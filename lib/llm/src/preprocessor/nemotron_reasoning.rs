// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Nemotron/Qwen reasoning boundaries are stateful: a reasoning delimiter in
//! TOOL_ARGS is argument data, not an instruction to reopen or close reasoning.
//! Keep the tool grammar bytes intact for the independent downstream tool parser.

use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::Arc;

use dynamo_protocols::types::ChatCompletionMessageContent;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt, stream};

use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Reasoning,
    Content,
    ToolPreamble,
    ToolName,
    ToolArgs,
    ToolBetween,
}

struct Splitter {
    phase: Phase,
    buffer: String,
    reasoning_whitespace: String,
    bypass_decided: bool,
    bypass: bool,
    emitted_content: bool,
    visible_content: bool,
    force_nonempty: bool,
    streaming: bool,
    reasoning_history: String,
    finished: bool,
    token_aware: bool,
    control_offsets: VecDeque<(usize, &'static str)>,
}

impl Splitter {
    fn with_options(
        inspect_bare_json: bool,
        thinking: bool,
        force_nonempty: bool,
        streaming: bool,
    ) -> Self {
        Self {
            phase: if thinking {
                Phase::Reasoning
            } else {
                Phase::Content
            },
            buffer: String::new(),
            reasoning_whitespace: String::new(),
            bypass_decided: !inspect_bare_json,
            bypass: false,
            emitted_content: false,
            visible_content: false,
            force_nonempty,
            streaming,
            reasoning_history: String::new(),
            finished: false,
            token_aware: false,
            control_offsets: VecDeque::new(),
        }
    }

    fn markers(&self) -> &'static [&'static str] {
        match self.phase {
            Phase::Reasoning => &["<think>", "</think>", "<tool_call>"],
            Phase::Content => &["</think>", "<tool_call>", "<function="],
            Phase::ToolPreamble => &["</tool_call>", "<function="],
            Phase::ToolName => &[">", "</function>"],
            Phase::ToolArgs => &["</function>"],
            Phase::ToolBetween => &["</tool_call>", "<tool_call>", "<function="],
        }
    }

    fn push(&mut self, text: &str, finished: bool) -> (String, String) {
        self.finished |= finished;
        self.buffer.push_str(text);
        if !self.bypass_decided {
            if let Some(first) = self.buffer.trim_start().chars().next() {
                self.bypass = matches!(first, '[' | '{');
                self.bypass_decided = true;
            } else if !finished {
                return (String::new(), String::new());
            }
        }
        if self.bypass {
            return (String::new(), std::mem::take(&mut self.buffer));
        }
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut cursor = 0;
        while cursor < self.buffer.len() {
            let remaining = &self.buffer[cursor..];
            let markers = self.markers();
            if let Some(marker) = markers.iter().find(|m| {
                remaining.starts_with(**m)
                    && (!self.token_aware
                        || !super::super_user_tokenization::MARKERS.contains(m)
                        || self.control_offsets.iter().any(|(offset, marker)| {
                            *offset == cursor && marker == *m
                        }))
            }) {
                let marker = *marker;
                let before = self.phase;
                self.phase = match (before, marker) {
                    (Phase::Reasoning, "</think>") => Phase::Content,
                    (Phase::Reasoning | Phase::Content | Phase::ToolBetween, "<tool_call>") => {
                        Phase::ToolPreamble
                    }
                    (Phase::Content | Phase::ToolPreamble | Phase::ToolBetween, "<function=") => {
                        Phase::ToolName
                    }
                    (Phase::ToolName, ">") => Phase::ToolArgs,
                    (Phase::ToolName | Phase::ToolArgs, "</function>") => Phase::ToolBetween,
                    (Phase::ToolPreamble | Phase::ToolBetween, "</tool_call>") => Phase::Content,
                    _ => before,
                };
                cursor += marker.len();
                if before == Phase::Reasoning && self.phase != Phase::Reasoning {
                    self.reasoning_whitespace.clear();
                }
                // Only reasoning-owned delimiters are removed. All tool syntax
                // remains byte-for-byte input to the tool parser.
                if !matches!(
                    (before, marker),
                    (Phase::Reasoning, "<think>" | "</think>") | (Phase::Content, "</think>")
                ) {
                    content.push_str(marker);
                }
                continue;
            }
            if !finished && markers.iter().any(|marker| marker.starts_with(remaining)) {
                break;
            }
            let ch = remaining.chars().next().unwrap();
            cursor += ch.len_utf8();
            if self.phase == Phase::Reasoning {
                if ch.is_whitespace() {
                    self.reasoning_whitespace.push(ch);
                } else {
                    reasoning.push_str(&std::mem::take(&mut self.reasoning_whitespace));
                    reasoning.push(ch);
                }
            } else {
                content.push(ch);
            }
        }
        self.buffer.drain(..cursor);
        self.control_offsets.retain(|(offset, _)| *offset >= cursor);
        for (offset, _) in &mut self.control_offsets { *offset -= cursor; }
        if finished {
            self.reasoning_whitespace.clear();
        }
        if !self.emitted_content {
            content = content.trim_start_matches(['\n', '\r']).to_string();
            self.emitted_content = !content.is_empty();
        }
        self.visible_content |= !content.trim().is_empty();
        if self.force_nonempty {
            self.reasoning_history.push_str(&reasoning);
            // This is an EOF fallback, not a request-wide reasoning disable.
            // Streaming reasoning cannot be retracted. Batch mode can retain
            // it until EOF and swap it into otherwise empty content instead.
            if finished
                && !self.visible_content
                && (!self.streaming || self.phase == Phase::Reasoning)
            {
                content = self.reasoning_history.clone();
            }
            if !self.streaming {
                reasoning = if finished && self.visible_content {
                    std::mem::take(&mut self.reasoning_history)
                } else {
                    String::new()
                };
            }
        }
        (reasoning, content)
    }
}

/// Split while BackendOutput still carries sampled IDs. A second incremental
/// decoder supplies exact byte offsets; the backend's already-stopped text must
/// be its prefix. Stop holdback never transfers a control's identity to a later
/// ordinary token, and repeated marker spellings are never located by search.
pub(crate) struct TokenAwareReasoning {
    tokenizer: Arc<dyn crate::tokenizers::traits::Tokenizer>,
    prompt: Vec<u32>,
    controls: Arc<HashMap<u32, &'static str>>,
    skip_special_tokens: bool,
    choices: HashMap<u32, TokenAwareChoice>,
    options: (bool, bool, bool, bool),
}

struct TokenAwareChoice {
    decoder: crate::tokenizers::DecodeStream,
    splitter: Splitter,
    pending_text: String,
    pending_controls: VecDeque<(usize, &'static str)>,
    visible_bytes: usize,
}

impl TokenAwareReasoning {
    pub(crate) fn finish_pending(&mut self) -> anyhow::Result<Vec<(u32, String, String)>> {
        let indices: Vec<_> = self.choices.iter()
            .filter_map(|(&index, choice)| (!choice.splitter.finished).then_some(index)).collect();
        let mut output = Vec::new();
        for index in indices {
            let (reasoning, content) = self.push(index, &[], "", true)?;
            if !reasoning.is_empty() || !content.is_empty() {
                output.push((index, reasoning, content));
            }
        }
        Ok(output)
    }

    pub(crate) fn new(
        tokenizer: Arc<dyn crate::tokenizers::traits::Tokenizer>,
        prompt: &[u32],
        controls: Arc<HashMap<u32, &'static str>>,
        skip_special_tokens: bool,
        inspect_bare_json: bool,
        thinking: bool,
        force_nonempty: bool,
        streaming: bool,
    ) -> Self {
        Self {
            tokenizer, prompt: prompt.to_vec(), controls, skip_special_tokens, choices: HashMap::new(),
            options: (inspect_bare_json, thinking, force_nonempty, streaming),
        }
    }

    pub(crate) fn push(&mut self, index: u32, ids: &[u32], text: &str, finished: bool)
        -> anyhow::Result<(String, String)>
    {
        let (inspect, thinking, force, streaming) = self.options;
        let choice = self.choices.entry(index).or_insert_with(|| {
            let mut splitter = Splitter::with_options(inspect, thinking, force, streaming);
            splitter.token_aware = true;
            TokenAwareChoice {
                decoder: crate::tokenizers::DecodeStream::new(self.tokenizer.clone(), &self.prompt, self.skip_special_tokens),
                splitter, pending_text: String::new(), pending_controls: VecDeque::new(),
                visible_bytes: 0,
            }
        });
        for &id in ids {
            if let Some(fragment) = choice.decoder.step(id)? {
                if let Some(&marker) = self.controls.get(&id) {
                    anyhow::ensure!(fragment.ends_with(marker),
                        "protocol token did not decode at its own byte boundary");
                    choice.pending_controls.push_back((
                        choice.pending_text.len() + fragment.len() - marker.len(), marker,
                    ));
                }
                choice.pending_text.push_str(&fragment);
            } else {
                anyhow::ensure!(!self.controls.contains_key(&id),
                    "protocol control unexpectedly has incomplete decoding");
            }
        }
        let end = choice.visible_bytes + text.len();
        anyhow::ensure!(choice.pending_text.get(choice.visible_bytes..end) == Some(text),
            "backend decoded text does not match sampled-token provenance");
        choice.visible_bytes = end;
        let mut emit = end;
        if !finished {
            for &(offset, marker) in &choice.pending_controls {
                if offset < emit && offset + marker.len() > emit { emit = offset; break; }
            }
        }
        for &(offset, marker) in &choice.pending_controls {
            if offset + marker.len() <= emit {
                choice.splitter.control_offsets.push_back((choice.splitter.buffer.len() + offset, marker));
            }
        }
        let (reasoning, content) = choice.splitter.push(&choice.pending_text[..emit], finished);
        choice.pending_text.drain(..emit);
        choice.visible_bytes -= emit;
        choice.pending_controls.retain(|(offset, _)| *offset >= emit);
        for (offset, _) in &mut choice.pending_controls { *offset -= emit; }
        if finished {
            // Remaining raw bytes are a hidden stop/EOS suffix, never output.
            choice.pending_text.clear();
            choice.pending_controls.clear();
        }
        Ok((reasoning, content))
    }
}

pub(super) fn parse_stream_with_options<S>(
    input: S,
    inspect_bare_json: bool,
    thinking: bool,
    force_nonempty: bool,
    streaming: bool,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    struct State<S> {
        input: Pin<Box<S>>,
        choices: HashMap<u32, Splitter>,
        inspect_bare_json: bool,
        thinking: bool,
        force_nonempty: bool,
        streaming: bool,
        templates: HashMap<u32, Annotated<NvCreateChatCompletionStreamResponse>>,
        pending: VecDeque<Annotated<NvCreateChatCompletionStreamResponse>>,
        eof_flushed: bool,
    }
    let state = State {
        input: Box::pin(input),
        choices: HashMap::new(),
        inspect_bare_json,
        thinking,
        force_nonempty,
        streaming,
        templates: HashMap::new(),
        pending: VecDeque::new(),
        eof_flushed: false,
    };
    stream::unfold(state, |mut state| async move {
        if let Some(response) = state.pending.pop_front() {
            return Some((response, state));
        }
        let Some(mut response) = state.input.next().await else {
            if state.eof_flushed {
                return None;
            }
            state.eof_flushed = true;
            for (index, splitter) in &mut state.choices {
                if splitter.finished {
                    continue;
                }
                let (reasoning, content) = splitter.push("", true);
                if reasoning.is_empty() && content.is_empty() {
                    continue;
                }
                let Some(mut response) = state.templates.remove(index) else {
                    continue;
                };
                let Some(data) = response.data.as_mut() else {
                    continue;
                };
                data.inner.usage = None;
                data.inner.choices.retain(|choice| choice.index == *index);
                for choice in &mut data.inner.choices {
                    choice.delta.role = None;
                    choice.delta.content = (!content.is_empty())
                        .then(|| ChatCompletionMessageContent::Text(content.clone()));
                    choice.delta.reasoning_content =
                        (!reasoning.is_empty()).then(|| reasoning.clone());
                    choice.delta.tool_calls = None;
                    choice.delta.function_call = None;
                    choice.delta.refusal = None;
                    choice.finish_reason = None;
                    choice.logprobs = None;
                }
                state.pending.push_back(response);
            }
            return state.pending.pop_front().map(|response| (response, state));
        };
        if let Some(data) = response.data.as_ref() {
            for choice in &data.inner.choices {
                state.templates.insert(choice.index, response.clone());
            }
        }
        if let Some(data) = response.data.as_mut() {
            for choice in &mut data.inner.choices {
                let splitter = state.choices.entry(choice.index).or_insert_with(|| {
                    Splitter::with_options(
                        state.inspect_bare_json,
                        state.thinking,
                        state.force_nonempty,
                        state.streaming,
                    )
                });
                let text = match choice.delta.content.take() {
                    Some(ChatCompletionMessageContent::Text(text)) => text,
                    other => {
                        choice.delta.content = other;
                        String::new()
                    }
                };
                let (reasoning, content) = splitter.push(&text, choice.finish_reason.is_some());
                if !reasoning.is_empty() {
                    choice
                        .delta
                        .reasoning_content
                        .get_or_insert_default()
                        .push_str(&reasoning);
                }
                if !content.is_empty() {
                    choice.delta.content = Some(ChatCompletionMessageContent::Text(content));
                }
            }
        }
        Some((response, state))
    })
}

#[cfg(test)]
mod token_provenance_tests {
    use super::*;

    /// CPU replay of captured sampled IDs: no model weights or generation.
    #[test]
    #[ignore = "requires SUPER35_CPU_MODEL_DIR and SUPER35_CPU_AUDIT_DIR"]
    fn super_user_actual_sampled_boundary_replay() -> anyhow::Result<()> {
        let model = std::path::PathBuf::from(std::env::var("SUPER35_CPU_MODEL_DIR")?);
        let audit = std::path::PathBuf::from(std::env::var("SUPER35_CPU_AUDIT_DIR")?);
        let tokenizer: Arc<dyn crate::tokenizers::traits::Tokenizer> = Arc::new(
            crate::tokenizers::HuggingFaceTokenizer::from_file(model.join("tokenizer.json").to_str().unwrap())?);
        let encoder = super::super::super_user_tokenization::UserDataEncoder::from_json(
            &std::fs::read(model.join("tokenizer.json"))?)?;
        let capture: serde_json::Value = serde_json::from_slice(&std::fs::read(
            audit.join("transformers-reference-whole-user-r1/raw-generation.json"))?)?;
        let ids: Vec<u32> = serde_json::from_value(capture["sampled_token_ids"].clone())?;
        let prompt: Vec<u32> = serde_json::from_value(capture["prompt_token_ids"].clone())?;
        let end_id = *encoder.controls.iter().find(|(_, marker)| **marker == "</think>").unwrap().0;
        let boundary = ids.iter().position(|id| *id == end_id).unwrap();
        assert_eq!(ids.iter().filter(|id| **id == end_id).count(), 1);
        assert_eq!(ids.last(), Some(&11));
        let expected_reasoning: String = tokenizer.decode(&ids[..boundary], false)?.into();
        let expected_content: String = tokenizer.decode(&ids[boundary + 1..ids.len() - 1], false)?.into();
        for chunk_size in [1, 2, 7, 128, ids.len()] {
            let mut parser = TokenAwareReasoning::new(tokenizer.clone(), &prompt, encoder.controls.clone(), false, false, true, false, true);
            let mut decoder = crate::tokenizers::DecodeStream::new(tokenizer.clone(), &prompt, false);
            let mut reasoning = String::new();
            let mut content = String::new();
            for (chunk_index, chunk) in ids.chunks(chunk_size).enumerate() {
                let mut text = String::new();
                for &id in chunk {
                    let fragment = decoder.step(id)?;
                    if id != 11 { text.push_str(fragment.as_deref().unwrap_or("")); }
                }
                let finished = (chunk_index + 1) * chunk_size >= ids.len();
                let (r, c) = parser.push(0, chunk, &text, finished)?;
                reasoning.push_str(&r);
                content.push_str(&c);
            }
            assert_eq!(reasoning, expected_reasoning.trim_end(), "chunk size {chunk_size}");
            assert_eq!(content, expected_content.trim_start_matches(['\n', '\r']), "chunk size {chunk_size}");
            assert!(content.contains("</think><tool_call>{\"x\":1}</tool_call>"));
        }
        println!("Super CPU actual sampled-ID replay: exact reasoning/content at chunk sizes 1,2,7,128,2337");
        Ok(())
    }
}
