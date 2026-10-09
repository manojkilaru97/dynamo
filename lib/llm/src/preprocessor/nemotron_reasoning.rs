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

/// Tool-argument closers the model sometimes writes inside reasoning without ever
/// opening a call (Super 3.5 Bug 5: `...\n</parameter>\n</function>\n</tool_call>`).
const ORPHAN_CLOSERS: [&str; 2] = ["</parameter>", "</function>"];
/// Call-body openers. Once one appears in reasoning, closers may belong to a call
/// written there, so residue removal is disabled for the rest of the reasoning.
const TOOL_OPENERS: [&str; 2] = ["<function=", "<parameter="];
/// Longest closer run held back before it is released as plain reasoning.
const MAX_ORPHAN_CLOSER_BYTES: usize = 256;

struct Splitter {
    phase: Phase,
    buffer: String,
    reasoning_whitespace: String,
    /// Removal of reasoning-phase closer residue. The rule is deliberately
    /// narrow: only the exact shape the model emits (captured from production) is
    /// dropped, and anything ambiguous is kept byte-for-byte as before.
    ///
    /// A run of `</parameter>`/`</function>` lines starting at column 0 is held
    /// in `orphan_closers`; a column-0 `</tool_call>` after it closes the run. A
    /// closed run is dropped only when reasoning then ends: at a protocol
    /// boundary (`<think>`, `</think>`, `<tool_call>`) or at a natural EOS/stop.
    /// Any visible character, an opener, or a length/unknown end releases it.
    orphan_closers: String,
    orphan_closed: bool,
    /// A closer of the current sequence was emitted as text (or the run outgrew
    /// the holdback cap): closers stay text until ordinary visible reasoning or
    /// a protocol boundary, so a sequence is never half-dropped.
    closer_passthrough: bool,
    /// A call-body opener appeared in this reasoning; no residue is removed.
    reasoning_saw_opener: bool,
    /// No visible character, and no indentation, since the last newline.
    line_column_zero: bool,
    /// The upcoming EOF is a natural stop (EOS or stop string), not a length
    /// limit or a cut stream; set by the caller before the final `push`.
    natural_stop: bool,
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
            orphan_closers: String::new(),
            orphan_closed: false,
            closer_passthrough: false,
            reasoning_saw_opener: false,
            line_column_zero: true,
            natural_stop: false,
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
            Phase::Reasoning => &[
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
                ORPHAN_CLOSERS[0],
                ORPHAN_CLOSERS[1],
                TOOL_OPENERS[0],
                TOOL_OPENERS[1],
            ],
            Phase::Content => &["</think>", "<tool_call>", "<function="],
            Phase::ToolPreamble => &["</tool_call>", "<function="],
            Phase::ToolName => &[">", "</function>"],
            Phase::ToolArgs => &["</function>"],
            Phase::ToolBetween => &["</tool_call>", "<tool_call>", "<function="],
        }
    }

    /// Route one reasoning character, holding whitespace until the next visible
    /// character so trailing whitespace before a boundary is never emitted.
    fn push_reasoning_char(&mut self, reasoning: &mut String, ch: char) {
        if ch.is_whitespace() {
            if self.orphan_closers.len() + ch.len_utf8() > MAX_ORPHAN_CLOSER_BYTES {
                self.release_orphan_closers(reasoning);
                self.closer_passthrough = true;
            }
            if self.orphan_closers.is_empty() {
                self.reasoning_whitespace.push(ch);
            } else {
                self.orphan_closers.push(ch);
            }
            // Column 0 survives only newlines; indentation leaves it.
            self.line_column_zero = ch == '\n';
            return;
        }
        self.release_orphan_closers(reasoning);
        reasoning.push_str(&std::mem::take(&mut self.reasoning_whitespace));
        reasoning.push(ch);
        self.line_column_zero = false;
    }

    /// Handle `</parameter>`, `</function>` or `</tool_call>` in the reasoning
    /// phase. `protocol` is false for a `</tool_call>` spelled with ordinary
    /// tokens in token-aware mode: it is text, but still a closer of the sequence.
    fn push_reasoning_closer(&mut self, reasoning: &mut String, marker: &'static str, protocol: bool) {
        let closer = ORPHAN_CLOSERS.contains(&marker);
        let structural = protocol
            && self.line_column_zero
            && !self.orphan_closed
            && !self.closer_passthrough
            && !self.reasoning_saw_opener
            && self.orphan_closers.len() + marker.len() <= MAX_ORPHAN_CLOSER_BYTES
            // A lone `</tool_call>` (even the sampled control token, which is
            // also how prose spells it) is kept; only a closer run proves residue.
            && (closer || !self.orphan_closers.is_empty());
        if structural {
            self.orphan_closers.push_str(marker);
            self.orphan_closed = !closer;
            self.line_column_zero = false;
            return;
        }
        self.release_orphan_closers(reasoning);
        self.closer_passthrough = true;
        for ch in marker.chars() {
            self.push_reasoning_char(reasoning, ch);
        }
    }

    /// Resolve a held run where reasoning ends: at a protocol boundary
    /// (`<think>`, `</think>`, `<tool_call>`) or at EOF. A closed run is dropped
    /// at a boundary or a natural stop; at a length limit or a cut stream it may
    /// be a quoted example and is kept. An unclosed run is always kept.
    fn settle_orphan_closers(&mut self, reasoning: &mut String, eof: bool) {
        if self.orphan_closed && (!eof || self.natural_stop) {
            self.orphan_closers.clear();
        }
        self.release_orphan_closers(reasoning);
        self.closer_passthrough = false;
    }

    /// Emit a held closer run as ordinary reasoning text; its trailing whitespace
    /// stays pending like any other.
    fn release_orphan_closers(&mut self, reasoning: &mut String) {
        self.orphan_closed = false;
        if self.orphan_closers.is_empty() {
            return;
        }
        let run = std::mem::take(&mut self.orphan_closers);
        let visible = run.trim_end();
        reasoning.push_str(&std::mem::take(&mut self.reasoning_whitespace));
        reasoning.push_str(visible);
        self.reasoning_whitespace.push_str(&run[visible.len()..]);
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
                if before == Phase::Reasoning {
                    if marker == "</tool_call>" || ORPHAN_CLOSERS.contains(&marker) {
                        cursor += marker.len();
                        self.push_reasoning_closer(&mut reasoning, marker, true);
                        continue;
                    }
                    if TOOL_OPENERS.contains(&marker) {
                        // A call body may be written inside reasoning; from here
                        // on, closers could be its own, so nothing is removed.
                        cursor += marker.len();
                        self.reasoning_saw_opener = true;
                        self.closer_passthrough = false;
                        for ch in marker.chars() {
                            self.push_reasoning_char(&mut reasoning, ch);
                        }
                        continue;
                    }
                    self.settle_orphan_closers(&mut reasoning, false);
                }
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
            if self.phase == Phase::Reasoning && remaining.starts_with("</tool_call>") {
                // Token-aware: `</tool_call>` spelled with ordinary tokens. Not
                // protocol, but still part of a closer sequence.
                cursor += "</tool_call>".len();
                self.push_reasoning_closer(&mut reasoning, "</tool_call>", false);
                continue;
            }
            let ch = remaining.chars().next().unwrap();
            cursor += ch.len_utf8();
            if self.phase == Phase::Reasoning {
                self.closer_passthrough &= ch.is_whitespace();
                self.push_reasoning_char(&mut reasoning, ch);
            } else {
                content.push(ch);
            }
        }
        self.buffer.drain(..cursor);
        self.control_offsets.retain(|(offset, _)| *offset >= cursor);
        for (offset, _) in &mut self.control_offsets { *offset -= cursor; }
        if finished {
            self.settle_orphan_closers(&mut reasoning, true);
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
            let (reasoning, content) = self.push(index, &[], "", true, false)?;
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

    /// `natural_stop`: this final chunk ends with EOS or a stop string (not a
    /// length limit or cancellation).
    pub(crate) fn push(
        &mut self,
        index: u32,
        ids: &[u32],
        text: &str,
        finished: bool,
        natural_stop: bool,
    ) -> anyhow::Result<(String, String)> {
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
        choice.splitter.natural_stop = finished && natural_stop;
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
                splitter.natural_stop = matches!(
                    choice.finish_reason,
                    Some(dynamo_protocols::types::FinishReason::Stop)
                );
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
                let (r, c) = parser.push(0, chunk, &text, finished, finished)?;
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

/// Super 3.5 Bug 5 (Cognition/Devin): sampled-token replays captured from the GA
/// endpoint stack (v64) under 1000-way concurrency, via
/// `nvext.extra_fields=["completion_token_ids"]`. Only the generation tail is kept;
/// each ID maps to its exact Super 3.5 tokenizer decoding.
#[cfg(test)]
mod bug5_sampled_replay_tests {
    use super::*;
    use crate::tokenizers::traits::{DecodeResult, Decoder, Encoder, Tokenizer};

    const THINK: u32 = 12;
    const END_THINK: u32 = 13;
    const TOOL_CALL: u32 = 14;
    const END_TOOL_CALL: u32 = 15;
    const IM_END: u32 = 11;

    /// The model wrote orphan argument closers after its last reasoning sentence,
    /// then `</tool_call>` and EOS; it never sampled `<tool_call>`, `<function=`,
    /// `<parameter=`, or `</think>` (774 tokens, finish_reason=stop).
    const ORPHAN_CLOSERS_THEN_EOS: &[(u32, &str)] = &[
        (1513, " at"), (1278, " the"), (84273, " `_"), (1689, "get"), (5198, "Path"),
        (1096, "`"), (2254, " function"), (3879, " around"), (3110, " line"), (1032, " "),
        (1049, "1"), (1048, "0"), (1053, "5"), (1048, "0"), (1045, "-"), (1049, "1"),
        (1049, "1"), (1048, "0"), (1048, "0"), (1626, ".\n"), (1885, "</"),
        (31960, "parameter"), (1561, ">\n"), (1885, "</"), (5165, "function"), (1561, ">\n"),
        (END_TOOL_CALL, "</tool_call>"), (1010, "\n"), (IM_END, "<|im_end|>"),
    ];

    /// EOS sampled inside open reasoning (p=0.008 teacher-forced; </think> p=0.89):
    /// no protocol token at all in 10644 tokens. The splitter must keep it reasoning.
    const EOS_INSIDE_REASONING: &[(u32, &str)] = &[
        (1045, "-"), (1051, "3"), (1056, "8"), (1050, "2"), (1562, " from"), (1278, " the"),
        (3323, " file"), (1925, " one"), (2081, " more"), (2142, " time"), (1044, ","),
        (3435, " very"), (21966, " carefully"), (1046, "."), (9246, " Let"), (1639, " me"),
        (1344, " re"), (41412, "-read"), (1046, "."), (IM_END, "<|im_end|>"),
    ];

    struct FakeTokenizer(HashMap<u32, String>);

    impl Encoder for FakeTokenizer {
        fn encode(&self, _: &str) -> anyhow::Result<crate::tokenizers::Encoding> {
            anyhow::bail!("decode-only test tokenizer")
        }
        fn encode_batch(&self, _: &[&str]) -> anyhow::Result<Vec<crate::tokenizers::Encoding>> {
            anyhow::bail!("decode-only test tokenizer")
        }
    }

    impl Decoder for FakeTokenizer {
        fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<DecodeResult> {
            let mut out = String::new();
            for id in ids {
                if skip_special_tokens && *id == IM_END {
                    continue;
                }
                out.push_str(self.0.get(id).map(String::as_str).unwrap_or(""));
            }
            Ok(DecodeResult::from_decoded(out))
        }
    }

    impl Tokenizer for FakeTokenizer {}

    fn controls() -> Arc<HashMap<u32, &'static str>> {
        Arc::new(HashMap::from([
            (THINK, "<think>"),
            (END_THINK, "</think>"),
            (TOOL_CALL, "<tool_call>"),
            (END_TOOL_CALL, "</tool_call>"),
        ]))
    }

    /// Replay `seq` (prefixed by reasoning `lead`) through TokenAwareReasoning in
    /// chunks of `chunk` tokens, the way DeltaGenerator feeds BackendOutput.
    fn replay(lead: &str, seq: &[(u32, &str)], chunk: usize) -> (String, String) {
        let mut vocab: HashMap<u32, String> =
            seq.iter().map(|(id, s)| (*id, (*s).to_string())).collect();
        vocab.insert(7, lead.to_string());
        vocab.insert(IM_END, "<|im_end|>".to_string());
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(FakeTokenizer(vocab));
        let ids: Vec<u32> = std::iter::once(7).chain(seq.iter().map(|(id, _)| *id)).collect();
        let mut parser =
            TokenAwareReasoning::new(tokenizer.clone(), &[], controls(), true, false, true, false, true);
        let mut decoder = crate::tokenizers::DecodeStream::new(tokenizer, &[], true);
        let (mut reasoning, mut content) = (String::new(), String::new());
        for (n, part) in ids.chunks(chunk).enumerate() {
            let mut text = String::new();
            for &id in part {
                text.push_str(decoder.step(id).unwrap().as_deref().unwrap_or(""));
            }
            let finished = (n + 1) * chunk >= ids.len();
            let (r, c) = parser.push(0, part, &text, finished, finished).unwrap();
            reasoning.push_str(&r);
            content.push_str(&c);
        }
        (reasoning, content)
    }

    #[test]
    fn orphan_closers_then_eos_leave_no_tool_markup_in_reasoning() {
        let lead = "If `path` is undefined... Let me check `_getPath`:\n\nActually let me look";
        for chunk in 1..=ORPHAN_CLOSERS_THEN_EOS.len() + 1 {
            let (reasoning, content) = replay(lead, ORPHAN_CLOSERS_THEN_EOS, chunk);
            assert_eq!(
                reasoning,
                format!("{lead} at the `_getPath` function around line 1050-1100."),
                "chunk={chunk}"
            );
            assert_eq!(content, "", "chunk={chunk}");
        }
    }

    #[test]
    fn orphan_closers_then_real_call_keep_the_call_for_the_tool_parser() {
        // Same captured prefix; the model's p=0.32 alternative after `</tool_call>\n`
        // (seen on the c256 run): it then opens a real call.
        let mut seq = ORPHAN_CLOSERS_THEN_EOS[..ORPHAN_CLOSERS_THEN_EOS.len() - 1].to_vec();
        seq.extend_from_slice(&[
            (TOOL_CALL, "<tool_call>"), (1010, "\n"), (40, "<function=read>\n"),
            (41, "<parameter=file_path>\n/repo/lib/schema.js\n</parameter>\n"),
            (42, "</function>"), (1010, "\n"), (END_TOOL_CALL, "</tool_call>"),
            (IM_END, "<|im_end|>"),
        ]);
        let call = "<tool_call>\n<function=read>\n<parameter=file_path>\n/repo/lib/schema.js\n</parameter>\n</function>\n</tool_call>";
        for chunk in 1..=seq.len() + 1 {
            let (reasoning, content) = replay("Reason", &seq, chunk);
            assert_eq!(
                reasoning,
                "Reason at the `_getPath` function around line 1050-1100.",
                "chunk={chunk}"
            );
            assert_eq!(content, call, "chunk={chunk}");
        }
    }

    #[test]
    fn eos_inside_reasoning_stays_reasoning() {
        for chunk in 1..=EOS_INSIDE_REASONING.len() + 1 {
            let (reasoning, content) = replay("Hmm", EOS_INSIDE_REASONING, chunk);
            assert_eq!(
                reasoning,
                "Hmm-382 from the file one more time, very carefully. Let me re-read.",
                "chunk={chunk}"
            );
            assert_eq!(content, "", "chunk={chunk}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Splitter;

    fn split_all(raw: &str, token_aware_controls: &[(usize, &'static str)]) -> Vec<(String, String)> {
        split_all_ending(raw, token_aware_controls, true)
    }

    /// Feed `raw` in two pieces at every char boundary; the stream then ends with
    /// a natural stop (EOS) or not (length limit / cut stream).
    fn split_all_ending(
        raw: &str,
        token_aware_controls: &[(usize, &'static str)],
        natural_stop: bool,
    ) -> Vec<(String, String)> {
        (0..=raw.len())
            .filter(|split| raw.is_char_boundary(*split))
            .map(|split| {
                let mut parser = Splitter::with_options(false, true, false, true);
                parser.token_aware = !token_aware_controls.is_empty();
                parser.control_offsets.extend(token_aware_controls.iter().copied());
                let a = parser.push(&raw[..split], false);
                parser.natural_stop = natural_stop;
                let b = parser.push(&raw[split..], true);
                (format!("{}{}", a.0, b.0), format!("{}{}", a.1, b.1))
            })
            .collect()
    }

    #[test]
    fn closed_orphan_closer_run_is_dropped_from_reasoning() {
        let raw = "Read more.\n</parameter>\n</function>\n</tool_call>\n";
        for (reasoning, content) in split_all(raw, &[]) {
            assert_eq!(reasoning, "Read more.");
            assert_eq!(content, "");
        }
        let at = raw.find("</tool_call>").unwrap();
        for (reasoning, content) in split_all(raw, &[(at, "</tool_call>")]) {
            assert_eq!(reasoning, "Read more.");
            assert_eq!(content, "");
        }
    }

    #[test]
    fn unclosed_or_interrupted_closer_text_stays_reasoning() {
        for (raw, expected) in [
            ("a </parameter> b", "a </parameter> b"),
            ("a\n</parameter>\n</function>\n", "a\n</parameter>\n</function>"),
            ("a\n</function> then b</tool_call>c", "a\n</function> then b</tool_call>c"),
            ("XML ends with </tool_call> here", "XML ends with </tool_call> here"),
        ] {
            for (reasoning, content) in split_all(raw, &[]) {
                assert_eq!(reasoning, expected, "{raw:?}");
                assert_eq!(content, "", "{raw:?}");
            }
        }
    }

    #[test]
    fn lone_tool_call_closer_is_kept_in_both_modes() {
        // The tokenizer spells prose `</tool_call>` with the control token too, so a
        // lone closer (inline or on its own line) is never treated as residue.
        for raw in ["says </tool_call> ok", "says\n</tool_call>\nok"] {
            let at = raw.find("</tool_call>").unwrap();
            for controls in [&[][..], &[(usize::MAX, "</tool_call>")][..], &[(at, "</tool_call>")][..]] {
                for (reasoning, _) in split_all(raw, controls) {
                    assert_eq!(reasoning, raw, "{controls:?}");
                }
            }
        }
    }

    #[test]
    fn inline_quoted_closer_sequence_is_reasoning_text() {
        let raw = "The expected suffix is `</parameter></function></tool_call>`.";
        let at = raw.find("</tool_call>").unwrap();
        for controls in [&[][..], &[(at, "</tool_call>")][..]] {
            for (reasoning, content) in split_all(raw, controls) {
                assert_eq!(reasoning, raw, "{controls:?}");
                assert_eq!(content, "");
            }
        }
        // Same line after the closer: not a protocol boundary, keep it all.
        let raw = "x\n</parameter>\n</function>\n</tool_call> is how it ends";
        for (reasoning, _) in split_all(raw, &[]) {
            assert_eq!(reasoning, raw);
        }
    }

    #[test]
    fn closed_run_at_eof_without_trailing_newline_is_dropped() {
        let raw = "Read more.\n</parameter>\n</function>\n</tool_call>";
        for (reasoning, content) in split_all(raw, &[]) {
            assert_eq!(reasoning, "Read more.");
            assert_eq!(content, "");
        }
    }

    #[test]
    fn degenerate_closer_repetition_is_released_not_held_until_eof() {
        let run = "</parameter>\n".repeat(64);
        let mut parser = Splitter::with_options(false, true, false, true);
        let (reasoning, content) = parser.push(&format!("x\n{run}"), false);
        // Streamed before EOF, and byte-for-byte reasoning.
        assert!(reasoning.len() > 200, "{reasoning:?}");
        let (tail, more) = parser.push("y", true);
        assert_eq!(format!("{reasoning}{tail}"), format!("x\n{run}y"));
        assert_eq!(format!("{content}{more}"), "");
    }

    #[test]
    fn oversized_run_stays_text_through_its_closing_tags() {
        let long = format!("x\n{}</function>\n</tool_call>", "</parameter>\n".repeat(20));
        for tail in ["\ny", "\n", ""] {
            let raw = format!("{long}{tail}");
            for (reasoning, content) in split_all(&raw, &[]) {
                assert_eq!(reasoning, raw.trim_end(), "{tail:?}");
                assert_eq!(content, "");
            }
        }
    }

    #[test]
    fn closer_lines_followed_by_more_reasoning_are_kept() {
        for raw in [
            "Expected suffix:\n```xml\n</parameter>\n</function>\n</tool_call>\n```\nContinue.",
            "Expected suffix:\n~~~\n</parameter>\n</function>\n</tool_call>\n~~~",
            "Read more.\n</parameter>\n</function>\n</tool_call>\nThen more.",
        ] {
            let at = raw.find("</tool_call>").unwrap();
            for controls in [&[][..], &[(at, "</tool_call>")][..]] {
                for (reasoning, content) in split_all(raw, controls) {
                    assert_eq!(reasoning, raw, "{controls:?}");
                    assert_eq!(content, "");
                }
            }
        }
    }

    #[test]
    fn a_closer_sequence_is_never_half_dropped() {
        for raw in [
            "Done.</parameter>\n</function>\n</tool_call>\n",
            "A.\n</parameter>\n</function>\n</tool_call>\n</parameter>\n</function>\n</tool_call>\n",
        ] {
            let at = raw.find("</tool_call>").unwrap();
            for controls in [&[][..], &[(at, "</tool_call>")][..]] {
                for (reasoning, _) in split_all(raw, controls) {
                    assert_eq!(reasoning, raw.trim_end(), "{controls:?}");
                }
            }
        }
    }

    #[test]
    fn indented_quotations_and_length_truncated_runs_are_kept() {
        // Residue is never indented (Markdown code block lines are).
        let raw = "Expected suffix:\n\n    </parameter>\n    </function>\n    </tool_call>";
        for (reasoning, _) in split_all(raw, &[]) {
            assert_eq!(reasoning, raw);
        }
        let raw = "Expected suffix:\n\n    </parameter>\n    </function>\n    </tool_call>\n</think>Answer";
        for (reasoning, content) in split_all(raw, &[]) {
            assert_eq!(
                reasoning,
                "Expected suffix:\n\n    </parameter>\n    </function>\n    </tool_call>"
            );
            assert_eq!(content, "Answer");
        }
        // A quoted example cut off by a length limit (no EOS) is kept, fenced or not.
        for raw in [
            "Expected suffix:\n```\n</parameter>\n</function>\n</tool_call>",
            "Expected suffix:\n~~~~xml\n~~~\n</parameter>\n</function>\n</tool_call>\n",
            "Read more.\n</parameter>\n</function>\n</tool_call>\n",
        ] {
            let at = raw.find("</tool_call>").unwrap();
            for controls in [&[][..], &[(at, "</tool_call>")][..]] {
                for (reasoning, _) in split_all_ending(raw, controls, false) {
                    assert_eq!(reasoning, raw.trim_end(), "{controls:?}");
                }
            }
        }
    }

    #[test]
    fn residue_after_an_unclosed_fence_is_dropped_where_reasoning_ends() {
        // Captured (c256): reasoning left a ```js block open, then wrote residue
        // and opened a real call; the more likely continuation is EOS.
        let head = "```js\nx();\nLet me re-read.\n</parameter>\n</function>\n</tool_call>\n";
        let raw = format!("{head}<tool_call>\n<function=read>\n</function>\n</tool_call>");
        for (reasoning, content) in split_all(&raw, &[]) {
            assert_eq!(reasoning, "```js\nx();\nLet me re-read.");
            assert_eq!(content, "<tool_call>\n<function=read>\n</function>\n</tool_call>");
        }
        let at = head.find("</tool_call>").unwrap();
        for controls in [&[][..], &[(at, "</tool_call>")][..]] {
            for (reasoning, content) in split_all(head, controls) {
                assert_eq!(reasoning, "```js\nx();\nLet me re-read.", "{controls:?}");
                assert_eq!(content, "");
            }
        }
    }

    #[test]
    fn nothing_is_removed_after_a_call_body_opener() {
        for raw in [
            // Residue, then a call body with no `<tool_call>`.
            "Plan.\n</parameter>\n</function>\n</tool_call>\n<parameter=file_path>\n/x.js\n</parameter>\n</function>\n</tool_call>\n",
            // An inline opener mention, then closer lines.
            "Use <parameter=path> here.\nThen think.\n</parameter>\n</function>\n</tool_call>\n",
            "I'll run <function=bash><parameter=cmd>ls\n</parameter>\n</function>\n</tool_call>\n",
            // `</tool_call>` inside an argument value.
            "<function=exec>\n<parameter=command>\ncat <<'EOF'\n</tool_call>\nEOF\n</parameter>\n</function>\n</tool_call>\n",
            // Captured variant: a stray `</parameter>`, then parameters with no opener.
            "Read from 329.\n</parameter>\n<parameter=file_path>\n/x.js\n</parameter>\n<parameter=offset>\n326\n</parameter>\n</function>\n</tool_call>\n",
        ] {
            let at = raw.rfind("</tool_call>").unwrap();
            for controls in [&[][..], &[(at, "</tool_call>")][..]] {
                for (reasoning, content) in split_all(raw, controls) {
                    assert_eq!(reasoning, raw.trim_end(), "{raw:?} {controls:?}");
                    assert_eq!(content, "");
                }
            }
        }
    }

    #[test]
    fn ordinary_token_tool_call_closer_keeps_the_sequence_text() {
        // Token-aware: the first `</tool_call>` is spelled with ordinary tokens
        // (no control offset), the second is the control token.
        for raw in [
            "A.\n</parameter>\n</function>\n</tool_call>\n</parameter>\n</function>\n</tool_call>\n",
            "Expected suffix:\n</parameter>\n</tool_call>\n</function>\n</tool_call>\n",
        ] {
            let second = raw.rfind("</tool_call>").unwrap();
            for (reasoning, _) in split_all(raw, &[(second, "</tool_call>")]) {
                assert_eq!(reasoning, raw.trim_end(), "{raw:?}");
            }
        }
    }

    #[test]
    fn closed_run_before_end_of_reasoning_is_dropped() {
        let raw = "Plan.\n</parameter>\n</function>\n</tool_call>\n</think>Answer";
        for (reasoning, content) in split_all(raw, &[]) {
            assert_eq!(reasoning, "Plan.");
            assert_eq!(content, "Answer");
        }
    }

    #[test]
    fn orphan_run_before_a_real_call_is_dropped_and_call_is_kept() {
        let raw = "Plan.\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=read>\n<parameter=p>\nx\n</parameter>\n</function>\n</tool_call>";
        for (reasoning, content) in split_all(raw, &[]) {
            assert_eq!(reasoning, "Plan.");
            assert_eq!(
                content,
                "<tool_call>\n<function=read>\n<parameter=p>\nx\n</parameter>\n</function>\n</tool_call>"
            );
        }
    }

    #[test]
    fn literal_reasoning_markers_in_tool_args_survive_all_splits() {
        for prefix in ["Reason", "Reason</think>"] {
            for marker in [
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
                "<exec>",
                "</exec>",
            ] {
                let raw = format!(
                    "{prefix}<tool_call><function=exec><parameter=command>before {marker} after</parameter></function></tool_call>"
                );
                for split in 0..=raw.len() {
                    let mut parser = Splitter::with_options(false, true, false, true);
                    let a = parser.push(&raw[..split], false);
                    let b = parser.push(&raw[split..], true);
                    assert_eq!(
                        format!("{}{}", a.0, b.0),
                        "Reason",
                        "{marker} split={split}"
                    );
                    assert_eq!(
                        format!("{}{}", a.1, b.1),
                        raw[prefix.len()..],
                        "{marker} split={split}"
                    );
                }
            }
        }
    }
}
