// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Nemotron/Qwen reasoning boundaries are stateful: a reasoning delimiter in
//! TOOL_ARGS is argument data, not an instruction to reopen or close reasoning.
//! Keep the tool grammar bytes intact for the independent downstream tool parser.

use std::collections::{HashMap, VecDeque};
use std::pin::Pin;

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
}

impl Splitter {
    #[cfg(test)]
    fn new(inspect_bare_json: bool) -> Self {
        Self::with_options(inspect_bare_json, true, false, true)
    }

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
            if let Some(marker) = markers.iter().find(|m| remaining.starts_with(**m)) {
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

pub(super) fn parse_stream<S>(
    input: S,
    inspect_bare_json: bool,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    parse_stream_with_options(input, inspect_bare_json, true, false, true)
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
mod tests {
    use super::Splitter;

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
                    let mut parser = Splitter::new(false);
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
