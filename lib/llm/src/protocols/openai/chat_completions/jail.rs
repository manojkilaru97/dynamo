// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_stream::stream;
use dynamo_protocols::types::{
    ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionMessageToolCallChunk,
    ChatCompletionStreamResponseDelta, FinishReason, FunctionCallStream, FunctionType, Role,
};

use dynamo_parsers::tool_calling::config::JsonParserConfig;
use dynamo_parsers::tool_calling::json::try_tool_call_parse_basic_json;
use dynamo_parsers::tool_calling::parsers::get_tool_parser_map;
use dynamo_parsers::tool_calling::{
    detect_tool_call_start, find_tool_call_end_position, try_tool_call_parse_aggregate,
    try_tool_call_parse_aggregate_finalize,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use uuid::Uuid;

use crate::utils::{MarkerMatcher, MatchResult};

use super::NvCreateChatCompletionStreamResponse;

const MAX_SUPPRESSED_REQUIRED_TOOL_DUPLICATES: usize = 2;

/// Represents what a choice wants to emit after processing content
#[derive(Debug, Clone)]
pub enum ChoiceEmission {
    /// Pass through content unchanged (choice is not jailed)
    PassThrough(ChatChoiceStream),
    /// Emit parsed tool calls (choice finished jailing with tool calls)
    ToolCall(ChatChoiceStream),
    /// Emit accumulated content (choice finished jailing without tool calls)
    Content(ChatChoiceStream),
    /// Emit trailing content after tool call end (choice has trailing after unjail)
    Trailing(ChatChoiceStream),
}

impl ChoiceEmission {
    /// Extract the ChatChoiceStream from any emission type
    pub fn into_choice(self) -> ChatChoiceStream {
        match self {
            ChoiceEmission::PassThrough(choice) => choice,
            ChoiceEmission::ToolCall(choice) => choice,
            ChoiceEmission::Content(choice) => choice,
            ChoiceEmission::Trailing(choice) => choice,
        }
    }

    /// Get the choice index
    pub fn index(&self) -> u32 {
        match self {
            ChoiceEmission::PassThrough(choice) => choice.index,
            ChoiceEmission::ToolCall(choice) => choice.index,
            ChoiceEmission::Content(choice) => choice.index,
            ChoiceEmission::Trailing(choice) => choice.index,
        }
    }

    /// Get mutable access to the underlying choice.
    fn choice_mut(&mut self) -> &mut ChatChoiceStream {
        match self {
            ChoiceEmission::PassThrough(choice) => choice,
            ChoiceEmission::ToolCall(choice) => choice,
            ChoiceEmission::Content(choice) => choice,
            ChoiceEmission::Trailing(choice) => choice,
        }
    }
}

/// Configuration for jail detection and parsing
#[derive(Debug, Clone)]
pub struct JailConfig<'a> {
    pub jail_start_sequences: &'a [String],
    pub jail_end_sequences: &'a [String],
    pub tool_call_parser: Option<&'a str>,
}

/// Jail activation mode
#[derive(Debug, Clone, PartialEq)]
pub enum JailMode {
    /// Traditional: wait for start marker, then jail
    MarkerBased,
    /// Immediate: start jailed from first token (for tool_choice)
    Immediate { format: ToolChoiceFormat },
}

/// Format for tool_choice immediate jail mode
#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoiceFormat {
    /// tool_choice=named: expect single object {"location": "Paris", ...}
    SingleObject { tool_name: String },
    /// tool_choice=required: expect array [{name:"search", parameters:{...}}, ...]
    ArrayOfTools { terminal_after_first: bool },
}

fn immediate_tool_choice_json_fragment(content: &str) -> &str {
    if let Some((_, suffix)) = content.rsplit_once("</think>") {
        let suffix = suffix.trim_start();
        if !suffix.is_empty() {
            return suffix;
        }
    }
    content.trim()
}

fn escape_json_string_control_chars(input: &str) -> Cow<'_, str> {
    let mut output = String::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut changed = false;

    for ch in input.chars() {
        if escaped {
            output.push(ch);
            escaped = false;
            continue;
        }

        if in_string {
            match ch {
                '\\' => {
                    output.push(ch);
                    escaped = true;
                }
                '"' => {
                    output.push(ch);
                    in_string = false;
                }
                '\n' => {
                    output.push_str("\\n");
                    changed = true;
                }
                '\r' => {
                    output.push_str("\\r");
                    changed = true;
                }
                '\t' => {
                    output.push_str("\\t");
                    changed = true;
                }
                c if c < ' ' => {
                    use std::fmt::Write;
                    let _ = write!(output, "\\u{:04x}", c as u32);
                    changed = true;
                }
                _ => output.push(ch),
            }
        } else {
            output.push(ch);
            if ch == '"' {
                in_string = true;
            }
        }
    }

    if changed {
        Cow::Owned(output)
    } else {
        Cow::Borrowed(input)
    }
}

#[derive(Debug, Clone, Default)]
struct PoolsideV1StreamDelta {
    content: Option<String>,
    tool_calls: Vec<ChatCompletionMessageToolCallChunk>,
}

#[derive(Debug, Clone)]
struct PoolsideV1StreamState {
    buffer: String,
    in_tool_call: bool,
    current_tool_name_sent: bool,
    current_tool_id: i32,
    current_tool_name: Option<String>,
    pending_key: Option<String>,
    streaming_string_value: bool,
    tool_call_ids: Vec<String>,
    streamed_args_for_tool: Vec<String>,
    args_started: Vec<bool>,
    args_closed: Vec<bool>,
    seen_keys: Vec<HashSet<String>>,
}

impl Default for PoolsideV1StreamState {
    fn default() -> Self {
        Self {
            buffer: String::new(),
            in_tool_call: false,
            current_tool_name_sent: false,
            current_tool_id: -1,
            current_tool_name: None,
            pending_key: None,
            streaming_string_value: false,
            tool_call_ids: Vec::new(),
            streamed_args_for_tool: Vec::new(),
            args_started: Vec::new(),
            args_closed: Vec::new(),
            seen_keys: Vec::new(),
        }
    }
}

impl PoolsideV1StreamState {
    const TOOL_CALL_START: &'static str = "<tool_call>";
    const TOOL_CALL_END: &'static str = "</tool_call>";
    const ARG_KEY_START: &'static str = "<arg_key>";
    const ARG_KEY_END: &'static str = "</arg_key>";
    const ARG_VALUE_START: &'static str = "<arg_value>";
    const ARG_VALUE_END: &'static str = "</arg_value>";

    fn process_delta(
        &mut self,
        delta_text: &str,
        tools: Option<&[dynamo_parsers::tool_calling::ToolDefinition]>,
    ) -> PoolsideV1StreamDelta {
        self.buffer.push_str(delta_text);

        let mut pending_deltas: BTreeMap<u32, ChatCompletionMessageToolCallChunk> = BTreeMap::new();
        let mut content: Option<String> = None;

        loop {
            if !self.in_tool_call {
                let Some(start_idx) = self.buffer.find(Self::TOOL_CALL_START) else {
                    let safe_len = self.safe_len_before_partial_start();
                    if safe_len > 0 {
                        let out = self.buffer[..safe_len].to_string();
                        self.buffer.drain(..safe_len);
                        append_optional(&mut content, out);
                    }
                    break;
                };

                if start_idx > 0 {
                    let out = self.buffer[..start_idx].to_string();
                    append_optional(&mut content, out);
                    self.buffer.drain(..start_idx);
                }

                self.buffer.drain(..Self::TOOL_CALL_START.len());
                self.begin_tool_call();
                continue;
            }

            if !self.current_tool_name_sent {
                let newline = self.buffer.find('\n');
                let arg_key = self.buffer.find(Self::ARG_KEY_START);
                let end = self.buffer.find(Self::TOOL_CALL_END);
                let Some(cut) = [newline, arg_key, end].into_iter().flatten().min() else {
                    break;
                };

                let tool_name = self.buffer[..cut].trim().to_string();
                if tool_name.is_empty() && end == Some(cut) {
                    self.buffer.drain(..cut + Self::TOOL_CALL_END.len());
                    self.finish_tool_call();
                    self.revert_last_tool_call_state();
                    continue;
                }

                if newline == Some(cut) {
                    self.buffer.drain(..cut + 1);
                } else {
                    self.buffer.drain(..cut);
                }

                self.current_tool_name = Some(tool_name.clone());
                self.current_tool_name_sent = true;
                self.update_tool_name(&mut pending_deltas, tool_name);
                continue;
            }

            if self.streaming_string_value {
                if let Some(value_end) = self.buffer.find(Self::ARG_VALUE_END) {
                    let raw_content = self.buffer[..value_end].to_string();
                    self.buffer.drain(..value_end + Self::ARG_VALUE_END.len());
                    self.streaming_string_value = false;
                    self.pending_key = None;

                    let fragment = format!("{}\"", json_escape_string_content(&raw_content));
                    self.append_current_args_fragment(&fragment);
                    self.update_tool_args(&mut pending_deltas, &fragment);
                    continue;
                }

                let safe_len = safe_len_before_partial_suffix(&self.buffer, Self::ARG_VALUE_END);
                if safe_len > 0 {
                    let to_emit = self.buffer[..safe_len].to_string();
                    self.buffer.drain(..safe_len);
                    let escaped = json_escape_string_content(&to_emit);
                    if !escaped.is_empty() {
                        self.append_current_args_fragment(&escaped);
                        self.update_tool_args(&mut pending_deltas, &escaped);
                    }
                }
                break;
            }

            if let Some(key) = self.pending_key.clone() {
                let Some(value_start) = self.buffer.find(Self::ARG_VALUE_START) else {
                    break;
                };
                if value_start > 0 {
                    self.buffer.drain(..value_start);
                }

                let key = key.trim().to_string();
                let tool_name = self.current_tool_name.as_deref().unwrap_or_default();
                let is_string = is_poolside_string_type(tool_name, &key, tools);

                if is_string {
                    self.buffer.drain(..Self::ARG_VALUE_START.len());

                    let index = self.current_index();
                    if self.seen_keys[index].contains(&key) {
                        self.pending_key = None;
                        continue;
                    }

                    self.seen_keys[index].insert(key.clone());
                    let key_json = serde_json::to_string(&key).unwrap_or_else(|_| "\"\"".into());
                    let fragment = if !self.args_started[index] {
                        self.args_started[index] = true;
                        format!("{{{key_json}:\"")
                    } else {
                        format!(",{key_json}:\"")
                    };

                    self.append_current_args_fragment(&fragment);
                    self.streaming_string_value = true;
                    self.update_tool_args(&mut pending_deltas, &fragment);
                    continue;
                }

                let Some(value_end) = self.buffer.find(Self::ARG_VALUE_END) else {
                    break;
                };

                let raw_value = self.buffer[Self::ARG_VALUE_START.len()..value_end]
                    .trim()
                    .to_string();
                self.buffer.drain(..value_end + Self::ARG_VALUE_END.len());
                self.pending_key = None;

                if let Some(fragment) = self.append_arg_fragment(&key, &raw_value) {
                    self.update_tool_args(&mut pending_deltas, &fragment);
                }
                continue;
            }

            let end_pos = self.buffer.find(Self::TOOL_CALL_END);
            let key_pos = self.buffer.find(Self::ARG_KEY_START);
            if let Some(end_pos) = end_pos
                && (key_pos.is_none() || end_pos < key_pos.unwrap())
            {
                self.buffer.drain(..end_pos + Self::TOOL_CALL_END.len());
                let fragment = self.close_args_if_needed();
                self.finish_tool_call();
                if let Some(fragment) = fragment {
                    self.update_tool_args(&mut pending_deltas, &fragment);
                }
                continue;
            }

            let Some(key_pos) = key_pos else {
                break;
            };
            if key_pos > 0 {
                self.buffer.drain(..key_pos);
            }
            let Some(key_end) = self.buffer.find(Self::ARG_KEY_END) else {
                break;
            };
            let key = self.buffer[Self::ARG_KEY_START.len()..key_end].to_string();
            self.buffer.drain(..key_end + Self::ARG_KEY_END.len());
            self.pending_key = Some(key);
        }

        PoolsideV1StreamDelta {
            content,
            tool_calls: pending_deltas.into_values().collect(),
        }
    }

    fn finalize_content(&mut self) -> Option<String> {
        if self.in_tool_call {
            self.buffer.clear();
            self.finish_tool_call();
            return None;
        }

        if self.buffer.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.buffer))
        }
    }

    fn safe_len_before_partial_start(&self) -> usize {
        safe_len_before_partial_suffix(&self.buffer, Self::TOOL_CALL_START)
    }

    fn begin_tool_call(&mut self) {
        self.current_tool_id += 1;
        self.ensure_tool_state();
        self.current_tool_name_sent = false;
        self.current_tool_name = None;
        self.pending_key = None;
        self.streaming_string_value = false;
        self.in_tool_call = true;
    }

    fn finish_tool_call(&mut self) {
        self.in_tool_call = false;
        self.current_tool_name = None;
        self.pending_key = None;
        self.streaming_string_value = false;
    }

    fn revert_last_tool_call_state(&mut self) {
        if self.current_tool_id < 0 {
            return;
        }
        self.tool_call_ids.pop();
        self.streamed_args_for_tool.pop();
        self.args_started.pop();
        self.args_closed.pop();
        self.seen_keys.pop();
        self.current_tool_id -= 1;
    }

    fn ensure_tool_state(&mut self) {
        while self.tool_call_ids.len() <= self.current_index() {
            self.tool_call_ids.push(Uuid::new_v4().to_string());
        }
        while self.streamed_args_for_tool.len() <= self.current_index() {
            self.streamed_args_for_tool.push(String::new());
        }
        while self.args_started.len() <= self.current_index() {
            self.args_started.push(false);
        }
        while self.args_closed.len() <= self.current_index() {
            self.args_closed.push(false);
        }
        while self.seen_keys.len() <= self.current_index() {
            self.seen_keys.push(HashSet::new());
        }
    }

    fn current_index(&self) -> usize {
        self.current_tool_id.max(0) as usize
    }

    fn get_or_create_delta<'a>(
        &self,
        pending: &'a mut BTreeMap<u32, ChatCompletionMessageToolCallChunk>,
    ) -> &'a mut ChatCompletionMessageToolCallChunk {
        let index = self.current_index() as u32;
        pending
            .entry(index)
            .or_insert_with(|| ChatCompletionMessageToolCallChunk {
                index,
                id: None,
                r#type: None,
                function: Some(FunctionCallStream {
                    name: None,
                    arguments: None,
                }),
            })
    }

    fn update_tool_name(
        &mut self,
        pending: &mut BTreeMap<u32, ChatCompletionMessageToolCallChunk>,
        tool_name: String,
    ) {
        let index = self.current_index();
        let id = self.tool_call_ids[index].clone();
        let delta = self.get_or_create_delta(pending);
        delta.id = Some(id);
        delta.r#type = Some(FunctionType::Function);
        let function = delta.function.get_or_insert(FunctionCallStream {
            name: None,
            arguments: None,
        });
        function.name = Some(tool_name);
        function.arguments.get_or_insert_with(String::new);
    }

    fn update_tool_args(
        &self,
        pending: &mut BTreeMap<u32, ChatCompletionMessageToolCallChunk>,
        fragment: &str,
    ) {
        let delta = self.get_or_create_delta(pending);
        let function = delta.function.get_or_insert(FunctionCallStream {
            name: None,
            arguments: None,
        });
        function
            .arguments
            .get_or_insert_with(String::new)
            .push_str(fragment);
    }

    fn append_current_args_fragment(&mut self, fragment: &str) {
        let index = self.current_index();
        self.streamed_args_for_tool[index].push_str(fragment);
    }

    fn append_arg_fragment(&mut self, key: &str, raw_value: &str) -> Option<String> {
        if key.is_empty() || self.seen_keys[self.current_index()].contains(key) {
            return None;
        }

        let value = dynamo_parsers::tool_calling::xml::deserialize_poolside_literal(raw_value);
        let key_json = serde_json::to_string(key).ok()?;
        let value_json = serde_json::to_string(&value).ok()?;
        let index = self.current_index();
        let fragment = if !self.args_started[index] {
            self.args_started[index] = true;
            format!("{{{key_json}:{value_json}")
        } else {
            format!(",{key_json}:{value_json}")
        };

        self.seen_keys[index].insert(key.to_string());
        self.streamed_args_for_tool[index].push_str(&fragment);
        Some(fragment)
    }

    fn close_args_if_needed(&mut self) -> Option<String> {
        let index = self.current_index();
        if self.args_closed[index] {
            return None;
        }
        self.args_closed[index] = true;
        let fragment = if !self.args_started[index] {
            self.streamed_args_for_tool[index] = "{}".to_string();
            "{}".to_string()
        } else {
            self.streamed_args_for_tool[index].push('}');
            "}".to_string()
        };
        Some(fragment)
    }
}

fn append_optional(target: &mut Option<String>, fragment: String) {
    if fragment.is_empty() {
        return;
    }
    target.get_or_insert_with(String::new).push_str(&fragment);
}

fn safe_len_before_partial_suffix(buffer: &str, marker: &str) -> usize {
    for i in (1..marker.len()).rev() {
        if buffer.ends_with(&marker[..i]) {
            return buffer.len() - i;
        }
    }
    buffer.len()
}

fn json_escape_string_content(value: &str) -> String {
    serde_json::to_string(value)
        .ok()
        .and_then(|encoded| {
            encoded
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .map(ToOwned::to_owned)
        })
        .unwrap_or_default()
}

fn is_poolside_string_type(
    tool_name: &str,
    arg_name: &str,
    tools: Option<&[dynamo_parsers::tool_calling::ToolDefinition]>,
) -> bool {
    let Some(tool) = tools.and_then(|tools| tools.iter().find(|tool| tool.name == tool_name))
    else {
        return false;
    };
    tool.parameters
        .as_ref()
        .and_then(|schema| schema.get("properties"))
        .and_then(|properties| properties.get(arg_name))
        .and_then(|arg_schema| arg_schema.get("type"))
        .and_then(|arg_type| arg_type.as_str())
        == Some("string")
}

/// State tracking for an individual choice during jail processing
#[derive(Debug, Clone)]
struct ChoiceJailState {
    /// The choice index (0, 1, 2, ...)
    index: u32,
    /// Whether this choice is currently jailed
    is_jailed: bool,
    /// Accumulated content for this choice while jailed
    accumulated_content: String,
    /// Accumulated logprobs for this choice while jailed.
    /// Logprobs from each jailed chunk are appended so the full token-level
    /// log-probability information is preserved when the jail emits.
    accumulated_logprobs: Option<ChatChoiceLogprobs>,
    /// Buffer for partial marker matches across chunks
    partial_match_buffer: String,
    /// Stream finish reason
    stream_finish_reason: Option<FinishReason>,
    /// Number of tool calls already emitted for this choice
    emitted_tool_calls_count: usize,
    /// Exact tool calls already emitted in required-mode immediate jail.
    emitted_required_tool_call_keys: HashMap<String, usize>,
    /// Stop the upstream stream after emitting/suppressing a duplicate
    /// required-mode tool call.
    terminate_after_tool_call: bool,
    /// Reasoning content collected while waiting for a suitable emission.
    pending_reasoning_content: Option<String>,
    /// Poolside/vLLM streams schema-declared string args incrementally instead
    /// of buffering until `</tool_call>`.
    poolside_v1_state: PoolsideV1StreamState,
}

fn create_choice_stream(
    index: u32,
    role: Option<Role>,
    content: &str,
    tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    finish_reason: Option<FinishReason>,
    logprobs: Option<ChatChoiceLogprobs>,
) -> ChatChoiceStream {
    #[allow(deprecated)]
    ChatChoiceStream {
        index,
        delta: ChatCompletionStreamResponseDelta {
            role,
            content: Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(
                content.to_string(),
            )),
            tool_calls,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason,
        logprobs,
    }
}

fn create_choice_stream_optional_content(
    index: u32,
    role: Option<Role>,
    content: Option<String>,
    tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    finish_reason: Option<FinishReason>,
    logprobs: Option<ChatChoiceLogprobs>,
) -> ChatChoiceStream {
    #[allow(deprecated)]
    ChatChoiceStream {
        index,
        delta: ChatCompletionStreamResponseDelta {
            role,
            content: content.map(dynamo_protocols::types::ChatCompletionMessageContent::Text),
            tool_calls,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason,
        logprobs,
    }
}

impl ChoiceJailState {
    /// Create a new jail state for a choice
    fn new(index: u32, starts_jailed: bool) -> Self {
        Self {
            index,
            is_jailed: starts_jailed,
            accumulated_content: String::new(),
            accumulated_logprobs: None,
            partial_match_buffer: String::new(),
            stream_finish_reason: None,
            emitted_tool_calls_count: 0,
            emitted_required_tool_call_keys: HashMap::new(),
            terminate_after_tool_call: false,
            pending_reasoning_content: None,
            poolside_v1_state: PoolsideV1StreamState::default(),
        }
    }

    /// Add content and logprobs to this choice's accumulation
    fn accumulate(&mut self, content: &str, logprobs: Option<&ChatChoiceLogprobs>) {
        if self.is_jailed {
            self.accumulated_content.push_str(content);
            // Accumulate logprobs so they are preserved across jailed chunks.
            if let Some(lp) = logprobs {
                let state_lps = self.accumulated_logprobs.get_or_insert(ChatChoiceLogprobs {
                    content: None,
                    refusal: None,
                });
                if let Some(content_lps) = &lp.content {
                    state_lps
                        .content
                        .get_or_insert_with(Vec::new)
                        .extend(content_lps.clone());
                }
                if let Some(refusal_lps) = &lp.refusal {
                    state_lps
                        .refusal
                        .get_or_insert_with(Vec::new)
                        .extend(refusal_lps.clone());
                }
            }
        }
    }

    /// Consume the accumulated logprobs, replacing them with `None`.
    fn take_accumulated_logprobs(&mut self) -> Option<ChatChoiceLogprobs> {
        self.accumulated_logprobs.take()
    }

    /// End jailing and return the accumulated content
    fn end_jail(&mut self) -> String {
        self.is_jailed = false;
        self.accumulated_logprobs = None;
        std::mem::take(&mut self.accumulated_content)
    }

    fn should_suppress_non_tool_trailing(&self, jail_stream: &JailedStream) -> bool {
        matches!(
            &jail_stream.jail_mode,
            JailMode::Immediate {
                format: ToolChoiceFormat::ArrayOfTools { .. }
            }
        ) && self.emitted_tool_calls_count > 0
    }

    fn should_suppress_post_tool_whitespace(&self, content: &str) -> bool {
        self.emitted_tool_calls_count > 0 && content.trim().is_empty()
    }

    fn should_guard_required_duplicates(&self, jail_stream: &JailedStream) -> bool {
        matches!(
            &jail_stream.jail_mode,
            JailMode::Immediate {
                format: ToolChoiceFormat::ArrayOfTools {
                    terminal_after_first: false,
                }
            }
        )
    }

    fn tool_call_dedupe_key(tool_call: &ChatCompletionMessageToolCallChunk) -> Option<String> {
        let function = tool_call.function.as_ref()?;
        let name = function.name.as_deref()?.trim();
        if name.is_empty() {
            return None;
        }
        let args = function.arguments.as_deref().unwrap_or_default().trim();
        let normalized_args = serde_json::from_str::<serde_json::Value>(args)
            .map(Self::canonicalize_json_value)
            .ok()
            .and_then(|value| serde_json::to_string(&value).ok())
            .unwrap_or_else(|| args.to_string());
        Some(format!("{name}\x1f{normalized_args}"))
    }

    fn canonicalize_json_value(value: serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Array(values) => serde_json::Value::Array(
                values
                    .into_iter()
                    .map(Self::canonicalize_json_value)
                    .collect(),
            ),
            serde_json::Value::Object(map) => {
                let mut entries: Vec<_> = map.into_iter().collect();
                entries.sort_by(|(left, _), (right, _)| left.cmp(right));
                serde_json::Value::Object(
                    entries
                        .into_iter()
                        .map(|(key, value)| (key, Self::canonicalize_json_value(value)))
                        .collect(),
                )
            }
            other => other,
        }
    }

    fn prepare_tool_call_emission(
        &mut self,
        choice: &mut ChatChoiceStream,
        jail_stream: &JailedStream,
    ) -> usize {
        let Some(tool_calls) = choice.delta.tool_calls.take() else {
            return 0;
        };

        if !self.should_guard_required_duplicates(jail_stream) {
            let len = tool_calls.len();
            choice.delta.tool_calls = Some(tool_calls);
            return len;
        }

        let mut kept = Vec::with_capacity(tool_calls.len());
        for tool_call in tool_calls {
            if let Some(key) = Self::tool_call_dedupe_key(&tool_call) {
                if let Some(count) = self.emitted_required_tool_call_keys.get_mut(&key) {
                    *count += 1;
                    if *count > MAX_SUPPRESSED_REQUIRED_TOOL_DUPLICATES + 1 {
                        self.terminate_after_tool_call = true;
                        tracing::warn!(
                            duplicate_count = *count,
                            "tool_choice=required: repeated duplicate tool call detected; terminating stream"
                        );
                        break;
                    }
                    tracing::warn!(
                        duplicate_count = *count,
                        "tool_choice=required: duplicate tool call detected; suppressing duplicate"
                    );
                    continue;
                }
                self.emitted_required_tool_call_keys.insert(key, 1);
            }
            kept.push(tool_call);
        }

        let len = kept.len();
        if len > 0 {
            choice.delta.tool_calls = Some(kept);
        }
        len
    }

    fn take_terminate_after_tool_call(&mut self) -> bool {
        let terminate = self.terminate_after_tool_call;
        self.terminate_after_tool_call = false;
        terminate
    }

    /// Process incoming content and return what should be emitted (if anything)
    async fn process_content(
        &mut self,
        choice: &ChatChoiceStream,
        content: &str,
        jail_stream: &JailedStream,
    ) -> Vec<ChoiceEmission> {
        if jail_stream.is_poolside_v1_parser() {
            return self.process_poolside_v1_content(choice, content, jail_stream);
        }

        let mut emissions = Vec::new();
        if !self.is_jailed {
            // Use the marker matcher to detect complete/partial markers
            let match_result = jail_stream
                .marker_matcher
                .process_chunk(content, &self.partial_match_buffer);

            match match_result {
                MatchResult::Complete {
                    prefix,
                    marker,
                    suffix,
                    ..
                } => {
                    // Emit prefix if any
                    if !prefix.is_empty()
                        && !self.should_suppress_non_tool_trailing(jail_stream)
                        && !self.should_suppress_post_tool_whitespace(&prefix)
                    {
                        #[allow(deprecated)]
                        let prefix_choice = create_choice_stream(
                            choice.index,
                            choice.delta.role,
                            &prefix,
                            None,
                            choice.finish_reason,
                            choice.logprobs.clone(),
                        );
                        emissions.push(ChoiceEmission::PassThrough(prefix_choice));
                    }

                    // Build the potential full content
                    let full_content = format!("{}{}", marker, suffix);

                    // Check if this already contains the end marker
                    let (should_end, split_pos) = jail_stream.should_end_jail(&full_content).await;

                    if should_end {
                        // Complete tool call found in this chunk
                        let (jailed_part, trailing_part) = full_content.split_at(split_pos);

                        // Create the tool call choice
                        let tool_choice = jail_stream
                            .create_tool_call_choice(
                                choice.index,
                                jailed_part,
                                choice,
                                self.emitted_tool_calls_count,
                                false, // streaming early-exit, no EOF recovery
                            )
                            .await;

                        let mut tool_choice = tool_choice;
                        let emitted_tool_calls =
                            self.prepare_tool_call_emission(&mut tool_choice, jail_stream);
                        if emitted_tool_calls > 0 {
                            self.emitted_tool_calls_count += emitted_tool_calls;
                            emissions.push(ChoiceEmission::ToolCall(tool_choice));
                        } else if !self.terminate_after_tool_call {
                            emissions.push(ChoiceEmission::Content(tool_choice));
                        }

                        // Handle trailing content if any
                        if !trailing_part.is_empty() {
                            if jail_stream.should_start_jail(trailing_part) {
                                self.is_jailed = true;
                                self.accumulated_content = trailing_part.to_string();
                                // No logprobs to seed here — they were already emitted with the tool call
                                self.accumulated_logprobs = None;
                            } else if self.should_suppress_post_tool_whitespace(trailing_part) {
                                // Drop whitespace emitted after a parsed tool call. Non-streaming
                                // aggregation already removes this, and streaming should match it.
                            } else if self.should_suppress_non_tool_trailing(jail_stream) {
                                self.is_jailed = true;
                                self.accumulated_content = trailing_part.to_string();
                                self.accumulated_logprobs = None;
                            } else {
                                #[allow(deprecated)]
                                let trailing_choice = create_choice_stream(
                                    choice.index,
                                    choice.delta.role,
                                    trailing_part,
                                    None,
                                    choice.finish_reason,
                                    choice.logprobs.clone(),
                                );
                                emissions.push(ChoiceEmission::Trailing(trailing_choice));
                            }
                        }
                    } else {
                        // Start jailing with the marker and suffix
                        self.is_jailed = true;
                        self.accumulated_content = full_content;
                        // Seed accumulated logprobs with this chunk's logprobs
                        self.accumulated_logprobs = choice.logprobs.clone();
                    }

                    self.partial_match_buffer.clear();
                }

                MatchResult::Partial {
                    prefix,
                    partial,
                    possible_patterns,
                } => {
                    // Emit the safe prefix
                    if !prefix.is_empty()
                        && !self.should_suppress_non_tool_trailing(jail_stream)
                        && !self.should_suppress_post_tool_whitespace(&prefix)
                    {
                        #[allow(deprecated)]
                        let prefix_choice = create_choice_stream(
                            choice.index,
                            choice.delta.role,
                            &prefix,
                            None,
                            choice.finish_reason,
                            choice.logprobs.clone(),
                        );
                        emissions.push(ChoiceEmission::PassThrough(prefix_choice));
                    }

                    // Hold the partial for next chunk
                    self.partial_match_buffer = partial;

                    tracing::trace!(
                        "Choice {} holding partial '{}' for patterns: {:?}",
                        choice.index,
                        self.partial_match_buffer,
                        possible_patterns
                    );
                }

                MatchResult::None { content } => {
                    // Check if this content (combined with partial buffer) should start jailing
                    let combined_content = if self.partial_match_buffer.is_empty() {
                        content.clone()
                    } else {
                        format!("{}{}", self.partial_match_buffer, content)
                    };

                    if jail_stream.should_start_jail(&combined_content) {
                        // Start jailing with the combined content
                        self.is_jailed = true;
                        self.accumulated_content = combined_content;
                        // Seed accumulated logprobs with this chunk's logprobs
                        self.accumulated_logprobs = choice.logprobs.clone();
                        self.partial_match_buffer.clear();
                    } else if self.should_suppress_post_tool_whitespace(&content) {
                        self.partial_match_buffer.clear();
                    } else if self.should_suppress_non_tool_trailing(jail_stream) {
                        self.is_jailed = true;
                        self.accumulated_content = combined_content;
                        self.accumulated_logprobs = choice.logprobs.clone();
                        self.partial_match_buffer.clear();
                    } else {
                        // No markers - emit everything
                        if !content.is_empty() {
                            #[allow(deprecated)]
                            let pass_through_choice = create_choice_stream(
                                choice.index,
                                choice.delta.role,
                                &content,
                                None,
                                choice.finish_reason,
                                choice.logprobs.clone(),
                            );
                            emissions.push(ChoiceEmission::PassThrough(pass_through_choice));
                        }
                        self.partial_match_buffer.clear();
                    }
                }
            }
        } else {
            // Already jailed - accumulate content AND logprobs, then check for unjail
            self.accumulate(content, choice.logprobs.as_ref());

            let (should_end, split_pos) =
                jail_stream.should_end_jail(&self.accumulated_content).await;

            if should_end {
                // Take accumulated logprobs before borrowing accumulated_content
                let jail_logprobs = self.take_accumulated_logprobs();

                // Split the content
                let (jailed_part, trailing_part) = self.accumulated_content.split_at(split_pos);
                let trailing_owned = trailing_part.to_string();
                let jailed_owned = jailed_part.to_string();

                // Create the unjailed choice, using accumulated logprobs
                let mut unjailed_choice = jail_stream
                    .create_tool_call_choice(
                        choice.index,
                        &jailed_owned,
                        choice,
                        self.emitted_tool_calls_count,
                        false, // streaming unjail, no EOF recovery
                    )
                    .await;
                unjailed_choice.logprobs = jail_logprobs;

                // Determine emission type based on whether tool calls were parsed
                let emitted_tool_calls =
                    self.prepare_tool_call_emission(&mut unjailed_choice, jail_stream);
                if emitted_tool_calls > 0 {
                    self.emitted_tool_calls_count += emitted_tool_calls;
                    emissions.push(ChoiceEmission::ToolCall(unjailed_choice));
                } else if !self.terminate_after_tool_call {
                    emissions.push(ChoiceEmission::Content(unjailed_choice));
                }

                // End jailing before processing trailing content
                self.end_jail();

                // Handle trailing content if any
                if !trailing_owned.is_empty() {
                    if jail_stream.should_start_jail(&trailing_owned) {
                        self.is_jailed = true;
                        self.accumulated_content = trailing_owned;
                    } else if self.should_suppress_post_tool_whitespace(&trailing_owned) {
                        // Drop whitespace emitted after a parsed tool call. Non-streaming
                        // aggregation already removes this, and streaming should match it.
                    } else if self.should_suppress_non_tool_trailing(jail_stream) {
                        self.is_jailed = true;
                        self.accumulated_content = trailing_owned;
                        self.accumulated_logprobs = None;
                    } else {
                        #[allow(deprecated)]
                        let trailing_choice = create_choice_stream(
                            choice.index,
                            choice.delta.role,
                            &trailing_owned,
                            None,
                            choice.finish_reason,
                            choice.logprobs.clone(),
                        );
                        emissions.push(ChoiceEmission::Trailing(trailing_choice));
                    }
                }
            }
            // If not unjailing, don't emit anything (still accumulating)
        }

        // create_tool_call_choice drops finish_reason; the worker may also put it on a
        // suppressed post-tool whitespace chunk. Ensure the chunk's finish_reason rides
        // the last emission, or emit a terminal chunk if none. tool_calls -> ToolCalls.
        // Ensure the chunk's finish_reason is not dropped. create_tool_call_choice
        // omits it, and the worker may put it on a suppressed post-tool whitespace
        // chunk. tool_calls -> ToolCalls.
        if let Some(fr) = choice.finish_reason {
            let fr = if self.emitted_tool_calls_count > 0 {
                FinishReason::ToolCalls
            } else {
                fr
            };
            if let Some(last) = emissions.last_mut() {
                // Auto/marker path: ride finish on the last emission. Terminal
                // tool-choice modes emit their own finish + break, so don't double.
                if matches!(jail_stream.jail_mode, JailMode::MarkerBased) {
                    last.choice_mut().finish_reason = Some(fr);
                    let keep = emissions.len() - 1;
                    for e in emissions.iter_mut().take(keep) {
                        e.choice_mut().finish_reason = None;
                    }
                }
            } else {
                // Finish on a suppressed chunk (no emission). Terminal modes break
                // before here, so a terminal finish chunk is safe for any mode.
                emissions.push(ChoiceEmission::PassThrough(ChatChoiceStream {
                    index: choice.index,
                    delta: ChatCompletionStreamResponseDelta {
                        role: Some(Role::Assistant),
                        content: None,
                        tool_calls: None,
                        function_call: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: Some(fr),
                    logprobs: None,
                }));
            }
        }

        emissions
    }

    fn process_poolside_v1_content(
        &mut self,
        choice: &ChatChoiceStream,
        content: &str,
        jail_stream: &JailedStream,
    ) -> Vec<ChoiceEmission> {
        if !jail_stream.poolside_v1_tools_enabled() {
            return vec![ChoiceEmission::PassThrough(create_choice_stream(
                choice.index,
                choice.delta.role,
                content,
                None,
                choice.finish_reason,
                choice.logprobs.clone(),
            ))];
        }

        let parsed = self
            .poolside_v1_state
            .process_delta(content, jail_stream.tool_definitions.as_deref());

        if parsed.content.is_none() && parsed.tool_calls.is_empty() {
            return Vec::new();
        }

        let has_tool_calls = !parsed.tool_calls.is_empty();
        let choice = create_choice_stream_optional_content(
            choice.index,
            choice.delta.role.or(Some(Role::Assistant)),
            parsed.content,
            has_tool_calls.then_some(parsed.tool_calls),
            choice.finish_reason,
            choice.logprobs.clone(),
        );

        if has_tool_calls {
            vec![ChoiceEmission::ToolCall(choice)]
        } else {
            vec![ChoiceEmission::PassThrough(choice)]
        }
    }

    /// Finalize any remaining content when stream ends
    async fn finalize(&mut self, jail_stream: &JailedStream) -> Option<ChoiceEmission> {
        if jail_stream.is_poolside_v1_parser() {
            let content = self.poolside_v1_state.finalize_content()?;
            let final_choice = create_choice_stream_optional_content(
                self.index,
                Some(Role::Assistant),
                Some(content),
                None,
                self.stream_finish_reason,
                None,
            );
            return Some(ChoiceEmission::Content(final_choice));
        }

        if self.is_jailed && !self.accumulated_content.is_empty() {
            // Create a dummy choice for the method call
            #[allow(deprecated)]
            let dummy_choice = create_choice_stream(
                self.index,
                Some(Role::Assistant),
                &self.accumulated_content,
                None,
                self.stream_finish_reason, // For the accumulated content, assign the original stream finish reason, otherwise it will get lost
                self.accumulated_logprobs.clone(),
            );

            let mut final_choice = jail_stream
                .create_tool_call_choice(
                    self.index,
                    &self.accumulated_content,
                    &dummy_choice,
                    self.emitted_tool_calls_count,
                    true, // finalize: enable EOF recovery for missing-end-token / truncated-JSON
                )
                .await;
            // Attach the full accumulated logprobs to the final choice
            final_choice.logprobs = self.take_accumulated_logprobs();

            // Preserve any pending reasoning content collected while jailed.
            if let Some(pending_reasoning) = self.pending_reasoning_content.take() {
                if let Some(existing_reasoning) = final_choice.delta.reasoning_content.as_mut() {
                    existing_reasoning.push_str(&pending_reasoning);
                } else {
                    final_choice.delta.reasoning_content = Some(pending_reasoning);
                }
            }

            let emitted_tool_calls =
                self.prepare_tool_call_emission(&mut final_choice, jail_stream);
            if emitted_tool_calls > 0 {
                self.emitted_tool_calls_count += emitted_tool_calls;
            }

            // End jailing
            self.end_jail();

            // Determine emission type
            if final_choice.delta.tool_calls.is_some() {
                // EOF-jailed tool call: ensure terminal finish_reason=tool_calls.
                if final_choice.finish_reason.is_none() {
                    final_choice.finish_reason = Some(FinishReason::ToolCalls);
                }
                Some(ChoiceEmission::ToolCall(final_choice))
            } else if self.terminate_after_tool_call {
                None
            } else {
                // Preserve worker finish_reason for finalized content.
                if final_choice.finish_reason.is_none() {
                    final_choice.finish_reason = self.stream_finish_reason;
                }
                Some(ChoiceEmission::Content(final_choice))
            }
        } else {
            None
        }
    }
}

/// Collection of choice jail states with deterministic ordering
#[derive(Debug, Clone)]
struct ChoiceJailStateCollection {
    /// Vec of states, always kept sorted by choice index for deterministic iteration
    states: Vec<ChoiceJailState>,
}

impl ChoiceJailStateCollection {
    /// Create a new empty collection
    fn new() -> Self {
        Self { states: Vec::new() }
    }

    /// Get or create state for a choice index
    fn get_or_create_state(&mut self, index: u32, starts_jailed: bool) -> &mut ChoiceJailState {
        // Find the position where this index should be
        match self.states.binary_search_by_key(&index, |s| s.index) {
            Ok(pos) => {
                // Found existing state
                if starts_jailed
                    && !self.states[pos].is_jailed
                    && self.states[pos].accumulated_content.is_empty()
                    && self.states[pos].emitted_tool_calls_count == 0
                {
                    self.states[pos].is_jailed = true;
                }
                &mut self.states[pos]
            }
            Err(insert_pos) => {
                // Need to create new state
                let new_state = ChoiceJailState::new(index, starts_jailed);
                self.states.insert(insert_pos, new_state);
                &mut self.states[insert_pos]
            }
        }
    }
}

/// Emission mode for handling multiple choices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmissionMode {
    /// Pack multiple choices in the same chunk (default, matches original behavior)
    #[default]
    Packed,
    /// Emit one choice per chunk for OpenAI compatibility
    SingleChoicePerChunk,
}

/// A stream transformer that can "jail" tokens based on configurable start/end sequences
/// When jailed, tokens are accumulated rather than yielded immediately
/// When the jail ends (via end sequence or stream completion), accumulated content is processed and released
pub struct JailedStream {
    jail_start_sequences: Vec<String>,
    jail_end_sequences: Vec<String>,
    tool_call_parser: Option<String>,
    /// When set, only tool calls with this name are emitted (enforces tool_choice=named
    /// when a tool_call_parser is active and the parser-aware MarkerBased path is used).
    named_tool_name: Option<String>,
    tool_definitions: Option<Vec<dynamo_parsers::tool_calling::ToolDefinition>>,
    emission_mode: EmissionMode,
    marker_matcher: MarkerMatcher,
    jail_mode: JailMode,
    defer_terminal_until_usage: bool,
}

impl JailedStream {
    /// Create a new builder for configuring a JailedStream
    pub fn builder() -> JailedStreamBuilder {
        JailedStreamBuilder::new()
    }

    /// Whether the jail starts already-jailed (tool_choice=required/named path).
    fn is_immediate(&self) -> bool {
        matches!(self.jail_mode, JailMode::Immediate { .. })
    }

    fn is_poolside_v1_parser(&self) -> bool {
        self.tool_call_parser.as_deref() == Some("poolside_v1")
            && matches!(self.jail_mode, JailMode::MarkerBased)
    }

    fn poolside_v1_tools_enabled(&self) -> bool {
        self.tool_definitions
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
    }
    /// Apply jail stream transformation with finish_reason fix
    /// This is a convenience method that applies both apply() and fix_finish_reason()
    pub fn apply_with_finish_reason<S>(
        self,
        stream: S,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        let jail_mode = self.jail_mode.clone();
        let named_tool_active = self.named_tool_name.is_some();
        let jailed_stream = self.apply(stream);
        JailedStream::fix_finish_reason(jailed_stream, jail_mode, named_tool_active)
    }

    /// Apply the jail transformation to a stream of chat completion responses
    /// Consumes self and returns the transformed stream
    pub fn apply<S>(
        self,
        stream: S,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        // Use the stream! macro for cleaner async stream processing
        stream! {
            // State variables - clean architecture with choice state collection
            let mut choice_states = ChoiceJailStateCollection::new();
            // Track Annotated metadata for preservation
            let mut last_annotated_id: Option<String> = None;
            let mut last_annotated_event: Option<String> = None;
            let mut last_annotated_comment: Option<Vec<String>> = None;
            // Track stream response metadata so finalization chunks carry real values
            let mut last_stream_id = String::new();
            let mut last_stream_model = String::new();
            let mut last_stream_created: u32 = 0;
            let mut waiting_for_usage_after_terminal = false;

            // Pin the stream for iteration (stack pinning is more efficient)
            tokio::pin!(stream);


            // Process each item in the stream
            while let Some(response) = stream.next().await {
                if let Some(chat_response) = response.data.as_ref() {
                    last_stream_id.clone_from(&chat_response.inner.id);
                    last_stream_model.clone_from(&chat_response.inner.model);
                    last_stream_created = chat_response.inner.created;

                    let mut all_emissions = Vec::new();
                    let mut forced_terminal_indices = Vec::new();

                    if chat_response.inner.choices.is_empty() {
                        let is_usage_chunk = chat_response.inner.usage.is_some();
                        // No choices processed (e.g., usage-only chunk)
                        // Pass through as-is to preserve usage and other metadata
                        yield response;
                        if waiting_for_usage_after_terminal && is_usage_chunk {
                            break;
                        }
                        continue;
                    }

                    if waiting_for_usage_after_terminal {
                        continue;
                    }

                    // Process each choice independently using the new architecture
                    for choice in &chat_response.inner.choices {
                        if choice
                            .delta
                            .tool_calls
                            .as_ref()
                            .is_some_and(|tool_calls| !tool_calls.is_empty())
                        {
                            let choice_state = choice_states.get_or_create_state(choice.index, false);
                            if choice.finish_reason.is_some() {
                                choice_state.stream_finish_reason = choice.finish_reason;
                            }
                            all_emissions.push(ChoiceEmission::PassThrough(choice.clone()));
                            continue;
                        }

                        if let Some(ref content) = choice.delta.content {
                            // Jailing only applies to text content
                            let text_content = match content {
                                dynamo_protocols::types::ChatCompletionMessageContent::Text(text) => Some(text.as_str()),
                                dynamo_protocols::types::ChatCompletionMessageContent::Parts(_) => None,
                            };

                            if let Some(text) = text_content {
                                let starts_jailed = matches!(self.jail_mode, JailMode::Immediate { .. });
                                let choice_state = choice_states.get_or_create_state(choice.index, starts_jailed);

                                if let Some(reasoning_content) = &choice.delta.reasoning_content {
                                    let pending = choice_state
                                        .pending_reasoning_content
                                        .get_or_insert_with(String::new);
                                    pending.push_str(reasoning_content);
                                }

                                // Store metadata when any choice becomes jailed (first time only)
                                if !choice_state.is_jailed && self.should_start_jail(text)
                                    && last_annotated_id.is_none() {
                                        last_annotated_id = response.id.clone();
                                        last_annotated_event = response.event.clone();
                                        last_annotated_comment = response.comment.clone();
                                    }

                                // Track actual stream finish reason in the choice state
                                choice_state.stream_finish_reason = choice.finish_reason;

                                // Process this choice and get emissions
                                let mut emissions = choice_state.process_content(choice, text, &self).await;
                                if !emissions.is_empty()
                                    && let Some(reasoning) = choice_state.pending_reasoning_content.take()
                                    && let Some(first) = emissions.first_mut()
                                {
                                    first.choice_mut().delta.reasoning_content = Some(reasoning);
                                }
                                if choice_state.take_terminate_after_tool_call()
                                    && !forced_terminal_indices.contains(&choice.index)
                                {
                                    forced_terminal_indices.push(choice.index);
                                }
                                all_emissions.extend(emissions);
                            }
                            // For multimodal content, pass through unchanged (no jailing)
                        } else {
                            // Handle choices without content (e.g., final chunks with finish_reason)
                            // Only filter out if this choice was ever jailed and lacks role
                            // (to avoid aggregator issues with deltas missing role after unjail)
                            let choice_state = choice_states.get_or_create_state(choice.index, false);
                            // Also track stream finish reason from content-less final chunks
                            // (e.g. finish_reason=Stop arriving in a chunk with content=None) so
                            // the Immediate-mode finalize path can emit the correct finish_reason.
                            if choice.finish_reason.is_some() {
                                choice_state.stream_finish_reason = choice.finish_reason;
                            }
                            let was_ever_jailed = !choice_state.accumulated_content.is_empty() || choice_state.is_jailed;

                            let should_emit = choice.delta.role.is_some()
                                || choice.delta.tool_calls.is_some()
                                || !was_ever_jailed; // Always pass through if never jailed

                            if should_emit {
                                let pass_through_choice = ChatChoiceStream {
                                    index: choice.index,
                                    delta: choice.delta.clone(),
                                    finish_reason: choice.finish_reason,
                                    logprobs: choice.logprobs.clone(),
                                };
                                all_emissions.push(ChoiceEmission::PassThrough(pass_through_choice));
                            }
                        }
                    }

                    // Emit all results based on emission mode
                    if !all_emissions.is_empty() {
                        // Group emissions by type for proper ordering and separation
                        let mut tool_content_emissions = Vec::new();
                        let mut trailing_emissions = Vec::new();
                        let mut passthrough_emissions = Vec::new();
                        let terminal_tool_choice = matches!(
                            self.jail_mode,
                            JailMode::Immediate {
                                format: ToolChoiceFormat::SingleObject { .. }
                                    | ToolChoiceFormat::ArrayOfTools {
                                        terminal_after_first: true,
                                    }
                            }
                        );
                        let mut terminal_tool_choice_indices = forced_terminal_indices.clone();

                        for emission in all_emissions {
                            match emission {
                                ChoiceEmission::PassThrough(_) => passthrough_emissions.push(emission),
                                ChoiceEmission::ToolCall(choice) => {
                                    if terminal_tool_choice
                                        && choice
                                            .delta
                                            .tool_calls
                                            .as_ref()
                                            .is_some_and(|tool_calls| !tool_calls.is_empty())
                                        && !terminal_tool_choice_indices.contains(&choice.index)
                                    {
                                        terminal_tool_choice_indices.push(choice.index);
                                    }
                                    tool_content_emissions.push(ChoiceEmission::ToolCall(choice));
                                }
                                ChoiceEmission::Content(_) => {
                                    tool_content_emissions.push(emission);
                                }
                                ChoiceEmission::Trailing(_) => {
                                    trailing_emissions.push(emission);
                                }
                            }
                        }

                        // Emit tool calls and content with preserved metadata
                        if !tool_content_emissions.is_empty() {
                            let preserved_metadata = (
                                last_annotated_id.clone(),
                                last_annotated_event.clone(),
                                last_annotated_comment.clone(),
                            );
                            let responses = self.emit_choice_emissions(tool_content_emissions, chat_response, preserved_metadata);
                            for emitted_response in responses {
                                yield emitted_response;
                            }
                        }

                        // A named forced tool choice is complete after the first valid
                        // call. Required tool_choice behaves the same only when the
                        // request has exactly one available tool; with multiple tools,
                        // required remains multi-call capable.
                        if !terminal_tool_choice_indices.is_empty() {
                            let final_choices = terminal_tool_choice_indices
                                .into_iter()
                                .map(|choice_index| ChatChoiceStream {
                                    index: choice_index,
                                    delta: ChatCompletionStreamResponseDelta {
                                        role: Some(Role::Assistant),
                                        content: None,
                                        tool_calls: None,
                                        function_call: None,
                                        refusal: None,
                                        reasoning_content: None,
                                    },
                                    finish_reason: Some(FinishReason::ToolCalls),
                                    logprobs: None,
                                })
                                .collect();
                            let mut final_response = chat_response.clone();
                            final_response.inner.choices = final_choices;
                            yield Annotated {
                                data: Some(final_response),
                                id: response.id.clone(),
                                event: response.event.clone(),
                                comment: response.comment.clone(),
                                error: None,
                            };
                            if self.defer_terminal_until_usage {
                                waiting_for_usage_after_terminal = true;
                                continue;
                            }
                            break;
                        }

                        // Emit trailing content separately (always as individual chunks)
                        if !trailing_emissions.is_empty() {
                            let preserved_metadata = (
                                last_annotated_id.clone(),
                                last_annotated_event.clone(),
                                last_annotated_comment.clone(),
                            );
                            let responses = self.emit_choice_emissions(trailing_emissions, chat_response, preserved_metadata);
                            for emitted_response in responses {
                                yield emitted_response;
                            }
                        }

                        // Emit pass-through content with current metadata
                        if !passthrough_emissions.is_empty() {
                            let current_metadata = (response.id.clone(), response.event.clone(), response.comment.clone());
                            let responses = self.emit_choice_emissions(passthrough_emissions, chat_response, current_metadata);
                            for emitted_response in responses {
                                yield emitted_response;
                            }
                        }
                    } else if !forced_terminal_indices.is_empty() {
                        let final_choices = forced_terminal_indices
                            .into_iter()
                            .map(|choice_index| ChatChoiceStream {
                                index: choice_index,
                                delta: ChatCompletionStreamResponseDelta {
                                    role: Some(Role::Assistant),
                                    content: None,
                                    tool_calls: None,
                                    function_call: None,
                                    refusal: None,
                                    reasoning_content: None,
                                },
                                finish_reason: Some(FinishReason::ToolCalls),
                                logprobs: None,
                            })
                            .collect();
                        let mut final_response = chat_response.clone();
                        final_response.inner.choices = final_choices;
                        yield Annotated {
                            data: Some(final_response),
                            id: response.id.clone(),
                            event: response.event.clone(),
                            comment: response.comment.clone(),
                            error: None,
                        };
                        if self.defer_terminal_until_usage {
                            waiting_for_usage_after_terminal = true;
                            continue;
                        }
                        break;
                    }
                } else {
                    // No response data, pass through as-is
                    yield response;
                }
            }

            // Stream ended - finalize any remaining jailed choices
            let mut final_emissions = Vec::new();
            for state in choice_states.states.iter_mut() {
                if let Some(emission) = state.finalize(&self).await {
                    final_emissions.push(emission);
                }
            }

            if !final_emissions.is_empty() {
                tracing::debug!("Stream ended while jailed, releasing accumulated content");
                // Create a finalization response carrying forward real stream metadata
                let dummy_response = NvCreateChatCompletionStreamResponse {
                    inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                        id: last_stream_id,
                    object: "chat.completion.chunk".to_string(),
                        created: last_stream_created,
                        model: last_stream_model,
                    choices: Vec::new(),
                    usage: None,
                    service_tier: None,
                    system_fingerprint: None,
                    },
                    nvext: None,
                };

                let final_metadata = (last_annotated_id, last_annotated_event, last_annotated_comment);
                let responses = self.emit_choice_emissions(final_emissions, &dummy_response, final_metadata);
                for emitted_response in responses {
                    yield emitted_response;
                }
            }
        }
    }

    /// Emit choice emissions based on the configured emission mode
    fn emit_choice_emissions(
        &self,
        emissions: Vec<ChoiceEmission>,
        base_response: &NvCreateChatCompletionStreamResponse,
        annotated_metadata: (Option<String>, Option<String>, Option<Vec<String>>),
    ) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
        if emissions.is_empty() {
            return Vec::new();
        }

        let (id, event, comment) = annotated_metadata;

        match self.emission_mode {
            EmissionMode::Packed => {
                // Pack all choices into a single response
                let mut response = base_response.clone();
                response.inner.choices = emissions.into_iter().map(|e| e.into_choice()).collect();

                vec![Annotated {
                    data: Some(response),
                    id,
                    event,
                    comment,
                    error: None,
                }]
            }
            EmissionMode::SingleChoicePerChunk => {
                // Emit each choice in a separate response
                emissions
                    .into_iter()
                    .map(|emission| {
                        let mut response = base_response.clone();
                        response.inner.choices = vec![emission.into_choice()];

                        Annotated {
                            data: Some(response),
                            id: id.clone(),
                            event: event.clone(),
                            comment: comment.clone(),
                            error: None,
                        }
                    })
                    .collect()
            }
        }
    }

    /// Check if content matches any jail start patterns
    fn should_start_jail(&self, content: &str) -> bool {
        // Path 1: Check configured start sequences
        let sequence_match = !self.jail_start_sequences.is_empty()
            && self
                .jail_start_sequences
                .iter()
                .any(|seq| content.contains(seq));

        // Path 2: Check for tool call start pattern
        let tool_call_match = self.tool_call_parser.is_some()
            && detect_tool_call_start(content, self.tool_call_parser.as_deref()).unwrap_or(false);

        sequence_match || tool_call_match
    }

    /// Check if accumulated content should end jail
    async fn should_end_jail(&self, accumulated_content: &str) -> (bool, usize) {
        match &self.jail_mode {
            JailMode::MarkerBased => {
                // Path 1: End sequence detected via naive string search.
                let end_marker_info = if !self.jail_end_sequences.is_empty() {
                    self.jail_end_sequences.iter().find_map(|seq| {
                        accumulated_content
                            .find(seq)
                            .map(|pos| (pos + seq.len(), seq.clone()))
                    })
                } else {
                    None
                };

                // Path 2: Complete tool call(s) can be parsed (early exit)
                let early_exit = self.should_exit_jail_early(accumulated_content).await;

                // When a tool_call_parser is active, prefer Path 2 over Path 1 so
                // that `find_tool_call_end_position` advances past all consecutive
                // parallel tool calls instead of splitting at the first end tag.
                // Fall back to Path 1 when parsing fails (e.g. malformed content).
                if early_exit {
                    // For early exit, find where the complete tool call ends.
                    // `find_tool_call_end_position` returns `None` when the
                    // section wrapper isn't closed (e.g. kimi_k2 without
                    // section_end). In that case, don't early-exit — more
                    // parallel calls may follow. The calls will be recovered
                    // by `finalize()` at stream end.
                    if let Some(parser) = &self.tool_call_parser {
                        let tools_slice = self.tool_definitions.as_deref();
                        if let Ok((_, _)) = try_tool_call_parse_aggregate(
                            accumulated_content,
                            Some(parser),
                            tools_slice,
                        )
                        .await
                        {
                            if let Some(split_pos) =
                                find_tool_call_end_position(accumulated_content, Some(parser))
                            {
                                (true, split_pos)
                            } else {
                                (false, accumulated_content.len())
                            }
                        } else {
                            (false, accumulated_content.len())
                        }
                    } else {
                        (false, accumulated_content.len())
                    }
                } else if let Some((end_pos, _)) = end_marker_info {
                    (true, end_pos)
                } else {
                    (false, accumulated_content.len())
                }
            }
            JailMode::Immediate { format } => {
                // For tool_choice, check if we have valid complete JSON
                let json_fragment = immediate_tool_choice_json_fragment(accumulated_content);
                let json_fragment = escape_json_string_control_chars(json_fragment);
                match format {
                    ToolChoiceFormat::SingleObject { .. } => {
                        // Expect single object: {"location": "Paris", "unit": "celsius"}
                        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&json_fragment)
                            && value.is_object()
                        {
                            return (true, accumulated_content.len());
                        }
                    }
                    ToolChoiceFormat::ArrayOfTools { .. } => {
                        // Expect array: [{"name":"search","parameters":{...}}, ...]
                        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&json_fragment)
                            && let Some(arr) = value.as_array()
                            && !arr.is_empty()
                        {
                            return (true, accumulated_content.len());
                        }
                    }
                }

                // Some model-native structural-tag paths still emit XML/tool
                // markers for forced/required tool_choice. Immediate jail must
                // be able to end on those complete native tool calls too;
                // otherwise it waits until EOF/max_tokens even though the
                // configured parser can already recover the call.
                if let Some(parser) = &self.tool_call_parser
                    && let Ok((tool_calls, _)) = try_tool_call_parse_aggregate(
                        accumulated_content,
                        Some(parser),
                        self.tool_definitions.as_deref(),
                    )
                    .await
                    && !tool_calls.is_empty()
                    && let Some(split_pos) =
                        find_tool_call_end_position(accumulated_content, Some(parser))
                {
                    return (true, split_pos);
                }

                (false, accumulated_content.len())
            }
        }
    }

    /// Parse tool calls from accumulated content and create choice.
    ///
    /// `is_finalize` selects the recovery-enabled aggregator (missing
    /// outer end-token / truncated JSON). Streaming early-exit callers pass
    /// `false`; the stream-end finalize path passes `true`.
    async fn create_tool_call_choice(
        &self,
        choice_index: u32,
        accumulated_content: &str,
        base_choice: &ChatChoiceStream,
        tool_call_offset: usize,
        is_finalize: bool,
    ) -> ChatChoiceStream {
        match &self.jail_mode {
            JailMode::MarkerBased => {
                // Traditional marker-based tool call parsing
                let tools_slice = self.tool_definitions.as_deref();
                let parse_result = if is_finalize {
                    try_tool_call_parse_aggregate_finalize(
                        accumulated_content,
                        self.tool_call_parser.as_deref(),
                        tools_slice,
                    )
                    .await
                } else {
                    try_tool_call_parse_aggregate(
                        accumulated_content,
                        self.tool_call_parser.as_deref(),
                        tools_slice,
                    )
                    .await
                };
                match parse_result {
                    Ok((tool_calls, normal_text)) if !tool_calls.is_empty() => {
                        // If a named tool filter is set (tool_choice=named + parser path), reject
                        // tool calls that don't match the required tool name.
                        let tool_calls = if let Some(ref required_name) = self.named_tool_name {
                            let filtered: Vec<_> = tool_calls
                                .into_iter()
                                .filter(|tc| tc.function.name == *required_name)
                                .collect();
                            if filtered.is_empty() {
                                tracing::warn!(
                                    required = %required_name,
                                    "tool_choice=named: parser emitted no matching tool calls; dropping jail output"
                                );
                            }
                            filtered
                        } else {
                            tool_calls
                        };

                        if tool_calls.is_empty() {
                            // All parsed calls were filtered out — emit the parser's stripped
                            // normal_text, not accumulated_content (which still contains the
                            // raw tool-call markers).
                            return create_choice_stream(
                                choice_index,
                                Some(Role::Assistant),
                                normal_text.as_deref().unwrap_or(""),
                                None,
                                base_choice.finish_reason,
                                base_choice.logprobs.clone(),
                            );
                        }

                        // Convert to streaming format
                        let tool_call_chunks: Vec<ChatCompletionMessageToolCallChunk> = tool_calls
                            .into_iter()
                            .enumerate()
                            .map(|(idx, tool_call)| ChatCompletionMessageToolCallChunk {
                                index: (tool_call_offset + idx) as u32,
                                id: Some(tool_call.id),
                                r#type: Some(FunctionType::Function),
                                function: Some(FunctionCallStream {
                                    name: Some(tool_call.function.name),
                                    arguments: Some(tool_call.function.arguments),
                                }),
                            })
                            .collect();
                        create_choice_stream(
                            choice_index,
                            Some(Role::Assistant),
                            normal_text.as_deref().unwrap_or(""),
                            Some(tool_call_chunks),
                            None,
                            base_choice.logprobs.clone(),
                        )
                    }
                    Ok((_, normal_text)) => {
                        if let Ok(tool_call_chunks) = self.parse_jsonish_tool_call_chunks(
                            accumulated_content,
                            tool_call_offset,
                            is_finalize,
                        ) && !tool_call_chunks.is_empty()
                        {
                            return create_choice_stream(
                                choice_index,
                                Some(Role::Assistant),
                                "",
                                Some(tool_call_chunks),
                                None,
                                base_choice.logprobs.clone(),
                            );
                        }

                        // Parser succeeded but extracted no structured tool calls. The parser
                        // signals which sub-case via normal_text:
                        //   - Some(""):  parser detected markers but couldn't form a complete
                        //                call (e.g. kimi truncated mid-arg, or start token with
                        //                no valid JSON). Drop the buffer — accumulated_content
                        //                still has the raw markers and would leak.
                        //   - otherwise: parser saw no markers (false positive entry, e.g.
                        //                mistral on a stray `{` in prose, or default `<tool_call>`
                        //                token when manual sequences are configured). Pass
                        //                accumulated_content through verbatim — it's regular text
                        //                and may carry leading/trailing whitespace the parser
                        //                would have trimmed.
                        let content = if normal_text.as_deref() == Some("") {
                            ""
                        } else {
                            accumulated_content
                        };
                        create_choice_stream(
                            choice_index,
                            Some(Role::Assistant),
                            content,
                            None,
                            base_choice.finish_reason,
                            base_choice.logprobs.clone(),
                        )
                    }
                    Err(e) => {
                        if let Ok(tool_call_chunks) = self.parse_jsonish_tool_call_chunks(
                            accumulated_content,
                            tool_call_offset,
                            is_finalize,
                        ) && !tool_call_chunks.is_empty()
                        {
                            return create_choice_stream(
                                choice_index,
                                Some(Role::Assistant),
                                "",
                                Some(tool_call_chunks),
                                None,
                                base_choice.logprobs.clone(),
                            );
                        }

                        // Parser errored — emit empty content rather than the raw buffer.
                        // accumulated_content may still contain tool-call markers, and
                        // surfacing those to the user is the leak we're guarding against.
                        // The warn! gives operators visibility into the failure.
                        tracing::warn!(
                            error = %e,
                            "tool-call parser errored; dropping buffered content to avoid marker leak"
                        );
                        create_choice_stream(
                            choice_index,
                            Some(Role::Assistant),
                            "",
                            None,
                            base_choice.finish_reason,
                            base_choice.logprobs.clone(),
                        )
                    }
                }
            }
            JailMode::Immediate { format } => {
                // tool_choice=required/named path (SGLang/vLLM-style).
                //
                // Primary parser is try_tool_call_parse_basic_json (the
                // base_json_parser) since guided decoding constrains output
                // to a bare JSON shape. Fallbacks cover two edge cases:
                //
                //   * Named tool_choice when the schema produces just the
                //     parameters object (no {name, parameters} wrapper) —
                //     handled by parse_tool_choice_json, which knows the
                //     target tool_name from ToolChoiceFormat::SingleObject.
                //
                //   * Backends that do not honor guided decoding and emit
                //     the model's native format instead (e.g. qwen3_coder
                //     XML). In that case try_tool_call_parse_aggregate with
                //     the configured tool_call_parser recovers the call.
                let is_named_tool_choice = matches!(format, ToolChoiceFormat::SingleObject { .. });
                if is_named_tool_choice && tool_call_offset > 0 {
                    return create_choice_stream(
                        choice_index,
                        Some(Role::Assistant),
                        "",
                        None,
                        base_choice.finish_reason,
                        base_choice.logprobs.clone(),
                    );
                }

                let mut tool_call_chunks: Vec<ChatCompletionMessageToolCallChunk> = Vec::new();
                let json_fragment = immediate_tool_choice_json_fragment(accumulated_content);
                let json_fragment = escape_json_string_control_chars(json_fragment);

                // 1. Primary: bare-JSON extraction — handles
                //    `[{name,parameters}, ...]`, `{name,parameters}`,
                //    `{name,arguments}`, and arrays of either.
                let basic_json_cfg = JsonParserConfig {
                    bare_json_mode: true,
                    ..Default::default()
                };
                // Per-path indices are placeholders — final indices are assigned
                // below after the named filter so dropped entries don't leave
                // gaps and multi-emission streams don't collide.
                if let Ok((parsed, _)) = try_tool_call_parse_basic_json(
                    &json_fragment,
                    &basic_json_cfg,
                    self.tool_definitions.as_deref(),
                ) && !parsed.is_empty()
                {
                    tool_call_chunks.extend(parsed.into_iter().map(|tc| {
                        ChatCompletionMessageToolCallChunk {
                            index: 0,
                            id: Some(tc.id),
                            r#type: Some(FunctionType::Function),
                            function: Some(FunctionCallStream {
                                name: Some(tc.function.name),
                                arguments: Some(tc.function.arguments),
                            }),
                        }
                    }));
                }

                // 2. Named-only fallback: output is just the parameters object
                //    (tool_name is supplied by SingleObject format).
                if tool_call_chunks.is_empty()
                    && let Ok(chunks) = self.parse_tool_choice_json(&json_fragment, format)
                {
                    tool_call_chunks = chunks;
                }

                // 3. Marker-based fallback for backends that did not enforce
                //    guided decoding and emitted the model's native format.
                if tool_call_chunks.is_empty()
                    && self.tool_call_parser.is_some()
                    && let Ok((tool_calls, _)) = try_tool_call_parse_aggregate(
                        accumulated_content,
                        self.tool_call_parser.as_deref(),
                        self.tool_definitions.as_deref(),
                    )
                    .await
                {
                    tool_call_chunks.extend(tool_calls.into_iter().map(|tc| {
                        ChatCompletionMessageToolCallChunk {
                            index: 0,
                            id: Some(tc.id),
                            r#type: Some(FunctionType::Function),
                            function: Some(FunctionCallStream {
                                name: Some(tc.function.name),
                                arguments: Some(tc.function.arguments),
                            }),
                        }
                    }));
                }

                // Named filter: drop any parsed calls whose name doesn't match.
                // Track whether the filter drained a non-empty list so we can
                // suppress the content fallback below — otherwise the raw
                // wrong-tool JSON would leak to the client as assistant text.
                let mut filter_dropped_all = false;
                if let Some(ref required_name) = self.named_tool_name {
                    let pre_filter_len = tool_call_chunks.len();
                    tool_call_chunks.retain(|tc| {
                        tc.function.as_ref().and_then(|f| f.name.as_deref())
                            == Some(required_name.as_str())
                    });
                    if pre_filter_len > 0 && tool_call_chunks.is_empty() {
                        filter_dropped_all = true;
                        tracing::warn!(
                            required = %required_name,
                            "tool_choice=named: parsers emitted no matching tool calls; dropping jail output"
                        );
                    }
                }

                if is_named_tool_choice && tool_call_chunks.len() > 1 {
                    tracing::warn!(
                        count = tool_call_chunks.len(),
                        "tool_choice=named: parsers emitted multiple calls; keeping first"
                    );
                    tool_call_chunks.truncate(1);
                }

                // Assign final indices: renumber survivors 0..n (no gaps from
                // the filter) then add the cumulative offset for consistency
                // with the MarkerBased branch across multi-emission streams.
                for (new_idx, chunk) in tool_call_chunks.iter_mut().enumerate() {
                    chunk.index = (tool_call_offset + new_idx) as u32;
                }

                if !tool_call_chunks.is_empty() {
                    create_choice_stream(
                        choice_index,
                        Some(Role::Assistant),
                        "",
                        Some(tool_call_chunks),
                        base_choice.finish_reason,
                        base_choice.logprobs.clone(),
                    )
                } else if filter_dropped_all {
                    // Named filter rejected every parsed call — do not leak
                    // the wrong-tool JSON back as content.
                    create_choice_stream(
                        choice_index,
                        Some(Role::Assistant),
                        "",
                        None,
                        base_choice.finish_reason,
                        base_choice.logprobs.clone(),
                    )
                } else if matches!(format, ToolChoiceFormat::ArrayOfTools { .. })
                    && tool_call_offset > 0
                {
                    // Required tool choice may keep receiving native-format
                    // explanatory/final-answer spill after valid tool calls.
                    // Once a required-mode call was emitted, any unparsable
                    // jailed remainder is not assistant content.
                    create_choice_stream(
                        choice_index,
                        Some(Role::Assistant),
                        "",
                        None,
                        base_choice.finish_reason,
                        base_choice.logprobs.clone(),
                    )
                } else {
                    // All parsing paths failed — return accumulated content as text.
                    create_choice_stream(
                        choice_index,
                        Some(Role::Assistant),
                        accumulated_content,
                        None,
                        base_choice.finish_reason,
                        base_choice.logprobs.clone(),
                    )
                }
            }
        }
    }

    /// Helper to create a ChatCompletionMessageToolCallChunk
    fn create_tool_call_chunk(
        index: u32,
        name: String,
        arguments: String,
    ) -> ChatCompletionMessageToolCallChunk {
        ChatCompletionMessageToolCallChunk {
            index,
            id: Some(format!("call-{}", Uuid::new_v4())),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some(name),
                arguments: Some(arguments),
            }),
        }
    }

    fn parse_jsonish_tool_call_chunks(
        &self,
        accumulated_content: &str,
        tool_call_offset: usize,
        allow_eof_recovery: bool,
    ) -> anyhow::Result<Vec<ChatCompletionMessageToolCallChunk>> {
        let mut config = JsonParserConfig::default();
        config.tool_call_start_tokens = vec!["<tools>".to_string()];
        config.tool_call_end_tokens = vec!["</tools>".to_string()];
        config.allow_eof_recovery = allow_eof_recovery;

        let (parsed, _) = try_tool_call_parse_basic_json(
            accumulated_content,
            &config,
            self.tool_definitions.as_deref(),
        )?;

        let known_tools = self.tool_definitions.as_ref();
        let chunks = parsed
            .into_iter()
            .filter(|tc| {
                known_tools
                    .is_none_or(|tools| tools.iter().any(|tool| tool.name == tc.function.name))
            })
            .enumerate()
            .map(|(idx, tc)| ChatCompletionMessageToolCallChunk {
                index: (tool_call_offset + idx) as u32,
                id: Some(tc.id),
                r#type: Some(FunctionType::Function),
                function: Some(FunctionCallStream {
                    name: Some(tc.function.name),
                    arguments: Some(tc.function.arguments),
                }),
            })
            .collect();
        Ok(chunks)
    }

    /// Parse tool_choice JSON output into tool call chunks
    fn parse_tool_choice_json(
        &self,
        json_content: &str,
        format: &ToolChoiceFormat,
    ) -> anyhow::Result<Vec<ChatCompletionMessageToolCallChunk>> {
        let parsed = serde_json::from_str::<serde_json::Value>(json_content)?;

        match format {
            ToolChoiceFormat::SingleObject { tool_name } => {
                // For named tool choice: JSON is the parameters object
                if parsed.is_object() {
                    Ok(vec![Self::create_tool_call_chunk(
                        0,
                        tool_name.clone(),
                        json_content.to_string(),
                    )])
                } else {
                    Ok(vec![])
                }
            }
            ToolChoiceFormat::ArrayOfTools { .. } => {
                // For required tool choice: JSON is array of {name, parameters}
                if let Some(array) = parsed.as_array() {
                    let chunks: Vec<ChatCompletionMessageToolCallChunk> = array
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, entry)| {
                            let name = entry.get("name")?.as_str()?.to_string();
                            let parameters = entry.get("parameters")?;
                            let args = serde_json::to_string(parameters).ok()?;
                            Some(Self::create_tool_call_chunk(idx as u32, name, args))
                        })
                        .collect();
                    Ok(chunks)
                } else {
                    Ok(vec![])
                }
            }
        }
    }

    /// Check if accumulated content contains complete tool calls that can be parsed
    /// Returns true if we should exit the jail early
    async fn should_exit_jail_early(&self, accumulated: &str) -> bool {
        if let Some(ref parser) = self.tool_call_parser {
            // Try to parse - if successful and we have complete tool calls, exit early
            let tools_slice = self.tool_definitions.as_deref();
            match try_tool_call_parse_aggregate(accumulated, Some(parser), tools_slice).await {
                Ok((tool_calls, _normal_text)) => {
                    let result = !tool_calls.is_empty();
                    return result;
                }
                Err(_e) => {}
            }
        }
        false
    }

    /// Post-processor that sets finish_reason to ToolCalls when tool calls were emitted
    /// This should be called after apply() to fix the finish_reason for tool call chunks
    fn fix_finish_reason<S>(
        input_stream: S,
        jail_mode: JailMode,
        named_tool_active: bool,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        stream! {
            tokio::pin!(input_stream);
            let mut has_tool_calls_per_choice: HashMap<u32, bool> = HashMap::new();
            let mut finish_seen: HashMap<u32, bool> = HashMap::new();
            let mut last_inner: Option<dynamo_protocols::types::CreateChatCompletionStreamResponse> = None;
            let mut last_ann: (Option<String>, Option<String>, Option<Vec<String>>) = (None, None, None);

            while let Some(mut response) = input_stream.next().await {
                // Track if any choice emitted tool calls
                if let Some(ref data) = response.data {
                    for choice in &data.inner.choices {
                        if choice.delta.tool_calls.is_some() {
                            has_tool_calls_per_choice.insert(choice.index, true);
                        }
                    }
                }

                // Fix finish_reason based on jail mode and whether tool calls were emitted
                if let Some(ref mut data) = response.data {
                    for choice in &mut data.inner.choices {
                        if let Some(finish) = choice.finish_reason {
                            // Only modify Stop finish reason, preserve Length/ContentFilter
                            if finish == FinishReason::Stop {
                                let has_tool_calls = has_tool_calls_per_choice.get(&choice.index).copied().unwrap_or(false);

                                // OpenAI spec: whenever tool_calls were emitted on this
                                // choice, finish_reason MUST be "tool_calls" — regardless of
                                // whether tool_choice was "auto", "required", or a named
                                // function.
                                let _ = named_tool_active;
                                match &jail_mode {
                                    JailMode::MarkerBased => {
                                        if has_tool_calls {
                                            choice.finish_reason = Some(FinishReason::ToolCalls);
                                        }
                                    }
                                    JailMode::Immediate { format: _ } => {
                                        if has_tool_calls {
                                            choice.finish_reason = Some(FinishReason::ToolCalls);
                                        }
                                    }
                                }
                            }
                            // Length and ContentFilter are preserved as-is
                        }
                        if choice.finish_reason.is_some() {
                            finish_seen.insert(choice.index, true);
                        }
                    }
                    last_inner = Some(data.inner.clone());
                    last_ann = (response.id.clone(), response.event.clone(), response.comment.clone());
                }

                yield response;
            }

            // Safety net: a choice that emitted tool calls but never a finish_reason
            // (some MTP chunk patterns drop the terminal chunk) must still terminate
            // with finish_reason=tool_calls. Fires only when no finish was seen, so it
            // cannot duplicate an existing terminal chunk.
            if let Some(mut inner) = last_inner {
                let mut missing: Vec<u32> = Vec::new();
                for (idx, has) in has_tool_calls_per_choice.iter() {
                    if *has && !finish_seen.get(idx).copied().unwrap_or(false) {
                        missing.push(*idx);
                    }
                }
                if !missing.is_empty() {
                    inner.choices = missing
                        .into_iter()
                        .map(|index| ChatChoiceStream {
                            index,
                            delta: ChatCompletionStreamResponseDelta {
                                role: Some(Role::Assistant),
                                content: None,
                                tool_calls: None,
                                function_call: None,
                                refusal: None,
                                reasoning_content: None,
                            },
                            finish_reason: Some(FinishReason::ToolCalls),
                            logprobs: None,
                        })
                        .collect();
                    inner.usage = None;
                    yield Annotated {
                        data: Some(NvCreateChatCompletionStreamResponse { inner, nvext: None }),
                        id: last_ann.0,
                        event: last_ann.1,
                        comment: last_ann.2,
                        error: None,
                    };
                }
            }
        }
    }
}

/// Builder for configuring a JailedStream
pub struct JailedStreamBuilder {
    jail_start_sequences: Vec<String>,
    jail_end_sequences: Vec<String>,
    tool_call_parser: Option<String>,
    /// When set, only tool calls with this name are emitted (enforces tool_choice=named
    /// when a tool_call_parser is active and the parser-aware MarkerBased path is used).
    named_tool_name: Option<String>,
    tool_definitions: Option<Vec<dynamo_parsers::tool_calling::ToolDefinition>>,
    emission_mode: EmissionMode,
    jail_mode: JailMode,
    defer_terminal_until_usage: bool,
}

impl JailedStreamBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            jail_start_sequences: Vec::new(),
            jail_end_sequences: Vec::new(),
            tool_call_parser: None,
            named_tool_name: None,
            tool_definitions: None,
            emission_mode: EmissionMode::default(),
            jail_mode: JailMode::MarkerBased,
            defer_terminal_until_usage: false,
        }
    }

    /// Add a sequence that triggers jailing when detected
    pub fn jail_start_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.jail_start_sequences.push(sequence.into());
        self
    }

    /// Add multiple sequences that trigger jailing when detected
    pub fn jail_start_sequences(
        mut self,
        sequences: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.jail_start_sequences
            .extend(sequences.into_iter().map(Into::into));
        self
    }

    /// Add a sequence that ends jailing when detected
    pub fn jail_end_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.jail_end_sequences.push(sequence.into());
        self
    }

    /// Add multiple sequences that end jailing when detected
    pub fn jail_end_sequences(
        mut self,
        sequences: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.jail_end_sequences
            .extend(sequences.into_iter().map(Into::into));
        self
    }

    /// Set the tool call parser to use for detection and parsing
    pub fn tool_call_parser(mut self, parser: impl Into<String>) -> Self {
        self.tool_call_parser = Some(parser.into());
        self
    }

    /// Constrain parsed output to a single named tool (for tool_choice=named + parser path).
    /// When set, tool calls emitted by the parser that don't match `tool_name` are silently
    /// filtered out, enforcing the named-tool contract even when the model emits the wrong tool.
    pub fn named_tool_filter(mut self, tool_name: impl Into<String>) -> Self {
        self.named_tool_name = Some(tool_name.into());
        self
    }

    /// Set the tool definitions for runtime validation and parsing
    pub fn tool_definitions(
        mut self,
        tools: Vec<dynamo_parsers::tool_calling::ToolDefinition>,
    ) -> Self {
        self.tool_definitions = Some(tools);
        self
    }

    /// Set the emission mode for handling multiple choices
    pub fn emission_mode(mut self, mode: EmissionMode) -> Self {
        self.emission_mode = mode;
        self
    }

    /// Enable single choice per chunk emission for OpenAI compatibility
    pub fn single_choice_per_chunk(mut self) -> Self {
        self.emission_mode = EmissionMode::SingleChoicePerChunk;
        self
    }

    /// Enable packed emission mode (multiple choices per chunk)
    pub fn packed_emission(mut self) -> Self {
        self.emission_mode = EmissionMode::Packed;
        self
    }

    /// Enable immediate jail mode for tool_choice=named
    pub fn tool_choice_named(mut self, tool_name: String) -> Self {
        self.jail_mode = JailMode::Immediate {
            format: ToolChoiceFormat::SingleObject { tool_name },
        };
        self
    }

    /// Enable immediate jail mode for tool_choice=required
    pub fn tool_choice_required(mut self) -> Self {
        self.jail_mode = JailMode::Immediate {
            format: ToolChoiceFormat::ArrayOfTools {
                terminal_after_first: false,
            },
        };
        self
    }

    pub fn defer_terminal_until_usage(mut self, enabled: bool) -> Self {
        self.defer_terminal_until_usage = enabled;
        self
    }

    /// Build the configured JailedStream
    pub fn build(mut self) -> JailedStream {
        if let JailMode::Immediate {
            format:
                ToolChoiceFormat::ArrayOfTools {
                    terminal_after_first,
                },
        } = &mut self.jail_mode
            && self
                .tool_definitions
                .as_ref()
                .is_some_and(|tools| tools.len() == 1)
        {
            *terminal_after_first = true;
        }

        // Auto-populate jail sequences from parser config if not manually configured
        if let Some(ref parser_name) = self.tool_call_parser {
            let parser_map = get_tool_parser_map();
            if let Some(config) = parser_map.get(parser_name.as_str()) {
                // Auto-populate start sequences if none configured
                if self.jail_start_sequences.is_empty() {
                    self.jail_start_sequences = config.parser_config.tool_call_start_tokens();
                }

                // Auto-populate end sequences if none configured
                if self.jail_end_sequences.is_empty() {
                    self.jail_end_sequences = config
                        .parser_config
                        .tool_call_end_tokens()
                        .iter()
                        .filter(|&s| !s.is_empty())
                        .cloned()
                        .collect();
                }
            }
        }

        // Collect all possible marker patterns for the MarkerMatcher
        let mut all_patterns = Vec::new();

        // Add configured start sequences (now auto-populated if needed)
        all_patterns.extend(self.jail_start_sequences.clone());

        // Add patterns from tool call parser if configured (for redundancy)
        if let Some(ref parser_name) = self.tool_call_parser {
            let parser_map = get_tool_parser_map();
            if let Some(config) = parser_map.get(parser_name.as_str()) {
                // Add start tokens from the parser config
                all_patterns.extend(config.parser_config.tool_call_start_tokens());
            }
        }

        // Add common tool call markers to ensure we detect all formats
        // Only include these when a specific parser is NOT configured,
        // to avoid unexpected false positives for explicit formats
        if self.tool_call_parser.is_none() {
            let common_markers = vec![
                "<TOOLCALL>".to_string(),     // nemotron_deci format
                "<tool_call>".to_string(),    // hermes format
                "[TOOL_CALLS]".to_string(),   // mistral format
                "<|python_tag|>".to_string(), // llama3_json format
                "functools[".to_string(),     // phi4 format
                // Add JSON start patterns for Mistral-style tool calls
                "[{".to_string(),
                "{".to_string(),
                // Note: Harmony parser uses JSON patterns, covered by "{" above
            ];
            for marker in common_markers {
                if !all_patterns.contains(&marker) {
                    all_patterns.push(marker);
                }
            }
        }

        // Create the marker matcher (fallback to empty patterns if none configured)
        let marker_matcher = if all_patterns.is_empty() {
            // If no patterns, create a dummy matcher that never matches
            MarkerMatcher::new(vec!["__NEVER_MATCH__".to_string()])
                .expect("Failed to create dummy MarkerMatcher")
        } else {
            tracing::debug!("Creating MarkerMatcher with patterns: {:?}", all_patterns);
            MarkerMatcher::new(all_patterns)
                .expect("Failed to create MarkerMatcher with configured patterns")
        };

        JailedStream {
            jail_start_sequences: self.jail_start_sequences,
            jail_end_sequences: self.jail_end_sequences,
            tool_call_parser: self.tool_call_parser,
            named_tool_name: self.named_tool_name,
            tool_definitions: self.tool_definitions,
            emission_mode: self.emission_mode,
            marker_matcher,
            jail_mode: self.jail_mode,
            defer_terminal_until_usage: self.defer_terminal_until_usage,
        }
    }
}

impl Default for JailedStreamBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::CreateChatCompletionStreamResponse;
    use futures::stream;

    /// Helper: build a single-choice stream chunk with text content
    #[allow(deprecated)]
    fn text_chunk(text: &str) -> Annotated<NvCreateChatCompletionStreamResponse> {
        text_chunk_with_tool_calls(text, None)
    }

    #[allow(deprecated)]
    fn text_chunk_with_tool_calls(
        text: &str,
        tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(
                    text.to_string(),
                )),
                tool_calls,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: None,
        };

        Annotated {
            data: Some(NvCreateChatCompletionStreamResponse {
                inner: CreateChatCompletionStreamResponse {
                    id: "id-42".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 0,
                    model: "test-model".to_string(),
                    choices: vec![choice],
                    usage: None,
                    service_tier: None,
                    system_fingerprint: None,
                },
                nvext: None,
            }),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    #[allow(deprecated)]
    fn finish_chunk(
        finish_reason: FinishReason,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: None,
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: Some(finish_reason),
            logprobs: None,
        };

        Annotated {
            data: Some(NvCreateChatCompletionStreamResponse {
                inner: CreateChatCompletionStreamResponse {
                    id: "id-42".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 0,
                    model: "test-model".to_string(),
                    choices: vec![choice],
                    usage: None,
                    service_tier: None,
                    system_fingerprint: None,
                },
                nvext: None,
            }),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    fn usage_chunk(
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        Annotated {
            data: Some(NvCreateChatCompletionStreamResponse {
                inner: CreateChatCompletionStreamResponse {
                    id: "id-42".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 0,
                    model: "test-model".to_string(),
                    choices: vec![],
                    usage: Some(dynamo_protocols::types::CompletionUsage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                        prompt_tokens_details: None,
                        completion_tokens_details: None,
                    }),
                    service_tier: None,
                    system_fingerprint: None,
                },
                nvext: None,
            }),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    /// Collect all emitted tool calls from the jailed stream output
    fn collect_tool_calls(
        responses: &[Annotated<NvCreateChatCompletionStreamResponse>],
    ) -> Vec<(String, String)> {
        let mut tool_calls = Vec::new();
        for resp in responses {
            if let Some(ref data) = resp.data {
                for choice in &data.inner.choices {
                    if let Some(ref tcs) = choice.delta.tool_calls {
                        for tc in tcs {
                            if let Some(ref func) = tc.function {
                                let name = func.name.clone().unwrap_or_default();
                                let args = func.arguments.clone().unwrap_or_default();
                                tool_calls.push((name, args));
                            }
                        }
                    }
                }
            }
        }
        tool_calls
    }

    /// Collect all emitted text content from the jailed stream output
    fn collect_text_content(
        responses: &[Annotated<NvCreateChatCompletionStreamResponse>],
    ) -> String {
        responses
            .iter()
            .flat_map(|r| r.data.iter())
            .flat_map(|d| d.inner.choices.iter())
            .filter_map(|c| {
                if let Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(t)) =
                    &c.delta.content
                {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    #[tokio::test]
    async fn test_immediate_named_tool_choice_skips_reasoning_prefix() {
        let jail = JailedStream::builder()
            .tool_choice_named("get_weather".to_string())
            .build();

        let chunks = vec![text_chunk(
            "I should call the weather tool.</think>{\"location\":{\"city\":\"Paris\"}}",
        )];

        let input_stream = Box::pin(stream::iter(chunks));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert_eq!(tool_calls.len(), 1, "Expected named tool call");
        assert_eq!(tool_calls[0].0, "get_weather");
        assert_eq!(tool_calls[0].1, "{\"location\":{\"city\":\"Paris\"}}");
        assert_eq!(collect_text_content(&responses).trim(), "");
    }

    #[tokio::test]
    async fn test_immediate_named_tool_choice_preserves_usage_chunk() {
        let jail = JailedStream::builder()
            .tool_choice_named("get_weather".to_string())
            .defer_terminal_until_usage(true)
            .build();

        let chunks = vec![
            text_chunk("{\"location\":{\"city\":\"Paris\"}}"),
            usage_chunk(11, 7),
        ];

        let input_stream = Box::pin(stream::iter(chunks));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);
        let usage = responses
            .iter()
            .filter_map(|r| r.data.as_ref())
            .find_map(|d| d.inner.usage.as_ref())
            .expect("usage chunk should pass through after terminal tool call");

        assert_eq!(tool_calls.len(), 1, "Expected named tool call");
        assert_eq!(usage.completion_tokens, 7);
        assert!(
            responses
                .iter()
                .flat_map(|r| r.data.iter())
                .flat_map(|d| d.inner.choices.iter())
                .any(|choice| choice.finish_reason == Some(FinishReason::ToolCalls))
        );
    }

    #[tokio::test]
    async fn test_marker_jail_parses_content_with_empty_tool_calls_delta() {
        let jail = JailedStream::builder()
            .tool_call_parser("qwen3_coder")
            .tool_definitions(vec![dynamo_parsers::tool_calling::ToolDefinition {
                name: "name_a_color".to_string(),
                parameters: None,
            }])
            .build();

        let native_call = concat!(
            "<tool_call>\n",
            "<function=name_a_color>\n",
            "<parameter=color_hex>\n#ff00ee\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        );

        let input_stream = Box::pin(stream::iter(vec![text_chunk_with_tool_calls(
            native_call,
            Some(Vec::new()),
        )]));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].0, "name_a_color");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&tool_calls[0].1).unwrap()["color_hex"],
            "#ff00ee"
        );
        assert_eq!(collect_text_content(&responses).trim(), "");
    }

    #[tokio::test]
    async fn test_marker_jail_drops_whitespace_after_tool_call() {
        let jail = JailedStream::builder()
            .tool_call_parser("qwen3_coder")
            .tool_definitions(vec![dynamo_parsers::tool_calling::ToolDefinition {
                name: "name_a_color".to_string(),
                parameters: None,
            }])
            .build();

        let native_call = concat!(
            "<tool_call>\n",
            "<function=name_a_color>\n",
            "<parameter=color_hex>\n#ff00ee\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        );

        let input_stream = Box::pin(stream::iter(vec![
            text_chunk(native_call),
            text_chunk("\n"),
            finish_chunk(FinishReason::Stop),
        ]));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert_eq!(tool_calls.len(), 1, "responses: {responses:#?}");
        assert_eq!(tool_calls[0].0, "name_a_color");
        assert_eq!(collect_text_content(&responses), "");
    }

    #[tokio::test]
    async fn test_marker_jail_parses_split_qwen_xml_before_stop_chunk() {
        let jail = JailedStream::builder()
            .tool_call_parser("qwen3_coder")
            .tool_definitions(vec![dynamo_parsers::tool_calling::ToolDefinition {
                name: "name_a_color".to_string(),
                parameters: None,
            }])
            .build();

        let input_stream = Box::pin(stream::iter(vec![
            text_chunk("<tool_call>"),
            text_chunk("\n<function=name_a_color>"),
            text_chunk("\n<parameter=color_hex>"),
            text_chunk("\n#"),
            text_chunk("ff"),
            text_chunk("00"),
            text_chunk("ee"),
            text_chunk("\n</parameter>"),
            text_chunk("\n</function>"),
            text_chunk("\n</tool_call>"),
            finish_chunk(FinishReason::Stop),
        ]));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert_eq!(tool_calls.len(), 1, "responses: {responses:#?}");
        assert_eq!(tool_calls[0].0, "name_a_color");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&tool_calls[0].1).unwrap()["color_hex"],
            "#ff00ee"
        );
        assert_eq!(collect_text_content(&responses).trim(), "");
        assert!(
            responses
                .iter()
                .flat_map(|r| r.data.iter())
                .flat_map(|d| d.inner.choices.iter())
                .any(|choice| choice.finish_reason == Some(FinishReason::ToolCalls))
        );
    }

    #[tokio::test]
    async fn test_immediate_named_tool_choice_jails_after_role_only_chunk() {
        let jail = JailedStream::builder()
            .tool_choice_named("send_message".to_string())
            .build();

        let role_choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: None,
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: None,
        };
        let role_chunk = Annotated {
            data: Some(NvCreateChatCompletionStreamResponse {
                inner: CreateChatCompletionStreamResponse {
                    id: "id-42".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 0,
                    model: "test-model".to_string(),
                    choices: vec![role_choice],
                    usage: None,
                    service_tier: None,
                    system_fingerprint: None,
                },
                nvext: None,
            }),
            id: None,
            event: None,
            comment: None,
            error: None,
        };

        let chunks = vec![
            role_chunk,
            text_chunk(
                "I should send it.</think>{\"to\":\"ops@test.com\",\"body\":\"line1\\nline2\\tquoted\"}",
            ),
        ];

        let input_stream = Box::pin(stream::iter(chunks));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert_eq!(tool_calls.len(), 1, "Expected named tool call");
        assert_eq!(tool_calls[0].0, "send_message");
        assert!(tool_calls[0].1.contains("ops@test.com"));
        assert_eq!(collect_text_content(&responses).trim(), "");
    }

    #[tokio::test]
    async fn test_immediate_named_tool_choice_escapes_raw_control_chars() {
        let jail = JailedStream::builder()
            .tool_choice_named("send_message".to_string())
            .build();

        let chunks = vec![text_chunk(
            "{\"to\":\"ops@test.com\",\"body\":\"line1\tquoted\"}",
        )];

        let input_stream = Box::pin(stream::iter(chunks));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert_eq!(tool_calls.len(), 1, "Expected named tool call");
        assert_eq!(tool_calls[0].0, "send_message");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&tool_calls[0].1).unwrap()["body"],
            "line1\tquoted"
        );
        assert!(tool_calls[0].1.contains("\\t"));
        assert_eq!(collect_text_content(&responses).trim(), "");
    }

    #[tokio::test]
    async fn test_immediate_named_tool_choice_exits_on_native_xml_tool_call() {
        let jail = JailedStream::builder()
            .tool_choice_named("send_message".to_string())
            .tool_call_parser("qwen3_coder")
            .tool_definitions(vec![dynamo_parsers::tool_calling::ToolDefinition {
                name: "send_message".to_string(),
                parameters: None,
            }])
            .build();

        let native_call = concat!(
            "<tool_call>\n",
            "<function=send_message>\n",
            "<parameter=to>\nops@test.com\n</parameter>\n",
            "<parameter=body>\nline1\nline2\tquoted\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        );

        let (should_end, split_pos) = jail.should_end_jail(native_call).await;

        assert!(should_end, "native tool parser should end immediate jail");
        assert_eq!(split_pos, native_call.len());
    }

    #[tokio::test]
    async fn test_immediate_named_tool_choice_emits_only_first_native_xml_tool_call() {
        let jail = JailedStream::builder()
            .tool_choice_named("send_message".to_string())
            .named_tool_filter("send_message")
            .tool_call_parser("qwen3_coder")
            .tool_definitions(vec![dynamo_parsers::tool_calling::ToolDefinition {
                name: "send_message".to_string(),
                parameters: None,
            }])
            .build();

        let native_call = concat!(
            "<tool_call>\n",
            "<function=send_message>\n",
            "<parameter=to>\nops@test.com\n</parameter>\n",
            "<parameter=body>\nline1\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        );

        let input_stream = Box::pin(stream::iter(vec![
            text_chunk(native_call),
            text_chunk(native_call),
        ]));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert_eq!(tool_calls.len(), 1, "named tool_choice must emit one call");
        assert_eq!(tool_calls[0].0, "send_message");
        assert_eq!(collect_text_content(&responses).trim(), "");
    }

    #[tokio::test]
    async fn test_immediate_required_tool_choice_drops_native_xml_final_answer_spill() {
        let jail = JailedStream::builder()
            .tool_choice_required()
            .tool_call_parser("qwen3_coder")
            .tool_definitions(vec![
                dynamo_parsers::tool_calling::ToolDefinition {
                    name: "get_weather".to_string(),
                    parameters: None,
                },
                dynamo_parsers::tool_calling::ToolDefinition {
                    name: "calculate".to_string(),
                    parameters: None,
                },
            ])
            .build();

        let native_calls = concat!(
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=location>\nNew York, NY\n</parameter>\n",
            "<parameter=unit>\nfahrenheit\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
            "<tool_call>\n",
            "<function=calculate>\n",
            "<parameter=expression>\n(99 - 17) / 4\n</parameter>\n",
            "<parameter=precision>\n2\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        );
        let final_answer_spill = concat!(
            "\n<tool_call>\n",
            "<final_answer>\n",
            "The weather is being fetched and the result is 20.5.\n",
            "</final_answer>\n",
            "</tool_call>"
        );

        let input_stream = Box::pin(stream::iter(vec![
            text_chunk(native_calls),
            text_chunk(final_answer_spill),
        ]));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert_eq!(
            tool_calls.len(),
            2,
            "required tool_choice should emit both calls"
        );
        assert_eq!(tool_calls[0].0, "get_weather");
        assert_eq!(tool_calls[1].0, "calculate");
        assert_eq!(collect_text_content(&responses).trim(), "");
    }

    #[tokio::test]
    async fn test_immediate_required_single_tool_choice_stops_after_first_native_xml_call() {
        let jail = JailedStream::builder()
            .tool_choice_required()
            .tool_call_parser("qwen3_coder")
            .tool_definitions(vec![dynamo_parsers::tool_calling::ToolDefinition {
                name: "execute_sql".to_string(),
                parameters: None,
            }])
            .build();

        let native_call = concat!(
            "<tool_call>\n",
            "<function=execute_sql>\n",
            "<parameter=query>\nSELECT 42 AS answer\n</parameter>\n",
            "<parameter=dialect>\nsqlite\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        );

        let input_stream = Box::pin(stream::iter(vec![
            text_chunk(native_call),
            text_chunk(native_call),
        ]));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert_eq!(
            tool_calls.len(),
            1,
            "single-tool required choice should terminate after the first call"
        );
        assert_eq!(tool_calls[0].0, "execute_sql");
        assert_eq!(collect_text_content(&responses).trim(), "");
        assert!(
            responses
                .iter()
                .flat_map(|r| r.data.iter())
                .flat_map(|d| d.inner.choices.iter())
                .any(|choice| choice.finish_reason == Some(FinishReason::ToolCalls))
        );
    }

    #[tokio::test]
    async fn test_immediate_required_tool_choice_stops_repeated_native_xml_call() {
        let jail = JailedStream::builder()
            .tool_choice_required()
            .tool_call_parser("qwen3_coder")
            .tool_definitions(vec![
                dynamo_parsers::tool_calling::ToolDefinition {
                    name: "execute_sql".to_string(),
                    parameters: None,
                },
                dynamo_parsers::tool_calling::ToolDefinition {
                    name: "get_weather".to_string(),
                    parameters: None,
                },
            ])
            .build();

        let native_call = concat!(
            "<tool_call>\n",
            "<function=execute_sql>\n",
            "<parameter=query>\nSELECT COUNT(*) AS n FROM users;\n</parameter>\n",
            "<parameter=dialect>\nsqlite\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        );
        let native_call_reordered = concat!(
            "<tool_call>\n",
            "<function=execute_sql>\n",
            "<parameter=dialect>\nsqlite\n</parameter>\n",
            "<parameter=query>\nSELECT COUNT(*) AS n FROM users;\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        );

        let input_stream = Box::pin(stream::iter(vec![
            text_chunk(native_call),
            text_chunk(native_call_reordered),
            text_chunk(native_call),
            text_chunk(native_call_reordered),
        ]));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert_eq!(
            tool_calls.len(),
            1,
            "required tool_choice should stop on repeated identical calls"
        );
        assert_eq!(tool_calls[0].0, "execute_sql");
        assert!(
            responses
                .iter()
                .flat_map(|r| r.data.iter())
                .flat_map(|d| d.inner.choices.iter())
                .any(|choice| choice.finish_reason == Some(FinishReason::ToolCalls))
        );
    }

    #[tokio::test]
    async fn test_immediate_required_tool_choice_suppresses_duplicate_and_continues() {
        let jail = JailedStream::builder()
            .tool_choice_required()
            .tool_call_parser("qwen3_coder")
            .tool_definitions(vec![
                dynamo_parsers::tool_calling::ToolDefinition {
                    name: "execute_sql".to_string(),
                    parameters: None,
                },
                dynamo_parsers::tool_calling::ToolDefinition {
                    name: "get_weather".to_string(),
                    parameters: None,
                },
            ])
            .build();

        let native_call = concat!(
            "<tool_call>\n",
            "<function=execute_sql>\n",
            "<parameter=query>\nSELECT COUNT(*) AS n FROM users;\n</parameter>\n",
            "<parameter=dialect>\nsqlite\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        );
        let native_call_reordered = concat!(
            "<tool_call>\n",
            "<function=execute_sql>\n",
            "<parameter=dialect>\nsqlite\n</parameter>\n",
            "<parameter=query>\nSELECT COUNT(*) AS n FROM users;\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        );
        let second_call = concat!(
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=location>\nBoston\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        );

        let input_stream = Box::pin(stream::iter(vec![
            text_chunk(native_call),
            text_chunk(native_call_reordered),
            text_chunk(second_call),
        ]));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);
        let names: Vec<&str> = tool_calls.iter().map(|(name, _)| name.as_str()).collect();

        assert_eq!(
            names,
            vec!["execute_sql", "get_weather"],
            "required tool_choice should suppress one duplicate and continue"
        );
        assert_eq!(collect_text_content(&responses).trim(), "");
    }

    #[tokio::test]
    async fn test_immediate_required_tool_choice_drops_dangling_native_xml_function_spill() {
        let jail = JailedStream::builder()
            .tool_choice_required()
            .tool_call_parser("qwen3_coder")
            .tool_definitions(vec![
                dynamo_parsers::tool_calling::ToolDefinition {
                    name: "get_weather".to_string(),
                    parameters: None,
                },
                dynamo_parsers::tool_calling::ToolDefinition {
                    name: "calculate".to_string(),
                    parameters: None,
                },
            ])
            .build();

        let native_call = concat!(
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=location>\nNew York, NY\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        );
        let dangling_function_spill = concat!(
            "\n<function=calculate>\n",
            "<parameter=expression>\n(99 - 17) / 4\n</parameter>\n",
            "</function>\n",
            "</tool_call>"
        );

        let input_stream = Box::pin(stream::iter(vec![
            text_chunk(native_call),
            text_chunk(dangling_function_spill),
        ]));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert_eq!(
            tool_calls.len(),
            1,
            "malformed spill must not leak as content"
        );
        assert_eq!(tool_calls[0].0, "get_weather");
        assert_eq!(collect_text_content(&responses).trim(), "");
    }

    /// Helper: build a single-choice stream chunk with text content and logprobs
    #[allow(deprecated)]
    fn text_chunk_with_logprobs(text: &str) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let logprobs = ChatChoiceLogprobs {
            content: Some(
                text.chars()
                    .enumerate()
                    .map(
                        |(i, c)| dynamo_protocols::types::ChatCompletionTokenLogprob {
                            token: c.to_string(),
                            logprob: -(i as f32 + 1.0) * 0.1,
                            bytes: Some(c.to_string().into_bytes()),
                            top_logprobs: vec![],
                        },
                    )
                    .collect(),
            ),
            refusal: None,
        };

        let choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(
                    text.to_string(),
                )),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: Some(logprobs),
        };

        Annotated {
            data: Some(NvCreateChatCompletionStreamResponse {
                inner: CreateChatCompletionStreamResponse {
                    id: "id-42".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 0,
                    model: "test-model".to_string(),
                    choices: vec![choice],
                    usage: None,
                    service_tier: None,
                    system_fingerprint: None,
                },
                nvext: None,
            }),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    /// Collect all logprobs from jailed stream output choices
    fn collect_logprobs(
        responses: &[Annotated<NvCreateChatCompletionStreamResponse>],
    ) -> Vec<Option<ChatChoiceLogprobs>> {
        responses
            .iter()
            .flat_map(|r| r.data.iter())
            .flat_map(|d| d.inner.choices.iter())
            .map(|c| c.logprobs.clone())
            .collect()
    }

    #[tokio::test]
    async fn test_tool_call_preserves_logprobs_single_chunk() {
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let chunks = vec![text_chunk_with_logprobs(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"SF\"}}\n</tool_call>",
        )];

        let input_stream = Box::pin(stream::iter(chunks));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);
        assert_eq!(
            tool_calls.len(),
            1,
            "Expected 1 tool call, got {:?}",
            tool_calls
        );
        assert_eq!(tool_calls[0].0, "get_weather");

        // Logprobs must be preserved even though the entire output is a tool call
        let all_logprobs = collect_logprobs(&responses);
        let has_some_logprobs = all_logprobs.iter().any(|lp| lp.is_some());
        assert!(
            has_some_logprobs,
            "Logprobs should be preserved for tool call responses, got all None: {:?}",
            all_logprobs
        );
    }

    #[tokio::test]
    async fn test_tool_call_preserves_logprobs_multiple_chunks() {
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let chunks = vec![
            text_chunk_with_logprobs("<tool_call>\n{\"name\": \"get_weather\", \"arguments\""),
            text_chunk_with_logprobs(": {\"location\": \"SF\"}}\n</tool_call>"),
        ];

        let input_stream = Box::pin(stream::iter(chunks));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);
        assert!(!tool_calls.is_empty(), "Expected tool calls, got none");

        let all_logprobs = collect_logprobs(&responses);
        let has_some_logprobs = all_logprobs.iter().any(|lp| lp.is_some());
        assert!(
            has_some_logprobs,
            "Logprobs should be preserved for tool call responses across chunks, got all None",
        );
    }

    #[tokio::test]
    async fn test_tool_call_with_text_preserves_logprobs() {
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let chunks = vec![text_chunk_with_logprobs(
            "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"SF\"}}\n</tool_call>",
        )];

        let input_stream = Box::pin(stream::iter(chunks));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);
        assert_eq!(tool_calls.len(), 1);

        let all_logprobs = collect_logprobs(&responses);
        let has_some_logprobs = all_logprobs.iter().any(|lp| lp.is_some());
        assert!(
            has_some_logprobs,
            "Logprobs should be preserved for mixed text+tool_call responses",
        );

        // Verify the logprobs content is non-empty
        let logprob_entries: Vec<_> = all_logprobs
            .iter()
            .filter_map(|lp| lp.as_ref())
            .filter_map(|lp| lp.content.as_ref())
            .collect();
        assert!(
            logprob_entries.iter().any(|entries| !entries.is_empty()),
            "Logprobs content should have entries",
        );
    }

    #[tokio::test]
    async fn test_multi_tool_call_single_chunk() {
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let chunks = vec![text_chunk(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"SF\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"timezone\": \"PST\"}}\n</tool_call>",
        )];

        let input_stream = Box::pin(stream::iter(chunks));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert!(
            tool_calls.len() >= 2,
            "Expected at least 2 tool calls, got {}: {:?}",
            tool_calls.len(),
            tool_calls
        );

        let names: Vec<&str> = tool_calls.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains(&"get_weather"),
            "Missing get_weather tool call. Got: {:?}",
            names
        );
        assert!(
            names.contains(&"get_time"),
            "Missing get_time tool call. Got: {:?}",
            names
        );
    }

    #[tokio::test]
    async fn test_qwen_parser_recovers_tools_json_blocks() {
        let jail = JailedStream::builder()
            .jail_start_sequence("<tools>")
            .jail_end_sequence("</tools>")
            .tool_call_parser("qwen3_coder")
            .tool_definitions(vec![
                dynamo_parsers::tool_calling::ToolDefinition {
                    name: "get_weather".to_string(),
                    parameters: None,
                },
                dynamo_parsers::tool_calling::ToolDefinition {
                    name: "calculate".to_string(),
                    parameters: None,
                },
            ])
            .build();

        let chunks = vec![text_chunk(
            "\n<tools>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Boston, MA\",\"unit\":\"celsius\"}}\n</tools>\n<tools>\n{\"name\":\"calculate\",\"arguments\":{\"expression\":\"128*47\"}}\n</tools>",
        )];

        let input_stream = Box::pin(stream::iter(chunks));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);
        let names: Vec<&str> = tool_calls.iter().map(|(name, _)| name.as_str()).collect();

        assert!(
            names.contains(&"get_weather"),
            "Missing get_weather tool call. Got: {:?}",
            tool_calls
        );
        assert!(
            names.contains(&"calculate"),
            "Missing calculate tool call. Got: {:?}",
            tool_calls
        );
        assert_eq!(collect_text_content(&responses).trim(), "");
    }

    #[tokio::test]
    async fn test_multi_tool_call_multiple_chunks() {
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let chunks = vec![
            text_chunk("<tool_call>\n{\"name\": \"get_weather\", \"arguments\""),
            text_chunk(
                ": {\"location\": \"SF\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_time\"",
            ),
            text_chunk(", \"arguments\": {\"timezone\": \"PST\"}}\n</tool_call>"),
        ];

        let input_stream = Box::pin(stream::iter(chunks));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert!(
            tool_calls.len() >= 2,
            "Expected at least 2 tool calls, got {}: {:?}",
            tool_calls.len(),
            tool_calls
        );

        let names: Vec<&str> = tool_calls.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains(&"get_weather"),
            "Missing get_weather tool call. Got: {:?}",
            names
        );
        assert!(
            names.contains(&"get_time"),
            "Missing get_time tool call. Got: {:?}",
            names
        );
    }

    #[tokio::test]
    async fn test_trailing_text_not_re_jailed() {
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let chunks = vec![text_chunk(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"SF\"}}\n</tool_call>\nDone!",
        )];

        let input_stream = Box::pin(stream::iter(chunks));
        let output_stream = jail.apply_with_finish_reason(input_stream);

        let responses: Vec<_> = output_stream.collect().await;
        let tool_calls = collect_tool_calls(&responses);

        assert_eq!(
            tool_calls.len(),
            1,
            "Expected exactly 1 tool call, got {}: {:?}",
            tool_calls.len(),
            tool_calls
        );
        assert_eq!(tool_calls[0].0, "get_weather");

        let all_text = collect_text_content(&responses);
        assert!(
            all_text.contains("Done!"),
            "Trailing text 'Done!' should appear in output. Got text: {:?}",
            all_text
        );
    }
}
