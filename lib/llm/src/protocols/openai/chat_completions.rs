// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

use crate::engines::ValidateRequest;
use crate::preprocessor::media::MediaDecoder;

use super::{
    OpenAIOutputOptionsProvider, OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider,
    common_ext::{CommonExt, CommonExtProvider},
    nvext::NvExt,
    nvext::NvExtProvider,
    tools, validate,
};

pub mod aggregator;
mod delta;
pub mod jail;

pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

const DEFAULT_WILDCARD_PATTERN_MAX_LENGTH: u64 = 512;
const DEFAULT_CONSTRAINED_MAX_THINKING_TOKENS_ENV: &str =
    "DYN_DEFAULT_CONSTRAINED_MAX_THINKING_TOKENS";
const TOOL_CHOICE_GUIDED_JSON_MARKER: &str = "x-dynamo-tool-choice-schema";

fn wildcard_pattern_max_length() -> u64 {
    std::env::var("DYN_XGRAMMAR_DEFAULT_MAX_STRING_LENGTH")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_WILDCARD_PATTERN_MAX_LENGTH)
}

fn json_object_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "minProperties": 1,
        "maxProperties": 64,
        "additionalProperties": {
            "anyOf": [
                {"type": "string", "maxLength": wildcard_pattern_max_length()},
                {"type": "number"},
                {"type": "integer"},
                {"type": "boolean"},
                {"type": "null"},
                {"type": "array", "maxItems": 64},
                {"type": "object", "maxProperties": 64}
            ]
        }
    })
}

fn json_string_len(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::String(text) => text.len(),
        serde_json::Value::Array(values) => values.iter().map(json_string_len).sum(),
        serde_json::Value::Object(map) => map.values().map(json_string_len).sum(),
        _ => 0,
    }
}

fn pattern_starts_with_unbounded_wildcard(pattern: &str) -> bool {
    pattern.starts_with(".*")
        || pattern.starts_with(".+")
        || pattern.starts_with("[\\s\\S]*")
        || pattern.starts_with("[\\s\\S]+")
        || pattern.starts_with("[\\w\\W]*")
        || pattern.starts_with("[\\w\\W]+")
}

fn pattern_ends_with_unbounded_wildcard(pattern: &str) -> bool {
    pattern.ends_with(".*")
        || pattern.ends_with(".+")
        || pattern.ends_with("[\\s\\S]*")
        || pattern.ends_with("[\\s\\S]+")
        || pattern.ends_with("[\\w\\W]*")
        || pattern.ends_with("[\\w\\W]+")
}

fn pattern_has_unbounded_wildcard(pattern: &str) -> bool {
    pattern_starts_with_unbounded_wildcard(pattern)
        || pattern_ends_with_unbounded_wildcard(pattern)
        || pattern.contains("[\\s\\S]*")
        || pattern.contains("[\\s\\S]+")
        || pattern.contains("[\\w\\W]*")
        || pattern.contains("[\\w\\W]+")
}

fn strip_prefix_wildcard(pattern: &str) -> &str {
    for prefix in [
        ".*",
        ".+",
        "[\\s\\S]*",
        "[\\s\\S]+",
        "[\\w\\W]*",
        "[\\w\\W]+",
    ] {
        if let Some(stripped) = pattern.strip_prefix(prefix) {
            return stripped;
        }
    }
    pattern
}

fn strip_suffix_wildcard(pattern: &str) -> &str {
    for suffix in [
        ".*",
        ".+",
        "[\\s\\S]*",
        "[\\s\\S]+",
        "[\\w\\W]*",
        "[\\w\\W]+",
    ] {
        if let Some(stripped) = pattern.strip_suffix(suffix) {
            return stripped;
        }
    }
    pattern
}

fn bound_wildcard_pattern(pattern: &str, max_length: u64) -> Option<String> {
    let starts = pattern_starts_with_unbounded_wildcard(pattern);
    let ends = pattern_ends_with_unbounded_wildcard(pattern);
    if !starts && !ends {
        return None;
    }

    let mut inner = pattern;
    if starts {
        inner = strip_prefix_wildcard(inner);
    }
    if ends {
        inner = strip_suffix_wildcard(inner);
    }

    let mut bounded = String::new();
    if starts {
        bounded.push_str(&format!("^.{{0,{max_length}}}"));
    }
    bounded.push_str(inner);
    if ends {
        bounded.push_str(&format!(".{{0,{max_length}}}$"));
    }
    Some(bounded)
}

fn schema_type_includes_string(schema: &serde_json::Map<String, serde_json::Value>) -> bool {
    match schema.get("type") {
        Some(serde_json::Value::String(value)) => value == "string",
        Some(serde_json::Value::Array(values)) => values
            .iter()
            .any(|value| matches!(value, serde_json::Value::String(s) if s == "string")),
        _ => false,
    }
}

fn bound_wildcard_pattern_strings(schema: &mut serde_json::Value) {
    match schema {
        serde_json::Value::Object(object) => {
            let pattern = object
                .get("pattern")
                .and_then(|value| value.as_str())
                .map(str::to_string);
            let has_risky_pattern = pattern
                .as_deref()
                .is_some_and(pattern_has_unbounded_wildcard);
            let is_string_schema = schema_type_includes_string(object);
            let should_bound = is_string_schema
                && has_risky_pattern
                && !object.contains_key("enum")
                && !object.contains_key("const");

            if should_bound {
                let max_length = object
                    .get("maxLength")
                    .and_then(|value| value.as_u64())
                    .filter(|value| *value > 0)
                    .unwrap_or_else(wildcard_pattern_max_length);
                if let Some(pattern) = pattern
                    .as_deref()
                    .and_then(|pattern| bound_wildcard_pattern(pattern, max_length))
                {
                    object.insert("pattern".to_string(), serde_json::Value::String(pattern));
                }
                if !object.contains_key("maxLength") {
                    object.insert(
                        "maxLength".to_string(),
                        serde_json::Value::Number(max_length.into()),
                    );
                }
            }

            for value in object.values_mut() {
                bound_wildcard_pattern_strings(value);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                bound_wildcard_pattern_strings(value);
            }
        }
        _ => {}
    }
}

fn guided_json_schema(schema: serde_json::Value) -> serde_json::Value {
    let mut schema = schema;
    bound_wildcard_pattern_strings(&mut schema);
    schema
}

fn tool_choice_guided_json_schema(schema: serde_json::Value) -> serde_json::Value {
    let mut schema = guided_json_schema(schema);
    if let serde_json::Value::Object(object) = &mut schema {
        object.insert(
            TOOL_CHOICE_GUIDED_JSON_MARKER.to_string(),
            serde_json::Value::Bool(true),
        );
    }
    schema
}

fn default_constrained_max_thinking_tokens() -> Option<u32> {
    let value = std::env::var(DEFAULT_CONSTRAINED_MAX_THINKING_TOKENS_ENV).ok()?;
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let value = value.parse::<i64>().ok()?;
    if value < 0 {
        None
    } else {
        u32::try_from(value).ok()
    }
}

/// A request structure for creating a chat completion, extending OpenAI's
/// `CreateChatCompletionRequest` with [`NvExt`] extensions and common fields.
///
/// # Fields
/// - `inner`: The base OpenAI chat completion request, embedded using `serde(flatten)`.
/// - `common`: Common extension fields (ignore_eos, min_tokens) at root level, embedded using `serde(flatten)`.
/// - `nvext`: The optional NVIDIA extension field. See [`NvExt`] for more details.
///   Note: If ignore_eos is specified in both common and nvext, the common (root-level) value takes precedence.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateChatCompletionRequest {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::CreateChatCompletionRequest,

    #[serde(flatten, default)]
    pub common: CommonExt,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,

    /// Extra args to pass to the chat template rendering context
    /// Also accepts "chat_template_kwargs" as an alias for compatibility
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "chat_template_kwargs"
    )]
    pub chat_template_args: Option<std::collections::HashMap<String, serde_json::Value>>,

    /// Runtime media decoding parameters.
    /// When provided, these override the MDC defaults
    /// Example: `{"video": {"num_frames": 16}}`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_io_kwargs: Option<MediaDecoder>,

    /// When true, logprob token fields are returned as "token_id:<id>" instead
    /// of decoded text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_tokens_as_token_ids: Option<bool>,

    /// Catch-all for unsupported fields - checked during validation
    #[serde(flatten, default, skip_serializing)]
    pub unsupported_fields: std::collections::HashMap<String, serde_json::Value>,
}

/// A response structure for unary chat completion responses, embedding OpenAI's
/// `CreateChatCompletionResponse` with optional NVIDIA extension metadata.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NvCreateChatCompletionResponse {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::CreateChatCompletionResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<serde_json::Value>,
}

/// A response structure for streamed chat completions, embedding OpenAI's
/// `CreateChatCompletionStreamResponse` with optional NVIDIA extension metadata.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NvCreateChatCompletionStreamResponse {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::CreateChatCompletionStreamResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<serde_json::Value>,
}

impl NvCreateChatCompletionRequest {
    fn request_text_len(&self) -> Option<usize> {
        serde_json::to_value(&self.inner.messages)
            .ok()
            .map(|value| json_string_len(&value))
            .filter(|length| *length > 0)
    }

    pub(crate) fn uses_pure_json_structured_output(&self) -> bool {
        if self
            .inner
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
        {
            return false;
        }

        if self.common.guided_json.is_some() {
            return true;
        }

        if self
            .common
            .structured_outputs
            .as_ref()
            .is_some_and(|structured| {
                structured.json.is_some() || structured.json_object.unwrap_or(false)
            })
        {
            return true;
        }

        matches!(
            self.inner.response_format.as_ref(),
            Some(
                dynamo_protocols::types::ResponseFormat::JsonObject
                    | dynamo_protocols::types::ResponseFormat::JsonSchema { .. }
            )
        )
    }

    fn uses_qwen_xml_tool_structural_tag(&self) -> bool {
        let has_named_tool_choice = matches!(
            self.inner.tool_choice.as_ref(),
            Some(dynamo_protocols::types::ChatCompletionToolChoiceOption::Named(_))
        );
        let has_required_tool_choice = matches!(
            self.inner.tool_choice.as_ref(),
            Some(dynamo_protocols::types::ChatCompletionToolChoiceOption::Required)
        );
        if has_named_tool_choice
            && tools::named_tool_choice_has_freeform_text_field(
                self.inner.tool_choice.as_ref(),
                self.inner.tools.as_deref(),
            )
        {
            return false;
        }
        if !(has_named_tool_choice || has_required_tool_choice)
            || !self
                .inner
                .tools
                .as_ref()
                .is_some_and(|tools| !tools.is_empty())
        {
            return false;
        }

        let parser = std::env::var("DYN_DYNAMO_TOOL_CALL_PARSER")
            .or_else(|_| std::env::var("DYN_TOOL_CALL_PARSER"))
            .unwrap_or_default();
        matches!(parser.trim(), "qwen3_coder" | "qwen3_xml" | "nemotron_nano")
    }

    pub fn normalize_reasoning_controls(&mut self) {
        self.normalize_template_thinking_aliases();

        let explicit_template_thinking = self.has_explicit_template_thinking();
        if !explicit_template_thinking {
            if let Some(effort) = self.reasoning_effort_string() {
                self.apply_reasoning_effort_template_args(&effort);
            }
        }

        let Some(reasoning) = self.unsupported_fields.remove("reasoning") else {
            return;
        };
        let Some(reasoning) = reasoning.as_object() else {
            return;
        };

        if !self.unsupported_fields.contains_key("reasoning_budget")
            && let Some(max_tokens) = reasoning.get("max_tokens")
        {
            self.unsupported_fields
                .insert("reasoning_budget".to_string(), max_tokens.clone());
        }

        if reasoning
            .get("enabled")
            .is_some_and(|enabled| enabled == &serde_json::Value::Bool(false))
        {
            if !explicit_template_thinking {
                self.set_chat_template_arg_if_absent("enable_thinking", serde_json::json!(false));
            }
            return;
        }

        if explicit_template_thinking || self.inner.reasoning_effort.is_some() {
            return;
        }

        if let Some(effort) = reasoning.get("effort").and_then(|value| value.as_str()) {
            if let Ok(parsed) = serde_json::from_value::<dynamo_protocols::types::ReasoningEffort>(
                serde_json::Value::String(effort.to_string()),
            ) {
                self.inner.reasoning_effort = Some(parsed);
            }
            self.apply_reasoning_effort_template_args(effort);
        } else if reasoning
            .get("enabled")
            .is_some_and(|enabled| enabled == &serde_json::Value::Bool(true))
        {
            self.set_chat_template_arg_if_absent("enable_thinking", serde_json::json!(true));
        }
    }

    fn has_explicit_template_thinking(&self) -> bool {
        self.chat_template_args.as_ref().is_some_and(|args| {
            args.contains_key("enable_thinking")
                || args.contains_key("thinking")
                || args.contains_key("low_effort")
                || args.contains_key("medium_effort")
        })
    }

    fn normalize_template_thinking_aliases(&mut self) {
        let Some(thinking) = self
            .chat_template_args
            .as_ref()
            .and_then(|args| args.get("thinking"))
            .and_then(serde_json::Value::as_bool)
        else {
            return;
        };

        self.set_chat_template_arg_if_absent("enable_thinking", serde_json::json!(thinking));
    }

    fn reasoning_effort_string(&self) -> Option<String> {
        let value = serde_json::to_value(self.inner.reasoning_effort.as_ref()?).ok()?;
        value.as_str().map(|effort| effort.to_ascii_lowercase())
    }

    fn set_chat_template_arg_if_absent(&mut self, key: &str, value: serde_json::Value) {
        self.chat_template_args
            .get_or_insert_with(Default::default)
            .entry(key.to_string())
            .or_insert(value);
    }

    fn apply_reasoning_effort_template_args(&mut self, effort: &str) {
        match effort {
            "none" => {
                self.set_chat_template_arg_if_absent("enable_thinking", serde_json::json!(false));
            }
            "minimal" | "low" | "medium" => {
                self.set_chat_template_arg_if_absent("enable_thinking", serde_json::json!(true));
                self.set_chat_template_arg_if_absent("low_effort", serde_json::json!(true));
                self.set_chat_template_arg_if_absent("medium_effort", serde_json::json!(true));
            }
            "high" | "xhigh" | "max" => {
                self.set_chat_template_arg_if_absent("enable_thinking", serde_json::json!(true));
            }
            _ => {}
        }
    }

    fn template_thinking_enabled(&self) -> bool {
        self.chat_template_args.as_ref().is_some_and(|args| {
            let enable_thinking = args
                .get("enable_thinking")
                .and_then(serde_json::Value::as_bool);
            let thinking = args.get("thinking").and_then(serde_json::Value::as_bool);
            if enable_thinking == Some(false) || thinking == Some(false) {
                return false;
            }

            enable_thinking == Some(true)
                || thinking == Some(true)
                || args.get("low_effort").and_then(serde_json::Value::as_bool) == Some(true)
                || args
                    .get("medium_effort")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
        })
    }

    fn uses_constrained_decoding(&self) -> bool {
        let tools_enabled = self
            .inner
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
            && !matches!(
                self.inner.tool_choice.as_ref(),
                Some(dynamo_protocols::types::ChatCompletionToolChoiceOption::None)
            );
        if tools_enabled {
            return true;
        }

        if self.common.guided_json.is_some()
            || self.common.guided_regex.is_some()
            || self.common.guided_grammar.is_some()
            || self.common.guided_choice.is_some()
        {
            return true;
        }

        if self
            .common
            .structured_outputs
            .as_ref()
            .is_some_and(|structured| {
                structured.json.is_some()
                    || structured.json_object.unwrap_or(false)
                    || structured.regex.is_some()
                    || structured.grammar.is_some()
                    || structured.choice.is_some()
                    || structured.structural_tag.is_some()
            })
        {
            return true;
        }

        matches!(
            self.inner.response_format.as_ref(),
            Some(
                dynamo_protocols::types::ResponseFormat::JsonObject
                    | dynamo_protocols::types::ResponseFormat::JsonSchema { .. }
            )
        )
    }
}

/// Implements `NvExtProvider` for `NvCreateChatCompletionRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateChatCompletionRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    /// Returns `None`, as raw prompt extraction is not implemented.
    fn raw_prompt(&self) -> Option<String> {
        None
    }
}

/// Implements `AnnotationsProvider` for `NvCreateChatCompletionRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    ///
    /// # Arguments
    /// * `annotation` - A string slice representing the annotation to check.
    ///
    /// # Returns
    /// `true` if the annotation exists, `false` otherwise.
    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

/// Implements `OpenAISamplingOptionsProvider` for `NvCreateChatCompletionRequest`,
/// exposing OpenAI's sampling parameters for chat completion.
impl OpenAISamplingOptionsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the temperature parameter for sampling, if set.
    fn get_temperature(&self) -> Option<f32> {
        self.inner.temperature
    }

    /// Retrieves the top-p (nucleus sampling) parameter, if set.
    fn get_top_p(&self) -> Option<f32> {
        self.inner.top_p
    }

    /// Retrieves the frequency penalty parameter, if set.
    fn get_frequency_penalty(&self) -> Option<f32> {
        self.inner.frequency_penalty
    }

    /// Retrieves the presence penalty parameter, if set.
    fn get_presence_penalty(&self) -> Option<f32> {
        self.inner.presence_penalty
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
    /// Retrieves the seed value for random number generation, if set.
    fn get_seed(&self) -> Option<i64> {
        self.inner.seed
    }

    /// Retrieves the number of completions to generate for each prompt, if set.
    fn get_n(&self) -> Option<u8> {
        self.inner.n
    }

    /// Retrieves the best_of parameter, if set.
    fn get_best_of(&self) -> Option<u8> {
        None // Not supported in chat completions
    }
}

/// Implements `CommonExtProvider` for `NvCreateChatCompletionRequest`,
/// providing access to common extension fields.
impl CommonExtProvider for NvCreateChatCompletionRequest {
    /// Returns a reference to the CommonExt struct.
    fn common_ext(&self) -> Option<&CommonExt> {
        Some(&self.common)
    }

    /// Guided Decoding Options
    fn get_guided_json(&self) -> Option<serde_json::Value> {
        if self.uses_qwen_xml_tool_structural_tag() {
            return None;
        }

        if let Some(value) = self.common.guided_json.clone() {
            return Some(guided_json_schema(value));
        }

        let has_explicit_structural_tag = self
            .common
            .structured_outputs
            .as_ref()
            .is_some_and(|structured| structured.structural_tag.is_some());

        // 1) Tool-call guided decoding (highest precedence after explicit guided_json)
        if !has_explicit_structural_tag {
            if let (Some(tool_choice), Some(tools)) =
                (self.inner.tool_choice.as_ref(), self.inner.tools.as_deref())
            {
                match tools::get_json_schema_from_tools(
                    Some(tool_choice),
                    Some(tools),
                    self.request_text_len(),
                ) {
                    Ok(Some(schema)) => return Some(tool_choice_guided_json_schema(schema)),
                    Ok(None) => {}
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            "failed to derive guided_json from tool_choice"
                        );
                    }
                }
            }
        }

        // 2) vLLM `structured_outputs` alias (applies to assistant content, not tool calls)
        if let Some(value) = self
            .common
            .structured_outputs
            .as_ref()
            .and_then(|structured| structured.json.clone())
        {
            return Some(guided_json_schema(value));
        }
        if self
            .common
            .structured_outputs
            .as_ref()
            .and_then(|structured| structured.json_object)
            .unwrap_or(false)
        {
            return Some(json_object_schema());
        }

        // 3) OpenAI `response_format` (applies to assistant content, not tool calls)
        if let Some(response_format) = self.inner.response_format.as_ref() {
            use dynamo_protocols::types::ResponseFormat;
            match response_format {
                ResponseFormat::Text => {}
                ResponseFormat::JsonObject => {
                    return Some(json_object_schema());
                }
                ResponseFormat::JsonSchema { json_schema } => {
                    // validate_response_format ensures schema is present when type=json_schema
                    if let Some(schema) = json_schema.schema.clone() {
                        return Some(guided_json_schema(schema));
                    }
                }
            }
        }

        None
    }

    fn get_guided_structural_tag(&self) -> Option<serde_json::Value> {
        if let Some(value) = self
            .common
            .structured_outputs
            .as_ref()
            .and_then(|structured| structured.structural_tag.clone())
        {
            return Some(value);
        }

        if !self.uses_qwen_xml_tool_structural_tag() {
            return None;
        }

        if let (Some(tool_choice), Some(tools)) =
            (self.inner.tool_choice.as_ref(), self.inner.tools.as_deref())
        {
            match tools::get_qwen_xml_structural_tag_from_tools(
                Some(tool_choice),
                Some(tools),
                self.request_text_len(),
            ) {
                Ok(Some(structural_tag)) => return Some(structural_tag),
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "failed to derive structural_tag from tool_choice"
                    );
                }
            }
        }

        None
    }

    fn get_guided_regex(&self) -> Option<String> {
        self.common.guided_regex.clone().or_else(|| {
            self.common
                .structured_outputs
                .as_ref()
                .and_then(|structured| structured.regex.clone())
        })
    }

    fn get_guided_grammar(&self) -> Option<String> {
        self.common.guided_grammar.clone().or_else(|| {
            self.common
                .structured_outputs
                .as_ref()
                .and_then(|structured| structured.grammar.clone())
        })
    }

    fn get_guided_choice(&self) -> Option<Vec<String>> {
        self.common.guided_choice.clone().or_else(|| {
            self.common
                .structured_outputs
                .as_ref()
                .and_then(|structured| structured.choice.clone())
        })
    }

    fn get_guided_decoding_backend(&self) -> Option<String> {
        self.common.guided_decoding_backend.clone()
    }

    fn get_guided_whitespace_pattern(&self) -> Option<String> {
        self.common.guided_whitespace_pattern.clone()
    }

    fn get_top_k(&self) -> Option<i32> {
        self.common.top_k
    }

    fn get_min_p(&self) -> Option<f32> {
        self.common.min_p
    }

    fn get_repetition_penalty(&self) -> Option<f32> {
        self.common.repetition_penalty
    }

    fn get_include_stop_str_in_output(&self) -> Option<bool> {
        self.common.include_stop_str_in_output
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        self.common.skip_special_tokens
    }
}

/// Implements `OpenAIStopConditionsProvider` for `NvCreateChatCompletionRequest`,
/// providing access to stop conditions that control chat completion behavior.
impl OpenAIStopConditionsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the maximum number of tokens allowed in the response.
    #[allow(deprecated)]
    fn get_max_tokens(&self) -> Option<u32> {
        self.inner.max_completion_tokens.or(self.inner.max_tokens)
    }

    /// Retrieves the minimum number of tokens required in the response.
    /// Returns `min_tokens` Value
    /// `min_tokens` is not an OpenAI-supported parameter.
    fn get_min_tokens(&self) -> Option<u32> {
        self.common.min_tokens
    }

    /// Retrieves the stop conditions that terminate the chat completion response.
    ///
    /// Converts OpenAI's `Stop` enum to a `Vec<String>`, normalizing the representation.
    ///
    /// # Returns
    /// * `Some(Vec<String>)` if stop conditions are set.
    /// * `None` if no stop conditions are defined.
    fn get_stop(&self) -> Option<Vec<String>> {
        self.inner.stop.as_ref().and_then(|stop| stop.strings())
    }

    fn get_stop_token_ids(&self) -> Option<Vec<crate::types::TokenIdType>> {
        self.inner.stop.as_ref().and_then(|stop| stop.token_ids())
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn get_max_thinking_tokens(&self) -> Option<u32> {
        if let Some(value) = self.nvext.as_ref().and_then(|nv| nv.max_thinking_tokens) {
            return Some(value);
        }

        if let Some(value) = self.unsupported_fields.get("reasoning_budget").or_else(|| {
            self.chat_template_args.as_ref().and_then(|args| {
                args.get("reasoning_budget")
                    .or_else(|| args.get("thinking_token_budget"))
            })
        }) {
            if let Some(value) = value.as_u64() {
                return u32::try_from(value).ok();
            }
            if let Some(value) = value.as_i64() {
                return if value < 0 {
                    None
                } else {
                    u32::try_from(value).ok()
                };
            }
            return None;
        }

        if self.template_thinking_enabled() && self.uses_constrained_decoding() {
            return default_constrained_max_thinking_tokens();
        }

        None
    }

    /// Get ignore_eos from CommonExt.
    fn get_common_ignore_eos(&self) -> Option<bool> {
        self.common.ignore_eos
    }

    /// Get the effective ignore_eos value from CommonExt.
    fn get_ignore_eos(&self) -> Option<bool> {
        self.common.ignore_eos
    }
}

impl OpenAIOutputOptionsProvider for NvCreateChatCompletionRequest {
    fn get_logprobs(&self) -> Option<u32> {
        match self.inner.logprobs {
            Some(true) => match self.inner.top_logprobs {
                Some(top_logprobs) => Some(top_logprobs as u32),
                None => Some(1_u32),
            },
            Some(false) => None,
            None => None,
        }
    }

    fn get_prompt_logprobs(&self) -> Option<u32> {
        None
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        CommonExtProvider::get_skip_special_tokens(self)
    }

    fn get_formatted_prompt(&self) -> Option<bool> {
        None
    }

    fn get_return_tokens_as_token_ids(&self) -> Option<bool> {
        self.return_tokens_as_token_ids
    }
}

/// Implements `ValidateRequest` for `NvCreateChatCompletionRequest`,
/// allowing us to validate the data.
impl ValidateRequest for NvCreateChatCompletionRequest {
    fn validate(&self) -> Result<(), anyhow::Error> {
        validate::validate_no_unsupported_fields(&self.unsupported_fields)?;
        validate::validate_messages(&self.inner.messages)?;
        validate::validate_model(&self.inner.model)?;
        // none for store
        validate::validate_reasoning_effort(&self.inner.reasoning_effort)?;
        // none for metadata
        validate::validate_frequency_penalty(self.inner.frequency_penalty)?;
        validate::validate_logit_bias(&self.inner.logit_bias)?;
        // none for logprobs
        validate::validate_top_logprobs(self.inner.top_logprobs)?;
        // validate::validate_max_tokens(self.inner.max_tokens)?; // warning depricated field
        validate::validate_max_completion_tokens(self.inner.max_completion_tokens)?;
        validate::validate_n(self.inner.n)?;
        // none for modalities
        // none for prediction
        // none for audio
        validate::validate_presence_penalty(self.inner.presence_penalty)?;
        validate::validate_response_format(&self.inner.response_format)?;
        validate::validate_structured_outputs(&self.common.structured_outputs)?;
        // none for seed
        validate::validate_service_tier(&self.inner.service_tier)?;
        validate::validate_stop(&self.inner.stop)?;
        // none for stream
        // none for stream_options
        validate::validate_temperature(self.inner.temperature)?;
        validate::validate_top_p(self.inner.top_p)?;
        validate::validate_tools(&self.inner.tools.as_deref())?;
        validate::validate_tool_choice(&self.inner.tool_choice, &self.inner.tools.as_deref())?;
        // none for parallel_tool_calls
        validate::validate_user(self.inner.user.as_deref())?;
        // none for function call
        // none for functions
        // Common Ext
        validate::validate_repetition_penalty(self.get_repetition_penalty())?;
        validate::validate_min_p(self.get_min_p())?;
        validate::validate_top_k(self.get_top_k())?;
        // Cross-field validation
        validate::validate_n_with_temperature(self.inner.n, self.inner.temperature)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::ValidateRequest;
    use crate::protocols::common::{
        OutputOptionsProvider, SamplingOptionsProvider, StopConditionsProvider,
    };
    use serde_json::json;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_tool_parser<T>(parser: &str, f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let old = std::env::var_os("DYN_DYNAMO_TOOL_CALL_PARSER");
        unsafe {
            std::env::set_var("DYN_DYNAMO_TOOL_CALL_PARSER", parser);
        }
        let result = f();
        unsafe {
            match old {
                Some(value) => std::env::set_var("DYN_DYNAMO_TOOL_CALL_PARSER", value),
                None => std::env::remove_var("DYN_DYNAMO_TOOL_CALL_PARSER"),
            }
        }
        result
    }

    fn with_default_constrained_max_thinking_tokens<T>(value: &str, f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let old = std::env::var_os(DEFAULT_CONSTRAINED_MAX_THINKING_TOKENS_ENV);
        unsafe {
            std::env::set_var(DEFAULT_CONSTRAINED_MAX_THINKING_TOKENS_ENV, value);
        }
        let result = f();
        unsafe {
            match old {
                Some(value) => {
                    std::env::set_var(DEFAULT_CONSTRAINED_MAX_THINKING_TOKENS_ENV, value)
                }
                None => std::env::remove_var(DEFAULT_CONSTRAINED_MAX_THINKING_TOKENS_ENV),
            }
        }
        result
    }

    #[test]
    fn test_skip_special_tokens_none() {
        let json_str = json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        assert_eq!(request.common.skip_special_tokens, None);

        let output_options = request
            .extract_output_options()
            .expect("Failed to extract output options");

        assert_eq!(output_options.skip_special_tokens, None);
    }

    #[test]
    fn test_skip_special_tokens_propagates() {
        for skip_value in [true, false] {
            let json_str = json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "skip_special_tokens": skip_value
            });

            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(json_str).expect("Failed to deserialize request");

            let output_options = request
                .extract_output_options()
                .expect("Failed to extract output options");

            assert_eq!(output_options.skip_special_tokens, Some(skip_value));
        }
    }

    #[test]
    fn test_reasoning_effort_low_sets_size_specific_template_flags() {
        let mut request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "low"
        }))
        .unwrap();

        request.normalize_reasoning_controls();

        let args = request.chat_template_args.unwrap();
        assert_eq!(args.get("enable_thinking"), Some(&json!(true)));
        assert_eq!(args.get("low_effort"), Some(&json!(true)));
        assert_eq!(args.get("medium_effort"), Some(&json!(true)));
    }

    #[test]
    fn test_reasoning_effort_high_sets_max_thinking_template_flags() {
        let mut request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "max"
        }))
        .unwrap();

        request.normalize_reasoning_controls();

        let args = request.chat_template_args.unwrap();
        assert_eq!(args.get("enable_thinking"), Some(&json!(true)));
        assert!(!args.contains_key("low_effort"));
        assert!(!args.contains_key("medium_effort"));
    }

    #[test]
    fn test_reasoning_effort_none_disables_template_thinking() {
        let mut request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "none"
        }))
        .unwrap();

        request.normalize_reasoning_controls();

        let args = request.chat_template_args.unwrap();
        assert_eq!(args.get("enable_thinking"), Some(&json!(false)));
    }

    #[test]
    fn test_template_thinking_alias_sets_enable_thinking() {
        let mut request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"thinking": false}
        }))
        .unwrap();

        request.normalize_reasoning_controls();

        let args = request.chat_template_args.unwrap();
        assert_eq!(args.get("thinking"), Some(&json!(false)));
        assert_eq!(args.get("enable_thinking"), Some(&json!(false)));
    }

    #[test]
    fn test_explicit_template_reasoning_flags_win() {
        let mut request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .unwrap();

        request.normalize_reasoning_controls();

        let args = request.chat_template_args.unwrap();
        assert_eq!(args.get("enable_thinking"), Some(&json!(false)));
        assert!(!args.contains_key("low_effort"));
        assert!(!args.contains_key("medium_effort"));
    }

    #[test]
    fn test_structured_outputs_alias_maps_to_guided_decoding() {
        let schema = json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"]
        });
        let json_str = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Answer as JSON"}],
            "structured_outputs": {
                "json": schema,
                "regex": "[A-Z]{2}-\\d{3}",
                "grammar": "root ::= \"PASS\"",
                "choice": ["PASS", "FAIL"],
                "structural_tag": {"type": "structural_tag", "format": {"type": "tag"}}
            }
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        assert!(request.unsupported_fields.is_empty());
        assert_eq!(request.get_guided_json(), Some(schema));
        assert_eq!(
            request.get_guided_regex().as_deref(),
            Some("[A-Z]{2}-\\d{3}")
        );
        assert_eq!(
            request.get_guided_grammar().as_deref(),
            Some("root ::= \"PASS\"")
        );
        assert_eq!(
            request.get_guided_choice(),
            Some(vec!["PASS".to_string(), "FAIL".to_string()])
        );
        assert_eq!(
            request.get_guided_structural_tag(),
            Some(json!({"type": "structural_tag", "format": {"type": "tag"}}))
        );
    }

    #[test]
    fn test_structured_outputs_json_object_maps_to_bounded_object_schema() {
        let json_str = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Answer as JSON"}],
            "structured_outputs": {"json_object": true}
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        let guided = request.get_guided_json().expect("guided json");

        assert_eq!(guided["type"], json!("object"));
        assert_eq!(guided["minProperties"], json!(1));
        assert_eq!(guided["maxProperties"], json!(64));
    }

    #[test]
    fn test_tool_choice_requires_tools() {
        let json_str = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Call a tool"}],
            "tool_choice": "required"
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        let err = ValidateRequest::validate(&request).expect_err("tool_choice requires tools");

        assert!(
            err.to_string()
                .contains("When using `tool_choice`, `tools` must be set")
        );
    }

    #[test]
    fn test_wildcard_pattern_string_schema_gets_bounded() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let old = std::env::var_os("DYN_XGRAMMAR_DEFAULT_MAX_STRING_LENGTH");
        unsafe {
            std::env::set_var("DYN_XGRAMMAR_DEFAULT_MAX_STRING_LENGTH", "64");
        }

        let schema = json!({
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "pattern": ".*He said \\\"ready\\\", proceed\\..*"
                            }
                        }
                    }
                },
                "code": {
                    "type": "string",
                    "pattern": "^[A-Z]+$"
                }
            }
        });
        let json_str = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Answer as JSON"}],
            "structured_outputs": {"json": schema}
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        let guided = request.get_guided_json().expect("guided json");

        assert_eq!(
            guided["properties"]["records"]["items"]["properties"]["message"]["maxLength"],
            json!(64)
        );
        assert_eq!(
            guided["properties"]["records"]["items"]["properties"]["message"]["pattern"],
            json!("^.{0,64}He said \\\"ready\\\", proceed\\..{0,64}$")
        );
        assert!(
            guided["properties"]["code"].get("maxLength").is_none(),
            "non-wildcard patterns should not be capped"
        );

        unsafe {
            match old {
                Some(value) => std::env::set_var("DYN_XGRAMMAR_DEFAULT_MAX_STRING_LENGTH", value),
                None => std::env::remove_var("DYN_XGRAMMAR_DEFAULT_MAX_STRING_LENGTH"),
            }
        }
    }

    #[test]
    fn test_named_tool_choice_schema_precedes_structured_outputs_json() {
        let structured_schema = json!({
            "type": "object",
            "properties": {
                "wrong": {"type": "string"}
            },
            "required": ["wrong"]
        });
        let tool_schema = json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        });
        let json_str = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Call get_weather"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": tool_schema
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"}
            },
            "structured_outputs": {"json": structured_schema}
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        with_tool_parser("hermes", || {
	            assert_eq!(
	                request.get_guided_json(),
	                Some(json!({
	                    "type": "object",
	                    "properties": {
	                        "location": {"type": "string", "maxLength": 1024}
	                    },
	                    "required": ["location"],
	                    "additionalProperties": false,
	                    "x-dynamo-tool-choice-schema": true
	                }))
	            );
        });
    }

    #[test]
    fn test_named_tool_choice_qwen_parser_uses_structural_tag() {
        let tool_schema = json!({
            "type": "object",
            "properties": {
                "note": {"type": "string"}
            },
            "required": ["note"]
        });
        let json_str = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Call send_message"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "send_message",
                    "parameters": tool_schema
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "send_message"}
            }
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        with_tool_parser("qwen3_coder", || {
            assert_eq!(request.get_guided_json(), None);
            let structural_tag = request.get_guided_structural_tag().expect("structural tag");
            let tag_format = &structural_tag["format"];
            assert_eq!(tag_format["type"], "tags_with_separator");
            assert_eq!(tag_format["at_least_one"], true);
            assert_eq!(tag_format["stop_after_first"], true);
            let tag = &tag_format["tags"][0];
            assert_eq!(tag["begin"], "<tool_call>\n<function=send_message>\n");
            assert_eq!(tag["content"]["style"], "qwen_xml");
            assert_eq!(
                tag["content"]["json_schema"]["properties"]["note"],
                json!({"type": "string", "maxLength": 1024})
            );

            let sampling = request
                .extract_sampling_options()
                .expect("extract sampling options");
            let guided = sampling.guided_decoding.expect("guided decoding");
            assert!(guided.json.is_none());
            assert_eq!(guided.structural_tag, Some(structural_tag));
        });
    }

    #[test]
    fn test_named_tool_choice_qwen_parser_text_field_uses_json_schema() {
        let tool_schema = json!({
            "type": "object",
            "properties": {
                "body": {"type": "string"}
            },
            "required": ["body"]
        });
        let json_str = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Call send_message"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "send_message",
                    "parameters": tool_schema
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "send_message"}
            }
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        with_tool_parser("qwen3_coder", || {
            assert!(request.get_guided_structural_tag().is_none());
	            assert_eq!(
	                request.get_guided_json(),
	                Some(json!({
	                    "type": "object",
	                    "properties": {
	                        "body": {"type": "string", "maxLength": 256}
	                    },
	                    "required": ["body"],
	                    "additionalProperties": false,
	                    "x-dynamo-tool-choice-schema": true
	                }))
	            );
        });
    }

    #[test]
    fn test_named_tool_choice_non_qwen_parser_keeps_json_schema() {
        let tool_schema = json!({
            "type": "object",
            "properties": {
                "body": {"type": "string"}
            },
            "required": ["body"]
        });
        let json_str = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Call send_message"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "send_message",
                    "parameters": tool_schema
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "send_message"}
            }
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        with_tool_parser("hermes", || {
            assert!(request.get_guided_structural_tag().is_none());
	            assert_eq!(
	                request.get_guided_json(),
	                Some(json!({
	                    "type": "object",
	                    "properties": {
	                        "body": {"type": "string", "maxLength": 256}
	                    },
	                    "required": ["body"],
	                    "additionalProperties": false,
	                    "x-dynamo-tool-choice-schema": true
	                }))
	            );
        });
    }

    #[test]
    fn test_named_tool_choice_long_request_expands_body_budget() {
        let tool_schema = json!({
            "type": "object",
            "properties": {
                "body": {"type": "string"}
            },
            "required": ["body"]
        });
        let json_str = json!({
            "model": "nvidia/nemotron-3-nano-30b-a3b",
            "messages": [{
                "role": "user",
                "content": "Send this exact long body: ".to_string() + &"abcdef ".repeat(600)
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "send_message",
                    "parameters": tool_schema
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "send_message"}
            }
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        with_tool_parser("qwen3_coder", || {
            assert!(request.get_guided_structural_tag().is_none());
            let guided_json = request.get_guided_json().expect("guided json");
            assert_eq!(
                guided_json["properties"]["body"],
                json!({"type": "string", "maxLength": 8192})
            );
        });
    }

    #[test]
    fn test_named_tool_choice_nemotron_model_uses_structural_tag() {
        let tool_schema = json!({
            "type": "object",
            "properties": {
                "note": {"type": "string"}
            },
            "required": ["note"]
        });
        let json_str = json!({
            "model": "nvidia/nemotron-3-nano-30b-a3b",
            "messages": [{"role": "user", "content": "Call send_message"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "send_message",
                    "parameters": tool_schema
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "send_message"}
            }
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        with_tool_parser("qwen3_coder", || {
            assert!(request.get_guided_json().is_none());
            let structural_tag = request.get_guided_structural_tag().expect("structural tag");
            let format = &structural_tag["format"];
            assert_eq!(format["type"], json!("tags_with_separator"));
            assert_eq!(format["at_least_one"], json!(true));
            assert_eq!(format["stop_after_first"], json!(true));
        });
    }

    #[test]
    fn test_required_tool_choice_nemotron_model_uses_structural_tag() {
        let tool_schema = json!({
            "type": "object",
            "properties": {
                "body": {"type": "string"}
            },
            "required": ["body"]
        });
        let json_str = json!({
            "model": "nvidia/nemotron-3-nano-30b-a3b",
            "messages": [{"role": "user", "content": "Call send_message"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "send_message",
                    "parameters": tool_schema
                }
            }],
            "tool_choice": "required"
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        with_tool_parser("qwen3_coder", || {
            assert!(request.get_guided_json().is_none());
            let structural_tag = request.get_guided_structural_tag().expect("structural tag");
            let format = &structural_tag["format"];
            assert_eq!(format["type"], json!("tags_with_separator"));
            assert_eq!(format["at_least_one"], json!(true));
            assert_eq!(format["stop_after_first"], json!(true));
            assert_eq!(format["tags"].as_array().unwrap().len(), 1);
        });
    }

    #[test]
    fn test_explicit_reasoning_budget_maps_to_max_thinking_tokens() {
        let json_str = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Answer as JSON"}],
            "max_tokens": 65536,
            "chat_template_kwargs": {
                "enable_thinking": true,
                "reasoning_budget": 1024
            },
            "structured_outputs": {
                "json": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"]
                }
            }
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        let stop_conditions = request
            .extract_stop_conditions()
            .expect("extract stop conditions");

        assert_eq!(stop_conditions.max_thinking_tokens, Some(1024));
    }

    #[test]
    fn test_root_reasoning_budget_maps_to_max_thinking_tokens() {
        let json_str = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Call get_weather"}],
            "max_tokens": 65536,
            "reasoning_budget": 1024,
            "chat_template_kwargs": {"enable_thinking": true},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }],
            "tool_choice": "auto"
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        let stop_conditions = request
            .extract_stop_conditions()
            .expect("extract stop conditions");

        assert_eq!(stop_conditions.max_thinking_tokens, Some(1024));
    }

    #[test]
    fn test_constrained_thinking_default_applies_to_tool_requests() {
        with_default_constrained_max_thinking_tokens("0", || {
            let json_str = json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Call get_weather"}],
                "max_tokens": 65536,
                "chat_template_kwargs": {"enable_thinking": true},
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"]
                        }
                    }
                }],
                "tool_choice": "auto"
            });

            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(json_str).expect("Failed to deserialize request");
            let stop_conditions = request
                .extract_stop_conditions()
                .expect("extract stop conditions");

            assert_eq!(stop_conditions.max_thinking_tokens, Some(0));
        });
    }

    #[test]
    fn test_constrained_thinking_default_does_not_override_explicit_budget() {
        with_default_constrained_max_thinking_tokens("0", || {
            let json_str = json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Answer as JSON"}],
                "max_tokens": 65536,
                "chat_template_kwargs": {
                    "enable_thinking": true,
                    "reasoning_budget": 1024
                },
                "structured_outputs": {
                    "json": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"]
                    }
                }
            });

            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(json_str).expect("Failed to deserialize request");
            let stop_conditions = request
                .extract_stop_conditions()
                .expect("extract stop conditions");

            assert_eq!(stop_conditions.max_thinking_tokens, Some(1024));
        });
    }

    #[test]
    fn test_constrained_thinking_default_skips_plain_chat_and_thinking_off() {
        with_default_constrained_max_thinking_tokens("0", || {
            let plain_chat = json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 65536,
                "chat_template_kwargs": {"enable_thinking": true}
            });
            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(plain_chat).expect("Failed to deserialize request");
            let stop_conditions = request
                .extract_stop_conditions()
                .expect("extract stop conditions");
            assert_eq!(stop_conditions.max_thinking_tokens, None);

            let thinking_off_tool = json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Call get_weather"}],
                "max_tokens": 65536,
                "chat_template_kwargs": {"enable_thinking": false},
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }],
                "tool_choice": "auto"
            });
            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(thinking_off_tool).expect("Failed to deserialize request");
            let stop_conditions = request
                .extract_stop_conditions()
                .expect("extract stop conditions");
            assert_eq!(stop_conditions.max_thinking_tokens, None);
        });
    }

    #[test]
    fn test_stop_contract() {
        let one_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": " The"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(one_stop).expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec![" The".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let many_stops = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ["A", "B"]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(many_stops).expect("Failed to deserialize request");
        assert_eq!(
            request.get_stop(),
            Some(vec!["A".to_string(), "B".to_string()])
        );
        assert_eq!(request.get_stop_token_ids(), None);

        let token_id_stops = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": [32, 34]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(token_id_stops).expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), None);
        assert_eq!(request.get_stop_token_ids(), Some(vec![32, 34]));

        let stop_conditions = request
            .extract_stop_conditions()
            .expect("extract stop conditions");
        assert_eq!(stop_conditions.stop, None);
        assert_eq!(stop_conditions.stop_token_ids, Some(vec![32, 34]));

        let token_id_display_string_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": "token_id:576"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(token_id_display_string_stop)
                .expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec!["token_id:576".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let token_id_display_string_array_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ["token_id:576"]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(token_id_display_string_array_stop)
                .expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec!["token_id:576".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let scalar_token_id_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": 576
        });
        let result: Result<NvCreateChatCompletionRequest, _> =
            serde_json::from_value(scalar_token_id_stop);
        assert!(result.is_err());

        let unsupported_stop_token_ids = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop_token_ids": [576]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(unsupported_stop_token_ids)
                .expect("Failed to deserialize request");
        assert!(ValidateRequest::validate(&request).is_err());
    }
}
