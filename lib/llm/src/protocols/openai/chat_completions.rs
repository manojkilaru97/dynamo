// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

use crate::engines::ValidateRequest;
use crate::preprocessor::media::MediaDecoder;

use super::{
    OpenAIOutputOptionsProvider, OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider,
    common_ext::{CommonExt, CommonExtProvider},
    validate,
};
use crate::protocols::common::extensions::{
    NvExt, NvExtProvider, validate_completion_token_ids_single_choice,
};

pub mod aggregator;
mod delta;
pub mod tool_parser_v2;

pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

use dynamo_parsers::tool_calling::{ToolCallResponse, ToolCallResponseChunk};
use dynamo_protocols::types::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk, FunctionCall,
    FunctionCallStream, FunctionType,
};

/// Map a parser-native [`ToolCallResponse`] onto the protocol/wire
/// [`ChatCompletionMessageToolCall`].
///
/// `dynamo-parsers` is decoupled from `dynamo-protocols`, so this consumer —
/// which already depends on both — owns the mapping between the parser-native
/// types and the OpenAI wire types. The field shapes are identical, so this is
/// a straight re-map that preserves the previous wire output.
pub(crate) fn tool_call_response_to_protocol(
    parsed: ToolCallResponse,
) -> ChatCompletionMessageToolCall {
    ChatCompletionMessageToolCall {
        id: parsed.id,
        r#type: FunctionType::Function,
        function: FunctionCall {
            name: parsed.function.name,
            arguments: parsed.function.arguments,
        },
    }
}

/// Map a parser-native [`ToolCallResponseChunk`] onto the protocol/wire
/// [`ChatCompletionMessageToolCallChunk`]. See
/// [`tool_call_response_to_protocol`] for the rationale.
///
/// Exposed so consumers of the decoupled streaming parser entrypoint
/// ([`dynamo_parsers::tool_calling::try_tool_call_parse_stream`]) can recover
/// the wire type without `dynamo-parsers` depending on `dynamo-protocols`.
#[allow(dead_code)]
pub(crate) fn tool_call_response_chunk_to_protocol(
    parsed: ToolCallResponseChunk,
) -> ChatCompletionMessageToolCallChunk {
    ChatCompletionMessageToolCallChunk {
        index: parsed.index,
        id: parsed.id,
        r#type: parsed.tp.map(|_| FunctionType::Function),
        function: parsed.function.map(|f| FunctionCallStream {
            name: f.name,
            arguments: f.arguments,
        }),
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
    #[schema(value_type = Object)]
    pub nvext: Option<NvExt>,

    /// Extra args to pass to the chat template rendering context
    /// Also accepts "chat_template_kwargs" as an alias for compatibility
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "chat_template_kwargs"
    )]
    pub chat_template_args: Option<std::collections::HashMap<String, serde_json::Value>>,

    /// OpenAI-style thinking control from client request payloads.
    /// Normalized into `chat_template_args` before preprocessing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<serde_json::Value>,

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

impl NvCreateChatCompletionRequest {
    /// Normalize OpenAI-style DS-V4 reasoning controls into the template kwargs
    /// consumed by the SGLang/DeepSeek-V4 prompt formatter.
    pub fn normalize_reasoning_template_args(&mut self) -> anyhow::Result<()> {
        let thinking_mode = self
            .thinking
            .as_ref()
            .map(openai_thinking_mode)
            .transpose()?
            .flatten();
        let reasoning_effort = self
            .inner
            .reasoning_effort
            .as_ref()
            .and_then(|effort| serde_json::to_value(effort).ok());

        if thinking_mode.is_none() && reasoning_effort.is_none() {
            return Ok(());
        }

        let args = self.chat_template_args.get_or_insert_with(HashMap::new);
        if let Some(mode) = thinking_mode {
            match mode {
                OpenAiThinkingMode::Enabled => {
                    args.insert("thinking".to_string(), serde_json::Value::Bool(true));
                    args.insert(
                        "thinking_mode".to_string(),
                        serde_json::Value::String("enabled".to_string()),
                    );
                }
                OpenAiThinkingMode::Disabled => {
                    args.insert("thinking".to_string(), serde_json::Value::Bool(false));
                    args.insert(
                        "thinking_mode".to_string(),
                        serde_json::Value::String("disabled".to_string()),
                    );
                }
                OpenAiThinkingMode::Adaptive => {
                    args.insert(
                        "thinking_mode".to_string(),
                        serde_json::Value::String("adaptive".to_string()),
                    );
                }
            }
        }
        if let Some(effort) = reasoning_effort {
            args.insert("reasoning_effort".to_string(), effort);
        }

        // The raw `thinking` payload has been folded into `chat_template_args`;
        // drop it so it isn't double-shipped downstream (and so it can't be
        // re-interpreted with different precedence by the worker preprocessor).
        self.thinking = None;
        Ok(())
    }
}

enum OpenAiThinkingMode {
    Enabled,
    Disabled,
    Adaptive,
}

fn openai_thinking_mode(value: &serde_json::Value) -> anyhow::Result<Option<OpenAiThinkingMode>> {
    if let Some(enabled) = value.as_bool() {
        return Ok(Some(if enabled {
            OpenAiThinkingMode::Enabled
        } else {
            OpenAiThinkingMode::Disabled
        }));
    }

    let Some(thinking_object) = value.as_object() else {
        anyhow::bail!(
            "`thinking` must be a boolean or an object with `type` set to `enabled`, `disabled`, or `adaptive`"
        );
    };
    let Some(thinking_type) = thinking_object.get("type").and_then(|v| v.as_str()) else {
        anyhow::bail!("`thinking.type` must be `enabled`, `disabled`, or `adaptive`");
    };
    match thinking_type {
        "enabled" => Ok(Some(OpenAiThinkingMode::Enabled)),
        "disabled" => Ok(Some(OpenAiThinkingMode::Disabled)),
        "adaptive" => Ok(Some(OpenAiThinkingMode::Adaptive)),
        _ => anyhow::bail!("`thinking.type` must be `enabled`, `disabled`, or `adaptive`"),
    }
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
    /// Internal frontend metrics payload. This must never be serialized to
    /// client-facing OpenAI-compatible streams.
    #[serde(skip)]
    pub llm_metrics: Option<crate::protocols::common::metrics::LLMMetricAnnotation>,
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
        let tools = self.inner.tools.as_deref().unwrap_or(&[]);

        if has_named_tool_choice
            && tools::named_tool_choice_has_freeform_text_field(
                self.inner.tool_choice.as_ref(),
                Some(tools),
            )
        {
            return false;
        }
        if !(has_named_tool_choice || has_required_tool_choice) || tools.is_empty() {
            return false;
        }
        if has_required_tool_choice && tools.len() != 1 {
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

    fn unsupported_fields(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        Some(&self.unsupported_fields)
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

    fn get_prompt_logprobs_count(&self) -> Option<u32> {
        self.common.prompt_logprobs
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
        // Token IDs may be provided in the standard OpenAI `stop` array.
        if let Some(ids) = self.inner.stop.as_ref().and_then(|stop| stop.token_ids()) {
            return Some(ids);
        }
        // Also accept top-level `stop_token_ids` from passthrough clients.
        self.unsupported_fields
            .get("stop_token_ids")
            .and_then(|v| serde_json::from_value::<Vec<crate::types::TokenIdType>>(v.clone()).ok())
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
        // Top-level `prompt_logprobs` is carried through CommonExt.
        self.common.prompt_logprobs
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
        validate_completion_token_ids_single_choice(
            self.inner.n.unwrap_or(1) as usize,
            self.nvext.as_ref(),
        )?;
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
        validate::validate_tool_choice(&self.inner.tool_choice, self.inner.tools.as_deref())?;
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
    use crate::protocols::common::{OutputOptionsProvider, StopConditionsProvider};
    use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolType, FunctionObject};
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
    fn test_required_multi_tool_choice_nemotron_model_keeps_json_array_schema() {
        let json_str = json!({
            "model": "nvidia/nemotron-3-nano-30b-a3b",
            "messages": [{"role": "user", "content": "Call the needed tools"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"}
                            },
                            "required": ["expression"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            },
                            "required": ["location"]
                        }
                    }
                }
            ],
            "tool_choice": "required"
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        with_tool_parser("qwen3_coder", || {
            assert!(request.get_guided_structural_tag().is_none());
            let guided_json = request.get_guided_json().expect("guided json");
            assert_eq!(guided_json["type"], json!("array"));
            assert_eq!(guided_json["x-dynamo-tool-choice-schema"], json!(true));
            assert_eq!(
                guided_json["items"]["anyOf"].as_array().unwrap().len(),
                2
            );
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

        // `stop_token_ids` is accepted and plumbed by the provider trait.
        let whitelisted_stop_token_ids = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop_token_ids": [576]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(whitelisted_stop_token_ids)
                .expect("Failed to deserialize request");
        assert_eq!(request.get_stop_token_ids(), Some(vec![576]));
        assert!(
            ValidateRequest::validate(&request).is_ok(),
            "stop_token_ids must be accepted via PASSTHROUGH_EXTRA_FIELDS"
        );

        let invalid_stop_token_ids = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop_token_ids": "bad"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(invalid_stop_token_ids).expect("Failed to deserialize request");
        let err = ValidateRequest::validate(&request).expect_err("invalid stop_token_ids");
        assert!(err.to_string().contains("stop_token_ids"));
    }

    #[test]
    fn test_passthrough_token_constraints_validate() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "allowed_token_ids": [10, 11],
            "bad_words_token_ids": [[12, 13]]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        assert_eq!(
            request.unsupported_fields.get("allowed_token_ids"),
            Some(&serde_json::json!([10, 11]))
        );
        assert_eq!(
            request.unsupported_fields.get("bad_words_token_ids"),
            Some(&serde_json::json!([[12, 13]]))
        );
        assert!(ValidateRequest::validate(&request).is_ok());
    }

    #[test]
    fn test_completion_token_ids_rejected_for_multi_choice() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "n": 2,
            "nvext": {
                "extra_fields": ["completion_token_ids"]
            }
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        let err = ValidateRequest::validate(&request).expect_err("multi-choice token ids");
        assert!(err.to_string().contains("completion_token_ids"));
    }

    #[test]
    fn test_validate_tool_choice_required_rejects_empty_tools() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": "required"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        let err = ValidateRequest::validate(&request).expect_err("required needs tools");
        assert!(
            err.to_string()
                .contains("tool_choice is \"required\" but tools is empty")
        );
    }

    #[test]
    fn test_validate_tool_choice_named_rejects_missing_tool() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "search"}
            }
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        let err = ValidateRequest::validate(&request).expect_err("named tool must exist");
        assert!(
            err.to_string()
                .contains("tool named \"search\" in tool_choice is not present in tools")
        );
    }

    #[test]
    fn test_truncate_prompt_tokens_rejected_until_supported() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "truncate_prompt_tokens": 2
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        assert!(ValidateRequest::validate(&request).is_err());
    }

    // -----------------------------------------------------------------------
    // Parser -> protocol mapping (decoupling guard).
    //
    // `dynamo-parsers` no longer depends on `dynamo-protocols`; the mapping
    // moved into this consumer. These tests pin the mapper output to the
    // *exact* struct + serialized JSON the old protocol-typed parser path
    // produced, proving the wire output is unchanged.
    // -----------------------------------------------------------------------
    use dynamo_parsers::tool_calling::{
        CalledFunction, CalledFunctionStream, ToolCallResponse, ToolCallResponseChunk, ToolCallType,
    };

    fn native_call(id: &str, name: &str, args: &str) -> ToolCallResponse {
        ToolCallResponse {
            id: id.to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    fn native_chunk(index: u32, id: &str, name: &str, args: &str) -> ToolCallResponseChunk {
        ToolCallResponseChunk {
            index,
            id: Some(id.to_string()),
            tp: Some(ToolCallType::Function),
            function: Some(CalledFunctionStream {
                name: Some(name.to_string()),
                arguments: Some(args.to_string()),
            }),
        }
    }

    /// Reference reconstruction of the pre-decoupling unary mapping that lived
    /// inside `dynamo-parsers`. Kept inline so a divergence in the live mapper
    /// fails the test.
    fn legacy_unary(id: &str, name: &str, args: &str) -> ChatCompletionMessageToolCall {
        ChatCompletionMessageToolCall {
            id: id.to_string(),
            r#type: FunctionType::Function,
            function: FunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    /// Reference reconstruction of the pre-decoupling streaming mapping.
    fn legacy_chunk(
        index: u32,
        id: &str,
        name: &str,
        args: &str,
    ) -> ChatCompletionMessageToolCallChunk {
        ChatCompletionMessageToolCallChunk {
            index,
            id: Some(id.to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some(name.to_string()),
                arguments: Some(args.to_string()),
            }),
        }
    }

    #[test]
    fn unary_mapping_matches_legacy_struct_and_json() {
        for (id, name, args) in [
            (
                "call_1",
                "get_weather",
                r#"{"location":"SF","unit":"celsius"}"#,
            ),
            ("call_2", "ping", "{}"), // empty arguments
        ] {
            let mapped = tool_call_response_to_protocol(native_call(id, name, args));
            let legacy = legacy_unary(id, name, args);
            assert_eq!(mapped, legacy, "struct mismatch for {name}");
            assert_eq!(
                serde_json::to_string(&mapped).unwrap(),
                serde_json::to_string(&legacy).unwrap(),
                "serialized JSON mismatch for {name}"
            );
        }
    }

    #[test]
    fn unary_mapping_multi_call_matches_legacy() {
        let inputs = [
            ("a", "first", r#"{"k":"v1"}"#),
            ("b", "second", r#"{"k":"v2"}"#),
        ];
        let mapped: Vec<_> = inputs
            .iter()
            .map(|(id, n, a)| tool_call_response_to_protocol(native_call(id, n, a)))
            .collect();
        let legacy: Vec<_> = inputs
            .iter()
            .map(|(id, n, a)| legacy_unary(id, n, a))
            .collect();
        assert_eq!(mapped, legacy);
        assert_eq!(
            serde_json::to_string(&mapped).unwrap(),
            serde_json::to_string(&legacy).unwrap()
        );
    }

    #[test]
    fn stream_mapping_matches_legacy_struct_and_json() {
        for (idx, id, name, args) in [
            (0u32, "call_1", "get_weather", r#"{"location":"SF"}"#),
            (1u32, "call_2", "ping", "{}"), // empty arguments
        ] {
            let mapped = tool_call_response_chunk_to_protocol(native_chunk(idx, id, name, args));
            let legacy = legacy_chunk(idx, id, name, args);
            assert_eq!(mapped, legacy, "struct mismatch for {name}");
            assert_eq!(
                serde_json::to_string(&mapped).unwrap(),
                serde_json::to_string(&legacy).unwrap(),
                "serialized JSON mismatch for {name}"
            );
        }
    }

    #[test]
    fn stream_mapping_multi_call_indexes_and_matches_legacy() {
        let inputs = [
            (0u32, "a", "first", r#"{"k":"v1"}"#),
            (1u32, "b", "second", r#"{"k":"v2"}"#),
        ];
        let mapped: Vec<_> = inputs
            .iter()
            .map(|(i, id, n, a)| tool_call_response_chunk_to_protocol(native_chunk(*i, id, n, a)))
            .collect();
        let legacy: Vec<_> = inputs
            .iter()
            .map(|(i, id, n, a)| legacy_chunk(*i, id, n, a))
            .collect();
        assert_eq!(mapped, legacy);
        assert_eq!(
            serde_json::to_string(&mapped).unwrap(),
            serde_json::to_string(&legacy).unwrap()
        );
    }

    #[test]
    fn test_validate_messages_rejects_bad_tool_call_arguments() {
        for arguments in ["{invalid json}", "[]", "null", "\"not an object\""] {
            let request_json = json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": arguments
                            }
                        }]
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }]
            });

            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(request_json).expect("Failed to deserialize request");
            let err = ValidateRequest::validate(&request)
                .expect_err("bad tool_call arguments should fail validation");
            let err = err.to_string();
            assert!(
                err.contains("`messages[1].tool_calls[0].function.arguments`"),
                "unexpected error for {arguments:?}: {err}"
            );
            assert!(
                err.contains("valid JSON object string"),
                "unexpected error for {arguments:?}: {err}"
            );
        }
    }

    #[test]
    fn test_validate_messages_accepts_empty_tool_call_arguments() {
        for arguments in ["", " \n\t ", "{}"] {
            let request_json = json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": arguments
                            }
                        }]
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }]
            });

            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(request_json).expect("Failed to deserialize request");
            ValidateRequest::validate(&request)
                .unwrap_or_else(|err| panic!("empty tool_call arguments should validate: {err}"));
        }
    }

    #[test]
    fn test_validate_tools_valid_names() {
        fn make_tool(name: &str) -> ChatCompletionTool {
            ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: name.to_string(),
                    description: None,
                    parameters: Some(json!({"type": "object", "properties": {}})),
                    strict: None,
                },
            }
        }

        let tools = vec![
            make_tool("func_name"),
            make_tool("func-name_v2"),
            make_tool("FuncName"),
            make_tool("Func_Name-123"),
        ];
        assert!(validate::validate_tools(&Some(&tools)).is_ok());
    }

    #[test]
    fn test_validate_tools_invalid_names() {
        for name in ["<func_name>", "func name", "func@name", "func,name", ""] {
            let tools = vec![ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: name.to_string(),
                    description: None,
                    parameters: Some(json!({"type": "object", "properties": {}})),
                    strict: None,
                },
            }];
            assert!(
                validate::validate_tools(&Some(&tools)).is_err(),
                "expected error for name: {name:?}"
            );
        }
    }

    #[test]
    fn test_openai_thinking_payload_normalizes_to_template_args() {
        let json_str = json!({
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "reasoning_effort": "max",
            "thinking": {"type": "enabled"}
        });

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("thinking payload should normalize");

        let args = request
            .chat_template_args
            .as_ref()
            .expect("chat_template_args should be populated");
        assert_eq!(args.get("thinking"), Some(&json!(true)));
        assert_eq!(args.get("thinking_mode"), Some(&json!("enabled")));
        assert_eq!(args.get("reasoning_effort"), Some(&json!("max")));
    }

    #[test]
    fn test_openai_thinking_adaptive_normalizes_to_template_mode() {
        let json_str = json!({
            "model": "MiniMaxAI/MiniMax-M3",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "thinking": {"type": "adaptive"}
        });

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("adaptive thinking payload should normalize");

        let args = request
            .chat_template_args
            .as_ref()
            .expect("chat_template_args should be populated");
        assert_eq!(args.get("thinking_mode"), Some(&json!("adaptive")));
        assert_eq!(args.get("thinking"), None);
        assert!(request.thinking.is_none());
    }

    #[test]
    fn test_openai_thinking_disabled_normalizes_to_template_mode() {
        let json_str = json!({
            "model": "MiniMaxAI/MiniMax-M3",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "thinking": {"type": "disabled"}
        });

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("disabled thinking payload should normalize");

        let args = request
            .chat_template_args
            .as_ref()
            .expect("chat_template_args should be populated");
        assert_eq!(args.get("thinking"), Some(&json!(false)));
        assert_eq!(args.get("thinking_mode"), Some(&json!("disabled")));
    }

    #[test]
    fn test_openai_thinking_top_level_overrides_stale_template_args() {
        let json_str = json!({
            "model": "MiniMaxAI/MiniMax-M3",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "chat_template_args": {
                "thinking": true,
                "thinking_mode": "thinking",
                "reasoning_effort": "high"
            },
            "reasoning_effort": "none",
            "thinking": {"type": "disabled"}
        });

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("top-level thinking payload should normalize");

        let args = request
            .chat_template_args
            .as_ref()
            .expect("chat_template_args should be populated");
        assert_eq!(args.get("thinking"), Some(&json!(false)));
        assert_eq!(args.get("thinking_mode"), Some(&json!("disabled")));
        assert_eq!(args.get("reasoning_effort"), Some(&json!("none")));
        assert!(request.thinking.is_none());
    }

    #[test]
    fn test_invalid_openai_thinking_payload_is_rejected() {
        for invalid_thinking in [
            json!("enabled"),
            json!({"type": "auto"}),
            json!({"type": true}),
            json!({}),
        ] {
            let json_str = json!({
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "thinking": invalid_thinking
            });

            let mut request: NvCreateChatCompletionRequest =
                serde_json::from_value(json_str).expect("Failed to deserialize request");
            assert!(request.normalize_reasoning_template_args().is_err());
        }
    }
}
