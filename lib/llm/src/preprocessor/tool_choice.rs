// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-choice guided decoding policy for OpenAI chat requests.

use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::openai::tools::get_json_schema_from_tools;

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolChoiceOption, ResponseFormat};
use dynamo_runtime::error::{DynamoError, ErrorType};

fn invalid_argument(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message)
        .build()
}

fn prefer_structural_tag_over_legacy_json(common_request: &mut PreprocessedRequest) -> bool {
    if let Some(guided_decoding) = common_request.sampling_options.guided_decoding.as_mut()
        && guided_decoding.structural_tag.is_some()
    {
        // Request conversion may already have installed the legacy forced-tool
        // JSON schema. A backend accepts only one structured-output constraint,
        // and the native structural tag is the preferred tool-call constraint.
        guided_decoding.json = None;
        return true;
    }

    false
}

impl OpenAIPreprocessor {
    /// Apply guided decoding for OpenAI tool-choice requests.
    ///
    /// Structural tags are preferred when enabled and supported by the configured
    /// tool-call parser. Forced tool-choice requests fall back to the legacy
    /// JSON-schema constraint when structural tags are not applied.
    pub(super) fn apply_tool_choice_guided_decoding(
        &self,
        request: &NvCreateChatCompletionRequest,
        common_request: &mut PreprocessedRequest,
        prompt_injected_reasoning: bool,
    ) -> Result<bool, DynamoError> {
        let tool_choice = request
            .inner
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
        let tools = request.inner.tools.as_deref().unwrap_or(&[]);
        let is_forced_tool_choice = matches!(
            tool_choice,
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
        );
        let has_explicit_guided_decoding = has_explicit_guided_decoding(request);
        let has_response_format_constraint = has_response_format_constraint(request);

        if is_forced_tool_choice && has_explicit_guided_decoding {
            return Err(invalid_argument(concat!(
                "guided decoding cannot be used in the same request as ",
                "tool_choice=\"required\" or a named tool_choice.",
            )));
        }

        // For non-forced tool choice, explicit guided decoding and response_format
        // constrain assistant content, so tool-choice guided decoding stays inactive.
        let has_assistant_constraint =
            has_explicit_guided_decoding || has_response_format_constraint;
        if !is_forced_tool_choice && has_assistant_constraint {
            return Ok(false);
        }

        if is_forced_tool_choice
            && has_response_format_constraint
            && let Some(gd) = common_request.sampling_options.guided_decoding.as_mut()
        {
            // OpenAI `response_format` applies to assistant content, not tool calls.
            gd.json = None;
        }

        // Request conversion can already provide a model-specific structural
        // tag. Preserve that normalized constraint instead of replacing it
        // with the generic parser builder's format.
        if is_forced_tool_choice && prefer_structural_tag_over_legacy_json(common_request) {
            return Ok(true);
        }

        if self.apply_tool_choice_structural_tag(
            &convert_tool_choice(tool_choice),
            &convert_tools(tools),
            request.inner.parallel_tool_calls,
            prompt_injected_reasoning,
            common_request,
        )? {
            let removed_conflict = prefer_structural_tag_over_legacy_json(common_request);
            debug_assert!(removed_conflict);
            return Ok(true);
        }

        match get_json_schema_from_tools(Some(tool_choice), Some(tools), request.request_text_len())
        {
            Ok(Some(schema)) => {
                let gd = common_request
                    .sampling_options
                    .guided_decoding
                    .get_or_insert_default();
                gd.json = Some(schema);
            }
            Ok(None) => {}
            Err(err) => {
                return Err(invalid_argument(err.to_string()));
            }
        }

        // Auto/None requests can reach here when neither structural tags nor a
        // tool-choice JSON fallback were needed.
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_card::ModelDeploymentCard;
    use crate::protocols::common::{
        GuidedDecodingOptions, OutputOptions, SamplingOptions, StopConditions,
    };

    const TEST_MODEL_PATH: &str = "tests/data/sample-models/mock-llama-3.1-8b-instruct";

    fn test_preprocessor() -> std::sync::Arc<OpenAIPreprocessor> {
        let mut mdc = ModelDeploymentCard::load_from_disk(TEST_MODEL_PATH, None)
            .expect("load test model deployment card");
        mdc.set_name("test-model");
        OpenAIPreprocessor::new(mdc).expect("construct test preprocessor")
    }

    fn tool_request(tool_choice: serde_json::Value) -> NvCreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Call get_weather for Paris"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": false
                    }
                }
            }],
            "tool_choice": tool_choice
        }))
        .expect("valid chat request")
    }

    fn forced_single_tool_structural_tag() -> serde_json::Value {
        serde_json::json!({
            "type": "structural_tag",
            "format": {
                "type": "tags_with_separator",
                "tags": [],
                "triggers": [""],
                "separator": "",
                "at_least_one": true,
                "stop_after_first": true,
            }
        })
    }

    fn request_with_guided_decoding(guided_decoding: GuidedDecodingOptions) -> PreprocessedRequest {
        let mut builder = PreprocessedRequest::builder();
        builder
            .model("test-model".to_string())
            .token_ids(vec![1])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions {
                guided_decoding: Some(guided_decoding),
                ..Default::default()
            })
            .output_options(OutputOptions::default());
        builder.build().expect("valid preprocessed request")
    }

    #[test]
    fn existing_forced_single_tool_tag_is_preserved_over_legacy_json() {
        let structural_tag = forced_single_tool_structural_tag();
        let mut request = request_with_guided_decoding(GuidedDecodingOptions::new(
            Some(serde_json::json!({"type": "object"})),
            None,
            None,
            None,
            None,
            None,
            Some(structural_tag.clone()),
        ));

        assert!(prefer_structural_tag_over_legacy_json(&mut request));

        let guided_decoding = request
            .sampling_options
            .guided_decoding
            .expect("guided decoding remains configured");
        assert_eq!(guided_decoding.json, None);
        assert_eq!(guided_decoding.structural_tag, Some(structural_tag));
        let format = guided_decoding
            .structural_tag
            .as_ref()
            .and_then(|tag| tag.get("format"))
            .expect("preserved structural-tag format");
        assert_eq!(format["type"], "tags_with_separator");
        assert_eq!(format["stop_after_first"], true);
    }

    #[test]
    fn guided_decoding_preserves_existing_forced_qwen_single_tool_tag() {
        let structural_tag = forced_single_tool_structural_tag();
        let mut common_request = request_with_guided_decoding(GuidedDecodingOptions::new(
            Some(serde_json::json!({"type": "object"})),
            None,
            None,
            None,
            None,
            None,
            Some(structural_tag.clone()),
        ));
        let request = tool_request(serde_json::json!({
            "type": "function",
            "function": {"name": "get_weather"}
        }));

        let uses_tool_call_structural_tag = test_preprocessor()
            .apply_tool_choice_guided_decoding(&request, &mut common_request, true)
            .expect("apply forced tool-choice guidance");

        assert!(uses_tool_call_structural_tag);
        let guided_decoding = common_request
            .sampling_options
            .guided_decoding
            .expect("guided decoding remains configured");
        assert_eq!(guided_decoding.json, None);
        assert_eq!(guided_decoding.structural_tag, Some(structural_tag));
    }

    #[test]
    fn guided_decoding_does_not_mislabel_existing_auto_structural_tag() {
        let structural_tag = serde_json::json!({
            "type": "structural_tag",
            "format": {"type": "tag", "begin": "<custom>", "content": {"type": "any_text"}, "end": "</custom>"}
        });
        let mut common_request = request_with_guided_decoding(GuidedDecodingOptions::new(
            None,
            None,
            None,
            None,
            None,
            None,
            Some(structural_tag.clone()),
        ));
        let request = tool_request(serde_json::json!("auto"));

        let uses_tool_call_structural_tag = test_preprocessor()
            .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
            .expect("apply automatic tool-choice guidance");

        assert!(!uses_tool_call_structural_tag);
        let guided_decoding = common_request
            .sampling_options
            .guided_decoding
            .expect("guided decoding remains configured");
        assert_eq!(guided_decoding.structural_tag, Some(structural_tag));
    }

    #[test]
    fn legacy_json_constraint_is_preserved_without_structural_tag() {
        let guided_json = serde_json::json!({"type": "object"});
        let mut request = request_with_guided_decoding(GuidedDecodingOptions::new(
            Some(guided_json.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        ));

        assert!(!prefer_structural_tag_over_legacy_json(&mut request));

        let guided_decoding = request
            .sampling_options
            .guided_decoding
            .expect("guided decoding remains configured");
        assert_eq!(guided_decoding.json, Some(guided_json));
        assert_eq!(guided_decoding.structural_tag, None);
    }
}

fn has_explicit_guided_decoding(request: &NvCreateChatCompletionRequest) -> bool {
    request.common.guided_json.is_some()
        || request.common.guided_regex.is_some()
        || request
            .common
            .guided_choice
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        || request.common.guided_grammar.is_some()
}

fn has_response_format_constraint(request: &NvCreateChatCompletionRequest) -> bool {
    request
        .inner
        .response_format
        .as_ref()
        .is_some_and(|format| !matches!(format, ResponseFormat::Text))
}

fn convert_tool_choice(tool_choice: &ChatCompletionToolChoiceOption) -> ToolChoice {
    match tool_choice {
        ChatCompletionToolChoiceOption::None => ToolChoice::None,
        ChatCompletionToolChoiceOption::Auto => ToolChoice::Auto,
        ChatCompletionToolChoiceOption::Required => ToolChoice::Required,
        ChatCompletionToolChoiceOption::Named(named) => {
            ToolChoice::Named(named.function.name.clone())
        }
    }
}

fn convert_tools(tools: &[ChatCompletionTool]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.function.name.clone(),
            parameters: tool.function.parameters.clone(),
            strict: tool.function.strict,
        })
        .collect()
}
