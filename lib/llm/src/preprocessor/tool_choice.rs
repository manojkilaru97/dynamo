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

fn clear_legacy_json_constraint(common_request: &mut PreprocessedRequest) -> bool {
    if let Some(guided_decoding) = common_request.sampling_options.guided_decoding.as_mut()
        && guided_decoding.structural_tag.is_some()
    {
        // Request conversion may already have installed the legacy forced-tool
        // JSON schema. vLLM accepts only one structured-output constraint.
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

        if self.apply_tool_choice_structural_tag(
            &convert_tool_choice(tool_choice),
            &convert_tools(tools),
            request.inner.parallel_tool_calls,
            prompt_injected_reasoning,
            common_request,
        )? {
            let cleared = clear_legacy_json_constraint(common_request);
            debug_assert!(cleared);
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
    use crate::protocols::common::{
        GuidedDecodingOptions, OutputOptions, SamplingOptions, StopConditions,
    };

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
    fn structural_tag_replaces_legacy_json_constraint() {
        let structural_tag = serde_json::json!({
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "tags": [],
                "triggers": ["<tool_call>\n<function="],
                "at_least_one": true,
                "stop_after_first": true,
            }
        });
        let mut request = request_with_guided_decoding(GuidedDecodingOptions::new(
            Some(serde_json::json!({"type": "object"})),
            None,
            None,
            None,
            None,
            None,
            Some(structural_tag.clone()),
        ));

        assert!(clear_legacy_json_constraint(&mut request));
        let guided = request
            .sampling_options
            .guided_decoding
            .expect("guided decoding remains configured");
        assert_eq!(guided.json, None);
        assert_eq!(guided.structural_tag, Some(structural_tag));
    }

    #[test]
    fn legacy_json_is_unchanged_without_structural_tag() {
        let legacy_json = serde_json::json!({"type": "object"});
        let mut request = request_with_guided_decoding(GuidedDecodingOptions::new(
            Some(legacy_json.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        ));

        assert!(!clear_legacy_json_constraint(&mut request));
        assert_eq!(
            request
                .sampling_options
                .guided_decoding
                .expect("guided decoding remains configured")
                .json,
            Some(legacy_json)
        );
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
