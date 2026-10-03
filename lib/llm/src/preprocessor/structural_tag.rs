// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural tag policy for chat tool-call guided decoding.

use crate::local_model::runtime_config::{StructuralTagMode, StructuralTagScope};
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_runtime::error::{DynamoError, ErrorType};

impl OpenAIPreprocessor {
    /// Apply structural tag guided decoding when enabled for this request.
    pub(super) fn apply_tool_choice_structural_tag(
        &self,
        tool_choice: &ToolChoice,
        tools: &[ToolDefinition],
        parallel_tool_calls: Option<bool>,
        prompt_injected_reasoning: bool,
        preprocessed_request: &mut PreprocessedRequest,
    ) -> Result<bool, DynamoError> {
        if self.runtime_config.structural_tag_mode == StructuralTagMode::Off {
            return Ok(false);
        }

        let Some(parser_name) = self.tool_call_parser.as_deref() else {
            tracing::warn!(
                "Structural tag is enabled but --dyn-tool-call-parser is not set; \
                 structural tags will not be applied"
            );
            return Ok(false);
        };

        let Some(builder) = Self::structural_tag_builder_for_parser(parser_name) else {
            return Ok(false);
        };

        if matches!(tool_choice, ToolChoice::None) {
            // Match stock vLLM: explicit `tool_choice=none` does not install a
            // guided-decoding constraint. Prompt formatting independently omits
            // tools when configured to do so.
            return Ok(false);
        }

        if !Self::should_apply_tool_call_format(
            self.runtime_config.structural_tag_scope,
            tool_choice,
            tools,
            parallel_tool_calls,
        ) {
            return Ok(false);
        }

        let ctx = Self::tool_call_format_context(
            &self.runtime_config,
            parser_name,
            tool_choice,
            tools,
            parallel_tool_calls,
            prompt_injected_reasoning,
        );

        Self::apply_tool_call_format(parser_name, builder, &ctx, preprocessed_request)
    }

    /// Build context for the tool-call tag of one request.
    fn tool_call_format_context<'a>(
        runtime_config: &crate::local_model::runtime_config::ModelRuntimeConfig,
        parser_name: &str,
        tool_choice: &'a ToolChoice,
        tools: &'a [ToolDefinition],
        parallel_tool_calls: Option<bool>,
        prompt_injected_reasoning: bool,
    ) -> dynamo_parsers::tool_calling::ToolCallFormatBuildContext<'a> {
        // Nemotron-v3's native vLLM reasoner owns the prompt-seeded reasoning
        // phase and consumes its single `</think>` boundary before advancing
        // guided decoding. Start Qwen3-coder's structural grammar at the tool
        // suffix so the two layers do not both wait for the same boundary.
        let native_reasoning_owns_prefix = prompt_injected_reasoning
            && parser_name == "qwen3_coder"
            && runtime_config.reasoning_parser.as_deref() == Some("nemotron_v3");
        dynamo_parsers::tool_calling::ToolCallFormatBuildContext {
            tool_choice,
            tools,
            parallel_tool_calls: Self::tag_parallel_tool_calls(tool_choice, parallel_tool_calls),
            schema_mode: runtime_config.structural_tag_schema,
            starts_in_reasoning: prompt_injected_reasoning && !native_reasoning_owns_prefix,
        }
    }

    /// `parallel_tool_calls` for the tool-call tag. A named `tool_choice` forces
    /// one call of that tool, as in vLLM; the parser builders allow repeated calls
    /// unless `parallel_tool_calls` is `false`, which let the forced call repeat
    /// until `max_tokens`.
    fn tag_parallel_tool_calls(
        tool_choice: &ToolChoice,
        parallel_tool_calls: Option<bool>,
    ) -> Option<bool> {
        match tool_choice {
            ToolChoice::Named(_) => Some(false),
            _ => parallel_tool_calls,
        }
    }

    /// Find the structural tag builder for a parser, if supported.
    fn structural_tag_builder_for_parser(
        parser_name: &str,
    ) -> Option<&'static dynamo_parsers::tool_calling::StructuralTagBuilder> {
        let parser_map = dynamo_parsers::tool_calling::parsers::get_tool_parser_map();
        let builder = parser_map
            .get(parser_name)
            .and_then(|tc| tc.structural_tag_builder.as_ref());

        if builder.is_none() {
            tracing::warn!(
                parser = parser_name,
                "Structural tag enabled but parser does not support it; \
                 falling back to default behaviour"
            );
        }

        builder
    }

    /// Build and inject the tool-call format tag, if one is needed.
    fn apply_tool_call_format(
        parser_name: &str,
        builder: &dynamo_parsers::tool_calling::StructuralTagBuilder,
        ctx: &dynamo_parsers::tool_calling::ToolCallFormatBuildContext<'_>,
        common_request: &mut PreprocessedRequest,
    ) -> Result<bool, DynamoError> {
        let structural_tag = match builder.build_tool_call_format(ctx) {
            Ok(Some(tag)) => tag,
            Ok(None) => {
                tracing::debug!(
                    parser = parser_name,
                    "Builder returned None for structural_tag (tool_choice={:?})",
                    ctx.tool_choice,
                );
                return Ok(false);
            }
            Err(e) => {
                return Err(DynamoError::builder()
                    .error_type(ErrorType::Unknown)
                    .message(format!(
                        "failed to build structural_tag for parser '{parser_name}': {e}"
                    ))
                    .build());
            }
        };

        let gd = common_request
            .sampling_options
            .guided_decoding
            .get_or_insert_default();
        gd.structural_tag = Some(structural_tag);
        Ok(true)
    }

    /// Decide whether this request should use a tool-call format tag.
    fn should_apply_tool_call_format(
        scope: StructuralTagScope,
        tool_choice: &ToolChoice,
        tools: &[ToolDefinition],
        parallel_tool_calls: Option<bool>,
    ) -> bool {
        match tool_choice {
            ToolChoice::None => false,
            ToolChoice::Required | ToolChoice::Named(_) => true,
            ToolChoice::Auto => match scope {
                StructuralTagScope::Always => true,
                StructuralTagScope::Auto => {
                    let explicit_single_call = parallel_tool_calls == Some(false);
                    tools.iter().any(|t| t.strict.unwrap_or(false)) || explicit_single_call
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tag `apply_tool_choice_structural_tag` installs for qwen3_coder with
    /// Super 3.5's nemotron_v3 reasoning parser.
    fn tag(tool_choice: &ToolChoice, parallel_tool_calls: Option<bool>) -> serde_json::Value {
        let tools = [ToolDefinition {
            name: "record".to_string(),
            parameters: Some(serde_json::json!({"type": "object"})),
            strict: None,
        }];
        let builder = OpenAIPreprocessor::structural_tag_builder_for_parser("qwen3_coder")
            .expect("qwen3_coder has a structural tag builder");
        let runtime_config = crate::local_model::runtime_config::ModelRuntimeConfig {
            reasoning_parser: Some("nemotron_v3".to_string()),
            ..Default::default()
        };
        let ctx = OpenAIPreprocessor::tool_call_format_context(
            &runtime_config,
            "qwen3_coder",
            tool_choice,
            &tools,
            parallel_tool_calls,
            true,
        );
        builder
            .build_tool_call_format(&ctx)
            .expect("tag builds")
            .expect("tag is needed")
    }

    #[test]
    fn named_tool_choice_allows_one_call() {
        let named = ToolChoice::Named("record".to_string());
        for parallel_tool_calls in [None, Some(true), Some(false)] {
            let value = tag(&named, parallel_tool_calls);
            assert_eq!(
                value["format"]["stop_after_first"],
                serde_json::json!(true),
                "{parallel_tool_calls:?}: {value}"
            );
        }
    }

    #[test]
    fn required_tool_choice_keeps_parallel_tool_calls() {
        let value = tag(&ToolChoice::Required, None);
        assert_eq!(
            value["format"]["stop_after_first"],
            serde_json::json!(false)
        );
        let value = tag(&ToolChoice::Required, Some(false));
        assert_eq!(value["format"]["stop_after_first"], serde_json::json!(true));
    }
}
