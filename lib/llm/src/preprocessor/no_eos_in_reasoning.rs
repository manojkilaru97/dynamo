// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Super 3.5 opt-in for vLLM's "no end-of-turn while reasoning is open" mask.
//!
//! Super 3.5 occasionally samples `<|im_end|>` inside open reasoning, which ends a
//! tool-using turn with no content and no tool call (Bug 5, `reasoning_stop_empty`).
//! For thinking requests that may call tools, the frontend asks the vLLM worker to
//! mask stop tokens until the model has generated `</think>` or `<tool_call>`.
//! `DYN_SUPER35_BLOCK_EOS_IN_REASONING=0` turns this off without a new image.

use std::sync::LazyLock;

use dynamo_protocols::types::ChatCompletionToolChoiceOption;
use dynamo_runtime::config::is_falsey;

use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

/// `PreprocessedRequest.extra_args` key, copied by the vLLM worker into
/// `SamplingParams.extra_args`.
pub(crate) const EXTRA_ARG: &str = "nemotron_block_eos_in_reasoning";
const ENV: &str = "DYN_SUPER35_BLOCK_EOS_IN_REASONING";

static ENV_ENABLED: LazyLock<bool> =
    LazyLock::new(|| env_value_enabled(std::env::var(ENV).ok().as_deref()));

/// Default on; only an explicit false value (`0`, `false`, `off`, `no`) disables it.
fn env_value_enabled(value: Option<&str>) -> bool {
    !value.is_some_and(|value| !value.trim().is_empty() && is_falsey(value))
}

/// Whether a Super 3.5 request opts into the mask: generation starts inside
/// reasoning opened by the prompt, thinking is parsed, and the model may call a tool.
fn applies(
    request: &NvCreateChatCompletionRequest,
    thinking: bool,
    prompt_injected_reasoning: bool,
) -> bool {
    thinking
        && prompt_injected_reasoning
        && request.inner.tools.as_ref().is_some_and(|tools| !tools.is_empty())
        && !matches!(
            request.inner.tool_choice,
            Some(ChatCompletionToolChoiceOption::None)
        )
}

/// Add the opt-in to `extra_args` when it applies. Only called for the Super 3.5
/// user-data profile.
pub(crate) fn apply(
    request: &NvCreateChatCompletionRequest,
    thinking: bool,
    prompt_injected_reasoning: bool,
    extra_args: &mut Option<serde_json::Value>,
) -> anyhow::Result<bool> {
    apply_with(*ENV_ENABLED, request, thinking, prompt_injected_reasoning, extra_args)
}

fn apply_with(
    enabled: bool,
    request: &NvCreateChatCompletionRequest,
    thinking: bool,
    prompt_injected_reasoning: bool,
    extra_args: &mut Option<serde_json::Value>,
) -> anyhow::Result<bool> {
    if !enabled || !applies(request, thinking, prompt_injected_reasoning) {
        return Ok(false);
    }
    let args = extra_args.get_or_insert_with(|| serde_json::json!({}));
    let Some(args) = args.as_object_mut() else {
        anyhow::bail!("preprocessed extra_args must be an object");
    };
    args.insert(EXTRA_ARG.to_string(), serde_json::Value::Bool(true));
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: serde_json::Value) -> NvCreateChatCompletionRequest {
        serde_json::from_value(body).expect("valid chat request")
    }

    fn with_tools(tool_choice: Option<serde_json::Value>) -> NvCreateChatCompletionRequest {
        let mut body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        });
        if let Some(choice) = tool_choice {
            body["tool_choice"] = choice;
        }
        request(body)
    }

    fn run(
        enabled: bool,
        request: &NvCreateChatCompletionRequest,
        thinking: bool,
        prompt_injected_reasoning: bool,
        mut extra_args: Option<serde_json::Value>,
    ) -> Option<serde_json::Value> {
        apply_with(enabled, request, thinking, prompt_injected_reasoning, &mut extra_args).unwrap();
        extra_args
    }

    #[test]
    fn tool_requests_in_open_reasoning_opt_in() {
        for choice in [None, Some(serde_json::json!("auto")), Some(serde_json::json!("required")),
            Some(serde_json::json!({"type": "function", "function": {"name": "f"}}))]
        {
            let req = with_tools(choice.clone());
            assert_eq!(run(true, &req, true, true, None), Some(serde_json::json!({EXTRA_ARG: true})), "{choice:?}");
        }
    }

    #[test]
    fn existing_extra_args_are_preserved() {
        let req = with_tools(None);
        let out = run(true, &req, true, true, Some(serde_json::json!({"nvext": {"a": 1}})));
        assert_eq!(out, Some(serde_json::json!({"nvext": {"a": 1}, EXTRA_ARG: true})));
        let mut bad = Some(serde_json::json!([1]));
        assert!(apply_with(true, &req, true, true, &mut bad).is_err());
    }

    #[test]
    fn other_requests_are_untouched() {
        let tools = with_tools(None);
        let none = with_tools(Some(serde_json::json!("none")));
        let text_only = request(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
        }));
        let empty_tools = request(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [],
        }));
        let original = Some(serde_json::json!({"nvext": {"a": 1}}));
        for (req, thinking, injected) in [
            (&tools, false, true),
            (&tools, true, false),
            (&none, true, true),
            (&text_only, true, true),
            (&empty_tools, true, true),
        ] {
            assert_eq!(run(true, req, thinking, injected, None), None);
            assert_eq!(run(true, req, thinking, injected, original.clone()), original);
        }
        assert_eq!(run(false, &tools, true, true, None), None);
    }

    #[test]
    fn env_kill_switch() {
        assert!(env_value_enabled(None));
        assert!(env_value_enabled(Some("")));
        assert!(env_value_enabled(Some("1")));
        assert!(env_value_enabled(Some("true")));
        for off in ["0", "false", "OFF", " no "] {
            assert!(!env_value_enabled(Some(off)), "{off}");
        }
    }
}
