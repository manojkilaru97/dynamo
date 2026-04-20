# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

from vllm.sampling_params import StructuredOutputsParams

import dynamo.vllm.main as vllm_main_module
from dynamo.frontend.vllm_processor import (
    _maybe_wrap_guided_decoding_for_minimax_reasoning,
    _request_needs_cache_isolation,
    _structured_outputs_to_guided_decoding,
    _tool_choice_to_guided_decoding,
    _vllm_reasoning_parser_name,
)
from dynamo.vllm.handlers import build_sampling_params
from dynamo.vllm.main import setup_vllm_engine


def test_structured_outputs_bridge_preserves_guided_fields():
    structured_outputs = StructuredOutputsParams(
        json={"type": "object", "properties": {"x": {"type": "string"}}},
        disable_fallback=True,
        disable_any_whitespace=True,
        disable_additional_properties=True,
        whitespace_pattern=r"\s*",
    )

    guided = _structured_outputs_to_guided_decoding(structured_outputs)

    assert guided == {
        "json": {"type": "object", "properties": {"x": {"type": "string"}}},
        "disable_fallback": True,
        "disable_any_whitespace": True,
        "disable_additional_properties": True,
        "whitespace_pattern": r"\s*",
    }


def test_build_sampling_params_restores_full_guided_decoding_payload():
    request = {
        "sampling_options": {
            "guided_decoding": {
                "json": {"type": "object"},
                "json_object": False,
                "disable_fallback": True,
                "disable_any_whitespace": True,
                "disable_additional_properties": True,
                "whitespace_pattern": r"\s*",
            }
        },
        "stop_conditions": {},
        "output_options": {},
    }

    sampling_params = build_sampling_params(request, default_sampling_params={})

    assert sampling_params.structured_outputs is not None
    assert sampling_params.structured_outputs.json == {"type": "object"}
    assert sampling_params.structured_outputs.json_object is None
    assert sampling_params.structured_outputs.disable_fallback is True
    assert sampling_params.structured_outputs.disable_any_whitespace is True
    assert sampling_params.structured_outputs.disable_additional_properties is True
    assert sampling_params.structured_outputs.whitespace_pattern == r"\s*"
    assert sampling_params.skip_reading_prefix_cache is True


def test_build_sampling_params_prefers_lmfe_for_regex_guidance():
    request = {
        "sampling_options": {
            "guided_decoding": {
                "regex": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            }
        },
        "stop_conditions": {},
        "output_options": {},
    }

    sampling_params = build_sampling_params(request, default_sampling_params={})

    assert sampling_params.structured_outputs is not None
    assert sampling_params.structured_outputs.regex == (
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    )
    assert sampling_params.structured_outputs._backend == "lm-format-enforcer"
    assert sampling_params.structured_outputs._backend_was_auto is True
    assert sampling_params.skip_reading_prefix_cache is True


def test_build_sampling_params_prefers_lmfe_for_minimax_forced_tool_json():
    request = {
        "model": "minimaxai/minimax-m2.7",
        "sampling_options": {
            "guided_decoding": {
                "json": {
                    "type": "object",
                    "properties": {
                        "japanese": {"type": "string"},
                        "arabic": {"type": "string"},
                    },
                    "required": ["japanese", "arabic"],
                    "additionalProperties": False,
                }
            }
        },
        "request_for_sampling": {"tool_choice": "required"},
        "stop_conditions": {},
        "output_options": {},
    }

    sampling_params = build_sampling_params(request, default_sampling_params={})

    assert sampling_params.structured_outputs is not None
    assert sampling_params.structured_outputs.json is not None
    assert sampling_params.structured_outputs._backend == "lm-format-enforcer"
    assert sampling_params.structured_outputs._backend_was_auto is True
    assert sampling_params.skip_reading_prefix_cache is True


def test_build_sampling_params_prefers_guidance_for_minimax_grammar():
    request = {
        "model": "minimaxai/minimax-m2.7",
        "sampling_options": {
            "guided_decoding": {
                "grammar": 'root ::= "yes" | "no"',
            }
        },
        "stop_conditions": {},
        "output_options": {},
    }

    sampling_params = build_sampling_params(request, default_sampling_params={})

    assert sampling_params.structured_outputs is not None
    assert sampling_params.structured_outputs.grammar == 'root ::= "yes" | "no"'
    assert sampling_params.structured_outputs._backend == "guidance"
    assert sampling_params.structured_outputs._backend_was_auto is True
    assert sampling_params.skip_reading_prefix_cache is True


def test_build_sampling_params_restores_structural_tag_guided_decoding():
    request = {
        "sampling_options": {
            "guided_decoding": {
                "structural_tag": '{"type":"sequence","elements":[{"type":"tag","begin":"","content":{"type":"any_text"},"end":"</think>"},{"type":"json_schema","json_schema":{"type":"object"}}]}'
            }
        },
        "stop_conditions": {},
        "output_options": {},
    }

    sampling_params = build_sampling_params(request, default_sampling_params={})

    assert sampling_params.structured_outputs is not None
    assert sampling_params.structured_outputs.structural_tag == (
        '{"type":"sequence","elements":[{"type":"tag","begin":"","content":{"type":"any_text"},"end":"</think>"},{"type":"json_schema","json_schema":{"type":"object"}}]}'
    )
    assert sampling_params.structured_outputs._backend == "xgrammar"
    assert sampling_params.structured_outputs._backend_was_auto is True
    assert sampling_params.skip_reading_prefix_cache is True


def test_build_sampling_params_prefers_guidance_for_minimax_structural_tag_json_schema():
    request = {
        "model": "minimaxai/minimax-m2.7",
        "sampling_options": {
            "guided_decoding": {
                "structural_tag": '{"type":"sequence","elements":[{"type":"tag","begin":"","content":{"type":"any_text"},"end":"</think>"},{"type":"json_schema","json_schema":{"type":"object"}}]}',
                "_dynamo_structural_content_type": "json_schema",
            }
        },
        "stop_conditions": {},
        "output_options": {},
    }

    sampling_params = build_sampling_params(request, default_sampling_params={})

    assert sampling_params.structured_outputs is not None
    assert sampling_params.structured_outputs._backend == "guidance"
    assert sampling_params.structured_outputs._backend_was_auto is True
    assert sampling_params.skip_reading_prefix_cache is True


def test_build_sampling_params_prefers_xgrammar_for_minimax_structural_tag_regex():
    request = {
        "model": "minimaxai/minimax-m2.7",
        "sampling_options": {
            "guided_decoding": {
                "structural_tag": '{"type":"sequence","elements":[{"type":"tag","begin":"","content":{"type":"any_text"},"end":"</think>"},{"type":"regex","pattern":"yes|no"}]}',
                "_dynamo_structural_content_type": "regex",
            }
        },
        "stop_conditions": {},
        "output_options": {},
    }

    sampling_params = build_sampling_params(request, default_sampling_params={})

    assert sampling_params.structured_outputs is not None
    assert sampling_params.structured_outputs._backend == "xgrammar"
    assert sampling_params.structured_outputs._backend_was_auto is True
    assert sampling_params.skip_reading_prefix_cache is True


def test_request_needs_cache_isolation_for_forced_tool_choice():
    request = {"tool_choice": {"type": "function", "function": {"name": "calculate"}}}
    request_for_sampling = SimpleNamespace(tool_choice=None, structured_outputs=None)

    assert _request_needs_cache_isolation(request, request_for_sampling) is True


def test_request_needs_cache_isolation_for_structured_outputs():
    request: dict[str, Any] = {}
    request_for_sampling = SimpleNamespace(
        tool_choice=None,
        structured_outputs=StructuredOutputsParams(choice=["yes", "no"]),
    )

    assert _request_needs_cache_isolation(request, request_for_sampling) is True


def test_request_needs_cache_isolation_skips_plain_requests():
    request: dict[str, Any] = {}
    request_for_sampling = SimpleNamespace(tool_choice="auto", structured_outputs=None)

    assert _request_needs_cache_isolation(request, request_for_sampling) is False


def test_tool_choice_to_guided_decoding_for_named_function():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                    },
                    "required": ["expression"],
                },
            },
        }
    ]

    guided = _tool_choice_to_guided_decoding(
        {"type": "function", "function": {"name": "calculate"}},
        tools,
    )

    assert guided == {
        "json": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        }
    }


def test_tool_choice_to_guided_decoding_for_required_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        },
    ]

    guided = _tool_choice_to_guided_decoding("required", tools)

    assert guided is not None
    assert guided["json"]["type"] == "array"
    assert guided["json"]["minItems"] == 1


def test_minimax_guided_decoding_wraps_structured_json_with_reasoning_handoff():
    guided = _maybe_wrap_guided_decoding_for_minimax_reasoning(
        {"json": {"type": "object", "properties": {"x": {"type": "string"}}}},
        type("MiniMaxM2AppendThinkReasoningParser", (), {}),
        source="structured_outputs",
    )

    assert guided is not None
    structural_tag = json.loads(guided["structural_tag"])
    assert structural_tag["elements"][0] == {
        "type": "tag",
        "begin": "",
        "content": {"type": "any_text"},
        "end": "</think>",
    }
    assert structural_tag["elements"][1] == {
        "type": "json_schema",
        "json_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
    }
    assert guided.get("json") is None


def test_minimax_guided_decoding_wraps_json_object_with_reasoning_handoff():
    guided = _maybe_wrap_guided_decoding_for_minimax_reasoning(
        {"json_object": True, "disable_any_whitespace": True},
        type("MiniMaxM2AppendThinkReasoningParser", (), {}),
        source="response_format",
    )

    assert guided is not None
    structural_tag = json.loads(guided["structural_tag"])
    assert structural_tag["elements"][0]["end"] == "</think>"
    assert structural_tag["elements"][1] == {
        "type": "json_schema",
        "json_schema": {"type": "object"},
    }
    assert guided["disable_any_whitespace"] is True
    assert guided.get("json_object") is None


def test_minimax_guided_decoding_wraps_choice_with_reasoning_handoff():
    guided = _maybe_wrap_guided_decoding_for_minimax_reasoning(
        {"choice": ["yes", "no"], "disable_any_whitespace": True},
        type("MiniMaxM2AppendThinkReasoningParser", (), {}),
        source="structured_outputs",
    )

    assert guided is not None
    structural_tag = json.loads(guided["structural_tag"])
    assert structural_tag["elements"][0]["end"] == "</think>"
    assert structural_tag["elements"][1] == {
        "type": "regex",
        "pattern": "yes|no",
    }
    assert guided["disable_any_whitespace"] is True
    assert guided.get("choice") is None


def test_minimax_guided_decoding_wraps_tool_choice_json_with_reasoning_handoff():
    guided = _maybe_wrap_guided_decoding_for_minimax_reasoning(
        {"json": {"type": "object", "properties": {"expression": {"type": "string"}}}},
        type("MiniMaxM2AppendThinkReasoningParser", (), {}),
        source="tool_choice",
    )

    assert guided is not None
    structural_tag = json.loads(guided["structural_tag"])
    assert structural_tag["elements"][0]["end"] == "</think>"
    assert structural_tag["elements"][1] == {
        "type": "json_schema",
        "json_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
        },
    }


def test_setup_vllm_engine_propagates_dyn_reasoning_parser(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeMetrics:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def set_model_load_time(self, value):
            captured["load_time"] = value

    def _fake_from_vllm_config(**kwargs):
        captured["vllm_config"] = kwargs["vllm_config"]
        return SimpleNamespace()

    monkeypatch.setattr(vllm_main_module, "setup_multiprocess_prometheus", lambda: None)
    monkeypatch.setattr(vllm_main_module, "LLMBackendMetrics", _FakeMetrics)
    monkeypatch.setattr(
        vllm_main_module.AsyncLLM,
        "from_vllm_config",
        staticmethod(_fake_from_vllm_config),
    )

    engine_args = SimpleNamespace(
        enable_lora=False,
        load_format="dummy",
        create_model_config=lambda: SimpleNamespace(
            get_diff_sampling_param=lambda: {}
        ),
        create_engine_config=lambda usage_context=None: SimpleNamespace(
            structured_outputs_config=SimpleNamespace(reasoning_parser=""),
            additional_config={},
        ),
        enable_log_requests=False,
        disable_log_stats=False,
    )
    config = SimpleNamespace(
        engine_args=engine_args,
        route_to_encoder=False,
        multimodal_embedding_cache_capacity_gb=0,
        dyn_reasoning_parser="minimax_append_think",
        served_model_name="minimaxai/minimax-m2.7",
        component="backend",
        namespace="dynamo",
        model="minimaxai/minimax-m2.7",
    )

    _, vllm_config, *_ = setup_vllm_engine(cast(Any, config))

    assert (
        vllm_config.structured_outputs_config.reasoning_parser
        == "minimax_m2_append_think"
    )
    assert (
        cast(Any, captured["vllm_config"]).structured_outputs_config.reasoning_parser
        == "minimax_m2_append_think"
    )


def test_vllm_reasoning_parser_name_maps_minimax_alias():
    assert _vllm_reasoning_parser_name("minimax_append_think") == (
        "minimax_m2_append_think"
    )
    assert _vllm_reasoning_parser_name("qwen3") == "qwen3"
    assert _vllm_reasoning_parser_name(None) is None
