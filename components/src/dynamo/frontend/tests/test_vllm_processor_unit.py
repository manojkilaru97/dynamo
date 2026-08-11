#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM processor components.

Tests for the tool-stripping behaviour of _prepare_request when
tool_choice='none' and the exclude_tools_when_tool_choice_none flag.
"""

import asyncio
import importlib.util
import json
from types import SimpleNamespace

import pytest
from _routed_engine_fakes import FakeRoutedEngine as _FakeRoutedEngine
from transformers import AutoTokenizer
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from dynamo.frontend.prepost import StreamingPostProcessor, _prepare_request
from dynamo.frontend.vllm_processor import (
    _wire_chat_logprobs_content,
    _with_parser_visible_engine_text,
)

# NOTE: dynamo.frontend.vllm_processor is imported lazily inside the tests that
# need it. Importing it at module top level would run its `from vllm.tasks import ...`
# / `from vllm.v1.engine.parallel_sampling import ...` imports during pytest
# collection, which breaks the pytest-marker-report pre-commit hook.

HAS_QWEN3_TOOL_PARSER = (
    importlib.util.find_spec("vllm.tool_parsers.qwen3_engine_tool_parser") is not None
    or importlib.util.find_spec("vllm.tool_parsers.qwen3coder_tool_parser") is not None
)


def _resolve_qwen3_tool_parser_class():
    try:
        from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser

        return Qwen3EngineToolParser
    except ImportError:
        from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

        return Qwen3CoderToolParser


# Needs vllm packages (gpu_1 container), but does not allocate GPU VRAM.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.xpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
    pytest.mark.skipif(
        not HAS_QWEN3_TOOL_PARSER,
        reason="requires vllm qwen3 tool parser",
    ),
]

MODEL = "Qwen/Qwen3-0.6B"

TOOL_REQUEST = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Hello"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ],
}


def test_parser_visible_engine_text_overrides_trimmed_delta():
    output = SimpleNamespace(
        index=0,
        text="",
        token_ids=[13],
        finish_reason=None,
        logprobs=None,
    )
    visible = _with_parser_visible_engine_text(
        output, "</think>", parser_needs_raw_delta=True
    )
    assert visible.text == "</think>"


def test_parser_visible_engine_text_preserves_utf8_buffering():
    output = SimpleNamespace(
        index=0,
        text="",
        token_ids=[1226],
        finish_reason=None,
        logprobs=None,
    )
    visible = _with_parser_visible_engine_text(
        output, "�", parser_needs_raw_delta=False
    )
    assert visible is output
    assert visible.text == ""


def test_wire_chat_logprobs_content_uses_selected_and_top_tokens():
    tokenizer = SimpleNamespace(
        decode=lambda token_ids, **_: {7: "yes", 8: "no"}[token_ids[0]]
    )
    content = _wire_chat_logprobs_content(
        {
            "token_ids": [7],
            "log_probs": [-0.25],
            "top_logprobs": [
                [
                    {
                        "token_id": 7,
                        "token": "yes",
                        "logprob": -0.25,
                        "bytes": [121, 101, 115],
                    },
                    {"token_id": 8, "token": "no", "logprob": -1.5},
                ]
            ],
        },
        tokenizer,
    )

    assert content == [
        {
            "token": "yes",
            "logprob": -0.25,
            "bytes": [121, 101, 115],
            "top_logprobs": [
                {
                    "token": "yes",
                    "logprob": -0.25,
                    "bytes": [121, 101, 115],
                },
                {"token": "no", "logprob": -1.5, "bytes": [110, 111]},
            ],
        }
    ]


def test_wire_chat_logprobs_content_rejects_misaligned_payload():
    tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "unused")
    assert (
        _wire_chat_logprobs_content(
            {"token_ids": [7, 8], "log_probs": [-0.25]}, tokenizer
        )
        is None
    )


def test_reasoning_parser_plugin_is_loaded_before_lookup(monkeypatch):
    from dynamo.frontend import vllm_processor

    events = []
    plugin_path = "/models/custom_reasoning_parser.py"
    parser_class = object()

    monkeypatch.setattr(
        vllm_processor.ReasoningParserManager,
        "import_reasoning_parser",
        lambda path: events.append(("import", path)),
    )
    monkeypatch.setattr(
        vllm_processor.ReasoningParserManager,
        "get_reasoning_parser",
        lambda name: events.append(("get", name)) or parser_class,
    )

    flags = SimpleNamespace(
        reasoning_parser="custom",
        reasoning_parser_plugin=plugin_path,
    )
    resolved = vllm_processor._resolve_reasoning_parser(flags, {})

    assert resolved is parser_class
    assert events == [("import", plugin_path), ("get", "custom")]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


# ---------------------------------------------------------------------------
# _prepare_request: tool_choice=none tool-stripping
# ---------------------------------------------------------------------------


class TestPrepareRequestToolStripping:  # FRONTEND.1 + FRONTEND.3 — tool stripping when tool_choice=none on chat-template input
    """Test that _prepare_request strips/keeps tools based on the flag."""

    def test_tool_choice_none_strips_tools_from_template(self, tokenizer):
        """When exclude flag is on and tool_choice=none, tools are excluded from template kwargs."""
        _, _, _, _, chat_params = _prepare_request(
            {**TOOL_REQUEST, "tool_choice": "none"},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )
        assert (
            chat_params.chat_template_kwargs["tools"] is None
        ), "tool_choice=none with exclude flag should strip tools from template"

    def test_tool_choice_none_keeps_tools_when_flag_off(self, tokenizer):
        """When exclude flag is off, tool_choice=none still includes tools in template kwargs."""
        _, _, _, _, chat_params = _prepare_request(
            {**TOOL_REQUEST, "tool_choice": "none"},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=False,
        )
        tools = chat_params.chat_template_kwargs["tools"]
        assert (
            tools is not None and len(tools) == 1
        ), "tool_choice=none with flag off should keep tools in template"

    def test_tool_choice_auto_keeps_tools(self, tokenizer):
        """tool_choice=auto should always include tools regardless of flag."""
        _, _, _, _, chat_params = _prepare_request(
            {**TOOL_REQUEST, "tool_choice": "auto"},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )
        tools = chat_params.chat_template_kwargs["tools"]
        assert (
            tools is not None and len(tools) == 1
        ), "tool_choice=auto should keep tools in template"

    def test_structured_outputs_skip_auto_tool_parser(self, tokenizer):
        """Explicit structured outputs stay on grammar path, not auto tool parsing."""
        _, parser, template_kwargs, _, _ = _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Return JSON"}],
                "structured_outputs": {"json": {"type": "object"}},
                "chat_template_kwargs": {"enable_thinking": True},
            },
            tokenizer=tokenizer,
            tool_parser_class=_resolve_qwen3_tool_parser_class(),
            enable_auto_tool_choice=True,
        )

        assert parser is None
        assert template_kwargs["enable_thinking"] is False
        assert template_kwargs["thinking"] is False

    def test_omitted_tool_choice_with_tools_defaults_to_auto(self, tokenizer):
        """OpenAI-compatible default: omitted tool_choice plus tools means auto."""
        request_for_sampling, _, _, _, chat_params = _prepare_request(
            TOOL_REQUEST,
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )
        assert request_for_sampling.tool_choice == "auto"
        tools = chat_params.chat_template_kwargs["tools"]
        assert (
            tools is not None and len(tools) == 1
        ), "omitted tool_choice with tools should keep tools in template"

    def test_tool_choice_required_keeps_tools(self, tokenizer):
        """tool_choice=required should always include tools regardless of flag."""
        _, _, _, _, chat_params = _prepare_request(
            {**TOOL_REQUEST, "tool_choice": "required"},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )
        tools = chat_params.chat_template_kwargs["tools"]
        assert (
            tools is not None and len(tools) == 1
        ), "tool_choice=required should keep tools in template"

    def test_tool_choice_required_preserves_template_thinking(self, tokenizer):
        _, _, chat_template_kwargs, _, chat_params = _prepare_request(
            {
                **TOOL_REQUEST,
                "tool_choice": "required",
                "chat_template_kwargs": {"enable_thinking": True, "thinking": True},
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )

        assert chat_template_kwargs["enable_thinking"] is True
        assert chat_template_kwargs["thinking"] is True
        assert chat_params.chat_template_kwargs["enable_thinking"] is True
        assert chat_params.chat_template_kwargs["thinking"] is True

    def test_named_tool_choice_preserves_template_thinking(self, tokenizer):
        _, _, chat_template_kwargs, _, chat_params = _prepare_request(
            {
                **TOOL_REQUEST,
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
                "chat_template_kwargs": {"enable_thinking": True, "thinking": True},
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )

        assert chat_template_kwargs["enable_thinking"] is True
        assert chat_template_kwargs["thinking"] is True
        assert chat_params.chat_template_kwargs["enable_thinking"] is True
        assert chat_params.chat_template_kwargs["thinking"] is True

    def test_tool_choice_auto_preserves_template_thinking(self, tokenizer):
        _, _, chat_template_kwargs, _, chat_params = _prepare_request(
            {
                **TOOL_REQUEST,
                "tool_choice": "auto",
                "chat_template_kwargs": {"enable_thinking": True, "thinking": True},
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )

        assert chat_template_kwargs["enable_thinking"] is True
        assert chat_template_kwargs["thinking"] is True
        assert chat_params.chat_template_kwargs["enable_thinking"] is True
        assert chat_params.chat_template_kwargs["thinking"] is True

    def test_thinking_alias_disables_enable_thinking(self, tokenizer):
        _, _, chat_template_kwargs, _, chat_params = _prepare_request(
            {
                **TOOL_REQUEST,
                "chat_template_kwargs": {"thinking": False},
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )

        assert chat_template_kwargs["thinking"] is False
        assert chat_template_kwargs["enable_thinking"] is False
        assert chat_params.chat_template_kwargs["enable_thinking"] is False

    def test_reasoning_budget_template_kwargs_are_preserved(self, tokenizer):
        _, _, chat_template_kwargs, _, chat_params = _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "chat_template_kwargs": {
                    "reasoning_budget": 24000,
                    "reasoning_budget_grace_period": 128,
                },
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
        )

        assert chat_template_kwargs["reasoning_budget"] == 24000
        assert chat_template_kwargs["reasoning_budget_grace_period"] == 128
        assert chat_params.chat_template_kwargs["reasoning_budget"] == 24000
        assert chat_params.chat_template_kwargs["reasoning_budget_grace_period"] == 128

    def test_no_tools_in_request(self, tokenizer):
        """Request without tools should produce None tools in template kwargs."""
        _, _, _, _, chat_params = _prepare_request(
            {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}]},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )
        assert (
            chat_params.chat_template_kwargs["tools"] is None
        ), "No tools in request should produce None tools in template"


class TestReasoningEffortTemplateKwargs:
    @pytest.mark.parametrize("effort", ["minimal", "low", "medium"])
    def test_non_high_efforts_set_size_specific_flags(self, tokenizer, effort):
        _, _, chat_template_kwargs, _, chat_params = _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": effort,
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
        )

        assert chat_template_kwargs["enable_thinking"] is True
        assert chat_template_kwargs["low_effort"] is True
        assert chat_template_kwargs["medium_effort"] is True
        assert chat_params.chat_template_kwargs["enable_thinking"] is True
        assert chat_params.chat_template_kwargs["low_effort"] is True
        assert chat_params.chat_template_kwargs["medium_effort"] is True

    @pytest.mark.parametrize("effort", ["high", "xhigh", "max"])
    def test_high_efforts_enable_max_thinking(self, tokenizer, effort):
        _, _, chat_template_kwargs, _, chat_params = _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": effort,
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
        )

        assert chat_template_kwargs["enable_thinking"] is True
        assert "low_effort" not in chat_template_kwargs
        assert "medium_effort" not in chat_template_kwargs
        assert chat_params.chat_template_kwargs["enable_thinking"] is True
        assert "low_effort" not in chat_params.chat_template_kwargs
        assert "medium_effort" not in chat_params.chat_template_kwargs

    def test_none_disables_thinking(self, tokenizer):
        _, _, chat_template_kwargs, _, chat_params = _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "none",
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
        )

        assert chat_template_kwargs["enable_thinking"] is False
        assert chat_params.chat_template_kwargs["enable_thinking"] is False

    def test_explicit_template_flags_win(self, tokenizer):
        _, _, chat_template_kwargs, _, chat_params = _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "low",
                "chat_template_kwargs": {"enable_thinking": False},
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
        )

        assert chat_template_kwargs["enable_thinking"] is False
        assert "low_effort" not in chat_template_kwargs
        assert "medium_effort" not in chat_template_kwargs
        assert chat_params.chat_template_kwargs["enable_thinking"] is False
        assert "low_effort" not in chat_params.chat_template_kwargs
        assert "medium_effort" not in chat_params.chat_template_kwargs


class TestReasoningParserMetadata:
    def test_no_reasoning_parser_returns_none(self):
        from dynamo.frontend.vllm_processor import _build_reasoning_parser_metadata

        assert _build_reasoning_parser_metadata(
            None,
            object(),
            {},
            SimpleNamespace(include_reasoning=True),
            [1, 2, 3],
        ) == (None, None)

    def test_include_reasoning_false_marks_reasoning_ended(self):
        from dynamo.frontend.vllm_processor import _build_reasoning_parser_metadata

        class ParserShouldNotBeBuilt:
            def __init__(self, *args, **kwargs):
                raise AssertionError("parser should not be constructed")

        reasoning_ended, parser_kwargs = _build_reasoning_parser_metadata(
            ParserShouldNotBeBuilt,
            object(),
            {"reasoning_effort": "low"},
            SimpleNamespace(include_reasoning=False),
            [1, 2, 3],
        )

        assert reasoning_ended is True
        assert parser_kwargs == {"chat_template_kwargs": {"reasoning_effort": "low"}}

    def test_enable_thinking_false_marks_reasoning_ended(self):
        from dynamo.frontend.vllm_processor import _build_reasoning_parser_metadata

        class ParserShouldNotBeBuilt:
            def __init__(self, *args, **kwargs):
                raise AssertionError("parser should not be constructed")

        reasoning_ended, parser_kwargs = _build_reasoning_parser_metadata(
            ParserShouldNotBeBuilt,
            object(),
            {"enable_thinking": False},
            SimpleNamespace(include_reasoning=True),
            [1, 2, 3],
        )

        assert reasoning_ended is True
        assert parser_kwargs == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_enable_thinking_false_without_frontend_parser_marks_reasoning_ended(self):
        from dynamo.frontend.vllm_processor import _build_reasoning_parser_metadata

        reasoning_ended, parser_kwargs = _build_reasoning_parser_metadata(
            None,
            object(),
            {"enable_thinking": False},
            SimpleNamespace(include_reasoning=True),
            [1, 2, 3],
        )

        assert reasoning_ended is True
        assert parser_kwargs == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_parser_receives_chat_template_kwargs(self):
        from dynamo.frontend.vllm_processor import _build_reasoning_parser_metadata

        class FakeReasoningParser:
            def __init__(self, tokenizer, *, chat_template_kwargs):
                self.tokenizer = tokenizer
                self.chat_template_kwargs = chat_template_kwargs

            def is_reasoning_end(self, prompt_token_ids):
                return prompt_token_ids == [9, 9]

        tokenizer = object()
        reasoning_ended, parser_kwargs = _build_reasoning_parser_metadata(
            FakeReasoningParser,
            tokenizer,
            {"reasoning_effort": "high"},
            SimpleNamespace(include_reasoning=True),
            [9, 9],
        )

        assert reasoning_ended is True
        assert parser_kwargs == {"chat_template_kwargs": {"reasoning_effort": "high"}}

    def test_structured_outputs_serialize_to_guided_decoding(self):
        from dynamo.frontend.vllm_processor import _structured_outputs_to_guided_decoding

        guided = _structured_outputs_to_guided_decoding(
            StructuredOutputsParams(
                json={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                disable_any_whitespace=True,
            )
        )

        assert guided == {
            "json": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            "disable_any_whitespace": True,
            "disable_additional_properties": False,
        }

    def test_constraints_empty_structured_outputs_are_discarded(self):
        from dynamo.frontend.vllm_processor import _active_structured_outputs

        empty = SimpleNamespace(all_constraints_none=lambda: True)
        active = StructuredOutputsParams(json={"type": "object"})

        assert _active_structured_outputs(empty) is None
        assert _active_structured_outputs(active) is active

    def test_response_format_json_schema_serializes_to_guided_decoding(self):
        from dynamo.frontend.vllm_processor import (
            _structured_outputs_from_response_format,
            _structured_outputs_to_guided_decoding,
        )

        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        structured_outputs = _structured_outputs_from_response_format(
            {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            }
        )

        guided = _structured_outputs_to_guided_decoding(structured_outputs)

        assert guided == {
            "json": schema,
            "disable_any_whitespace": False,
            "disable_additional_properties": False,
        }

    def test_request_extra_structured_outputs_serializes_to_guided_decoding(self):
        from dynamo.frontend.vllm_processor import (
            _request_structured_outputs,
            _structured_outputs_to_guided_decoding,
        )

        schema = {
            "type": "object",
            "properties": {"records": {"type": "array"}},
            "required": ["records"],
        }
        structured_outputs = _request_structured_outputs(
            SimpleNamespace(
                structured_outputs=None,
                model_extra={"structured_outputs": {"json": schema}},
                response_format=None,
            ),
            SamplingParams(max_tokens=128),
        )

        assert _structured_outputs_to_guided_decoding(structured_outputs)["json"] == schema

    def test_kv_router_copies_reasoning_metadata_to_extra_args(self):
        from dynamo.frontend.vllm_processor import _inject_routing_metadata

        kv_kwargs = {"extra_args": {"mm_hashes": [123]}}
        _inject_routing_metadata(
            {
                "reasoning_ended": False,
                "reasoning_parser_kwargs": {
                    "chat_template_kwargs": {"reasoning_effort": "high"}
                },
                "reasoning_budget": 24000,
                "reasoning_budget_grace_period": 128,
            },
            kv_kwargs,
        )

        assert kv_kwargs["extra_args"] == {
            "mm_hashes": [123],
            "reasoning_ended": False,
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"reasoning_effort": "high"}
            },
            "reasoning_budget": 24000,
            "reasoning_budget_grace_period": 128,
        }


class _FakeOutputProcessor:
    def __init__(self):
        self.request_states = {}
        self.added_requests = []
        self.aborted_requests = []

    def add_request(self, preproc, *args, **kwargs):
        self.added_requests.append((preproc, args, kwargs))
        self.request_states[preproc.request_id] = object()

    def process_outputs(self, outputs):
        return SimpleNamespace(
            reqs_to_abort=[],
            request_outputs=[SimpleNamespace(outputs=[SimpleNamespace(index=0)])],
        )

    def abort_requests(self, request_ids, internal=False):
        self.aborted_requests.append((request_ids, internal))
        for request_id in request_ids:
            self.request_states.pop(request_id, None)


class _FakePostProcessor:
    def needs_raw_parser_delta(self, raw_delta_token_ids):
        return False

    def process_output(self, output, raw_delta_token_ids=None):
        return {
            "index": output.index,
            "delta": {"content": "x"},
            "finish_reason": None,
        }


@pytest.fixture
def vllm_processor_module(monkeypatch):
    import dynamo.frontend.vllm_processor as module

    class FakeEngineCoreOutput:
        __struct_fields__ = ()

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(module, "EngineCoreOutput", FakeEngineCoreOutput)
    monkeypatch.setattr(module._nvtx, "start_range", lambda *args, **kwargs: object())
    monkeypatch.setattr(module._nvtx, "end_range", lambda rng: None)
    return module


def _make_processor(module, routed_engine):
    processor = module.VllmProcessor.__new__(module.VllmProcessor)
    processor.routed_engine = routed_engine
    processor.output_processor = _FakeOutputProcessor()
    return processor


def _base_preproc():
    return {
        "model": MODEL,
        "token_ids": [1, 2, 3],
        "stop_conditions": {"max_tokens": 4},
        "sampling_options": {"temperature": 0.0},
        "output_options": {},
        "eos_token_ids": [],
        "annotations": [],
        "routing": None,
    }


async def _run_generate(processor, preproc, *, mm_routing_info=None, context=None):
    vllm_preproc = SimpleNamespace(
        sampling_params=SimpleNamespace(n=1),
        request_id="vllm-request",
        external_req_id=None,
    )
    post_processors = {0: _FakePostProcessor()}

    return [
        item
        async for item in processor._generate_and_stream(
            "request-id",
            {"model": MODEL},
            preproc,
            preproc["token_ids"],
            vllm_preproc,
            post_processors,
            request_for_sampling=SimpleNamespace(tool_choice=None),
            mm_routing_info=mm_routing_info,
            context=context,
        )
    ]


@pytest.mark.parametrize(
    ("raw_reason", "expected"),
    [
        ({"type": "stop"}, "STOP"),
        ({"type": "error", "message": "engine overloaded"}, "ERROR"),
        ({}, "ERROR"),
    ],
)
def test_map_finish_reason_accepts_router_objects(
    vllm_processor_module, raw_reason, expected
):
    assert vllm_processor_module.map_finish_reason(raw_reason) == getattr(
        vllm_processor_module.FinishReason, expected
    )


class TestRoutedEnginePath:
    @pytest.mark.parametrize(
        "raw_finish_reason", ["stop", {"type": "stop"}], ids=["string", "object"]
    )
    def test_terminal_chunk_without_vllm_output_flushes_postprocessor(
        self, vllm_processor_module, raw_finish_reason
    ):
        routed_engine = _FakeRoutedEngine(
            [{"token_ids": [], "index": 0, "finish_reason": raw_finish_reason}]
        )
        processor = _make_processor(vllm_processor_module, routed_engine)
        processor.output_processor.process_outputs = lambda outputs: SimpleNamespace(
            reqs_to_abort=[], request_outputs=[]
        )

        class TerminalPostProcessor(_FakePostProcessor):
            def process_output(self, output, raw_delta_token_ids=None):
                assert output.finish_reason == "stop"
                return {
                    "index": output.index,
                    "delta": {"role": "assistant", "tool_calls": [{"index": 0}]},
                    "finish_reason": "tool_calls",
                }

        async def run():
            vllm_preproc = SimpleNamespace(
                sampling_params=SimpleNamespace(n=1),
                request_id="vllm-request",
                external_req_id=None,
            )
            return [
                item
                async for item in processor._generate_and_stream(
                    "request-id",
                    {"model": MODEL},
                    _base_preproc(),
                    [1, 2, 3],
                    vllm_preproc,
                    {0: TerminalPostProcessor()},
                    request_for_sampling=SimpleNamespace(tool_choice=None),
                )
            ]

        chunks = asyncio.run(run())
        assert chunks[0]["data"]["choices"][0]["delta"]["tool_calls"] == [
            {"index": 0}
        ]

    @pytest.mark.asyncio
    async def test_routed_engine_gets_extra_args_metadata(self, vllm_processor_module):
        routed_engine = _FakeRoutedEngine()
        processor = _make_processor(vllm_processor_module, routed_engine)
        preproc = _base_preproc()
        preproc["extra_args"] = {"mm_hashes": [123]}
        preproc["reasoning_ended"] = False
        preproc["reasoning_parser_kwargs"] = {
            "chat_template_kwargs": {"reasoning_effort": "high"}
        }
        preproc["mm_processor_kwargs"] = {"use_audio_in_video": True}

        await _run_generate(processor, preproc)

        assert routed_engine.requests[0]["extra_args"] == {
            "mm_hashes": [123],
            "reasoning_ended": False,
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"reasoning_effort": "high"}
            },
            "mm_processor_kwargs": {"use_audio_in_video": True},
        }

    @pytest.mark.asyncio
    async def test_routed_stream_produces_openai_chunks(self, vllm_processor_module):
        routed_engine = _FakeRoutedEngine(
            [{"token_ids": [101], "index": 0, "finish_reason": None}]
        )
        processor = _make_processor(vllm_processor_module, routed_engine)

        chunks = await _run_generate(processor, _base_preproc())

        # One annotated envelope per iteration carries both data and the
        # llm_metrics annotation; observer strips the annotation before SSE.
        assert len(chunks) == 1
        envelope = chunks[0]

        assert envelope["_dynamo_annotated"] is True
        assert envelope["data"] == {
            "id": "request-id",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "x"},
                    "finish_reason": None,
                }
            ],
            "created": envelope["data"]["created"],
            "model": MODEL,
            "object": "chat.completion.chunk",
        }

        assert envelope["event"] == "llm_metrics"
        assert len(envelope["comment"]) == 1
        assert json.loads(envelope["comment"][0]) == {
            "input_tokens": 3,
            "output_tokens": 1,
            "chunk_tokens": 1,
        }


OBJECT_TYPED_TOOL_REQUEST = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "set my profile"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "set_profile",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"},
                            },
                        }
                    },
                    "required": ["profile"],
                },
            },
        }
    ],
    "tool_choice": "auto",
}


# ---------------------------------------------------------------------------
# _prepare_request: schema-aware tool-parser end-to-end regression
# ---------------------------------------------------------------------------


class TestSchemaAwareToolParser:
    """Schema-aware parsers (e.g. qwen3_coder) need ``tools`` at construction
    to coerce object/array-typed parameter values from raw text into JSON;
    without them, the value comes through as a string-in-a-string inside the
    final ``arguments`` JSON.
    """

    def test_qwen3_coder_coerces_object_typed_arg(self, tokenizer):
        """qwen3_coder must coerce object-typed parameter values into nested
        objects, not leave them as JSON-encoded strings inside ``arguments``.
        """
        model_output = (
            "<tool_call><function=set_profile>\n"
            "<parameter=profile>\n"
            '{"name": "Alice", "age": 30}\n'
            "</parameter>\n"
            "</function></tool_call>"
        )

        request_for_sampling, parser, _, _, _ = _prepare_request(
            OBJECT_TYPED_TOOL_REQUEST,
            tokenizer=tokenizer,
            tool_parser_class=_resolve_qwen3_tool_parser_class(),
        )
        assert parser is not None, "Expected _prepare_request to construct the parser"

        result = parser.extract_tool_calls(model_output, request_for_sampling)

        assert result.tools_called, f"Expected tools_called=True; got {result!r}"
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert isinstance(args["profile"], dict), (
            f"Schema-aware parser should coerce object-typed arg to dict; "
            f"got {type(args['profile']).__name__}: {args['profile']!r}"
        )
        assert args["profile"] == {"name": "Alice", "age": 30}


# ---------------------------------------------------------------------------
# _prepare_request: chat_template_kwargs forwarding
# ---------------------------------------------------------------------------


@pytest.mark.core
class TestChatTemplateKwargsForwarding:
    """chat_template_kwargs from the request are forwarded to ChatParams.

    Uses Qwen3 which supports enable_thinking: False to suppress <think> blocks.
    """

    @staticmethod
    def _messages():
        return [{"role": "user", "content": "Hello"}]

    def _prepare(self, request, tokenizer):
        """Return (chat_params, messages) from _prepare_request."""
        _, _, _, messages, chat_params = _prepare_request(
            request,
            tokenizer=tokenizer,
            tool_parser_class=None,
        )
        return chat_params, messages

    def _render(self, tokenizer, chat_params) -> str:
        """Render prompt text using the chat_params template kwargs."""
        kwargs = {**chat_params.chat_template_kwargs, "tokenize": False}
        return tokenizer.apply_chat_template(self._messages(), **kwargs)

    def test_qwen3_enable_thinking_true_no_closed_think_block(self, tokenizer):
        """enable_thinking=True leaves reasoning open (model generates <think> itself)."""
        chat_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_kwargs": {"enable_thinking": True},
            },
            tokenizer,
        )
        prompt = self._render(tokenizer, chat_params)
        assert "</think>" not in prompt

    def test_qwen3_thinking_flag_changes_tokens(self, tokenizer):
        """enable_thinking=True vs False produces different rendered prompts."""
        think_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_kwargs": {"enable_thinking": True},
            },
            tokenizer,
        )
        no_think_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_kwargs": {"enable_thinking": False},
            },
            tokenizer,
        )
        assert self._render(tokenizer, think_params) != self._render(
            tokenizer, no_think_params
        )

    def test_reasoning_effort_forwarded_to_template_kwargs(self, tokenizer):
        """reasoning_effort is always present in chat_params.chat_template_kwargs."""
        chat_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "reasoning_effort": "low",
            },
            tokenizer,
        )
        assert chat_params.chat_template_kwargs.get("reasoning_effort") == "low"


@pytest.mark.parametrize(
    ("runtime_config", "expected"),
    [
        ({"context_length": 1048576}, 1048576),
        ({}, None),
        ({"context_length": None}, None),
        ({"context_length": 0}, None),
        ({"context_length": -1}, None),
        ({"context_length": "1048576"}, None),
        ({"context_length": True}, None),
        (None, None),
    ],
)
def test_runtime_config_context_length(vllm_processor_module, runtime_config, expected):
    mdc = SimpleNamespace(runtime_config=lambda: runtime_config)

    assert vllm_processor_module._runtime_config_context_length(mdc) == expected


class _ToolMarkerTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {
            "<think>": [12],
            "</think>": [13],
            "<tool_call>": [14],
            "</tool_call>": [15],
        }.get(text, [])

    def decode(self, token_ids, skip_special_tokens=False):
        token_text = {
            12: "<think>",
            13: "</think>",
            14: "<tool_call>",
            15: "</tool_call>",
        }
        return "".join(token_text.get(token_id, "") for token_id in token_ids)


class _ThinkingParser:
    start_token = "<think>"
    end_token = "</think>"
    start_token_id = 12
    end_token_id = 13

    def __init__(self, tokenizer, **kwargs):
        pass

    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ):
        if self.end_token_id in delta_token_ids:
            _, _, content = delta_text.partition(self.end_token)
            return DeltaMessage(content=content or None)
        return DeltaMessage(reasoning=delta_text or None)

    def is_reasoning_end_streaming(self, current_token_ids, delta_token_ids):
        return self.end_token_id in delta_token_ids


def _postprocessor_output(text, token_ids, finish_reason=None):
    return SimpleNamespace(
        index=0,
        text=text,
        token_ids=token_ids,
        finish_reason=finish_reason,
        logprobs=None,
    )


def _plain_request():
    return SimpleNamespace(
        tool_choice=None,
        tools=None,
        structured_outputs=None,
        response_format=None,
        include_reasoning=True,
    )


def test_duplicate_reasoning_end_marker_is_not_emitted_after_budget_cutoff():
    post = StreamingPostProcessor(
        tokenizer=_ToolMarkerTokenizer(),
        request_for_sampling=_plain_request(),
        sampling_params=SamplingParams(max_tokens=128),
        prompt_token_ids=[],
        tool_parser=None,
        reasoning_parser_class=_ThinkingParser,
        chat_template_kwargs={"enable_thinking": True},
    )
    post.reasoning_is_done = True

    marker_choice = post.process_output(
        _postprocessor_output("</think>", [13])
    )
    content_choice = post.process_output(
        _postprocessor_output("answer", [21], "stop")
    )

    assert marker_choice is None
    assert content_choice["delta"]["content"] == "answer"


def test_structured_json_with_reasoning_parses_content_after_thinking():
    post = StreamingPostProcessor(
        tokenizer=_ToolMarkerTokenizer(),
        request_for_sampling=_plain_request(),
        sampling_params=SamplingParams(
            max_tokens=128,
            structured_outputs=StructuredOutputsParams(
                json={"type": "object"}
            ),
        ),
        prompt_token_ids=[],
        tool_parser=None,
        reasoning_parser_class=_ThinkingParser,
        chat_template_kwargs={"enable_thinking": True},
    )

    reasoning_choice = post.process_output(
        _postprocessor_output("analysis", [21])
    )
    marker_choice = post.process_output(
        _postprocessor_output("</think>", [13])
    )
    content_choice = post.process_output(
        _postprocessor_output('{"answer":4}', [22], "stop")
    )

    assert reasoning_choice["delta"]["reasoning_content"] == "analysis"
    assert marker_choice is None
    assert content_choice["delta"]["content"] == '{"answer":4}'
    assert content_choice["finish_reason"] == "stop"


def test_tool_markup_is_removed_without_frontend_parser():
    post = StreamingPostProcessor(
        tokenizer=_ToolMarkerTokenizer(),
        request_for_sampling=SimpleNamespace(
            tool_choice="auto",
            tools=[{"type": "function"}],
            structured_outputs=None,
            response_format=None,
        ),
        sampling_params=SamplingParams(max_tokens=128),
        prompt_token_ids=[],
        tool_parser=None,
        reasoning_parser_class=None,
        chat_template_kwargs={"enable_thinking": True},
    )
    delta = {"reasoning_content": "Need the file.<tool_call><function=Read>"}

    post._strip_tool_markup_from_delta(delta)

    assert delta == {"reasoning_content": "Need the file."}


def test_terminal_reasoning_only_stream_gets_nonempty_content_fallback():
    post = StreamingPostProcessor(
        tokenizer=_ToolMarkerTokenizer(),
        request_for_sampling=SimpleNamespace(
            tool_choice=None,
            tools=None,
            structured_outputs=None,
            response_format=None,
        ),
        sampling_params=SamplingParams(max_tokens=128),
        prompt_token_ids=[],
        tool_parser=None,
        reasoning_parser_class=None,
        chat_template_kwargs={"enable_thinking": True},
    )
    post._emitted_reasoning_text = "I reached the token limit while reasoning."

    choice = post._build_choice(
        SimpleNamespace(index=0, finish_reason="length", logprobs=None), {}
    )

    assert choice["delta"]["content"] == post._emitted_reasoning_text
    assert choice["finish_reason"] == "length"


def test_orphaned_qwen_tool_closing_markup_is_removed():
    post = StreamingPostProcessor(
        tokenizer=_ToolMarkerTokenizer(),
        request_for_sampling=SimpleNamespace(
            tool_choice="auto",
            tools=[{"type": "function"}],
            structured_outputs=None,
            response_format=None,
        ),
        sampling_params=SamplingParams(max_tokens=128),
        prompt_token_ids=[],
        tool_parser=None,
        reasoning_parser_class=None,
        chat_template_kwargs={"enable_thinking": True},
    )
    first_delta = {"reasoning_content": "Let me update the fix:\n</"}
    leaked_tail = {
        "reasoning_content": (
            "parameter>\n<parameter=command>python3 -c 'print(42)'\n"
            "</parameter>\n</function>\n</tool_call>"
        )
    }

    post._strip_tool_markup_from_delta(first_delta)
    post._strip_tool_markup_from_delta(leaked_tail)

    assert first_delta == {"reasoning_content": "Let me update the fix:\n"}
    assert leaked_tail == {}


def test_non_tool_closing_markup_is_preserved():
    post = StreamingPostProcessor(
        tokenizer=_ToolMarkerTokenizer(),
        request_for_sampling=SimpleNamespace(
            tool_choice="auto",
            tools=[{"type": "function"}],
            structured_outputs=None,
            response_format=None,
        ),
        sampling_params=SamplingParams(max_tokens=128),
        prompt_token_ids=[],
        tool_parser=None,
        reasoning_parser_class=None,
        chat_template_kwargs={"enable_thinking": True},
    )
    delta = {"reasoning_content": "The HTML closes with </div>."}

    post._strip_tool_markup_from_delta(delta)

    assert delta == {"reasoning_content": "The HTML closes with </div>."}


def test_tool_markers_fall_back_to_parser_engine_ids():
    parser = SimpleNamespace(
        tool_call_start_token=None,
        tool_call_end_token=None,
        _parser_engine=SimpleNamespace(
            _tool_call_token_id=14,
            _tool_call_end_token_id=15,
        ),
    )
    post = StreamingPostProcessor(
        tokenizer=_ToolMarkerTokenizer(),
        request_for_sampling=SimpleNamespace(tool_choice="auto", tools=[]),
        sampling_params=SamplingParams(max_tokens=128),
        prompt_token_ids=[],
        tool_parser=parser,
        reasoning_parser_class=None,
        chat_template_kwargs={"enable_thinking": True},
    )

    assert post._tool_call_start_token() == "<tool_call>"
    assert post._tool_call_end_token() == "</tool_call>"
