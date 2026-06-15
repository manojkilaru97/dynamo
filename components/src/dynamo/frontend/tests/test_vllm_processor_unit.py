#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM processor components.

Tests for the tool-stripping behaviour of _prepare_request when
tool_choice='none' and the exclude_tools_when_tool_choice_none flag.
"""

import json
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

from dynamo.frontend.prepost import StreamingPostProcessor, _prepare_request

# Needs vllm packages (gpu_1 container).  No need for parallel marker.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
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
        _, parser, _, _, _ = _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Return JSON"}],
                "structured_outputs": {"json": {"type": "object"}},
            },
            tokenizer=tokenizer,
            tool_parser_class=Qwen3CoderToolParser,
            enable_auto_tool_choice=True,
        )

        assert parser is None

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
        from dynamo.frontend.vllm_processor import (
            _copy_reasoning_metadata_to_extra_args,
        )

        kv_kwargs = {"extra_args": {"mm_hashes": [123]}}
        _copy_reasoning_metadata_to_extra_args(
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
            tool_parser_class=Qwen3CoderToolParser,
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


class _FallbackReasoningParser:
    end_token_id = 13

    def __init__(self, tokenizer, *, chat_template_kwargs=None):
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
        if 13 in delta_token_ids:
            return DeltaMessage(content=delta_text.replace("</think>", ""))
        return DeltaMessage(reasoning=delta_text)

    def is_reasoning_end_streaming(self, current_token_ids, delta_token_ids):
        return 13 in delta_token_ids


class _FallbackToolParser:
    tool_call_start_token = "<tool_call>"
    tool_call_end_token = "</tool_call>"
    tool_call_start_token_id = 14
    tool_call_end_token_id = 15

    def extract_tool_calls_streaming(self, **kwargs):
        return None

    def extract_tool_calls(self, text, request):
        called = (
            self.tool_call_start_token in text
            and self.tool_call_end_token in text
        )
        return SimpleNamespace(
            tools_called=called,
            content=None if called else text,
            tool_calls=[
                SimpleNamespace(
                    id=None,
                    function=SimpleNamespace(
                        name="Read",
                        arguments='{"path":"package.json"}',
                    ),
                )
            ]
            if called
            else [],
        )


class _RawToolTokenizer:
    all_special_tokens = ("<|im_end|>",)

    _token_text = {
        1: "Need the file.",
        13: "</think>",
        14: "<tool_call>",
        15: "</tool_call>",
        20: "\n<function=Read>\n",
        21: "<parameter=path>package.json</parameter>\n</function>\n",
        99: "<|im_end|>",
    }

    def decode(self, token_ids, skip_special_tokens=False):
        text = "".join(self._token_text.get(token_id, "") for token_id in token_ids)
        if skip_special_tokens:
            text = text.replace("<|im_end|>", "")
        return text

    def convert_ids_to_tokens(self, token_ids):
        return [self._token_text.get(token_id, "") for token_id in token_ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


class TestStreamingPostProcessorStructuredJson:
    def test_pure_structured_json_emits_first_complete_value_once(self):
        post = StreamingPostProcessor(
            tokenizer=SimpleNamespace(all_special_tokens=()),
            request_for_sampling=SimpleNamespace(
                tool_choice="auto",
                tools=[],
                structured_outputs={
                    "json": {
                        "type": "object",
                        "properties": {"a": {"type": "number"}},
                        "required": ["a"],
                    }
                },
                response_format=None,
            ),
            sampling_params=SamplingParams(max_tokens=128),
            prompt_token_ids=[],
            tool_parser=_FallbackToolParser(),
            reasoning_parser_class=None,
            chat_template_kwargs={"enable_thinking": False},
        )

        chunks = [
            SimpleNamespace(
                index=0,
                token_ids=[1],
                text='{"a":1',
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[2],
                text='} {"a":2}',
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[3],
                text=' {"a":3}',
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[],
                text="",
                finish_reason="stop",
                logprobs=None,
            ),
        ]

        choices = [choice for chunk in chunks if (choice := post.process_output(chunk))]

        assert len(choices) == 2
        assert choices[0]["delta"]["content"] == '{"a":1}'
        assert choices[-1]["finish_reason"] == "stop"


class TestStreamingPostProcessorToolFallback:
    def test_complete_buffered_tool_call_emits_before_finish_chunk(self):
        post = StreamingPostProcessor(
            tokenizer=SimpleNamespace(all_special_tokens=()),
            request_for_sampling=SimpleNamespace(tool_choice="auto"),
            sampling_params=SamplingParams(max_tokens=65536),
            prompt_token_ids=[],
            tool_parser=_FallbackToolParser(),
            reasoning_parser_class=_FallbackReasoningParser,
            chat_template_kwargs={"enable_thinking": True},
        )

        chunks = [
            SimpleNamespace(
                index=0,
                token_ids=[1],
                text="Need the file.",
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[13],
                text="</think>",
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[14],
                text=(
                    "<tool_call>\n<function=Read>\n"
                    "<parameter=path>package.json</parameter>\n"
                    "</function>\n</tool_call>"
                ),
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[99],
                text="ignored continuation",
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[],
                text="",
                finish_reason="length",
                logprobs=None,
            ),
        ]

        choices = [choice for chunk in chunks if (choice := post.process_output(chunk))]

        assert len(choices) == 3
        assert choices[1]["delta"]["tool_calls"][0]["function"]["name"] == "Read"
        assert "ignored continuation" not in json.dumps(choices)
        assert choices[-1]["finish_reason"] == "tool_calls"

    def test_final_chunk_recovers_buffered_post_reasoning_tool_call(self):
        post = StreamingPostProcessor(
            tokenizer=SimpleNamespace(all_special_tokens=()),
            request_for_sampling=SimpleNamespace(tool_choice="auto"),
            sampling_params=SamplingParams(max_tokens=128),
            prompt_token_ids=[],
            tool_parser=_FallbackToolParser(),
            reasoning_parser_class=_FallbackReasoningParser,
            chat_template_kwargs={"enable_thinking": True},
        )

        chunks = [
            SimpleNamespace(
                index=0,
                token_ids=[1],
                text="Need the file.",
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[13],
                text="</think>",
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[14],
                text=(
                    "<tool_call>\n<function=Read>\n"
                    "<parameter=path>package.json</parameter>\n"
                ),
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[],
                text="</function>\n</tool_call>",
                finish_reason="stop",
                logprobs=None,
            ),
        ]

        choices = [choice for chunk in chunks if (choice := post.process_output(chunk))]
        final_choice = choices[-1]

        assert final_choice["finish_reason"] == "tool_calls"
        tool_calls = final_choice["delta"]["tool_calls"]
        assert tool_calls[0]["function"]["name"] == "Read"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {
            "path": "package.json"
        }

    def test_final_chunk_prefers_raw_token_tool_call_over_forced_content(self):
        post = StreamingPostProcessor(
            tokenizer=_RawToolTokenizer(),
            request_for_sampling=SimpleNamespace(tool_choice="auto"),
            sampling_params=SamplingParams(max_tokens=128),
            prompt_token_ids=[],
            tool_parser=_FallbackToolParser(),
            reasoning_parser_class=_FallbackReasoningParser,
            chat_template_kwargs={"enable_thinking": True},
        )

        chunks = [
            SimpleNamespace(
                index=0,
                token_ids=[1],
                text="Need the file.",
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[13],
                text="</think>",
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[14, 20, 21, 15],
                text="",
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[99],
                text="Need the file.",
                finish_reason="stop",
                logprobs=None,
            ),
        ]

        choices = [choice for chunk in chunks if (choice := post.process_output(chunk))]
        tool_choice = next(
            choice for choice in choices if "tool_calls" in choice["delta"]
        )
        final_choice = choices[-1]

        assert final_choice["finish_reason"] == "tool_calls"
        assert "content" not in final_choice["delta"]
        tool_calls = tool_choice["delta"]["tool_calls"]
        assert tool_calls[0]["function"]["name"] == "Read"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {
            "path": "package.json"
        }

    def test_tool_call_inside_reasoning_tokens_beats_forced_content_replay(self):
        post = StreamingPostProcessor(
            tokenizer=_RawToolTokenizer(),
            request_for_sampling=SimpleNamespace(tool_choice="auto"),
            sampling_params=SamplingParams(max_tokens=128),
            prompt_token_ids=[],
            tool_parser=_FallbackToolParser(),
            reasoning_parser_class=_FallbackReasoningParser,
            chat_template_kwargs={"enable_thinking": True},
        )

        chunks = [
            (
                SimpleNamespace(
                    index=0,
                    token_ids=[1],
                    text="Need the file.",
                    finish_reason=None,
                    logprobs=None,
                ),
                [1],
            ),
            (
                SimpleNamespace(
                    index=0,
                    token_ids=[],
                    text="",
                    finish_reason=None,
                    logprobs=None,
                ),
                [14, 20],
            ),
            (
                SimpleNamespace(
                    index=0,
                    token_ids=[],
                    text="",
                    finish_reason=None,
                    logprobs=None,
                ),
                [21, 15],
            ),
            (
                SimpleNamespace(
                    index=0,
                    token_ids=[1],
                    text="Need the file.",
                    finish_reason="stop",
                    logprobs=None,
                ),
                [99],
            ),
        ]

        choices = [
            choice
            for chunk, raw_token_ids in chunks
            if (
                choice := post.process_output(
                    chunk,
                    raw_delta_token_ids=raw_token_ids,
                )
            )
        ]
        tool_choice = next(
            choice for choice in choices if "tool_calls" in choice["delta"]
        )
        final_choice = choices[-1]

        assert final_choice["finish_reason"] == "tool_calls"
        assert "content" not in final_choice["delta"]
        tool_calls = tool_choice["delta"]["tool_calls"]
        assert tool_calls[0]["function"]["name"] == "Read"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {
            "path": "package.json"
        }
