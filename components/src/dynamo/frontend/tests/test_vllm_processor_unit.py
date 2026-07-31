#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM processor components.

Tests for the tool-stripping behaviour of _prepare_request when
tool_choice='none' and the exclude_tools_when_tool_choice_none flag.
"""

import asyncio
import json
from enum import IntEnum
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

try:
    from vllm.tool_parsers.qwen3_engine_tool_parser import (
        Qwen3EngineToolParser as Qwen3CoderToolParser,
    )
except ImportError:
    from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

from dynamo.frontend.prepost import StreamingPostProcessor, _prepare_request
from dynamo.frontend.vllm_processor import (
    VllmProcessor,
    _strip_malformed_composite_tool_tail,
    _strip_raw_tool_protocol_from_choice,
    _tools_enabled,
    _with_parser_visible_engine_text,
)

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


def test_async_tokenizer_fallback_is_cached_and_preserves_kwargs(monkeypatch):
    from dynamo.frontend import prepost

    calls = []

    class _Tokenizer:
        def __call__(self, text, **kwargs):
            calls.append((text, kwargs))
            return SimpleNamespace(input_ids=[7, 8, 9])

    tokenizer = _Tokenizer()
    monkeypatch.setattr(prepost, "_AsyncMicrobatchTokenizer", None)
    prepost._ASYNC_TOKENIZER_POOL.pop(id(tokenizer), None)

    async def exercise():
        first = prepost._get_async_tokenizer(tokenizer)
        second = prepost._get_async_tokenizer(tokenizer)
        encoded = await first("hello", add_special_tokens=False)
        return first, second, encoded

    first, second, encoded = asyncio.run(exercise())
    prepost._ASYNC_TOKENIZER_POOL.pop(id(tokenizer), None)

    assert first is second
    assert encoded.input_ids == [7, 8, 9]
    assert calls == [("hello", {"add_special_tokens": False})]


def test_async_tokenizer_prefers_microbatch_implementation(monkeypatch):
    from dynamo.frontend import prepost

    class _Tokenizer:
        pass

    class _MicrobatchTokenizer:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    tokenizer = _Tokenizer()
    monkeypatch.setattr(
        prepost,
        "_AsyncMicrobatchTokenizer",
        _MicrobatchTokenizer,
    )
    prepost._ASYNC_TOKENIZER_POOL.pop(id(tokenizer), None)

    wrapped = prepost._get_async_tokenizer(tokenizer)
    prepost._ASYNC_TOKENIZER_POOL.pop(id(tokenizer), None)

    assert isinstance(wrapped, _MicrobatchTokenizer)
    assert wrapped.tokenizer is tokenizer


def test_raw_tool_request_remains_detectable_after_sampling_normalization():
    assert _tools_enabled(TOOL_REQUEST)
    assert not _tools_enabled({**TOOL_REQUEST, "tool_choice": "none"})


def test_final_openai_boundary_strips_protocol_and_continuations():
    states: dict[int, dict] = {}
    first = {
        "index": 0,
        "delta": {
            "reasoning_content": (
                "Need the file.<tool_call><function=Read>"
            )
        },
    }
    continuation = {
        "index": 0,
        "delta": {
            "reasoning_content": "<parameter=path>package.json</parameter>"
        },
    }

    _strip_raw_tool_protocol_from_choice(first, states)
    _strip_raw_tool_protocol_from_choice(continuation, states)

    assert first["delta"]["reasoning_content"] == "Need the file."
    assert "reasoning_content" not in continuation["delta"]


@pytest.mark.parametrize("split_at", range(1, len("<tool_call>")))
def test_final_openai_boundary_strips_split_protocol(split_at):
    states: dict[int, dict] = {}
    marker = "<tool_call>"
    first = {
        "index": 0,
        "delta": {"reasoning_content": "Need the file." + marker[:split_at]},
    }
    second = {
        "index": 0,
        "delta": {"reasoning_content": marker[split_at:] + "<function=Read>"},
    }

    _strip_raw_tool_protocol_from_choice(first, states)
    _strip_raw_tool_protocol_from_choice(second, states)

    assert first["delta"]["reasoning_content"] == "Need the file."
    assert "reasoning_content" not in second["delta"]


def test_final_openai_boundary_tracks_split_protocol_across_fields():
    states: dict[int, dict] = {}
    first = {
        "index": 0,
        "delta": {"reasoning_content": "Need the file.<tool_"},
    }
    second = {
        "index": 0,
        "delta": {"content": "call><function=Read>"},
    }

    _strip_raw_tool_protocol_from_choice(first, states)
    _strip_raw_tool_protocol_from_choice(second, states)

    assert first["delta"]["reasoning_content"] == "Need the file."
    assert "content" not in second["delta"]


def test_final_openai_boundary_flushes_partial_marker_at_finish():
    choice = {
        "index": 0,
        "delta": {"reasoning_content": "This ends with <tool_"},
        "finish_reason": "stop",
    }

    _strip_raw_tool_protocol_from_choice(choice, {})

    assert choice["delta"]["reasoning_content"] == "This ends with <tool_"


@pytest.mark.parametrize(
    "marker",
    ("</parameter>", "</function>", "</tool_call>", "<|im_end|>"),
)
def test_final_openai_boundary_strips_orphaned_protocol_suffixes(marker):
    choice = {
        "index": 0,
        "delta": {"reasoning_content": f"Malformed tool attempt.{marker}"},
    }

    _strip_raw_tool_protocol_from_choice(choice, {})

    assert choice["delta"]["reasoning_content"] == "Malformed tool attempt."


def test_final_openai_boundary_strips_observed_incomplete_parameter_call():
    visible = "Let me test this more carefully:"
    malformed = (
        "\n</parameter>\n<parameter=description>\n"
        "Debug why _store_type_annotation_node does not work"
    )
    output = visible + malformed

    for split_at in range(len(output) + 1):
        states: dict[int, dict] = {}
        choices = [
            {
                "index": 0,
                "delta": {"reasoning_content": output[:split_at]},
            },
            {
                "index": 0,
                "delta": {"reasoning_content": output[split_at:]},
            },
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        ]

        for choice in choices:
            _strip_malformed_composite_tool_tail(choice, states)

        assert _reasoning_from_choices(choices) == visible


def test_composite_boundary_suppresses_only_reasoning_after_protocol_restart():
    states: dict[int, dict] = {}
    malformed = {
        "index": 0,
        "delta": {
            "reasoning_content": (
                "Reasoning.\n</parameter>\n<parameter=description>"
            ),
            "tool_calls": [{"index": 0, "function": {"name": "bash"}}],
        },
    }
    continuation = {
        "index": 0,
        "delta": {
            "reasoning_content": "hidden",
            "content": "visible content",
        },
        "finish_reason": "tool_calls",
    }

    _strip_malformed_composite_tool_tail(malformed, states)
    _strip_malformed_composite_tool_tail(continuation, states)

    assert malformed["delta"] == {
        "reasoning_content": "Reasoning.",
        "tool_calls": [{"index": 0, "function": {"name": "bash"}}],
    }
    assert continuation["delta"] == {"content": "visible content"}


def test_composite_boundary_preserves_protocol_restart_near_mismatch():
    output = "Reasoning.\n</parameter>\n<parameterX> is literal."
    split_at = output.index("X")
    choices = [
        {
            "index": 0,
            "delta": {"reasoning_content": output[:split_at]},
        },
        {
            "index": 0,
            "delta": {"reasoning_content": output[split_at:]},
            "finish_reason": "stop",
        },
    ]
    states: dict[int, dict] = {}

    for choice in choices:
        _strip_malformed_composite_tool_tail(choice, states)

    assert _reasoning_from_choices(choices) == output


_ORPHAN_COMPOSITE_TOOL_TAIL = "\n</parameter>\n</function>\n</tool_call>"


def _reasoning_from_choices(choices):
    return "".join(choice["delta"].get("reasoning_content", "") for choice in choices)


@pytest.mark.parametrize("terminal", ("", "<|im_end|>"))
def test_composite_boundary_strips_every_binary_partition_at_empty_finish(
    terminal,
):
    visible = "Let me test with the axis directly:"
    output = f"{visible}{_ORPHAN_COMPOSITE_TOOL_TAIL}{terminal}"

    for split_at in range(len(output) + 1):
        states: dict[int, dict] = {}
        choices = [
            {
                "index": 0,
                "delta": {"reasoning_content": output[:split_at]},
            },
            {
                "index": 0,
                "delta": {"reasoning_content": output[split_at:]},
            },
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        ]

        for choice in choices:
            _strip_malformed_composite_tool_tail(choice, states)

        assert _reasoning_from_choices(choices) == visible


@pytest.mark.parametrize("terminal", ("", "<|im_end|>"))
def test_composite_boundary_strips_character_partition_at_empty_finish(
    terminal,
):
    visible = "Reasoning."
    output = f"{visible}{_ORPHAN_COMPOSITE_TOOL_TAIL}{terminal}"
    choices = [
        {"index": 0, "delta": {"reasoning_content": character}} for character in output
    ]
    choices.append({"index": 0, "delta": {}, "finish_reason": "stop"})
    states: dict[int, dict] = {}

    for choice in choices:
        _strip_malformed_composite_tool_tail(choice, states)

    assert _reasoning_from_choices(choices) == visible


def test_composite_boundary_flushes_longest_partial_prefix_on_mismatch():
    output = "Reasoning.\n</parameterX> is a literal string."
    choices = [
        {
            "index": 0,
            "delta": {"reasoning_content": "Reasoning.\n</param"},
        },
        {
            "index": 0,
            "delta": {"reasoning_content": "eterX> is a literal string."},
            "finish_reason": "stop",
        },
    ]
    states: dict[int, dict] = {}

    for choice in choices:
        _strip_malformed_composite_tool_tail(choice, states)

    assert _reasoning_from_choices(choices) == output


@pytest.mark.parametrize("terminal", ("", "<|im_end|>"))
def test_composite_boundary_flushes_complete_tail_before_visible_text(
    terminal,
):
    output = f"Reasoning.{_ORPHAN_COMPOSITE_TOOL_TAIL}{terminal} is literal text."
    choices = [
        {
            "index": 0,
            "delta": {
                "reasoning_content": (
                    f"Reasoning.{_ORPHAN_COMPOSITE_TOOL_TAIL}{terminal}"
                )
            },
        },
        {
            "index": 0,
            "delta": {"reasoning_content": " is literal text."},
            "finish_reason": "stop",
        },
    ]
    states: dict[int, dict] = {}

    for choice in choices:
        _strip_malformed_composite_tool_tail(choice, states)

    assert _reasoning_from_choices(choices) == output


def test_composite_boundary_flushes_held_reasoning_before_content():
    states: dict[int, dict] = {}
    held = {
        "index": 0,
        "delta": {"reasoning_content": _ORPHAN_COMPOSITE_TOOL_TAIL},
    }
    content = {"index": 0, "delta": {"content": "visible content"}}

    _strip_malformed_composite_tool_tail(held, states)
    _strip_malformed_composite_tool_tail(content, states)

    assert content["delta"] == {
        "reasoning_content": _ORPHAN_COMPOSITE_TOOL_TAIL,
        "content": "visible content",
    }


def test_composite_boundary_flushes_partial_tail_at_finish():
    choice = {
        "index": 0,
        "delta": {"reasoning_content": "\n</parameter>"},
        "finish_reason": "stop",
    }

    _strip_malformed_composite_tool_tail(choice, {})

    assert choice["delta"]["reasoning_content"] == "\n</parameter>"


def test_composite_boundary_flushes_partial_tail_at_empty_finish():
    states: dict[int, dict] = {}
    held = {"index": 0, "delta": {"reasoning_content": "\n</param"}}
    finish = {"index": 0, "delta": {}, "finish_reason": "stop"}

    _strip_malformed_composite_tool_tail(held, states)
    _strip_malformed_composite_tool_tail(finish, states)

    assert not held["delta"]
    assert finish["delta"]["reasoning_content"] == "\n</param"


def test_composite_boundary_strips_complete_tail_at_finish():
    states: dict[int, dict] = {}
    choices = [
        {"index": 0, "delta": {"reasoning_content": "\n</parameter>"}},
        {"index": 0, "delta": {"reasoning_content": "\n</function>"}},
        {
            "index": 0,
            "delta": {"reasoning_content": "\n</tool_call>"},
            "finish_reason": "stop",
        },
    ]

    for choice in choices:
        _strip_malformed_composite_tool_tail(choice, states)

    assert all(not choice["delta"] for choice in choices)


@pytest.mark.parametrize("terminal", ("", "<|im_end|>"))
def test_composite_boundary_strips_coalesced_tail_on_same_terminal_chunk(
    terminal,
):
    reasoning = "Let me test with the axis directly:"
    choice = {
        "index": 0,
        "delta": {
            "reasoning_content": (
                f"{reasoning}\n</parameter>\n</function>\n</tool_call>{terminal}"
            )
        },
        "finish_reason": "stop",
    }

    _strip_malformed_composite_tool_tail(choice, {})

    assert choice["delta"]["reasoning_content"] == reasoning


def test_composite_boundary_buffers_nonterminal_tail_until_continuation():
    reasoning = f"Literal example:{_ORPHAN_COMPOSITE_TOOL_TAIL}<|im_end|>"
    held = {
        "index": 0,
        "delta": {"reasoning_content": reasoning},
    }
    continuation = {
        "index": 0,
        "delta": {"reasoning_content": " is visible."},
        "finish_reason": "stop",
    }
    states: dict[int, dict] = {}

    _strip_malformed_composite_tool_tail(held, states)
    _strip_malformed_composite_tool_tail(continuation, states)

    assert _reasoning_from_choices([held, continuation]) == f"{reasoning} is visible."


def test_composite_boundary_preserves_mismatched_coalesced_terminal_tail():
    reasoning = (
        "Malformed example:\n"
        "</parameter>\n</function>\n</tool_calls><|im_end|>"
    )
    choice = {
        "index": 0,
        "delta": {"reasoning_content": reasoning},
        "finish_reason": "stop",
    }

    _strip_malformed_composite_tool_tail(choice, {})

    assert choice["delta"]["reasoning_content"] == reasoning


def test_composite_boundary_preserves_visible_content():
    choice = {
        "index": 0,
        "delta": {"content": "The literal string <tool_call> is in the file."},
    }

    _strip_malformed_composite_tool_tail(choice, {})

    assert choice["delta"]["content"] == (
        "The literal string <tool_call> is in the file."
    )


def test_vllm_composite_parser_owns_reasoning_tool_transition():
    class _CompositeParser:
        def __init__(self):
            self.calls = []

        def parse_delta(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["delta_text"] == "reason":
                return DeltaMessage(reasoning="Need the file.")
            if kwargs["delta_text"] == "tool":
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=0,
                            id="call-1",
                            type="function",
                            function=DeltaFunctionCall(
                                name="Read",
                                arguments='{"path":"package.json"}',
                            ),
                        )
                    ]
                )
            return None

    parser = _CompositeParser()
    post = StreamingPostProcessor(
        tokenizer=_RawToolTokenizer(),
        request_for_sampling=SimpleNamespace(
            tool_choice="auto",
            tools=TOOL_REQUEST["tools"],
        ),
        sampling_params=SamplingParams(max_tokens=128),
        prompt_token_ids=[101, 102],
        tool_parser=_FallbackToolParser(),
        reasoning_parser_class=_FallbackReasoningParser,
        chat_template_kwargs={"enable_thinking": True},
        composite_parser=parser,
    )
    chunks = [
        SimpleNamespace(
            index=0,
            token_ids=[1],
            text="reason",
            finish_reason=None,
            logprobs=None,
        ),
        SimpleNamespace(
            index=0,
            token_ids=[14],
            text="tool",
            finish_reason=None,
            logprobs=None,
        ),
        SimpleNamespace(
            index=0,
            token_ids=[11],
            text="",
            finish_reason="stop",
            logprobs=None,
        ),
    ]

    choices = [choice for chunk in chunks if (choice := post.process_output(chunk))]

    assert post.uses_vllm_composite_parser
    assert choices[0]["delta"]["reasoning_content"] == "Need the file."
    assert "reasoning" not in choices[0]["delta"]
    assert choices[1]["delta"]["tool_calls"][0]["function"]["name"] == "Read"
    assert choices[-1]["finish_reason"] == "tool_calls"
    assert [call["finished"] for call in parser.calls] == [False, False, True]
    assert all(call["prompt_token_ids"] == [101, 102] for call in parser.calls)


def test_composite_parser_adjusts_request_before_sampling():
    class _CompositeParser:
        reasoning_parser_cls = object

        def __init__(self, tokenizer, tools, **kwargs):
            self.tokenizer = tokenizer
            self.tools = tools
            self.kwargs = kwargs

        def adjust_request(self, request):
            request.skip_special_tokens = False
            return request

    processor = object.__new__(VllmProcessor)
    processor.composite_parser_class = _CompositeParser
    processor.tokenizer = object()
    processor.model_config = object()
    request = SimpleNamespace(
        tool_choice="none",
        tools=None,
        structured_outputs=None,
        model_extra=None,
        response_format=None,
        skip_special_tokens=True,
    )

    adjusted = processor._adjust_request_for_composite_parser(
        request,
        {"enable_thinking": True},
    )

    assert adjusted.skip_special_tokens is False


def test_composite_parser_preserves_generated_structural_tag():
    class _CompositeParser:
        reasoning_parser_cls = object

        def __init__(self, tokenizer, tools, **kwargs):
            pass

        def adjust_request(self, request):
            request.structured_outputs = {"structural_tag": "generated"}
            return request

    processor = object.__new__(VllmProcessor)
    processor.composite_parser_class = _CompositeParser
    processor.tokenizer = object()
    processor.model_config = object()
    request = SimpleNamespace(
        tool_choice="auto",
        tools=TOOL_REQUEST["tools"],
        structured_outputs=None,
        model_extra=None,
        response_format=None,
    )
    enabled = processor._composite_parser_enabled_for_request(request)

    adjusted = processor._adjust_request_for_composite_parser(
        request,
        {},
        enabled=enabled,
    )
    streaming_parser = processor._new_composite_parser(
        adjusted,
        {},
        enabled=enabled,
    )

    assert adjusted.structured_outputs == {"structural_tag": "generated"}
    assert streaming_parser is not None


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


class TestVllmProcessorParserTextBridge:
    def test_parser_visible_engine_text_fills_empty_vllm_delta(self):
        output = SimpleNamespace(
            index=0,
            token_ids=[14],
            text="",
            finish_reason=None,
            logprobs=None,
        )

        bridged = _with_parser_visible_engine_text(output, "<tool_call>")

        assert bridged.text == "<tool_call>"
        assert bridged.token_ids == [14]

    def test_parser_visible_engine_text_does_not_replace_visible_content(self):
        output = SimpleNamespace(
            index=0,
            token_ids=[42],
            text="hello",
            finish_reason=None,
            logprobs=None,
        )

        assert _with_parser_visible_engine_text(output, "<tool_call>") is output


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


class _EngineAdapterToolParser(_FallbackToolParser):
    tool_call_start_token = None
    tool_call_end_token = None
    tool_call_start_token_id = None
    tool_call_end_token_id = None

    def __init__(self):
        self._parser_engine = SimpleNamespace(
            _tool_call_token_id=14,
            _tool_call_end_token_id=15,
        )

    def extract_tool_calls(self, text, request):
        called = "<tool_call>" in text and "</tool_call>" in text
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


class _OpaqueToolParser(_EngineAdapterToolParser):
    def __init__(self):
        self._parser_engine = SimpleNamespace()


class _ZeroValuedFinishReason(IntEnum):
    STOP = 0


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

    def encode(self, text, add_special_tokens=False):
        token_ids = {
            "<tool_call>": [14],
            "</tool_call>": [15],
        }
        return token_ids.get(text, [20, 21])

    def convert_ids_to_tokens(self, token_ids):
        return [self._token_text.get(token_id, "") for token_id in token_ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


class TestStreamingPostProcessorStructuredJson:
    def test_pure_structured_json_emits_complete_value_at_finish(self):
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
                text="}",
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

        assert len(choices) == 1
        assert choices[0]["delta"]["content"] == '{"a":1}'
        assert choices[-1]["finish_reason"] == "stop"

    def test_pure_structured_json_does_not_complete_number_prefix_early(self):
        post = StreamingPostProcessor(
            tokenizer=SimpleNamespace(all_special_tokens=()),
            request_for_sampling=SimpleNamespace(
                tool_choice="auto",
                tools=[],
                structured_outputs={"json": {"type": "number"}},
                response_format=None,
            ),
            sampling_params=SamplingParams(max_tokens=128),
            prompt_token_ids=[],
            tool_parser=_FallbackToolParser(),
            reasoning_parser_class=None,
            chat_template_kwargs={"enable_thinking": False},
        )

        first = post.process_output(
            SimpleNamespace(
                index=0,
                token_ids=[1],
                text="1",
                finish_reason=None,
                logprobs=None,
            )
        )
        final = post.process_output(
            SimpleNamespace(
                index=0,
                token_ids=[2],
                text="23",
                finish_reason="stop",
                logprobs=None,
            )
        )

        assert first is None
        assert final["delta"]["content"] == "123"


class TestStreamingPostProcessorToolFallback:
    def test_tool_enabled_request_filters_markup_without_frontend_parser(self):
        post = StreamingPostProcessor(
            tokenizer=_RawToolTokenizer(),
            request_for_sampling=SimpleNamespace(
                tool_choice="auto",
                tools=[{"type": "function"}],
                structured_outputs=None,
                response_format=None,
            ),
            sampling_params=SamplingParams(max_tokens=128),
            prompt_token_ids=[],
            tool_parser=None,
            reasoning_parser_class=_FallbackReasoningParser,
            chat_template_kwargs={"enable_thinking": True},
        )
        delta = {
            "reasoning_content": (
                "Need the file.<tool_call><function=Read>"
            )
        }

        post._strip_tool_markup_from_delta(delta)

        assert delta["reasoning_content"] == "Need the file."

    def test_forced_final_guard_filters_after_tool_metadata_is_erased(self):
        post = StreamingPostProcessor(
            tokenizer=_RawToolTokenizer(),
            request_for_sampling=SimpleNamespace(tool_choice="auto"),
            sampling_params=SamplingParams(max_tokens=128),
            prompt_token_ids=[],
            tool_parser=None,
            reasoning_parser_class=_FallbackReasoningParser,
            chat_template_kwargs={"enable_thinking": True},
        )
        first = {
            "reasoning_content": (
                "Need the file.<tool_call><function=Read>"
            )
        }
        continuation = {
            "reasoning_content": "<parameter=path>package.json</parameter>"
        }

        post._strip_tool_markup_from_delta(first, force=True)
        post._strip_tool_markup_from_delta(continuation, force=True)

        assert first["reasoning_content"] == "Need the file."
        assert "reasoning_content" not in continuation

    def test_qwen3_marker_fallback_without_parser_metadata(self):
        post = StreamingPostProcessor(
            tokenizer=_RawToolTokenizer(),
            request_for_sampling=SimpleNamespace(tool_choice="auto"),
            sampling_params=SamplingParams(max_tokens=128),
            prompt_token_ids=[],
            tool_parser=_OpaqueToolParser(),
            reasoning_parser_class=_FallbackReasoningParser,
            chat_template_kwargs={"enable_thinking": True},
        )

        assert post._tool_call_start_token() == "<tool_call>"
        assert post._tool_call_end_token() == "</tool_call>"

    def test_tool_markup_is_defensively_removed_from_reasoning(self):
        post = StreamingPostProcessor(
            tokenizer=_RawToolTokenizer(),
            request_for_sampling=SimpleNamespace(tool_choice="auto"),
            sampling_params=SamplingParams(max_tokens=128),
            prompt_token_ids=[],
            tool_parser=_EngineAdapterToolParser(),
            reasoning_parser_class=_FallbackReasoningParser,
            chat_template_kwargs={"enable_thinking": True},
        )
        first = DeltaMessage(
            reasoning="Need the file.<tool_call><function=Read>"
        )
        continuation = DeltaMessage(reasoning="<parameter=path>package.json")

        post._strip_tool_markup_from_reasoning(first)
        post._strip_tool_markup_from_reasoning(continuation)

        assert first.reasoning == "Need the file."
        assert continuation.reasoning is None

    def test_tool_markup_is_removed_at_choice_boundary(self):
        post = StreamingPostProcessor(
            tokenizer=_RawToolTokenizer(),
            request_for_sampling=SimpleNamespace(tool_choice="auto"),
            sampling_params=SamplingParams(max_tokens=128),
            prompt_token_ids=[],
            tool_parser=_EngineAdapterToolParser(),
            reasoning_parser_class=_FallbackReasoningParser,
            chat_template_kwargs={"enable_thinking": True},
        )
        output = SimpleNamespace(
            index=0,
            finish_reason=None,
            logprobs=None,
        )

        first = post._build_choice(
            output,
            {
                "reasoning_content": (
                    "Need the file.<tool_call><function=Read>"
                ),
                "reasoning": "duplicate reasoning",
                "content": "duplicate content",
            },
        )
        continuation = post._build_choice(
            output,
            {"reasoning_content": "<parameter=path>package.json"},
        )

        assert first["delta"]["reasoning_content"] == "Need the file."
        assert "reasoning" not in first["delta"]
        assert "content" not in first["delta"]
        assert "reasoning_content" not in continuation["delta"]

    def test_tool_call_without_reasoning_end_does_not_leak_markup(self):
        post = StreamingPostProcessor(
            tokenizer=_RawToolTokenizer(),
            request_for_sampling=SimpleNamespace(tool_choice="auto"),
            sampling_params=SamplingParams(max_tokens=128),
            prompt_token_ids=[],
            tool_parser=_EngineAdapterToolParser(),
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
                token_ids=[14, 20],
                text="<tool_call>\n<function=Read>\n",
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[21, 15],
                text=(
                    "<parameter=path>package.json</parameter>\n"
                    "</function>\n</tool_call>"
                ),
                finish_reason="stop",
                logprobs=None,
            ),
        ]

        choices = [choice for chunk in chunks if (choice := post.process_output(chunk))]
        serialized = json.dumps(choices)
        reasoning = "".join(
            choice["delta"].get("reasoning_content", "") for choice in choices
        )
        tool_choice = next(
            choice for choice in choices if "tool_calls" in choice["delta"]
        )

        assert reasoning == "Need the file."
        assert "<tool_call>" not in serialized
        assert "<parameter=" not in serialized
        assert tool_choice["finish_reason"] == "tool_calls"
        assert tool_choice["delta"]["tool_calls"][0]["function"]["name"] == "Read"

    def test_zero_valued_stop_still_recovers_terminal_tool_call(self):
        post = StreamingPostProcessor(
            tokenizer=_RawToolTokenizer(),
            request_for_sampling=SimpleNamespace(tool_choice="auto"),
            sampling_params=SamplingParams(max_tokens=128),
            prompt_token_ids=[],
            tool_parser=_EngineAdapterToolParser(),
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
                token_ids=[14, 20, 21, 15],
                text=(
                    "<tool_call>\n<function=Read>\n"
                    "<parameter=path>package.json</parameter>\n"
                    "</function>\n</tool_call>"
                ),
                finish_reason=_ZeroValuedFinishReason.STOP,
                logprobs=None,
            ),
        ]

        choices = [choice for chunk in chunks if (choice := post.process_output(chunk))]
        tool_choice = next(
            choice for choice in choices if "tool_calls" in choice["delta"]
        )

        assert tool_choice["finish_reason"] == "tool_calls"
        assert tool_choice["delta"]["tool_calls"][0]["function"]["name"] == "Read"

    def test_complete_buffered_tool_call_emits_on_finish_chunk(self):
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

        assert len(choices) == 2
        assert choices[1]["delta"]["tool_calls"][0]["function"]["name"] == "Read"
        assert "ignored continuation" not in json.dumps(choices)
        assert choices[-1]["finish_reason"] == "tool_calls"

    def test_split_raw_tool_body_with_marker_text_only(self):
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
                token_ids=[14],
                text="<tool_call>",
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[20],
                text="",
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[21],
                text="",
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[15],
                text="</tool_call>",
                finish_reason=None,
                logprobs=None,
            ),
            SimpleNamespace(
                index=0,
                token_ids=[99],
                text="",
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
        tool_calls = tool_choice["delta"]["tool_calls"]
        assert tool_calls[0]["function"]["name"] == "Read"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {
            "path": "package.json"
        }

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
