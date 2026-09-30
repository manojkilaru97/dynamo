# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.exceptions import VLLMValidationError
from vllm.v1.engine.exceptions import EngineGenerateError

from dynamo.vllm.handlers import (
    _map_request_validation_errors,
    build_sampling_params_openai,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _engine_generate_error(cause: Exception) -> EngineGenerateError:
    try:
        raise EngineGenerateError() from cause
    except EngineGenerateError as error:
        return error


async def _stream(error: Exception, outputs: int = 0):
    for index in range(outputs):
        yield index
    raise error


async def _collect(stream, structured_output: bool = False):
    return [
        item async for item in _map_request_validation_errors(stream, structured_output)
    ]


def _output(finish_reason, token_ids=(), finished=True, stop_reason=None):
    return SimpleNamespace(
        finished=finished,
        outputs=[
            SimpleNamespace(
                finish_reason=finish_reason,
                token_ids=list(token_ids),
                stop_reason=stop_reason,
            )
        ],
    )


def _compile_error():
    return _output("error", stop_reason="structured_output_compile_error")


async def _outputs(*items):
    for item in items:
        yield item


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(VLLMValidationError("bad request"), id="client_error"),
        pytest.param(
            _engine_generate_error(ValueError("Grammar error")), id="validation"
        ),
    ],
)
async def test_rejection_before_output_is_a_value_error(error):
    """Maps to InvalidArgument (HTTP 400) in map_python_exception."""
    with pytest.raises(ValueError):
        await _collect(_stream(error))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error, outputs, expected",
    [
        pytest.param(
            _engine_generate_error(ValueError("late")),
            1,
            EngineGenerateError,
            id="validation_after_output",
        ),
        pytest.param(
            VLLMValidationError("late"),
            1,
            VLLMValidationError,
            id="client_after_output",
        ),
        pytest.param(
            _engine_generate_error(RuntimeError("boom")),
            0,
            EngineGenerateError,
            id="server_error",
        ),
    ],
)
async def test_other_failures_stay_server_errors(error, outputs, expected):
    with pytest.raises(expected):
        await _collect(_stream(error, outputs=outputs))


@pytest.mark.asyncio
@pytest.mark.parametrize("finished", [True, False], ids=["n1", "child_of_n2"])
async def test_structured_output_compile_failure_is_a_value_error(finished):
    failure = _compile_error()
    failure.finished = finished
    with pytest.raises(ValueError, match="could not be compiled"):
        await _collect(_outputs(failure), structured_output=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "items, structured_output",
    [
        pytest.param([_compile_error()], False, id="not_structured"),
        pytest.param([_output(None, [1], False), _compile_error()], True, id="late"),
        pytest.param([_output("error")], True, id="other_engine_error"),
        pytest.param([_output("stop", [1])], True, id="normal"),
    ],
)
async def test_other_error_finishes_pass_through(items, structured_output):
    assert len(await _collect(_outputs(*items), structured_output)) == len(items)


@pytest.mark.parametrize(
    "request_fields, expected",
    [
        pytest.param(
            {
                "guided_json": {"type": "object"},
                "guided_whitespace_pattern": "[ ]?",
            },
            {"json_is_set": True, "whitespace_pattern": "[ ]?"},
            id="legacy_json_with_pattern",
        ),
        pytest.param(
            {
                "guided_json": {"type": "object"},
                "structured_outputs": {"whitespace_pattern": "[ ]?"},
            },
            {"json_is_set": True, "whitespace_pattern": "[ ]?"},
            id="legacy_json_with_alias_pattern",
        ),
        pytest.param(
            {"response_format": {"type": "json_object"}},
            {"json_object": True, "whitespace_pattern": None},
            id="json_object",
        ),
    ],
)
def test_text_mode_sampling_params_carry_structured_outputs(request_fields, expected):
    """--use-vllm-tokenizer builds SamplingParams from the OpenAI request."""
    params = build_sampling_params_openai(
        {"max_tokens": 8, **request_fields}, default_sampling_params={}
    ).structured_outputs
    assert params is not None
    if expected.get("json_is_set"):
        assert params.json is not None
    if "json_object" in expected:
        assert params.json_object is expected["json_object"]
    assert params.whitespace_pattern == expected["whitespace_pattern"]


def test_text_mode_without_constraints_has_no_structured_outputs():
    params = build_sampling_params_openai({"max_tokens": 8}, default_sampling_params={})
    assert params.structured_outputs is None


_TOOLS = [
    {
        "type": "function",
        "function": {"name": "record", "parameters": {"type": "object"}},
    }
]


def test_text_mode_forced_tool_takes_precedence_over_output_constraints():
    params = build_sampling_params_openai(
        {
            "max_tokens": 8,
            "tools": _TOOLS,
            "tool_choice": "required",
            "response_format": {"type": "json_object"},
        },
        default_sampling_params={},
    )
    assert params.structured_outputs is None


def test_text_mode_rejects_structural_tag_with_forced_tool():
    with pytest.raises(ValueError, match="structural_tag"):
        build_sampling_params_openai(
            {
                "max_tokens": 8,
                "tools": _TOOLS,
                "tool_choice": {"type": "function", "function": {"name": "record"}},
                "structured_outputs": {
                    "structural_tag": '{"type": "structural_tag", "format": '
                    '{"type": "const_string", "value": "x"}}'
                },
            },
            default_sampling_params={},
        )


def test_text_mode_auto_tools_keep_output_constraints():
    params = build_sampling_params_openai(
        {
            "max_tokens": 8,
            "tools": _TOOLS,
            "response_format": {"type": "json_object"},
        },
        default_sampling_params={},
    )
    assert params.structured_outputs is not None
    assert params.structured_outputs.json_object is True
