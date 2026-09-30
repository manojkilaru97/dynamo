# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.exceptions import VLLMValidationError
from vllm.v1.engine.exceptions import EngineGenerateError

from dynamo.vllm.handlers import _map_request_validation_errors

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


def _output(finish_reason, token_ids=(), finished=True):
    return SimpleNamespace(
        finished=finished,
        outputs=[
            SimpleNamespace(finish_reason=finish_reason, token_ids=list(token_ids))
        ],
    )


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
async def test_structured_output_compile_failure_is_a_value_error():
    with pytest.raises(ValueError, match="could not be compiled"):
        await _collect(_outputs(_output("error")), structured_output=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "items, structured_output",
    [
        pytest.param([_output("error")], False, id="not_structured"),
        pytest.param([_output(None, [1], False), _output("error")], True, id="late"),
        pytest.param([_output("stop", [1])], True, id="normal"),
    ],
)
async def test_other_error_finishes_pass_through(items, structured_output):
    assert len(await _collect(_outputs(*items), structured_output)) == len(items)
