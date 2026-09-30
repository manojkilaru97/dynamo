# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from dynamo.vllm.handlers import _map_request_validation_errors
from vllm.exceptions import VLLMValidationError
from vllm.v1.engine.exceptions import EngineGenerateError

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


async def _collect(stream):
    return [item async for item in _map_request_validation_errors(stream)]


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
