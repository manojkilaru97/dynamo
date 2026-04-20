# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from prometheus_client import generate_latest
from dynamo.vllm.publisher import DYNAMO_COMPONENT_REGISTRY
from vllm.sampling_params import StructuredOutputsParams

from dynamo.vllm.request_metrics import (
    create_request_metrics_context,
    record_request_cancelled,
    record_request_failure,
    record_request_start,
    record_structured_output_backend,
    record_request_success,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
]


def _metric_value(metric_name: str) -> float:
    metrics_text = generate_latest(DYNAMO_COMPONENT_REGISTRY).decode("utf-8")
    total = 0.0
    matched = False
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(metric_name):
            continue
        total += float(line.rsplit(" ", 1)[1])
        matched = True
    if not matched:
        return 0.0
    return total


def test_request_metrics_start_captures_request_shape() -> None:
    request = {
        "model": "minimaxai/minimax-m2.7",
        "prompt_embeds": "Zm9v",
        "stream": True,
        "multi_modal_data": {
            "image_url": [{"url": "https://example.com/cat.png"}],
            "video_url": [{"url": "https://example.com/cat.mp4"}],
        },
        "sampling_options": {
            "guided_decoding": {
                "json": {"type": "object"},
            }
        },
        "request_for_sampling": {
            "tool_choice": {"type": "function", "function": {"name": "weather"}},
        },
    }

    before = {
        "request_type_image_total": _metric_value("request_type_image_total"),
        "request_type_video_total": _metric_value("request_type_video_total"),
        "request_type_tool_call_total": _metric_value("request_type_tool_call_total"),
        "request_type_structured_output_total": _metric_value("request_type_structured_output_total"),
        "request_type_prompt_embeds_total": _metric_value("request_type_prompt_embeds_total"),
        "request_type_lora_total": _metric_value("request_type_lora_total"),
        "request_type_streaming_total": _metric_value("request_type_streaming_total"),
        "request_type_prefill_total": _metric_value("request_type_prefill_total"),
        "request_mode_total": _metric_value("request_mode_total"),
        "request_tool_choice_total": _metric_value("request_tool_choice_total"),
        "request_structured_output_kind_total": _metric_value("request_structured_output_kind_total"),
    }

    context = create_request_metrics_context(
        request,
        mode="decode_tokens",
        lora_request=Mock(),
    )
    record_request_start(context)

    assert _metric_value("request_type_image_total") == before["request_type_image_total"] + 1.0
    assert _metric_value("request_type_video_total") == before["request_type_video_total"] + 1.0
    assert _metric_value("request_type_tool_call_total") == before["request_type_tool_call_total"] + 1.0
    assert _metric_value("request_type_structured_output_total") == before["request_type_structured_output_total"] + 1.0
    assert _metric_value("request_type_prompt_embeds_total") == before["request_type_prompt_embeds_total"] + 1.0
    assert _metric_value("request_type_lora_total") == before["request_type_lora_total"] + 1.0
    assert _metric_value("request_type_streaming_total") == before["request_type_streaming_total"] + 1.0
    assert _metric_value("request_type_prefill_total") == before["request_type_prefill_total"]
    assert _metric_value("request_mode_total") == before["request_mode_total"] + 1.0
    assert _metric_value("request_tool_choice_total") == before["request_tool_choice_total"] + 1.0
    assert _metric_value("request_structured_output_kind_total") == before["request_structured_output_kind_total"] + 1.0


def test_request_metrics_uses_request_for_sampling_shape_for_openai_requests() -> None:
    request = {
        "messages": [{"role": "user", "content": "Return JSON and call the tool."}],
        "request_for_sampling": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "weather"}},
            "structured_outputs": {"json": {"type": "object"}},
        },
    }

    context = create_request_metrics_context(request, mode="decode_text")

    assert context.tool_choice == "named_function"
    assert context.structured_output_kind == "json"


def test_request_metrics_backend_counter_uses_sampling_backend() -> None:
    context = create_request_metrics_context(
        {
            "sampling_options": {
                "guided_decoding": {
                    "regex": "yes|no",
                }
            }
        },
        mode="decode_tokens",
    )
    record_request_start(context)

    before_backend = _metric_value("request_structured_output_backend_total")
    sampling_params = Mock()
    sampling_params.structured_outputs = StructuredOutputsParams(regex="yes|no")
    sampling_params.structured_outputs._backend = "guidance"

    record_structured_output_backend(context, sampling_params)
    record_structured_output_backend(context, sampling_params)

    assert _metric_value("request_structured_output_backend_total") == before_backend + 1.0


def test_request_metrics_terminal_success_records_once() -> None:
    context = create_request_metrics_context({}, mode="decode_tokens")
    record_request_start(context)

    before_outcome = _metric_value("request_outcome_total")
    before_finish = _metric_value("request_finish_reason_total")

    record_request_success(context, finish_reason="stop")
    record_request_failure(context, failure_type="no_outputs", finish_reason="error")
    record_request_cancelled(context, reason="client_cancelled")

    assert _metric_value("request_outcome_total") == before_outcome + 1.0
    assert _metric_value("request_finish_reason_total") == before_finish + 1.0


def test_request_metrics_terminal_failure_and_cancel() -> None:
    failure_context = create_request_metrics_context({}, mode="decode_tokens")
    cancel_context = create_request_metrics_context({}, mode="decode_tokens")

    before_failure = _metric_value("request_failure_total")
    before_outcome = _metric_value("request_outcome_total")
    before_finish = _metric_value("request_finish_reason_total")

    record_request_failure(
        failure_context,
        failure_type="request_setup_error",
        finish_reason="error: invalid request",
    )
    record_request_cancelled(cancel_context, reason="client_cancelled")

    assert _metric_value("request_failure_total") == before_failure + 2.0
    assert _metric_value("request_outcome_total") == before_outcome + 2.0
    assert _metric_value("request_finish_reason_total") == before_finish + 2.0
