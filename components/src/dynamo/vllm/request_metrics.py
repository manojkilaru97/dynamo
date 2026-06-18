# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from prometheus_client import Counter

from .publisher import DYNAMO_COMPONENT_REGISTRY


def _counter(name: str, doc: str, labels: tuple[str, ...] = ()) -> Counter:
    return Counter(
        name=name,
        documentation=doc,
        labelnames=labels,
        registry=DYNAMO_COMPONENT_REGISTRY,
    )


request_type_image = _counter("request_type_image_total", "Requests containing images")
request_type_video = _counter("request_type_video_total", "Requests containing videos")
request_type_audio = _counter("request_type_audio_total", "Requests containing audio")
request_type_tool_call = _counter("request_type_tool_call_total", "Requests with tools enabled")
request_type_structured_output = _counter(
    "request_type_structured_output_total", "Requests with structured outputs enabled"
)
request_input_images = _counter("request_input_images_total", "Input image count")
request_input_videos = _counter("request_input_videos_total", "Input video count")
request_input_audios = _counter("request_input_audios_total", "Input audio count")
request_input_tools = _counter("request_input_tools_total", "Input tool count")
request_mode_total = _counter("request_mode_total", "Requests by handler mode", ("mode",))
request_tool_choice_total = _counter(
    "request_tool_choice_total", "Requests by tool choice", ("tool_choice",)
)
request_structured_output_kind_total = _counter(
    "request_structured_output_kind_total",
    "Requests by structured output kind",
    ("kind",),
)
request_structured_output_backend_total = _counter(
    "request_structured_output_backend_total",
    "Requests by structured output backend",
    ("backend",),
)
request_outcome_total = _counter(
    "request_outcome_total", "Requests by terminal outcome", ("outcome",)
)
request_finish_reason_total = _counter(
    "request_finish_reason_total", "Requests by finish reason", ("finish_reason",)
)
request_failure_total = _counter(
    "request_failure_total", "Request failures by class", ("failure_type",)
)


@dataclass
class RequestMetricsContext:
    mode: str
    tool_choice: str
    structured_output_kind: str
    image_count: int
    video_count: int
    audio_count: int
    tool_count: int
    structured_output_backend: str = "none"
    backend_recorded: bool = False
    terminal_recorded: bool = False
    _terminal_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def claim_terminal(self) -> bool:
        with self._terminal_lock:
            if self.terminal_recorded:
                return False
            self.terminal_recorded = True
            return True


def _coerce_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    for attr in ("model_dump", "dict"):
        if hasattr(value, attr):
            try:
                dumped = getattr(value, attr)()
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                return dumped
    return None


def _count_message_modalities(messages: Any) -> tuple[int, int, int]:
    counts = [0, 0, 0]
    if not isinstance(messages, list):
        return 0, 0, 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "")
            if part_type in ("image_url", "input_image"):
                counts[0] += 1
            elif part_type == "video_url":
                counts[1] += 1
            elif part_type in ("audio_url", "input_audio"):
                counts[2] += 1
    return counts[0], counts[1], counts[2]


def _count_internal_modalities(request: dict[str, Any]) -> tuple[int, int, int]:
    mm_data = _coerce_dict(request.get("multi_modal_data")) or {}

    def _count(value: Any) -> int:
        if value is None:
            return 0
        return len(value) if isinstance(value, list) else 1

    return (
        _count(mm_data.get("image") or mm_data.get("image_url")),
        _count(mm_data.get("video") or mm_data.get("video_url")),
        _count(mm_data.get("audio") or mm_data.get("audio_url")),
    )


def _normalize_tool_choice(tool_choice: Any, tools: Any = None) -> str:
    if tool_choice is None:
        return "auto" if isinstance(tools, list) and tools else "none"
    if isinstance(tool_choice, str):
        return tool_choice
    choice_dict = _coerce_dict(tool_choice)
    if choice_dict is None:
        return "object"
    function_dict = _coerce_dict(choice_dict.get("function")) or {}
    if function_dict.get("name"):
        return "named_function"
    return str(choice_dict.get("type") or "object")


def _detect_structured_output_kind(
    request: dict[str, Any],
    request_for_sampling: dict[str, Any] | None = None,
) -> str:
    sampling_options = _coerce_dict(request.get("sampling_options")) or {}
    guided_decoding = _coerce_dict(sampling_options.get("guided_decoding"))
    if guided_decoding:
        for key in ("structural_tag", "json", "json_object", "regex", "choice", "grammar"):
            if guided_decoding.get(key) not in (None, False):
                return key
    request_for_sampling = request_for_sampling or {}
    response_format = _coerce_dict(request.get("response_format"))
    if response_format and response_format.get("type"):
        return str(response_format["type"])
    structured_outputs = _coerce_dict(
        request_for_sampling.get("structured_outputs")
    ) or _coerce_dict(request.get("structured_outputs"))
    if structured_outputs:
        for key in ("structural_tag", "json", "json_object", "regex", "choice", "grammar"):
            if structured_outputs.get(key) not in (None, False):
                return key
        return "structured_outputs"
    return "none"


def create_request_metrics_context(
    request: dict[str, Any],
    *,
    mode: str,
) -> RequestMetricsContext:
    extra_args = _coerce_dict(request.get("extra_args")) or {}
    request_for_sampling = (
        _coerce_dict(request.get("request_for_sampling"))
        or _coerce_dict(extra_args.get("request_for_sampling"))
        or {}
    )
    tools = request.get("tools", request_for_sampling.get("tools"))
    tool_choice = request.get("tool_choice", request_for_sampling.get("tool_choice"))
    image_count, video_count, audio_count = _count_internal_modalities(request)
    if image_count == 0 and video_count == 0 and audio_count == 0:
        image_count, video_count, audio_count = _count_message_modalities(
            request.get("messages")
        )
    tool_count = len(tools) if isinstance(tools, list) else 0
    return RequestMetricsContext(
        mode=mode,
        tool_choice=_normalize_tool_choice(tool_choice, tools),
        structured_output_kind=_detect_structured_output_kind(request, request_for_sampling),
        image_count=image_count,
        video_count=video_count,
        audio_count=audio_count,
        tool_count=tool_count,
    )


def record_request_start(context: RequestMetricsContext) -> None:
    request_mode_total.labels(mode=context.mode).inc()
    if context.image_count:
        request_type_image.inc()
        request_input_images.inc(context.image_count)
    if context.video_count:
        request_type_video.inc()
        request_input_videos.inc(context.video_count)
    if context.audio_count:
        request_type_audio.inc()
        request_input_audios.inc(context.audio_count)
    if context.tool_count:
        request_type_tool_call.inc()
        request_input_tools.inc(context.tool_count)
        request_tool_choice_total.labels(tool_choice=context.tool_choice).inc()
    if context.structured_output_kind != "none":
        request_type_structured_output.inc()
        request_structured_output_kind_total.labels(
            kind=context.structured_output_kind
        ).inc()


def record_structured_output_backend(context: RequestMetricsContext, sampling_params: Any) -> None:
    if context.backend_recorded or context.structured_output_kind == "none":
        return
    backend = "default"
    structured_outputs = getattr(sampling_params, "structured_outputs", None)
    if structured_outputs is not None:
        backend = str(getattr(structured_outputs, "_backend", None) or "default")
    request_structured_output_backend_total.labels(backend=backend).inc()
    context.backend_recorded = True


def _finish_reason_label(finish_reason: str | None) -> str:
    if not finish_reason:
        return "unknown"
    finish_reason = str(finish_reason)
    return "error" if finish_reason.startswith("error") else finish_reason


def record_request_success(context: RequestMetricsContext, *, finish_reason: str | None) -> None:
    if not context.claim_terminal():
        return
    request_outcome_total.labels(outcome="success").inc()
    request_finish_reason_total.labels(
        finish_reason=_finish_reason_label(finish_reason)
    ).inc()


def record_request_failure(
    context: RequestMetricsContext,
    *,
    failure_type: str,
    finish_reason: str | None = None,
) -> None:
    if not context.claim_terminal():
        return
    request_outcome_total.labels(outcome="failure").inc()
    request_failure_total.labels(failure_type=failure_type).inc()
    request_finish_reason_total.labels(
        finish_reason=_finish_reason_label(finish_reason)
    ).inc()
