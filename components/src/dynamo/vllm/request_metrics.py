# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from prometheus_client import Counter
from .publisher import DYNAMO_COMPONENT_REGISTRY

request_type_image = Counter(
    name="request_type_image_total",
    documentation="Total Dynamo requests containing images",
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_type_video = Counter(
    name="request_type_video_total",
    documentation="Total Dynamo requests containing videos",
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_type_tool_call = Counter(
    name="request_type_tool_call_total",
    documentation="Total Dynamo requests with tool calls enabled",
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_type_structured_output = Counter(
    name="request_type_structured_output_total",
    documentation="Total Dynamo requests with structured outputs enabled",
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_type_prompt_embeds = Counter(
    name="request_type_prompt_embeds_total",
    documentation="Total Dynamo requests using prompt embeddings",
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_type_lora = Counter(
    name="request_type_lora_total",
    documentation="Total Dynamo requests resolved to a LoRA adapter",
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_type_streaming = Counter(
    name="request_type_streaming_total",
    documentation="Total Dynamo requests using streaming responses",
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_type_prefill = Counter(
    name="request_type_prefill_total",
    documentation="Total Dynamo requests handled by the prefill worker path",
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_mode_total = Counter(
    name="request_mode_total",
    documentation="Total Dynamo requests by handler mode",
    labelnames=("mode",),
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_tool_choice_total = Counter(
    name="request_tool_choice_total",
    documentation="Total Dynamo requests by tool choice",
    labelnames=("tool_choice",),
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_structured_output_kind_total = Counter(
    name="request_structured_output_kind_total",
    documentation="Total Dynamo requests by structured output kind",
    labelnames=("kind",),
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_structured_output_backend_total = Counter(
    name="request_structured_output_backend_total",
    documentation="Total Dynamo requests by structured output backend",
    labelnames=("backend",),
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_outcome_total = Counter(
    name="request_outcome_total",
    documentation="Total Dynamo requests by terminal outcome",
    labelnames=("outcome",),
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_finish_reason_total = Counter(
    name="request_finish_reason_total",
    documentation="Total Dynamo requests by normalized finish reason",
    labelnames=("finish_reason",),
    registry=DYNAMO_COMPONENT_REGISTRY,
)
request_failure_total = Counter(
    name="request_failure_total",
    documentation="Total Dynamo request failures by class",
    labelnames=("failure_type",),
    registry=DYNAMO_COMPONENT_REGISTRY,
)


@dataclass
class RequestMetricsContext:
    mode: str
    tool_choice: str
    structured_output_kind: str
    has_image: bool
    has_video: bool
    has_prompt_embeds: bool
    has_lora: bool
    is_streaming: bool
    is_prefill: bool
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
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "dict"):
        try:
            dumped = value.dict()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "__dict__"):
        try:
            dumped = {k: v for k, v in vars(value).items() if not k.startswith("_")}
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            return dumped
    return None


def _extract_modalities_from_messages(messages: Any) -> tuple[bool, bool]:
    has_image = False
    has_video = False
    if not isinstance(messages, list):
        return has_image, has_video
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
            if part_type == "image_url":
                has_image = True
            elif part_type == "video_url":
                has_video = True
    return has_image, has_video


def _extract_modalities_from_internal_request(request: dict[str, Any]) -> tuple[bool, bool]:
    mm_data = _coerce_dict(request.get("multi_modal_data")) or {}
    has_image = bool(mm_data.get("image") or mm_data.get("image_url"))
    has_video = bool(mm_data.get("video") or mm_data.get("video_url"))
    return has_image, has_video


def _normalize_tool_choice(tool_choice: Any, tools: Any = None) -> str:
    if tool_choice is None:
        return "auto" if isinstance(tools, list) and len(tools) > 0 else "none"
    if isinstance(tool_choice, str):
        return tool_choice
    choice_dict = _coerce_dict(tool_choice)
    if choice_dict is None:
        return "object"
    function_dict = _coerce_dict(choice_dict.get("function")) or {}
    if function_dict.get("name"):
        return "named_function"
    return str(choice_dict.get("type") or "object")


def _detect_structured_output_kind(request: dict[str, Any], request_for_sampling: dict[str, Any] | None = None) -> str:
    sampling_options = _coerce_dict(request.get("sampling_options")) or {}
    guided_decoding = _coerce_dict(sampling_options.get("guided_decoding"))
    if guided_decoding:
        for key in (
            "structural_tag",
            "json",
            "json_object",
            "regex",
            "choice",
            "grammar",
        ):
            value = guided_decoding.get(key)
            if value not in (None, False):
                return key
    request_for_sampling = request_for_sampling or {}
    response_format = _coerce_dict(request.get("response_format"))
    if response_format and response_format.get("type"):
        return str(response_format["type"])
    structured_outputs = _coerce_dict(request_for_sampling.get("structured_outputs")) or _coerce_dict(request.get("structured_outputs"))
    if structured_outputs:
        for key in (
            "structural_tag",
            "json",
            "json_object",
            "regex",
            "choice",
            "grammar",
        ):
            if structured_outputs.get(key) not in (None, False):
                return key
        return "structured_outputs"
    return "none"


def _normalize_backend_name(backend: Any) -> str:
    if backend in (None, "", False):
        return "default"
    return str(backend)


def _normalize_finish_reason_label(finish_reason: str | None) -> str:
    if not finish_reason:
        return "unknown"
    finish_reason = str(finish_reason)
    if finish_reason.startswith("error"):
        return "error"
    return finish_reason


def create_request_metrics_context(
    request: dict[str, Any],
    *,
    mode: str,
    is_prefill: bool = False,
    lora_request: Any = None,
) -> RequestMetricsContext:
    request_for_sampling = _coerce_dict(request.get("request_for_sampling")) or {}
    tools = request.get("tools", request_for_sampling.get("tools"))
    tool_choice = request.get("tool_choice", request_for_sampling.get("tool_choice"))
    normalized_tool_choice = _normalize_tool_choice(tool_choice, tools)

    has_image, has_video = _extract_modalities_from_internal_request(request)
    if not has_image and not has_video:
        has_image, has_video = _extract_modalities_from_messages(request.get("messages"))

    return RequestMetricsContext(
        mode=mode,
        tool_choice=normalized_tool_choice,
        structured_output_kind=_detect_structured_output_kind(request, request_for_sampling),
        has_image=has_image,
        has_video=has_video,
        has_prompt_embeds=bool(request.get("prompt_embeds")),
        has_lora=lora_request is not None,
        is_streaming=bool(request.get("stream", False)),
        is_prefill=is_prefill,
    )


def record_request_start(context: RequestMetricsContext) -> None:
    request_mode_total.labels(mode=context.mode).inc()
    if context.is_prefill:
        request_type_prefill.inc()
    if context.is_streaming:
        request_type_streaming.inc()
    if context.has_image:
        request_type_image.inc()
    if context.has_video:
        request_type_video.inc()
    if context.has_prompt_embeds:
        request_type_prompt_embeds.inc()
    if context.has_lora:
        request_type_lora.inc()
    if context.tool_choice != "none":
        request_type_tool_call.inc()
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
        backend = _normalize_backend_name(getattr(structured_outputs, "_backend", None))
    context.structured_output_backend = backend
    request_structured_output_backend_total.labels(backend=backend).inc()
    context.backend_recorded = True


def record_request_success(
    context: RequestMetricsContext,
    *,
    finish_reason: str | None,
) -> None:
    normalized_finish_reason = _normalize_finish_reason_label(finish_reason)
    outcome = "cancelled" if normalized_finish_reason == "cancelled" else "success"
    if not context.claim_terminal():
        return
    request_outcome_total.labels(outcome=outcome).inc()
    request_finish_reason_total.labels(finish_reason=normalized_finish_reason).inc()


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
        finish_reason=_normalize_finish_reason_label(finish_reason)
    ).inc()


def record_request_cancelled(
    context: RequestMetricsContext,
    *,
    reason: str,
) -> None:
    if not context.claim_terminal():
        return
    request_outcome_total.labels(outcome="cancelled").inc()
    request_failure_total.labels(failure_type=reason).inc()
    request_finish_reason_total.labels(finish_reason="cancelled").inc()
