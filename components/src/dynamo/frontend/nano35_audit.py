# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default-off, correlation-ID-scoped Nano 3.5 rendered-prompt audit."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

_AUDIT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_LOCK = threading.Lock()
_EFFICIENT_MARKER = "{reasoning effort: efficient}"
_THINKING_SUFFIX = "<|im_start|>assistant\n<think>\n"
_DISABLED_SUFFIX = "<|im_start|>assistant\n<think></think>"


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _audit_id(context: Any | None, request: dict[str, Any]) -> str | None:
    metadata = getattr(context, "metadata", None) if context is not None else None
    value = metadata.get("nano35-audit-id") if metadata is not None else None
    if value is None:
        template_args = request.get("chat_template_args")
        if not isinstance(template_args, dict):
            template_args = request.get("chat_template_kwargs")
        if isinstance(template_args, dict):
            value = template_args.get("__dynamo_nano35_audit_id")
    return value if isinstance(value, str) and _AUDIT_ID_RE.fullmatch(value) else None


def _rendered_prompt(
    engine_prompt: dict[str, Any], prompt_token_ids: list[int], tokenizer: Any
) -> str:
    prompt = engine_prompt.get("prompt")
    if isinstance(prompt, str):
        return prompt
    return tokenizer.decode(prompt_token_ids, skip_special_tokens=False)


def record_nano35_render_audit(
    *,
    request: dict[str, Any],
    context: Any | None,
    response_id: str,
    engine_prompt: dict[str, Any],
    prompt_token_ids: list[int],
    tokenizer: Any,
    chat_template_kwargs: dict[str, Any],
) -> None:
    """Write one redacted JSONL record only for an explicitly tagged request."""

    output = os.getenv("DYN_NANO35_AUDIT_FILE")
    if not output or not (audit_id := _audit_id(context, request)):
        return

    template_path_value = os.getenv("DYN_NANO35_AUDIT_TEMPLATE")
    if not template_path_value:
        raise RuntimeError("DYN_NANO35_AUDIT_TEMPLATE is required when audit is enabled")
    template_path = Path(template_path_value).resolve(strict=True)
    prompt = _rendered_prompt(engine_prompt, prompt_token_ids, tokenizer)
    marker_count = prompt.count(_EFFICIENT_MARKER)
    if prompt.endswith(_THINKING_SUFFIX):
        generation_prefix = "thinking"
        suffix = _THINKING_SUFFIX
    elif prompt.endswith(_DISABLED_SUFFIX):
        generation_prefix = "disabled"
        suffix = _DISABLED_SUFFIX
    else:
        generation_prefix = "unknown"
        suffix = prompt[-96:]

    redacted_suffix = (
        f"<redacted>{_EFFICIENT_MARKER}" if marker_count else "<redacted>"
    )
    record = {
        "audit_id": audit_id,
        "response_id": response_id,
        "wire_reasoning_effort": request.get("reasoning_effort"),
        "enable_thinking": chat_template_kwargs.get("enable_thinking"),
        "medium_effort": chat_template_kwargs.get("medium_effort"),
        "efficient_marker_count": marker_count,
        "template_path": str(template_path),
        "template_sha256": _sha256(template_path.read_bytes()),
        "rendered_prompt_sha256": _sha256(prompt),
        "redacted_final_user_suffix": redacted_suffix,
        "generation_prefix": generation_prefix,
        "generation_suffix_sha256": _sha256(suffix),
    }
    line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, destination.open("a", encoding="utf-8") as audit_file:
        audit_file.write(line)
        audit_file.flush()
