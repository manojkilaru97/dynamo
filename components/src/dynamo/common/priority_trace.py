# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import threading
import time
from typing import Any

_LOGGER = logging.getLogger("dynamo.priority")
_DEFAULT_LIMIT = 200_000
_MAX_LIMIT = 1_000_000
_ALLOWED_FIELDS = frozenset(
    {
        "request_id",
        "processor_request_id",
        "client_request_id",
        "priority_present",
        "priority",
        "priority_jump",
        "strict_priority_present",
        "strict_priority",
        "engine_priority",
        "worker_id",
        "dp_rank",
        "choice_index",
        "choice_count",
        "finish_reason",
        "all_choices_terminated",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "total_tokens_source",
        "cached_tokens_present",
        "cached_tokens",
        "cache_count_source",
        "reasoning_tokens_present",
        "reasoning_tokens",
        "reasoning_count_source",
    }
)
_lock = threading.Lock()
_emitted = 0


def _is_enabled() -> bool:
    return os.getenv("DYN_PRIORITY_TRACE", "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _limit() -> int:
    try:
        configured = int(os.getenv("DYN_PRIORITY_TRACE_LIMIT", _DEFAULT_LIMIT))
    except ValueError:
        configured = _DEFAULT_LIMIT
    return min(max(configured, 0), _MAX_LIMIT)


def emit(stage: str, **fields: Any) -> None:
    """Emit one bounded payload-free scheduling observation."""
    global _emitted
    if not _is_enabled():
        return

    with _lock:
        if _emitted >= _limit():
            return
        _emitted += 1
        sequence = _emitted

    record = {
        "schema": "dynamo.priority.v1",
        "stage": stage,
        "sequence": sequence,
        "monotonic_ns": time.monotonic_ns(),
        "process_id": os.getpid(),
    }
    record.update(
        {
            key: value
            for key, value in fields.items()
            if key in _ALLOWED_FIELDS and value is not None
        }
    )
    _LOGGER.info("priority_trace %s", json.dumps(record, separators=(",", ":")))


def _reset_for_test() -> None:
    global _emitted
    with _lock:
        _emitted = 0
