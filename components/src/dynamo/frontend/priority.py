# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any


def normalize_routing_hints(request: dict[str, Any]) -> dict[str, Any] | None:
    """Project already-resolved nvext agent hints into routing metadata."""
    existing = request.get("routing")
    routing = dict(existing) if isinstance(existing, dict) else {}
    hints = (request.get("nvext") or {}).get("agent_hints") or {}

    if "priority" in hints and hints["priority"] is not None:
        priority = hints["priority"]
        routing["priority"] = priority
        routing["priority_jump"] = float(max(priority, 0))
    elif "latency_sensitivity" in hints and hints["latency_sensitivity"] is not None:
        routing["priority_jump"] = hints["latency_sensitivity"]

    if "strict_priority" in hints and hints["strict_priority"] is not None:
        routing["strict_priority"] = hints["strict_priority"]

    return routing or None
