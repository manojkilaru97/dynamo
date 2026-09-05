# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any


def engine_priority_from_routing(
    routing: dict[str, Any],
) -> tuple[bool, int | None, int]:
    """Return presence, Dynamo priority, and the once-negated vLLM priority."""
    priority_present = "priority" in routing
    priority = int(routing["priority"]) if priority_present else None
    return priority_present, priority, -(priority or 0)
