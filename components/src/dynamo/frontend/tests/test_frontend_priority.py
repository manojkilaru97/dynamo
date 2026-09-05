# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.frontend.priority import normalize_routing_hints


@pytest.mark.parametrize(
    ("input_request", "expected"),
    [
        ({}, None),
        ({"nvext": {}}, None),
        ({"nvext": {"agent_hints": {}}}, None),
        (
            {"nvext": {"agent_hints": {"priority": 0}}},
            {"priority": 0, "priority_jump": 0.0},
        ),
        (
            {"nvext": {"agent_hints": {"priority": -2147483648}}},
            {"priority": -2147483648, "priority_jump": 0.0},
        ),
        (
            {"nvext": {"agent_hints": {"priority": 2147483647}}},
            {"priority": 2147483647, "priority_jump": 2147483647.0},
        ),
        (
            {"nvext": {"agent_hints": {"strict_priority": 4294967295}}},
            {"strict_priority": 4294967295},
        ),
    ],
)
def test_normalize_routing_preserves_presence_and_integer_boundaries(
    input_request, expected
):
    assert normalize_routing_hints(input_request) == expected


def test_normalize_routing_uses_legacy_only_without_ordinary_priority():
    legacy_only = {
        "nvext": {
            "agent_hints": {
                "latency_sensitivity": 6.5,
                "strict_priority": 1,
            }
        }
    }
    explicit_zero = {
        "nvext": {
            "agent_hints": {
                "priority": 0,
                "latency_sensitivity": 6.5,
                "strict_priority": 1,
            }
        }
    }

    assert normalize_routing_hints(legacy_only) == {
        "priority_jump": 6.5,
        "strict_priority": 1,
    }
    assert normalize_routing_hints(explicit_zero) == {
        "priority": 0,
        "priority_jump": 0.0,
        "strict_priority": 1,
    }


def test_normalize_routing_merges_unrelated_data_and_resolved_hints_win():
    request = {
        "routing": {
            "backend_instance_id": 42,
            "custom": {"keep": True},
            "priority": 99,
            "strict_priority": 98,
        },
        "nvext": {
            "agent_hints": {
                "priority": -3,
                "strict_priority": 7,
                "latency_sensitivity": 100.0,
            }
        },
    }

    assert normalize_routing_hints(request) == {
        "backend_instance_id": 42,
        "custom": {"keep": True},
        "priority": -3,
        "priority_jump": 0.0,
        "strict_priority": 7,
    }
