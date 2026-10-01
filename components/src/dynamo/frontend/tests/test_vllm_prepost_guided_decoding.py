# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.frontend.prepost import (
    _build_assistant_guided_decoding,
    build_tool_call_guided_decoding,
    _lift_pattern_only_structured_outputs,
    _validate_chat_completion_request,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}


def _guided_decoding(**fields):
    request = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        **fields,
    }
    return _build_assistant_guided_decoding(_validate_chat_completion_request(request))


def test_pattern_only_structured_outputs_modifies_legacy_guided_json():
    guided = _guided_decoding(
        guided_json=SCHEMA, structured_outputs={"whitespace_pattern": "[ ]?"}
    )
    assert guided == {"json": SCHEMA, "whitespace_pattern": "[ ]?"}


def test_top_level_pattern_modifies_structured_outputs_json():
    guided = _guided_decoding(
        structured_outputs={"json": SCHEMA}, guided_whitespace_pattern="[ ]?"
    )
    assert guided["json"] == SCHEMA
    assert guided["whitespace_pattern"] == "[ ]?"


def test_pattern_does_not_attach_to_non_json_constraints():
    guided = _guided_decoding(guided_regex="a+", guided_whitespace_pattern="[ ]?")
    assert guided == {"regex": "a+"}


def test_lifted_pattern_drops_unset_sentinels():
    lifted = _lift_pattern_only_structured_outputs(
        {
            "structured_outputs": {"whitespace_pattern": "[ ]?", "json_object": False},
            "tool_choice": {"type": "function", "function": {"name": "record"}},
        }
    )
    assert "structured_outputs" not in lifted
    assert lifted["guided_whitespace_pattern"] == "[ ]?"


def test_false_json_schema_with_pattern_is_not_lifted():
    request = {"structured_outputs": {"json": False, "whitespace_pattern": "[ ]?"}}
    assert _lift_pattern_only_structured_outputs(request) is request


def test_empty_choice_with_pattern_is_lifted():
    lifted = _lift_pattern_only_structured_outputs(
        {"structured_outputs": {"choice": [], "whitespace_pattern": "[ ]?"}}
    )
    assert "structured_outputs" not in lifted
    assert lifted["guided_whitespace_pattern"] == "[ ]?"


@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "function", "function": {"name": "record"}}],
    ids=["required", "named"],
)
@pytest.mark.parametrize(
    "pattern_fields",
    [
        {"guided_whitespace_pattern": "[ ]?"},
        {"structured_outputs": {"whitespace_pattern": "[ ]?"}},
    ],
    ids=["legacy", "alias"],
)
def test_json_tool_fallback_keeps_whitespace_pattern(tool_choice, pattern_fields):
    request = _validate_chat_completion_request(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "record", "parameters": {"type": "object"}},
                }
            ],
            "tool_choice": tool_choice,
            **pattern_fields,
        }
    )
    guided = build_tool_call_guided_decoding(request, None)
    assert guided is not None and "json" in guided
    assert guided["whitespace_pattern"] == "[ ]?"
