# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.frontend.prepost import (
    _build_assistant_guided_decoding,
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
