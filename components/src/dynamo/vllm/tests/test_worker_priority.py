# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging

from dynamo.common import priority_trace
from dynamo.vllm.priority import engine_priority_from_routing


def test_engine_priority_negates_ordinary_priority_exactly_once():
    assert engine_priority_from_routing({}) == (False, None, 0)
    assert engine_priority_from_routing({"strict_priority": 1}) == (False, None, 0)
    assert engine_priority_from_routing({"priority": 0}) == (True, 0, 0)
    assert engine_priority_from_routing({"priority": -2147483648}) == (
        True,
        -2147483648,
        2147483648,
    )
    assert engine_priority_from_routing({"priority": 2147483647}) == (
        True,
        2147483647,
        -2147483647,
    )


def test_priority_trace_is_opt_in_bounded_and_payload_free(
    monkeypatch, caplog
):
    monkeypatch.setenv("DYN_PRIORITY_TRACE", "1")
    monkeypatch.setenv("DYN_PRIORITY_TRACE_LIMIT", "1")
    priority_trace._reset_for_test()

    with caplog.at_level(logging.INFO, logger="dynamo.priority"):
        priority_trace.emit(
            "normalization",
            request_id="request-1",
            priority=4,
            prompt="private prompt",
            authorization="secret",
        )
        priority_trace.emit("normalization", request_id="request-2", priority=5)

    records = [
        json.loads(record.message.removeprefix("priority_trace "))
        for record in caplog.records
        if record.message.startswith("priority_trace ")
    ]
    assert len(records) == 1
    assert records[0]["request_id"] == "request-1"
    assert records[0]["priority"] == 4
    assert "prompt" not in records[0]
    assert "authorization" not in records[0]
