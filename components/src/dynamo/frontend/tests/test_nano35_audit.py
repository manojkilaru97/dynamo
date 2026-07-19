# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace

from dynamo.frontend.nano35_audit import record_nano35_render_audit


def test_audit_is_default_off(tmp_path, monkeypatch):
    monkeypatch.delenv("DYN_NANO35_AUDIT_FILE", raising=False)
    record_nano35_render_audit(
        request={"reasoning_effort": "low"},
        context=SimpleNamespace(metadata={"nano35-audit-id": "case-low"}),
        response_id="response-1",
        engine_prompt={"prompt": "secret"},
        prompt_token_ids=[],
        tokenizer=None,
        chat_template_kwargs={"enable_thinking": True, "medium_effort": True},
    )
    assert list(tmp_path.iterdir()) == []


def test_audit_records_redacted_render_contract(tmp_path, monkeypatch):
    output = tmp_path / "audit.jsonl"
    template = tmp_path / "chat_template.jinja"
    template.write_text("template")
    monkeypatch.setenv("DYN_NANO35_AUDIT_FILE", str(output))
    monkeypatch.setenv("DYN_NANO35_AUDIT_TEMPLATE", str(template))
    prompt = (
        "private user text {reasoning effort: efficient}"
        "<|im_start|>assistant\n<think>\n"
    )

    record_nano35_render_audit(
        request={"reasoning_effort": "low"},
        context=SimpleNamespace(metadata={"nano35-audit-id": "case-low"}),
        response_id="response-1",
        engine_prompt={"prompt": prompt},
        prompt_token_ids=[],
        tokenizer=None,
        chat_template_kwargs={"enable_thinking": True, "medium_effort": True},
    )

    row = json.loads(output.read_text())
    assert row["audit_id"] == "case-low"
    assert row["response_id"] == "response-1"
    assert row["efficient_marker_count"] == 1
    assert row["generation_prefix"] == "thinking"
    assert row["medium_effort"] is True
    assert row["rendered_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert "private user text" not in output.read_text()


def test_audit_accepts_internal_request_carrier(tmp_path, monkeypatch):
    output = tmp_path / "audit.jsonl"
    template = tmp_path / "chat_template.jinja"
    template.write_text("template")
    monkeypatch.setenv("DYN_NANO35_AUDIT_FILE", str(output))
    monkeypatch.setenv("DYN_NANO35_AUDIT_TEMPLATE", str(template))

    record_nano35_render_audit(
        request={
            "reasoning_effort": "max",
            "chat_template_args": {"__dynamo_nano35_audit_id": "case-max"},
        },
        context=None,
        response_id="response-2",
        engine_prompt={"prompt": "<|im_start|>assistant\n<think>\n"},
        prompt_token_ids=[],
        tokenizer=None,
        chat_template_kwargs={"enable_thinking": True},
    )

    assert json.loads(output.read_text())["audit_id"] == "case-max"
