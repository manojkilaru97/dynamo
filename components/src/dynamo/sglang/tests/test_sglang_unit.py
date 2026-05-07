# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang backend components."""

import asyncio
import json
import re
import sys
from types import SimpleNamespace
from pathlib import Path
from contextlib import asynccontextmanager

import pytest
import yaml

from dynamo.sglang.args import parse_args
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler
from dynamo.sglang.tests.conftest import make_cli_args_fixture

# Get path relative to this test file
REPO_ROOT = Path(__file__).resolve().parents[5]
TEST_DIR = REPO_ROOT / "tests"
# Now construct the full path to the shared test fixture
JINJA_TEMPLATE_PATH = str(
    REPO_ROOT / "tests" / "serve" / "fixtures" / "custom_template.jinja"
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]
# Create SGLang-specific CLI args fixture
# This will use monkeypatch to write to argv
mock_sglang_cli = make_cli_args_fixture("dynamo.sglang")


class _AbortRecorder:
    def __init__(self):
        self.aborted = []

    def abort_request(self, rid, abort_all=False):
        self.aborted.append((rid, abort_all))


def _make_decode_handler_for_slot_tests(limit=2, lease_secs=600.0):
    handler = object.__new__(DecodeWorkerHandler)
    recorder = _AbortRecorder()
    handler.max_total_requests = limit
    handler.max_total_requests_per_dp = max(1, limit // 2)
    handler.request_slot_lease_secs = lease_secs
    handler._request_admission_lock = asyncio.Lock()
    handler._active_request_admissions = 0
    handler._active_request_admissions_high_water = 0
    handler._request_admissions = {}
    handler._request_admission_dp_counts = {}
    handler._request_slots_reaped_total = 0
    handler.stale_full_unhealthy_secs = 60.0
    handler._last_stream_progress_at = 0.0
    handler.engine = SimpleNamespace(tokenizer_manager=recorder)
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(served_model_name="test-model")
    )
    return handler, recorder


class _FakeContext:
    def __init__(self, context_id="ctx"):
        self._id = context_id

    def id(self):
        return self._id

    def is_stopped(self):
        return False


@asynccontextmanager
async def _noop_cancellation_monitor(*_args, **_kwargs):
    yield


@pytest.mark.asyncio
async def test_custom_jinja_template_invalid_path(mock_sglang_cli):
    """Test that invalid file path raises FileNotFoundError."""
    invalid_path = "/nonexistent/path/to/template.jinja"
    mock_sglang_cli(
        "--model", "Qwen/Qwen3-0.6B", "--custom-jinja-template", invalid_path
    )

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"Custom Jinja template file not found: {invalid_path}"),
    ):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_custom_jinja_template_valid_path(mock_sglang_cli):
    """Test that valid absolute path is stored correctly."""
    mock_sglang_cli(model="Qwen/Qwen3-0.6B", custom_jinja_template=JINJA_TEMPLATE_PATH)

    config = await parse_args(sys.argv[1:])

    assert config.dynamo_args.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.dynamo_args.custom_jinja_template}"
    )


@pytest.mark.asyncio
async def test_custom_jinja_template_env_var_expansion(monkeypatch, mock_sglang_cli):
    """Test that environment variables in paths are expanded by Python code."""
    jinja_dir = str(TEST_DIR / "serve" / "fixtures")
    monkeypatch.setenv("JINJA_DIR", jinja_dir)

    cli_path = "$JINJA_DIR/custom_template.jinja"
    mock_sglang_cli(model="Qwen/Qwen3-0.6B", custom_jinja_template=cli_path)

    config = await parse_args(sys.argv[1:])

    assert "$JINJA_DIR" not in config.dynamo_args.custom_jinja_template
    assert config.dynamo_args.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.dynamo_args.custom_jinja_template}"
    )


# --- Tool Call Parser Validation Tests ---


@pytest.mark.asyncio
async def test_tool_call_parser_valid_with_dynamo_tokenizer(mock_sglang_cli):
    """Valid parser name works when using Dynamo's tokenizer."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dyn-tool-call-parser",
        "hermes",  # supported by Dynamo
    )

    config = await parse_args(sys.argv[1:])

    assert config.dynamo_args.dyn_tool_call_parser == "hermes"


@pytest.mark.asyncio
async def test_tool_call_parser_invalid_with_dynamo_tokenizer(mock_sglang_cli):
    """Invalid parser name exits when using Dynamo's tokenizer."""
    mock_sglang_cli(
        "--model", "Qwen/Qwen3-0.6B", "--dyn-tool-call-parser", "nonexistent_parser"
    )

    with pytest.raises(SystemExit):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_tool_call_parser_both_flags_error(mock_sglang_cli):
    """Setting both --dyn-tool-call-parser and --tool-call-parser exits with error."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dyn-tool-call-parser",
        "hermes",
        "--tool-call-parser",
        "qwen25",
    )

    with pytest.raises(SystemExit):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_namespace_flag_drives_default_endpoint_namespace(mock_sglang_cli):
    """CLI namespace should be used for auto-derived endpoint."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--namespace",
        "custom-ns",
    )

    config = await parse_args(sys.argv[1:])
    assert config.dynamo_args.namespace == "custom-ns"


@pytest.mark.asyncio
async def test_obsolete_dyn_endpoint_types_flag_is_supported(mock_sglang_cli):
    """Obsolete --dyn-endpoint-types alias should map to endpoint_types."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dyn-endpoint-types",
        "completions",
    )

    config = await parse_args(sys.argv[1:])
    assert config.dynamo_args.endpoint_types == "completions"


@pytest.mark.asyncio
async def test_disagg_config_requires_disagg_config_key(mock_sglang_cli):
    """--disagg-config and --disagg-config-key must be provided together."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        "/tmp/nonexistent.yaml",
    )

    with pytest.raises(ValueError, match="disagg_config.*disagg_config_key.*together"):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_disagg_config_key_requires_disagg_config(mock_sglang_cli):
    """--disagg-config-key alone should fail."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config-key",
        "prefill",
    )

    with pytest.raises(ValueError, match="disagg_config.*disagg_config_key.*together"):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_disagg_config_key_not_found_error(tmp_path, mock_sglang_cli):
    """Missing disagg section key should raise a clear ValueError."""
    config_path = tmp_path / "disagg.yaml"
    config_path.write_text(
        yaml.safe_dump({"prefill": {"tensor_parallel_size": 1}}), encoding="utf-8"
    )

    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        str(config_path),
        "--disagg-config-key",
        "decode",
    )

    with pytest.raises(ValueError, match="Disagg config key 'decode' not found"):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_disagg_config_section_must_be_dict(tmp_path, mock_sglang_cli):
    """Selected disagg section must be a dictionary."""
    config_path = tmp_path / "disagg.yaml"
    config_path.write_text(yaml.safe_dump({"prefill": "not-a-dict"}), encoding="utf-8")

    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        str(config_path),
        "--disagg-config-key",
        "prefill",
    )

    with pytest.raises(
        ValueError, match="Disagg config section 'prefill' must be a dictionary"
    ):
        await parse_args(sys.argv[1:])


def test_xgrammar_schema_normalization_adds_runaway_bounds():
    """Unbounded JSON schemas should not let constrained decoding run forever."""
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "metadata": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["title", "tags", "metadata"],
    }

    normalized = DecodeWorkerHandler._normalize_json_schema_for_xgrammar(schema)

    assert normalized["maxProperties"] == 64
    assert normalized["properties"]["title"]["maxLength"] == 4096
    assert normalized["properties"]["tags"]["maxItems"] == 64
    assert normalized["properties"]["tags"]["items"]["maxLength"] == 4096
    assert normalized["properties"]["metadata"]["maxProperties"] == 64
    assert (
        normalized["properties"]["metadata"]["additionalProperties"]["maxLength"]
        == 4096
    )


def test_xgrammar_schema_normalization_preserves_explicit_bounds():
    schema = {
        "type": "object",
        "maxProperties": 3,
        "properties": {
            "title": {"type": "string", "maxLength": 32},
            "tags": {"type": "array", "maxItems": 2, "items": {"type": "string"}},
        },
    }

    normalized = DecodeWorkerHandler._normalize_json_schema_for_xgrammar(schema)

    assert normalized["maxProperties"] == 3
    assert normalized["properties"]["title"]["maxLength"] == 32
    assert normalized["properties"]["tags"]["maxItems"] == 2
    assert normalized["properties"]["tags"]["items"]["maxLength"] == 4096


def test_xgrammar_schema_normalization_disambiguates_oneof_object_branches():
    schema = {
        "type": "object",
        "oneOf": [
            {"properties": {"key": {"type": "string"}}},
            {"properties": {"keys": {"type": "array", "items": {"type": "string"}}}},
        ],
    }

    normalized = DecodeWorkerHandler._normalize_json_schema_for_xgrammar(schema)

    assert normalized["oneOf"][0]["required"] == ["key"]
    assert normalized["oneOf"][1]["required"] == ["keys"]


def test_xgrammar_schema_normalization_keeps_root_object_over_side_anyof():
    schema = {
        "type": "object",
        "properties": {"Age": {"type": "number"}},
        "required": ["Age"],
        "anyOf": [{"not": {"required": ["FirstName", "LastName"]}}],
    }

    normalized = DecodeWorkerHandler._normalize_json_schema_for_xgrammar(schema)

    assert "anyOf" not in normalized
    assert normalized["type"] == "object"
    assert normalized["required"] == ["Age"]
    assert normalized["properties"]["Age"]["type"] == "number"


def test_thinking_guided_json_uses_bounded_reasoning_region():
    params = DecodeWorkerHandler._guided_to_sglang_params(
        {
            "enable_thinking": True,
            "json": {"type": "string", "enum": ["low", "medium", "high"]},
        }
    )

    structural_tag = json.loads(params["structural_tag"])
    reasoning_region = structural_tag["format"]["elements"][0]["content"]

    assert reasoning_region["type"] == "regex"
    assert "8192" in reasoning_region["pattern"]
    assert reasoning_region["pattern"].startswith("[^<]")
    assert "any_text" not in params["structural_tag"]


def test_openai_sampling_params_preserve_sglang_controls():
    handler = object.__new__(DecodeWorkerHandler)

    params = handler._build_sampling_params(
        {
            "presence_penalty": 1.5,
            "frequency_penalty": 0.25,
            "repetition_penalty": 1.0,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "seed": 1,
            "max_tokens": 65536,
            "ignore_eos": True,
        }
    )

    assert params == {
        "presence_penalty": 1.5,
        "frequency_penalty": 0.25,
        "repetition_penalty": 1.0,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "max_new_tokens": 65536,
        "ignore_eos": True,
    }


def test_preprocessed_sampling_params_preserve_sglang_controls():
    handler = object.__new__(DecodeWorkerHandler)

    params = handler._build_sampling_params(
        {
            "sampling_options": {
                "presence_penalty": 1.5,
                "frequency_penalty": 0.25,
                "repetition_penalty": 1.0,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "seed": 1,
            },
            "stop_conditions": {"max_tokens": 65536, "ignore_eos": False},
        }
    )

    assert params == {
        "presence_penalty": 1.5,
        "frequency_penalty": 0.25,
        "repetition_penalty": 1.0,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "max_new_tokens": 65536,
        "ignore_eos": False,
    }


@pytest.mark.asyncio
async def test_worker_admission_slots_track_release_by_context():
    handler, recorder = _make_decode_handler_for_slot_tests(limit=2)

    reserved, active, limit, reason = await handler._try_reserve_request_slot(
        "ctx-1", "rid-1"
    )
    assert (reserved, active, limit) == (True, 1, 2)
    assert reason is None

    reserved, active, limit, reason = await handler._try_reserve_request_slot(
        "ctx-2", "rid-2"
    )
    assert (reserved, active, limit) == (True, 2, 2)
    assert reason is None

    reserved, active, limit, reason = await handler._try_reserve_request_slot(
        "ctx-3", "rid-3"
    )
    assert (reserved, active, limit) == (False, 2, 2)
    assert reason is None

    await handler._release_request_slot_reservation("ctx-1")
    reserved, active, limit, reason = await handler._try_reserve_request_slot(
        "ctx-3", "rid-3"
    )
    assert (reserved, active, limit) == (True, 2, 2)
    assert reason is None
    assert set(handler._request_admissions) == {"ctx-2", "ctx-3"}
    assert recorder.aborted == []


@pytest.mark.asyncio
async def test_worker_admission_slots_release_on_cancel_or_error():
    handler, _ = _make_decode_handler_for_slot_tests(limit=1)

    reserved, active, limit, reason = await handler._try_reserve_request_slot(
        "ctx-1", "rid-1"
    )
    assert (reserved, active, limit) == (True, 1, 1)
    assert reason is None

    await handler._release_request_slot_reservation("ctx-1")

    reserved, active, limit, reason = await handler._try_reserve_request_slot(
        "ctx-2", "rid-2"
    )
    assert (reserved, active, limit) == (True, 1, 1)
    assert reason is None
    assert set(handler._request_admissions) == {"ctx-2"}


@pytest.mark.asyncio
async def test_worker_admission_duplicate_context_is_idempotent():
    handler, _ = _make_decode_handler_for_slot_tests(limit=4)

    reserved, active, limit, reason = await handler._try_reserve_request_slot(
        "ctx-1", "rid-1", dp_rank=0
    )
    assert (reserved, active, limit, reason) == (True, 1, 4, None)
    assert handler._request_admission_dp_counts == {0: 1}

    reserved, active, limit, reason = await handler._try_reserve_request_slot(
        "ctx-1", "rid-1b", dp_rank=0
    )

    assert (reserved, active, limit, reason) == (True, 1, 4, None)
    assert len(handler._request_admissions) == 1
    assert handler._request_admissions["ctx-1"].sglang_request_id == "rid-1b"
    assert handler._request_admission_dp_counts == {0: 1}


@pytest.mark.asyncio
async def test_worker_admission_duplicate_context_moves_dp_count():
    handler, _ = _make_decode_handler_for_slot_tests(limit=4)

    await handler._try_reserve_request_slot("ctx-1", "rid-1", dp_rank=0)
    await handler._try_reserve_request_slot("ctx-1", "rid-1", dp_rank=1)

    assert len(handler._request_admissions) == 1
    assert handler._request_admissions["ctx-1"].dp_rank == 1
    assert handler._request_admission_dp_counts == {1: 1}

    await handler._release_request_slot_reservation("ctx-1")

    assert handler._request_admissions == {}
    assert handler._request_admission_dp_counts == {}


@pytest.mark.asyncio
async def test_worker_admission_slots_reap_and_abort_stale_slots():
    handler, recorder = _make_decode_handler_for_slot_tests(limit=1, lease_secs=10.0)

    reserved, active, limit, reason = await handler._try_reserve_request_slot(
        "ctx-old", "rid-old"
    )
    assert (reserved, active, limit) == (True, 1, 1)
    assert reason is None
    handler._request_admissions["ctx-old"].created_at -= 11.0

    reserved, active, limit, reason = await handler._try_reserve_request_slot(
        "ctx-new", "rid-new"
    )

    assert (reserved, active, limit) == (False, 1, 1)
    assert reason is None
    assert set(handler._request_admissions) == {"ctx-old"}
    assert handler._request_slots_reaped_total == 0
    assert recorder.aborted == []

    handler._request_admissions["ctx-old"].last_progress_at -= 11.0

    reserved, active, limit, reason = await handler._try_reserve_request_slot(
        "ctx-new", "rid-new"
    )

    assert (reserved, active, limit) == (True, 1, 1)
    assert reason is None
    assert set(handler._request_admissions) == {"ctx-new"}
    assert handler._request_slots_reaped_total == 1
    assert recorder.aborted == [("rid-old", False)]


@pytest.mark.asyncio
async def test_worker_admission_health_check_fails_when_full_and_progress_stale():
    handler, _ = _make_decode_handler_for_slot_tests(limit=1, lease_secs=600.0)
    handler.stale_full_unhealthy_secs = 60.0

    reserved, active, limit, reason = await handler._try_reserve_request_slot(
        "ctx-old", "rid-old"
    )
    assert (reserved, active, limit, reason) == (True, 1, 1, None)
    handler._last_stream_progress_at -= 61.0

    reserved, active, limit, reason = await handler._try_reserve_request_slot(
        "ctx-health", "rid-health", health_check=True
    )

    assert (reserved, active, limit) == (False, 1, 1)
    assert reason is not None
    assert "worker unhealthy" in reason


def test_sglang_health_check_payload_is_marked():
    from dynamo.sglang.health_check import SglangHealthCheckPayload

    payload = SglangHealthCheckPayload().to_dict()

    assert DecodeWorkerHandler._is_health_check_request(payload)


@pytest.mark.asyncio
async def test_process_token_stream_tolerates_missing_final_usage_metadata():
    handler, _ = _make_decode_handler_for_slot_tests(limit=1)
    handler._cancellation_monitor = _noop_cancellation_monitor
    context = _FakeContext("ctx-usage")
    await handler._try_reserve_request_slot("ctx-usage", "rid-usage")

    async def stream():
        yield {
            "meta_info": {
                "id": "rid-usage",
                "finish_reason": {"type": "stop"},
            },
            "output_ids": [1, 2, 3],
        }

    released = False

    async def release_once():
        nonlocal released
        released = True

    chunks = []
    async for chunk in handler._process_token_stream(stream(), context, release_once):
        chunks.append(chunk)
        assert released is True

    assert chunks == [{"finish_reason": "stop", "token_ids": [1, 2, 3]}]
    assert released is True


@pytest.mark.asyncio
async def test_process_text_stream_releases_slot_before_final_chunk():
    handler, _ = _make_decode_handler_for_slot_tests(limit=1)
    handler._cancellation_monitor = _noop_cancellation_monitor
    context = _FakeContext("ctx-text")
    await handler._try_reserve_request_slot("ctx-text", "rid-text")

    async def stream():
        yield {
            "index": 0,
            "text": "done",
            "meta_info": {
                "id": "rid-text",
                "finish_reason": {"type": "stop"},
            },
        }

    released_before_yield = False

    async def release_once():
        nonlocal released_before_yield
        released_before_yield = True

    chunks = []
    async for chunk in handler._process_text_stream(
        stream(), context, release_once
    ):
        chunks.append(chunk)
        assert released_before_yield is True

    assert chunks[0]["choices"][0]["finish_reason"] == "stop"
    assert chunks[0]["choices"][0]["delta"]["content"] == "done"


@pytest.mark.asyncio
async def test_disagg_config_preserves_bootstrap_port(tmp_path, mock_sglang_cli):
    """Bootstrap port from disagg section should not be overridden by auto-port logic."""
    config_path = tmp_path / "disagg.yaml"
    config_path.write_text(
        yaml.safe_dump({"prefill": {"disaggregation-bootstrap-port": 42345}}),
        encoding="utf-8",
    )

    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        str(config_path),
        "--disagg-config-key",
        "prefill",
    )

    config = await parse_args(sys.argv[1:])
    assert config.server_args.disaggregation_bootstrap_port == 42345


@pytest.mark.asyncio
async def test_disagg_config_rejects_dynamo_keys(tmp_path, mock_sglang_cli, capfd):
    """Disagg config should only accept SGLang-native keys."""
    config_path = tmp_path / "disagg.yaml"
    config_path.write_text(
        yaml.safe_dump({"prefill": {"store-kv": "mem"}}), encoding="utf-8"
    )

    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        str(config_path),
        "--disagg-config-key",
        "prefill",
    )

    with pytest.raises(SystemExit):
        await parse_args(sys.argv[1:])

    out, err = capfd.readouterr()
    assert "unrecognized arguments: --store-kv mem" in err
