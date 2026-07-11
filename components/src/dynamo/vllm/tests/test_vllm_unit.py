# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM backend components."""

import asyncio
import json
import re
import socket
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dynamo.vllm.args import (
    _connector_to_kv_transfer_json,
    _is_routable,
    _uses_dynamo_connector,
    _uses_nixl_connector,
    ensure_side_channel_host,
    get_host_ip,
    parse_args,
)
from dynamo.vllm.constants import DisaggregationMode
from dynamo.vllm.tests.conftest import make_cli_args_fixture

# Get path relative to this test file
REPO_ROOT = Path(__file__).resolve().parents[5]
TEST_DIR = REPO_ROOT / "tests"
# Now construct the full path to the shared test fixture
JINJA_TEMPLATE_PATH = str(
    REPO_ROOT / "tests" / "serve" / "fixtures" / "custom_template.jinja"
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    # gpu_1 not gpu_0: vLLM DeviceConfig(device='auto') fails on CPU-only arm64
    # runners with "Failed to infer device type" even for mock tests.
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]

# Create vLLM-specific CLI args fixture
# This will use monkeypatch to write to argv
mock_vllm_cli = make_cli_args_fixture("dynamo.vllm")


def test_custom_jinja_template_invalid_path(mock_vllm_cli):
    """Test that invalid file path raises FileNotFoundError."""
    invalid_path = "/nonexistent/path/to/template.jinja"

    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--custom-jinja-template", invalid_path)

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"Custom Jinja template file not found: {invalid_path}"),
    ):
        parse_args()


def test_custom_jinja_template_valid_path(mock_vllm_cli):
    """Test that valid absolute path is stored correctly."""
    mock_vllm_cli(model="Qwen/Qwen3-0.6B", custom_jinja_template=JINJA_TEMPLATE_PATH)

    config = parse_args()

    assert config.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.custom_jinja_template}"
    )


def test_custom_jinja_template_env_var_expansion(monkeypatch, mock_vllm_cli):
    """Test that environment variables in paths are expanded by Python code."""
    jinja_dir = str(TEST_DIR / "serve" / "fixtures")
    monkeypatch.setenv("JINJA_DIR", jinja_dir)

    cli_path = "$JINJA_DIR/custom_template.jinja"
    mock_vllm_cli(model="Qwen/Qwen3-0.6B", custom_jinja_template=cli_path)

    config = parse_args()

    assert "$JINJA_DIR" not in config.custom_jinja_template
    assert config.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.custom_jinja_template}"
    )


def test_served_model_aliases_are_separate_from_primary_name(mock_vllm_cli):
    """Test Dynamo aliases do not require multiple vLLM served model names."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--served-model-name",
        "primary-model",
        "--dyn-served-model-alias",
        "alias-a",
        "alias-b",
    )

    config = parse_args()

    assert config.served_model_name == "primary-model"
    assert config.dyn_served_model_alias == ["alias-a", "alias-b"]
    assert config.engine_args.served_model_name == ["primary-model"]


def test_multiple_served_model_names_fail_with_alias_hint(mock_vllm_cli):
    """Test startup fails before vLLM registration when aliases are misconfigured."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--served-model-name",
        "primary-model",
        "alias-a",
    )

    with pytest.raises(ValueError, match="Fix multiple model names issue"):
        parse_args()
# --endpoint flag tests


def test_endpoint_overrides_defaults(mock_vllm_cli):
    """Test that --endpoint overrides default namespace/component/endpoint."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--endpoint",
        "dyn://mynamespace.mycomponent.myendpoint",
    )
    config = parse_args()
    assert config.namespace == "mynamespace"
    assert config.component == "mycomponent"
    assert config.endpoint == "myendpoint"


def test_endpoint_not_provided_preserves_defaults(mock_vllm_cli):
    """Test that without --endpoint, defaults are preserved."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
    config = parse_args()
    assert config.namespace == "dynamo"
    assert config.component == "backend"
    assert config.endpoint == "generate"


def test_endpoint_overrides_with_prefill_worker(mock_vllm_cli):
    """Test that --endpoint overrides even with --disaggregation-mode prefill."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--endpoint",
        "dyn://custom.worker.serve",
        "--disaggregation-mode",
        "prefill",
        "--kv-transfer-config",
        '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
    )
    config = parse_args()
    assert config.namespace == "custom"
    assert config.component == "worker"
    assert config.endpoint == "serve"


def test_endpoint_invalid_format_raises(mock_vllm_cli):
    """Test that invalid --endpoint format raises ValueError."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--endpoint",
        "invalid-endpoint",
    )
    with pytest.raises(ValueError, match="Invalid endpoint format"):
        parse_args()


# --connector removal tests


def test_connector_nixl_raises_error_with_migration_hint(mock_vllm_cli):
    """Test that --connector nixl raises ValueError with --kv-transfer-config hint."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--connector", "nixl")
    with pytest.raises(ValueError, match="--connector is no longer supported"):
        parse_args()


def test_connector_none_raises_error(mock_vllm_cli):
    """Test that --connector none raises ValueError telling user it's no longer needed."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--connector", "none")
    with pytest.raises(ValueError, match="no longer needed"):
        parse_args()


def test_env_var_dyn_connector_raises_error(monkeypatch, mock_vllm_cli):
    """Test that DYN_CONNECTOR env var raises error for vLLM backend."""
    monkeypatch.setenv("DYN_CONNECTOR", "nixl")
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
    with pytest.raises(ValueError, match="no longer supported"):
        parse_args()


def test_model_express_url_is_accepted_for_compatibility(mock_vllm_cli):
    """Test that legacy ModelExpress manifests still parse."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--model-express-url",
        "http://model-express:8080",
    )

    config = parse_args()

    assert config.model_express_url == "http://model-express:8080"


def test_model_express_url_env_is_accepted_for_compatibility(
    monkeypatch, mock_vllm_cli
):
    """Test that legacy MODEL_EXPRESS_URL still maps to config."""
    monkeypatch.setenv("MODEL_EXPRESS_URL", "http://model-express:8080")
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")

    config = parse_args()

    assert config.model_express_url == "http://model-express:8080"


def test_prefill_worker_without_kv_transfer_config_raises(mock_vllm_cli):
    """Test that --disaggregation-mode prefill without --kv-transfer-config raises ValueError."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--disaggregation-mode", "prefill")
    with pytest.raises(ValueError, match="--kv-transfer-config"):
        parse_args()


def test_connector_to_kv_transfer_json_single():
    """Test _connector_to_kv_transfer_json returns valid JSON for a single connector."""
    result = json.loads(_connector_to_kv_transfer_json(["nixl"]))
    assert result == {"kv_connector": "NixlConnector", "kv_role": "kv_both"}


def test_connector_to_kv_transfer_json_multi():
    """Test _connector_to_kv_transfer_json wraps multiple connectors in PdConnector."""
    result = json.loads(_connector_to_kv_transfer_json(["kvbm", "nixl"]))
    assert result["kv_connector"] == "PdConnector"
    nested = result["kv_connector_extra_config"]["connectors"]
    nested_names = [c["kv_connector"] for c in nested]
    assert "DynamoConnector" in nested_names
    assert "NixlConnector" in nested_names


# _uses_nixl_connector / _uses_dynamo_connector tests


def _make_engine_cfg(kv_connector=None, extra_config=None):
    """Build a minimal fake engine config for connector detection tests."""
    if kv_connector is None:
        return SimpleNamespace(kv_transfer_config=None)
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector=kv_connector,
            kv_connector_extra_config=extra_config,
        )
    )


_PD_KVBM_NIXL = {
    "connectors": [
        {"kv_connector": "DynamoConnector", "kv_role": "kv_both"},
        {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
    ]
}


def test_uses_nixl_connector_direct_and_nested():
    """Test _uses_nixl_connector for direct, nested-in-PdConnector, and absent cases."""
    assert _uses_nixl_connector(_make_engine_cfg("NixlConnector")) is True
    assert _uses_nixl_connector(_make_engine_cfg("PdConnector", _PD_KVBM_NIXL)) is True
    assert _uses_nixl_connector(_make_engine_cfg("LMCacheConnectorV1")) is False
    assert _uses_nixl_connector(_make_engine_cfg("FlexKVConnectorV1")) is False
    assert _uses_nixl_connector(_make_engine_cfg()) is False


def test_uses_dynamo_connector_direct_and_nested():
    """Test _uses_dynamo_connector for direct, nested-in-PdConnector, and absent cases."""
    assert _uses_dynamo_connector(_make_engine_cfg("DynamoConnector")) is True
    assert (
        _uses_dynamo_connector(_make_engine_cfg("PdConnector", _PD_KVBM_NIXL)) is True
    )
    assert _uses_dynamo_connector(_make_engine_cfg("NixlConnector")) is False
    assert _uses_dynamo_connector(_make_engine_cfg()) is False


def test_headless_namespace_has_required_fields(mock_vllm_cli):
    """Test that build_headless_namespace produces a Namespace with fields
    required by vLLM's run_headless(), including the api_server_count fallback."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--headless",
    )
    config = parse_args()
    assert config.headless is True

    from dynamo.vllm.main import build_headless_namespace

    ns = build_headless_namespace(config)

    # Required by run_headless()
    assert hasattr(ns, "api_server_count")
    assert ns.api_server_count == 0
    # Core engine fields must survive the round-trip
    assert hasattr(ns, "model")
    assert hasattr(ns, "tensor_parallel_size")


def test_should_prefetch_model_for_default_load_format():
    from dynamo.vllm.main import should_prefetch_model

    config = SimpleNamespace(
        model="Qwen/Qwen3-0.6B",
        engine_args=SimpleNamespace(load_format="auto"),
    )

    assert should_prefetch_model(config) is True


@pytest.mark.parametrize("load_format", ["modelexpress", "mx"])
def test_should_not_prefetch_model_for_modelexpress_load_formats(load_format):
    from dynamo.vllm.main import (
        should_prefetch_model,
        should_register_model_ignore_weights,
        uses_modelexpress_load_format,
    )

    config = SimpleNamespace(
        model="Qwen/Qwen3-0.6B",
        engine_args=SimpleNamespace(load_format=load_format),
    )

    assert uses_modelexpress_load_format(config) is True
    assert should_prefetch_model(config) is False
    assert should_register_model_ignore_weights(config) is True


def test_should_not_prefetch_existing_local_model(tmp_path):
    from dynamo.vllm.main import should_prefetch_model

    config = SimpleNamespace(
        model=str(tmp_path),
        engine_args=SimpleNamespace(load_format="auto"),
    )

    assert should_prefetch_model(config) is False


def test_should_register_model_fetch_weights_for_default_load_format():
    from dynamo.vllm.main import should_register_model_ignore_weights

    config = SimpleNamespace(
        model="Qwen/Qwen3-0.6B",
        engine_args=SimpleNamespace(load_format="auto"),
    )

    assert should_register_model_ignore_weights(config) is False


def test_setup_vllm_engine_reuses_engine_config_model_config(monkeypatch):
    from dynamo.vllm import main as vllm_main

    class FakeModelConfig:
        def get_diff_sampling_param(self):
            return {"temperature": 0.7}

    vllm_config = SimpleNamespace(
        additional_config={},
        cache_config=SimpleNamespace(block_size=None),
        model_config=FakeModelConfig(),
    )

    class FakeEngineArgs:
        enable_log_requests = False
        enable_lora = False
        disable_log_stats = True
        load_format = "modelexpress"

        def create_model_config(self):
            raise AssertionError("setup_vllm_engine must not create ModelConfig twice")

        def create_engine_config(self, usage_context):
            return vllm_config

    engine_client = SimpleNamespace(vllm_config=vllm_config)

    class FakeAsyncLLM:
        @staticmethod
        def from_vllm_config(**_kwargs):
            return engine_client

    class FakeMetrics:
        def __init__(self, **_kwargs):
            pass

        def set_model_load_time(self, _load_time):
            pass

    monkeypatch.setattr(vllm_main, "setup_multiprocess_prometheus", lambda: None)
    monkeypatch.setattr(vllm_main, "LLMBackendMetrics", FakeMetrics)
    monkeypatch.setattr(vllm_main, "_uses_dynamo_connector", lambda _args: False)
    monkeypatch.setattr(vllm_main, "AsyncLLM", FakeAsyncLLM)
    monkeypatch.setattr(
        vllm_main,
        "get_engine_cache_info",
        lambda _engine: {"block_size": 16},
    )

    config = SimpleNamespace(
        component="backend",
        engine_args=FakeEngineArgs(),
        gms_shadow_mode=False,
        multimodal_embedding_cache_capacity_gb=0,
        route_to_encoder=False,
        served_model_name="Qwen/Qwen3-0.6B",
    )

    _, _, default_sampling_params, _, _ = vllm_main.setup_vllm_engine(config)

    assert default_sampling_params == {"temperature": 0.7}

def test_reasoning_parser_propagates_to_structured_outputs(mock_vllm_cli):
    """Tool/JSON constraints must not apply while a reasoning model is thinking."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--reasoning-parser",
        "nemotron_v3",
    )

    config = parse_args()

    assert config.engine_args.reasoning_parser == "nemotron_v3"
    assert (
        config.engine_args.structured_outputs_config.reasoning_parser == "nemotron_v3"
    )


# --disaggregation-mode tests


def test_disaggregation_mode_default(mock_vllm_cli):
    """Test that default disaggregation mode is AGGREGATED."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
    config = parse_args()
    assert config.disaggregation_mode == DisaggregationMode.AGGREGATED
    assert config.is_prefill_worker is False
    assert config.is_decode_worker is False


def test_kv_events_disabled_by_default_without_explicit_config(mock_vllm_cli):
    """Test that vLLM no longer auto-creates kv_events_config."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
    config = parse_args()
    assert config.engine_args.kv_events_config is None
    assert config.use_kv_events is False


def test_disaggregation_mode_prefill(mock_vllm_cli):
    """Test --disaggregation-mode prefill sets correct state."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disaggregation-mode",
        "prefill",
        "--kv-transfer-config",
        '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
    )
    config = parse_args()
    assert config.disaggregation_mode == DisaggregationMode.PREFILL
    assert config.is_prefill_worker is True
    assert config.is_decode_worker is False
    assert config.component == "prefill"


def test_disaggregation_mode_decode(mock_vllm_cli):
    """Test --disaggregation-mode decode sets correct state."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--disaggregation-mode", "decode")
    config = parse_args()
    assert config.disaggregation_mode == DisaggregationMode.DECODE
    assert config.is_prefill_worker is False
    assert config.is_decode_worker is True


def test_legacy_is_prefill_worker_emits_deprecation(mock_vllm_cli):
    """Test that --is-prefill-worker still works but emits DeprecationWarning."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--is-prefill-worker",
        "--kv-transfer-config",
        '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = parse_args()
    deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecation_warnings) >= 1
    assert "deprecated" in str(deprecation_warnings[0].message).lower()
    assert config.disaggregation_mode == DisaggregationMode.PREFILL
    assert config.is_prefill_worker is True


def test_legacy_is_decode_worker_emits_deprecation(mock_vllm_cli):
    """Test that --is-decode-worker still works but emits DeprecationWarning."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--is-decode-worker")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = parse_args()
    deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecation_warnings) >= 1
    assert "deprecated" in str(deprecation_warnings[0].message).lower()
    assert config.disaggregation_mode == DisaggregationMode.DECODE
    assert config.is_decode_worker is True


def test_conflicting_legacy_and_new_flags_raises(mock_vllm_cli):
    """Test that combining legacy flags with explicit --disaggregation-mode raises ValueError."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disaggregation-mode",
        "prefill",
        "--is-decode-worker",
    )
    with pytest.raises(ValueError, match="Cannot combine"):
        parse_args()


def test_explicit_default_mode_with_legacy_flag_raises(mock_vllm_cli):
    """Test that --disaggregation-mode agg --is-decode-worker raises ValueError."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disaggregation-mode",
        "agg",
        "--is-decode-worker",
    )
    with pytest.raises(ValueError, match="Cannot combine"):
        parse_args()


# --- _is_routable tests (pure logic, no mocking) ---


class TestIsRoutable:
    def test_accepts_private_ipv4(self):
        assert _is_routable("10.0.0.5") is True
        assert _is_routable("192.168.1.1") is True

    def test_accepts_private_ipv6(self):
        assert _is_routable("fd00::1") is True

    def test_rejects_loopback_v4(self):
        assert _is_routable("127.0.0.1") is False

    def test_rejects_loopback_v6(self):
        assert _is_routable("::1") is False

    def test_rejects_link_local_v4(self):
        assert _is_routable("169.254.1.1") is False

    def test_rejects_link_local_v6(self):
        assert _is_routable("fe80::1") is False

    def test_rejects_unspecified(self):
        assert _is_routable("0.0.0.0") is False
        assert _is_routable("::") is False

    def test_rejects_multicast(self):
        assert _is_routable("224.0.0.1") is False

    def test_rejects_invalid(self):
        assert _is_routable("not-an-ip") is False


# --- get_host_ip tests (mock socket module functions) ---


class TestGetHostIp:
    def test_hostname_resolution_success(self):
        """getaddrinfo returns routable IPv4 → returns it."""
        with patch(
            "dynamo.vllm.args._try_hostname_resolution", return_value="10.0.0.5"
        ):
            result = get_host_ip()
        assert result == "10.0.0.5"

    def test_hostname_loopback_falls_through_to_udp(self):
        """getaddrinfo returns 127.0.0.1, UDP returns 10.0.0.5 → returns 10.0.0.5."""
        with (
            patch(
                "dynamo.vllm.args._try_hostname_resolution", return_value="127.0.0.1"
            ),
            patch("dynamo.vllm.args._try_udp_connect") as mock_udp,
        ):
            mock_udp.side_effect = lambda family, target: (
                "10.0.0.5" if family == socket.AF_INET else None
            )
            result = get_host_ip()
        assert result == "10.0.0.5"

    def test_hostname_link_local_falls_through_to_udp(self):
        """getaddrinfo returns 169.254.1.1, UDP returns 10.0.0.5 → returns 10.0.0.5."""
        with (
            patch(
                "dynamo.vllm.args._try_hostname_resolution", return_value="169.254.1.1"
            ),
            patch("dynamo.vllm.args._try_udp_connect") as mock_udp,
        ):
            mock_udp.side_effect = lambda family, target: (
                "10.0.0.5" if family == socket.AF_INET else None
            )
            result = get_host_ip()
        assert result == "10.0.0.5"

    def test_ipv6_fallback(self):
        """IPv4 strategies fail, IPv6 UDP returns fd00::1 → returns fd00::1."""
        with (
            patch("dynamo.vllm.args._try_hostname_resolution", return_value=None),
            patch("dynamo.vllm.args._try_udp_connect") as mock_udp,
        ):
            mock_udp.side_effect = lambda family, target: (
                "fd00::1" if family == socket.AF_INET6 else None
            )
            result = get_host_ip()
        assert result == "fd00::1"

    def test_all_fail_raises_runtime_error(self):
        """All strategies fail → RuntimeError with VLLM_NIXL_SIDE_CHANNEL_HOST in message."""
        with (
            patch("dynamo.vllm.args._try_hostname_resolution", return_value=None),
            patch("dynamo.vllm.args._try_udp_connect", return_value=None),
        ):
            with pytest.raises(RuntimeError, match="VLLM_NIXL_SIDE_CHANNEL_HOST"):
                get_host_ip()


# --- ensure_side_channel_host tests ---


class TestEnsureSideChannelHost:
    def test_preserves_existing_env_var(self, monkeypatch):
        """Pre-set env var → verify not overwritten."""
        monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_HOST", "192.168.99.99")
        with patch("dynamo.vllm.args.get_host_ip") as mock_get:
            ensure_side_channel_host()
            mock_get.assert_not_called()
        import os

        assert os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "192.168.99.99"

    def test_sets_env_var_on_successful_detection(self, monkeypatch):
        """No env var set, successful detection populates the side-channel host."""
        monkeypatch.delenv("VLLM_NIXL_SIDE_CHANNEL_HOST", raising=False)
        with patch("dynamo.vllm.args.get_host_ip", return_value="10.0.0.5"):
            ensure_side_channel_host()

        import os

        assert os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "10.0.0.5"

    def test_raises_when_detection_fails_and_no_env(self, monkeypatch):
        """All strategies fail, no env var → RuntimeError."""
        monkeypatch.delenv("VLLM_NIXL_SIDE_CHANNEL_HOST", raising=False)
        with patch(
            "dynamo.vllm.args.get_host_ip",
            side_effect=RuntimeError("Unable to determine"),
        ):
            with pytest.raises(RuntimeError, match="Unable to determine"):
                ensure_side_channel_host()


# --- vllm_omni optional dependency tests ---


class TestVllmOmniOptionalDependency:
    def test_dynamo_vllm_main_importable_without_vllm_omni(self):
        """dynamo.vllm.main must import cleanly even when vllm_omni is absent.

        Setting sys.modules["vllm_omni"] = None blocks ALL imports from the
        vllm_omni package — Python always resolves the top-level package first,
        so a None sentinel at the root raises ImportError for any submodule import.
        """
        # Save and evict any already-cached vllm_omni and dynamo.vllm.omni modules
        saved = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "vllm_omni"
            or k.startswith("vllm_omni.")
            or k == "dynamo.vllm.main"
            or k.startswith("dynamo.vllm.omni")
        }
        # Explicitly block the top-level vllm_omni package regardless of prior imports
        sys.modules["vllm_omni"] = None  # type: ignore[assignment]

        try:
            import dynamo.vllm.main  # noqa: F401
        except ImportError as e:
            pytest.fail(f"dynamo.vllm.main has a hard dependency on vllm_omni: {e}")
        finally:
            sys.modules.pop("vllm_omni", None)
            # Remove any modules imported during this test
            for mod in list(sys.modules):
                if mod == "dynamo.vllm.main" or mod.startswith("dynamo.vllm.omni"):
                    sys.modules.pop(mod, None)
            # Restore original state
            sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Benchmark mode unit tests
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass and grid generation."""

    def test_benchmark_config_defaults(self):
        from dynamo.vllm.instrumented_scheduler import BenchmarkConfig

        cfg = BenchmarkConfig()
        assert cfg.mode == "agg"
        assert cfg.prefill_isl_granularity == 16
        assert cfg.decode_length_granularity == 6
        assert cfg.decode_batch_size_granularity == 6
        assert cfg.warmup_iterations == 5
        assert cfg.output_path == "/tmp/benchmark_results.json"

    def test_benchmark_config_from_dict(self):
        from dynamo.vllm.instrumented_scheduler import BenchmarkConfig

        cfg = BenchmarkConfig(
            mode="decode",
            prefill_isl_granularity=4,
            decode_length_granularity=3,
            decode_batch_size_granularity=3,
            warmup_iterations=2,
            output_path="/tmp/test.json",
        )
        assert cfg.mode == "decode"
        assert cfg.prefill_isl_granularity == 4

    def test_benchmark_config_kwargs_unpack(self):
        from dynamo.vllm.instrumented_scheduler import BenchmarkConfig

        d = {"mode": "prefill", "warmup_iterations": 1}
        cfg = BenchmarkConfig(**d)
        assert cfg.mode == "prefill"
        assert cfg.warmup_iterations == 1
        assert cfg.prefill_isl_granularity == 16


class TestBenchmarkGrid:
    """Tests for benchmark grid generation logic (no GPU required)."""

    def _make_grid_helper(self):
        """Return (prefill_grid_fn, decode_grid_fn) that operate on plain params."""
        import numpy as np

        def generate_prefill_grid(max_num_scheduled_tokens, granularity):
            isls = np.unique(
                np.linspace(10, max_num_scheduled_tokens, granularity, dtype=int)
            )
            return [int(x) for x in isls]

        def generate_decode_grid(
            block_size,
            max_model_len,
            max_num_running_reqs,
            num_gpu_blocks,
            length_granularity,
            batch_granularity,
        ):
            total_kv_tokens = num_gpu_blocks * block_size
            ctx_lens = np.unique(
                np.linspace(block_size, max_model_len, length_granularity, dtype=int)
            )
            points = []
            for ctx_len in ctx_lens:
                ctx_len = int(ctx_len)
                max_batch = min(max_num_running_reqs, total_kv_tokens // ctx_len)
                if max_batch < 1:
                    continue
                batch_sizes = np.unique(
                    np.linspace(1, max_batch, batch_granularity, dtype=int)
                )
                for bs in batch_sizes:
                    points.append((ctx_len, int(bs)))
            return points

        return generate_prefill_grid, generate_decode_grid

    def test_prefill_grid_count(self):
        gen_prefill, _ = self._make_grid_helper()
        isls = gen_prefill(max_num_scheduled_tokens=8192, granularity=10)
        assert len(isls) == 10
        assert isls[0] == 10
        assert isls[-1] == 8192

    def test_prefill_grid_dedup(self):
        gen_prefill, _ = self._make_grid_helper()
        isls = gen_prefill(max_num_scheduled_tokens=20, granularity=100)
        assert len(isls) == len(set(isls))

    def test_decode_grid_batch_capped(self):
        _, gen_decode = self._make_grid_helper()
        points = gen_decode(
            block_size=16,
            max_model_len=4096,
            max_num_running_reqs=64,
            num_gpu_blocks=256,
            length_granularity=3,
            batch_granularity=3,
        )
        total_kv = 256 * 16
        for ctx_len, bs in points:
            assert bs <= min(64, total_kv // ctx_len)
            assert bs >= 1

    def test_decode_grid_skips_large_ctx(self):
        _, gen_decode = self._make_grid_helper()
        points = gen_decode(
            block_size=16,
            max_model_len=100000,
            max_num_running_reqs=64,
            num_gpu_blocks=100,
            length_granularity=5,
            batch_granularity=3,
        )
        total_kv = 100 * 16
        for ctx_len, bs in points:
            assert ctx_len <= total_kv


def test_build_sampling_params_maps_max_thinking_tokens():
    from dynamo.vllm.handlers import build_sampling_params

    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {"max_thinking_tokens": 1024},
        "output_options": {},
    }
    sp = build_sampling_params(request, default_sampling_params={})
    assert sp.thinking_token_budget == 1024
    assert sp.extra_args["reasoning_budget"] == 1024


def test_build_sampling_params_forwards_reasoning_budget_extra_args():
    from dynamo.vllm.handlers import build_sampling_params

    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {},
        "output_options": {},
        "reasoning_budget": "24000",
        "reasoning_budget_grace_period": 16,
        "chat_template_kwargs": {"enable_thinking": True},
        "extra_args": {"request_tag": "kept"},
    }
    sp = build_sampling_params(request, default_sampling_params={})

    assert sp.extra_args["reasoning_budget"] == "24000"
    assert sp.extra_args["reasoning_budget_grace_period"] == 16
    assert sp.extra_args["enable_thinking"] is True
    assert sp.extra_args["request_tag"] == "kept"


def test_build_sampling_params_reasoning_budget_syncs_hidden_eos_stop_ids():
    from dynamo.vllm.handlers import build_sampling_params

    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {"stop_token_ids": [7]},
        "output_options": {},
        "reasoning_budget": 24000,
        "eos_token_ids": [11],
    }
    sp = build_sampling_params(request, default_sampling_params={})

    assert sp.stop_token_ids == [7, 11]
    assert {7, 11}.issubset(sp.all_stop_token_ids)


def test_build_sampling_params_openai_forwards_reasoning_budget_extra_args():
    from dynamo.vllm.handlers import build_sampling_params_openai

    request = {
        "max_tokens": 32,
        "reasoning_budget": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    sp = build_sampling_params_openai(request, default_sampling_params={})

    assert sp.extra_args["reasoning_budget"] == 8
    assert sp.extra_args["enable_thinking"] is False


def test_build_sampling_params_caps_omitted_max_tokens(monkeypatch):
    from dynamo.vllm.handlers import build_sampling_params

    monkeypatch.setenv("DYN_MAX_OUTPUT_TOKENS", "128")
    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {},
        "output_options": {},
    }
    sp = build_sampling_params(
        request, default_sampling_params={}, model_max_len=1_000_000
    )
    assert sp.max_tokens == 128


def test_build_sampling_params_openai_caps_oversized_max_tokens(monkeypatch):
    from dynamo.vllm.handlers import build_sampling_params_openai

    monkeypatch.setenv("DYN_MAX_OUTPUT_LEN", "64")
    sp = build_sampling_params_openai(
        {"max_tokens": 1_000_000}, default_sampling_params={}
    )
    assert sp.max_tokens == 64


def test_apply_reasoning_budget_derives_end_token_ids():
    from vllm.sampling_params import SamplingParams

    from dynamo.vllm.handlers import BaseWorkerHandler

    class FakeTokenizer:
        vocab_size = 2

        def encode(self, text, add_special_tokens=False):
            if text == "</think>":
                return [13]
            return [10]

        def decode(self, token_ids):
            return "\n" if token_ids == [10] else "x"

    class FakeHandler(BaseWorkerHandler):
        def generate(self, request, context):
            raise NotImplementedError

    handler = object.__new__(FakeHandler)
    handler.engine_client = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        vllm_config=SimpleNamespace(
            reasoning_config=SimpleNamespace(reasoning_end_str="</think>")
        ),
    )
    handler.config = SimpleNamespace(engine_args=SimpleNamespace(reasoning_parser=None))

    sp = SamplingParams(extra_args={"reasoning_budget": "8"})
    handler._apply_reasoning_budget_extra_args(sp)

    assert sp.extra_args["reasoning_budget"] == 8
    assert sp.extra_args["reasoning_budget_grace_period"] == 0
    assert sp.extra_args["think_end_token_id"] == 13
    assert sp.extra_args["end_token_ids"] == [13]
    assert 10 in sp.extra_args["newline_token_ids"]


def _make_fake_worker_handler():
    from dynamo.vllm.handlers import BaseWorkerHandler

    class FakeHandler(BaseWorkerHandler):
        def generate(self, request, context):
            raise NotImplementedError

    handler = object.__new__(FakeHandler)
    handler._request_admission_lock = asyncio.Lock()
    handler._admitted_request_ids = set()
    handler._pending_request_admissions = 0
    handler.max_decode_wall_clock_secs = None
    handler.dp_range = (0, 1)
    return handler


@pytest.mark.asyncio
async def test_worker_admission_rejects_at_total_request_limit():
    handler = _make_fake_worker_handler()
    handler.max_total_requests = 2
    handler.engine_client = SimpleNamespace(
        output_processor=SimpleNamespace(get_num_unfinished_requests=lambda: 2)
    )

    reserved, current, limit = await handler._try_reserve_request_slot("req-1", {})

    assert reserved is False
    assert current == 2
    assert limit == 2
    assert handler._pending_request_admissions == 0


@pytest.mark.asyncio
async def test_worker_admission_fails_closed_when_unfinished_count_unavailable():
    handler = _make_fake_worker_handler()
    handler.max_total_requests = 4
    handler.engine_client = SimpleNamespace(output_processor=None)

    reserved, current, limit = await handler._try_reserve_request_slot("req-1", {})

    assert reserved is False
    assert current is None
    assert limit == 4
    assert handler._pending_request_admissions == 0


@pytest.mark.asyncio
async def test_worker_admission_fails_closed_when_unfinished_getter_raises():
    handler = _make_fake_worker_handler()
    handler.max_total_requests = 4

    def _boom():
        raise RuntimeError("output processor unavailable")

    handler.engine_client = SimpleNamespace(
        output_processor=SimpleNamespace(get_num_unfinished_requests=_boom)
    )

    reserved, current, limit = await handler._try_reserve_request_slot("req-2", {})

    assert reserved is False
    assert current is None
    assert limit == 4
    assert handler._pending_request_admissions == 0


@pytest.mark.asyncio
async def test_worker_admission_releases_on_stream_error_before_first_item():
    handler = _make_fake_worker_handler()
    handler.max_total_requests = 4
    handler.engine_client = SimpleNamespace(abort=None)
    handler._admitted_request_ids = {"req-err"}
    handler._pending_request_admissions = 1

    async def stream():
        raise RuntimeError("engine enqueue failed")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="engine enqueue failed"):
        async for _ in handler._iterate_engine_stream(
            stream(), "req-err", release_request_admission=True
        ):
            pass

    assert handler._pending_request_admissions == 0
    assert handler._admitted_request_ids == set()


def test_worker_admission_ignores_per_dp_limit_for_single_local_rank(monkeypatch):
    handler = _make_fake_worker_handler()
    handler.engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=64))
    )
    monkeypatch.setenv("DYN_REQUEST_MAX_TOTAL_REQUESTS_PER_DP", "16")
    monkeypatch.setenv("DYN_REQUEST_MAX_TOTAL_REQUESTS", "32")

    assert handler._configured_max_total_requests() == 32


def test_worker_admission_uses_per_dp_limit_for_multi_local_rank(monkeypatch):
    handler = _make_fake_worker_handler()
    handler.dp_range = (0, 2)
    handler.engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=64))
    )
    monkeypatch.setenv("DYN_REQUEST_MAX_TOTAL_REQUESTS_PER_DP", "16")
    monkeypatch.setenv("DYN_REQUEST_MAX_TOTAL_REQUESTS", "32")

    assert handler._configured_max_total_requests() == 16


@pytest.mark.asyncio
async def test_worker_admission_health_check_bypasses_limit():
    from dynamo.health_check import HEALTH_CHECK_KEY

    handler = _make_fake_worker_handler()
    handler.max_total_requests = 1
    handler.engine_client = SimpleNamespace(
        output_processor=SimpleNamespace(get_num_unfinished_requests=lambda: 99)
    )

    reserved, current, limit = await handler._try_reserve_request_slot(
        "health", {HEALTH_CHECK_KEY: True}
    )

    assert reserved is True
    assert current is None
    assert limit is None
    assert handler._pending_request_admissions == 0


@pytest.mark.asyncio
async def test_worker_admission_holds_until_stream_ends():
    handler = _make_fake_worker_handler()
    handler.max_total_requests = 4
    handler.engine_client = SimpleNamespace(abort=None)
    handler._admitted_request_ids = {"req-1"}
    handler._pending_request_admissions = 1

    async def stream():
        yield "first"
        yield "second"

    observed = []
    async for item in handler._iterate_engine_stream(
        stream(), "req-1", release_request_admission=True
    ):
        observed.append(item)
        # Reservation must still be held mid-stream (waiting+running bound).
        assert handler._pending_request_admissions == 1
        assert "req-1" in handler._admitted_request_ids

    assert observed == ["first", "second"]
    assert handler._pending_request_admissions == 0
    assert handler._admitted_request_ids == set()


@pytest.mark.asyncio
async def test_worker_admission_counts_pending_against_limit():
    handler = _make_fake_worker_handler()
    handler.max_total_requests = 2
    handler._admitted_request_ids = {"req-1"}
    handler._pending_request_admissions = 1
    handler.engine_client = SimpleNamespace(
        output_processor=SimpleNamespace(get_num_unfinished_requests=lambda: 0)
    )

    reserved, current, limit = await handler._try_reserve_request_slot("req-2", {})

    assert reserved is True
    assert current == 2
    assert limit == 2
    assert handler._pending_request_admissions == 2
    assert handler._admitted_request_ids == {"req-1", "req-2"}

    reserved2, current2, _ = await handler._try_reserve_request_slot("req-3", {})
    assert reserved2 is False
    assert current2 == 2
    assert handler._pending_request_admissions == 2


@pytest.mark.asyncio
async def test_worker_admission_rejects_when_engine_reports_zero_but_held_full(
    monkeypatch,
):
    """Prod failure mode: engine unfinished undercounts while Waiting is huge."""
    handler = _make_fake_worker_handler()
    monkeypatch.setenv("DYN_REQUEST_MAX_TOTAL_REQUESTS", "16")
    handler.engine_client = SimpleNamespace(
        output_processor=SimpleNamespace(get_num_unfinished_requests=lambda: 0),
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=16)),
    )
    handler._admitted_request_ids = {f"req-{i}" for i in range(16)}
    handler._pending_request_admissions = 16

    reserved, current, limit = await handler._try_reserve_request_slot("req-17", {})
    assert reserved is False
    assert current == 16
    assert limit == 16
    assert len(handler._admitted_request_ids) == 16


@pytest.mark.asyncio
async def test_worker_admission_releases_on_cancelled_error():
    handler = _make_fake_worker_handler()
    handler.max_total_requests = 4
    handler.engine_client = SimpleNamespace(abort=None)
    handler._admitted_request_ids = {"req-cancel"}
    handler._pending_request_admissions = 1

    async def stream():
        yield "first"
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        async for _ in handler._iterate_engine_stream(
            stream(), "req-cancel", release_request_admission=True
        ):
            pass

    assert handler._pending_request_admissions == 0
    assert handler._admitted_request_ids == set()


def test_configured_max_output_tokens_env_prefers_tokens(monkeypatch):
    from dynamo.vllm import handlers as handlers_mod

    monkeypatch.setenv("DYN_MAX_OUTPUT_TOKENS", "64")
    monkeypatch.setenv("DYN_MAX_OUTPUT_LEN", "128")
    assert handlers_mod._configured_max_output_tokens_env() == 64


def test_configured_max_output_tokens_env_falls_back_to_len(monkeypatch):
    from dynamo.vllm import handlers as handlers_mod

    monkeypatch.delenv("DYN_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.setenv("DYN_MAX_OUTPUT_LEN", "131072")
    assert handlers_mod._configured_max_output_tokens_env() == 131072
