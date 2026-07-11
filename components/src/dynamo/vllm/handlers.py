# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import inspect
import json
import logging
import os

# MM kwargs NIXL transfer (frontend → backend pre-rendered path)
import pickle
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Final, Generic, Optional, TypeVar

import torch
from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange
from vllm.outputs import RequestOutput
from vllm.renderers.embed_utils import safe_load_prompt_embeds
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine.exceptions import EngineDeadError

from dynamo._core import Context
from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal.audio_loader import AudioLoader
from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingReceiver,
    NixlReadEmbeddingReceiver,
    NixlWriteEmbeddingReceiver,
)
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.multimodal.mm_kwargs_transfer import (
    MmKwargsNixlReceiver,
    MmKwargsReceiver,
    MmKwargsShmReceiver,
    MmKwargsShmTransferMetadata,
    MmKwargsTransferMetadata,
)
from dynamo.common.multimodal.video_loader import VideoLoader
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.common.utils.input_params import InputParamManager
from dynamo.common.utils.otel_tracing import build_trace_headers
from dynamo.common.utils.time_section import time_and_log_code_section
from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.llm import (
    KvEventPublisher,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    lora_name_to_id,
    register_model,
    unregister_model,
)
from dynamo.llm.exceptions import EngineShutdown
from dynamo.runtime import Client
from dynamo.runtime.logging import configure_dynamo_logging

from .args import Config
from .constants import DisaggregationMode, EmbeddingTransferMode
from .engine_monitor import VllmEngineMonitor
from .multimodal_utils.hash_utils import compute_mm_uuids_from_images
from .multimodal_utils.model import (
    ModelFamily,
    construct_qwen_decode_mm_data,
    resolve_model_family,
)
from .multimodal_utils.models.qwen import (
    build_qwen_embedding_params,
    load_qwen_grid_params,
)
from .multimodal_utils.prefill_worker_utils import MultiModalEmbeddingLoader

# Multimodal data dictionary keys
IMAGE_URL_KEY: Final = "image_url"
VIDEO_URL_KEY: Final = "video_url"
AUDIO_URL_KEY: Final = "audio_url"
URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"

configure_dynamo_logging()
logger = logging.getLogger(__name__)

DEFAULT_TOOL_CALL_MAX_ARRAY_ITEMS = 8
DEFAULT_SCHEMA_MAX_STRING_LENGTH = 4096
DEFAULT_SCHEMA_MAX_ARRAY_ITEMS = 32
DEFAULT_TOOL_SHORT_TEXT_MAX_LENGTH = 256
DEFAULT_TOOL_LONG_TEXT_MAX_LENGTH = 8192
TOOL_LONG_REQUEST_THRESHOLD = 2048
TOOL_LONG_REQUEST_MARGIN = 512
TOOL_FIELD_STRING_BUDGETS = {
    "expression": 256,
}
TOOL_LONG_TEXT_FIELD_NAMES = {"body", "content", "message"}
SERVICE_OVERLOADED_ERROR_TYPE: Final = "service_overloaded"
QWEN_XML_STRUCTURAL_TAG_TOOL_PARSERS = {"qwen3_coder", "qwen3_xml"}
FORCED_TOOL_STRUCTURAL_TAG_TRIGGERS = ["<tool_call>"]
PARSER_VISIBLE_MARKER_TOKENS: Final = (
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tools>",
    "</tools>",
)
UNSUPPORTED_STRUCTURED_REGEX_TOKENS: Final = ("(?=", "(?!", "(?<=", "(?<!")
TOOL_CHOICE_SCHEMA_MARKER: Final = "x-dynamo-tool-choice-schema"

_GENERATE_REASONING_SUPPORT_CACHE_ATTR = "_dynamo_generate_reasoning_support"


def _build_token_error_response(message: str) -> dict[str, Any]:
    # newtype form: Rust FinishReason::Error(String)
    return {
        "finish_reason": {"error": message},
        "token_ids": [],
    }


def _build_prefill_error_response(message: str) -> dict[str, Any]:
    # newtype form: Rust FinishReason::Error(String)
    return {
        "finish_reason": {"error": message},
        "token_ids": [],
        "disaggregated_params": None,
        "completion_usage": None,
    }


def _build_overload_extra_args(
    message: str,
    *,
    current_total_requests: int | None = None,
    total_request_limit: int | None = None,
) -> dict[str, Any]:
    extra_args: dict[str, Any] = {
        "dynamo_error_type": SERVICE_OVERLOADED_ERROR_TYPE,
        "error_message": message,
    }
    if current_total_requests is not None:
        extra_args["worker_total_requests"] = current_total_requests
    if total_request_limit is not None:
        extra_args["worker_total_request_limit"] = total_request_limit
    return extra_args


def _build_token_overload_response(
    message: str,
    *,
    current_total_requests: int | None = None,
    total_request_limit: int | None = None,
) -> dict[str, Any]:
    return {
        "finish_reason": {"error": message},
        "token_ids": [],
        "extra_args": _build_overload_extra_args(
            message,
            current_total_requests=current_total_requests,
            total_request_limit=total_request_limit,
        ),
    }


def _build_prefill_overload_response(
    message: str,
    *,
    current_total_requests: int | None = None,
    total_request_limit: int | None = None,
) -> dict[str, Any]:
    response = _build_prefill_error_response(message)
    response["finish_reason"] = {"error": message}
    response["extra_args"] = _build_overload_extra_args(
        message,
        current_total_requests=current_total_requests,
        total_request_limit=total_request_limit,
    )
    return response


def _build_text_error_chunk(message: str, request_id: str) -> dict[str, Any]:
    return {
        "id": request_id,
        "created": int(time.time()),
        "object": "chat.completion.chunk",
        "model": "unknown",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": "error",
            }
        ],
        "error": {"message": message},
    }


def _build_text_overload_chunk(
    message: str,
    request_id: str,
    *,
    current_total_requests: int | None = None,
    total_request_limit: int | None = None,
) -> dict[str, Any]:
    chunk = _build_text_error_chunk(message, request_id)
    chunk["extra_args"] = _build_overload_extra_args(
        message,
        current_total_requests=current_total_requests,
        total_request_limit=total_request_limit,
    )
    return chunk


class DecodeWallClockTimeoutError(RuntimeError):
    pass


def _coerce_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (str, bytes)):
        return []
    if not isinstance(value, list):
        return []

    token_ids: list[int] = []
    for item in value:
        try:
            token_ids.append(int(item))
        except Exception:
            continue
    return token_ids


def _merge_stop_token_ids(existing: Any, added: Any) -> list[int]:
    merged: list[int] = []
    seen: set[int] = set()
    for token_id in [*_coerce_token_ids(existing), *_coerce_token_ids(added)]:
        if token_id not in seen:
            merged.append(token_id)
            seen.add(token_id)
    return merged


def _sync_all_stop_token_ids(
    sampling_params: SamplingParams, token_ids: Any
) -> None:
    all_stop_token_ids = getattr(sampling_params, "_all_stop_token_ids", None)
    if isinstance(all_stop_token_ids, set):
        all_stop_token_ids.update(_coerce_token_ids(token_ids))


class _DeferredAbort:
    """Defers engine_client.abort(request_id) until the first engine output.

    In disaggregated decode mode, calling engine_client.abort() while a NIXL
    KV transfer is still in flight on the decode worker can crash EngineCore.
    This guard delays the real abort call until the first generation result
    has been yielded (which, for a decode worker, means the KV transfer has
    completed and the engine has produced at least one token).

    When abort() is called before the first token, a background asyncio.Task
    is spawned to wait for the first-token signal and then perform the real
    abort.
    """

    def __init__(self, engine_client: Any, request_id: str):
        self._engine_client = engine_client
        self._request_id = request_id
        self._first_token_received = False
        self._first_token_event = asyncio.Event()
        # Strong reference to the deferred-abort background task so it is not
        # garbage collected mid-execution (asyncio.create_task only holds a
        # weak reference via the event loop).
        self._abort_task: Optional[asyncio.Task] = None

    def signal_first_token(self) -> None:
        """Called when the first engine output for the request is received."""
        if not self._first_token_received:
            self._first_token_received = True
            self._first_token_event.set()

    async def abort(self) -> None:
        """Abort the request. Creates a Task to hold a strong reference so the
        engine abort cannot be dropped if this coroutine is concurrently
        cancelled."""
        if self._abort_task is None:
            if self._first_token_received:
                logger.debug(
                    f"Deferred abort: first token already received, "
                    f"aborting request {self._request_id} now"
                )
                self._abort_task = asyncio.create_task(self._run_abort())
            else:
                logger.debug(
                    f"Deferred abort: first token not received for request "
                    f"{self._request_id}, spawning background task"
                )
                self._abort_task = asyncio.create_task(self._wait_and_abort())
        try:
            await self._abort_task
        except asyncio.CancelledError:
            logger.debug(
                f"Deferred abort: shielded from cancellation for request "
                f"{self._request_id}, abort continues in background"
            )

    async def _run_abort(self) -> None:
        """Execute engine.abort() and emit the canonical completion log."""
        try:
            await self._engine_client.abort(self._request_id)
            logger.debug(f"Aborted Request ID: {self._request_id}")
        except Exception as e:
            logger.warning(
                f"Deferred abort: engine abort raised for request "
                f"{self._request_id}: {e}"
            )

    async def _wait_and_abort(self) -> None:
        """Background task: wait for first token, then abort."""
        try:
            await self._first_token_event.wait()
        except Exception:
            pass
        await self._run_abort()

    async def close(self) -> None:
        """Clean up the deferred-abort waiter when generation exits.

        Handles case 1b: if abort() was requested before the first token
        arrived AND the generation loop exits without ever producing output,
        the background _wait_and_abort task would otherwise remain parked on
        first_token_event.wait() forever. Cancel it so it does not leak.

        Safety invariant: this method must NOT call engine_client.abort() in
        the pre-first-token window. Issuing abort before the engine has
        produced output is exactly what this guard exists to avoid (it can
        crash EngineCore while a NIXL KV transfer is still in flight on the
        decode worker).
        """
        if self._abort_task is None:
            return

        if not self._first_token_received:
            # Case 1b: cancel the local waiter without firing the real abort.
            self._abort_task.cancel()

        try:
            await self._abort_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(
                f"Deferred abort: cleanup observed error for request "
                f"{self._request_id}: {e}"
            )


@asynccontextmanager
async def _deferred_abort_guard(
    engine_client: Any, request_id: str, is_decode_only: bool
) -> AsyncIterator[Optional[_DeferredAbort]]:
    """Own the _DeferredAbort lifecycle for a single request.

    Yields a _DeferredAbort in disaggregated-decode mode, otherwise yields
    None. On exit, awaits guard.close() so the background waiter cannot leak
    when generation finishes without producing output (case 1b). close() is
    specifically designed not to call engine_client.abort() in the unsafe
    pre-first-token window.
    """
    guard = _DeferredAbort(engine_client, request_id) if is_decode_only else None
    try:
        yield guard
    finally:
        if guard is not None:
            await guard.close()


class VllmEngineQuiesceController:
    def __init__(self, engine_client: Any):
        self._engine_client = engine_client
        self._is_quiesced = False

    @property
    def is_quiesced(self) -> bool:
        return self._is_quiesced

    async def quiesce(self, *args: object) -> bool:
        if self._is_quiesced:
            return False

        level = args[0] if args else None
        await self._engine_client.pause_generation()
        if level is None:
            await self._engine_client.sleep()
        else:
            await self._engine_client.sleep(level)
        self._is_quiesced = True
        return True

    async def resume(self, tags: list[str] | None = None) -> bool:
        if not self._is_quiesced:
            return False

        if tags is None:
            await self._engine_client.wake_up()
        else:
            await self._engine_client.wake_up(tags)
        await self._engine_client.resume_generation()
        return True

    def mark_resumed(self) -> None:
        self._is_quiesced = False


@dataclass(frozen=True)
class LoRAInfo:
    """Metadata for a loaded LoRA adapter."""

    id: int
    path: str


def _compute_mm_uuids(
    multi_modal_data: Dict[str, Any] | None
) -> Dict[str, list[str]] | None:
    """
    Compute multi_modal_uuids from multi_modal_data.

    Each image gets a blake3 hex digest as its UUID (computed by
    compute_mm_uuids_from_images over a fixed-length header + pixel
    preimage), ensuring consistent hashing across the MM Router, vLLM
    handler, and Rust KV publisher.
    """
    if not multi_modal_data or "image" not in multi_modal_data:
        return None
    images = multi_modal_data["image"]
    # [gluo FIXME] Dict being returned when the mm data has been processed,
    # in this case, we skip computing mm_uuids for now until we better understand
    # what info should be hash on.
    if isinstance(images, dict):
        return None
    if not isinstance(images, list):
        images = [images]
    if not images:
        return None
    uuids = compute_mm_uuids_from_images(images)
    return {"image": uuids}


# LoRAManager singleton - initialized lazily when DYN_LORA_ENABLED is set
# None = not yet initialized, False = disabled/failed, LoRAManager = initialized
_lora_manager = None


def get_lora_manager():
    """Get the LoRAManager singleton, initializing it on first call if enabled."""
    global _lora_manager

    if _lora_manager is not None:
        return _lora_manager

    if os.environ.get("DYN_LORA_ENABLED", "").lower() in ("true", "1", "yes"):
        try:
            from dynamo.common.lora import LoRAManager

            _lora_manager = LoRAManager()
            logger.info("LoRAManager initialized successfully")
            return _lora_manager
        except Exception as e:
            logger.warning(
                f"Failed to initialize LoRAManager: {e}. URI-based LoRA loading will be disabled."
            )

    return None


def _normalize_structural_tag(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


def _merge_sampling_extra_args(
    sampling_params: SamplingParams, extra_args: dict[str, Any]
) -> None:
    if not extra_args:
        return
    merged: dict[str, Any] = {}
    existing = getattr(sampling_params, "extra_args", None)
    if isinstance(existing, dict):
        merged.update(existing)
    merged.update(extra_args)
    sampling_params.extra_args = merged


def _reasoning_budget_extra_args(request: Dict[str, Any]) -> dict[str, Any]:
    extra_args: dict[str, Any] = {}
    request_extra_args = request.get("extra_args")
    if isinstance(request_extra_args, dict):
        extra_args.update(request_extra_args)

    chat_template_args = request.get("chat_template_args")
    if not isinstance(chat_template_args, dict):
        chat_template_args = request.get("chat_template_kwargs")
    if not isinstance(chat_template_args, dict):
        chat_template_args = {}

    for key in ("reasoning_budget", "reasoning_budget_grace_period"):
        value = request.get(key)
        if value is None:
            value = chat_template_args.get(key)
        if value is not None:
            extra_args[key] = value

    stop_conditions = request.get("stop_conditions")
    if isinstance(stop_conditions, dict) and "reasoning_budget" not in extra_args:
        max_thinking_tokens = stop_conditions.get("max_thinking_tokens")
        if max_thinking_tokens is not None:
            extra_args["reasoning_budget"] = max_thinking_tokens

    if "enable_thinking" in chat_template_args:
        extra_args["enable_thinking"] = chat_template_args["enable_thinking"]
    return extra_args


def build_sampling_params(
    request: Dict[str, Any],
    default_sampling_params: Dict[str, Any],
    model_max_len: int | None = None,
) -> SamplingParams:
    """
    Build SamplingParams from a PreprocessedRequest (internal protocol format).

    Args:
        request: The PreprocessedRequest dict with 'sampling_options', 'stop_conditions',
                 and 'output_options'
        default_sampling_params: Default sampling parameters to initialize with

    Returns:
        SamplingParams configured from the request
    """
    sampling_params = SamplingParams(**default_sampling_params)
    sampling_params.detokenize = False

    # Handle guided_decoding - convert to StructuredOutputsParams
    sampling_options = request.get("sampling_options", {})
    guided_decoding = sampling_options.get("guided_decoding")
    if guided_decoding is not None and isinstance(guided_decoding, dict):
        structured_outputs = _qwen_xml_structural_tag_for_tool_choice(request)
        if structured_outputs is None:
            structured_outputs = _structured_outputs_from_fields(guided_decoding)
        if structured_outputs is not None:
            sampling_params.structured_outputs = structured_outputs

    # Apply remaining sampling_options
    for key, value in sampling_options.items():
        # Skip guided_decoding - already handled above
        if key == "guided_decoding":
            continue
        if value is not None and hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    # Apply stop_conditions
    for key, value in request.get("stop_conditions", {}).items():
        if value is not None and hasattr(sampling_params, key):
            # Do not add stop key to sampling params - dynamo handles stop conditions directly
            if key == "stop":
                continue
            setattr(sampling_params, key, value)
            if key == "stop_token_ids":
                _sync_all_stop_token_ids(sampling_params, value)
        if (
            key == "stop_token_ids_hidden"
            and value is not None
            and hasattr(sampling_params, "stop_token_ids")
        ):
            sampling_params.stop_token_ids = _merge_stop_token_ids(
                sampling_params.stop_token_ids, value
            )
            _sync_all_stop_token_ids(sampling_params, value)
        # Dynamo's StopConditions uses `max_thinking_tokens`; vLLM 0.20+ exposes
        # the same concept as `thinking_token_budget` on SamplingParams and
        # enforces it via the builtin thinking-budget logits processor.
        if (
            key == "max_thinking_tokens"
            and value is not None
            and hasattr(sampling_params, "thinking_token_budget")
        ):
            sampling_params.thinking_token_budget = value

    _merge_sampling_extra_args(sampling_params, _reasoning_budget_extra_args(request))
    if (
        isinstance(sampling_params.extra_args, dict)
        and "reasoning_budget" in sampling_params.extra_args
        and not sampling_params.ignore_eos
    ):
        eos_token_ids = request.get("eos_token_ids")
        if eos_token_ids is None:
            eos_token_ids = sampling_params.extra_args.get("eos_token_ids")
        if eos_token_ids is not None:
            sampling_params.stop_token_ids = _merge_stop_token_ids(
                sampling_params.stop_token_ids, eos_token_ids
            )
            _sync_all_stop_token_ids(sampling_params, eos_token_ids)

    # Apply output_options (logprobs, prompt_logprobs, etc.)
    output_options = request.get("output_options", {})
    if output_options:
        # Handle logprobs - vLLM expects this as an integer or None
        logprobs_value = output_options.get("logprobs")
        if logprobs_value is not None and logprobs_value != "":
            try:
                parsed_logprobs = int(logprobs_value)
                if parsed_logprobs < 0:
                    logger.warning(
                        f"Invalid logprobs value: {logprobs_value} (must be non-negative), ignoring"
                    )
                else:
                    sampling_params.logprobs = parsed_logprobs
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid logprobs value: {logprobs_value} (must be integer), ignoring"
                )

        # Handle prompt_logprobs - vLLM expects this as an integer or None
        prompt_logprobs_value = output_options.get("prompt_logprobs")
        if prompt_logprobs_value is not None and prompt_logprobs_value != "":
            try:
                parsed_prompt_logprobs = int(prompt_logprobs_value)
                if parsed_prompt_logprobs < 0:
                    logger.warning(
                        f"Invalid prompt_logprobs value: {prompt_logprobs_value} (must be non-negative), ignoring"
                    )
                else:
                    sampling_params.prompt_logprobs = parsed_prompt_logprobs
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid prompt_logprobs value: {prompt_logprobs_value} (must be integer), ignoring"
                )

    # If max_tokens wasn't provided (None or missing), compute a dynamic default
    provided_max_tokens = request.get("stop_conditions", {}).get("max_tokens", None)
    token_ids = request.get("token_ids", [])
    input_length = len(token_ids)
    if model_max_len is not None and (provided_max_tokens is None):
        # Ensure at least 1 token generation by default when possible
        dynamic_default = max(1, model_max_len - input_length)
        sampling_params.max_tokens = dynamic_default

    # Defense-in-depth: honor DYN_MAX_OUTPUT_TOKENS / DYN_MAX_OUTPUT_LEN.
    output_cap = _configured_max_output_tokens_env()
    if output_cap is not None:
        current = getattr(sampling_params, "max_tokens", None)
        if current is None or current > output_cap:
            sampling_params.max_tokens = output_cap

    return sampling_params


_OUTPUT_CAP_ENV_UNSET = object()
_OUTPUT_CAP_ENV_CACHE: int | None | object = _OUTPUT_CAP_ENV_UNSET


def _configured_max_output_tokens_env() -> int | None:
    """Cached process-wide output cap (matches Rust OnceLock)."""
    global _OUTPUT_CAP_ENV_CACHE
    if _OUTPUT_CAP_ENV_CACHE is not _OUTPUT_CAP_ENV_UNSET:
        return _OUTPUT_CAP_ENV_CACHE  # type: ignore[return-value]
    result: int | None = None
    for name in ("DYN_MAX_OUTPUT_TOKENS", "DYN_MAX_OUTPUT_LEN"):
        raw = os.environ.get(name)
        if not raw:
            continue
        try:
            parsed = int(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", name, raw)
            continue
        if parsed > 0:
            result = parsed
            break
    _OUTPUT_CAP_ENV_CACHE = result
    return result


def _clear_configured_max_output_tokens_env_cache() -> None:
    """Test helper: allow monkeypatched env to take effect."""
    global _OUTPUT_CAP_ENV_CACHE
    _OUTPUT_CAP_ENV_CACHE = _OUTPUT_CAP_ENV_UNSET


def _value_from_mapping_or_object(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def bound_json_schema_for_constrained_decoding(
    schema: Any,
    field_name: str | None = None,
    field_string_budgets: dict[str, int] | None = None,
    request_text_len: int | None = None,
) -> Any:
    if isinstance(schema, list):
        return [
            bound_json_schema_for_constrained_decoding(
                item, field_name, field_string_budgets, request_text_len
            )
            for item in schema
        ]
    if not isinstance(schema, dict):
        return schema

    bounded = copy.deepcopy(schema)
    bounded.pop(TOOL_CHOICE_SCHEMA_MARKER, None)
    for key in ("format", "pattern"):
        if bounded.get(key) == "":
            bounded.pop(key)
    pattern = bounded.get("pattern")
    if isinstance(pattern, str) and ".*" in pattern:
        bounded.pop("pattern")

    schema_type = bounded.get("type")
    schema_types = schema_type if isinstance(schema_type, list) else [schema_type]
    if (
        "string" in schema_types
        and "maxLength" not in bounded
        and "enum" not in bounded
        and "const" not in bounded
    ):
        max_length = DEFAULT_SCHEMA_MAX_STRING_LENGTH
        if field_string_budgets is not None and field_name is not None:
            max_length = field_string_budgets.get(field_name, max_length)
        if field_name in TOOL_LONG_TEXT_FIELD_NAMES:
            max_length = _tool_long_text_budget(request_text_len)
        bounded["maxLength"] = max_length
    if "array" in schema_types and "maxItems" not in bounded:
        bounded["maxItems"] = DEFAULT_SCHEMA_MAX_ARRAY_ITEMS

    for key in ("properties", "$defs", "definitions", "patternProperties"):
        value = bounded.get(key)
        if isinstance(value, dict):
            bounded[key] = {
                name: bound_json_schema_for_constrained_decoding(
                    subschema,
                    name if key == "properties" else None,
                    field_string_budgets,
                    request_text_len,
                )
                for name, subschema in value.items()
            }

    for key in ("items", "additionalProperties"):
        value = bounded.get(key)
        if isinstance(value, (dict, list)):
            bounded[key] = bound_json_schema_for_constrained_decoding(
                value, field_name, field_string_budgets, request_text_len
            )

    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        value = bounded.get(key)
        if isinstance(value, list):
            bounded[key] = [
                bound_json_schema_for_constrained_decoding(
                    subschema, field_name, field_string_budgets, request_text_len
                )
                for subschema in value
            ]

    return bounded


def _tool_long_text_budget(request_text_len: int | None) -> int:
    if request_text_len is None or request_text_len < TOOL_LONG_REQUEST_THRESHOLD:
        return DEFAULT_TOOL_SHORT_TEXT_MAX_LENGTH
    return max(
        DEFAULT_TOOL_SHORT_TEXT_MAX_LENGTH,
        min(DEFAULT_TOOL_LONG_TEXT_MAX_LENGTH, request_text_len + TOOL_LONG_REQUEST_MARGIN),
    )


def _request_text_len(request: Dict[str, Any]) -> int | None:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return None
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    total += len(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        total += len(text)
    return total


def _schema_has_freeform_long_text_field(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, subschema in properties.items():
            if (
                name in TOOL_LONG_TEXT_FIELD_NAMES
                and _schema_type_includes(subschema, "string")
                and isinstance(subschema, dict)
                and "enum" not in subschema
                and "const" not in subschema
            ):
                return True
            if _schema_has_freeform_long_text_field(subschema):
                return True

    for key in ("items", "additionalProperties", "anyOf", "oneOf", "allOf"):
        value = schema.get(key)
        if isinstance(value, list):
            if any(_schema_has_freeform_long_text_field(item) for item in value):
                return True
        elif isinstance(value, dict) and _schema_has_freeform_long_text_field(value):
            return True

    return False


def _schema_type_includes(schema: Any, expected: str) -> bool:
    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return expected in schema_type
    return schema_type == expected


def _tool_function(tool: Any) -> Any:
    return _value_from_mapping_or_object(tool, "function", {})


def _tool_name(tool: Any) -> str | None:
    return _value_from_mapping_or_object(_tool_function(tool), "name")


def _tool_parameters(tool: Any) -> dict[str, Any]:
    params = _value_from_mapping_or_object(_tool_function(tool), "parameters")
    if isinstance(params, dict):
        return copy.deepcopy(params)
    return {"type": "object", "properties": {}}


def _named_tool_choice_name(tool_choice: Any) -> str | None:
    function = _value_from_mapping_or_object(tool_choice, "function")
    if function is None:
        return None
    return _value_from_mapping_or_object(function, "name")


def _complete_json_text(text: str) -> str | None:
    candidate = text.strip()
    if not candidate:
        return None
    try:
        _, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return None
    if candidate[end:].strip():
        return None
    return candidate[:end]


def _forced_tool_calls_from_post_reasoning_json(
    request: Dict[str, Any],
    content: str,
) -> list[dict[str, Any]] | None:
    tools = request.get("tools")
    tool_choice = request.get("tool_choice")
    if not tools or tool_choice in (None, "none", "auto"):
        return None
    if any(
        request.get(key) is not None
        for key in (
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_grammar",
            "structured_outputs",
            "response_format",
        )
    ):
        return None

    _, end_marker, post_reasoning = content.rpartition("</think>")
    arguments = _complete_json_text(post_reasoning if end_marker else content)
    if arguments is None:
        return None

    named_tool = _named_tool_choice_name(tool_choice)
    if named_tool is not None:
        return [
            {
                "index": 0,
                "id": make_tool_call_id(),
                "type": "function",
                "function": {
                    "name": named_tool,
                    "arguments": arguments,
                },
            }
        ]

    if tool_choice != "required":
        return None
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return None

    items = decoded if isinstance(decoded, list) else [decoded]
    tool_calls: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        name = _value_from_mapping_or_object(item, "name")
        parameters = _value_from_mapping_or_object(item, "parameters", {})
        if not isinstance(name, str):
            continue
        if not isinstance(parameters, str):
            parameters = json.dumps(parameters, ensure_ascii=False)
        tool_calls.append(
            {
                "index": index,
                "id": make_tool_call_id(),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": parameters,
                },
            }
        )
    return tool_calls or None


def _schema_for_required_tool_choice(
    tools: list[Any],
    request_text_len: int | None = None,
) -> dict[str, Any]:
    if not tools:
        raise ValueError("tool_choice='required' needs at least one tool")

    defs: dict[str, Any] = {}
    any_of: list[dict[str, Any]] = []
    for tool in tools:
        name = _tool_name(tool)
        if not name:
            continue

        params = _tool_parameters(tool)
        tool_defs = params.pop("$defs", None)
        if isinstance(tool_defs, dict):
            for def_name, def_schema in tool_defs.items():
                if def_name in defs and defs[def_name] != def_schema:
                    raise ValueError(
                        f"Tool definition '{def_name}' has multiple schemas"
                    )
                defs[def_name] = def_schema

        any_of.append(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": [name]},
                    "parameters": params,
                },
                "required": ["name", "parameters"],
                "additionalProperties": False,
            }
        )

    if not any_of:
        raise ValueError("tool_choice='required' needs at least one named tool")

    schema: dict[str, Any] = {
        "type": "array",
        "minItems": 1,
        "items": {
            "anyOf": any_of,
        },
    }
    if defs:
        schema["$defs"] = defs
    return bound_json_schema_for_constrained_decoding(
        schema,
        field_string_budgets=TOOL_FIELD_STRING_BUDGETS,
        request_text_len=request_text_len,
    )


def _schema_for_tool_choice(request: Dict[str, Any]) -> dict[str, Any] | None:
    tools = request.get("tools")
    tool_choice = request.get("tool_choice")
    if tool_choice in (None, "none", "auto") or not tools:
        return None

    if tool_choice == "required":
        return _schema_for_required_tool_choice(tools, _request_text_len(request))

    if isinstance(tool_choice, dict) or _named_tool_choice_name(tool_choice):
        tool_name = _named_tool_choice_name(tool_choice)
        if not tool_name:
            return None
        for tool in tools:
            if _tool_name(tool) == tool_name:
                return bound_json_schema_for_constrained_decoding(
                    _tool_parameters(tool),
                    field_string_budgets=TOOL_FIELD_STRING_BUDGETS,
                    request_text_len=_request_text_len(request),
                )
        raise ValueError(f"Tool '{tool_name}' has not been passed in `tools`")

    return None


def _tool_call_parser_name(parser_name: str | None = None) -> str:
    return (parser_name or "").strip() or (
        os.environ.get("DYN_DYNAMO_TOOL_CALL_PARSER")
        or os.environ.get("DYN_TOOL_CALL_PARSER")
        or ""
    ).strip()


def _qwen_xml_tool_tag(
    tool: Any,
    request_text_len: int | None = None,
) -> dict[str, Any]:
    name = _tool_name(tool)
    if not name:
        raise ValueError("Tool choice needs a named tool")
    return {
        "type": "tag",
        "begin": f"<tool_call>\n<function={name}>\n",
        "content": {
            "type": "json_schema",
            "json_schema": bound_json_schema_for_constrained_decoding(
                _tool_parameters(tool),
                field_string_budgets=TOOL_FIELD_STRING_BUDGETS,
                request_text_len=request_text_len,
            ),
            "style": "qwen_xml",
        },
        "end": "\n</function>\n</tool_call>",
    }


def _qwen_xml_structural_tag_for_tool_choice(
    request: Dict[str, Any],
    tool_call_parser_name: str | None = None,
) -> StructuredOutputsParams | None:
    if (
        _tool_call_parser_name(tool_call_parser_name)
        not in QWEN_XML_STRUCTURAL_TAG_TOOL_PARSERS
    ):
        return None

    tools = request.get("tools")
    tool_choice = request.get("tool_choice")
    if not tools or tool_choice in (None, "none", "auto"):
        return None
    request_text_len = _request_text_len(request)

    if tool_choice == "required":
        tags = [
            _qwen_xml_tool_tag(tool, request_text_len)
            for tool in tools
            if _tool_name(tool)
        ]
        if not tags:
            raise ValueError("tool_choice='required' needs at least one named tool")
        if len(tags) != 1:
            return None
        structural_tag = {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "tags": tags[:DEFAULT_TOOL_CALL_MAX_ARRAY_ITEMS],
                "triggers": FORCED_TOOL_STRUCTURAL_TAG_TRIGGERS,
                "stop_after_first": False,
            },
        }
        return StructuredOutputsParams(structural_tag=json.dumps(structural_tag))

    tool_name = _named_tool_choice_name(tool_choice)
    if not tool_name:
        return None
    for tool in tools:
        if _tool_name(tool) == tool_name:
            if _schema_has_freeform_long_text_field(_tool_parameters(tool)):
                return None
            structural_tag = {
                "type": "structural_tag",
                "format": {
                    "type": "triggered_tags",
                    "tags": [_qwen_xml_tool_tag(tool, request_text_len)],
                    "triggers": FORCED_TOOL_STRUCTURAL_TAG_TRIGGERS,
                    "stop_after_first": True,
                },
            }
            return StructuredOutputsParams(structural_tag=json.dumps(structural_tag))
    raise ValueError(f"Tool '{tool_name}' has not been passed in `tools`")


def _validate_structured_regex(pattern: Any) -> None:
    if not isinstance(pattern, str):
        return
    if any(token in pattern for token in UNSUPPORTED_STRUCTURED_REGEX_TOKENS):
        raise ValueError(
            "Unsupported structured output regex: lookaround assertions are not "
            "supported by the configured guided-decoding backend"
        )


def _validate_structured_json_schema(schema: Any) -> None:
    if isinstance(schema, dict):
        if schema.get(TOOL_CHOICE_SCHEMA_MARKER) is True:
            return

        pattern = schema.get("pattern")
        if pattern is not None:
            _validate_structured_regex(pattern)

        pattern_properties = schema.get("patternProperties")
        if isinstance(pattern_properties, dict):
            for property_pattern in pattern_properties:
                _validate_structured_regex(property_pattern)

        for value in schema.values():
            _validate_structured_json_schema(value)
    elif isinstance(schema, list):
        for item in schema:
            _validate_structured_json_schema(item)


def _validate_structured_grammar(grammar: Any) -> None:
    if not isinstance(grammar, str):
        return

    in_char_class = False
    escaped = False
    for char in grammar:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if in_char_class:
            if char in "\r\n":
                raise ValueError(
                    "Unsupported structured output grammar: raw newlines inside "
                    "character classes are not supported"
                )
            if char == "]":
                in_char_class = False
            continue
        if char == "[":
            in_char_class = True


def _structured_outputs_from_fields(fields: Any) -> StructuredOutputsParams | None:
    if isinstance(fields, StructuredOutputsParams):
        return fields
    if not isinstance(fields, dict):
        return None

    params: dict[str, Any] = {}
    if fields.get("json") is not None:
        schema = fields["json"]
        _validate_structured_json_schema(schema)
        if isinstance(schema, dict) and schema.get(TOOL_CHOICE_SCHEMA_MARKER) is True:
            params["json"] = bound_json_schema_for_constrained_decoding(schema)
        else:
            # User structured-output schemas should stay equivalent to vLLM's
            # OpenAI path. Tool-choice schemas are explicitly marked above.
            params["json"] = schema
    for key in ("regex", "choice", "grammar", "json_object"):
        if fields.get(key) is not None:
            if key == "regex":
                _validate_structured_regex(fields[key])
            elif key == "grammar":
                _validate_structured_grammar(fields[key])
            params[key] = fields[key]
    if fields.get("structural_tag") is not None:
        params["structural_tag"] = _normalize_structural_tag(fields["structural_tag"])
    if not params:
        return None
    for key in (
        "disable_any_whitespace",
        "disable_additional_properties",
        "whitespace_pattern",
    ):
        if fields.get(key) is not None:
            params[key] = fields[key]
    return StructuredOutputsParams(**params)


def _is_required_single_tool_repeat_structural_tag(
    value: Any,
    request: Dict[str, Any],
) -> bool:
    if request.get("tool_choice") != "required":
        return False

    tool_names = [_tool_name(tool) for tool in request.get("tools") or []]
    tool_names = [name for name in tool_names if name]
    if len(tool_names) != 1:
        return False

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    if not isinstance(value, dict):
        return False

    tag_format = value.get("format")
    if not isinstance(tag_format, dict) or tag_format.get("type") != "repeat":
        return False
    if tag_format.get("min") != 1 or tag_format.get("max") != 1:
        return False

    content = tag_format.get("content")
    if not isinstance(content, dict) or content.get("type") != "or":
        return False
    elements = content.get("elements")
    if not isinstance(elements, list) or len(elements) != 1:
        return False
    expected_begin = f"<tool_call>\n<function={tool_names[0]}>\n"
    return isinstance(elements[0], dict) and elements[0].get("begin") == expected_begin


def _request_has_pure_json_structured_output(request: dict[str, Any]) -> bool:
    if request.get("tools"):
        return False

    guided_decoding = request.get("guided_decoding")
    if isinstance(guided_decoding, dict) and (
        guided_decoding.get("json") is not None
        or guided_decoding.get("json_object") is not None
    ):
        return True

    sampling_options = request.get("sampling_options")
    if isinstance(sampling_options, dict):
        guided_decoding = sampling_options.get("guided_decoding")
        if isinstance(guided_decoding, dict) and (
            guided_decoding.get("json") is not None
            or guided_decoding.get("json_object") is not None
        ):
            return True

    if request.get("guided_json") is not None:
        return True

    structured_outputs = request.get("structured_outputs")
    if isinstance(structured_outputs, dict) and (
        structured_outputs.get("json") is not None
        or structured_outputs.get("json_object") is not None
    ):
        return True

    response_format = request.get("response_format")
    if isinstance(response_format, dict):
        return response_format.get("type") in {"json_schema", "json_object"}
    return False


class _StructuredJsonStreamGuard:
    def __init__(self) -> None:
        self._buffer = ""
        self._emitted = False

    @property
    def emitted(self) -> bool:
        return self._emitted

    def filter_delta(self, delta_text: str) -> str:
        if self._emitted:
            return ""
        self._buffer += delta_text
        candidate = self._buffer.lstrip()
        if not candidate:
            return ""
        try:
            _, end = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            return ""
        self._emitted = True
        return candidate[:end]


def _structured_outputs_from_openai_request(
    request: Dict[str, Any],
    tool_call_parser_name: str | None = None,
) -> StructuredOutputsParams | None:
    guided_decoding = request.get("guided_decoding")
    tool_structural_tag = _qwen_xml_structural_tag_for_tool_choice(
        request, tool_call_parser_name
    )
    guided_outputs = _structured_outputs_from_fields(guided_decoding)
    if (
        guided_outputs is not None
        and isinstance(guided_decoding, dict)
        and guided_decoding.get("structural_tag") is not None
    ):
        if tool_structural_tag is not None and _is_required_single_tool_repeat_structural_tag(
            guided_decoding.get("structural_tag"),
            request,
        ):
            return tool_structural_tag
        return guided_outputs

    if tool_structural_tag is not None:
        return tool_structural_tag

    if guided_outputs is not None:
        return guided_outputs

    if request.get("guided_json") is not None:
        _validate_structured_json_schema(request["guided_json"])
        return StructuredOutputsParams(
            json=bound_json_schema_for_constrained_decoding(request["guided_json"])
        )
    if request.get("guided_regex") is not None:
        _validate_structured_regex(request["guided_regex"])
        return StructuredOutputsParams(regex=request["guided_regex"])
    if request.get("guided_choice") is not None:
        return StructuredOutputsParams(choice=request["guided_choice"])
    if request.get("guided_grammar") is not None:
        _validate_structured_grammar(request["guided_grammar"])
        return StructuredOutputsParams(grammar=request["guided_grammar"])

    tool_schema = _schema_for_tool_choice(request)
    if tool_schema is not None:
        return StructuredOutputsParams(json=tool_schema)

    structured_outputs = _structured_outputs_from_fields(
        request.get("structured_outputs")
    )
    if structured_outputs is not None:
        return structured_outputs

    response_format = request.get("response_format")
    if isinstance(response_format, dict):
        response_type = response_format.get("type")
        if response_type == "json_object":
            return StructuredOutputsParams(json_object=True)
        if response_type == "json_schema":
            json_schema = response_format.get("json_schema") or {}
            schema = json_schema.get("schema")
            if schema is not None:
                _validate_structured_json_schema(schema)
                return StructuredOutputsParams(
                    json=bound_json_schema_for_constrained_decoding(schema)
                )

    return None


def build_sampling_params_openai(
    request: Dict[str, Any],
    default_sampling_params: Dict[str, Any],
    tool_call_parser_name: str | None = None,
) -> SamplingParams:
    """
    Build SamplingParams from an OpenAI-compatible request format.

    Args:
        request: The OpenAI-style request dict with parameters like temperature, max_tokens, etc.
        default_sampling_params: Default sampling parameters to initialize with

    Returns:
        SamplingParams configured from the request
    """
    sampling_params = SamplingParams(**default_sampling_params)
    sampling_params.detokenize = True

    # Map common OpenAI parameters to SamplingParams
    openai_mapping = {
        "n": "n",
        "temperature": "temperature",
        "top_p": "top_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "seed": "seed",
        "top_k": "top_k",
        "repetition_penalty": "repetition_penalty",
        "min_p": "min_p",
        "length_penalty": "length_penalty",
        "use_beam_search": "use_beam_search",
    }

    for req_key, param_key in openai_mapping.items():
        if req_key in request and request[req_key] is not None:
            if hasattr(sampling_params, param_key):
                setattr(sampling_params, param_key, request[req_key])

    # Handle max_tokens
    if "max_tokens" in request and request["max_tokens"] is not None:
        sampling_params.max_tokens = request["max_tokens"]

    output_cap = _configured_max_output_tokens_env()
    if output_cap is not None:
        current = getattr(sampling_params, "max_tokens", None)
        if current is None or current > output_cap:
            sampling_params.max_tokens = output_cap

    # Handle stop sequences
    if "stop" in request and request["stop"] is not None:
        sampling_params.stop = request["stop"]

    # Handle ignore_eos (custom extension)
    if "ignore_eos" in request and request["ignore_eos"] is not None:
        sampling_params.ignore_eos = request["ignore_eos"]

    # Handle min_tokens (custom extension)
    if "min_tokens" in request and request["min_tokens"] is not None:
        sampling_params.min_tokens = request["min_tokens"]

    structured_outputs = _structured_outputs_from_openai_request(
        request, tool_call_parser_name
    )
    if structured_outputs is not None:
        sampling_params.structured_outputs = structured_outputs

    _merge_sampling_extra_args(sampling_params, _reasoning_budget_extra_args(request))

    return sampling_params


def _engine_generate_reasoning_kwargs(
    engine_client: Any,
    reasoning_ended: bool | None,
    reasoning_parser_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    if reasoning_ended is None and reasoning_parser_kwargs is None:
        return {}

    support = _engine_generate_reasoning_support(engine_client)
    if support is None:
        return {}
    accepts_reasoning_ended, accepts_reasoning_parser_kwargs = support

    kwargs: dict[str, Any] = {}
    if accepts_reasoning_ended:
        kwargs["reasoning_ended"] = reasoning_ended
    if accepts_reasoning_parser_kwargs:
        kwargs["reasoning_parser_kwargs"] = reasoning_parser_kwargs

    if not kwargs:
        logger.debug(
            "vLLM generate does not accept reasoning parser kwargs; "
            "running without request-local reasoning parser metadata"
        )
    return kwargs


def _engine_generate_reasoning_support(
    engine_client: Any,
) -> tuple[bool, bool] | None:
    try:
        cached = vars(engine_client).get(_GENERATE_REASONING_SUPPORT_CACHE_ATTR)
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    try:
        parameters = inspect.signature(engine_client.generate).parameters
    except (TypeError, ValueError):
        logger.debug(
            "Unable to inspect vLLM generate signature; dropping reasoning parser kwargs"
        )
        return None

    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    support = (
        accepts_kwargs or "reasoning_ended" in parameters,
        accepts_kwargs or "reasoning_parser_kwargs" in parameters,
    )
    try:
        setattr(engine_client, _GENERATE_REASONING_SUPPORT_CACHE_ATTR, support)
    except Exception:
        pass
    return support


def _request_reasoning_metadata(
    request: dict[str, Any],
) -> tuple[bool | None, dict[str, Any] | None]:
    reasoning_ended = request.get("reasoning_ended")
    reasoning_parser_kwargs = request.get("reasoning_parser_kwargs")

    extra_args = request.get("extra_args")
    if isinstance(extra_args, dict):
        if reasoning_ended is None:
            reasoning_ended = extra_args.get("reasoning_ended")
        if reasoning_parser_kwargs is None:
            reasoning_parser_kwargs = extra_args.get("reasoning_parser_kwargs")

    return reasoning_ended, reasoning_parser_kwargs


def get_dp_range_for_worker(vllm_config: VllmConfig) -> tuple[int, int]:
    """
    Get the global DP rank range that this worker is responsible for based on vLLM config.
    Note that the 'vllm_config' is normalized so the load balancing flags are set properly.
    The return value is in the format of (start_dp_rank, managed_dp_size)."""
    if vllm_config.parallel_config.data_parallel_external_lb:
        # external load balancing, each worker is responsible for exactly 1 rank
        return (vllm_config.parallel_config.data_parallel_rank, 1)
    elif vllm_config.parallel_config.data_parallel_hybrid_lb:
        # hybrid load balancing, each worker is responsible for a subset of local ranks
        return (
            vllm_config.parallel_config.data_parallel_rank,
            vllm_config.parallel_config.data_parallel_size_local,
        )
    else:
        # internal load balancing, the worker is responsible for all DP ranks
        logger.warning(
            "vLLM selects internal DP load balancing. If you are launching multiple workers for DP deployment,"
            " hybrid or external load balancing is recommended."
        )
        return (
            vllm_config.parallel_config.data_parallel_rank,
            vllm_config.parallel_config.data_parallel_size,
        )


RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class BaseWorkerHandler(ABC, Generic[RequestT, ResponseT]):
    """
    Request handler for the generate and clear_kv_blocks endpoints.
    """

    _benchmark_results: Optional[dict] = None
    # Class-level default so test doubles that bypass __init__ via
    # __new__ still have a sane value; __init__ overrides this from
    # hf_config.use_unified_vision_chunk on real instances.
    _use_unified_vision_chunk: bool = False

    def __init__(
        self,
        runtime,
        config: Config,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        model_config: ModelConfig | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
        encode_worker_client: Optional[Client] = None,
    ):
        self.runtime = runtime
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params
        self.kv_publishers: list[KvEventPublisher] | None = None
        self.fpm_relays: list | None = None
        self.generate_endpoint = generate_endpoint
        self.config = config
        self.engine_monitor = VllmEngineMonitor(runtime, engine, shutdown_event)
        self.temp_dirs: list[tempfile.TemporaryDirectory] = []
        self.model_max_len = model_max_len
        self.model_config = model_config
        self.enable_multimodal = enable_multimodal
        # LoRA tracking: name -> LoRAInfo(id, path)
        self.loaded_loras: dict[str, LoRAInfo] = {}
        # Per-LoRA locks to prevent concurrent load operations for the same LoRA
        self._lora_load_locks: dict[str, asyncio.Lock] = {}
        # Guard lock-map access in case handlers are invoked from multiple threads.
        self._lora_load_locks_guard = threading.Lock()

        self.image_loader = ImageLoader(
            enable_frontend_decoding=enable_frontend_decoding
        )
        self.audio_loader = AudioLoader(
            enable_frontend_decoding=enable_frontend_decoding
        )
        video_io_kwargs = None
        multimodal_config = getattr(model_config, "multimodal_config", None)
        media_io_kwargs = getattr(multimodal_config, "media_io_kwargs", None)
        if isinstance(media_io_kwargs, dict):
            configured_video_kwargs = media_io_kwargs.get("video")
            if isinstance(configured_video_kwargs, dict):
                video_io_kwargs = configured_video_kwargs
        self.video_loader = VideoLoader(
            enable_frontend_decoding=enable_frontend_decoding,
            media_io_kwargs=video_io_kwargs,
        )
        self.embedding_loader = self.init_embedding_loader(config, encode_worker_client)

        self.use_vllm_tokenizer = use_vllm_tokenizer

        self.dp_range = get_dp_range_for_worker(self.engine_client.vllm_config)
        self._quiesce_controller = VllmEngineQuiesceController(self.engine_client)
        self._quiesce_lock = asyncio.Lock()
        self._mm_kwargs_receiver: MmKwargsNixlReceiver | None = None

        # Some models (Kimi-K2.5) declare their image modality as
        # "vision_chunk" rather than "image". vLLM's openai entrypoint
        # renames the dict key + wraps images via the chat_utils tracker,
        # but dynamo bypasses chat_utils and builds `multi_modal_data`
        # directly — so we mirror the rename + wrap here. The flag lives
        # on the model's HF config; defaults to False for all other
        # families.
        self._use_unified_vision_chunk = bool(
            getattr(
                self.engine_client.vllm_config.model_config.hf_config,
                "use_unified_vision_chunk",
                False,
            )
        )

        # Serialise concurrent scale_elastic_ep calls.  vLLM's elastic-EP
        # bootstrap creates a fresh TCPStore per scale operation and stores it
        # in engine_client._coord_store.  When two callers race through
        # _setup_elastic_ep_reconfig_bootstrap concurrently the first caller's
        # store gets garbage-collected before the new Ray actor has had a chance
        # to connect, causing a 300 s TCPStore timeout on the worker node.
        # One handler is created per worker process (worker_factory.py), so all
        # concurrent HTTP callers share this lock and only one scale operation
        # can mutate _coord_store at a time.
        self._scale_ep_lock = asyncio.Lock()

        # Initialize InputParamManager for text-in-text-out mode
        tokenizer = None
        if use_vllm_tokenizer and hasattr(engine, "tokenizer"):
            tokenizer = engine.tokenizer
        self.input_param_manager = InputParamManager(tokenizer)

        # Store shutdown event for graceful shutdown monitoring
        self.shutdown_event = shutdown_event
        self.max_decode_wall_clock_secs = self._get_positive_float_env(
            "DYN_REQUEST_MAX_DECODE_WALL_CLOCK_SECS"
        )
        self._request_admission_lock = asyncio.Lock()
        # Request-id set is the source of truth (no counter drift on double-release).
        self._admitted_request_ids: set[str] = set()
        self._pending_request_admissions = 0  # len(_admitted_request_ids); tests/compat
        self.max_total_requests = self._configured_max_total_requests(
            log_ignored_per_dp=True
        )

    @staticmethod
    def _get_positive_float_env(name: str) -> float | None:
        value = os.environ.get(name)
        if value in (None, ""):
            return None
        try:
            parsed = float(value)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", name, value)
            return None
        if parsed <= 0:
            return None
        return parsed

    @staticmethod
    def _get_positive_int_env(name: str) -> int | None:
        value = os.environ.get(name)
        if value in (None, ""):
            return None
        try:
            parsed = int(value)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", name, value)
            return None
        if parsed <= 0:
            return None
        return parsed

    def _get_default_max_total_requests(self) -> int | None:
        scheduler_config = getattr(
            getattr(self.engine_client, "vllm_config", None),
            "scheduler_config",
            None,
        )
        value = getattr(scheduler_config, "max_num_seqs", None)
        if value in (None, ""):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid scheduler max_num_seqs=%r", value)
            return None
        if parsed <= 0:
            return None
        return parsed

    def _configured_max_total_requests(self, *, log_ignored_per_dp: bool = False) -> int | None:
        per_dp_limit = self._get_positive_int_env("DYN_REQUEST_MAX_TOTAL_REQUESTS_PER_DP")
        total_limit = self._get_positive_int_env("DYN_REQUEST_MAX_TOTAL_REQUESTS")
        local_dp_size = self.dp_range[1] if len(self.dp_range) > 1 else 1

        if per_dp_limit is not None and local_dp_size > 1:
            return per_dp_limit
        if per_dp_limit is not None and local_dp_size <= 1 and log_ignored_per_dp:
            logger.info(
                "Ignoring DYN_REQUEST_MAX_TOTAL_REQUESTS_PER_DP=%s because this "
                "worker manages a single local DP rank",
                per_dp_limit,
            )
        return total_limit or self._get_default_max_total_requests()

    def _current_total_local_requests(self) -> int | None:
        output_processor = getattr(self.engine_client, "output_processor", None)
        if output_processor is None:
            return None
        getter = getattr(output_processor, "get_num_unfinished_requests", None)
        if getter is None:
            return None
        try:
            return int(getter())
        except Exception as e:
            logger.warning("Failed to read local unfinished request count: %s", e)
            return None

    @staticmethod
    def _is_health_check_request(request: Any) -> bool:
        return isinstance(request, dict) and bool(request.get(HEALTH_CHECK_KEY))

    async def _try_reserve_request_slot(
        self, request_id: str, request: Any | None = None
    ) -> tuple[bool, int | None, int | None]:
        if self._is_health_check_request(request):
            return True, None, None

        # Cached at init (and test overrides via self.max_total_requests).
        limit = self.max_total_requests
        if limit is None:
            return True, None, None

        async with self._request_admission_lock:
            if request_id in self._admitted_request_ids:
                logger.debug(
                    "Rejecting duplicate request_id %s still holding an admission slot",
                    request_id,
                )
                return False, len(self._admitted_request_ids), limit

            current_total = self._current_total_local_requests()
            # Fail closed when the engine count is unreadable — never admit unboundedly.
            if current_total is None:
                logger.warning(
                    "Rejecting request %s: unfinished request count unavailable "
                    "with total request limit %s configured",
                    request_id,
                    limit,
                )
                return False, None, limit

            # held + engine zombies (engine unfinished not covered by held).
            # Equals max(held, engine) when one set contains the other.
            held = len(self._admitted_request_ids)
            observed = held + max(0, current_total - held)
            if observed >= limit:
                logger.debug(
                    "Rejecting request %s due to local total request limit: %s/%s "
                    "(held=%s engine_unfinished=%s)",
                    request_id,
                    observed,
                    limit,
                    held,
                    current_total,
                )
                return False, observed, limit

            self._admitted_request_ids.add(request_id)
            self._pending_request_admissions = len(self._admitted_request_ids)
            return True, self._pending_request_admissions, limit

    async def _release_request_slot_reservation(self, request_id: str) -> None:
        async with self._request_admission_lock:
            self._admitted_request_ids.discard(request_id)
            self._pending_request_admissions = len(self._admitted_request_ids)

    async def _iterate_engine_stream(
        self,
        gen,
        request_id: str,
        *,
        enforce_decode_timeout: bool = False,
        release_request_admission: bool = False,
        abort_guard: Optional[_DeferredAbort] = None,
    ):
        timeout_secs = (
            self.max_decode_wall_clock_secs if enforce_decode_timeout else None
        )
        deadline = time.monotonic() + timeout_secs if timeout_secs is not None else None
        admission_released = not release_request_admission

        async def release_admission_once() -> None:
            nonlocal admission_released
            if not admission_released:
                admission_released = True
                await self._release_request_slot_reservation(request_id)

        try:
            while True:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        await self._raise_decode_timeout(
                            request_id, abort_guard=abort_guard
                        )
                    next_item = asyncio.wait_for(gen.__anext__(), timeout=remaining)
                else:
                    next_item = gen.__anext__()

                try:
                    item = await next_item
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    await self._raise_decode_timeout(
                        request_id, abort_guard=abort_guard
                    )
                else:
                    # Hold reservation for the full stream so Running+Waiting <= limit.
                    yield item
        finally:
            # CancelledError/GeneratorExit bypass except Exception — always release.
            await release_admission_once()

    async def _raise_decode_timeout(
        self,
        request_id: str,
        *,
        abort_guard: Optional[_DeferredAbort] = None,
    ) -> None:
        timeout_secs = self.max_decode_wall_clock_secs
        if timeout_secs is None:
            raise DecodeWallClockTimeoutError(
                "decode timeout triggered without configuration"
            )

        message = f"Decode wall clock timeout after {timeout_secs:g}s"
        logger.warning("%s for request %s", message, request_id)
        try:
            if abort_guard is not None and not getattr(
                abort_guard, "_first_token_received", False
            ):
                logger.warning(
                    "Skipping immediate abort for request %s after decode timeout "
                    "because disaggregated decode has not produced its first token",
                    request_id,
                )
            elif abort_guard is not None:
                await abort_guard.abort()
            else:
                await self.engine_client.abort(request_id)
        except Exception as abort_error:
            logger.warning(
                "Failed to abort request %s after decode timeout: %s",
                request_id,
                abort_error,
            )

        raise DecodeWallClockTimeoutError(message)

    def _tokenizer_for_reasoning_budget(self):
        tokenizer = getattr(self.engine_client, "tokenizer", None)
        return getattr(tokenizer, "tokenizer", tokenizer)

    @staticmethod
    def _is_unittest_mock(value: Any) -> bool:
        return type(value).__module__.startswith("unittest.mock")

    def _parser_visible_marker_token_ids(self) -> frozenset[int]:
        cached = getattr(self, "_dynamo_parser_visible_marker_token_ids", None)
        if isinstance(cached, frozenset):
            return cached
        cached = frozenset(self._parser_visible_marker_token_map().keys())
        try:
            setattr(self, "_dynamo_parser_visible_marker_token_ids", cached)
        except Exception:
            pass
        return cached

    def _parser_visible_marker_token_map(self) -> dict[int, str]:
        cached = getattr(self, "_dynamo_parser_visible_marker_token_map", None)
        if isinstance(cached, dict):
            return cached

        tokenizer = self._tokenizer_for_reasoning_budget()
        marker_ids: dict[int, str] = {}
        if tokenizer is not None and not self._is_unittest_mock(tokenizer):
            unk_token_id = getattr(tokenizer, "unk_token_id", None)

            def add_token_id(value: Any, marker: str) -> None:
                if isinstance(value, bool):
                    return
                try:
                    token_id = int(value)
                except (TypeError, ValueError):
                    return
                if token_id < 0:
                    return
                try:
                    if unk_token_id is not None and token_id == int(unk_token_id):
                        return
                except (TypeError, ValueError):
                    pass
                marker_ids[token_id] = marker

            convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
            encode = getattr(tokenizer, "encode", None)
            for marker in PARSER_VISIBLE_MARKER_TOKENS:
                if callable(convert_tokens_to_ids):
                    try:
                        add_token_id(convert_tokens_to_ids(marker), marker)
                    except Exception:
                        pass
                if callable(encode):
                    try:
                        encoded = encode(marker, add_special_tokens=False)
                    except TypeError:
                        try:
                            encoded = encode(marker)
                        except Exception:
                            continue
                    except Exception:
                        continue
                    if isinstance(encoded, (list, tuple)) and len(encoded) == 1:
                        add_token_id(encoded[0], marker)

        try:
            setattr(self, "_dynamo_parser_visible_marker_token_map", marker_ids)
        except Exception:
            pass
        return marker_ids

    def _decode_parser_visible_text(self, token_ids: list[int]) -> str | None:
        if not token_ids:
            return None
        tokenizer = self._tokenizer_for_reasoning_budget()
        if tokenizer is None or self._is_unittest_mock(tokenizer):
            return None
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            return None

        marker_map = self._parser_visible_marker_token_map()

        def decode_span(span: list[int]) -> str | None:
            if not span:
                return ""
            try:
                text = decode(span, skip_special_tokens=True)
            except TypeError:
                try:
                    text = decode(span)
                except Exception:
                    return None
            except Exception:
                return None
            return text if isinstance(text, str) else None

        parts: list[str] = []
        span: list[int] = []
        for token_id in token_ids:
            marker = marker_map.get(int(token_id))
            if marker is None:
                span.append(token_id)
                continue
            decoded = decode_span(span)
            if decoded is None:
                return None
            parts.append(decoded)
            parts.append(marker)
            span = []

        decoded = decode_span(span)
        if decoded is None:
            return None
        parts.append(decoded)
        return "".join(parts)

    def _compute_newline_token_ids(
        self,
        tokenizer,
        strings: list[str] | None = None,
    ) -> list[int]:
        cached = getattr(tokenizer, "_dynamo_newline_token_ids", None)
        if isinstance(cached, list):
            return cached

        if strings is None:
            strings = ["\n", "\r\n", "\n\n"]

        newline_ids: set[int] = set()
        for s in strings:
            try:
                encoded = tokenizer.encode(s, add_special_tokens=False)
            except TypeError:
                try:
                    encoded = tokenizer.encode(s)
                except Exception:
                    continue
            except Exception:
                continue
            for token_id in encoded:
                try:
                    newline_ids.add(int(token_id))
                except Exception:
                    continue

        vocab_size = getattr(tokenizer, "vocab_size", None)
        if isinstance(vocab_size, int):
            for token_id in range(min(vocab_size, 300000)):
                try:
                    token_text = tokenizer.decode([token_id])
                except Exception:
                    continue
                if any(token_text.endswith(s) for s in strings):
                    newline_ids.add(token_id)

        ids_list = sorted(newline_ids)
        try:
            setattr(tokenizer, "_dynamo_newline_token_ids", ids_list)
        except Exception:
            pass
        return ids_list

    def _apply_reasoning_budget_extra_args(
        self, sampling_params: SamplingParams
    ) -> None:
        extra = sampling_params.extra_args
        if not isinstance(extra, dict):
            return
        if "reasoning_budget" not in extra:
            return

        try:
            budget_int = int(extra["reasoning_budget"])
        except Exception:
            logger.warning(
                "Invalid reasoning_budget=%r; skipping", extra.get("reasoning_budget")
            )
            return
        if budget_int == -1:
            return
        extra["reasoning_budget"] = budget_int

        try:
            extra["reasoning_budget_grace_period"] = int(
                extra.get("reasoning_budget_grace_period", 0) or 0
            )
        except Exception:
            extra["reasoning_budget_grace_period"] = 0

        end_token_ids = extra.get("end_token_ids")
        parsed_end_ids: list[int] = []
        if isinstance(end_token_ids, list):
            try:
                parsed_end_ids = [int(tid) for tid in end_token_ids]
            except Exception:
                parsed_end_ids = []

        if not parsed_end_ids and extra.get("think_end_token_id") is not None:
            try:
                parsed_end_ids = [int(extra["think_end_token_id"])]
            except Exception:
                parsed_end_ids = []

        if not parsed_end_ids:
            reasoning_config = getattr(
                self.engine_client.vllm_config, "reasoning_config", None
            )
            config_end_ids = getattr(reasoning_config, "reasoning_end_token_ids", None)
            if isinstance(config_end_ids, list):
                try:
                    parsed_end_ids = [int(tid) for tid in config_end_ids]
                except Exception:
                    parsed_end_ids = []

        tokenizer = None
        if not parsed_end_ids:
            tokenizer = self._tokenizer_for_reasoning_budget()
            end_token = getattr(
                getattr(self.engine_client.vllm_config, "reasoning_config", None),
                "reasoning_end_str",
                None,
            )
            parser_name = getattr(
                getattr(self.config, "engine_args", None), "reasoning_parser", None
            )
            if parser_name and tokenizer is not None:
                try:
                    from vllm.reasoning import ReasoningParserManager

                    parser_class = ReasoningParserManager.get_reasoning_parser(
                        parser_name
                    )
                    reasoning_parser = parser_class(tokenizer)
                    end_token_ids = getattr(reasoning_parser, "end_token_ids", None)
                    if isinstance(end_token_ids, list):
                        parsed_end_ids = [int(tid) for tid in end_token_ids]
                    if (
                        not parsed_end_ids
                        and getattr(reasoning_parser, "end_token_id", None) is not None
                    ):
                        parsed_end_ids = [int(reasoning_parser.end_token_id)]
                    if not parsed_end_ids:
                        end_token = (
                            getattr(reasoning_parser, "end_token", None)
                            or getattr(reasoning_parser, "reasoning_end_str", None)
                            or end_token
                        )
                except Exception:
                    parsed_end_ids = []

            if not parsed_end_ids and isinstance(end_token, str) and end_token:
                try:
                    parsed_end_ids = [
                        int(tid)
                        for tid in tokenizer.encode(
                            end_token, add_special_tokens=False
                        )
                    ]
                except TypeError:
                    try:
                        parsed_end_ids = [int(tid) for tid in tokenizer.encode(end_token)]
                    except Exception:
                        parsed_end_ids = []
                except Exception:
                    parsed_end_ids = []
                if not parsed_end_ids:
                    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
                    token_id = vocab.get(end_token)
                    if token_id is not None:
                        try:
                            parsed_end_ids = [int(token_id)]
                        except Exception:
                            parsed_end_ids = []

        if parsed_end_ids:
            extra.setdefault("think_end_token_id", parsed_end_ids[0])
            extra.setdefault("end_token_ids", parsed_end_ids)

        if "newline_token_ids" not in extra:
            if tokenizer is None:
                tokenizer = self._tokenizer_for_reasoning_budget()
            if tokenizer is not None:
                try:
                    newline_ids = self._compute_newline_token_ids(tokenizer)
                except Exception:
                    newline_ids = []
                if newline_ids:
                    extra["newline_token_ids"] = newline_ids

    def init_embedding_loader(
        self, config: Config, encode_worker_client: Optional[Client] = None
    ) -> Optional[MultiModalEmbeddingLoader]:
        """Initialize the embedding loader with the given encode worker client."""
        # Without encode worker, the embedding will be generated internally by vLLM.
        if encode_worker_client is None:
            return None
        logger.warning(
            "Separate multimodal encode-worker routing only applies to image_url "
            "inputs. video_url inputs are not sent to the encode worker and will "
            "be processed on the prefill/PD worker instead."
        )
        # Embedding loader consist of two main components:
        # 1) An remote encode worker client and matching embedding receiver,
        #    which can request remote encode and handle the transfer of embeddings
        #    from the encode worker to this prefill worker.
        # 2) A local embedding cache manager, which can store previously fetched embeddings
        #    and used to determine whether remote encode is necessary for a given mm data.
        self.encode_worker_client = encode_worker_client
        if config.embedding_transfer_mode == EmbeddingTransferMode.LOCAL:
            self.embedding_receiver = LocalEmbeddingReceiver()  # type: ignore
        elif config.embedding_transfer_mode == EmbeddingTransferMode.NIXL_WRITE:
            self.embedding_receiver = NixlWriteEmbeddingReceiver()  # type: ignore
        elif config.embedding_transfer_mode == EmbeddingTransferMode.NIXL_READ:
            # [gluo FIXME] can't use pre-registered tensor as NIXL requires descriptors
            # to be at matching size, need to overwrite nixl connect library
            self.embedding_receiver = NixlReadEmbeddingReceiver(max_items=0)  # type: ignore
        else:
            raise ValueError(
                f"Invalid embedding transfer mode: {config.embedding_transfer_mode}"
            )
        # [gluo FIXME/NOTE] This embedding cache manager is purely used for caching embedding
        # results from encode worker, but 'config.multimodal_embedding_cache_capacity_gb' is
        # also used to configure the DynamoMultimodalEmbeddingCacheConnector within the vLLM.
        # This results in duplication of memory and ideally we should have single cache manager
        # which can be used by vLLM internal and here. Then we can explore asynchrous embedding
        # transfer as we can process and block until the embedding is actually used within vLLM.
        self.embedding_cache_manager: MultimodalEmbeddingCacheManager | None = None
        if config.multimodal_embedding_cache_capacity_gb > 0:
            capacity_bytes = int(
                config.multimodal_embedding_cache_capacity_gb * 1024**3
            )
            self.embedding_cache_manager = MultimodalEmbeddingCacheManager(
                capacity_bytes
            )
        return MultiModalEmbeddingLoader(
            encode_worker_client=self.encode_worker_client,  # type: ignore
            receiver=self.embedding_receiver,
            embedding_cache_manager=self.embedding_cache_manager,
        )

    async def sleep(self, body: dict) -> dict:
        """Sleep the engine to release GPU memory and unregister from discovery.

        Args:
            body: Dict with optional 'level' key (1=weights only, 2=weights+buffers, 3=everything)

        Order of operations:
        1. Unregister from discovery - stop accepting new requests
        2. Abort and drain in-flight requests
        3. Sleep engine - safe now that GPU is quiesced
        """
        body = body or {}
        level = body.get("level", 1)
        async with self._quiesce_lock:
            if self._quiesce_controller.is_quiesced:
                return {
                    "status": "ok",
                    "message": "Engine already sleeping",
                }

            try:
                # Step 1: Unregister endpoint instance before memory transitions.
                if self.generate_endpoint is not None:
                    await self.generate_endpoint.unregister_endpoint_instance()
                    logger.info(
                        "[Sleep] Unregistered endpoint from discovery - worker removed from routing pool"
                    )

                # Step 2: Abort in-flight requests and wait for them to drain so the
                # GPU is fully quiesced before unmapping memory.
                if not await self._quiesce_controller.quiesce(level):
                    return {
                        "status": "ok",
                        "message": "Engine already sleeping",
                    }

                return {
                    "status": "ok",
                    "message": f"Engine slept (level={level})",
                }
            except Exception as e:
                logger.error(f"Failed to sleep engine: {e}")
                return {"status": "error", "message": str(e)}

    async def scale_elastic_ep(self, body: dict) -> dict:
        """Scale the elastic expert-parallelism data-parallel size live.

        Args:
            body: Dict with required 'new_data_parallel_size' key (int).
                Example::

                    {"new_data_parallel_size": 4}

        The vLLM Ray DP backend will spin up / tear down DP workers on the GPUs
        already reserved by the pod, then hot-swap the expert routing table.
        No pod restart is needed.
        """
        body = body or {}
        new_dp_size = body.get("new_data_parallel_size")
        if new_dp_size is None:
            return {
                "status": "error",
                "message": "Missing required field: new_data_parallel_size",
            }
        try:
            new_dp_size = int(new_dp_size)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "message": f"new_data_parallel_size must be an integer, got: {new_dp_size!r}",
            }
        if new_dp_size < 2:
            return {
                "status": "error",
                "message": (
                    "new_data_parallel_size must be >= 2 when elastic EP/ePLB is enabled"
                ),
            }

        logger.info(f"[ElasticEP] Scaling to new_data_parallel_size={new_dp_size}")

        # Early-reject if another scale is already in progress rather than
        # queuing behind it: a queued caller would garbage-collect the first
        # caller's TCPStore before its Ray actor connects, causing a 300 s
        # timeout on the worker node.
        # The locked() check followed immediately by async with is safe:
        # asyncio.Lock.acquire() only suspends (yields to the event loop) when
        # the lock is already held. When locked() returns False the lock is
        # free, so acquire() completes synchronously — no other coroutine can
        # run between the check and the acquisition.
        if self._scale_ep_lock.locked():
            msg = (
                f"A scale_elastic_ep operation is already in progress; "
                f"rejecting concurrent request for new_data_parallel_size={new_dp_size}"
            )
            logger.warning(f"[ElasticEP] {msg}")
            return {"status": "error", "message": msg}

        async with self._scale_ep_lock:
            try:
                # TODO(upstream-vllm): remove this patch once vLLM fixes
                # add_dp_placement_groups in vllm/v1/engine/utils.py to use ray.nodes()
                # instead of ray.util.state.list_nodes().
                #
                # Patch ray.util.state.list_nodes to use the GCS API instead of the
                # dashboard HTTP API (127.0.0.1:8265/api/v0/nodes). The dynamo image
                # installs ray core only (not ray[default]), so the dashboard HTTP server
                # starts in --minimal mode with the HTTP server disabled. vLLM's
                # add_dp_placement_groups calls list_nodes() which requires that HTTP
                # endpoint, causing scale_elastic_ep to fail with "Failed to connect to
                # API server".
                #
                # ray.nodes() uses the GCS gRPC channel directly (no dashboard process
                # needed) and returns the same information. Imported lazily so ray is not
                # required at module load time (absent in non-elastic-EP deployments).
                #
                # Format mapping:
                #   list_nodes() → objects with .node_ip and .node_id
                #   ray.nodes()  → dicts with "NodeManagerAddress" and "NodeID"
                import ray
                import ray.util.state as _ray_util_state

                class _NodeInfo:
                    __slots__ = ("node_id", "node_ip")

                    def __init__(self, d: dict) -> None:
                        self.node_ip: str = d["NodeManagerAddress"]
                        self.node_id: str = d["NodeID"]

                _ray_util_state.list_nodes = lambda **kw: [
                    _NodeInfo(n) for n in ray.nodes() if n.get("Alive", False)
                ]

                await self.engine_client.scale_elastic_ep(new_dp_size)
                logger.info(f"[ElasticEP] Scaling to dp={new_dp_size} complete")
                return {
                    "status": "ok",
                    "message": f"Scaled to data_parallel_size={new_dp_size}",
                    "new_data_parallel_size": new_dp_size,
                }
            except Exception as e:
                logger.error(f"[ElasticEP] Scaling failed: {e}")
                return {"status": "error", "message": str(e)}

    async def wake_up(self, body: dict) -> dict:
        """Wake the engine to restore GPU memory and re-register to discovery.

        Args:
            body: Optional dict with "tags" to request a partial wake.

        Order of operations:
        1. Wake engine - restore GPU memory
        2. Re-register endpoint instance - allow frontend to route requests here again
        """
        body = body or {}
        tags = body.get("tags")
        async with self._quiesce_lock:
            if not self._quiesce_controller.is_quiesced:
                return {"status": "ok", "message": "Engine already awake"}

            try:
                # Step 1: Wake engine first - must be ready before accepting requests
                await self._quiesce_controller.resume(tags)
                if self.generate_endpoint is not None:
                    await self.generate_endpoint.register_endpoint_instance()
                    logger.info(
                        "[Wake] Re-registered endpoint to discovery - worker added back to routing pool"
                    )
                self._quiesce_controller.mark_resumed()

                return {
                    "status": "ok",
                    "message": "Engine woke",
                }
            except Exception as e:
                logger.error(f"Failed to wake up engine: {e}")
                return {"status": "error", "message": str(e)}

    async def start_profile(self, body: dict) -> dict:
        """Start profiling on the engine.

        Args:
            body: Dict with profiling parameters. Supported keys:
                - profile_prefix (str|None): Optional prefix for profile output files.
        """
        profile_prefix = body.get("profile_prefix")
        try:
            await self.engine_client.start_profile(profile_prefix=profile_prefix)
            return {"status": "ok", "message": "Profiling started"}
        except Exception as e:
            logger.error(f"Failed to start profiling: {e}")
            return {"status": "error", "message": str(e)}

    async def stop_profile(self, body: dict) -> dict:
        """Stop profiling on the engine.

        Args:
            body: Unused, but required for handler signature.
        """
        try:
            await self.engine_client.stop_profile()
            return {"status": "ok", "message": "Profiling stopped"}
        except Exception as e:
            logger.error(f"Failed to stop profiling: {e}")
            return {"status": "error", "message": str(e)}

    @abstractmethod
    def generate(self, request: RequestT, context: Context) -> AsyncIterator[ResponseT]:
        raise NotImplementedError

    async def _monitor_abort(self, context, request_id, is_prefill, abort_guard=None):
        """
        Background task that monitors for context cancellation and shutdown.
        Aborts the request if either occurs. Raises EngineShutdown if shutdown was triggered.

        If abort_guard is provided, the abort call is routed through it so that
        it can be deferred until the first engine output (used in disagg decode
        mode to avoid aborting during an active NIXL KV transfer).
        """
        try:
            # Build list of futures/tasks to wait for
            wait_for = [context.async_killed_or_stopped()]
            shutdown_task = None

            if self.shutdown_event:
                # Create task for shutdown monitoring and add to wait list
                shutdown_task = asyncio.create_task(self.shutdown_event.wait())
                wait_for.append(shutdown_task)

            # Wait for whichever happens first
            done, pending = await asyncio.wait(
                wait_for,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the pending task/future
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Log intent before the abort — synchronous, no await, so it is
            # guaranteed to appear even if this task is concurrently cancelled
            # by _abort_monitor's cleanup.
            logger.debug(
                f"Aborting {'Prefill ' if is_prefill else ''}Request ID: {request_id}"
            )

            # Abort the request (via guard if provided, otherwise direct).
            if abort_guard is not None:
                # Guard owns completion logging: "Aborted Request ID:" is
                # emitted from _run_abort() once engine.abort() returns,
                # even if this task is concurrently cancelled.
                await abort_guard.abort()
            else:
                # No guard: shield the abort so a concurrent CancelledError
                # from _abort_monitor cleanup cannot interrupt it mid-flight.
                try:
                    await asyncio.shield(self.engine_client.abort(request_id))
                except asyncio.CancelledError:
                    logger.debug(
                        f"Abort shielded from cancellation for request "
                        f"{request_id}, continuing in background"
                    )
                logger.debug(
                    f"Aborted {'Prefill ' if is_prefill else ''}Request ID: {request_id}"
                )

            # Check which event triggered and raise EngineShutdown if shutdown
            if shutdown_task and shutdown_task in done:
                raise EngineShutdown("Engine was shut down during generation.")

        except asyncio.CancelledError:
            # Task was cancelled, normal cleanup if not aborted
            pass
        except EngineShutdown:
            raise
        except Exception as e:
            logger.error(f"Error in abort monitor for request {request_id}: {e}")

    @asynccontextmanager
    async def _abort_monitor(
        self, context, request_id, is_prefill=False, abort_guard=None
    ):
        """
        Context manager that creates and automatically cleans up an abort monitoring task.
        If shutdown event was triggered, raises EngineShutdown on exit.

        If abort_guard is provided, the abort call is routed through it so the
        abort can be deferred until the first engine output.
        """
        task = asyncio.create_task(
            self._monitor_abort(context, request_id, is_prefill, abort_guard)
        )
        try:
            yield task
        finally:
            # Clean up the abort monitoring task
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:
                # If the task completed, check if it raised EngineShutdown
                task.result()

    async def clear_kv_blocks(self, request=None):
        try:
            await self.engine_client.reset_prefix_cache()
            yield {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    async def get_perf_metrics(self, request=None):
        """Return self-benchmark FPM results, or an error dict if none."""
        result = getattr(self, "_benchmark_results", None)
        if result is None:
            yield {"status": "error", "message": "no benchmark data"}
        else:
            yield result

    def add_temp_dir(self, temp_dir: tempfile.TemporaryDirectory) -> None:
        """Add a temporary directory to be cleaned up later."""
        if temp_dir is not None:
            self.temp_dirs.append(temp_dir)

    def _to_local_dp_rank(self, dp_rank: int | None) -> int | None:
        """Convert global DP rank to local DP rank based on engine config."""
        if dp_rank is None:
            return None
        if dp_rank < self.dp_range[0] or dp_rank >= self.dp_range[0] + self.dp_range[1]:
            logger.warning(
                f"Received DP rank {dp_rank} is out of range [{self.dp_range[0]} - {self.dp_range[0] + self.dp_range[1]}), fallback to vLLM internal DP selection"
            )
            return None
        local_dp_rank = (dp_rank - self.dp_range[0]) % self.dp_range[1]
        logger.debug(
            f"Converted global DP rank {dp_rank} to local DP rank {local_dp_rank}"
        )
        return local_dp_rank

    def _resolve_lora_request(self, model_name: str | None) -> LoRARequest | None:
        """Return a LoRARequest if model_name is a loaded adapter, else None."""
        if model_name and (lora := self.loaded_loras.get(model_name)):
            return LoRARequest(
                lora_name=model_name,
                lora_int_id=lora.id,
                lora_path=lora.path,
            )
        return None

    def _get_lora_lock(self, lora_name: str) -> asyncio.Lock:
        """Get/create the per-LoRA lock without eagerly allocating a new lock each call."""
        with self._lora_load_locks_guard:
            lock = self._lora_load_locks.get(lora_name)
            if lock is None:
                lock = asyncio.Lock()
                self._lora_load_locks[lora_name] = lock
            return lock

    async def load_lora(self, request=None):
        """
        Load a LoRA adapter dynamically into the vLLM's AsyncLLM engine.

        Request format:
        {
            "lora_name": str,
            "source": {
                "uri": str  # e.g., "s3://bucket/path" or "file:///path"
            }
        }

        This method is idempotent - concurrent calls for the same LoRA will be
        serialized and only one load operation will happen.
        """
        try:
            if request is None:
                yield {
                    "status": "error",
                    "message": "Request is required with 'lora_name' and 'source.uri'",
                }
                return

            lora_name = request.get("lora_name")
            if not lora_name:
                yield {
                    "status": "error",
                    "message": "'lora_name' is required in request",
                }
                return

            # Debug: Log the incoming request
            logger.debug(f"load_lora request keys: {list(request.keys())}")
            logger.debug(f"load_lora request: {request}")

            # Check for URI-based API format (source.uri)
            source = request.get("source")
            if not source or not isinstance(source, dict):
                yield {
                    "status": "error",
                    "message": "'source' object is required in request",
                }
                return

            lora_uri = source.get("uri")
            if not lora_uri:
                yield {
                    "status": "error",
                    "message": "'source.uri' is required in request",
                }
                return

            # Use LoRAManager to download from URI
            lora_manager = get_lora_manager()
            if lora_manager is None:
                yield {
                    "status": "error",
                    "message": "LoRAManager not initialized. Set DYN_LORA_ENABLED=true to enable URI-based LoRA loading.",
                }
                return

            # Serialize load/unload operations per lora_name.
            lock = self._get_lora_lock(lora_name)
            async with lock:
                try:
                    # Check if already loaded (idempotency check after acquiring lock).
                    # Another concurrent request may have loaded this LoRA while we waited.
                    if lora_name in self.loaded_loras:
                        lora_id = self.loaded_loras[lora_name].id
                        logger.info(
                            f"LoRA adapter already loaded (concurrent request completed): "
                            f"{lora_name} with ID {lora_id}"
                        )
                        yield {
                            "status": "success",
                            "message": f"LoRA adapter '{lora_name}' already loaded",
                            "lora_name": lora_name,
                            "lora_id": lora_id,
                        }
                        return

                    logger.info(
                        f"Downloading LoRA adapter: {lora_name} from {lora_uri}"
                    )
                    download_result = await lora_manager.download_lora(lora_uri)

                    if download_result["status"] != "success":
                        yield {
                            "status": "error",
                            "message": f"Failed to download LoRA: {download_result.get('message', 'Unknown error')}",
                        }
                        return

                    lora_path = download_result["local_path"]
                    logger.debug(f"LoRA downloaded to: {lora_path}")

                    # Generate deterministic ID from lora_name before using it
                    lora_id = lora_name_to_id(lora_name)

                    # Add the LoRA to the engine
                    await self.engine_client.add_lora(
                        LoRARequest(
                            lora_name=lora_name,
                            lora_int_id=lora_id,
                            lora_path=lora_path,
                        )
                    )

                    # Track the LoRA
                    self.loaded_loras[lora_name] = LoRAInfo(id=lora_id, path=lora_path)
                    logger.info(
                        f"Successfully loaded LoRA adapter: {lora_name} with ID {lora_id}"
                    )

                    # Publish LoRA as a ModelDeploymentCard with format:
                    # v1/mdc/{namespace}/{component}/{endpoint}/{instance_id}/{lora_slug}
                    # This allows the frontend to discover it and route correctly to the worker instance
                    if self.generate_endpoint is not None:
                        logger.debug(
                            f"Publishing LoRA '{lora_name}' ModelDeploymentCard to {self.generate_endpoint}"
                        )
                        try:
                            logger.debug(
                                f"Publishing LoRA '{lora_name}' ModelDeploymentCard"
                            )

                            # Mark this as a LoRA in user_data
                            user_data = {
                                "lora_adapter": True,
                                "lora_id": lora_id,
                            }

                            runtime_config = ModelRuntimeConfig()
                            runtime_config.tool_call_parser = (
                                self.config.dyn_tool_call_parser
                            )
                            runtime_config.reasoning_parser = (
                                self.config.dyn_reasoning_parser
                            )

                            # Publish with format: v1/mdc/dynamo/backend/generate/{instance_id}/{lora_slug}
                            await register_model(
                                model_input=ModelInput.Tokens,
                                model_type=ModelType.Chat | ModelType.Completions,
                                endpoint=self.generate_endpoint,
                                model_path=self.config.model,
                                kv_cache_block_size=self.config.engine_args.block_size,
                                runtime_config=runtime_config,
                                user_data=user_data,
                                lora_name=lora_name,
                                base_model_path=self.config.model,
                            )
                            logger.info(
                                f"Successfully published LoRA '{lora_name}' ModelDeploymentCard"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Failed to publish LoRA {lora_name} ModelDeploymentCard: {e}"
                            )

                            # Rollback: remove the LoRA from the engine to maintain consistency
                            try:
                                logger.debug(
                                    f"Rolling back: removing LoRA '{lora_name}' from engine"
                                )
                                await self.engine_client.remove_lora(lora_id)
                                self.loaded_loras.pop(lora_name, None)
                                logger.debug(
                                    f"Successfully rolled back LoRA '{lora_name}'"
                                )
                            except Exception as rollback_error:
                                logger.exception(
                                    f"Failed to rollback LoRA {lora_name}: {rollback_error}"
                                )

                            # Return error status since registration failed
                            yield {
                                "status": "error",
                                "message": f"Failed to register LoRA '{lora_name}' in discovery registry: {str(e)}",
                                "lora_name": lora_name,
                            }
                            return
                    else:
                        logger.debug(
                            f"Cannot publish LoRA '{lora_name}': generate_endpoint={self.generate_endpoint}, config={self.config}"
                        )

                    yield {
                        "status": "success",
                        "message": f"LoRA adapter '{lora_name}' loaded successfully",
                        "lora_name": lora_name,
                        "lora_id": lora_id,
                    }
                finally:
                    # Avoid lock-map growth on failed loads: if this attempt did not leave the LoRA
                    # loaded, remove the lock entry (best-effort).
                    with self._lora_load_locks_guard:
                        if (
                            lora_name not in self.loaded_loras
                            and self._lora_load_locks.get(lora_name) is lock
                        ):
                            self._lora_load_locks.pop(lora_name, None)
        except Exception as e:
            logger.exception(f"Failed to load LoRA adapter: {e}")
            yield {"status": "error", "message": str(e)}

    async def unload_lora(self, request=None):
        """
        Unload a LoRA adapter dynamically from the vLLM's AsyncLLM engine.
        Expected request format:
        {
            "lora_name": str,
        }
        """
        try:
            if request is None:
                yield {
                    "status": "error",
                    "message": "Request is required with 'lora_name' field",
                }
                return
            lora_name = request.get("lora_name")
            if not lora_name:
                yield {
                    "status": "error",
                    "message": "'lora_name' is required in request",
                }
                return

            # Serialize load/unload operations per lora_name.
            lock = self._get_lora_lock(lora_name)
            async with lock:
                try:
                    # Check if the LoRA exists *after* waiting for any in-progress load.
                    lora = self.loaded_loras.get(lora_name)
                    if lora is None:
                        yield {
                            "status": "error",
                            "message": f"LoRA adapter '{lora_name}' not found. Available LoRAs: {list(self.loaded_loras.keys())}",
                        }
                        return

                    logger.debug(f"Unloading LoRA adapter: {lora_name}")
                    lora_id = lora.id
                    lora_path = lora.path

                    await self.engine_client.remove_lora(lora_id)

                    # Remove from tracking
                    del self.loaded_loras[lora_name]

                    # Unregister the LoRA model from the model registry
                    if self.generate_endpoint is not None:
                        logger.debug(
                            f"Unregistering LoRA '{lora_name}' ModelDeploymentCard"
                        )
                        try:
                            await unregister_model(
                                endpoint=self.generate_endpoint,
                                lora_name=lora_name,
                            )
                            logger.info(
                                f"Successfully unregistered LoRA '{lora_name}' ModelDeploymentCard"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Failed to unregister LoRA {lora_name} ModelDeploymentCard: {e}"
                            )

                            # Rollback: re-add the LoRA to the engine to maintain consistency
                            try:
                                logger.debug(
                                    f"Rolling back: re-adding LoRA '{lora_name}' to engine"
                                )
                                await self.engine_client.add_lora(
                                    LoRARequest(
                                        lora_name=lora_name,
                                        lora_int_id=lora_id,
                                        lora_path=lora_path,
                                    )
                                )
                                # Re-add to tracking
                                self.loaded_loras[lora_name] = LoRAInfo(
                                    id=lora_id, path=lora_path
                                )
                                logger.debug(
                                    f"Successfully rolled back LoRA '{lora_name}'"
                                )
                            except Exception as rollback_error:
                                logger.exception(
                                    f"Failed to rollback LoRA {lora_name}: {rollback_error}"
                                )

                            # Return error status since unregistration failed
                            yield {
                                "status": "error",
                                "message": f"Failed to unregister LoRA '{lora_name}' from discovery registry: {str(e)}",
                                "lora_name": lora_name,
                            }
                            return
                    else:
                        logger.debug(
                            f"Cannot unregister LoRA '{lora_name}': generate_endpoint={self.generate_endpoint}"
                        )

                    logger.info(
                        f"Successfully unloaded LoRA adapter: {lora_name} with ID {lora_id}"
                    )
                    yield {
                        "status": "success",
                        "message": f"LoRA adapter '{lora_name}' unloaded successfully",
                        "lora_name": lora_name,
                        "lora_id": lora_id,
                    }
                finally:
                    # Remove lock entry once the LoRA is not loaded (or never was).
                    with self._lora_load_locks_guard:
                        if (
                            lora_name not in self.loaded_loras
                            and self._lora_load_locks.get(lora_name) is lock
                        ):
                            self._lora_load_locks.pop(lora_name, None)
        except Exception as e:
            logger.exception(f"Failed to unload LoRA adapter: {e}")
            yield {"status": "error", "message": str(e)}

    async def list_loras(self, request=None):
        """
        List all loaded LoRA adapters.
        Returns a dictionary of lora_name -> lora_id mappings.
        """
        try:
            loras = {name: lora.id for name, lora in self.loaded_loras.items()}
            yield {
                "status": "success",
                "loras": loras,
                "count": len(loras),
            }
        except Exception as e:
            logger.error(f"Failed to list LoRA adapters: {e}")
            yield {"status": "error", "message": str(e)}

    def cleanup(self):
        """Clean up resources including temporary directories."""
        for temp_dir in self.temp_dirs:
            try:
                temp_dir.cleanup()
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")

    def _decode_prompt_embeds(self, prompt_embeds_base64: str):
        """
        Decode base64-encoded prompt embeddings in PyTorch format.

        Use vllm's safe loader to prevent out-of-bounds writes from maliciously crafted tensors.

        Format: PyTorch tensor serialized with torch.save() and base64-encoded.

        Args:
            prompt_embeds_base64: Base64-encoded PyTorch tensor

        Returns:
            torch.Tensor: Decoded prompt embeddings with dim == 2

        Raises:
            ValueError: If decoding fails or format is invalid
        """
        if not isinstance(prompt_embeds_base64, str):
            raise ValueError(
                f"Prompt embeds must be base64 encoded string. Got {type(prompt_embeds_base64)}."
            )

        if self.model_config is None:
            raise ValueError("ModelConfig is unavailable for prompt_embeds validation.")

        try:
            return safe_load_prompt_embeds(
                self.model_config, prompt_embeds_base64.encode()
            )
        except Exception as e:
            logger.error(f"Failed to decode prompt_embeds: {e}")
            raise ValueError(f"Failed to decode prompt_embeds as PyTorch tensor: {e}")

    def _create_prompt_from_embeddings(
        self, prompt_embeds_base64: str
    ) -> tuple[EmbedsPrompt, int, torch.Tensor]:
        """
        Decode prompt embeddings and create EmbedsPrompt for vLLM.

        Args:
            prompt_embeds_base64: Base64-encoded PyTorch tensor

        Returns:
            Tuple of (EmbedsPrompt, sequence_length, tensor) where:
            - EmbedsPrompt: The vLLM prompt input
            - sequence_length: Extracted from tensor shape for usage statistics
            - tensor: The decoded tensor (for logging shape/dtype)

        Raises:
            ValueError: If decoding fails or tensor is invalid
        """
        embeddings_tensor = self._decode_prompt_embeds(prompt_embeds_base64)
        if embeddings_tensor.dim() != 2:
            raise ValueError(
                f"prompt embeds should have dim 2 after vllm processing, but found dim {embeddings_tensor.dim()}"
            )

        # Extract sequence length from tensor shape for usage reporting
        sequence_length = embeddings_tensor.shape[0]

        # EmbedsInputs TypedDict has: {type: 'embeds', prompt_embeds: Tensor, cache_salt?: str}
        prompt = EmbedsPrompt(prompt_embeds=embeddings_tensor)

        return prompt, sequence_length, embeddings_tensor

    async def _try_receive_mm_kwargs(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """Try to receive pre-processed mm_kwargs from the frontend (SHM or NIXL).

        If ``extra_args`` contains ``mm_kwargs_shm`` or ``mm_kwargs_nixl``, fetch
        the tensors via the corresponding transport and construct a pre-rendered
        MultiModalInput dict the vLLM engine can consume directly, skipping the
        HF processor. Returns None if no transfer metadata is present or the
        receive fails (caller falls back to normal processing).
        """
        extra_args = request.get("extra_args") or {}
        logger.debug(
            "[mm-routing] _try_receive_mm_kwargs: extra_args keys=%s",
            list(extra_args.keys()),
        )

        # SHM path first (same-node, ~1.5ms). Only works when frontend and
        # backend share /dev/shm; if the read fails (cross-node), fall through
        # to normal processing.
        shm_meta_raw = extra_args.get("mm_kwargs_shm")
        if shm_meta_raw:
            shm_meta = MmKwargsShmTransferMetadata.model_validate(shm_meta_raw)
            return await self._receive_mm_kwargs(
                extra_args, "shm", MmKwargsShmReceiver(), shm_meta
            )

        nixl_meta_raw = extra_args.get("mm_kwargs_nixl")
        if nixl_meta_raw:
            nixl_meta = MmKwargsTransferMetadata.model_validate(nixl_meta_raw)
            if self._mm_kwargs_receiver is None:
                self._mm_kwargs_receiver = MmKwargsNixlReceiver()
            return await self._receive_mm_kwargs(
                extra_args, "nixl", self._mm_kwargs_receiver, nixl_meta
            )

        logger.debug("[mm-routing] No mm_kwargs transfer metadata in extra_args")
        return None

    async def _receive_mm_kwargs(
        self,
        extra_args: Dict[str, Any],
        transport: str,
        receiver: MmKwargsReceiver,
        metadata: Any,
    ) -> Dict[str, Any] | None:
        """Shared NIXL/SHM receive path.

        Calls ``receiver.receive(metadata)`` to fetch pickled
        ``MultiModalKwargsItem`` bytes, deserializes them, builds an
        ``EngineInput`` dict, and injects into the engine's MM processor
        cache. Returns None on any validation or transport failure
        (caller falls back).
        """
        color = "magenta" if transport == "nixl" else "cyan"
        rng = _nvtx.start_range(f"mm_backend:{transport}_receive", color=color)
        try:
            mm_hashes = extra_args.get("mm_hashes")
            mm_placeholders = extra_args.get("mm_placeholders")
            if not mm_hashes or not mm_placeholders:
                logger.warning(
                    "[mm-routing] %s present but mm_hashes/mm_placeholders missing",
                    transport,
                )
                return None

            # Receive pickled kwargs items (NVTX wrap is owned by the receiver).
            results = await receiver.receive(metadata)

            pickled_items = results.get("__pickled_kwargs_item__")
            if not pickled_items:
                logger.warning(
                    "[mm-routing] %s: no pickled kwargs items received", transport
                )
                return None

            # Unpickle and validate each item.
            kwargs_items: list[MultiModalKwargsItem] = []
            with _nvtx.annotate(f"mm_backend:{transport}_pickle_loads", color=color):
                for pi in pickled_items:
                    item = pickle.loads(pi)
                    if not isinstance(item, MultiModalKwargsItem):
                        logger.warning(
                            "[mm-routing] %s: deserialized object is %s, expected "
                            "MultiModalKwargsItem; falling back to normal path",
                            transport,
                            type(item).__name__,
                        )
                        return None
                    kwargs_items.append(item)

            # Use the expanded token IDs (with image placeholders) from the
            # frontend, not the unexpanded request["token_ids"]. The
            # mm_placeholders and transferred kwargs are aligned to the
            # expanded sequence — using unexpanded tokens would misplace
            # every placeholder.
            expanded_token_ids = extra_args.get("expanded_token_ids")
            if not expanded_token_ids:
                logger.warning(
                    "[mm-routing] %s: no expanded_token_ids in extra_args, "
                    "cannot use pre-rendered mm_kwargs; falling back",
                    transport,
                )
                return None

            mm_hashes_dict = {metadata.modality: mm_hashes}
            mm_kwargs_dict = {metadata.modality: kwargs_items}
            with _nvtx.annotate(
                f"mm_backend:{transport}_build_engine_input", color=color
            ):
                engine_input = {
                    "type": "multimodal",
                    "prompt_token_ids": expanded_token_ids,
                    "mm_kwargs": mm_kwargs_dict,
                    "mm_hashes": mm_hashes_dict,
                    "mm_placeholders": {
                        metadata.modality: [
                            PlaceholderRange(offset=off, length=length)
                            for off, length in mm_placeholders
                        ],
                    },
                }

            # Inject into the engine's MM processor cache so subsequent
            # requests with the same images get cache hits.
            try:
                self.engine_client.input_processor.inject_into_mm_cache(
                    mm_hashes_dict, mm_kwargs_dict
                )
            except Exception:
                logger.debug(
                    "[mm-routing] Failed to inject into mm_cache", exc_info=True
                )

            logger.debug(
                "[mm-routing] %s: constructed pre-rendered MultiModalInput from "
                "%d kwargs_items, %d hashes, %d placeholders",
                transport,
                len(kwargs_items),
                len(mm_hashes),
                len(mm_placeholders),
            )
            return engine_input
        except Exception:
            logger.exception("[mm-routing] %s receive failed, falling back", transport)
            return None
        finally:
            _nvtx.end_range(rng)

    @staticmethod
    def _get_mm_processor_kwargs(
        request: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """Extract mm_processor_kwargs from a request dict.

        Checks the top-level key (client router / Rust preprocessor path)
        and falls back to ``extra_args`` (KV router path).
        """
        mm_processor_kwargs = request.get("mm_processor_kwargs")
        if mm_processor_kwargs is None:
            req_extra_args = request.get("extra_args")
            if isinstance(req_extra_args, dict):
                mm_processor_kwargs = req_extra_args.get("mm_processor_kwargs")
        return mm_processor_kwargs

    async def _extract_multimodal_data(
        self,
        request: Dict[str, Any],
        request_id: str,
        context,
        mm_processor_kwargs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        """
        Extract and decode multimodal data from PreprocessedRequest.
        """
        rng = _nvtx.start_range("mm_backend:extract_multimodal_data", color="orange")
        if "multi_modal_data" not in request or request["multi_modal_data"] is None:
            _nvtx.end_range(rng)
            return None

        # Security check: reject multimodal data if not explicitly enabled
        if not self.enable_multimodal:
            raise ValueError(
                "Received multimodal data but multimodal processing is not enabled. "
                "Use --enable-multimodal flag to enable multimodal processing."
            )

        mm_map = request["multi_modal_data"]

        vllm_mm_data = {}

        # [gluo NOTE] If embedding loader is configured, fetch image embeddings first.
        # Still continue below so mixed image+video requests can attach `video`.
        if self.embedding_loader is not None:
            # [gluo FIXME] couldn't simply pass 'mm_map.get(IMAGE_URL_KEY, [])' like below
            # as currently the encode worker is using 'ImageLoader.load_image()' which doesn't
            # support 'Decoded' variant. Need to update the encode worker to unify handling
            image_urls = []
            supported = True
            for item in mm_map.get(IMAGE_URL_KEY, []):
                if isinstance(item, dict) and "Url" in item:
                    image_urls.append(item["Url"])
                elif isinstance(item, dict) and "Decoded" in item:
                    supported = False
            if supported:
                vllm_mm_data = await self.embedding_loader.load_multimodal_embeddings(
                    image_urls, request_id, model=self.config.model, context=context
                )
                logger.debug(
                    f"Fetched multimodal embeddings for {len(vllm_mm_data)} items"
                )

        image_mm_items = mm_map.get(IMAGE_URL_KEY, [])
        # Kimi-K2.5 (and any model with `use_unified_vision_chunk=True`)
        # consumes images under the `vision_chunk` modality, not `image`,
        # and expects each item to be a `VisionChunkImage` TypedDict.
        # See chat_utils.use_unified_vision_chunk_modality for the
        # upstream rename + wrap path.
        image_modality_key = (
            "vision_chunk" if self._use_unified_vision_chunk else "image"
        )
        if image_modality_key not in vllm_mm_data and image_mm_items:
            with _nvtx.annotate("mm_backend:image_download", color="green"):
                images = await self.image_loader.load_image_batch(
                    image_mm_items,
                )

            if images:
                if self._use_unified_vision_chunk:
                    # `VisionChunkImage` is a TypedDict — a plain dict
                    # with `type`/`image`/`uuid` keys is structurally
                    # equivalent. uuid=None matches vLLM's chat_utils
                    # path when the request doesn't pre-supply one.
                    chunks = [
                        {"type": "image", "image": img, "uuid": None} for img in images
                    ]
                    vllm_mm_data["vision_chunk"] = (
                        chunks[0] if len(chunks) == 1 else chunks
                    )
                else:
                    # vLLM expects single image or list
                    vllm_mm_data["image"] = images[0] if len(images) == 1 else images
                logger.debug(
                    f"Extracted {len(images)} image(s) for multimodal "
                    f"processing under modality={image_modality_key!r}"
                )

        video_mm_items = mm_map.get(VIDEO_URL_KEY, [])
        if video_mm_items:
            videos = await self.video_loader.load_video_batch(video_mm_items)

            if videos:
                # vLLM expects single video or list
                vllm_mm_data["video"] = videos[0] if len(videos) == 1 else videos
                logger.debug(
                    f"Extracted {len(videos)} video(s) for multimodal processing"
                )

        # Handle audio_url entries
        audio_mm_items = mm_map.get(AUDIO_URL_KEY, [])
        if audio_mm_items:
            audios = await self.audio_loader.load_audio_batch(audio_mm_items)
            if audios:
                vllm_mm_data["audio"] = audios[0] if len(audios) == 1 else audios
                logger.debug(
                    f"Extracted {len(audios)} audio item(s) for multimodal processing"
                )

        # Extract audio from video URLs when use_audio_in_video is set.
        # Models expect 1:1 audio/video pairing in the same order.
        # We load per-video sequentially to preserve ordering; a video
        # without an audio track raises immediately to avoid corrupting
        # the alignment.
        if (
            video_mm_items
            and mm_processor_kwargs
            and mm_processor_kwargs.get("use_audio_in_video", False)
        ):
            video_audios: list = []
            for item in video_mm_items:
                url = item.get(URL_VARIANT_KEY) if isinstance(item, dict) else None
                if not url:
                    raise ValueError(
                        "use_audio_in_video requires all video items to be "
                        "URL-based. Got a non-URL video item (e.g. frontend-"
                        "decoded). Audio extraction from decoded video data "
                        "is not yet supported."
                    )
                try:
                    audio = await self.audio_loader.load_audio(url)
                    video_audios.append(audio)
                except Exception:
                    logger.error(
                        "Failed to extract audio from video %s. "
                        "use_audio_in_video requires every video to "
                        "contain an audio stream.",
                        url[:80],
                    )
                    raise
            if video_audios:
                existing = vllm_mm_data.get("audio")
                if existing is not None:
                    all_audios = (
                        existing if isinstance(existing, list) else [existing]
                    ) + video_audios
                else:
                    all_audios = video_audios
                vllm_mm_data["audio"] = (
                    all_audios[0] if len(all_audios) == 1 else all_audios
                )
                logger.debug(
                    "Extracted %d audio track(s) from video URL(s) "
                    "(use_audio_in_video=True)",
                    len(video_audios),
                )

        _nvtx.end_range(rng)
        return vllm_mm_data if vllm_mm_data else None

    def _build_prompt_from_request(
        self,
        request: Dict[str, Any],
        request_id: str,
        multi_modal_data: Dict[str, Any] | None,
        log_prefix: str = "",
        mm_processor_kwargs: Dict[str, Any] | None = None,
    ) -> tuple[TokensPrompt | EmbedsPrompt | None, int | None, Dict[str, Any] | None]:
        """
        Build a prompt from request, handling both prompt_embeds and token_ids.

        Args:
            request: The request dict containing either prompt_embeds or token_ids
            request_id: Request ID for logging
            multi_modal_data: Optional multimodal data to attach to TokensPrompt
            log_prefix: Prefix for log messages (e.g., "Prefill " for prefill requests)
            mm_processor_kwargs: Optional multimodal processor kwargs (e.g.
                use_audio_in_video) forwarded to the vLLM engine.

        Returns:
            Tuple of (prompt, embedding_sequence_length, error_dict) where:
            - On success: (prompt, embedding_sequence_length or None, None)
            - On failure: (None, None, error_dict to yield)
        """
        embedding_sequence_length = None

        if "prompt_embeds" in request and request["prompt_embeds"]:
            if not self.config.engine_args.enable_prompt_embeds:
                msg = (
                    "Set `--enable-prompt-embeds` to allow `prompt_embeds` in request."
                )
                logger.error(
                    f"Rejected prompt_embeds for {log_prefix.lower().strip() or 'request'} "
                    f"{request_id}: {msg}"
                )
                return (
                    None,
                    None,
                    {
                        "finish_reason": f"error: Invalid prompt_embeds: {msg}",
                        "token_ids": [],
                    },
                )
            try:
                (
                    prompt,
                    embedding_sequence_length,
                    tensor,
                ) = self._create_prompt_from_embeddings(request["prompt_embeds"])
                logger.info(
                    f"{log_prefix}Using prompt embeddings: shape={tensor.shape}, "
                    f"dtype={tensor.dtype}, sequence_length={embedding_sequence_length}, "
                    f"request_id={request_id}"
                )
                return prompt, embedding_sequence_length, None
            except Exception as e:
                logger.error(
                    f"Failed to process prompt_embeds for {log_prefix.lower().strip() or 'request'} "
                    f"{request_id}: {e}"
                )
                return (
                    None,
                    None,
                    {
                        "finish_reason": f"error: Invalid prompt_embeds: {e}",
                        "token_ids": [],
                    },
                )
        # Normal path: use token IDs.
        # Prefer frontend-forwarded mm_hashes for hash consistency with the
        # routing layer. Fall back to computing from loaded image data when
        # not in EPD mode — in EPD mode multi_modal_data carries pre-computed
        # embeddings from the encode worker, not raw images, and raw-image
        # identity lives upstream at the Router / URL-keyed encoder cache.
        extra_args = request.get("extra_args") or {}
        forwarded_hashes = extra_args.get("mm_hashes")
        mm_uuids: dict[str, Any] | None = None
        if forwarded_hashes:
            # vLLM binds multi_modal_uuids by modality key string match.
            # For models with use_unified_vision_chunk=True (e.g. Kimi-K2.5)
            # images live under `vision_chunk`, not `image`; hardcoding
            # `image` here would silently fail to bind and force vLLM back
            # to its own content-derived hash, breaking router/worker
            # cache-key alignment.
            mm_modality_key = (
                "vision_chunk" if self._use_unified_vision_chunk else "image"
            )
            mm_uuids = {mm_modality_key: forwarded_hashes}
        elif self.embedding_loader is None:
            mm_uuids = _compute_mm_uuids(multi_modal_data)
            if mm_uuids and multi_modal_data:
                logger.warning(
                    "[mm-routing] No forwarded mm_hashes from frontend; "
                    "recomputed from image data. KV-cache-aware MM routing "
                    "may not match the frontend's routing decisions."
                )
        prompt_kwargs = dict[str, Any](
            prompt_token_ids=request["token_ids"],
            multi_modal_data=multi_modal_data,
        )
        if mm_uuids is not None:
            prompt_kwargs["multi_modal_uuids"] = mm_uuids
        if mm_processor_kwargs is not None:
            prompt_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

        prompt = TokensPrompt(**prompt_kwargs)
        return prompt, embedding_sequence_length, None

    @staticmethod
    def _build_completion_usage(
        request_output: RequestOutput,
        embedding_sequence_length: int | None = None,
    ) -> Dict[str, Any]:
        """
        Build completion usage statistics.

        Args:
            request_output: vLLM RequestOutput object
            embedding_sequence_length: If using prompt embeddings, the sequence length
                                     extracted from the embeddings tensor shape

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details
        """
        # Determine prompt token count:
        # - For embeddings: use embedding_sequence_length from tensor shape
        # - For normal text: use len(prompt_token_ids)
        if embedding_sequence_length is not None:
            prompt_tokens = embedding_sequence_length
        elif request_output.prompt_token_ids:
            prompt_tokens = len(request_output.prompt_token_ids)
        else:
            prompt_tokens = None

        completion_tokens = sum(
            len(output.token_ids) for output in request_output.outputs
        )

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens + completion_tokens if prompt_tokens is not None else None
            ),
            "prompt_tokens_details": (
                {"cached_tokens": num_cached}
                if (num_cached := getattr(request_output, "num_cached_tokens", None))
                else None
            ),
        }

    @staticmethod
    def _extract_logprobs(
        output, num_output_tokens_so_far: int, tokenizer=None
    ) -> tuple[list[float] | None, list[list[dict]] | None]:
        """
        Extract logprobs from vLLM CompletionOutput for new tokens.

        Args:
            output: vLLM CompletionOutput object
            num_output_tokens_so_far: Number of tokens already processed
            tokenizer: Optional tokenizer for decoding token IDs when
                       decoded_token is not populated by the engine

        Returns:
            Tuple of (log_probs, top_logprobs) in Dynamo's expected format:
            - log_probs: List of log probabilities for each new token
            - top_logprobs: List of top logprobs dicts for each new token
        """
        if output.logprobs is None:
            return None, None

        # Get logprobs for new tokens only
        new_logprobs = output.logprobs[num_output_tokens_so_far:]
        if not new_logprobs:
            return None, None

        log_probs = []
        top_logprobs = []

        for token_idx, token_logprobs_dict in enumerate(new_logprobs):
            if token_logprobs_dict is None:
                continue

            # Get the actual token_id that was generated at this position
            actual_token_id = output.token_ids[num_output_tokens_so_far + token_idx]

            # Extract log probability for the selected token
            # vLLM guarantees the selected token is always in the logprobs dict
            selected_logprob = token_logprobs_dict[actual_token_id]
            log_probs.append(float(selected_logprob.logprob))

            # Build top_logprobs list for this token position
            token_top_logprobs = []
            for tok_id, logprob_info in token_logprobs_dict.items():
                token_str = getattr(logprob_info, "decoded_token", None)
                if not token_str and tokenizer:
                    try:
                        token_str = tokenizer.decode([tok_id])
                    except Exception:
                        token_str = None
                token_top_logprobs.append(
                    {
                        "rank": (
                            logprob_info.rank if hasattr(logprob_info, "rank") else 0
                        ),
                        "token_id": tok_id,
                        "token": token_str,
                        "logprob": float(logprob_info.logprob),
                        "bytes": (
                            list(token_str.encode("utf-8")) if token_str else None
                        ),
                    }
                )
            top_logprobs.append(token_top_logprobs)

        return log_probs if log_probs else None, top_logprobs if top_logprobs else None

    @staticmethod
    def _log_with_lora_context(
        message: str,
        request_id: str,
        lora_request=None,
        level: str = "debug",
        **kwargs,
    ) -> None:
        """
        Log a message with optional LoRA context.

        Args:
            message: Base message to log (can include {lora_info} placeholder)
            request_id: Request ID for correlation
            lora_request: Optional LoRA request object
            level: Log level ("debug" or "info")
            **kwargs: Additional format arguments for the message
        """
        if lora_request:
            lora_info = f" with LoRA {lora_request.lora_name}"
        else:
            lora_info = ""

        formatted_message = message.format(
            request_id=request_id,
            lora_info=lora_info,
            **kwargs,
        )

        if level == "info":
            logger.info(formatted_message)
        else:
            logger.debug(formatted_message)

    async def generate_tokens(
        self,
        prompt,
        sampling_params,
        request_id,
        data_parallel_rank=None,
        lora_request=None,
        embedding_sequence_length=None,
        trace_headers=None,
        priority=0,
        reasoning_ended=None,
        reasoning_parser_kwargs=None,
        enforce_decode_timeout: bool = False,
        release_request_admission: bool = False,
        abort_guard: Optional[_DeferredAbort] = None,
    ):
        admission_handed_to_iterator = False
        try:
            # Log LoRA usage for this generation (debug level to avoid log spam)
            self._log_with_lora_context(
                "Starting token generation for request {request_id}{lora_info}",
                request_id,
                lora_request,
            )
            gen = self.engine_client.generate(
                prompt,
                sampling_params,
                request_id,
                lora_request=lora_request,
                data_parallel_rank=data_parallel_rank,
                trace_headers=trace_headers,
                priority=priority,
                **_engine_generate_reasoning_kwargs(
                    self.engine_client,
                    reasoning_ended,
                    reasoning_parser_kwargs,
                ),
            )

            num_output_tokens_so_far: dict[int, int] = {}
            emitted_parser_visible_marker: dict[int, bool] = {}
            parser_visible_marker_ids = self._parser_visible_marker_token_ids()
            admission_handed_to_iterator = True
            async for res in self._iterate_engine_stream(
                gen,
                request_id,
                enforce_decode_timeout=enforce_decode_timeout,
                release_request_admission=release_request_admission,
                abort_guard=abort_guard,
            ):
                # res is vllm's RequestOutput

                if not res.outputs:
                    # MTP/spec-decode: a non-finished step can yield no outputs
                    # (all draft tokens rejected). Transient -> skip, not fatal.
                    if not getattr(res, "finished", False):
                        continue
                    self._log_with_lora_context(
                        "Request {request_id}{lora_info} returned no outputs",
                        request_id,
                        lora_request,
                    )
                    # string form parsed by Rust into FinishReason::Error(message)
                    yield {
                        "finish_reason": "error: No outputs from vLLM engine",
                        "index": 0,
                        "token_ids": [],
                    }
                    break

                for output in res.outputs:
                    output_idx = getattr(output, "index", 0) or 0
                    previous_total_toks = num_output_tokens_so_far.get(output_idx, 0)
                    next_total_toks = len(output.token_ids)
                    delta_token_ids = output.token_ids[previous_total_toks:]
                    had_emitted_parser_marker = emitted_parser_visible_marker.get(
                        output_idx, False
                    )
                    out = {
                        "index": output_idx,
                        "token_ids": delta_token_ids,
                    }
                    delta_has_parser_marker = parser_visible_marker_ids and any(
                        token_id in parser_visible_marker_ids
                        for token_id in delta_token_ids
                    )
                    if delta_has_parser_marker:
                        decoded_text = self._decode_parser_visible_text(delta_token_ids)
                        if decoded_text is not None:
                            out["text"] = decoded_text
                            emitted_parser_visible_marker[output_idx] = True

                    # Extract logprobs for new tokens if available
                    tokenizer = getattr(self.engine_client, "tokenizer", None)
                    log_probs, top_logprobs = self._extract_logprobs(
                        output, previous_total_toks, tokenizer=tokenizer
                    )
                    if log_probs is not None:
                        out["log_probs"] = log_probs
                    if top_logprobs is not None:
                        out["top_logprobs"] = top_logprobs

                    if output.finish_reason:
                        out["finish_reason"] = normalize_finish_reason(
                            output.finish_reason
                        )
                        out[
                            "completion_usage"
                        ] = BaseWorkerHandler._build_completion_usage(
                            request_output=res,
                            embedding_sequence_length=embedding_sequence_length,
                        )
                        for _ctx_key in (
                            "spec_decode",
                            "scheduler_snapshot",
                            "request_throughput",
                        ):
                            _ctx_val = getattr(output, _ctx_key, None)
                            if _ctx_val:
                                extra_args = out.get("extra_args") or {}
                                extra_args[_ctx_key] = _ctx_val
                                out["extra_args"] = extra_args
                        # Log completion with LoRA info (debug level to avoid log spam)
                        self._log_with_lora_context(
                            "Completed token generation for request {request_id}{lora_info}: "
                            "{output_tokens} output tokens, finish_reason={finish_reason}",
                            request_id,
                            lora_request,
                            output_tokens=next_total_toks,
                            finish_reason=output.finish_reason,
                        )
                        if parser_visible_marker_ids and not had_emitted_parser_marker:
                            token_ids = list(output.token_ids)
                            first_marker_pos = next(
                                (
                                    idx
                                    for idx, token_id in enumerate(token_ids)
                                    if token_id in parser_visible_marker_ids
                                ),
                                None,
                            )
                            if (
                                first_marker_pos is not None
                                and first_marker_pos < previous_total_toks
                            ):
                                decoded_text = self._decode_parser_visible_text(
                                    token_ids[first_marker_pos:]
                                )
                                if decoded_text is not None:
                                    out["text"] = decoded_text
                                    emitted_parser_visible_marker[output_idx] = True
                        if os.environ.get("DYN_DEBUG_REASONING_BUDGET") == "1":
                            token_ids = list(output.token_ids)
                            marker_positions: dict[str, list[int]] = {}
                            for marker_id, marker_text in (
                                self._parser_visible_marker_token_map().items()
                            ):
                                positions = [
                                    idx
                                    for idx, token_id in enumerate(token_ids)
                                    if token_id == marker_id
                                ]
                                if positions:
                                    marker_positions[marker_text] = positions[-8:]
                            logger.warning(
                                "vllm final output request_id=%s index=%s "
                                "finish_reason=%s stop_reason=%s "
                                "contains_13=%s token_tail=%s "
                                "marker_positions=%s",
                                request_id,
                                output_idx,
                                output.finish_reason,
                                getattr(output, "stop_reason", None),
                                13 in token_ids,
                                token_ids[-16:],
                                marker_positions,
                            )
                    if output.stop_reason:
                        out["stop_reason"] = output.stop_reason
                    yield out
                    num_output_tokens_so_far[output_idx] = next_total_toks

        except EngineDeadError as e:
            if release_request_admission and not admission_handed_to_iterator:
                await self._release_request_slot_reservation(request_id)
            logger.error(f"vLLM EngineDeadError: {e}")
            logger.warning("Initiating Dynamo Runtime shutdown.")
            self.runtime.shutdown()
            os._exit(1)
        except Exception:
            if release_request_admission and not admission_handed_to_iterator:
                await self._release_request_slot_reservation(request_id)
            raise


class DecodeWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        runtime,
        config: Config,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        model_config: ModelConfig | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
        encode_worker_client: Client | None = None,
    ):
        super().__init__(
            runtime,
            config,
            engine,
            default_sampling_params,
            model_max_len=model_max_len,
            model_config=model_config,
            enable_multimodal=enable_multimodal,
            generate_endpoint=generate_endpoint,
            use_vllm_tokenizer=use_vllm_tokenizer,
            shutdown_event=shutdown_event,
            enable_frontend_decoding=enable_frontend_decoding,
            encode_worker_client=encode_worker_client,
        )

    async def generate(self, request, context):
        # Use context ID for request tracking and correlation
        request_id = context.id()
        logger.debug(f"Decode Request ID: {request_id}")
        first_token = True
        with time_and_log_code_section(
            f"[DECODE] request: {request_id} generate"
        ) as decode_timer:
            if self.use_vllm_tokenizer:
                # Text-in-text-out mode: use InputParamManager and OpenAI-compatible format
                generator = self._generate_text_mode(request, context, request_id)
            else:
                # Token-in-token-out mode: internal protocol format
                generator = self._generate_token_mode(request, context, request_id)

            async for chunk in generator:
                if first_token:
                    decode_timer.stop_interval()
                    first_token = False
                yield chunk

    async def _generate_token_mode(self, request, context, request_id):
        """Generate tokens using internal protocol format (token-in-token-out)."""
        # Firstly extract disaggregated params from prefill result if available
        prefill_result = request.get("prefill_result")
        if prefill_result and isinstance(prefill_result, dict):
            kv_params = prefill_result.get("disaggregated_params", {}).get(
                "kv_transfer_params"
            )
            embedding_params = prefill_result.get("disaggregated_params", {}).get(
                "embedding_params"
            )
            # Normalize embedding_params to None if it is an empty dict
            if not embedding_params:
                embedding_params = None
        else:
            kv_params = None
            embedding_params = None

        is_decode_only = self.config.disaggregation_mode == DisaggregationMode.DECODE
        has_mm_data = (
            "multi_modal_data" in request and request["multi_modal_data"] is not None
        )

        mm_processor_kwargs = self._get_mm_processor_kwargs(request)

        multi_modal_data: Dict[str, Any] | None = None
        pre_rendered: Dict[str, Any] | None = None
        if is_decode_only:
            # Decode mode: branch on model, not data.
            if resolve_model_family(self.config.model) is ModelFamily.QWEN_VL:
                # Qwen VL needs embedding_params for mRoPE initialization.
                if embedding_params is not None:
                    multi_modal_data = construct_qwen_decode_mm_data(
                        embedding_params["image_grid_thw"],
                        embedding_params["embeddings_shape"],
                        request_id,
                    )
                elif has_mm_data and request["multi_modal_data"].get(IMAGE_URL_KEY):
                    # Guard is on IMAGE_URL_KEY (not just has_mm_data) so
                    # text-only requests pass through and video/audio fall
                    # through to re-download below (TODO: proper support).
                    msg = (
                        "Decode worker received multimodal request without "
                        "prefill result"
                        if prefill_result is None
                        else "Prefill did not produce required multimodal "
                        "embedding metadata (image_grid_thw) for Qwen VL "
                        "decode. Use --route-to-encoder or the P/D launcher "
                        "with grid_thw computation support"
                    )
                    logger.error("Request %s: %s", request_id, msg)
                    yield {"status": "error", "message": msg}
                    return
            else:
                # Non-qwen model, assume the multi_modal_data has been consumed
                # in prefill, so we can use the expanded prompt token ids
                # without multimodal data
                if embedding_params and "expanded_prompt_token_ids" in embedding_params:
                    request["token_ids"] = embedding_params["expanded_prompt_token_ids"]
                    has_mm_data = False
            # TODO(DIS-1661): video/audio re-downloaded on decode.
            # TODO(DIS-1664): mixed image+video in disagg decode is not
            # supported — synthetic image data would be overwritten.
            if multi_modal_data is None and has_mm_data:
                mm = request["multi_modal_data"]
                if mm.get(VIDEO_URL_KEY) or mm.get(AUDIO_URL_KEY):
                    multi_modal_data = await self._extract_multimodal_data(
                        request,
                        request_id,
                        context,
                        mm_processor_kwargs=mm_processor_kwargs,
                    )
        else:
            # Fast path: check for pre-processed mm_kwargs via NIXL/SHM from frontend.
            # If available, we skip image downloading AND the HF processor.
            with _nvtx.annotate("mm_backend:receive_mm_kwargs", color="magenta"):
                pre_rendered = await self._try_receive_mm_kwargs(request)
            if pre_rendered is not None:
                logger.debug(
                    "[mm-routing] Request %s: received pre-rendered mm_kwargs via NIXL/SHM",
                    request_id,
                )
            else:
                # Aggregated mode: load images normally
                multi_modal_data = await self._extract_multimodal_data(
                    request,
                    request_id,
                    context,
                    mm_processor_kwargs=mm_processor_kwargs,
                )

        # Build prompt from request. `prompt` is either a pre-rendered
        # MultiModalInput dict (fast path) or a TokensPrompt/EmbedsPrompt from
        # `_build_prompt_from_request`. Declare as Any so mypy accepts both
        # branches without spelling out the full union.
        prompt: Any
        with _nvtx.annotate("mm_backend:build_prompt", color="yellow"):
            if pre_rendered is not None:
                # pre_rendered is a MultiModalInput dict with "type": "multimodal".
                # The engine's InputProcessor.process_inputs() will see the "type"
                # key and skip the HF processor entirely.
                prompt = pre_rendered
                embedding_sequence_length = None
                error = None
                logger.debug(
                    "[mm-routing] Request %s: using pre-rendered MultiModalInput",
                    request_id,
                )
            else:
                (
                    prompt,
                    embedding_sequence_length,
                    error,
                ) = self._build_prompt_from_request(
                    request,
                    request_id,
                    multi_modal_data,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
        if error is not None:
            yield error
            return

        # Build sampling params from request
        sampling_params = build_sampling_params(
            request, self.default_sampling_params, self.model_max_len
        )
        self._apply_reasoning_budget_extra_args(sampling_params)
        if os.environ.get("DYN_DEBUG_REASONING_BUDGET") == "1":
            extra_args = (
                sampling_params.extra_args
                if isinstance(sampling_params.extra_args, dict)
                else {}
            )
            logger.warning(
                "reasoning budget sampling params request_id=%s "
                "budget=%s grace=%s end_token_ids=%s stop_token_ids=%s "
                "all_stop_token_ids=%s",
                request_id,
                extra_args.get("reasoning_budget"),
                extra_args.get("reasoning_budget_grace_period"),
                extra_args.get("end_token_ids"),
                getattr(sampling_params, "stop_token_ids", None),
                getattr(sampling_params, "all_stop_token_ids", None),
            )

        if kv_params is not None:
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args["kv_transfer_params"] = kv_params
            logger.debug(
                f"Using disaggregated params from prefill for request {request_id}"
            )
        prefill_prompt_tokens_details = (
            prefill_result.get("prompt_tokens_details") if prefill_result else None
        )

        # Extract LoRA request if present
        model_name = request.get("model")
        lora_request = self._resolve_lora_request(model_name)
        if lora_request:
            logger.info(
                f"Decode request {request_id} will use LoRA adapter: {model_name} (ID: {lora_request.lora_int_id})"
            )
        else:
            logger.debug(
                f"Decode request {request_id} has no LoRA specified (model: {model_name})"
            )
        routing = request.get("routing") or {}
        dp_rank = self._to_local_dp_rank(routing.get("dp_rank"))
        priority = -int(routing.get("priority", 0))

        reserved, current_total_requests, total_request_limit = (
            await self._try_reserve_request_slot(request_id, request)
        )
        if not reserved:
            message = (
                f"Worker local total request limit reached "
                f"({current_total_requests}/{total_request_limit})"
            )
            yield _build_token_overload_response(
                message,
                current_total_requests=current_total_requests,
                total_request_limit=total_request_limit,
            )
            return

        trace_headers = build_trace_headers(context)
        reasoning_ended, reasoning_parser_kwargs = _request_reasoning_metadata(request)

        # In disagg decode mode, defer engine_client.abort() until the first
        # token so we don't abort while a NIXL KV transfer is still in flight
        # on the decode worker (which can crash EngineCore). The guard's
        # cleanup runs after _abort_monitor tears down its monitor task, so
        # any deferred-abort waiter spawned by the monitor is in a stable
        # state when close() is awaited.
        async with _deferred_abort_guard(
            self.engine_client, request_id, is_decode_only
        ) as abort_guard:
            async with self._abort_monitor(
                context, request_id, abort_guard=abort_guard
            ):
                try:
                    async for tok in self.generate_tokens(
                        prompt,
                        sampling_params,
                        request_id,
                        data_parallel_rank=dp_rank,
                        lora_request=lora_request,
                        embedding_sequence_length=embedding_sequence_length,
                        trace_headers=trace_headers,
                        priority=priority,
                        reasoning_ended=reasoning_ended,
                        reasoning_parser_kwargs=reasoning_parser_kwargs,
                        enforce_decode_timeout=not self._is_health_check_request(
                            request
                        ),
                        release_request_admission=reserved,
                        abort_guard=abort_guard,
                    ):
                        if abort_guard is not None:
                            abort_guard.signal_first_token()
                        if prefill_result is not None and "completion_usage" in tok:
                            tok["completion_usage"][
                                "prompt_tokens_details"
                            ] = prefill_prompt_tokens_details
                        yield tok
                except DecodeWallClockTimeoutError as e:
                    yield _build_token_error_response(str(e))
                except EngineDeadError as e:
                    logger.error(f"vLLM EngineDeadError: {e}")
                    logger.warning("Initiating Dynamo Runtime shutdown.")
                    self.runtime.shutdown()
                    os._exit(1)

    async def _generate_text_mode(self, request, context, request_id):
        """Generate text using OpenAI-compatible format (text-in-text-out)."""
        # Get text input using InputParamManager
        input_data = self.input_param_manager.get_input_param(
            request, use_tokenizer=True
        )

        # Build prompt for vLLM
        if isinstance(input_data, list):
            prompt = TokensPrompt(prompt_token_ids=input_data)
        else:
            prompt = TextPrompt(prompt=input_data)

        # Build sampling params from OpenAI-style request
        sampling_params = build_sampling_params_openai(
            request,
            self.default_sampling_params,
            tool_call_parser_name=self.config.dyn_tool_call_parser,
        )
        self._apply_reasoning_budget_extra_args(sampling_params)

        routing = request.get("routing") or {}
        dp_rank = self._to_local_dp_rank(routing.get("dp_rank"))
        priority = -int(routing.get("priority", 0))
        openai_request_id = request.get("id") or request.get("request_id", request_id)
        previous_text_per_choice: dict[int, str] = {}
        structured_json_guards: dict[int, _StructuredJsonStreamGuard] | None = (
            {} if _request_has_pure_json_structured_output(request) else None
        )
        early_stop_structured_json = (
            structured_json_guards is not None
            and int(getattr(sampling_params, "n", 1) or 1) == 1
        )

        reserved, current_total_requests, total_request_limit = (
            await self._try_reserve_request_slot(request_id, request)
        )
        if not reserved:
            message = (
                f"Worker local total request limit reached "
                f"({current_total_requests}/{total_request_limit})"
            )
            yield _build_text_overload_chunk(
                message,
                openai_request_id,
                current_total_requests=current_total_requests,
                total_request_limit=total_request_limit,
            )
            return

        trace_headers = build_trace_headers(context)
        admission_handed_to_iterator = False

        async with self._abort_monitor(context, request_id):
            try:
                gen = self.engine_client.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    data_parallel_rank=dp_rank,
                    trace_headers=trace_headers,
                    priority=priority,
                )

                admission_handed_to_iterator = True
                async for res in self._iterate_engine_stream(
                    gen,
                    request_id,
                    enforce_decode_timeout=not self._is_health_check_request(request),
                    release_request_admission=reserved,
                ):
                    if not res.outputs:
                        yield {
                            "id": openai_request_id,
                            "created": int(time.time()),
                            "object": "chat.completion.chunk",
                            "model": "unknown",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": "error",
                                }
                            ],
                        }
                        break

                    for output in res.outputs:
                        output_idx = getattr(output, "index", 0) or 0
                        previous_text = previous_text_per_choice.get(output_idx, "")
                        # Calculate the delta text (new text since last chunk)
                        delta_text = output.text[len(previous_text) :]
                        structured_json_complete = False
                        if structured_json_guards is not None:
                            guard = structured_json_guards.setdefault(
                                output_idx, _StructuredJsonStreamGuard()
                            )
                            delta_text = guard.filter_delta(delta_text)
                            structured_json_complete = bool(delta_text and guard.emitted)
                        finish_reason = normalize_finish_reason(output.finish_reason)
                        if structured_json_complete and finish_reason is None:
                            finish_reason = "stop"

                        tool_calls = (
                            _forced_tool_calls_from_post_reasoning_json(
                                request,
                                delta_text,
                            )
                            if output.finish_reason
                            else None
                        )
                        if tool_calls:
                            delta = {
                                "role": "assistant",
                                "tool_calls": tool_calls,
                            }
                            finish_reason = "tool_calls"
                        else:
                            delta = {
                                "role": "assistant",
                                "content": delta_text,
                            }

                        choice_data = {
                            "index": output_idx,
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }

                        chunk = {
                            "id": openai_request_id,
                            "created": int(time.time()),
                            "object": "chat.completion.chunk",
                            "model": "unknown",
                            "choices": [choice_data],
                        }

                        if output.finish_reason:
                            chunk["usage"] = BaseWorkerHandler._build_completion_usage(
                                request_output=res,
                            )

                        yield chunk
                        previous_text_per_choice[output_idx] = output.text
                        if (
                            early_stop_structured_json
                            and structured_json_complete
                            and output.finish_reason is None
                        ):
                            try:
                                await self.engine_client.abort(request_id)
                            except Exception as abort_error:
                                logger.warning(
                                    "Failed to abort structured JSON request %s "
                                    "after complete value: %s",
                                    request_id,
                                    abort_error,
                                )
                            return

            except DecodeWallClockTimeoutError as e:
                yield _build_text_error_chunk(str(e), openai_request_id)
            except EngineDeadError as e:
                if reserved and not admission_handed_to_iterator:
                    await self._release_request_slot_reservation(request_id)
                logger.error(f"vLLM EngineDeadError: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)
            except Exception:
                if reserved and not admission_handed_to_iterator:
                    await self._release_request_slot_reservation(request_id)
                raise


class PrefillWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        runtime,
        config: Config,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        model_config: ModelConfig | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
        encode_worker_client: Client | None = None,
    ):
        super().__init__(
            runtime,
            config,
            engine,
            default_sampling_params,
            model_max_len=model_max_len,
            model_config=model_config,
            enable_multimodal=enable_multimodal,
            generate_endpoint=generate_endpoint,
            use_vllm_tokenizer=use_vllm_tokenizer,
            shutdown_event=shutdown_event,
            enable_frontend_decoding=enable_frontend_decoding,
            encode_worker_client=encode_worker_client,
        )

        # Cache Qwen VL grid parameters for computing image_grid_thw from
        # PIL images in the P/D path (no separate encode worker).
        if resolve_model_family(config.model) is ModelFamily.QWEN_VL:
            self._qwen_grid_params = load_qwen_grid_params(config.model)
            if self._qwen_grid_params is None and self.embedding_loader is None:
                logger.error(
                    "Qwen VL grid params failed to load and no encode worker "
                    "is configured. P/D multimodal requests will fail because "
                    "prefill cannot produce embedding_params for decode. "
                    "Use --route-to-encoder or ensure the model is cached."
                )
        else:
            self._qwen_grid_params = None

    async def generate(self, request, context):
        # Use context ID for request tracking and correlation with decode phase
        request_id = context.id()
        logger.debug(f"Prefill Request ID: {request_id}")

        # Token-in-token-out mode: internal protocol format
        with time_and_log_code_section(f"[PREFILL] request: {request_id} generate"):
            async for chunk in self._generate_token_mode(request, context, request_id):
                yield chunk

    async def _generate_token_mode(self, request, context, request_id):
        """Generate prefill using internal protocol format (token-in-token-out)."""
        # TODO: Wire up NIXL mm_kwargs passthrough for disaggregated prefill
        # (similar to DecodeWorkerHandler). For now, prefill
        # always downloads and processes images via the standard path.
        mm_processor_kwargs = self._get_mm_processor_kwargs(request)

        # Extract and decode multimodal data if present
        multi_modal_data = await self._extract_multimodal_data(
            request,
            request_id,
            context,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        # Build prompt from request (handles both prompt_embeds and token_ids)
        prompt, embedding_sequence_length, error = self._build_prompt_from_request(
            request,
            request_id,
            multi_modal_data,
            log_prefix="Prefill ",
            mm_processor_kwargs=mm_processor_kwargs,
        )
        if error is not None:
            # Prefill errors need disaggregated_params field
            error["disaggregated_params"] = None
            yield error
            return

        # Build sampling params from request using shared utility
        sampling_params = build_sampling_params(
            request, self.default_sampling_params, self.model_max_len
        )
        self._apply_reasoning_budget_extra_args(sampling_params)

        # Configure for prefill-only mode with remote decode
        if sampling_params.extra_args is None:
            sampling_params.extra_args = {}
        sampling_params.extra_args["kv_transfer_params"] = {
            "do_remote_decode": True,
        }
        sampling_params_defaults = {
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        # Add only missing keys
        for k, v in sampling_params_defaults.items():
            sampling_params.extra_args["kv_transfer_params"].setdefault(k, v)
        # Override for prefill: only generate 1 token
        sampling_params.max_tokens = 1
        sampling_params.min_tokens = 1

        # Extract LoRA request if present
        model_name = request.get("model")
        lora_request = self._resolve_lora_request(model_name)
        if lora_request:
            logger.info(
                f"Prefill request {request_id} will use LoRA adapter: {model_name} "
                f"(ID: {lora_request.lora_int_id}), path: {lora_request.lora_path}"
            )
        else:
            logger.debug(
                f"Prefill request {request_id} has no LoRA specified (model: {model_name})"
            )

        routing = request.get("routing") or {}
        dp_rank = self._to_local_dp_rank(routing.get("dp_rank"))
        priority = -int(routing.get("priority", 0))

        reserved, current_total_requests, total_request_limit = (
            await self._try_reserve_request_slot(request_id, request)
        )
        if not reserved:
            message = (
                f"Worker local total request limit reached "
                f"({current_total_requests}/{total_request_limit})"
            )
            yield _build_prefill_overload_response(
                message,
                current_total_requests=current_total_requests,
                total_request_limit=total_request_limit,
            )
            return

        trace_headers = build_trace_headers(context)
        reasoning_ended, reasoning_parser_kwargs = _request_reasoning_metadata(request)
        admission_handed_to_iterator = False

        async with self._abort_monitor(context, request_id, is_prefill=True):
            try:
                gen = self.engine_client.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    data_parallel_rank=dp_rank,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=priority,
                    **_engine_generate_reasoning_kwargs(
                        self.engine_client,
                        reasoning_ended,
                        reasoning_parser_kwargs,
                    ),
                )
                admission_handed_to_iterator = True
            except EngineDeadError as e:
                if reserved and not admission_handed_to_iterator:
                    await self._release_request_slot_reservation(request_id)
                logger.error(f"vLLM EngineDeadError: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)
            except Exception:
                if reserved and not admission_handed_to_iterator:
                    await self._release_request_slot_reservation(request_id)
                raise

            async for res in self._iterate_engine_stream(
                gen,
                request_id,
                release_request_admission=reserved,
            ):
                logger.debug(f"kv transfer params: {res.kv_transfer_params}")

                token_ids = res.outputs[0].token_ids if res.outputs else []

                # For prefill worker, only one res will be generated,
                # so we can always build embedding params here without conditionals
                embedding_params = self._build_embedding_params(
                    multi_modal_data or {}, res.prompt_token_ids
                )
                output: Dict[str, Any] = {
                    "token_ids": list(token_ids),
                    "disaggregated_params": self._build_disaggregated_params(
                        res.kv_transfer_params,
                        embedding_params,
                    ),
                    "completion_usage": BaseWorkerHandler._build_completion_usage(
                        request_output=res,
                        embedding_sequence_length=embedding_sequence_length,
                    ),
                }

                # Log prefill completion with LoRA info
                self._log_with_lora_context(
                    "Prefill completed for request {request_id}{lora_info}: "
                    "generated {token_count} token(s), has_kv_params={has_kv_params}",
                    request_id,
                    lora_request,
                    level="info" if lora_request else "debug",
                    token_count=len(token_ids),
                    has_kv_params=res.kv_transfer_params is not None,
                )

                yield output

    def _build_disaggregated_params(
        self, kv_transfer_params, embedding_params=None, expanded_prompt_token_ids=None
    ):
        disaggregated_params = {}
        if kv_transfer_params is not None:
            disaggregated_params["kv_transfer_params"] = kv_transfer_params
        if embedding_params is not None:
            disaggregated_params["embedding_params"] = embedding_params
        if expanded_prompt_token_ids is not None:
            disaggregated_params[
                "expanded_prompt_token_ids"
            ] = expanded_prompt_token_ids

        return disaggregated_params if disaggregated_params else None

    def _build_embedding_params(
        self, multi_modal_data: dict[str, Any], prompt_token_ids: list[int]
    ) -> Dict[str, Any] | None:
        # [gluo NOTE] there could be different model architectures that
        # need different embedding params, will add more logic if needed
        if resolve_model_family(self.config.model) is not ModelFamily.QWEN_VL:
            # For non-qwen models, vLLM doesn't trigger mm preprocess so
            # decode worker only needs expanded prompt to properly fetch KV blocks
            # from prefill.
            if multi_modal_data:
                return {"expanded_prompt_token_ids": prompt_token_ids}
        else:
            # For qwen models, vLLM triggers mm preprocess so decode worker will
            # perform token expansion unconditionally, so we need to pass
            # original prompt and sufficient metadata to reconstruct mm embedding
            # as request input.
            return build_qwen_embedding_params(multi_modal_data, self._qwen_grid_params)
        return None
