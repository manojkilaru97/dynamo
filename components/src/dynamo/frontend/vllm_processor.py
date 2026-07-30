#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

#
# Use vllm for input and output processing
#

import asyncio
import json
import logging
import os
import time
from argparse import Namespace
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

from msgspec.structs import replace as msgspec_replace
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.config import CacheConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.tasks import GENERATION_TASKS
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, OutputProcessorOutput
from vllm.v1.engine.parallel_sampling import ParentRequest

from dynamo._internal import ModelDeploymentCard
from dynamo.common.multimodal.mm_kwargs_transfer import (
    MmKwargsNixlSender,
    MmKwargsSender,
    MmKwargsShmSender,
)
from dynamo.common.multimodal.routing_utils import build_mm_routing_info_from_features
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.frontend.frontend_args import FrontendConfig
from dynamo.llm import (
    KvRouter,
    ModelCardInstanceId,
    PythonAsyncEngine,
    RouterConfig,
    RouterMode,
    fetch_model,
)
from dynamo.runtime import Client, DistributedRuntime

from .prepost import StreamingPostProcessor, preprocess_chat_request
from .utils import (
    extract_mm_urls,
    handle_engine_error,
    make_internal_error,
    random_uuid,
)

logger = logging.getLogger(__name__)


_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "eos": FinishReason.STOP,
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "error": FinishReason.ERROR,
    "cancelled": FinishReason.ABORT,
    "content_filter": FinishReason.STOP,
}

DEFAULT_TOOL_SCHEMA_MAX_STRING_LENGTH = 4096
DEFAULT_TOOL_SCHEMA_MAX_ARRAY_ITEMS = 32
DEFAULT_TOOL_SHORT_TEXT_MAX_LENGTH = 256
DEFAULT_TOOL_LONG_TEXT_MAX_LENGTH = 8192
TOOL_LONG_REQUEST_THRESHOLD = 2048
TOOL_LONG_REQUEST_MARGIN = 512
TOOL_FIELD_STRING_BUDGETS = {
    "expression": 256,
}
TOOL_LONG_TEXT_FIELD_NAMES = {"body", "content", "message"}
TOOL_CHOICE_SCHEMA_MARKER = "x-dynamo-tool-choice-schema"
QWEN_XML_STRUCTURAL_TAG_TOOL_PARSERS = {"qwen3_coder", "qwen3_xml"}


def map_finish_reason(raw_reason: str | None) -> FinishReason | None:
    if raw_reason is None:
        return None
    if raw_reason.startswith("error"):
        return FinishReason.ERROR
    if raw_reason.startswith("abort"):
        return FinishReason.ABORT
    if raw_reason.startswith("content_filter"):
        logger.info("Router finish_reason indicates content filtering: %s", raw_reason)
        raw_reason = "content_filter"
    mapped = _FINISH_REASON_MAP.get(raw_reason)
    if mapped is None:
        logger.warning("Unknown finish_reason from router: %s", raw_reason)
    return mapped


def _with_parser_visible_engine_text(output: Any, engine_text: str | None) -> Any:
    if not engine_text or getattr(output, "text", None):
        return output
    return SimpleNamespace(
        index=output.index,
        token_ids=output.token_ids,
        text=engine_text,
        finish_reason=output.finish_reason,
        logprobs=output.logprobs,
    )


def _build_reasoning_parser_metadata(
    reasoning_parser_class: type[ReasoningParser] | None,
    tokenizer: TokenizerLike,
    chat_template_kwargs: dict[str, Any],
    request_for_sampling: Any,
    prompt_token_ids: list[int],
) -> tuple[bool | None, dict[str, Any] | None]:
    parser_kwargs = {"chat_template_kwargs": chat_template_kwargs}
    if not getattr(request_for_sampling, "include_reasoning", True):
        return True, parser_kwargs
    if getattr(request_for_sampling, "_grammar_from_tool_parser", False):
        return True, parser_kwargs
    if chat_template_kwargs.get("enable_thinking") is False:
        return True, parser_kwargs

    if reasoning_parser_class is None:
        return None, None

    reasoning_parser = reasoning_parser_class(
        tokenizer,
        chat_template_kwargs=chat_template_kwargs,
    )
    return reasoning_parser.is_reasoning_end(prompt_token_ids), parser_kwargs


def _structured_outputs_to_guided_decoding(
    structured_outputs: StructuredOutputsParams | None,
) -> dict[str, Any] | None:
    if structured_outputs is None or structured_outputs.all_constraints_none():
        return None

    guided_decoding: dict[str, Any] = {}
    for key in (
        "json",
        "regex",
        "choice",
        "grammar",
        "json_object",
        "structural_tag",
        "disable_any_whitespace",
        "disable_additional_properties",
        "whitespace_pattern",
    ):
        value = getattr(structured_outputs, key, None)
        if value is not None:
            guided_decoding[key] = value
    return guided_decoding or None


def _get_attr_or_item(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _copy_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    try:
        return json.loads(json.dumps(value))
    except TypeError:
        return value


def _tool_text_budget(request_text_len: int | None) -> int:
    if request_text_len is None or request_text_len < TOOL_LONG_REQUEST_THRESHOLD:
        return DEFAULT_TOOL_SHORT_TEXT_MAX_LENGTH
    return max(
        DEFAULT_TOOL_SHORT_TEXT_MAX_LENGTH,
        min(DEFAULT_TOOL_LONG_TEXT_MAX_LENGTH, request_text_len + TOOL_LONG_REQUEST_MARGIN),
    )


def _request_text_len(request: Any) -> int | None:
    messages = _get_attr_or_item(request, "messages")
    if not isinstance(messages, list):
        return None
    total = 0
    for message in messages:
        content = _get_attr_or_item(message, "content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    total += len(part)
                else:
                    text = _get_attr_or_item(part, "text")
                    if isinstance(text, str):
                        total += len(text)
    return total


def _bound_tool_schema(
    schema: Any,
    *,
    field_name: str | None = None,
    request_text_len: int | None = None,
) -> Any:
    if isinstance(schema, list):
        return [
            _bound_tool_schema(
                item,
                field_name=field_name,
                request_text_len=request_text_len,
            )
            for item in schema
        ]
    if not isinstance(schema, dict):
        return schema

    bounded = _copy_jsonable(schema)
    if not isinstance(bounded, dict):
        return schema

    schema_type = bounded.get("type")
    schema_types = schema_type if isinstance(schema_type, list) else [schema_type]
    if (
        "string" in schema_types
        and "maxLength" not in bounded
        and "enum" not in bounded
        and "const" not in bounded
    ):
        max_length = TOOL_FIELD_STRING_BUDGETS.get(
            field_name or "", DEFAULT_TOOL_SCHEMA_MAX_STRING_LENGTH
        )
        if field_name in TOOL_LONG_TEXT_FIELD_NAMES:
            max_length = _tool_text_budget(request_text_len)
        bounded["maxLength"] = max_length
    if "array" in schema_types and "maxItems" not in bounded:
        bounded["maxItems"] = DEFAULT_TOOL_SCHEMA_MAX_ARRAY_ITEMS

    for key in ("properties", "$defs", "definitions", "patternProperties"):
        value = bounded.get(key)
        if isinstance(value, dict):
            bounded[key] = {
                name: _bound_tool_schema(
                    subschema,
                    field_name=name if key == "properties" else None,
                    request_text_len=request_text_len,
                )
                for name, subschema in value.items()
            }

    for key in ("items", "additionalProperties"):
        value = bounded.get(key)
        if isinstance(value, (dict, list)):
            bounded[key] = _bound_tool_schema(
                value,
                field_name=field_name,
                request_text_len=request_text_len,
            )

    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        value = bounded.get(key)
        if isinstance(value, list):
            bounded[key] = [
                _bound_tool_schema(
                    subschema,
                    field_name=field_name,
                    request_text_len=request_text_len,
                )
                for subschema in value
            ]

    return bounded


def _tool_function(tool: Any) -> Any:
    return _get_attr_or_item(tool, "function") or {}


def _tool_name(tool: Any) -> str | None:
    return _get_attr_or_item(_tool_function(tool), "name")


def _tool_parameters(tool: Any) -> dict[str, Any]:
    params = _get_attr_or_item(_tool_function(tool), "parameters")
    if isinstance(params, dict):
        return _copy_jsonable(params)
    return {"type": "object", "properties": {}}


def _named_tool_choice_name(tool_choice: Any) -> str | None:
    function = _get_attr_or_item(tool_choice, "function")
    if function is None:
        return None
    return _get_attr_or_item(function, "name")


def _request_has_user_structured_output(request: Any) -> bool:
    if _get_attr_or_item(request, "guided_json") is not None:
        return True
    structured_outputs = _get_attr_or_item(request, "structured_outputs")
    if isinstance(structured_outputs, dict) and any(
        structured_outputs.get(key) is not None
        for key in ("json", "json_object", "regex", "choice", "grammar", "structural_tag")
    ):
        return True
    elif structured_outputs is not None and hasattr(
        structured_outputs, "all_constraints_none"
    ):
        if not structured_outputs.all_constraints_none():
            return True
    response_format = _get_attr_or_item(request, "response_format")
    if isinstance(response_format, dict):
        return response_format.get("type") in {
            "json_schema",
            "json_object",
            "structural_tag",
        }
    return False


def _complete_json_text(text: str) -> str | None:
    candidate = text.lstrip()
    if not candidate:
        return None
    try:
        _, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return None
    if candidate[end:].strip():
        return None
    return candidate[:end]


def _structured_tool_choice_name(request: Any) -> str | None:
    if _request_has_user_structured_output(request) or not _get_attr_or_item(
        request, "tools"
    ):
        return None
    tool_choice = _get_attr_or_item(request, "tool_choice")
    if tool_choice in (None, "none", "auto", "required"):
        return None
    return _named_tool_choice_name(tool_choice)


def _structured_tool_choice_required(request: Any) -> bool:
    return (
        bool(_get_attr_or_item(request, "tools"))
        and not _request_has_user_structured_output(request)
        and _get_attr_or_item(request, "tool_choice") == "required"
    )


def _structured_tool_calls_from_content(
    request: Any,
    content: str,
) -> list[dict[str, Any]] | None:
    _, end_marker, post_reasoning = content.rpartition("</think>")
    arguments = _complete_json_text(post_reasoning if end_marker else content)
    if arguments is None:
        return None

    tool_name = _structured_tool_choice_name(request)
    if tool_name is not None:
        return [
            {
                "id": make_tool_call_id(),
                "type": "function",
                "index": 0,
                "function": {"name": tool_name, "arguments": arguments},
            }
        ]

    if not _structured_tool_choice_required(request):
        return None

    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    items = decoded if isinstance(decoded, list) else [decoded]
    tool_calls: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        name = _get_attr_or_item(item, "name")
        parameters = _get_attr_or_item(item, "parameters")
        if not isinstance(name, str):
            continue
        if not isinstance(parameters, str):
            parameters = json.dumps(parameters or {}, ensure_ascii=False)
        tool_calls.append(
            {
                "id": make_tool_call_id(),
                "type": "function",
                "index": index,
                "function": {"name": name, "arguments": parameters},
            }
        )
    return tool_calls or None


def _bridge_structured_tool_content_choices(
    request: Any,
    choices: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if (
        _structured_tool_choice_name(request) is None
        and not _structured_tool_choice_required(request)
    ):
        return choices

    bridged: list[dict[str, Any]] = []
    for choice in choices:
        delta = choice.get("delta") or {}
        if delta.get("tool_calls"):
            bridged.append(choice)
            continue

        content = delta.get("content")
        if isinstance(content, str):
            tool_calls = _structured_tool_calls_from_content(request, content)
            if tool_calls:
                new_delta = {k: v for k, v in delta.items() if k != "content"}
                new_delta.setdefault("role", "assistant")
                new_delta["tool_calls"] = tool_calls
                bridged.append(
                    {
                        **choice,
                        "delta": new_delta,
                        "finish_reason": "tool_calls",
                    }
                )
                continue

        if choice.get("finish_reason") in {
            "stop",
            "length",
            FinishReason.STOP,
            FinishReason.LENGTH,
        } and set(delta.keys()) <= {"role"}:
            bridged.append({**choice, "finish_reason": None})
            continue
        bridged.append(choice)
    return bridged


def _tool_choice_guided_json_schema(request: Any) -> dict[str, Any] | None:
    tool_choice = _get_attr_or_item(request, "tool_choice")
    tools = _get_attr_or_item(request, "tools")
    if tool_choice in (None, "none", "auto") or not tools:
        return None

    request_text_len = _request_text_len(request)
    if tool_choice == "required":
        any_of = []
        for tool in tools:
            name = _tool_name(tool)
            if not name:
                continue
            any_of.append(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": [name]},
                        "parameters": _tool_parameters(tool),
                    },
                    "required": ["name", "parameters"],
                    "additionalProperties": False,
                }
            )
        if not any_of:
            return None
        schema = _bound_tool_schema(
            {
                "type": "array",
                "minItems": 1,
                "items": {"type": "object", "anyOf": any_of},
            },
            request_text_len=request_text_len,
        )
        schema[TOOL_CHOICE_SCHEMA_MARKER] = True
        return schema

    tool_name = _named_tool_choice_name(tool_choice)
    if not tool_name:
        return None
    for tool in tools:
        if _tool_name(tool) == tool_name:
            schema = _bound_tool_schema(
                _tool_parameters(tool),
                request_text_len=request_text_len,
            )
            schema[TOOL_CHOICE_SCHEMA_MARKER] = True
            return schema
    return None


def _forced_tool_choice_uses_qwen_xml_parser(
    request: Any,
    tool_parser_name: str | None,
) -> bool:
    tool_choice = _get_attr_or_item(request, "tool_choice")
    tools = _get_attr_or_item(request, "tools")
    if not tools or tool_choice in (None, "none", "auto"):
        return False
    return (tool_parser_name or "").strip() in QWEN_XML_STRUCTURAL_TAG_TOOL_PARSERS


def _has_structured_outputs(structured_outputs: StructuredOutputsParams | None) -> bool:
    return structured_outputs is not None and not structured_outputs.all_constraints_none()


def _parse_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected integer", name, value)
        return None


def _thinking_enabled(chat_template_kwargs: dict[str, Any]) -> bool:
    if chat_template_kwargs.get("enable_thinking") is False:
        return False
    if chat_template_kwargs.get("thinking") is False:
        return False
    return (
        chat_template_kwargs.get("enable_thinking") is True
        or chat_template_kwargs.get("thinking") is True
    )


def _tools_enabled(request_for_sampling: Any) -> bool:
    tools = _get_attr_or_item(request_for_sampling, "tools")
    tool_choice = _get_attr_or_item(request_for_sampling, "tool_choice")
    return bool(tools) and tool_choice != "none"


def _default_constrained_max_thinking_tokens(
    *,
    request_for_sampling: Any,
    sampling_params: SamplingParams,
    chat_template_kwargs: dict[str, Any],
) -> int | None:
    env_value = _parse_int_env("DYN_DEFAULT_CONSTRAINED_MAX_THINKING_TOKENS")
    if env_value is None:
        return None
    thinking_enabled = _thinking_enabled(chat_template_kwargs)
    tools_enabled = _tools_enabled(request_for_sampling)
    structured_enabled = _has_structured_outputs(
        _request_structured_outputs(request_for_sampling, sampling_params)
    )
    if not thinking_enabled:
        return None
    if not (tools_enabled or structured_enabled):
        return None
    return env_value


def _structured_outputs_from_response_format(
    response_format: Any,
) -> StructuredOutputsParams | None:
    response_type = _get_attr_or_item(response_format, "type")
    if response_type == "json_object":
        return StructuredOutputsParams(json_object=True)
    if response_type == "json_schema":
        json_schema = _get_attr_or_item(response_format, "json_schema")
        schema = _get_attr_or_item(json_schema, "json_schema")
        if schema is None:
            schema = _get_attr_or_item(json_schema, "schema")
        if schema is None:
            return None
        return StructuredOutputsParams(json=schema)
    if response_type == "structural_tag":
        if hasattr(response_format, "model_dump"):
            structural_tag = response_format.model_dump(by_alias=True)
        else:
            structural_tag = response_format
        return StructuredOutputsParams(structural_tag=json.dumps(structural_tag))
    return None


def _structured_outputs_from_request_field(
    request_for_sampling: Any,
) -> StructuredOutputsParams | None:
    structured_outputs = getattr(request_for_sampling, "structured_outputs", None)
    if structured_outputs is None:
        model_extra = getattr(request_for_sampling, "model_extra", None)
        if isinstance(model_extra, dict):
            structured_outputs = model_extra.get("structured_outputs")

    if structured_outputs is None:
        return None
    if isinstance(structured_outputs, StructuredOutputsParams):
        return structured_outputs
    if not isinstance(structured_outputs, dict):
        if hasattr(structured_outputs, "all_constraints_none"):
            return structured_outputs
        return None

    params: dict[str, Any] = {}
    for key in (
        "json",
        "regex",
        "choice",
        "grammar",
        "json_object",
        "structural_tag",
        "disable_any_whitespace",
        "disable_additional_properties",
        "whitespace_pattern",
    ):
        value = structured_outputs.get(key)
        if value is not None:
            params[key] = value
    if not params:
        return None
    return StructuredOutputsParams(**params)


def _request_structured_outputs(
    request_for_sampling: Any,
    sampling_params: SamplingParams,
) -> StructuredOutputsParams | None:
    if sampling_params.structured_outputs is not None:
        return sampling_params.structured_outputs
    structured_outputs = _structured_outputs_from_request_field(request_for_sampling)
    if structured_outputs is not None:
        return structured_outputs
    return _structured_outputs_from_response_format(
        getattr(request_for_sampling, "response_format", None)
    )


def _copy_reasoning_metadata_to_extra_args(
    dynamo_preproc: dict[str, Any], kv_kwargs: dict[str, Any]
) -> None:
    reasoning_extra_args: dict[str, Any] = {}
    for key in (
        "reasoning_ended",
        "reasoning_parser_kwargs",
        "reasoning_budget",
        "reasoning_budget_grace_period",
    ):
        if key in dynamo_preproc:
            reasoning_extra_args[key] = dynamo_preproc[key]

    if not reasoning_extra_args:
        return

    extra_args = kv_kwargs.get("extra_args")
    if not isinstance(extra_args, dict):
        extra_args = {}
        kv_kwargs["extra_args"] = extra_args
    extra_args.update(reasoning_extra_args)


class VllmProcessor:
    def __init__(
        self,
        tokenizer: TokenizerLike,
        input_processor: InputProcessor,
        router: Any,  # Client or KvRouter
        output_processor: OutputProcessor,
        tool_parser_class: type[ToolParser] | None,
        reasoning_parser_class: type[ReasoningParser] | None,
        tool_parser_name: str | None = None,
        block_size: int = 16,
        enable_auto_tool_choice: bool = False,
    ):
        self.tokenizer = tokenizer
        self.input_processor = input_processor
        self.router = router
        self.is_kv_router = isinstance(router, KvRouter)
        self.output_processor = output_processor
        self.tool_parser_class = tool_parser_class
        self.tool_parser_name = tool_parser_name
        self.reasoning_parser_class = reasoning_parser_class
        self.exclude_tools_when_tool_choice_none = True
        self.block_size = block_size
        self.enable_auto_tool_choice = enable_auto_tool_choice
        # Sender for mm_kwargs transfer — instantiated lazily on first MM request.
        # MmKwargsShmSender for same-node transfers (default), MmKwargsNixlSender
        # for cross-node RDMA. Controlled by DYNAMO_MM_TRANSFER env var.
        self._sender: MmKwargsSender | None = None
        # Set DYNAMO_DISABLE_NIXL_MM=1 to disable mm_kwargs transfer entirely.
        # Set DYNAMO_MM_TRANSFER to choose transfer mode:
        #   shm (default): shared memory. Same-node only (~2ms). If the
        #     backend can't read the segment (cross-node), it falls back to
        #     normal processing (backend runs HF processor).
        #   nixl: NIXL RDMA. Works cross-node via IB.
        self.nixl_mm_enabled = os.environ.get("DYNAMO_DISABLE_NIXL_MM", "") != "1"
        transfer_mode = os.environ.get("DYNAMO_MM_TRANSFER", "shm").lower()
        self.use_shm_transfer = transfer_mode == "shm"
        logger.info("[mm-routing] Transfer mode: %s", transfer_mode)

    def _get_eos_token_ids(self) -> list[int]:
        """Return EOS token ids using tokenizer metadata.

        vLLM 0.17.0 removed EngineCoreRequest.eos_token_id, so Dynamo can no
        longer read EOS ids from the preprocessed request object.
        """
        eos_token_ids = getattr(self.tokenizer, "eos_token_ids", None)
        if eos_token_ids is not None and not isinstance(eos_token_ids, int):
            return list(eos_token_ids)

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            return []
        return [eos_token_id]

    async def _prepare_mm_routing(
        self,
        vllm_preproc: EngineCoreRequest,
        dynamo_preproc: dict[str, Any],
    ) -> tuple[dict | None, list, bool]:
        """Extract MM routing info and prepare mm_kwargs transfer.

        Returns:
            (mm_routing_info, cleanup_items, transferred)
            cleanup_items: passed to self._sender.cleanup() after streaming.
            transferred: True when all features with data were sent successfully.
        """
        mm_routing_info = None
        cleanup_items: list = []
        nixl_transferred = False

        rng_routing = _nvtx.start_range("mm_frontend:build_routing_info", color="cyan")
        if self.is_kv_router and vllm_preproc.mm_features:
            mm_routing_info = build_mm_routing_info_from_features(
                vllm_preproc.mm_features,
                prompt_token_ids=list(vllm_preproc.prompt_token_ids),
                block_size=self.block_size,
            )
            # Forward mm_hashes to backend for hash consistency — the backend
            # will use these directly instead of recomputing.
            mm_hashes_list = [f.mm_hash for f in vllm_preproc.mm_features]
            mm_placeholders_list = [
                (f.mm_position.offset, f.mm_position.length)
                for f in vllm_preproc.mm_features
            ]
            # Transport mm_hashes and mm_placeholders to backend via extra_args.
            if "extra_args" not in dynamo_preproc:
                dynamo_preproc["extra_args"] = {}
            dynamo_preproc["extra_args"]["mm_hashes"] = mm_hashes_list
            dynamo_preproc["extra_args"]["mm_placeholders"] = mm_placeholders_list
            # Forward the expanded prompt_token_ids (with image placeholders)
            # so the backend can use them in the pre-rendered MultiModalInput.
            dynamo_preproc["extra_args"]["expanded_token_ids"] = list(
                vllm_preproc.prompt_token_ids
            )

            n_blocks = len(mm_routing_info["block_mm_infos"]) if mm_routing_info else 0
            n_mm_blocks = sum(
                1 for b in (mm_routing_info or {}).get("block_mm_infos", []) if b
            )
            logger.debug(
                "[mm-routing] Built mm_routing_info: %d mm_features, "
                "%d hashes, %d total blocks, %d blocks with MM content, "
                "block_size=%d",
                len(vllm_preproc.mm_features),
                len(mm_hashes_list),
                n_blocks,
                n_mm_blocks,
                self.block_size,
            )
            if logger.isEnabledFor(logging.DEBUG):
                for i, f in enumerate(vllm_preproc.mm_features):
                    logger.debug(
                        "[mm-routing]   feature[%d]: modality=%s, hash=%s..., "
                        "offset=%d, length=%d",
                        i,
                        f.modality,
                        f.mm_hash[:16] if f.mm_hash else "None",
                        f.mm_position.offset,
                        f.mm_position.length,
                    )

            # Transfer pre-processed mm_kwargs to the backend so it can skip
            # the HF processor.  Strategy:
            #   - shm (default): shared memory, same-node only (~2ms).
            #     Cross-node backends fail gracefully and fall back to
            #     normal processing.
            #   - nixl: NIXL RDMA (works cross-node via IB).
            if not self.nixl_mm_enabled:
                logger.debug(
                    "[mm-routing] mm_kwargs transfer disabled via DYNAMO_DISABLE_NIXL_MM"
                )
            else:
                try:
                    if self._sender is None:
                        self._sender = (
                            MmKwargsShmSender()
                            if self.use_shm_transfer
                            else MmKwargsNixlSender()
                        )
                    # NVTX annotation is owned by MmKwargsSender.prepare via
                    # the subclass's _nvtx_label/_nvtx_color class attrs.
                    extra_update, cleanup_items = await self._sender.prepare(
                        vllm_preproc.mm_features, modality="image"
                    )
                    if extra_update is not None:
                        dynamo_preproc["extra_args"].update(extra_update)
                        nixl_transferred = True
                    else:
                        logger.debug(
                            "[mm-routing] sender returned None — no tensors to transfer"
                        )
                except Exception:
                    logger.warning(
                        "[mm-routing] sender failed, backend will run HF processor",
                        exc_info=True,
                    )
                    cleanup_items = []

        elif self.is_kv_router:
            logger.debug("[mm-routing] No mm_features — text-only request")
        _nvtx.end_range(rng_routing)

        return mm_routing_info, cleanup_items, nixl_transferred

    # Ideally we would map NVCreateChatCompletionRequest into Python so it can be type checked, but
    # it has a lot of fields.
    # request: dynamo.NVCreateChatCompletionRequest
    async def generator(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run a single request through the engine. Does pre and post processing on this machine, delegates
        model inference to a backend using the router.
        """
        with _nvtx.annotate("mm_frontend:generator", color="blue"):
            async for item in self._generator_inner(request):
                yield item

    async def _generator_inner(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        request_id = random_uuid()

        # vLLM's Pydantic model requires image_url.detail to be 'auto'/'low'/'high'.
        # The Rust HTTP layer accepts None/missing, so normalize before validation.
        messages = request.get("messages") or []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    img_url = part.get("image_url")
                    if isinstance(img_url, dict) and img_url.get("detail") is None:
                        img_url["detail"] = "auto"

        # Images are fetched by vLLM's renderer via DynamoMediaConnector,
        # which wraps our ImageLoader (LRU cache + in-flight dedup).
        # No data URI encoding needed.
        with _nvtx.annotate("mm_frontend:preprocess_chat", color="yellow"):
            pre = await preprocess_chat_request(
                request,
                tokenizer=self.tokenizer,
                renderer=self.input_processor.renderer,
                tool_parser_class=self.tool_parser_class,
                exclude_tools_when_tool_choice_none=self.exclude_tools_when_tool_choice_none,
                enable_auto_tool_choice=self.enable_auto_tool_choice,
            )

        request_for_sampling = pre.request_for_sampling
        tool_parser = pre.tool_parser
        chat_template_kwargs = pre.chat_template_kwargs
        raw_chat_template_kwargs = request.get("chat_template_kwargs")
        if not isinstance(raw_chat_template_kwargs, dict):
            raw_chat_template_kwargs = request.get("chat_template_args")
        if isinstance(raw_chat_template_kwargs, dict):
            effective_chat_template_kwargs = {
                **raw_chat_template_kwargs,
                **chat_template_kwargs,
            }
        else:
            effective_chat_template_kwargs = chat_template_kwargs
        engine_prompt = pre.engine_prompt
        tokens = pre.prompt_token_ids

        if request_for_sampling.max_completion_tokens is not None:
            max_tokens = request_for_sampling.max_completion_tokens
        elif request_for_sampling.max_tokens is not None:
            max_tokens = request_for_sampling.max_tokens
        else:
            # This should mean model max - prompt len.
            max_tokens = None

        sampling_params = SamplingParams(
            output_kind=RequestOutputKind.DELTA,
            max_tokens=max_tokens,
        )
        # generation_config.json
        # Skip eos_token_id: vLLM 0.17.0 made SamplingParams.eos_token_id a
        # read-only property; eos tokens are handled via eos_token_ids below.
        for k, v in self.input_processor.generation_config_fields.items():
            if k == "eos_token_id":
                continue
            if hasattr(sampling_params, k):
                setattr(sampling_params, k, v)

        # User request: copy fields supported by both request schema and
        # SamplingParams, excluding fields handled separately below.
        sampling_fields = (
            set(getattr(SamplingParams, "__annotations__", ()))
            & set(type(request_for_sampling).model_fields)
        ) - {"max_tokens", "logprobs", "output_kind"}
        for k in sorted(sampling_fields):
            v = getattr(request_for_sampling, k, None)
            if v is not None:
                setattr(sampling_params, k, v)

        reasoning_ended, reasoning_parser_kwargs = _build_reasoning_parser_metadata(
            self.reasoning_parser_class,
            self.tokenizer,
            chat_template_kwargs,
            request_for_sampling,
            tokens,
        )
        skip_forced_tool_guidance_for_reasoning = (
            reasoning_ended is False
            and _forced_tool_choice_uses_qwen_xml_parser(
                request_for_sampling, self.tool_parser_name
            )
        )
        if not _has_structured_outputs(sampling_params.structured_outputs):
            tool_schema = (
                None
                if skip_forced_tool_guidance_for_reasoning
                else _tool_choice_guided_json_schema(request_for_sampling)
            )
            if tool_schema is not None:
                sampling_params.structured_outputs = StructuredOutputsParams(
                    json=tool_schema
                )
        # nvext.max_thinking_tokens is enforced on the worker, not here. The
        # frontend's InputProcessor is built without reasoning_config (it only
        # tokenizes), so setting sampling_params.thinking_token_budget would
        # cause process_inputs._validate_params to reject the request. Pluck
        # the value out of nvext and pass it directly into dynamo_preproc
        # below.
        nvext_max_thinking_tokens = (request.get("nvext") or {}).get(
            "max_thinking_tokens"
        )
        if nvext_max_thinking_tokens is None:
            nvext_max_thinking_tokens = _default_constrained_max_thinking_tokens(
                request_for_sampling=request_for_sampling,
                sampling_params=sampling_params,
                chat_template_kwargs=effective_chat_template_kwargs,
            )
        logprobs = request_for_sampling.logprobs
        top_logprobs = request_for_sampling.top_logprobs
        if logprobs is True:
            sampling_params.logprobs = top_logprobs or 1
        elif isinstance(logprobs, int) and not isinstance(logprobs, bool):
            sampling_params.logprobs = logprobs
        elif top_logprobs not in (None, 0):
            sampling_params.logprobs = top_logprobs
        if sampling_params.logprobs is not None and sampling_params.logprobs > 0:
            logger.warning(
                "Logprobs requested but not supported in distributed inference mode"
            )

        guided_decoding = _structured_outputs_to_guided_decoding(
            _request_structured_outputs(request_for_sampling, sampling_params)
        )

        # The renderer's process_for_engine() always returns a fully processed
        # EngineInput (TokenInputs or MultiModalInputs) with a "type" key.
        # Pass it directly to process_inputs() — no need to rebuild a
        # TokensPrompt, and this avoids the deprecation warning.
        prompt_inputs = engine_prompt
        if request_for_sampling.cache_salt is not None:
            prompt_inputs["cache_salt"] = request_for_sampling.cache_salt
        if request_for_sampling.mm_processor_kwargs is not None:
            prompt_inputs[
                "mm_processor_kwargs"
            ] = request_for_sampling.mm_processor_kwargs

        with _nvtx.annotate("mm_frontend:process_inputs", color="orange"):
            vllm_preproc: EngineCoreRequest = self.input_processor.process_inputs(
                request_id,
                prompt_inputs,
                sampling_params,
                GENERATION_TASKS,  # vLLM 0.17.0: required supported_tasks arg
            )

        InputProcessor.assign_request_id(vllm_preproc)

        # vLLM 0.17.0 removed EngineCoreRequest.eos_token_id. Dynamo now uses
        # tokenizer metadata for EOS ids when constructing the router payload.

        # Convert to a Python object that has fields that match our PreprocessedRequest
        sp = vllm_preproc.sampling_params
        dynamo_preproc = {
            "model": request["model"],
            "token_ids": tokens,
            "stop_conditions": {
                "max_tokens": sp.max_tokens,
                "stop": sp.stop,
                "stop_token_ids": sp.stop_token_ids,
                "min_tokens": sp.min_tokens,
                "ignore_eos": sp.ignore_eos,
                "max_thinking_tokens": nvext_max_thinking_tokens,
            },
            "sampling_options": {
                "n": sp.n,
                "presence_penalty": sp.presence_penalty,
                "frequency_penalty": sp.frequency_penalty,
                "repetition_penalty": sp.repetition_penalty,
                "temperature": sp.temperature,
                "top_p": sp.top_p,
                "top_k": sp.top_k,
                "min_p": sp.min_p,
                "seed": sp.seed,
            },
            "output_options": {
                "logprobs": sp.logprobs,
                "prompt_logprobs": sp.prompt_logprobs,
                "skip_special_tokens": sp.skip_special_tokens,
            },
            "eos_token_ids": self._get_eos_token_ids(),
            "annotations": [],
            "routing": request.get("routing"),
        }
        if guided_decoding is not None:
            dynamo_preproc["sampling_options"]["guided_decoding"] = guided_decoding
            tool_choice = _get_attr_or_item(request_for_sampling, "tool_choice")
            tools = _get_attr_or_item(request_for_sampling, "tools")
            if tools and tool_choice not in (None, "none", "auto"):
                dynamo_preproc["tools"] = [_copy_jsonable(tool) for tool in tools]
                dynamo_preproc["tool_choice"] = _copy_jsonable(tool_choice)
        if reasoning_ended is not None:
            dynamo_preproc["reasoning_ended"] = reasoning_ended
        if reasoning_parser_kwargs is not None:
            dynamo_preproc["reasoning_parser_kwargs"] = reasoning_parser_kwargs
        if request_for_sampling.reasoning_budget is not None:
            dynamo_preproc["reasoning_budget"] = request_for_sampling.reasoning_budget
        if request_for_sampling.reasoning_budget_grace_period is not None:
            dynamo_preproc["reasoning_budget_grace_period"] = (
                request_for_sampling.reasoning_budget_grace_period
            )

        # Extract MM routing metadata and prepare transfer.
        cleanup_items: list = []
        try:
            (
                mm_routing_info,
                cleanup_items,
                nixl_transferred,
            ) = await self._prepare_mm_routing(vllm_preproc, dynamo_preproc)

            # Forward multimodal URLs so the backend handler can load the media.
            # Only skip when ALL features were transferred — a partial transfer
            # (some features had data=None due to processor cache) still needs
            # URLs for the backend to process the missing features.
            n_features = (
                len(vllm_preproc.mm_features) if vllm_preproc.mm_features else 0
            )
            n_with_data = sum(
                1 for f in (vllm_preproc.mm_features or []) if f.data is not None
            )
            all_transferred = nixl_transferred and n_with_data == n_features
            if not all_transferred:
                mm_data = extract_mm_urls(request.get("messages") or [])
                if mm_data:
                    dynamo_preproc["multi_modal_data"] = mm_data

            # Forward mm_processor_kwargs (e.g. use_audio_in_video) to the backend.
            if request_for_sampling.mm_processor_kwargs is not None:
                dynamo_preproc[
                    "mm_processor_kwargs"
                ] = request_for_sampling.mm_processor_kwargs

            def new_post_processor() -> StreamingPostProcessor:
                return StreamingPostProcessor(
                    tokenizer=self.tokenizer,
                    request_for_sampling=request_for_sampling,
                    sampling_params=sp,
                    prompt_token_ids=tokens,
                    tool_parser=tool_parser,
                    reasoning_parser_class=self.reasoning_parser_class,
                    chat_template_kwargs=chat_template_kwargs,
                )

            # StreamingPostProcessor keeps delta/tool/reasoning parser state, so
            # parallel choices must not share one instance. Keep one state machine
            # per choice index while the backend interleaves n>1 token chunks.
            post_processors = {
                output_idx: new_post_processor() for output_idx in range(sp.n)
            }

            async for item in self._generate_and_stream(
                request_id,
                request,
                dynamo_preproc,
                tokens,
                vllm_preproc,
                post_processors,
                request_for_sampling=request_for_sampling,
                mm_routing_info=mm_routing_info,
            ):
                yield item
        finally:
            if cleanup_items and self._sender is not None:
                await self._sender.cleanup(cleanup_items)

    async def _generate_and_stream(
        self,
        request_id: str,
        request: dict[str, Any],
        dynamo_preproc: dict[str, Any],
        tokens: list[int],
        vllm_preproc: EngineCoreRequest,
        post_processors: dict[int, StreamingPostProcessor],
        request_for_sampling: Any,
        mm_routing_info: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        sp = vllm_preproc.sampling_params
        output_request_ids: dict[int, str]
        registered_request_ids: list[str]

        if sp.n == 1:
            self.output_processor.add_request(vllm_preproc, None)
            output_request_ids = {0: vllm_preproc.request_id}
            registered_request_ids = [vllm_preproc.request_id]
        else:
            # vLLM's normal engine path fans out SamplingParams.n>1 into
            # ParentRequest children before registering with OutputProcessor.
            # Dynamo bypasses that path here: the backend generates indexed
            # token chunks and this frontend feeds those chunks directly into
            # vLLM's OutputProcessor. Recreate the same parent/child request
            # state so each choice has its own request id, sampling params,
            # detokenizer/logprob state, and OpenAI choice index.
            #
            # See vLLM's implementation:
            # https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/engine/async_llm.py
            # https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/engine/output_processor.py
            # https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/engine/parallel_sampling.py
            parent_preproc = vllm_preproc
            if parent_preproc.external_req_id is None:
                parent_preproc = msgspec_replace(
                    parent_preproc, external_req_id=parent_preproc.request_id
                )
            parent_req = ParentRequest(parent_preproc)
            output_request_ids = {}
            registered_request_ids = []
            for output_idx in range(sp.n):
                child_request_id, child_sampling_params = parent_req.get_child_info(
                    output_idx
                )
                child_preproc = msgspec_replace(
                    parent_preproc,
                    request_id=child_request_id,
                    sampling_params=child_sampling_params,
                )
                self.output_processor.add_request(
                    child_preproc,
                    None,
                    parent_req=parent_req,
                    request_index=output_idx,
                )
                output_request_ids[output_idx] = child_request_id
                registered_request_ids.append(child_request_id)

        try:
            rng_route = _nvtx.start_range("mm_frontend:kv_router_generate", color="red")
            if self.is_kv_router:
                kv_kwargs: dict[str, Any] = {
                    "token_ids": tokens,
                    "model": dynamo_preproc["model"],
                    "stop_conditions": dynamo_preproc["stop_conditions"],
                    "sampling_options": dynamo_preproc["sampling_options"],
                    "output_options": dynamo_preproc["output_options"],
                    "multi_modal_data": dynamo_preproc.get("multi_modal_data"),
                }
                if dynamo_preproc.get("extra_args"):
                    kv_kwargs["extra_args"] = dynamo_preproc["extra_args"]
                    ea = dynamo_preproc["extra_args"]
                    logger.debug(
                        "[mm-routing] extra_args keys=%s, has_nixl=%s, "
                        "n_hashes=%d, n_placeholders=%d",
                        list(ea.keys()),
                        "mm_kwargs_nixl" in ea,
                        len(ea.get("mm_hashes", [])),
                        len(ea.get("mm_placeholders", [])),
                    )
                _copy_reasoning_metadata_to_extra_args(dynamo_preproc, kv_kwargs)
                # Forward mm_processor_kwargs (e.g. use_audio_in_video) to backend.
                mm_proc_kwargs = dynamo_preproc.get("mm_processor_kwargs")
                if mm_proc_kwargs is not None:
                    if "extra_args" not in kv_kwargs or kv_kwargs["extra_args"] is None:
                        kv_kwargs["extra_args"] = {}
                    kv_kwargs["extra_args"]["mm_processor_kwargs"] = mm_proc_kwargs
                if mm_routing_info is not None:
                    kv_kwargs["mm_routing_info"] = mm_routing_info
                    logger.debug(
                        "[mm-routing] KvRouter.generate() called with "
                        "mm_routing_info (%d routing tokens, %d blocks)",
                        len(mm_routing_info.get("routing_token_ids", [])),
                        len(mm_routing_info.get("block_mm_infos", [])),
                    )
                else:
                    logger.debug(
                        "[mm-routing] KvRouter.generate() called without "
                        "mm_routing_info (text-only)"
                    )
                dynamo_stream = await self.router.generate(**kv_kwargs)
            else:
                dynamo_stream = await self.router.generate(
                    dynamo_preproc, annotated=False
                )
            _nvtx.end_range(rng_route)

            rng_stream = _nvtx.start_range(
                "mm_frontend:stream_response", color="purple"
            )
            async for dynamo_response in dynamo_stream:
                if self.is_kv_router:
                    engine_response = dynamo_response
                elif hasattr(dynamo_response, "data"):
                    engine_response = dynamo_response.data()
                else:
                    engine_response = dynamo_response

                if engine_response is None or "token_ids" not in engine_response:
                    yield handle_engine_error(engine_response, request_id, logger)
                    break

                output_idx = engine_response.get("index", 0) or 0
                output_request_id = output_request_ids.get(output_idx)
                if output_request_id is None:
                    yield {
                        "error": {
                            "message": (
                                f"Invalid engine choice index {output_idx} "
                                f"for request {request_id}"
                            ),
                            "type": "internal_error",
                        }
                    }
                    break

                raw_finish_reason = engine_response.get("finish_reason")
                finish_reason = map_finish_reason(raw_finish_reason)
                stop_reason = engine_response.get("stop_reason")
                raw_token_ids = list(engine_response["token_ids"])
                engine_text = engine_response.get("text") or ""
                if os.environ.get("DYN_DEBUG_REASONING_BUDGET") == "1" and (
                    raw_finish_reason or 14 in raw_token_ids or 15 in raw_token_ids
                ):
                    logger.warning(
                        "frontend raw engine response request_id=%s index=%s "
                        "finish_reason=%s stop_reason=%s contains_14=%s "
                        "contains_15=%s raw_tail=%s",
                        request_id,
                        output_idx,
                        raw_finish_reason,
                        stop_reason,
                        14 in raw_token_ids,
                        15 in raw_token_ids,
                        raw_token_ids[-32:],
                    )

                output_kwargs: dict[str, Any] = {
                    "request_id": output_request_id,
                    "new_token_ids": raw_token_ids,
                    "finish_reason": finish_reason,
                    "stop_reason": stop_reason,
                }
                output_fields = getattr(EngineCoreOutput, "__struct_fields__", ())
                if "is_segment_finished" in output_fields:
                    output_kwargs["is_segment_finished"] = engine_response.get(
                        "is_segment_finished", False
                    )
                if "new_prompt_len_snapshot" in output_fields:
                    output_kwargs["new_prompt_len_snapshot"] = engine_response.get(
                        "new_prompt_len_snapshot"
                    )
                vllm_response = EngineCoreOutput(**output_kwargs)

                vllm_out: OutputProcessorOutput = self.output_processor.process_outputs(
                    [vllm_response]
                )

                if vllm_out.reqs_to_abort:
                    pass

                choices = []
                if not vllm_out.request_outputs:
                    post = post_processors.get(output_idx)
                    if post is not None and (
                        engine_text or post.needs_raw_parser_delta(raw_token_ids)
                    ):
                        choice = post.process_output(
                            SimpleNamespace(
                                index=output_idx,
                                token_ids=raw_token_ids,
                                text=engine_text,
                                finish_reason=raw_finish_reason,
                                logprobs=None,
                            ),
                            raw_delta_token_ids=raw_token_ids,
                        )
                        if choice:
                            choices.append(choice)
                    if not choices:
                        continue
                else:
                    for output in vllm_out.request_outputs[0].outputs:
                        post = post_processors.get(output.index)
                        if post is None:
                            yield {
                                "error": {
                                    "message": (
                                        f"Invalid postprocessor choice index {output.index} "
                                        f"for request {request_id}"
                                    ),
                                    "type": "internal_error",
                                }
                            }
                            break
                        if os.environ.get("DYN_DEBUG_REASONING_BUDGET") == "1" and (
                            raw_finish_reason
                            or 14 in raw_token_ids
                            or 15 in raw_token_ids
                        ):
                            processed_token_ids = list(output.token_ids or [])
                            logger.warning(
                                "frontend processed output request_id=%s index=%s "
                                "finish_reason=%s processed_contains_14=%s "
                                "processed_contains_15=%s processed_tail=%s text_tail=%s",
                                request_id,
                                output.index,
                                output.finish_reason,
                                14 in processed_token_ids,
                                15 in processed_token_ids,
                                processed_token_ids[-32:],
                                (output.text or "")[-240:],
                            )
                        parser_output = _with_parser_visible_engine_text(
                            output, engine_text
                        )
                        choice = post.process_output(
                            parser_output,
                            raw_delta_token_ids=raw_token_ids,
                        )
                        if choice:
                            choices.append(choice)

                if choices:
                    choices = _bridge_structured_tool_content_choices(
                        request_for_sampling,
                        choices,
                    )
                    for choice in choices:
                        post = post_processors.get(choice.get("index", 0))
                        if post is not None:
                            post._strip_tool_markup_from_delta(
                                choice.get("delta") or {}
                            )
                    dynamo_out = {
                        "id": request_id,
                        "choices": choices,
                        "created": int(time.time()),
                        "model": request["model"],
                        "object": "chat.completion.chunk",
                    }
                    if usage := engine_response.get("completion_usage"):
                        dynamo_out["usage"] = usage

                    yield dynamo_out
                    if sp.n == 1 and any(
                        post.structured_json_complete or post.structured_tool_complete
                        for post in post_processors.values()
                    ):
                        break
            _nvtx.end_range(rng_stream)
        except Exception as e:
            logger.exception("Error generating response for request %s", request_id)
            yield make_internal_error(request_id, str(e))
        finally:
            for output_request_id in registered_request_ids:
                if output_request_id in self.output_processor.request_states:
                    self.output_processor.abort_requests(
                        [output_request_id], internal=True
                    )


class EngineFactory:
    def __init__(
        self,
        runtime: DistributedRuntime,
        router_config: RouterConfig,
        config: FrontendConfig,
        flags: Namespace,
    ):
        self.runtime = runtime
        self.router_config = router_config
        self.config = config
        self.flags = flags
        self.stream_interval = 20
        raw_stream_interval = os.getenv("DYN_VLLM_STREAM_INTERVAL")
        if raw_stream_interval:
            try:
                self.stream_interval = max(1, int(raw_stream_interval))
            except ValueError:
                logger.warning(
                    "Invalid DYN_VLLM_STREAM_INTERVAL=%r, using default=%d",
                    raw_stream_interval,
                    self.stream_interval,
                )

    async def chat_engine_factory(
        self,
        instance_id: ModelCardInstanceId,
        mdc: ModelDeploymentCard,
    ) -> PythonAsyncEngine:
        """
        Called by Rust when a model is discovered.
        """
        model_type = mdc.model_type()
        if not model_type.supports_chat():
            raise RuntimeError(
                f"model type {model_type} is not supported by this factory"
            )
        if self.config.preprocess_workers != 0:
            raise RuntimeError(
                "preprocess_workers is not supported for vllm processor. "
                "Use the sglang processor for worker-pool preprocessing."
            )
        loop = asyncio.get_running_loop()

        # TODO(gh-8749): consume mdc.model_info.path()'s parent (slug_dir)
        # instead of re-running fetch_model. The MDC cache already has
        # blake3-verified copies; this path duplicates the download.
        source_path = mdc.source_path()
        if not os.path.exists(source_path):
            await fetch_model(source_path, ignore_weights=True)

        tokenizer_mode = getattr(self.flags, "tokenizer_mode", None) or "auto"
        config_format = getattr(self.flags, "config_format", None) or "auto"
        load_format = getattr(self.flags, "load_format", None) or "dummy"
        trust_remote_code = self.config.trust_remote_code
        enable_auto_tool_choice = getattr(self.flags, "enable_auto_tool_choice", False)

        model_config = ModelConfig(
            model=source_path,
            tokenizer_mode=tokenizer_mode,
            config_format=config_format,
            trust_remote_code=trust_remote_code,
        )
        # Use processor_only cache so tensor data persists across requests.
        # The default "lru" sender cache drops tensor data on cache hits
        # (designed for disagg where P1 holds tensors), but we need the
        # data to pickle and send via NIXL on repeated requests.
        if model_config.multimodal_config is not None:
            nixl_enabled = os.environ.get("DYNAMO_DISABLE_NIXL_MM", "") != "1"
            if nixl_enabled:
                model_config.multimodal_config.mm_processor_cache_type = (
                    "processor_only"
                )
        vllm_config = VllmConfig(
            model_config=model_config,
            load_config=LoadConfig(load_format=load_format),
            cache_config=CacheConfig(),
            # scheduler_config=SchedulerConfig(),
        )

        # Register dynamo's ImageLoader as vLLM's media connector so the
        # renderer uses our LRU cache + in-flight dedup for image fetching.
        # This eliminates data URI encoding overhead entirely.
        if os.environ.get("VLLM_MEDIA_CONNECTOR") != "dynamo":
            os.environ["VLLM_MEDIA_CONNECTOR"] = "dynamo"
        import dynamo.common.multimodal.media_connector  # noqa: F401

        input_processor = InputProcessor(vllm_config)
        tokenizer = input_processor.get_tokenizer()

        # Resolve stream_interval: env var override > backend config > default (20)
        stream_interval = self.stream_interval
        if not os.getenv("DYN_VLLM_STREAM_INTERVAL"):
            backend_interval = (
                mdc.runtime_config().get("runtime_data", {}).get("stream_interval")
            )
            if backend_interval is not None:
                try:
                    stream_interval = max(1, int(backend_interval))
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid stream_interval=%r from backend runtime_config, "
                        "using default=%d",
                        backend_interval,
                        stream_interval,
                    )

        output_processor = OutputProcessor(
            tokenizer,
            log_stats=False,
            stream_interval=stream_interval,
        )
        logger.info("vLLM OutputProcessor stream_interval=%d", stream_interval)

        tool_parser_name = self.flags.tool_call_parser or mdc.runtime_config().get(
            "tool_call_parser"
        )
        if tool_parser_name:
            tool_parser_class = ToolParserManager.get_tool_parser(tool_parser_name)
        else:
            tool_parser_class = None

        reasoning_parser_name = self.flags.reasoning_parser or mdc.runtime_config().get(
            "reasoning_parser"
        )
        if reasoning_parser_name:
            reasoning_parser_class = ReasoningParserManager.get_reasoning_parser(
                reasoning_parser_name
            )
        else:
            reasoning_parser_class = None

        namespace_name, component_name, endpoint_name = instance_id.triple()
        generate_endpoint = self.runtime.endpoint(
            f"{namespace_name}.{component_name}.{endpoint_name}"
        )
        router: Client | KvRouter
        if self.router_config.router_mode == RouterMode.KV:
            router = KvRouter(
                endpoint=generate_endpoint,
                block_size=self.config.kv_cache_block_size or 16,
                kv_router_config=self.router_config.kv_router_config,
            )
        else:
            router = await generate_endpoint.client(
                router_mode=self.router_config.router_mode
            )

        block_size = self.config.kv_cache_block_size or 16

        gen = VllmProcessor(
            tokenizer,
            input_processor,
            router,
            output_processor,
            tool_parser_class,
            reasoning_parser_class,
            tool_parser_name=tool_parser_name,
            block_size=block_size,
            enable_auto_tool_choice=enable_auto_tool_choice,
        )
        gen.exclude_tools_when_tool_choice_none = (
            self.config.exclude_tools_when_tool_choice_none
        )

        return PythonAsyncEngine(gen.generator, loop)
