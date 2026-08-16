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
from typing import Any

from msgspec.structs import replace as msgspec_replace
from vllm.config import CacheConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import RequestOutputKind, SamplingParams
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
from dynamo.llm import ModelCardInstanceId, PythonAsyncEngine, RoutedEngine

from .prepost import StreamingPostProcessor, preprocess_chat_request
from .thinking import runtime_default_thinking_mode
from .utils import (
    extract_mm_urls,
    handle_engine_error,
    random_uuid,
    resolve_chat_template,
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


def _with_parser_visible_engine_text(
    output: Any,
    engine_text: str | None,
    *,
    parser_needs_raw_delta: bool,
) -> Any:
    if (
        not parser_needs_raw_delta
        or not engine_text
        or getattr(output, "text", None)
    ):
        return output
    return SimpleNamespace(
        index=output.index,
        token_ids=output.token_ids,
        text=engine_text,
        finish_reason=output.finish_reason,
        logprobs=output.logprobs,
    )

def _normalize_router_finish_reason(raw_reason: Any) -> str | None:
    if raw_reason is None:
        return None
    if isinstance(raw_reason, dict):
        raw_reason = raw_reason.get("type")
    if not isinstance(raw_reason, str):
        logger.warning("Malformed finish_reason from router: %r", raw_reason)
        return "error"
    return raw_reason


def map_finish_reason(raw_reason: Any) -> FinishReason | None:
    raw_reason = _normalize_router_finish_reason(raw_reason)
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


def _runtime_config_context_length(mdc: ModelDeploymentCard) -> int | None:
    runtime_config = mdc.runtime_config()
    if not isinstance(runtime_config, dict):
        return None

    context_length = runtime_config.get("context_length")
    if type(context_length) is not int or context_length <= 0:
        return None
    return context_length


def _runtime_config_structural_tag_options(
    mdc: ModelDeploymentCard,
) -> tuple[str, str, str]:
    runtime_config = mdc.runtime_config()
    if not isinstance(runtime_config, dict):
        return "off", "auto", "auto"
    return (
        runtime_config.get("structural_tag_mode", "off"),
        runtime_config.get("structural_tag_scope", "auto"),
        runtime_config.get("structural_tag_schema", "auto"),
    )


def _ensure_chat_template(
    tokenizer: Any, local_dir: str, chat_template_flag: str | None
) -> None:
    """Set tokenizer.chat_template so vLLM's renderer handles tool calls.

    Skipped for MistralTokenizer (--tokenizer-mode mistral): it has no
    chat_template attribute and renders via mistral_common, so leave it
    untouched rather than attach an HF template it never uses.
    """
    if not hasattr(tokenizer, "chat_template"):
        return
    if tokenizer.chat_template is None:
        tokenizer.chat_template = resolve_chat_template(local_dir, backend="vllm")
    if chat_template_flag:
        tokenizer.chat_template = load_chat_template(chat_template_flag)


def _mm_feature_modality(feature: Any) -> str:
    return getattr(feature, "modality", None) or "image"


def _serialize_mm_placeholder(mm_position: Any) -> tuple[int, int] | dict[str, Any]:
    placeholder = (mm_position.offset, mm_position.length)
    is_embed = getattr(mm_position, "is_embed", None)
    if is_embed is None:
        return placeholder

    if hasattr(is_embed, "detach"):
        is_embed = is_embed.detach().cpu().tolist()

    return {
        "offset": mm_position.offset,
        "length": mm_position.length,
        "is_embed": [bool(value) for value in is_embed],
    }


def _group_mm_feature_metadata(
    mm_features: list[Any],
) -> tuple[
    list[str],
    list[tuple[int, int] | dict[str, Any]],
    dict[str, list[str]],
    dict[str, list[tuple[int, int] | dict[str, Any]]],
]:
    flat_hashes: list[str] = []
    flat_placeholders: list[tuple[int, int] | dict[str, Any]] = []
    hashes_by_modality: dict[str, list[str]] = {}
    placeholders_by_modality: dict[str, list[tuple[int, int] | dict[str, Any]]] = {}

    for feature in mm_features:
        mm_hash = getattr(feature, "mm_hash", None)
        if not mm_hash:
            continue
        modality = _mm_feature_modality(feature)
        placeholder = _serialize_mm_placeholder(feature.mm_position)
        hashes_by_modality.setdefault(modality, []).append(mm_hash)
        placeholders_by_modality.setdefault(modality, []).append(placeholder)

    # Legacy flat fields are image-only.
    if set(hashes_by_modality) == {"image"}:
        flat_hashes = hashes_by_modality["image"]
        flat_placeholders = placeholders_by_modality["image"]

    return flat_hashes, flat_placeholders, hashes_by_modality, placeholders_by_modality


def _single_transfer_modality(mm_features: list[Any]) -> str | None:
    modalities = {_mm_feature_modality(feature) for feature in mm_features}
    if len(modalities) != 1:
        return None
    return next(iter(modalities))


def _build_reasoning_parser_metadata(
    reasoning_parser_class: type[ReasoningParser] | None,
    tokenizer: TokenizerLike,
    chat_template_kwargs: dict[str, Any],
    request_for_sampling: Any,
    prompt_token_ids: list[int],
) -> tuple[bool | None, dict[str, Any] | None]:
    if reasoning_parser_class is None:
        return None, None

    parser_kwargs = {"chat_template_kwargs": chat_template_kwargs}
    if not getattr(request_for_sampling, "include_reasoning", True):
        return True, parser_kwargs
    if getattr(request_for_sampling, "_grammar_from_tool_parser", False):
        return True, parser_kwargs

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


def _wire_chat_logprobs_content(
    engine_response: dict[str, Any], tokenizer: TokenizerLike
) -> list[dict[str, Any]] | None:
    """Convert Dynamo's per-token wire logprobs to the OpenAI chat shape."""
    token_ids = list(engine_response.get("token_ids") or [])
    logprobs = engine_response.get("log_probs")
    if not token_ids or not isinstance(logprobs, list):
        return None
    if len(logprobs) != len(token_ids):
        logger.warning(
            "Dropping misaligned logprobs: token_ids=%d logprobs=%d",
            len(token_ids),
            len(logprobs),
        )
        return None

    top_by_position = engine_response.get("top_logprobs")
    if not isinstance(top_by_position, list):
        top_by_position = []

    content: list[dict[str, Any]] = []
    for position, (token_id, logprob) in enumerate(zip(token_ids, logprobs)):
        raw_top = (
            top_by_position[position]
            if position < len(top_by_position)
            and isinstance(top_by_position[position], list)
            else []
        )
        selected = next(
            (
                item
                for item in raw_top
                if isinstance(item, dict) and item.get("token_id") == token_id
            ),
            {},
        )
        token = selected.get("token")
        if not isinstance(token, str):
            try:
                token = tokenizer.decode([token_id], skip_special_tokens=False)
            except TypeError:
                token = tokenizer.decode([token_id])

        token_bytes = selected.get("bytes")
        if not isinstance(token_bytes, list):
            token_bytes = list(token.encode("utf-8"))

        top_logprobs: list[dict[str, Any]] = []
        for item in raw_top:
            if not isinstance(item, dict) or "logprob" not in item:
                continue
            top_token = item.get("token")
            if not isinstance(top_token, str):
                top_token_id = item.get("token_id")
                if not isinstance(top_token_id, int):
                    continue
                try:
                    top_token = tokenizer.decode(
                        [top_token_id], skip_special_tokens=False
                    )
                except TypeError:
                    top_token = tokenizer.decode([top_token_id])
            top_bytes = item.get("bytes")
            if not isinstance(top_bytes, list):
                top_bytes = list(top_token.encode("utf-8"))
            top_logprobs.append(
                {
                    "token": top_token,
                    "logprob": float(item["logprob"]),
                    "bytes": top_bytes,
                }
            )

        content.append(
            {
                "token": token,
                "logprob": float(logprob),
                "bytes": token_bytes,
                "top_logprobs": top_logprobs,
            }
        )
    return content


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
    if structured_outputs is not None and hasattr(
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


def _coerce_structured_outputs(
    structured_outputs: StructuredOutputsParams | dict[str, Any] | None,
) -> StructuredOutputsParams | None:
    if structured_outputs is None:
        return None
    if isinstance(structured_outputs, dict):
        params = {
            key: structured_outputs[key]
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
            )
            if structured_outputs.get(key) is not None
        }
        if not params:
            return None
        structured_outputs = StructuredOutputsParams(**params)
    if structured_outputs.all_constraints_none():
        return None
    return structured_outputs


def _has_structured_outputs(
    structured_outputs: StructuredOutputsParams | dict[str, Any] | None,
) -> bool:
    return _coerce_structured_outputs(structured_outputs) is not None


def _active_structured_outputs(
    structured_outputs: StructuredOutputsParams | dict[str, Any] | None,
) -> StructuredOutputsParams | None:
    """Discard request-model placeholders that carry no output constraint."""
    return _coerce_structured_outputs(structured_outputs)


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


def _inject_routing_metadata(
    dynamo_preproc: dict[str, Any],
    target: dict[str, Any],
    mm_routing_info: dict[str, Any] | None = None,
) -> None:
    extra_updates: dict[str, Any] = {}
    for key in ("reasoning_ended", "reasoning_parser_kwargs"):
        if key in dynamo_preproc:
            extra_updates[key] = dynamo_preproc[key]
    if dynamo_preproc.get("mm_processor_kwargs") is not None:
        extra_updates["mm_processor_kwargs"] = dynamo_preproc["mm_processor_kwargs"]

    if extra_updates:
        extra_args = target.get("extra_args")
        if not isinstance(extra_args, dict):
            extra_args = {}
            target["extra_args"] = extra_args
        extra_args.update(extra_updates)

    if mm_routing_info is not None:
        target["mm_routing_info"] = mm_routing_info


async def _build_engine_inputs(
    renderer: Any,
    engine_prompt: dict[str, Any],
    prompt_token_ids: list[int],
    *,
    cache_salt: str | None,
    mm_processor_kwargs: dict[str, Any] | None,
    defer_multimodal_processing: bool = False,
) -> dict[str, Any]:
    """Convert a rendered chat prompt into the EngineInput vLLM expects."""
    prompt_inputs = {**engine_prompt, "prompt_token_ids": prompt_token_ids}
    if cache_salt is not None:
        prompt_inputs["cache_salt"] = cache_salt
    if mm_processor_kwargs is not None:
        prompt_inputs["mm_processor_kwargs"] = mm_processor_kwargs

    if defer_multimodal_processing:
        # UUID-only media is resolved by the worker-side vLLM processor cache.
        # Processing it in the frontend would turn a frontend-local cache miss
        # into an error before the request can reach a worker whose cache may hit.
        prompt_inputs.pop("multi_modal_data", None)
        prompt_inputs.pop("multi_modal_uuids", None)

    return await renderer.process_for_engine_async(prompt_inputs, time.time())


def _normalize_vllm_image_parts(messages: list[Any]) -> None:
    """Normalize image parts before vLLM validates and renders them."""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            if image_url.get("detail") is None:
                image_url["detail"] = "auto"


class VllmProcessor:
    def __init__(
        self,
        tokenizer: TokenizerLike,
        input_processor: InputProcessor,
        output_processor: OutputProcessor,
        tool_parser_class: type[ToolParser] | None,
        reasoning_parser_class: type[ReasoningParser] | None,
        routed_engine: RoutedEngine,
        tool_parser_name: str | None = None,
        block_size: int = 16,
        enable_auto_tool_choice: bool = False,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        default_thinking_mode: str | None = None,
        structural_tag_mode: str = "off",
        structural_tag_scope: str = "auto",
        structural_tag_schema: str = "auto",
    ):
        self.tokenizer = tokenizer
        self.input_processor = input_processor
        self.routed_engine = routed_engine
        self.output_processor = output_processor
        self.tool_parser_class = tool_parser_class
        self.reasoning_parser_class = reasoning_parser_class
        self.exclude_tools_when_tool_choice_none = True
        self.block_size = block_size
        self.enable_auto_tool_choice = enable_auto_tool_choice
        self.default_chat_template_kwargs = default_chat_template_kwargs
        self.default_thinking_mode = default_thinking_mode
        self.structural_tag_mode = structural_tag_mode
        self.structural_tag_scope = structural_tag_scope
        self.structural_tag_schema = structural_tag_schema
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
        if dynamo_preproc.get("multi_modal_uuids"):
            # Keep the worker as the source of truth for UUID-backed media. A
            # worker-side processor-cache entry must include model-specific
            # prompt updates as well as tensors; frontend tensor transfer cannot
            # populate that complete entry for a later UUID-only request.
            logger.debug(
                "[mm-routing] User multimodal UUID present; using text-prefix "
                "routing and worker-side multimodal processing"
            )
            _nvtx.end_range(rng_routing)
            return None, cleanup_items, nixl_transferred

        if vllm_preproc.mm_features:
            mm_routing_info = build_mm_routing_info_from_features(
                vllm_preproc.mm_features,
                prompt_token_ids=list(vllm_preproc.prompt_token_ids),
            )
            (
                mm_hashes_list,
                mm_placeholders_list,
                mm_hashes_by_modality,
                mm_placeholders_by_modality,
            ) = _group_mm_feature_metadata(vllm_preproc.mm_features)
            if "extra_args" not in dynamo_preproc:
                dynamo_preproc["extra_args"] = {}
            dynamo_preproc["extra_args"]["mm_hashes"] = mm_hashes_list
            dynamo_preproc["extra_args"]["mm_placeholders"] = mm_placeholders_list
            dynamo_preproc["extra_args"][
                "mm_hashes_by_modality"
            ] = mm_hashes_by_modality
            dynamo_preproc["extra_args"][
                "mm_placeholders_by_modality"
            ] = mm_placeholders_by_modality
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
                        _mm_feature_modality(f),
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
                    transfer_modality = _single_transfer_modality(
                        vllm_preproc.mm_features
                    )
                    if transfer_modality is None:
                        extra_update = None
                        logger.debug(
                            "[mm-routing] mixed modalities; backend will run "
                            "HF processor"
                        )
                    else:
                        if self._sender is None:
                            self._sender = (
                                MmKwargsShmSender()
                                if self.use_shm_transfer
                                else MmKwargsNixlSender()
                            )
                        # NVTX annotation is owned by MmKwargsSender.prepare via
                        # the subclass's _nvtx_label/_nvtx_color class attrs.
                        extra_update, cleanup_items = await self._sender.prepare(
                            vllm_preproc.mm_features,
                            modality=transfer_modality,
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

        else:
            logger.debug("[mm-routing] No mm_features — text-only request")
        _nvtx.end_range(rng_routing)

        return mm_routing_info, cleanup_items, nixl_transferred

    # Ideally we would map NVCreateChatCompletionRequest into Python so it can be type checked, but
    # it has a lot of fields.
    # request: dynamo.NVCreateChatCompletionRequest
    async def generator(
        self, request: dict[str, Any], context: Any | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run a single request through the engine. Does pre and post processing on this machine, delegates
        model inference to a backend using the router.
        """
        with _nvtx.annotate("mm_frontend:generator", color="blue"):
            async for item in self._generator_inner(request, context=context):
                yield item

    async def _generator_inner(
        self, request: dict[str, Any], context: Any | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        request_id = random_uuid()

        messages = request.get("messages") or []
        _normalize_vllm_image_parts(messages)
        # Validate cache-UUID modality support before vLLM downloads or
        # processes media. Dynamo currently exposes vLLM cache UUIDs for
        # images only.
        mm_data, mm_uuids = extract_mm_urls(messages)

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
                default_chat_template_kwargs=self.default_chat_template_kwargs,
                default_thinking_mode=self.default_thinking_mode,
                structural_tag_mode=self.structural_tag_mode,
                structural_tag_scope=self.structural_tag_scope,
                structural_tag_schema=self.structural_tag_schema,
            )

        request_for_sampling = pre.request_for_sampling
        tool_parser = pre.tool_parser
        chat_template_kwargs = pre.chat_template_kwargs
        engine_prompt = pre.engine_prompt
        tokens = pre.prompt_token_ids
        guided_decoding = pre.guided_decoding

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
        # The OpenAI request model may expose a constraints-empty placeholder
        # even when the client omitted structured outputs. vLLM validates any
        # non-None value, so do not pass that placeholder to process_inputs().
        sampling_params.structured_outputs = _active_structured_outputs(
            sampling_params.structured_outputs
        )

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
                sampling_params.structured_outputs = StructuredOutputsParams(json=tool_schema)
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
            nvext_max_thinking_tokens = chat_template_kwargs.get("reasoning_budget")
        if nvext_max_thinking_tokens is None:
            nvext_max_thinking_tokens = _default_constrained_max_thinking_tokens(
                request_for_sampling=request_for_sampling,
                sampling_params=sampling_params,
                chat_template_kwargs=chat_template_kwargs,
            )
        logprobs = request_for_sampling.logprobs
        top_logprobs = request_for_sampling.top_logprobs
        if logprobs is True:
            sampling_params.logprobs = top_logprobs if top_logprobs is not None else 1
        elif isinstance(logprobs, int) and not isinstance(logprobs, bool):
            sampling_params.logprobs = logprobs
        elif top_logprobs not in (None, 0):
            sampling_params.logprobs = top_logprobs
        # TODO: Support logprobs in the distributed vLLM chat processor by
        # converting worker log_probs/top_logprobs into EngineCoreOutput.new_logprobs.
        if sampling_params.logprobs is not None:
            logger.warning(
                "Logprobs requested but not supported in distributed inference mode"
            )

        with _nvtx.annotate("mm_frontend:process_inputs", color="orange"):
            # render_messages_async returns a raw prompt. Convert it to a typed
            # EngineInput before process_inputs. User UUID requests deliberately
            # stay token-only here: the selected worker must populate and query
            # its own complete processor-cache entry.
            engine_inputs = await _build_engine_inputs(
                self.input_processor.renderer,
                engine_prompt,
                tokens,
                cache_salt=request_for_sampling.cache_salt,
                mm_processor_kwargs=request_for_sampling.mm_processor_kwargs,
                defer_multimodal_processing=mm_uuids is not None,
            )
            vllm_preproc: EngineCoreRequest = self.input_processor.process_inputs(
                request_id,
                engine_inputs,
                sampling_params,
                GENERATION_TASKS,  # vLLM 0.17.0: required supported_tasks arg
            )

        InputProcessor.assign_request_id(vllm_preproc)

        # vLLM 0.17.0 removed EngineCoreRequest.eos_token_id. Dynamo now uses
        # tokenizer metadata for EOS ids when constructing the router payload.

        reasoning_ended, reasoning_parser_kwargs = _build_reasoning_parser_metadata(
            self.reasoning_parser_class,
            self.tokenizer,
            chat_template_kwargs,
            request_for_sampling,
            tokens,
        )

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
        if reasoning_ended is not None:
            dynamo_preproc["reasoning_ended"] = reasoning_ended
        if reasoning_parser_kwargs is not None:
            dynamo_preproc["reasoning_parser_kwargs"] = reasoning_parser_kwargs
        # Attach user cache identities before building routing metadata. Opaque
        # UUIDs deliberately suppress multimodal exact routing and frontend
        # tensor transfer; the worker owns processor-cache fill and lookup.
        if mm_uuids:
            dynamo_preproc["multi_modal_uuids"] = mm_uuids
        reasoning_budget = _get_attr_or_item(request_for_sampling, "reasoning_budget")
        if reasoning_budget is None:
            reasoning_budget = chat_template_kwargs.get("reasoning_budget")
        if reasoning_budget is not None:
            dynamo_preproc["reasoning_budget"] = reasoning_budget
        reasoning_budget_grace_period = _get_attr_or_item(
            request_for_sampling, "reasoning_budget_grace_period"
        )
        if reasoning_budget_grace_period is None:
            reasoning_budget_grace_period = chat_template_kwargs.get(
                "reasoning_budget_grace_period"
            )
        if reasoning_budget_grace_period is not None:
            dynamo_preproc["reasoning_budget_grace_period"] = (
                reasoning_budget_grace_period
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
                    sampling_params=sampling_params,
                    prompt_token_ids=tokens,
                    tool_parser=tool_parser,
                    reasoning_parser_class=self.reasoning_parser_class,
                    chat_template_kwargs=chat_template_kwargs,
                    stream_response=bool(request.get("stream", False)),
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
                mm_routing_info=mm_routing_info,
                context=context,
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
        mm_routing_info: dict[str, Any] | None = None,
        context: Any | None = None,
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

        # Rust postprocessor is bypassed on this path, so emit the multimodal
        # content-part counts here too (else frontend metrics report zero media).
        input_tokens = len(tokens)
        cumulative_output_tokens = 0
        _mm_counts, _ = extract_mm_urls(request.get("messages") or [])
        _mm_counts = _mm_counts or {}
        image_count = len(_mm_counts.get("image_url", []))
        video_count = len(_mm_counts.get("video_url", []))
        audio_count = len(_mm_counts.get("audio_url", []))
        pending_logprobs: dict[int, list[dict[str, Any]]] = {
            output_idx: [] for output_idx in output_request_ids
        }

        try:
            _inject_routing_metadata(dynamo_preproc, dynamo_preproc, mm_routing_info)
            with _nvtx.annotate("mm_frontend:routed_engine_generate", color="red"):
                dynamo_stream = await self.routed_engine.generate(
                    dynamo_preproc, context=context
                )

            rng_stream = _nvtx.start_range(
                "mm_frontend:stream_response", color="purple"
            )
            async for dynamo_response in dynamo_stream:
                if dynamo_response.is_error():
                    comments = dynamo_response.comments() or []
                    message = "; ".join(comments) or "unknown routed_engine error"
                    logger.error(
                        "routed_engine error for request %s: %s",
                        request_id,
                        message,
                    )
                    raise RuntimeError(message)
                engine_response = dynamo_response.data()

                if engine_response is None:
                    if dynamo_response.is_error():
                        yield handle_engine_error(engine_response, request_id, logger)
                        break
                    # No data or error fields, means we may have a comment or other kind of event.
                    # I'm not sure what those are used for, so TODO. Skip for now.
                    continue

                if "token_ids" not in engine_response:
                    yield handle_engine_error(engine_response, request_id, logger)
                    break

                # Count before any choice gate — tool/reasoning parsers may
                # consume tokens without emitting a visible delta.
                chunk_tokens = len(engine_response.get("token_ids") or [])
                cumulative_output_tokens += chunk_tokens

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

                raw_finish_reason = _normalize_router_finish_reason(
                    engine_response.get("finish_reason")
                )
                finish_reason = map_finish_reason(raw_finish_reason)
                stop_reason = engine_response.get("stop_reason")
                raw_token_ids = list(engine_response["token_ids"])
                engine_text = engine_response.get("text") or ""
                chat_logprobs = _wire_chat_logprobs_content(
                    engine_response, self.tokenizer
                )
                if chat_logprobs:
                    pending_logprobs[output_idx].extend(chat_logprobs)

                output_kwargs: dict[str, Any] = {
                    "request_id": output_request_id,
                    "new_token_ids": engine_response["token_ids"],
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
                postprocess_error = False
                if not vllm_out.request_outputs:
                    post = post_processors.get(output_idx)
                    parser_needs_raw_delta = bool(
                        post is not None
                        and post.needs_raw_parser_delta(raw_token_ids)
                    )
                    if post is not None and (
                        raw_finish_reason or parser_needs_raw_delta
                    ):
                        choice = post.process_output(
                            SimpleNamespace(
                                index=output_idx,
                                token_ids=raw_token_ids,
                                # vLLM deliberately buffers incomplete UTF-8. Only
                                # bypass that buffer for parser-significant markers.
                                text=engine_text if parser_needs_raw_delta else "",
                                finish_reason=raw_finish_reason,
                                logprobs=None,
                            ),
                            raw_delta_token_ids=raw_token_ids,
                        )
                        if choice:
                            choices.append(choice)
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
                            postprocess_error = True
                            break
                        output = _with_parser_visible_engine_text(
                            output,
                            engine_text,
                            parser_needs_raw_delta=post.needs_raw_parser_delta(
                                raw_token_ids
                            ),
                        )
                        choice = post.process_output(
                            output, raw_delta_token_ids=raw_token_ids
                        )
                        if choice:
                            choices.append(choice)

                if postprocess_error:
                    continue

                # One envelope per iteration carries both data and metrics so
                # client cancellation can't drop the annotation between yields.
                envelope: dict[str, Any] = {"_dynamo_annotated": True}
                if choices:
                    choices = _bridge_structured_tool_content_choices(
                        request_for_sampling,
                        choices,
                    )
                    for choice in choices:
                        choice_index = choice.get("index", 0)
                        if sp.logprobs is not None:
                            choice["logprobs"] = {
                                "content": pending_logprobs[choice_index] or None,
                                "refusal": None,
                            }
                            pending_logprobs[choice_index] = []
                        post = post_processors.get(choice.get("index", 0))
                        strip_tool_markup = getattr(
                            post, "_strip_tool_markup_from_delta", None
                        )
                        if strip_tool_markup is not None:
                            strip_tool_markup(choice.get("delta") or {})
                    dynamo_out = {
                        "id": request_id,
                        "choices": choices,
                        "created": int(time.time()),
                        "model": request["model"],
                        "object": "chat.completion.chunk",
                    }
                    if usage := engine_response.get("completion_usage"):
                        dynamo_out["usage"] = usage
                    envelope["data"] = dynamo_out

                metrics = {
                    "input_tokens": input_tokens,
                    "output_tokens": cumulative_output_tokens,
                    "chunk_tokens": chunk_tokens,
                }
                # Include nonzero counts on every frame (text-only carries nothing).
                if image_count:
                    metrics["image_count"] = image_count
                if video_count:
                    metrics["video_count"] = video_count
                if audio_count:
                    metrics["audio_count"] = audio_count
                envelope["event"] = "llm_metrics"
                envelope["comment"] = [json.dumps(metrics)]

                yield envelope
            _nvtx.end_range(rng_stream)
        except Exception as e:
            logger.exception("Error generating response for request %s", request_id)
            raise
        finally:
            for output_request_id in registered_request_ids:
                if output_request_id in self.output_processor.request_states:
                    self.output_processor.abort_requests(
                        [output_request_id], internal=True
                    )


def _resolve_reasoning_parser(
    flags: Namespace, runtime_config: dict[str, Any]
) -> type[Any] | None:
    reasoning_parser_plugin = getattr(flags, "reasoning_parser_plugin", None)
    if reasoning_parser_plugin:
        ReasoningParserManager.import_reasoning_parser(reasoning_parser_plugin)

    reasoning_parser_name = getattr(
        flags, "reasoning_parser", None
    ) or runtime_config.get("reasoning_parser")
    if not reasoning_parser_name:
        return None
    return ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)


class EngineFactory:
    def __init__(
        self,
        config: FrontendConfig,
        flags: Namespace,
    ):
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
        routed_engine: RoutedEngine,
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

        local_dir = mdc.local_dir()
        if not os.path.isdir(local_dir):
            raise RuntimeError(
                f"MDC local_dir {local_dir!r} not populated for model {mdc.name()!r}; "
                f"download_config must run before the engine factory."
            )

        tokenizer_mode = getattr(self.flags, "tokenizer_mode", None) or "auto"
        config_format = getattr(self.flags, "config_format", None) or "auto"
        load_format = getattr(self.flags, "load_format", None) or "dummy"
        trust_remote_code = self.config.trust_remote_code
        enable_auto_tool_choice = getattr(self.flags, "enable_auto_tool_choice", False)

        model_config_kwargs = {
            "model": local_dir,
            "tokenizer_mode": tokenizer_mode,
            "config_format": config_format,
            "trust_remote_code": trust_remote_code,
        }
        context_length = _runtime_config_context_length(mdc)
        if context_length:
            os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
            model_config_kwargs["max_model_len"] = context_length
            logger.info(
                "vLLM frontend ModelConfig max_model_len=%d "
                "from runtime_config.context_length",
                context_length,
            )
        model_config = ModelConfig(**model_config_kwargs)
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

        _ensure_chat_template(
            tokenizer, local_dir, getattr(self.flags, "chat_template", None)
        )

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

        reasoning_parser_class = _resolve_reasoning_parser(
            self.flags, mdc.runtime_config()
        )
        default_thinking_mode = runtime_default_thinking_mode(mdc.runtime_config())
        (
            structural_tag_mode,
            structural_tag_scope,
            structural_tag_schema,
        ) = _runtime_config_structural_tag_options(mdc)

        block_size = self.config.kv_cache_block_size or 16

        gen = VllmProcessor(
            tokenizer,
            input_processor,
            output_processor,
            tool_parser_class,
            reasoning_parser_class,
            routed_engine,
            tool_parser_name=tool_parser_name,
            block_size=block_size,
            enable_auto_tool_choice=enable_auto_tool_choice,
            default_chat_template_kwargs=getattr(
                self.flags, "default_chat_template_kwargs", None
            ),
            default_thinking_mode=default_thinking_mode,
            structural_tag_mode=structural_tag_mode,
            structural_tag_scope=structural_tag_scope,
            structural_tag_schema=structural_tag_schema,
        )
        gen.exclude_tools_when_tool_choice_none = (
            self.config.exclude_tools_when_tool_choice_none
        )

        return PythonAsyncEngine(gen.generator, loop)
