# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional

import pybase64
import sglang as sgl

from dynamo._core import Context
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


class DecodeWorkerHandler(BaseWorkerHandler):
    """Handler for decode workers in both aggregated and disaggregated serving modes."""

    SERVICE_OVERLOADED_ERROR_TYPE = "service_overloaded"
    THINK_START = "<think>"
    THINK_END = "</think>"
    DEFAULT_XGRAMMAR_MAX_STRING_LENGTH = 4096
    DEFAULT_XGRAMMAR_MAX_ARRAY_ITEMS = 64
    DEFAULT_XGRAMMAR_MAX_OBJECT_PROPERTIES = 64
    DEFAULT_XGRAMMAR_MAX_REASONING_CHARS = 8192

    def __init__(
        self,
        engine: sgl.Engine,
        config: Config,
        publisher: DynamoSglangPublisher,
        generate_endpoint=None,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Initialize decode worker handler.

        Args:
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: Metrics publisher for the worker.
            shutdown_event: Optional event to signal shutdown.
            generate_endpoint: The endpoint handle for discovery registration.
        """
        super().__init__(
            engine,
            config,
            publisher,
            generate_endpoint,
            shutdown_event,
        )
        if self.serving_mode == DisaggregationMode.DECODE:
            logging.info(
                "Decode worker handler initialized (disaggregated decode mode)"
            )
        else:
            logging.info("Decode worker handler initialized (aggregated mode)")
        self.max_total_requests = self._get_positive_int_env(
            "DYN_REQUEST_MAX_TOTAL_REQUESTS"
        ) or self._get_default_max_total_requests()
        self._request_admission_lock = asyncio.Lock()
        self._active_request_admissions = 0
        self._active_request_admissions_high_water = 0

    @staticmethod
    def _get_positive_int_env(name: str) -> Optional[int]:
        value = os.environ.get(name)
        if value in (None, ""):
            return None
        try:
            parsed = int(value)
        except ValueError:
            logging.warning("Ignoring invalid %s=%r", name, value)
            return None
        if parsed <= 0:
            return None
        return parsed

    def _get_default_max_total_requests(self) -> Optional[int]:
        max_running_requests = getattr(
            self.config.server_args, "max_running_requests", None
        )
        if max_running_requests in (None, ""):
            return None
        try:
            total_limit = int(max_running_requests)
        except (TypeError, ValueError):
            logging.warning(
                "Ignoring invalid SGLang max_running_requests=%r",
                max_running_requests,
            )
            return None
        if total_limit <= 0:
            return None
        # SGLang treats --max-running-requests as the per-replica total and
        # divides it by dp_size internally. Do not multiply by dp_size here.
        return total_limit

    async def _try_reserve_request_slot(
        self, request_id: str
    ) -> tuple[bool, int | None, int | None]:
        limit = self.max_total_requests
        if limit is None:
            return True, None, None

        async with self._request_admission_lock:
            current_total = self._active_request_admissions
            if current_total >= limit:
                logging.info(
                    "Rejecting request %s due to local total request limit: %s/%s",
                    request_id,
                    current_total,
                    limit,
                )
                return False, current_total, limit

            self._active_request_admissions += 1
            if self._active_request_admissions > self._active_request_admissions_high_water:
                self._active_request_admissions_high_water = self._active_request_admissions
                logging.info(
                    "Worker local total request slots in use: %s/%s",
                    self._active_request_admissions,
                    limit,
                )
            return True, current_total + 1, limit

    async def _release_request_slot_reservation(self) -> None:
        async with self._request_admission_lock:
            if self._active_request_admissions > 0:
                self._active_request_admissions -= 1

    @classmethod
    def _build_overload_extra_args(
        cls,
        message: str,
        *,
        current_total_requests: int | None = None,
        total_request_limit: int | None = None,
    ) -> dict[str, Any]:
        extra_args: dict[str, Any] = {
            "dynamo_error_type": cls.SERVICE_OVERLOADED_ERROR_TYPE,
            "error_message": message,
        }
        if current_total_requests is not None:
            extra_args["worker_total_requests"] = current_total_requests
        if total_request_limit is not None:
            extra_args["worker_total_request_limit"] = total_request_limit
        return extra_args

    @classmethod
    def _build_token_overload_response(
        cls,
        message: str,
        *,
        current_total_requests: int | None = None,
        total_request_limit: int | None = None,
    ) -> dict[str, Any]:
        return {
            "finish_reason": {"error": message},
            "token_ids": [],
            "extra_args": cls._build_overload_extra_args(
                message,
                current_total_requests=current_total_requests,
                total_request_limit=total_request_limit,
            ),
        }

    @classmethod
    def _build_text_overload_chunk(
        cls,
        message: str,
        request_id: str,
        model: str,
        *,
        current_total_requests: int | None = None,
        total_request_limit: int | None = None,
    ) -> dict[str, Any]:
        return {
            "id": request_id,
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": "error",
                }
            ],
            "model": model,
            "object": "chat.completion.chunk",
            "error": {"message": message},
            "extra_args": cls._build_overload_extra_args(
                message,
                current_total_requests=current_total_requests,
                total_request_limit=total_request_limit,
            ),
        }

    def cleanup(self) -> None:
        """Shutdown the engine and cleanup resources."""
        super().cleanup()
        self.engine.shutdown()
        logging.info("Engine shutdown")

    @staticmethod
    def _choice_regex(choices: list[Any]) -> str:
        return "(?:" + "|".join(re.escape(str(choice)) for choice in choices) + ")"

    @staticmethod
    def _as_json_string(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, separators=(",", ":"))

    @staticmethod
    def _literal_from_contains_pattern(pattern: str) -> Optional[str]:
        if not (pattern.startswith(".*") and pattern.endswith(".*")):
            return None
        literal = pattern[2:-2]
        # Only handle the safe subset emitted by our tests/common schemas: a
        # contains-regex with no real regex operators except escaped literals.
        if re.search(r"(?<!\\)[\[\]\(\)\{\}\|\+\?\^\$]", literal):
            return None
        return re.sub(r"\\(.)", r"\1", literal)

    @staticmethod
    def _bounded_char_class_pattern(
        pattern: str,
        min_length: Any,
        max_length: Any,
    ) -> Optional[str]:
        match = re.fullmatch(r"^\^(\[(?:\\.|[^\]])+\])([+*])\$$", pattern)
        if not match:
            return None

        char_class, quantifier = match.groups()
        if '"' in char_class:
            # SGLang/xgrammar can emit unescaped quote characters from JSON
            # Schema string patterns, producing invalid JSON. Fall back to
            # the length-only normalization for such broad printable classes.
            return None
        lower = 1 if quantifier == "+" else 0
        if isinstance(min_length, int):
            lower = max(lower, min_length)
        if isinstance(max_length, int):
            return f"{char_class}{{{lower},{max_length}}}"
        return f"{char_class}{{{lower},}}"

    @classmethod
    def _normalize_json_schema_for_xgrammar(cls, schema: Any) -> Any:
        """Adapt common JSON Schema features to SGLang/xgrammar's supported subset.

        xgrammar currently ignores some combinations such as string pattern with
        minLength/maxLength/format, and it does not force required fields that are
        absent from properties. It can also keep generating forever when a schema
        permits unbounded strings, arrays, or open objects. Normalizing here keeps
        Dynamo's OpenAI-compatible behavior stricter without changing the
        request-facing schema contract.
        """
        if isinstance(schema, bool) or schema is None:
            return schema
        if isinstance(schema, list):
            return [cls._normalize_json_schema_for_xgrammar(item) for item in schema]
        if not isinstance(schema, dict):
            return schema

        normalized = {
            key: cls._normalize_json_schema_for_xgrammar(value)
            for key, value in schema.items()
        }

        required = normalized.get("required")
        properties = normalized.get("properties")

        # Some real payloads use a property-bag shape like
        # {"host": {...}, "port": {...}, "required": [...]}. Convert that to
        # normal JSON Schema object form before handing it to xgrammar.
        if properties is None and isinstance(required, list):
            schema_keywords = {
                "$schema",
                "$defs",
                "additionalProperties",
                "allOf",
                "anyOf",
                "description",
                "enum",
                "format",
                "items",
                "maxItems",
                "maxLength",
                "maximum",
                "minItems",
                "minLength",
                "minimum",
                "not",
                "oneOf",
                "pattern",
                "properties",
                "required",
                "title",
                "type",
            }
            inferred_properties = {
                key: value
                for key, value in normalized.items()
                if key not in schema_keywords and isinstance(value, dict)
            }
            if inferred_properties:
                for key in inferred_properties:
                    normalized.pop(key, None)
                normalized["type"] = normalized.get("type") or "object"
                normalized["properties"] = inferred_properties
                properties = inferred_properties

        if isinstance(required, list):
            if not isinstance(properties, dict):
                properties = {}
                normalized["properties"] = properties
            for key in required:
                if isinstance(key, str) and key not in properties:
                    properties[key] = {}

        for combinator in ("oneOf", "anyOf"):
            branches = normalized.get(combinator)
            if isinstance(branches, list):
                repaired_branches = []
                for branch in branches:
                    if isinstance(branch, dict):
                        branch_properties = branch.get("properties")
                        branch_required = branch.get("required")
                        if (
                            branch_required is None
                            and isinstance(branch_properties, dict)
                            and branch_properties
                        ):
                            branch = {
                                **branch,
                                "required": list(branch_properties.keys()),
                            }
                    repaired_branches.append(branch)
                normalized[combinator] = repaired_branches

        schema_type = normalized.get("type")
        if schema_type == "string":
            normalized.setdefault(
                "maxLength", cls.DEFAULT_XGRAMMAR_MAX_STRING_LENGTH
            )
        elif schema_type == "array" or "items" in normalized:
            normalized.setdefault("type", "array")
            normalized.setdefault("maxItems", cls.DEFAULT_XGRAMMAR_MAX_ARRAY_ITEMS)
        elif (
            schema_type == "object"
            or isinstance(properties, dict)
            or "additionalProperties" in normalized
        ):
            normalized.setdefault("type", "object")
            normalized.setdefault(
                "maxProperties", cls.DEFAULT_XGRAMMAR_MAX_OBJECT_PROPERTIES
            )

        if (
            normalized.get("type") == "object"
            and isinstance(normalized.get("properties"), dict)
            and isinstance(normalized.get("required"), list)
        ):
            # xgrammar can satisfy root anyOf/not constraints without preserving
            # sibling object/required constraints. Prefer the concrete root
            # object shape; it is the contract clients validate against.
            for combinator in ("anyOf", "allOf", "not"):
                normalized.pop(combinator, None)

        if normalized.get("type") == "string":
            pattern = normalized.get("pattern")
            if isinstance(pattern, str):
                literal = cls._literal_from_contains_pattern(pattern)
                if literal:
                    normalized.pop("pattern", None)
                    normalized["enum"] = [literal]
                elif bounded_pattern := cls._bounded_char_class_pattern(
                    pattern,
                    normalized.get("minLength"),
                    normalized.get("maxLength"),
                ):
                    # xgrammar may ignore minLength/maxLength when pattern is
                    # also present. Fold simple anchored char-class patterns
                    # into a single regex so both character and length
                    # constraints survive.
                    normalized["pattern"] = bounded_pattern
                    normalized.pop("minLength", None)
                    normalized.pop("maxLength", None)
                elif any(
                    key in normalized
                    for key in ("minLength", "maxLength", "format")
                ):
                    # xgrammar warns that this combination causes the length
                    # constraints to be ignored. Prefer length bounds over a
                    # loose character-class pattern because length violations
                    # are not repairable after generation.
                    normalized.pop("pattern", None)
            if normalized.get("format") == "":
                normalized.pop("format", None)

        return normalized

    @classmethod
    def _guided_to_sglang_params(cls, guided: Any) -> Dict[str, Any]:
        if not isinstance(guided, dict):
            return {}

        json_schema = guided.get("json")
        if guided.get("json_object"):
            json_schema = json_schema or {"type": "object"}
        if json_schema is False:
            return {"regex": "a" * 4096}
        json_schema = cls._normalize_json_schema_for_xgrammar(json_schema)

        regex = guided.get("regex")
        choice = guided.get("choice")
        grammar = guided.get("grammar")
        structural_tag = guided.get("structural_tag")
        enable_thinking = guided.get("enable_thinking")

        if grammar is not None:
            cls._validate_ebnf_grammar(grammar)

        if choice and not regex:
            regex = cls._choice_regex(choice)

        if enable_thinking and structural_tag is None:
            reasoning_content = {
                "type": "regex",
                "pattern": f"[^<]{{0,{cls.DEFAULT_XGRAMMAR_MAX_REASONING_CHARS}}}",
            }
            if json_schema is not None:
                structural_tag = {
                    "type": "sequence",
                    "elements": [
                        {
                            "type": "tag",
                            "begin": "",
                            "content": reasoning_content,
                            "end": "</think>",
                        },
                        {"type": "json_schema", "json_schema": json_schema},
                    ],
                }
                json_schema = None
            elif regex is not None:
                structural_tag = {
                    "type": "sequence",
                    "elements": [
                        {
                            "type": "tag",
                            "begin": "",
                            "content": reasoning_content,
                            "end": "</think>",
                        },
                        {"type": "regex", "pattern": regex},
                    ],
                }
                regex = None
            elif grammar is not None:
                structural_tag = {
                    "type": "sequence",
                    "elements": [
                        {
                            "type": "tag",
                            "begin": "",
                            "content": reasoning_content,
                            "end": "</think>",
                        },
                        {"type": "grammar", "grammar": grammar},
                    ],
                }
                grammar = None

        params: Dict[str, Any] = {}
        if structural_tag is not None:
            structural_tag = cls._normalize_structural_tag_for_sglang(structural_tag)
            params["structural_tag"] = cls._as_json_string(structural_tag)
        elif json_schema is not None:
            params["json_schema"] = cls._as_json_string(json_schema)
        elif regex is not None:
            params["regex"] = regex
        elif grammar is not None:
            params["ebnf"] = grammar

        return params

    @staticmethod
    def _validate_ebnf_grammar(grammar: Any) -> None:
        if not isinstance(grammar, str):
            raise ValueError("Grammar error: structured_outputs.grammar must be a string")
        try:
            import xgrammar as xgr

            xgr.Grammar.from_ebnf(grammar)
        except Exception as exc:
            raise ValueError(f"Grammar error: {exc}") from exc

    @staticmethod
    def _normalize_structural_tag_for_sglang(structural_tag: Any) -> Any:
        """Convert Dynamo/vLLM-style structural formats to SGLang's wrapper.

        SGLang accepts either the legacy structural_tag object with
        ``structures``/``triggers`` or the newer xgrammar wrapper:
        ``{"type": "structural_tag", "format": ...}``.  Dynamo's guided
        decoding path internally represents the inner xgrammar format directly
        (for example ``{"type": "sequence", ...}``), so wrap that shape before
        handing it to SGLang.
        """

        if not isinstance(structural_tag, dict):
            return structural_tag
        if "structures" in structural_tag:
            return structural_tag
        if structural_tag.get("type") == "structural_tag":
            return structural_tag
        if "format" in structural_tag:
            return {"type": "structural_tag", "format": structural_tag["format"]}
        return {"type": "structural_tag", "format": structural_tag}

    @staticmethod
    def _request_enable_thinking(request: Dict[str, Any]) -> bool:
        """Return whether this request asked the chat template to enable thinking."""

        sampling_options = request.get("sampling_options")
        if isinstance(sampling_options, dict):
            guided = sampling_options.get("guided_decoding")
            if isinstance(guided, dict) and guided.get("enable_thinking") is not None:
                return bool(guided.get("enable_thinking"))

        guided = request.get("guided_decoding")
        if isinstance(guided, dict) and guided.get("enable_thinking") is not None:
            return bool(guided.get("enable_thinking"))

        chat_template_kwargs = request.get("chat_template_kwargs")
        if isinstance(chat_template_kwargs, dict):
            if chat_template_kwargs.get("enable_thinking") is not None:
                return bool(chat_template_kwargs.get("enable_thinking"))
            if chat_template_kwargs.get("thinking") is not None:
                return bool(chat_template_kwargs.get("thinking"))

        return False

    @classmethod
    def _sampling_params_enable_thinking(cls, sampling_params: Dict[str, Any]) -> bool:
        """Detect reasoning handoff after guided decoding was lowered to SGLang."""

        structural_tag = sampling_params.get("structural_tag")
        return isinstance(structural_tag, str) and cls.THINK_END in structural_tag

    @classmethod
    def _split_qwen_reasoning_delta(
        cls,
        delta: str,
        state: Dict[str, Any],
        *,
        flush: bool,
    ) -> Dict[str, str]:
        """Split Qwen thinking text from answer content for SGLang text IO.

        SGLang's text stream is cumulative text. For Qwen thinking mode, the
        model may emit reasoning without an opening ``<think>`` and then close
        with ``</think>`` before the constrained answer. OpenAI clients expect
        that pre-answer span as ``reasoning_content``, not ``content``.
        """

        if state.get("done"):
            return {"content": delta} if delta else {}

        segment = state.get("pending", "") + delta
        state["pending"] = ""
        if not segment:
            return {}

        end = cls.THINK_END
        end_pos = segment.find(end)
        if end_pos >= 0:
            state["done"] = True
            reasoning = segment[:end_pos].replace(cls.THINK_START, "")
            content = segment[end_pos + len(end) :]
            split: Dict[str, str] = {}
            if reasoning:
                split["reasoning_content"] = reasoning
            if content:
                split["content"] = content
            return split

        keep = len(end) - 1
        if flush:
            reasoning = segment
        elif len(segment) <= keep:
            state["pending"] = segment
            reasoning = ""
        else:
            reasoning = segment[:-keep]
            state["pending"] = segment[-keep:]

        reasoning = reasoning.replace(cls.THINK_START, "")
        return {"reasoning_content": reasoning} if reasoning else {}

    def _build_sampling_params(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Build sampling params from request format.

        Args:
            request: Request dict in either token-based or OpenAI format.

        Returns:
            Dict of sampling parameters for SGLang engine.
        """
        if "sampling_options" in request or "stop_conditions" in request:
            # Dynamo-preprocessed request format. This is used both with
            # --skip-tokenizer-init and with tokenizer init enabled for SGLang
            # xgrammar support.
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})

            param_mapping = {
                "presence_penalty": sampling_opts.get("presence_penalty"),
                "frequency_penalty": sampling_opts.get("frequency_penalty"),
                "repetition_penalty": sampling_opts.get("repetition_penalty"),
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "min_p": sampling_opts.get("min_p"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
                "ignore_eos": stop_conditions.get("ignore_eos"),
            }
            param_mapping.update(
                self._guided_to_sglang_params(sampling_opts.get("guided_decoding"))
            )
        else:
            # OpenAI request format
            param_mapping = {
                "presence_penalty": request.get("presence_penalty"),
                "frequency_penalty": request.get("frequency_penalty"),
                "repetition_penalty": request.get("repetition_penalty"),
                "temperature": request.get("temperature"),
                "top_p": request.get("top_p"),
                "top_k": request.get("top_k"),
                "min_p": request.get("min_p"),
                "max_new_tokens": request.get("max_tokens"),
            }
            param_mapping.update(
                self._guided_to_sglang_params(request.get("guided_decoding"))
            )

        sampling_params = {k: v for k, v in param_mapping.items() if v is not None}
        if os.environ.get("DYN_SGLANG_LOG_SAMPLING_PARAMS") == "1":
            logging.warning(
                "Dynamo SGLang sampling params: %s",
                json.dumps(sampling_params, sort_keys=True)[:4000],
            )
        return sampling_params

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate response in aggregated or disaggregated mode.

        Args:
            request: Request dict with input and sampling parameters.
            context: Context object for cancellation handling.

        Yields:
            Response dicts with token_ids or OpenAI-formatted chunks.

        Raises:
            RuntimeError: If no bootstrap info received from prefill worker.
        """
        logging.debug(f"New Request ID: {context.id()}")
        trace_id = context.trace_id
        sampling_params = self._build_sampling_params(request)
        input_param = self._get_input_param(request)
        split_reasoning = self._request_enable_thinking(
            request
        ) or self._sampling_params_enable_thinking(sampling_params)
        return_routed_experts = getattr(
            self.config.server_args, "enable_return_routed_experts", False
        )
        priority = (request.get("routing") or {}).get("priority")

        reserved, current_total_requests, total_request_limit = (
            await self._try_reserve_request_slot(context.id())
        )
        if not reserved:
            message = (
                f"Worker local total request limit reached "
                f"({current_total_requests}/{total_request_limit})"
            )
            if self.use_sglang_text_io:
                yield self._build_text_overload_chunk(
                    message,
                    request.get("id") or request.get("request_id") or context.id(),
                    self.config.server_args.served_model_name,
                    current_total_requests=current_total_requests,
                    total_request_limit=total_request_limit,
                )
            else:
                yield self._build_token_overload_response(
                    message,
                    current_total_requests=current_total_requests,
                    total_request_limit=total_request_limit,
                )
            return

        slot_released = False

        async def release_request_slot_once() -> None:
            nonlocal slot_released
            if slot_released:
                return
            slot_released = True
            await self._release_request_slot_reservation()

        try:
            if self.serving_mode == DisaggregationMode.DECODE:
                # Check if bootstrap_info is pre-computed in the request (from frontend)
                bootstrap_info = request.get("bootstrap_info")

                if not bootstrap_info:
                    raise RuntimeError(
                        "bootstrap_info is required for disaggregated decode but was not provided"
                    )

                logging.debug(
                    f"Using bootstrap_info: "
                    f"host={bootstrap_info['bootstrap_host']}, "
                    f"port={bootstrap_info['bootstrap_port']}, "
                    f"room={bootstrap_info['bootstrap_room']}"
                )

                trace_header = (
                    self._get_trace_header(context) if self.enable_trace else None
                )

                # Extract dp_rank from routing info (set by KV router)
                routing = request.get("routing") or {}
                dp_rank = routing.get("dp_rank")

                decode = await self.engine.async_generate(
                    **input_param,
                    sampling_params=sampling_params,
                    stream=True,
                    return_routed_experts=return_routed_experts,
                    bootstrap_host=bootstrap_info["bootstrap_host"],
                    bootstrap_port=bootstrap_info["bootstrap_port"],
                    bootstrap_room=bootstrap_info["bootstrap_room"],
                    external_trace_header=trace_header,
                    rid=trace_id,
                    data_parallel_rank=dp_rank,
                    **self._priority_kwargs(priority),
                )

                if not self.use_sglang_text_io:
                    async for out in self._process_token_stream(
                        decode, context, release_request_slot_once
                    ):
                        yield out
                else:
                    async for out in self._process_text_stream(
                        decode,
                        context,
                        release_request_slot_once,
                        split_reasoning=split_reasoning,
                    ):
                        yield out
            else:
                # Extract image URLs for multimodal requests. SGLang's mm_data_processor
                # handles loading/preprocessing, and the scheduler does vision encoding.
                image_data = None
                image_items = request.get("multi_modal_data", {}).get("image_url")
                if image_items:
                    image_data = []
                    for item in image_items:
                        if isinstance(item, str):
                            image_data.append(item)
                        elif isinstance(item, dict) and "Url" in item:
                            image_data.append(item["Url"])
                    image_data = image_data or None

                trace_header = (
                    self._get_trace_header(context) if self.enable_trace else None
                )

                # Extract dp_rank from routing info (set by KV router)
                routing = request.get("routing") or {}
                dp_rank = routing.get("dp_rank")

                agg = await self.engine.async_generate(
                    **input_param,
                    image_data=image_data,
                    sampling_params=sampling_params,
                    stream=True,
                    return_routed_experts=return_routed_experts,
                    external_trace_header=trace_header,
                    rid=trace_id,
                    data_parallel_rank=dp_rank,
                    **self._priority_kwargs(priority),
                )
                if not self.use_sglang_text_io:
                    async for out in self._process_token_stream(
                        agg, context, release_request_slot_once
                    ):
                        yield out
                else:
                    async for out in self._process_text_stream(
                        agg,
                        context,
                        release_request_slot_once,
                        split_reasoning=split_reasoning,
                    ):
                        yield out
        finally:
            await release_request_slot_once()

    async def _process_token_stream(
        self,
        stream_source: AsyncGenerator[Dict[str, Any], None],
        context: Context,
        release_request_slot_once: Callable[[], Awaitable[None]],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process token-based stream output.

        With stream_output=True (enforced by Dynamo), SGLang sends disjoint segments
        containing only new tokens since the last output. We pass these through directly.

        Args:
            stream_source: Async generator from engine.async_generate.
            context: Context object for cancellation handling.

        Yields:
            Dict with token_ids and optional finish_reason.
        """
        # Use Future pattern for request ID - will be set when first response arrives
        request_id_future = asyncio.Future()
        async with self._cancellation_monitor(request_id_future, context):
            async for res in stream_source:
                # Extract SGLang request ID from the first response and set the future
                if not request_id_future.done():
                    meta_info = res.get("meta_info", {})
                    sglang_request_id = meta_info.get("id")
                    if sglang_request_id:
                        request_id_future.set_result(sglang_request_id)
                        logging.debug(f"New SGLang Request ID: {sglang_request_id}")

                # Check cancellation before yielding to allow proper cleanup.
                # This lets SGLang proceed to the second token generation, which will
                # async context switch and allow the abort monitor to signal cancellation.
                # The loop should exit by itself when context.is_stopped() returns True.
                out = {}
                finish_reason = res["meta_info"]["finish_reason"]
                if finish_reason:
                    out["finish_reason"] = normalize_finish_reason(
                        finish_reason["type"]
                    )

                # With stream_output=True, output_ids contains only new tokens (disjoint)
                output_ids = res.get("output_ids", [])
                # Empty, non-final chunks can happen during scheduler idle ticks.
                # Keep waiting for the next chunk unless cancellation was requested.
                if not output_ids and not finish_reason:
                    if context.is_stopped():
                        break
                    continue

                # Pass through disjoint token segments directly
                out["token_ids"] = output_ids
                routed_experts = res["meta_info"].get("routed_experts")
                if routed_experts is not None:
                    # Base64-encode tensor bytes to match sglang's output format.
                    routed_experts = pybase64.b64encode(
                        routed_experts.numpy().tobytes()
                    ).decode("utf-8")
                    # Internal transport field consumed by frontend nvext mapping.
                    out["disaggregated_params"] = {"routed_experts": routed_experts}
                if finish_reason:
                    input_tokens = res["meta_info"]["prompt_tokens"]
                    completion_tokens = res["meta_info"]["completion_tokens"]
                    cached_tokens = res["meta_info"]["cached_tokens"]
                    prefill_prompt_tokens_details = None
                    if cached_tokens is not None and cached_tokens > 0:
                        prefill_prompt_tokens_details = {"cached_tokens": cached_tokens}
                    out["completion_usage"] = {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": input_tokens + completion_tokens,
                        "prompt_tokens_details": prefill_prompt_tokens_details,
                    }
                if not context.is_stopped():
                    yield out

    async def _process_text_stream(
        self,
        stream_source: AsyncGenerator[Dict[str, Any], None],
        context: Context,
        release_request_slot_once: Callable[[], Awaitable[None]],
        *,
        split_reasoning: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process text-based stream output in OpenAI format.

        Args:
            stream_source: Async generator from engine.async_generate.
            context: Context object for cancellation handling.

        Yields:
            OpenAI-formatted chat completion chunk dicts.
        """
        count = 0
        reasoning_state: Dict[str, Any] = {"done": not split_reasoning, "pending": ""}

        # Use Future pattern for request ID - will be set when first response arrives
        request_id_future = asyncio.Future()
        async with self._cancellation_monitor(request_id_future, context):
            async for res in stream_source:
                # Extract SGLang request ID from the first response and set the future
                if not request_id_future.done():
                    meta_info = res.get("meta_info", {})
                    sglang_request_id = meta_info.get("id")
                    if sglang_request_id:
                        request_id_future.set_result(sglang_request_id)
                        logging.debug(f"New SGLang Request ID: {sglang_request_id}")

                # Check cancellation before yielding to allow proper cleanup.
                # This lets SGLang proceed to the second token generation, which will
                # async context switch and allow the abort monitor to signal cancellation.
                # The loop should exit by itself when context.is_stopped() returns True.

                index = res.get("index", 0)
                text = res.get("text", "")

                finish_reason = res["meta_info"]["finish_reason"]
                finish_reason_type = (
                    normalize_finish_reason(finish_reason["type"])
                    if finish_reason
                    else None
                )
                next_count = len(text)
                delta = text[count:]
                delta_payload = (
                    self._split_qwen_reasoning_delta(
                        delta, reasoning_state, flush=bool(finish_reason)
                    )
                    if split_reasoning
                    else ({"content": delta} if delta else {})
                )
                if not delta_payload and finish_reason:
                    delta_payload = {}

                choice_data = {
                    "index": index,
                    "delta": {"role": "assistant", **delta_payload},
                    "finish_reason": finish_reason_type,
                }

                response = {
                    "id": res["meta_info"]["id"],
                    "created": int(time.time()),
                    "choices": [choice_data],
                    "model": self.config.server_args.served_model_name,
                    "object": "chat.completion.chunk",
                }
                routed_experts = res["meta_info"].get("routed_experts")
                if routed_experts is not None:
                    # Base64-encode tensor bytes to match sglang's output format.
                    routed_experts = pybase64.b64encode(
                        routed_experts.numpy().tobytes()
                    ).decode("utf-8")
                    response["nvext"] = {"routed_experts": routed_experts}
                if not context.is_stopped():
                    yield response
                count = next_count
