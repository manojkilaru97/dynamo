# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
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


@dataclass
class _RequestAdmission:
    context_id: str
    created_at: float
    last_progress_at: float
    sglang_request_id: Optional[str] = None
    dp_rank: Optional[int] = None


class DecodeWorkerHandler(BaseWorkerHandler):
    """Handler for decode workers in both aggregated and disaggregated serving modes."""

    SERVICE_OVERLOADED_ERROR_TYPE = "service_overloaded"
    THINK_START = "<think>"
    THINK_END = "</think>"
    DEFAULT_XGRAMMAR_MAX_STRING_LENGTH = int(
        os.getenv("DYN_XGRAMMAR_DEFAULT_MAX_STRING_LENGTH", "4096")
    )
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
        )
        self.max_total_requests_per_dp = self._get_default_max_total_requests_per_dp()
        self.request_slot_lease_secs = self._get_request_slot_lease_secs()
        self.stale_full_unhealthy_secs = self._get_stale_full_unhealthy_secs()
        self._request_admission_lock = asyncio.Lock()
        self._active_request_admissions = 0
        self._active_request_admissions_high_water = 0
        self._request_admissions: dict[str, _RequestAdmission] = {}
        self._request_admission_dp_counts: dict[int, int] = {}
        self._request_admission_dp_high_water: dict[int, int] = {}
        self._request_slots_reaped_total = 0
        self._last_stream_progress_at = time.monotonic()

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

    def _get_default_max_total_requests_per_dp(self) -> Optional[int]:
        configured = self._get_positive_int_env("DYN_REQUEST_MAX_TOTAL_REQUESTS_PER_DP")
        if configured is not None:
            return configured

        return None

    @classmethod
    def _get_positive_float_env(cls, name: str) -> Optional[float]:
        value = os.environ.get(name)
        if value in (None, ""):
            return None
        try:
            parsed = float(value)
        except ValueError:
            logging.warning("Ignoring invalid %s=%r", name, value)
            return None
        if parsed <= 0:
            return None
        return parsed

    @classmethod
    def _get_request_slot_lease_secs(cls) -> Optional[float]:
        configured = cls._get_positive_float_env("DYN_REQUEST_SLOT_LEASE_SECS")
        if configured is not None:
            return configured

        configured = cls._get_positive_float_env(
            "DYN_REQUEST_MAX_DECODE_WALL_CLOCK_SECS"
        )
        if configured is not None:
            return configured

        # Admission slots are a protection mechanism, so they also need a
        # finite lease. Without this, one wedged async stream can leave a worker
        # permanently stuck at "local total request limit: N/N".
        return 600.0

    @classmethod
    def _get_stale_full_unhealthy_secs(cls) -> Optional[float]:
        configured = cls._get_positive_float_env(
            "DYN_SGLANG_STALE_FULL_UNHEALTHY_SECS"
        )
        if configured is not None:
            return configured
        return 60.0

    @staticmethod
    def _extract_multimodal_urls(
        request: Dict[str, Any], key: str
    ) -> Optional[list[Any]]:
        """Extract frontend-provided multimodal URL payloads for SGLang."""
        items = (request.get("multi_modal_data") or {}).get(key)
        if not items:
            return None
        urls: list[Any] = []
        for item in items:
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, dict):
                for item_key in ("Url", "url", "Decoded", "decoded"):
                    if item_key in item:
                        urls.append(item[item_key])
                        break
        return urls or None

    @staticmethod
    def _is_health_check_request(request: Dict[str, Any]) -> bool:
        annotations = request.get("annotations") or []
        if not isinstance(annotations, list):
            return False
        return any(
            isinstance(annotation, dict)
            and annotation.get("dynamo_health_check") is True
            for annotation in annotations
        )

    def _stale_full_unhealthy_reason_locked(self, now: float) -> str | None:
        limit = self.max_total_requests
        stale_after = self.stale_full_unhealthy_secs
        if limit is None or stale_after is None:
            return None

        active = len(self._request_admissions)
        if active < limit:
            return None

        progress_age = now - self._last_stream_progress_at
        if progress_age < stale_after:
            return None

        oldest_age = max(
            (now - admission.created_at for admission in self._request_admissions.values()),
            default=0.0,
        )
        return (
            "Dynamo SGLang worker unhealthy: local request slots are full "
            f"({active}/{limit}) and no SGLang stream progress for "
            f"{progress_age:.1f}s (oldest_slot_age={oldest_age:.1f}s)"
        )

    def _reap_expired_request_slots_locked(
        self, now: float
    ) -> list[_RequestAdmission]:
        lease_secs = self.request_slot_lease_secs
        if lease_secs is None:
            return []

        expired: list[_RequestAdmission] = []
        for context_id, admission in list(self._request_admissions.items()):
            if now - admission.last_progress_at < lease_secs:
                continue
            expired.append(admission)
            self._request_admissions.pop(context_id, None)

        if expired:
            for admission in expired:
                if admission.dp_rank is not None:
                    current = self._request_admission_dp_counts.get(admission.dp_rank, 0)
                    if current <= 1:
                        self._request_admission_dp_counts.pop(admission.dp_rank, None)
                    else:
                        self._request_admission_dp_counts[admission.dp_rank] = current - 1
            self._request_slots_reaped_total += len(expired)
            self._active_request_admissions = len(self._request_admissions)
            oldest_age = max(now - admission.created_at for admission in expired)
            longest_idle = max(now - admission.last_progress_at for admission in expired)
            logging.warning(
                "Reaped %s stale SGLang admission slot(s); active=%s/%s "
                "oldest_age=%.1fs longest_idle=%.1fs lease=%.1fs "
                "reaped_total=%s dp_counts=%s",
                len(expired),
                self._active_request_admissions,
                self.max_total_requests,
                oldest_age,
                longest_idle,
                lease_secs,
                self._request_slots_reaped_total,
                self._format_request_admission_dp_counts_locked(),
            )
        return expired

    def _format_request_admission_dp_counts_locked(self) -> dict[int, int]:
        return {
            dp_rank: self._request_admission_dp_counts[dp_rank]
            for dp_rank in sorted(self._request_admission_dp_counts)
        }

    def _abort_reaped_request_slot(self, admission: _RequestAdmission) -> None:
        request_id = admission.sglang_request_id
        if not request_id:
            return
        tokenizer_manager = getattr(self.engine, "tokenizer_manager", None)
        if tokenizer_manager is None:
            logging.warning(
                "Cannot abort stale SGLang request %s for context %s: "
                "tokenizer_manager missing",
                request_id,
                admission.context_id,
            )
            return
        try:
            tokenizer_manager.abort_request(rid=request_id, abort_all=False)
            logging.warning(
                "Aborted stale SGLang request %s for context %s after %.1fs",
                request_id,
                admission.context_id,
                time.monotonic() - admission.created_at,
            )
        except Exception as exc:
            logging.warning(
                "Failed to abort stale SGLang request %s for context %s: %s",
                request_id,
                admission.context_id,
                exc,
            )

    async def _try_reserve_request_slot(
        self,
        request_id: str,
        sglang_request_id: str | None = None,
        *,
        health_check: bool = False,
        dp_rank: int | None = None,
    ) -> tuple[bool, int | None, int | None, str | None, str | None]:
        limit = self.max_total_requests
        if limit is None:
            return True, None, None, None, None

        expired: list[_RequestAdmission] = []
        reject_result: tuple[bool, int | None, int | None, str | None, str | None] | None = None
        async with self._request_admission_lock:
            now = time.monotonic()
            expired = self._reap_expired_request_slots_locked(now)
            if health_check and expired:
                current_total = len(self._request_admissions)
                unhealthy_reason = (
                    "Dynamo SGLang worker unhealthy: health check reaped "
                    f"{len(expired)} stale admission slot(s) "
                    f"(active={current_total}/{limit}, lease={self.request_slot_lease_secs}s)"
                )
                reject_result = (
                    False,
                    current_total,
                    limit,
                    unhealthy_reason,
                    "worker",
                )
            existing = self._request_admissions.get(request_id)
            if reject_result is None and existing is not None:
                if sglang_request_id:
                    existing.sglang_request_id = sglang_request_id
                existing.last_progress_at = now
                if existing.dp_rank != dp_rank:
                    old_dp_rank = existing.dp_rank
                    if old_dp_rank is not None:
                        current = self._request_admission_dp_counts.get(
                            old_dp_rank, 0
                        )
                        if current <= 1:
                            self._request_admission_dp_counts.pop(
                                old_dp_rank, None
                            )
                        else:
                            self._request_admission_dp_counts[old_dp_rank] = current - 1
                    existing.dp_rank = dp_rank
                    if dp_rank is not None:
                        self._request_admission_dp_counts[dp_rank] = (
                            self._request_admission_dp_counts.get(dp_rank, 0) + 1
                        )
                    logging.warning(
                        "Duplicate SGLang admission for request %s moved from DP %s to DP %s",
                        request_id,
                        old_dp_rank,
                        dp_rank,
                    )
                self._active_request_admissions = len(self._request_admissions)
                reject_result = (
                    True,
                    self._active_request_admissions,
                    limit,
                    None,
                    None,
                )
            if reject_result is None:
                current_total = len(self._request_admissions)
                per_dp_limit = self.max_total_requests_per_dp
                current_dp_total = (
                    self._request_admission_dp_counts.get(dp_rank, 0)
                    if dp_rank is not None
                    else 0
                )
                if (
                    dp_rank is not None
                    and per_dp_limit is not None
                    and current_dp_total >= per_dp_limit
                ):
                    logging.info(
                        "Rejecting request %s due to local DP request limit: dp_rank=%s %s/%s",
                        request_id,
                        dp_rank,
                        current_dp_total,
                        per_dp_limit,
                    )
                    reject_result = (
                        False,
                        current_dp_total,
                        per_dp_limit,
                        None,
                        "dp",
                    )
                elif current_total >= limit:
                    unhealthy_reason = (
                        self._stale_full_unhealthy_reason_locked(now)
                        if health_check
                        else None
                    )
                    logging.info(
                        "Rejecting request %s due to local total request limit: %s/%s",
                        request_id,
                        current_total,
                        limit,
                    )
                    reject_result = (
                        False,
                        current_total,
                        limit,
                        unhealthy_reason,
                        "worker",
                    )
                else:
                    self._request_admissions[request_id] = _RequestAdmission(
                        context_id=request_id,
                        created_at=now,
                        last_progress_at=now,
                        sglang_request_id=sglang_request_id,
                        dp_rank=dp_rank,
                    )
                    if dp_rank is not None:
                        self._request_admission_dp_counts[dp_rank] = (
                            current_dp_total + 1
                        )
                        if (
                            self._request_admission_dp_counts[dp_rank]
                            > self._request_admission_dp_high_water.get(dp_rank, 0)
                        ):
                            self._request_admission_dp_high_water[dp_rank] = (
                                self._request_admission_dp_counts[dp_rank]
                            )
                            logging.info(
                                "Worker local DP request slots in use: "
                                "dp_rank=%s %s/%s total=%s/%s dp_counts=%s",
                                dp_rank,
                                self._request_admission_dp_counts[dp_rank],
                                self.max_total_requests_per_dp,
                                self._active_request_admissions,
                                limit,
                                self._format_request_admission_dp_counts_locked(),
                            )
                    self._active_request_admissions = len(self._request_admissions)
                    if (
                        self._active_request_admissions
                        > self._active_request_admissions_high_water
                    ):
                        self._active_request_admissions_high_water = (
                            self._active_request_admissions
                        )
                        logging.info(
                            "Worker local total request slots in use: %s/%s dp_counts=%s",
                            self._active_request_admissions,
                            limit,
                            self._format_request_admission_dp_counts_locked(),
                        )

        for admission in expired:
            self._abort_reaped_request_slot(admission)
        if reject_result is not None:
            return reject_result
        return True, current_total + 1, limit, None, None

    async def _release_request_slot_reservation(
        self, request_id: str | None = None
    ) -> None:
        async with self._request_admission_lock:
            admission = None
            if request_id is not None:
                admission = self._request_admissions.pop(request_id, None)
            elif self._request_admissions:
                admission = self._request_admissions.pop(next(iter(self._request_admissions)))
            if admission is not None and admission.dp_rank is not None:
                current = self._request_admission_dp_counts.get(admission.dp_rank, 0)
                if current <= 1:
                    self._request_admission_dp_counts.pop(admission.dp_rank, None)
                else:
                    self._request_admission_dp_counts[admission.dp_rank] = current - 1
            self._active_request_admissions = len(self._request_admissions)

    async def _record_request_slot_sglang_id(
        self, context_id: str, sglang_request_id: str
    ) -> None:
        async with self._request_admission_lock:
            admission = self._request_admissions.get(context_id)
            if admission is not None:
                admission.sglang_request_id = sglang_request_id
                now = time.monotonic()
                admission.last_progress_at = now
                self._last_stream_progress_at = now

    async def _touch_request_slot(self, context_id: str) -> None:
        async with self._request_admission_lock:
            admission = self._request_admissions.get(context_id)
            if admission is not None:
                now = time.monotonic()
                admission.last_progress_at = now
                self._last_stream_progress_at = now

    @classmethod
    def _build_overload_extra_args(
        cls,
        message: str,
        *,
        current_total_requests: int | None = None,
        total_request_limit: int | None = None,
        request_limit_scope: str | None = None,
    ) -> dict[str, Any]:
        extra_args: dict[str, Any] = {
            "dynamo_error_type": cls.SERVICE_OVERLOADED_ERROR_TYPE,
            "error_message": message,
        }
        if current_total_requests is not None:
            extra_args["worker_total_requests"] = current_total_requests
        if total_request_limit is not None:
            extra_args["worker_total_request_limit"] = total_request_limit
        if request_limit_scope is not None:
            extra_args["worker_request_limit_scope"] = request_limit_scope
        return extra_args

    @classmethod
    def _build_token_overload_response(
        cls,
        message: str,
        *,
        current_total_requests: int | None = None,
        total_request_limit: int | None = None,
        request_limit_scope: str | None = None,
    ) -> dict[str, Any]:
        return {
            "finish_reason": {"error": message},
            "token_ids": [],
            "extra_args": cls._build_overload_extra_args(
                message,
                current_total_requests=current_total_requests,
                total_request_limit=total_request_limit,
                request_limit_scope=request_limit_scope,
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
        request_limit_scope: str | None = None,
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
                request_limit_scope=request_limit_scope,
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

    @classmethod
    def _openai_structured_request_to_guided(cls, request: Dict[str, Any]) -> Dict[str, Any]:
        structured_outputs = request.get("structured_outputs")
        if isinstance(structured_outputs, dict):
            guided = {
                key: value
                for key, value in structured_outputs.items()
                if key
                in {
                    "choice",
                    "grammar",
                    "json",
                    "json_object",
                    "regex",
                    "structural_tag",
                }
                and value is not None
                and value is not False
            }
            if guided:
                return guided

        response_format = request.get("response_format")
        if not isinstance(response_format, dict):
            return {}

        response_type = response_format.get("type")
        if response_type == "json_object":
            return {"json_object": True}
        if response_type != "json_schema":
            return {}

        json_schema = response_format.get("json_schema") or {}
        schema = json_schema.get("schema")
        if schema is None:
            return {}
        return {"json": schema}

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
            guided_decoding = request.get("guided_decoding")
            if guided_decoding is None:
                guided_decoding = self._openai_structured_request_to_guided(request)
            param_mapping = {
                "presence_penalty": request.get("presence_penalty"),
                "frequency_penalty": request.get("frequency_penalty"),
                "repetition_penalty": request.get("repetition_penalty"),
                "temperature": request.get("temperature"),
                "top_p": request.get("top_p"),
                "top_k": request.get("top_k"),
                "min_p": request.get("min_p"),
                "max_new_tokens": request.get("max_tokens"),
                "ignore_eos": request.get("ignore_eos"),
            }
            param_mapping.update(self._guided_to_sglang_params(guided_decoding))

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
        routing = request.get("routing") or {}
        dp_rank = routing.get("dp_rank")
        try:
            dp_rank = int(dp_rank) if dp_rank is not None else None
        except (TypeError, ValueError):
            dp_rank = None

        (
            reserved,
            current_total_requests,
            total_request_limit,
            unhealthy_reason,
            request_limit_scope,
        ) = (
            await self._try_reserve_request_slot(
                context.id(),
                trace_id,
                health_check=self._is_health_check_request(request),
                dp_rank=dp_rank,
            )
        )
        if not reserved:
            if unhealthy_reason:
                logging.error(unhealthy_reason)
                raise RuntimeError(unhealthy_reason)
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
                    request_limit_scope=request_limit_scope,
                )
            else:
                yield self._build_token_overload_response(
                    message,
                    current_total_requests=current_total_requests,
                    total_request_limit=total_request_limit,
                    request_limit_scope=request_limit_scope,
                )
            return

        slot_released = False

        async def release_request_slot_once() -> None:
            nonlocal slot_released
            if slot_released:
                return
            slot_released = True
            await self._release_request_slot_reservation(context.id())

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
                        decode,
                        context,
                        release_request_slot_once,
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
                # Pass frontend-extracted multimodal URLs through to SGLang.
                # SGLang's mm_data_processor handles loading/preprocessing, and
                # the scheduler performs the vision/video encoding.
                image_data = self._extract_multimodal_urls(request, "image_url")
                video_data = self._extract_multimodal_urls(request, "video_url")

                trace_header = (
                    self._get_trace_header(context) if self.enable_trace else None
                )

                agg = await self.engine.async_generate(
                    **input_param,
                    image_data=image_data,
                    video_data=video_data,
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
                        agg,
                        context,
                        release_request_slot_once,
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
                        await self._record_request_slot_sglang_id(
                            context.id(), sglang_request_id
                        )
                        logging.debug(f"New SGLang Request ID: {sglang_request_id}")
                await self._touch_request_slot(context.id())

                # Check cancellation before yielding to allow proper cleanup.
                # This lets SGLang proceed to the second token generation, which will
                # async context switch and allow the abort monitor to signal cancellation.
                # The loop should exit by itself when context.is_stopped() returns True.
                out = {}
                meta_info = res.get("meta_info", {})
                finish_reason = meta_info.get("finish_reason")
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
                routed_experts = meta_info.get("routed_experts")
                if routed_experts is not None:
                    # Base64-encode tensor bytes to match sglang's output format.
                    routed_experts = pybase64.b64encode(
                        routed_experts.numpy().tobytes()
                    ).decode("utf-8")
                    # Internal transport field consumed by frontend nvext mapping.
                    out["disaggregated_params"] = {"routed_experts": routed_experts}
                if finish_reason:
                    input_tokens = meta_info.get("prompt_tokens")
                    completion_tokens = meta_info.get("completion_tokens")
                    if input_tokens is not None and completion_tokens is not None:
                        cached_tokens = meta_info.get("cached_tokens")
                        prefill_prompt_tokens_details = None
                        if cached_tokens is not None and cached_tokens > 0:
                            prefill_prompt_tokens_details = {
                                "cached_tokens": cached_tokens
                            }
                        out["completion_usage"] = {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": input_tokens + completion_tokens,
                            "prompt_tokens_details": prefill_prompt_tokens_details,
                        }
                    await release_request_slot_once()
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
                        await self._record_request_slot_sglang_id(
                            context.id(), sglang_request_id
                        )
                        logging.debug(f"New SGLang Request ID: {sglang_request_id}")
                await self._touch_request_slot(context.id())

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
                if finish_reason:
                    await release_request_slot_once()
                if not context.is_stopped():
                    yield response
                count = next_count
