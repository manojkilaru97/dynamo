#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.reasoning import ReasoningParser
from vllm.renderers import ChatParams
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.utils.async_utils import make_async

logger = logging.getLogger(__name__)


class _Renderer(Protocol):
    """Structural type for vLLM's chat-template renderer."""

    async def render_messages_async(
        self, messages: Any, params: ChatParams
    ) -> tuple[Any, dict[str, Any]]:
        ...


@dataclass
class PreprocessResult:
    request_for_sampling: ChatCompletionRequest
    tool_parser: ToolParser | None
    chat_template_kwargs: dict[str, Any]
    engine_prompt: dict[str, Any]
    prompt_token_ids: list[int]


_ASYNC_TOKENIZER_POOL: dict[int, Callable[..., Awaitable[Any]]] = {}
SKIP_REQUEST_VALIDATION = os.getenv("DYN_VLLM_SKIP_REQUEST_VALIDATION", "1") == "1"


def _get_async_tokenizer(tokenizer: TokenizerLike) -> Callable[..., Awaitable[Any]]:
    key = id(tokenizer)
    async_tokenizer = _ASYNC_TOKENIZER_POOL.get(key)
    if async_tokenizer is None:
        async_tokenizer = make_async(
            tokenizer, executor=ThreadPoolExecutor(max_workers=1)
        )
        _ASYNC_TOKENIZER_POOL[key] = async_tokenizer
    return async_tokenizer


def _materialize_assistant_tool_calls(
    messages: Sequence[Any],
) -> list[dict[str, Any] | Any]:
    # Mistral chat templating expects assistant tool_calls to be materialized
    # as a concrete list of dict-like values. Our validated message models may
    # still carry non-list sequence-like containers here, which can break or
    # mis-render when tokenize=True is used in-template. This helper converts
    # model objects to dicts and normalizes assistant.tool_calls to list when
    # possible, while preserving original values if they are not iterable.
    normalized: list[dict[str, Any] | Any] = []
    for message in messages:
        if hasattr(message, "model_dump"):
            msg: dict[str, Any] | Any = message.model_dump(exclude_none=False)
        else:
            msg = message

        if isinstance(msg, dict) and msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls is not None and not isinstance(tool_calls, list):
                try:
                    msg["tool_calls"] = list(tool_calls)
                except TypeError:
                    # Keep original object if it is not iterable.
                    pass

        normalized.append(msg)
    return normalized


def _resolve_chat_template_kwargs(
    request: ChatCompletionRequest,
) -> dict[str, Any]:
    resolver = getattr(request, "get_resolved_chat_template_kwargs", None)
    if callable(resolver):
        kwargs = dict(resolver())
    else:
        kwargs = dict(request.chat_template_kwargs or {})
    if "thinking" in kwargs and "enable_thinking" not in kwargs:
        kwargs["enable_thinking"] = kwargs["thinking"]
    if any(
        key in kwargs
        for key in ("enable_thinking", "thinking", "low_effort", "medium_effort")
    ):
        return kwargs

    effort = request.reasoning_effort
    if effort == "none":
        kwargs.setdefault("enable_thinking", False)
    elif effort in ("minimal", "low", "medium"):
        kwargs.setdefault("enable_thinking", True)
        kwargs.setdefault("low_effort", True)
        kwargs.setdefault("medium_effort", True)
    elif effort in ("high", "xhigh", "max"):
        kwargs.setdefault("enable_thinking", True)

    return kwargs


def _has_user_structured_output_constraint(request: ChatCompletionRequest) -> bool:
    structured_outputs = getattr(request, "structured_outputs", None)
    if structured_outputs is None:
        model_extra = getattr(request, "model_extra", None)
        if isinstance(model_extra, dict):
            structured_outputs = model_extra.get("structured_outputs")
    if StreamingPostProcessor._structured_outputs_has_json(structured_outputs):
        return True
    if structured_outputs is not None:
        if isinstance(structured_outputs, dict):
            if any(
                structured_outputs.get(key) is not None
                for key in ("regex", "choice", "grammar", "structural_tag")
            ):
                return True
        elif not structured_outputs.all_constraints_none():
            return True

    response_format = getattr(request, "response_format", None)
    if response_format is None:
        return False
    response_type = (
        response_format.get("type")
        if isinstance(response_format, dict)
        else getattr(response_format, "type", None)
    )
    return response_type in {"json_schema", "json_object", "structural_tag"}


def _value_from_mapping_or_object(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _named_tool_choice_name(tool_choice: Any) -> str | None:
    function = _value_from_mapping_or_object(tool_choice, "function")
    if function is None:
        return None
    return _value_from_mapping_or_object(function, "name")


def _prepare_request(
    request: dict[str, Any] | ChatCompletionRequest,
    *,
    tokenizer: TokenizerLike,
    tool_parser_class: type[ToolParser] | None,
    exclude_tools_when_tool_choice_none: bool = True,
    enable_auto_tool_choice: bool = False,
) -> tuple[ChatCompletionRequest, ToolParser | None, dict[str, Any], Any, ChatParams]:
    """Validate request and build arguments for template rendering.

    Returns:
        request_for_sampling: Validated ChatCompletionRequest.
        tool_parser: Instantiated tool parser, or None.
        chat_template_kwargs: Template kwargs (for PreprocessResult).
        messages_for_render: Messages to pass as first arg to render_messages.
        chat_params: ChatParams for render_messages / render_messages_async.
    """
    if (
        isinstance(request, dict)
        and "tool_choice" not in request
        and request.get("tools")
    ):
        request = {**request, "tool_choice": "auto"}

    if isinstance(request, ChatCompletionRequest):
        request_for_sampling = request
    elif SKIP_REQUEST_VALIDATION:
        # Trusted fast path; caller must provide OpenAI-compatible payload.
        request_for_sampling = ChatCompletionRequest.model_construct(**request)
        if request_for_sampling.tools and any(
            not hasattr(tool, "model_dump") for tool in request_for_sampling.tools
        ):
            request_for_sampling = ChatCompletionRequest.model_validate(request)
    else:
        request_for_sampling = ChatCompletionRequest.model_validate(request)

    tool_parser: ToolParser | None = None
    # With enable_auto_tool_choice the model may emit tool calls even when the
    # client did not supply an explicit `tools` list, so we activate the parser
    # whenever the tool_parser_class is available.
    has_tools = bool(request_for_sampling.tools)
    has_structured_output = _has_user_structured_output_constraint(request_for_sampling)
    tool_choice = request_for_sampling.tool_choice
    should_parse_tools = (
        tool_parser_class
        and (has_tools or enable_auto_tool_choice)
        and not has_structured_output
        and tool_choice in (None, "auto")
    )
    should_parse_tools_without_adjustment = (
        tool_parser_class
        and has_tools
        and not has_structured_output
        and tool_choice not in (None, "none", "auto")
    )
    if should_parse_tools or should_parse_tools_without_adjustment:
        tool_parser = tool_parser_class(tokenizer, request_for_sampling.tools)
    if should_parse_tools:
        request_for_sampling = tool_parser.adjust_request(request_for_sampling)

    # Strip tools from the template when tool_choice=none so the model doesn't
    # see them and generate raw XML tool calls in its response.
    tool_dicts = (
        [tool.model_dump() for tool in request_for_sampling.tools]
        if request_for_sampling.tools
        and not (
            exclude_tools_when_tool_choice_none
            and request_for_sampling.tool_choice == "none"
        )
        else None
    )
    chat_template_kwargs = _resolve_chat_template_kwargs(request_for_sampling)
    chat_template_kwargs["reasoning_effort"] = request_for_sampling.reasoning_effort

    # Mistral warns that tokenize=False is unsafe for chat templates.
    is_mistral_tokenizer = (
        tokenizer.__class__.__name__ == "MistralTokenizer"
        or "tokenizers.mistral" in tokenizer.__class__.__module__
    )
    tokenize_in_template = is_mistral_tokenizer
    messages_for_render = (
        _materialize_assistant_tool_calls(request_for_sampling.messages)
        if is_mistral_tokenizer
        else request_for_sampling.messages
    )

    chat_params = ChatParams(
        chat_template=request_for_sampling.chat_template,
        chat_template_content_format="auto",
        chat_template_kwargs=dict(
            add_generation_prompt=request_for_sampling.add_generation_prompt,
            continue_final_message=request_for_sampling.continue_final_message,
            tools=tool_dicts,
            documents=request_for_sampling.documents,
            tokenize=tokenize_in_template,
            **chat_template_kwargs,
        ),
    )

    return (
        request_for_sampling,
        tool_parser,
        chat_template_kwargs,
        messages_for_render,
        chat_params,
    )


async def preprocess_chat_request(
    request: dict[str, Any] | ChatCompletionRequest,
    *,
    tokenizer: TokenizerLike,
    renderer: _Renderer,
    tool_parser_class: type[ToolParser] | None,
    exclude_tools_when_tool_choice_none: bool = True,
    enable_auto_tool_choice: bool = False,
) -> PreprocessResult:
    (
        request_for_sampling,
        tool_parser,
        chat_template_kwargs,
        messages,
        chat_params,
    ) = _prepare_request(
        request,
        tokenizer=tokenizer,
        tool_parser_class=tool_parser_class,
        exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
        enable_auto_tool_choice=enable_auto_tool_choice,
    )

    _, engine_prompt = await renderer.render_messages_async(messages, chat_params)

    if "prompt_token_ids" in engine_prompt:
        tokens = list(engine_prompt["prompt_token_ids"])
    else:
        async_tokenizer = _get_async_tokenizer(tokenizer)
        encoded = await async_tokenizer(
            engine_prompt["prompt"],
            add_special_tokens=request_for_sampling.add_special_tokens,
        )
        tokens = list(encoded.input_ids)

    return PreprocessResult(
        request_for_sampling=request_for_sampling,
        tool_parser=tool_parser,
        chat_template_kwargs=chat_template_kwargs,
        engine_prompt=engine_prompt,
        prompt_token_ids=tokens,
    )


class StreamingPostProcessor:
    def __init__(
        self,
        *,
        tokenizer: TokenizerLike,
        request_for_sampling: ChatCompletionRequest,
        sampling_params: SamplingParams,
        prompt_token_ids: Sequence[int],
        tool_parser: ToolParser | None,
        reasoning_parser_class: type[ReasoningParser] | None,
        chat_template_kwargs: dict[str, Any],
        stream_response: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.request_for_sampling = request_for_sampling
        self.sampling_params = sampling_params
        self.tool_parser = tool_parser
        self.stream_response = stream_response
        # See https://github.com/ai-dynamo/dynamo/issues/8636 —
        # when the chat template runs with enable_thinking=False,
        # the reasoning open/close tags live in the prompt and the generated
        # output carries none — so is_reasoning_end_streaming() never fires,
        # reasoning_is_done stays false, and tool-call markup leaks into
        # reasoning_content. Skip the reasoning parser in that case.
        # `enable_thinking` is the convention adopted across the modern
        # reasoning-capable model families that vLLM supports; templates
        # that don't honor it simply leave it unset (no effect here).
        thinking_disabled = chat_template_kwargs.get("enable_thinking") is False
        self.reasoning_parser = (
            reasoning_parser_class(
                tokenizer,
                chat_template_kwargs=chat_template_kwargs,
            )
            if reasoning_parser_class and not thinking_disabled
            else None
        )
        self._structured_json_guard = self._is_pure_structured_json_request()
        self._structured_tool_call_name = self._structured_tool_call_name_from_request()
        self._structured_required_tool_choice = (
            self._structured_required_tool_choice_from_request()
        )
        if self.tool_parser is not None and not self._structured_outputs_active(
            getattr(self.sampling_params, "structured_outputs", None)
        ):
            self._structured_tool_call_name = None
            self._structured_required_tool_choice = False
        self._fast_plain_text = (
            (
                self.tool_parser is None
                and self.reasoning_parser is None
                and self._structured_tool_call_name is None
                and not self._structured_required_tool_choice
            )
            or self._structured_json_guard
        )
        self._structured_json_buffer = ""
        self._structured_json_emitted = False
        self._structured_tool_json_buffer = ""
        self._structured_tool_json_emitted = False

        self._control_markers = tuple(
            t for t in getattr(tokenizer, "all_special_tokens", ()) if t
        )

        self.previous_text = ""
        self.previous_token_ids: list[int] = []
        self.reasoning_is_done = False
        self.in_progress_tool_calls: dict[int, DeltaToolCall] = {}
        # Per-choice tracking (https://github.com/ai-dynamo/dynamo/issues/8636) of whether a tool_call delta was
        # emitted on that choice, keyed by `output.index`. Required because
        # `n > 1` requests stream multiple choices interleaved; a remap on
        # one choice must not bleed into another. See _remap_finish_reason().
        self._tool_call_choices_emitted: set[int] = set()
        # Buffer for post-reasoning tool text when </think> and <tool_call>
        # arrive in the same chunk.  The streaming tool parser cannot handle
        # this correctly, so we accumulate text here and fall back to the
        # non-streaming extract_tool_calls() once the buffer is complete.
        self._tool_text_buffer: str | None = None
        # Full text seen after reasoning ends. This lets the final chunk fall
        # back to non-streaming extraction if the streaming parser missed a
        # complete tool call split across many small chunks.
        self._post_reasoning_text_buffer = ""
        # Parser-only decoded token stream after reasoning. Some models emit
        # tool-call markers as token IDs while the visible text delta is empty;
        # keep those markers available to the fallback tool parser without
        # changing client-visible content deltas.
        self._post_reasoning_raw_text_buffer = ""
        # Parser-only capture for tool calls that start before the reasoning
        # parser declares </think>. This keeps force_nonempty_content from
        # hiding a valid tool call inside replayed reasoning text.
        self._raw_tool_text_buffer = ""
        self._raw_tool_capture_started = False
        self._reasoning_tool_markup_started = False
        self._tool_marker_ids = self._tool_marker_token_ids()
        self._emitted_reasoning_text = ""

    def _is_pure_structured_json_request(self) -> bool:
        if getattr(self.request_for_sampling, "tools", None):
            return False
        structured_outputs = getattr(self.sampling_params, "structured_outputs", None)
        if self._structured_outputs_has_json(structured_outputs):
            return True
        request_structured_outputs = getattr(
            self.request_for_sampling, "structured_outputs", None
        )
        if request_structured_outputs is None:
            model_extra = getattr(self.request_for_sampling, "model_extra", None)
            if isinstance(model_extra, dict):
                request_structured_outputs = model_extra.get("structured_outputs")
        if self._structured_outputs_has_json(request_structured_outputs):
            return True
        response_format = getattr(self.request_for_sampling, "response_format", None)
        if response_format is None:
            return False
        if isinstance(response_format, dict):
            response_type = response_format.get("type")
        else:
            response_type = getattr(response_format, "type", None)
        return response_type in {"json_schema", "json_object"}

    def _structured_tool_call_name_from_request(self) -> str | None:
        if (
            not getattr(self.request_for_sampling, "tools", None)
            or _has_user_structured_output_constraint(self.request_for_sampling)
            or getattr(self.request_for_sampling, "tool_choice", None) == "required"
        ):
            return None

        return _named_tool_choice_name(
            getattr(self.request_for_sampling, "tool_choice", None)
        )

    def _structured_required_tool_choice_from_request(self) -> bool:
        if (
            not getattr(self.request_for_sampling, "tools", None)
            or _has_user_structured_output_constraint(self.request_for_sampling)
            or getattr(self.request_for_sampling, "tool_choice", None) != "required"
        ):
            return False

        return True

    @staticmethod
    def _structured_outputs_has_json(structured_outputs: Any) -> bool:
        if structured_outputs is None:
            return False
        if isinstance(structured_outputs, dict):
            return (
                structured_outputs.get("json") is not None
                or structured_outputs.get("json_object") is not None
            )
        return (
            getattr(structured_outputs, "json", None) is not None
            or getattr(structured_outputs, "json_object", None) is not None
        )

    @staticmethod
    def _structured_outputs_active(structured_outputs: Any) -> bool:
        if structured_outputs is None:
            return False
        fields = ("json", "regex", "choice", "grammar", "json_object", "structural_tag")
        if isinstance(structured_outputs, dict):
            return any(structured_outputs.get(field) is not None for field in fields)
        return any(
            getattr(structured_outputs, field, None) is not None for field in fields
        )

    @property
    def structured_json_complete(self) -> bool:
        return self._structured_json_guard and self._structured_json_emitted

    @property
    def structured_tool_complete(self) -> bool:
        return self._structured_tool_json_emitted

    def _structured_json_delta(self, delta_text: str, *, finished: bool) -> str | None:
        if self._structured_json_emitted:
            return None
        self._structured_json_buffer += delta_text
        if not finished:
            return None
        candidate = self._structured_json_buffer.lstrip()
        if not candidate:
            return None
        try:
            _, end = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            return None
        if candidate[end:].strip():
            return None
        self._structured_json_emitted = True
        return candidate[:end]

    def _structured_tool_json_delta(
        self, delta_text: str, *, finished: bool
    ) -> str | None:
        if self._structured_tool_json_emitted:
            return None
        self._structured_tool_json_buffer += delta_text
        if not finished:
            return None
        candidate = self._structured_tool_json_buffer.lstrip()
        if not candidate:
            return None
        try:
            _, end = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            return None
        if candidate[end:].strip():
            return None
        self._structured_tool_json_emitted = True
        return candidate[:end]

    def _should_buffer_for_non_streaming_tool_parse(self) -> bool:
        return (
            not self.stream_response
            and self.tool_parser is not None
            and bool(getattr(self.request_for_sampling, "tools", None))
            and self.request_for_sampling.tool_choice != "none"
        )

    @staticmethod
    def _merge_tool_call(
        existing: DeltaToolCall | None, incoming: DeltaToolCall
    ) -> DeltaToolCall:
        if existing is None:
            if incoming.function and incoming.function.arguments is None:
                incoming.function.arguments = ""
            return incoming
        if incoming.id and not existing.id:
            existing.id = incoming.id
        if incoming.type and not existing.type:
            existing.type = incoming.type
        if incoming.function:
            if existing.function is None:
                existing.function = incoming.function
                if existing.function.arguments is None:
                    existing.function.arguments = ""
            else:
                if incoming.function.name and not existing.function.name:
                    existing.function.name = incoming.function.name
                if incoming.function.arguments:
                    if existing.function.arguments is None:
                        existing.function.arguments = ""
                    existing.function.arguments += incoming.function.arguments
        return existing

    def _is_control_only_content(self, content: str | None) -> bool:
        if not content:
            return True
        stripped = content
        for marker in self._control_markers:
            stripped = stripped.replace(marker, "")
        return stripped.strip() == ""

    def _should_parse_tools(self) -> bool:
        return (
            self.tool_parser is not None
            and bool(getattr(self.request_for_sampling, "tools", None))
            and self.request_for_sampling.tool_choice != "none"
            and self._structured_tool_call_name is None
            and not self._structured_required_tool_choice
        )

    def _decode_token_ids_for_parser(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        try:
            return self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
            )
        except TypeError:
            return self.tokenizer.decode(list(token_ids))
        except Exception:
            try:
                tokens = self.tokenizer.convert_ids_to_tokens(list(token_ids))
                return self.tokenizer.convert_tokens_to_string(tokens)
            except Exception:
                return ""

    def _buffer_tool_calls_until_finish(self) -> bool:
        if self.tool_parser is None:
            return False
        tool_call_start = self._tool_call_start_token()
        tool_call_end = self._tool_call_end_token()
        return tool_call_start == "<tool_call>" and tool_call_end == "</tool_call>"

    def _tool_call_marker(self, attr: str, engine_id_attr: str) -> str | None:
        marker = getattr(self.tool_parser, attr, None)
        if marker:
            return marker
        engine = getattr(self.tool_parser, "_parser_engine", None)
        token_id = getattr(engine, engine_id_attr, None)
        if not isinstance(token_id, int):
            return None
        return self._decode_token_ids_for_parser([token_id]) or None

    def _tool_call_start_token(self) -> str | None:
        marker = self._tool_call_marker(
            "tool_call_start_token",
            "_tool_call_token_id",
        )
        if marker:
            return marker
        bot_token = getattr(self.tool_parser, "bot_token", None)
        if isinstance(bot_token, str) and bot_token:
            return bot_token
        if self._has_single_token_tool_marker("<tool_call>"):
            return "<tool_call>"
        return None

    def _tool_call_end_token(self) -> str | None:
        marker = self._tool_call_marker(
            "tool_call_end_token",
            "_tool_call_end_token_id",
        )
        if marker:
            return marker
        if self._has_single_token_tool_marker("</tool_call>"):
            return "</tool_call>"
        return None

    def _has_single_token_tool_marker(self, marker: str) -> bool:
        try:
            token_ids = self.tokenizer.encode(
                marker,
                add_special_tokens=False,
            )
        except Exception:
            return False
        return (
            len(token_ids) == 1
            and self._decode_token_ids_for_parser(token_ids) == marker
        )

    def _strip_tool_markup_from_reasoning(
        self, delta_message: DeltaMessage | None
    ) -> None:
        if (
            delta_message is None
            or not delta_message.reasoning
            or not self._tool_markup_filter_enabled()
        ):
            return
        if self._reasoning_tool_markup_started:
            delta_message.reasoning = None
            return
        tool_call_start = self._tool_call_start_token()
        marker_offsets = [
            delta_message.reasoning.find(marker)
            for marker in (
                tool_call_start,
                "<function=",
                "<parameter=",
            )
            if marker and marker in delta_message.reasoning
        ]
        closing_offset = delta_message.reasoning.find("</")
        if closing_offset >= 0:
            closing_tail = delta_message.reasoning[closing_offset + 2 :]
            if not closing_tail or any(
                closing_tail.startswith(name) or name.startswith(closing_tail)
                for name in ("parameter", "function", "tool_call")
            ):
                marker_offsets.append(closing_offset)
        if marker_offsets:
            marker_offset = min(marker_offsets)
            delta_message.reasoning = (
                delta_message.reasoning[:marker_offset] or None
            )
            self._reasoning_tool_markup_started = True

    def _suppress_unclosed_reasoning_content(
        self,
        delta_message: DeltaMessage | None,
        current_text: str,
    ) -> None:
        if (
            self.reasoning_parser is None
            or delta_message is None
            or not delta_message.reasoning
            or not delta_message.content
        ):
            return
        end_token = getattr(self.reasoning_parser, "end_token", None)
        if (
            (not end_token or end_token not in current_text)
            and delta_message.content == delta_message.reasoning
        ):
            delta_message.content = None

    def _tool_markup_filter_enabled(self) -> bool:
        return (
            self.request_for_sampling.tool_choice != "none"
            and (
                self.tool_parser is not None
                or bool(getattr(self.request_for_sampling, "tools", None))
            )
        )

    def _strip_tool_markup_from_delta(self, delta: dict[str, Any]) -> None:
        for key in ("reasoning_content", "reasoning", "content"):
            value = delta.get(key)
            if not value:
                continue
            message = DeltaMessage(reasoning=value)
            self._strip_tool_markup_from_reasoning(message)
            if message.reasoning:
                delta[key] = message.reasoning
            else:
                delta.pop(key, None)

    def _raw_post_reasoning_text(
        self, current_token_ids: Sequence[int], raw_delta_text: str
    ) -> str:
        if not self.reasoning_parser:
            return raw_delta_text
        end_token = getattr(self.reasoning_parser, "end_token", None)
        if not end_token:
            return raw_delta_text
        raw_current_text = self._decode_token_ids_for_parser(current_token_ids)
        if end_token in raw_current_text:
            return raw_current_text.rpartition(end_token)[2]
        return raw_delta_text

    def _tool_marker_token_ids(self) -> set[int]:
        marker_ids: set[int] = set()
        if self.tool_parser:
            for attr in (
                "tool_call_start_token_id",
                "tool_call_end_token_id",
                "bot_token_id",
            ):
                token_id = getattr(self.tool_parser, attr, None)
                if isinstance(token_id, int):
                    marker_ids.add(token_id)
            engine = getattr(self.tool_parser, "_parser_engine", None)
            for attr in ("_tool_call_token_id", "_tool_call_end_token_id"):
                token_id = getattr(engine, attr, None)
                if isinstance(token_id, int):
                    marker_ids.add(token_id)
        if self.reasoning_parser:
            token_id = getattr(self.reasoning_parser, "end_token_id", None)
            if isinstance(token_id, int):
                marker_ids.add(token_id)
        return marker_ids

    def _maybe_capture_raw_tool_text(
        self,
        *,
        delta_text: str,
        raw_delta_token_ids: list[int],
        get_raw_delta_text: Any,
    ) -> None:
        if not self._should_parse_tools():
            return

        tool_call_start = self._tool_call_start_token()
        saw_marker_id = bool(
            self._tool_marker_ids
            and self._tool_marker_ids.intersection(raw_delta_token_ids)
        )
        saw_start_text = bool(tool_call_start and tool_call_start in delta_text)
        if (
            not self._raw_tool_capture_started
            and not saw_marker_id
            and not saw_start_text
        ):
            return

        # Prefer vLLM's incrementally decoded text whenever it is available.
        # Re-decoding token IDs is only a fallback for parser markers that
        # vLLM intentionally omits from the visible delta.
        raw_delta_text = delta_text or get_raw_delta_text()
        if not raw_delta_text:
            return

        if self._raw_tool_capture_started:
            self._raw_tool_text_buffer += raw_delta_text
            return

        if tool_call_start and tool_call_start in raw_delta_text:
            self._raw_tool_capture_started = True
            self._raw_tool_text_buffer += raw_delta_text[
                raw_delta_text.index(tool_call_start) :
            ]

    def needs_raw_parser_delta(self, raw_delta_token_ids: Sequence[int]) -> bool:
        return (
            bool(self._raw_tool_capture_started or self._tool_text_buffer is not None)
            or bool(
                self._tool_marker_ids
                and self._tool_marker_ids.intersection(raw_delta_token_ids)
            )
        )

    @staticmethod
    def _compose_delta_message(
        reasoning: str | None, content: str | None
    ) -> DeltaMessage | None:
        delta_message = DeltaMessage(reasoning=reasoning, content=content)
        if not delta_message.reasoning and not delta_message.content:
            return None
        return delta_message

    def _add_tool_call_from_extracted(self, index: int, tool_call: Any) -> None:
        tool_delta = DeltaToolCall(
            index=index,
            type="function",
            id=(tool_call.id if tool_call.id else make_tool_call_id()),
            function=DeltaFunctionCall(
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            ),
        )
        existing = self.in_progress_tool_calls.get(index)
        self.in_progress_tool_calls[index] = self._merge_tool_call(existing, tool_delta)

    def _extract_tool_calls_from_text(
        self, text: str, *, saved_reasoning: str | None = None
    ) -> DeltaMessage | None:
        if self.tool_parser is None:
            return self._compose_delta_message(saved_reasoning, None)

        extracted = self.tool_parser.extract_tool_calls(text, self.request_for_sampling)
        if extracted.tools_called:
            for i, tool_call in enumerate(extracted.tool_calls):
                self._add_tool_call_from_extracted(i, tool_call)
            content = extracted.content or None
            if content:
                marker_offsets = [
                    text.find(marker)
                    for marker in ("<tool_call>", "<function=", "[TOOL_CALLS]")
                    if marker in text
                ]
                if marker_offsets:
                    original_prefix = text[: min(marker_offsets)]
                    if original_prefix.strip() == content.strip():
                        content = original_prefix
            return self._compose_delta_message(
                saved_reasoning, content
            )

        structured_calls, structured_reasoning = (
            self._extract_post_reasoning_structured_tool_calls(text)
        )
        if structured_calls:
            for index, tool_call in enumerate(structured_calls):
                self.in_progress_tool_calls[index] = tool_call
            return self._compose_delta_message(
                saved_reasoning if saved_reasoning is not None else structured_reasoning,
                None,
            )

        return self._compose_delta_message(saved_reasoning, extracted.content or None)

    def _post_reasoning_text_and_reasoning(
        self, text: str
    ) -> tuple[str, str | None]:
        end_token = getattr(self.reasoning_parser, "end_token", None)
        if not end_token or end_token not in text:
            return text, None
        reasoning_text, _, post_text = text.rpartition(end_token)
        start_token = getattr(self.reasoning_parser, "start_token", None)
        if start_token and start_token in reasoning_text:
            reasoning_text = reasoning_text.rpartition(start_token)[2]
        reasoning_text = reasoning_text.strip()
        return post_text, reasoning_text or None

    @staticmethod
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

    def _extract_post_reasoning_structured_tool_calls(
        self, text: str
    ) -> tuple[list[DeltaToolCall] | None, str | None]:
        if _has_user_structured_output_constraint(self.request_for_sampling):
            return None, None
        tool_choice = getattr(self.request_for_sampling, "tool_choice", None)
        if tool_choice in (None, "none", "auto"):
            return None, None

        post_text, reasoning = self._post_reasoning_text_and_reasoning(text)
        arguments = self._complete_json_text(post_text)
        if arguments is None:
            return None, None

        named_tool = _named_tool_choice_name(tool_choice)
        if named_tool is not None:
            return [
                DeltaToolCall(
                    index=0,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=named_tool,
                        arguments=arguments,
                    ),
                )
            ], reasoning

        if tool_choice != "required":
            return None, None
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            return None, None
        items = decoded if isinstance(decoded, list) else [decoded]
        tool_calls: list[DeltaToolCall] = []
        for index, item in enumerate(items):
            name = _value_from_mapping_or_object(item, "name")
            parameters = _value_from_mapping_or_object(item, "parameters", {})
            if not isinstance(name, str):
                continue
            if not isinstance(parameters, str):
                parameters = json.dumps(parameters, ensure_ascii=False)
            tool_calls.append(
                DeltaToolCall(
                    index=index,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=name,
                        arguments=parameters,
                    ),
                )
            )
        return tool_calls or None, reasoning

    def _extract_tool_calls_streaming(
        self,
        *,
        current_text: str,
        delta_text: str,
        delta_token_ids: list[int],
        current_token_ids: list[int],
    ) -> DeltaMessage | None:
        if self.tool_parser is None:
            return None
        return self.tool_parser.extract_tool_calls_streaming(
            previous_text=self.previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=self.previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=self.request_for_sampling,
        )

    def _extract_buffered_post_reasoning_tool_calls(
        self,
        output: Any,
        *,
        extra_candidates: Sequence[str] | None = None,
    ) -> DeltaMessage | None:
        if (
            not self._should_parse_tools()
            or output.index in self._tool_call_choices_emitted
            or self.in_progress_tool_calls
        ):
            return None

        tool_call_start = self._tool_call_start_token()
        tool_call_end = self._tool_call_end_token()
        if not tool_call_start:
            return None

        buffered_text = ""
        candidates = (
            self._raw_tool_text_buffer,
            self._post_reasoning_raw_text_buffer,
            self._post_reasoning_text_buffer,
            *(extra_candidates or ()),
        )
        if tool_call_end:
            for candidate in candidates:
                if (
                    candidate
                    and tool_call_start in candidate
                    and tool_call_end in candidate
                ):
                    buffered_text = candidate
                    break

        if not buffered_text:
            if not output.finish_reason:
                return None
            for candidate in candidates:
                if not candidate:
                    continue
                delta_message = self._extract_tool_calls_from_text(candidate)
                if self.in_progress_tool_calls:
                    self._raw_tool_text_buffer = ""
                    self._raw_tool_capture_started = False
                    self._post_reasoning_text_buffer = ""
                    self._post_reasoning_raw_text_buffer = ""
                    return delta_message
            return None

        self._raw_tool_text_buffer = ""
        self._raw_tool_capture_started = False
        self._post_reasoning_text_buffer = ""
        self._post_reasoning_raw_text_buffer = ""
        return self._extract_tool_calls_from_text(buffered_text)

    def _merge_streaming_tool_calls(self, tool_calls: list[DeltaToolCall]) -> None:
        for tool_delta in tool_calls:
            existing = self.in_progress_tool_calls.get(tool_delta.index)
            merged = self._merge_tool_call(existing, tool_delta)
            self.in_progress_tool_calls[tool_delta.index] = merged

    def _in_progress_tool_calls_are_complete(self) -> bool:
        if not self.in_progress_tool_calls:
            return False
        for tool_call in self.in_progress_tool_calls.values():
            arguments = tool_call.function.arguments if tool_call.function else None
            if not arguments:
                return False
            try:
                json.loads(arguments)
            except (TypeError, json.JSONDecodeError):
                return False
        return True

    def _dump_in_progress_tool_calls(self) -> list[dict[str, Any]]:
        return [
            tool_call.model_dump(exclude_none=True)
            for _, tool_call in self.in_progress_tool_calls.items()
        ]

    def _remap_finish_reason(
        self, output_index: int, finish_reason: str | None
    ) -> str | None:
        # Per https://github.com/ai-dynamo/dynamo/issues/8636 — OpenAI ChatCompletion finish_reason must be "tool_calls"
        # when the model called a tool. vLLM stops at <|im_end|> and reports
        # "stop"; remap once a tool_call delta has been emitted on THIS
        # choice. Per-choice tracking is required for `n > 1` requests —
        # choice 0 emitting tool_calls must not remap choice 1's stop.
        # Spec: https://github.com/openai/openai-openapi/blob/master/openapi.yaml
        if (
            finish_reason in {"stop", "length"}
            and output_index in self._tool_call_choices_emitted
        ):
            return "tool_calls"
        return finish_reason

    def _emit_tool_calls_choice(
        self,
        output: Any,
        *,
        reasoning: str | None = None,
        finish_reason: str | None = None,
    ) -> dict[str, Any]:
        self._tool_call_choices_emitted.add(output.index)
        delta: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": self._dump_in_progress_tool_calls(),
        }
        if reasoning:
            delta["reasoning_content"] = reasoning
        self._strip_tool_markup_from_delta(delta)
        choice = {
            "index": output.index,
            "delta": delta,
            "finish_reason": finish_reason
            or self._remap_finish_reason(output.index, output.finish_reason),
            "logprobs": output.logprobs,
        }
        self.in_progress_tool_calls.clear()
        return choice

    def _maybe_emit_structured_tool_call(
        self,
        output: Any,
        delta_message: DeltaMessage | None,
    ) -> tuple[DeltaMessage | None, dict[str, Any] | None]:
        if (
            self._structured_tool_call_name is None
            and not self._structured_required_tool_choice
        ) or delta_message is None:
            return delta_message, None

        content = delta_message.content or ""
        finished = bool(output.finish_reason)
        if not content and not finished:
            return delta_message, None

        arguments = self._structured_tool_json_delta(content, finished=finished)
        if arguments is None:
            return self._compose_delta_message(delta_message.reasoning, None), None

        if self._structured_tool_call_name is not None:
            self.in_progress_tool_calls[output.index] = DeltaToolCall(
                index=output.index,
                type="function",
                id=make_tool_call_id(),
                function=DeltaFunctionCall(
                    name=self._structured_tool_call_name,
                    arguments=arguments,
                ),
            )
        else:
            try:
                decoded = json.loads(arguments)
            except json.JSONDecodeError:
                return self._compose_delta_message(delta_message.reasoning, None), None
            calls = decoded if isinstance(decoded, list) else [decoded]
            for index, item in enumerate(calls):
                name = _value_from_mapping_or_object(item, "name")
                parameters = _value_from_mapping_or_object(item, "parameters", {})
                if not isinstance(name, str):
                    continue
                if not isinstance(parameters, str):
                    parameters = json.dumps(parameters, ensure_ascii=False)
                self.in_progress_tool_calls[index] = DeltaToolCall(
                    index=index,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(name=name, arguments=parameters),
                )
            if not self.in_progress_tool_calls:
                return self._compose_delta_message(delta_message.reasoning, None), None

        return None, self._emit_tool_calls_choice(
            output,
            reasoning=delta_message.reasoning,
            finish_reason="tool_calls",
        )

    def _build_choice(self, output: Any, delta: dict[str, Any]) -> dict[str, Any]:
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            self._emitted_reasoning_text += reasoning
        if (
            output.finish_reason
            and not self.reasoning_is_done
            and delta.get("content")
            and delta["content"] == self._emitted_reasoning_text
        ):
            delta.pop("content")
        self._strip_tool_markup_from_delta(delta)
        if delta.get("tool_calls"):
            self._tool_call_choices_emitted.add(output.index)
        finish_reason = output.finish_reason
        if self.structured_json_complete and finish_reason is None:
            finish_reason = "stop"
        return {
            "index": output.index,
            "delta": delta,
            "finish_reason": self._remap_finish_reason(
                output.index, finish_reason
            ),
            "logprobs": output.logprobs,
        }

    def _process_non_streaming_tool_output(self, output: Any) -> dict[str, Any] | None:
        delta_token_ids = list(output.token_ids or [])
        delta_text = output.text or ""
        current_text = self.previous_text + delta_text
        current_token_ids = self.previous_token_ids + delta_token_ids

        self.previous_text = current_text
        self.previous_token_ids = current_token_ids
        if not output.finish_reason:
            return None

        saved_reasoning = None
        content = current_text
        if self.reasoning_parser:
            saved_reasoning, content = self.reasoning_parser.extract_reasoning(
                current_text,
                request=self.request_for_sampling,
            )
            end_token = getattr(self.reasoning_parser, "end_token", None)
            if (
                saved_reasoning
                and content == current_text
                and (not end_token or end_token not in current_text)
            ):
                content = None
            if not self.request_for_sampling.include_reasoning:
                saved_reasoning = None

        delta_message = self._extract_tool_calls_from_text(
            content or "",
            saved_reasoning=saved_reasoning,
        )
        if delta_message is None:
            if self.in_progress_tool_calls:
                return self._emit_tool_calls_choice(output)
            return self._build_choice(output, {})

        delta: dict[str, Any] = {"role": "assistant"}
        if delta_message.content:
            delta["content"] = delta_message.content
        if delta_message.reasoning:
            delta["reasoning_content"] = delta_message.reasoning
        if self.in_progress_tool_calls:
            delta["tool_calls"] = self._dump_in_progress_tool_calls()
            self.in_progress_tool_calls.clear()
        if len(delta) == 1:
            delta = {}
        return self._build_choice(output, delta)

    def process_output(
        self,
        output: Any,
        raw_delta_token_ids: Sequence[int] | None = None,
    ) -> dict[str, Any] | None:
        if self._should_buffer_for_non_streaming_tool_parse():
            return self._process_non_streaming_tool_output(output)

        delta_token_ids = list(raw_delta_token_ids or output.token_ids or [])
        raw_delta_token_ids = delta_token_ids
        # vLLM output_processor already applies stop-token/stop-string trimming
        # to text. Re-detokenizing from token_ids can reintroduce stop markers.
        delta_text = output.text or ""
        raw_delta_text: str | None = None

        def get_raw_delta_text() -> str:
            nonlocal raw_delta_text
            if raw_delta_text is None:
                raw_delta_text = (
                    self._decode_token_ids_for_parser(raw_delta_token_ids)
                    if self._should_parse_tools()
                    else ""
                )
            return raw_delta_text

        delta: dict[str, Any] = {}
        if self._fast_plain_text:
            content = delta_text
            if self._structured_json_guard:
                content = self._structured_json_delta(
                    delta_text, finished=bool(output.finish_reason)
                )
            if content:
                delta = {
                    "role": "assistant",
                    "content": content,
                }
            elif output.finish_reason:
                delta = {}
            else:
                return None
            return self._build_choice(output, delta)

        current_text = self.previous_text + delta_text
        current_token_ids = self.previous_token_ids + delta_token_ids

        self._maybe_capture_raw_tool_text(
            delta_text=delta_text,
            raw_delta_token_ids=raw_delta_token_ids,
            get_raw_delta_text=get_raw_delta_text,
        )

        if output.index in self._tool_call_choices_emitted:
            self.previous_text = current_text
            self.previous_token_ids = current_token_ids
            if output.finish_reason:
                return self._build_choice(output, {})
            return None

        delta_message: DeltaMessage | None = DeltaMessage(content=delta_text)

        # ------------------------------------------------------------------
        # Drain the tool-text buffer (populated when </think> and <tool_call>
        # arrived in the same chunk).  The streaming tool parser cannot
        # handle that transition correctly, so we accumulate text here and
        # use the non-streaming extract_tool_calls() once complete.
        # ------------------------------------------------------------------
        if self._tool_text_buffer is not None:
            self._tool_text_buffer += delta_text
            tool_call_end = self._tool_call_end_token()
            buffer_complete = (
                tool_call_end
                and tool_call_end in self._tool_text_buffer
                and not self._buffer_tool_calls_until_finish()
            ) or output.finish_reason
            if buffer_complete:
                buffered_text = self._tool_text_buffer
                self._tool_text_buffer = None
                delta_message = self._extract_tool_calls_from_text(buffered_text)
            else:
                # Still accumulating; emit nothing for this chunk.
                self.previous_text = current_text
                self.previous_token_ids = current_token_ids
                return None

        elif not self.reasoning_is_done and self.reasoning_parser:
            delta_message = self.reasoning_parser.extract_reasoning_streaming(
                self.previous_text,
                current_text,
                delta_text,
                self.previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

            # Some models transition directly from reasoning to a tool call
            # without emitting the reasoning end marker. A split reasoning
            # parser otherwise exposes the raw tool XML as reasoning_content
            # even though the tool parser also emits a valid tool call.
            tool_call_start = self._tool_call_start_token()
            reasoning_text = (
                delta_message.reasoning
                if delta_message and delta_message.reasoning
                else ""
            )
            raw_tool_text = (
                self._raw_tool_text_buffer
                if self._raw_tool_capture_started
                else ""
            )
            tool_in_reasoning = bool(
                tool_call_start
                and (
                    tool_call_start in reasoning_text
                    or tool_call_start in raw_tool_text
                )
            )
            if tool_in_reasoning:
                saved_reasoning = reasoning_text or None
                if tool_call_start in reasoning_text:
                    saved_reasoning = (
                        reasoning_text.partition(tool_call_start)[0] or None
                    )
                self.reasoning_is_done = True
                self.previous_text = ""
                self.previous_token_ids = []
                current_text = ""
                current_token_ids = []
                if output.finish_reason:
                    buffered_text = (
                        raw_tool_text
                        or reasoning_text[reasoning_text.index(tool_call_start) :]
                    )
                    self._raw_tool_text_buffer = ""
                    self._raw_tool_capture_started = False
                    delta_message = self._extract_tool_calls_from_text(
                        buffered_text,
                        saved_reasoning=saved_reasoning,
                    )
                else:
                    delta_message = self._compose_delta_message(
                        saved_reasoning,
                        None,
                    )

            # When reasoning ends in this chunk, reset accumulated state.
            # If there is post-reasoning content (e.g. <tool_call> markup),
            # buffer it for non-streaming extraction rather than feeding it
            # to the streaming tool parser which cannot handle the combined
            # reasoning-end + tool-start in a single chunk.
            elif self.reasoning_parser.is_reasoning_end_streaming(
                current_token_ids, delta_token_ids
            ):
                self.reasoning_is_done = True
                saved_reasoning = delta_message.reasoning if delta_message else None
                post_content = (delta_message.content if delta_message else None) or ""
                raw_post_content = (
                    self._raw_post_reasoning_text(
                        current_token_ids,
                        get_raw_delta_text(),
                    )
                    if self._should_parse_tools()
                    else ""
                )

                self.previous_text = ""
                self.previous_token_ids = []
                current_text = ""
                current_token_ids = []

                tool_call_start = self._tool_call_start_token()
                tool_post_content = (
                    raw_post_content
                    if raw_post_content
                    and tool_call_start
                    and tool_call_start in raw_post_content
                    else post_content
                )
                if (
                    tool_post_content
                    and tool_call_start
                    and tool_call_start in tool_post_content
                ):
                    # Tool call markup present — buffer for non-streaming
                    # extraction (streaming parser can't handle the combined
                    # reasoning-end + tool-start in a single chunk).
                    self._tool_text_buffer = tool_post_content
                    if output.finish_reason:
                        # If finish_reason is already set, this is the final
                        # chunk; parse buffered text now instead of waiting for
                        # a later call that will never happen.
                        buffered_text = self._tool_text_buffer
                        self._tool_text_buffer = None
                        delta_message = self._extract_tool_calls_from_text(
                            buffered_text,
                            saved_reasoning=saved_reasoning,
                        )
                    else:
                        delta_message = self._compose_delta_message(
                            saved_reasoning,
                            None,
                        )
                else:
                    if post_content and self._should_parse_tools():
                        self._post_reasoning_text_buffer += post_content
                    if raw_post_content and self._should_parse_tools():
                        self._post_reasoning_raw_text_buffer += raw_post_content
                    # Plain content (or no content) after reasoning end.
                    delta_message = self._compose_delta_message(
                        reasoning=saved_reasoning,
                        content=post_content if post_content else None,
                    )
            elif (
                delta_message
                and delta_message.content
                and not delta_message.reasoning
                and self._should_parse_tools()
            ):
                # Reasoning parser returned content (not reasoning).
                # The model may have skipped reasoning and gone straight
                # to tool calls (e.g. Mistral [TOOL_CALLS] without
                # [THINK]...[/THINK]).  Let the tool parser decide.
                delta_message = self._extract_tool_calls_streaming(
                    current_text=current_text,
                    delta_text=delta_text,
                    current_token_ids=current_token_ids,
                    delta_token_ids=delta_token_ids,
                )
        else:
            if self._should_parse_tools():
                tool_call_start = self._tool_call_start_token()
                if (
                    self._buffer_tool_calls_until_finish()
                    and self._tool_text_buffer is None
                    and tool_call_start
                    and tool_call_start in delta_text
                ):
                    self._tool_text_buffer = delta_text[
                        delta_text.index(tool_call_start) :
                    ]
                    if output.finish_reason:
                        buffered_text = self._tool_text_buffer
                        self._tool_text_buffer = None
                        delta_message = self._extract_tool_calls_from_text(
                            buffered_text
                        )
                    else:
                        self.previous_text = current_text
                        self.previous_token_ids = current_token_ids
                        return None

                if self.reasoning_is_done and delta_text:
                    self._post_reasoning_text_buffer += delta_text
                if self.reasoning_is_done:
                    raw_after_reasoning = get_raw_delta_text()
                    if raw_after_reasoning:
                        self._post_reasoning_raw_text_buffer += raw_after_reasoning
                no_prev_reasoning = (
                    delta_message
                    and delta_message.content
                    and not delta_message.reasoning
                )
                if self.reasoning_is_done or no_prev_reasoning:
                    delta_message = self._extract_tool_calls_streaming(
                        current_text=current_text,
                        delta_text=delta_text,
                        current_token_ids=current_token_ids,
                        delta_token_ids=delta_token_ids,
                    )
                    had_tool_calls = bool(self.in_progress_tool_calls)
                    fallback_message = (
                        self._extract_buffered_post_reasoning_tool_calls(
                            output,
                            extra_candidates=(current_text,),
                        )
                    )
                    if fallback_message is not None or (
                        not had_tool_calls and self.in_progress_tool_calls
                    ):
                        delta_message = fallback_message

        if self._should_parse_tools() and not self.in_progress_tool_calls:
            fallback_message = self._extract_buffered_post_reasoning_tool_calls(
                output,
                extra_candidates=(current_text,),
            )
            if fallback_message is not None or self.in_progress_tool_calls:
                delta_message = fallback_message

        # A streaming parser may leave a truncated argument fragment when a
        # terminal chunk contains malformed or repeated closing markers. At
        # finish, repair it with vLLM's batch parser over the complete text.
        if (
            output.finish_reason
            and self.in_progress_tool_calls
            and not self._in_progress_tool_calls_are_complete()
        ):
            self.in_progress_tool_calls.clear()
            batch_message = self._extract_tool_calls_from_text(current_text)
            if self.in_progress_tool_calls:
                delta_message = batch_message

        self._suppress_unclosed_reasoning_content(delta_message, current_text)
        self._strip_tool_markup_from_reasoning(delta_message)
        delta_message, structured_tool_choice = self._maybe_emit_structured_tool_call(
            output, delta_message
        )
        if structured_tool_choice is not None:
            self.previous_text = current_text
            self.previous_token_ids = current_token_ids
            return structured_tool_choice

        choice = None
        if delta_message is None:
            if self.in_progress_tool_calls:
                choice = self._emit_tool_calls_choice(output)
            elif output.finish_reason:
                choice = self._build_choice(output, {})
        elif delta_message.tool_calls:
            self._merge_streaming_tool_calls(delta_message.tool_calls)
            if output.finish_reason and self.in_progress_tool_calls:
                # Tool calls and finish_reason arrived in the same chunk.
                # Emit now — there will be no subsequent process_output call
                # to drain the buffer.
                choice = self._emit_tool_calls_choice(output)
        elif delta_message.content or delta_message.reasoning:
            delta = {"role": "assistant"}
            content = delta_message.content
            if self.in_progress_tool_calls and self._is_control_only_content(content):
                content = None
            if content:
                delta["content"] = content
            if delta_message.reasoning:
                delta["reasoning_content"] = delta_message.reasoning
            if self.in_progress_tool_calls:
                delta["tool_calls"] = self._dump_in_progress_tool_calls()
                self.in_progress_tool_calls.clear()
            if len(delta) > 1:
                choice = self._build_choice(output, delta)
        elif self.in_progress_tool_calls:
            choice = self._emit_tool_calls_choice(output)
        elif output.finish_reason:
            choice = self._build_choice(output, {})

        self.previous_text = current_text
        self.previous_token_ids = current_token_ids
        return choice
