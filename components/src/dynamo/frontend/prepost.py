#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from json import JSONDecoder
from typing import Any

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.reasoning import ReasoningParser
from vllm.renderers import ChatParams
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.utils.async_utils import AsyncMicrobatchTokenizer


@dataclass
class PreprocessResult:
    request_for_sampling: ChatCompletionRequest
    tool_parser: ToolParser | None
    chat_template_kwargs: dict[str, Any]
    engine_prompt: dict[str, Any]
    prompt_token_ids: list[int]


_ASYNC_TOKENIZER_POOL: dict[int, AsyncMicrobatchTokenizer] = {}
SKIP_REQUEST_VALIDATION = os.getenv("DYN_VLLM_SKIP_REQUEST_VALIDATION", "1") == "1"


def _get_async_tokenizer(tokenizer: TokenizerLike) -> AsyncMicrobatchTokenizer:
    key = id(tokenizer)
    async_tokenizer = _ASYNC_TOKENIZER_POOL.get(key)
    if async_tokenizer is None:
        async_tokenizer = AsyncMicrobatchTokenizer(tokenizer)
        _ASYNC_TOKENIZER_POOL[key] = async_tokenizer
    return async_tokenizer


def _strip_non_json_prefix(content: str, prefer_array: bool = False) -> str:
    stripped = content.lstrip()
    if not stripped:
        return stripped

    if "</think>" in stripped:
        stripped = stripped.rsplit("</think>", 1)[-1].lstrip()

    if prefer_array:
        idx = stripped.find("[")
        if idx != -1:
            return stripped[idx:]
    else:
        idx = stripped.find("{")
        if idx != -1:
            return stripped[idx:]

    first_obj = stripped.find("{")
    first_arr = stripped.find("[")
    starts = [i for i in (first_obj, first_arr) if i != -1]
    if not starts:
        return stripped
    return stripped[min(starts):]


def _iter_json_candidates(content: str, prefer_array: bool) -> Sequence[str]:
    stripped = _strip_non_json_prefix(content, prefer_array=prefer_array)
    if not stripped:
        return []

    candidates: list[str] = []
    seen: set[int] = set()

    def _add(index: int) -> None:
        if index >= 0 and index not in seen:
            seen.add(index)
            candidates.append(stripped[index:])

    preferred = "[" if prefer_array else "{"
    fallback = "{" if prefer_array else "["

    _add(stripped.find(preferred))
    _add(stripped.find(fallback))

    for idx, ch in enumerate(stripped):
        if ch in "{[":
            _add(idx)

    return candidates


def _iter_json_candidates_with_offsets(
    content: str, prefer_array: bool
) -> Sequence[tuple[int, str]]:
    stripped = _strip_non_json_prefix(content, prefer_array=prefer_array)
    if not stripped:
        return []

    stripped_start = content.find(stripped)
    if stripped_start < 0:
        stripped_start = len(content) - len(stripped)

    candidates: list[tuple[int, str]] = []
    seen: set[int] = set()

    def _add(index: int) -> None:
        if index >= 0 and index not in seen:
            seen.add(index)
            candidates.append((stripped_start + index, stripped[index:]))

    preferred = "[" if prefer_array else "{"
    fallback = "{" if prefer_array else "["

    _add(stripped.find(preferred))
    _add(stripped.find(fallback))

    for idx, ch in enumerate(stripped):
        if ch in "{[":
            _add(idx)

    return candidates


def _trailing_whitespace_run(content: str) -> int:
    run = 0
    for ch in reversed(content):
        if not ch.isspace():
            break
        run += 1
    return run


def _repair_unterminated_json(content: str) -> str | None:
    core = content.rstrip()
    if not core:
        return None

    expected_closers: list[str] = []
    in_string = False
    escaped = False
    for ch in core:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            expected_closers.append("}")
        elif ch == "[":
            expected_closers.append("]")
        elif ch in "}]":
            if not expected_closers or ch != expected_closers[-1]:
                return None
            expected_closers.pop()

    if in_string:
        return None

    if not expected_closers:
        return None

    return core + "".join(reversed(expected_closers))


def _decode_forced_tool_json(
    content: str,
    *,
    prefer_array: bool,
    allow_trailing_whitespace_repair: bool,
) -> Any | None:
    decoder = JSONDecoder()
    for candidate in _iter_json_candidates(content, prefer_array=prefer_array):
        normalized = candidate.lstrip()
        try:
            parsed, _ = decoder.raw_decode(normalized)
            return parsed
        except Exception:
            if (
                not allow_trailing_whitespace_repair
                or _trailing_whitespace_run(candidate) < 32
            ):
                continue
            repaired = _repair_unterminated_json(candidate)
            if repaired is None:
                continue
            try:
                parsed, _ = decoder.raw_decode(repaired.lstrip())
                return parsed
            except Exception:
                continue
    return None


def _decode_forced_tool_json_with_prefix(
    content: str,
    *,
    prefer_array: bool,
    allow_trailing_whitespace_repair: bool,
) -> tuple[Any, str, str] | None:
    decoder = JSONDecoder()
    for start, candidate in _iter_json_candidates_with_offsets(
        content, prefer_array=prefer_array
    ):
        normalized = candidate.lstrip()
        try:
            parsed, end = decoder.raw_decode(normalized)
            consumed = normalized[:end]
            return parsed, content[: start + (len(candidate) - len(normalized))], consumed
        except Exception:
            if (
                not allow_trailing_whitespace_repair
                or _trailing_whitespace_run(candidate) < 32
            ):
                continue
            repaired = _repair_unterminated_json(candidate)
            if repaired is None:
                continue
            try:
                parsed, end = decoder.raw_decode(repaired.lstrip())
                consumed = repaired.lstrip()[:end]
                return parsed, content[: start + (len(candidate) - len(normalized))], consumed
            except Exception:
                continue
    return None


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


def _structured_output_prompt_hint(structured_outputs: Any | None) -> str | None:
    if structured_outputs is None:
        return None

    if isinstance(structured_outputs, dict):
        so = structured_outputs
    else:
        so = {
            "json": getattr(structured_outputs, "json", None),
            "regex": getattr(structured_outputs, "regex", None),
            "choice": getattr(structured_outputs, "choice", None),
            "grammar": getattr(structured_outputs, "grammar", None),
            "json_object": getattr(structured_outputs, "json_object", None),
        }

    json_schema = so.get("json")
    if json_schema is not None:
        return (
            "After you finish reasoning, immediately output only a JSON value "
            "that matches this schema exactly. Do not add any prose or extra "
            f"keys.\nSchema: {json.dumps(json_schema, ensure_ascii=False)}"
        )

    if so.get("json_object"):
        return (
            "After you finish reasoning, immediately output only a valid JSON "
            "object. Do not add prose before or after the JSON."
        )

    regex = so.get("regex")
    if regex is not None:
        return (
            "After you finish reasoning, immediately output only text that "
            f"matches this regex exactly: {regex}"
        )

    choice = so.get("choice")
    if choice:
        return (
            "After you finish reasoning, immediately output exactly one of "
            f"these choices and nothing else: {json.dumps(choice, ensure_ascii=False)}"
        )

    grammar = so.get("grammar")
    if grammar:
        return (
            "After you finish reasoning, immediately output only text that "
            f"matches this grammar exactly:\n{grammar}"
        )

    return None


def _response_format_to_structured_outputs_params(
    response_format: Any | None,
) -> StructuredOutputsParams | None:
    if response_format is None:
        return None

    if isinstance(response_format, dict):
        rf_type = response_format.get("type")
        json_schema = response_format.get("json_schema")
    else:
        rf_type = getattr(response_format, "type", None)
        json_schema = getattr(response_format, "json_schema", None)

    if rf_type == "json_object":
        return StructuredOutputsParams(json_object=True)

    if rf_type != "json_schema" or json_schema is None:
        return None

    if isinstance(json_schema, dict):
        schema = json_schema.get("schema") or json_schema.get("json_schema")
    else:
        schema = getattr(json_schema, "json_schema", None) or getattr(
            json_schema, "schema", None
        )

    if schema is None:
        return None
    return StructuredOutputsParams(json=schema)


def _inject_system_hint(
    messages: Sequence[Any],
    hint: str | None,
) -> list[Any]:
    if not hint:
        return list(messages)

    system_prefix = (
        "Structured output requirement:\n"
        f"{hint}"
    )
    rendered = list(messages)
    if rendered:
        first = rendered[0]
        role = first.get("role") if isinstance(first, dict) else getattr(first, "role", None)
        if role == "system":
            content = (
                first.get("content")
                if isinstance(first, dict)
                else getattr(first, "content", "")
            )
            merged = f"{content}\n\n{system_prefix}" if content else system_prefix
            if isinstance(first, dict):
                updated = dict(first)
                updated["content"] = merged
            else:
                updated = first.model_copy(update={"content": merged})
            return [updated, *rendered[1:]]

    return [{"role": "system", "content": system_prefix}, *rendered]


def _forced_tool_choice_name(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        return tool_choice.function.name
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                return name
    return None


def _filter_tools_for_forced_choice(
    tools: Sequence[Any] | None,
    forced_tool_name: str | None,
) -> list[Any] | None:
    if not tools or not forced_tool_name:
        return None

    selected_tools: list[Any] = []
    for tool in tools:
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict) and function.get("name") == forced_tool_name:
                selected_tools.append(tool)
                continue

        function = getattr(tool, "function", None)
        if getattr(function, "name", None) == forced_tool_name:
            selected_tools.append(tool)

    return selected_tools or None


def _prepare_request(
    request: dict[str, Any] | ChatCompletionRequest,
    *,
    tokenizer: TokenizerLike,
    tool_parser_class: type[ToolParser] | None,
    reasoning_parser_class: type[ReasoningParser] | None = None,
) -> tuple[ChatCompletionRequest, ToolParser | None, dict[str, Any], Any, ChatParams]:
    """Validate request and build arguments for template rendering.

    Returns:
        request_for_sampling: Validated ChatCompletionRequest.
        tool_parser: Instantiated tool parser, or None.
        chat_template_kwargs: Template kwargs (for PreprocessResult).
        messages_for_render: Messages to pass as first arg to render_messages.
        chat_params: ChatParams for render_messages / render_messages_async.
    """
    if isinstance(request, ChatCompletionRequest):
        request_for_sampling = request
    elif SKIP_REQUEST_VALIDATION:
        # Trusted fast path; caller must provide OpenAI-compatible payload.
        request_for_sampling = ChatCompletionRequest.model_construct(**request)
        if (
            request.get("response_format") is not None
            or (
                request_for_sampling.tools
                and any(
                    not hasattr(tool, "model_dump")
                    for tool in request_for_sampling.tools
                )
            )
        ):
            # model_construct skips protocol validators, including the
            # response_format -> structured_outputs normalization that the
            # reasoning/guided-decoding path relies on.
            request_for_sampling = ChatCompletionRequest.model_validate(request)
    else:
        request_for_sampling = ChatCompletionRequest.model_validate(request)

    if (
        request_for_sampling.structured_outputs is None
        and not isinstance(request, ChatCompletionRequest)
    ):
        request_for_sampling.structured_outputs = (
            _response_format_to_structured_outputs_params(
                request.get("response_format")
            )
        )

    forced_tool_name = _forced_tool_choice_name(request_for_sampling.tool_choice)
    selected_tools = _filter_tools_for_forced_choice(
        request_for_sampling.tools,
        forced_tool_name,
    )
    if selected_tools is not None:
        request_for_sampling.tools = selected_tools

    tool_parser: ToolParser | None = None
    if tool_parser_class and request_for_sampling.tools:
        if request_for_sampling.tool_choice != "none":
            tool_parser = tool_parser_class(tokenizer)
            request_for_sampling = tool_parser.adjust_request(request_for_sampling)

    # Strip tools from the template when tool_choice=none so the model doesn't
    # see them and generate raw XML tool calls in its response.
    _exclude_tools = os.environ.get(
        "DYN_EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE", "true"
    ).lower() in ("true", "1", "yes", "on")
    tool_dicts = (
        [tool.model_dump() for tool in request_for_sampling.tools]
        if request_for_sampling.tools
        and not (_exclude_tools and request_for_sampling.tool_choice == "none")
        else None
    )
    if tool_dicts and forced_tool_name:
        selected_tool_dicts = [
            tool
            for tool in tool_dicts
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and tool["function"].get("name") == forced_tool_name
        ]
        if selected_tool_dicts:
            # Keep the original request schema intact for downstream parsing,
            # but only expose the selected tool in the prompt when a named
            # tool is forced. This avoids MiniMax describing unrelated tool
            # schemas instead of emitting the chosen call.
            tool_dicts = selected_tool_dicts
    chat_template_kwargs = dict(request_for_sampling.chat_template_kwargs or {})
    if (
        reasoning_parser_class is not None
        and reasoning_parser_class.__name__ == "MiniMaxM2AppendThinkReasoningParser"
        and (
            request_for_sampling.structured_outputs is not None
            or request_for_sampling.tool_choice == "required"
            or forced_tool_name is not None
        )
        and "enable_thinking" not in chat_template_kwargs
        and "thinking" not in chat_template_kwargs
    ):
        # MiniMax M2 always reasons before the constrained answer. Without an
        # explicit enable_thinking hint, Dynamo's Python frontend can render a
        # prompt that disagrees with the reasoning-aware constrained-decoding
        # path, and the final constrained token never reaches the client.
        chat_template_kwargs["enable_thinking"] = True
        request_for_sampling.chat_template_kwargs = dict(chat_template_kwargs)
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
        else list(request_for_sampling.messages)
    )
    if (
        reasoning_parser_class is not None
        and reasoning_parser_class.__name__ == "MiniMaxM2AppendThinkReasoningParser"
        and request_for_sampling.structured_outputs is not None
    ):
        messages_for_render = _inject_system_hint(
            messages_for_render,
            _structured_output_prompt_hint(request_for_sampling.structured_outputs),
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
    renderer,
    tool_parser_class: type[ToolParser] | None,
    reasoning_parser_class: type[ReasoningParser] | None = None,
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
        reasoning_parser_class=reasoning_parser_class,
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
        structured_decoding_active: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.request_for_sampling = request_for_sampling
        self.sampling_params = sampling_params
        self.tool_parser = tool_parser
        self._structured_outputs_active = structured_decoding_active or (
            request_for_sampling.structured_outputs is not None
        )
        keep_reasoning_parser = (
            reasoning_parser_class is not None
            and reasoning_parser_class.__name__ == "MiniMaxM2AppendThinkReasoningParser"
        )
        self.reasoning_parser = (
            reasoning_parser_class(
                tokenizer,
                chat_template_kwargs=chat_template_kwargs,
            )
            if reasoning_parser_class
            and (
                not self._structured_outputs_active or keep_reasoning_parser
            )
            else None
        )
        self._fast_plain_text = (
            self.tool_parser is None
            and self.reasoning_parser is None
            and not self._is_forced_tool_choice()
        )

        self._control_markers = tuple(
            t for t in getattr(tokenizer, "all_special_tokens", ()) if t
        )

        self.previous_text = ""
        self.previous_token_ids: list[int] = []
        self.reasoning_is_done = False
        self.in_progress_tool_calls: dict[int, DeltaToolCall] = {}
        self._forced_tool_json_buffer: str | None = None
        self._structured_json_buffer: str | None = None
        self._structured_response_completed = False
        # Buffer for post-reasoning tool text when </think> and <tool_call>
        # arrive in the same chunk.  The streaming tool parser cannot handle
        # this correctly, so we accumulate text here and fall back to the
        # non-streaming extract_tool_calls() once the buffer is complete.
        self._tool_text_buffer: str | None = None

    def _log_forced_tool_state(self, event: str, **fields: Any) -> None:
        return

    def _log_structured_state(self, event: str, **fields: Any) -> None:
        return

    @staticmethod
    def _strip_json_fence(text: str | None) -> str:
        raw = (text or "").strip()
        if not raw.startswith("```"):
            return raw
        lines = raw.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _extract_forced_tool_calls_from_json(self, text: str | None) -> bool:
        if self.request_for_sampling.tool_choice == "none":
            return False
        if not self._is_forced_tool_choice() and not self._should_parse_tools():
            return False

        raw = self._strip_json_fence(text)
        if not raw:
            return False

        tool_choice = self.request_for_sampling.tool_choice
        forced_tool_name = self._get_forced_tool_name()
        parsed = _decode_forced_tool_json(
            raw,
            prefer_array=forced_tool_name is None,
            allow_trailing_whitespace_repair=True,
        )
        if parsed is None:
            return False

        if forced_tool_name is not None:
            if not isinstance(parsed, dict):
                return False
            tool_delta = DeltaToolCall(
                index=0,
                type="function",
                id=make_tool_call_id(),
                function=DeltaFunctionCall(
                    name=forced_tool_name,
                    arguments=json.dumps(parsed, ensure_ascii=False),
                ),
            )
            existing = self.in_progress_tool_calls.get(0)
            self.in_progress_tool_calls[0] = self._merge_tool_call(
                existing, tool_delta
            )
            return True

        if tool_choice == "required":
            if isinstance(parsed, dict) and "name" in parsed and "parameters" in parsed:
                parsed = [parsed]
            if not isinstance(parsed, list):
                return False

            added = False
            for index, item in enumerate(parsed):
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                tool_delta = DeltaToolCall(
                    index=index,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=item["name"],
                        arguments=json.dumps(
                            item.get("parameters", {}),
                            ensure_ascii=False,
                        ),
                    ),
                )
                existing = self.in_progress_tool_calls.get(index)
                self.in_progress_tool_calls[index] = self._merge_tool_call(
                    existing, tool_delta
                )
                added = True
            return added

        return False

    def _recover_forced_tool_calls_from_text_with_prefix(
        self,
        text: str | None,
    ) -> str | None:
        raw = self._strip_json_fence(text)
        if not raw:
            return None

        decoded = _decode_forced_tool_json_with_prefix(
            raw,
            prefer_array=self._get_forced_tool_name() is None,
            allow_trailing_whitespace_repair=True,
        )
        if decoded is None:
            return None

        _, prefix, parsed_json = decoded
        if not self._extract_forced_tool_calls_from_json(parsed_json):
            return None
        return prefix or None

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
            and self.request_for_sampling.tool_choice != "none"
        )

    def _get_forced_tool_name(self) -> str | None:
        tool_choice = self.request_for_sampling.tool_choice
        if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
            return tool_choice.function.name
        if isinstance(tool_choice, dict):
            function = tool_choice.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str) and name:
                    return name
        return None

    def _is_forced_tool_choice(self) -> bool:
        tool_choice = self.request_for_sampling.tool_choice
        return tool_choice == "required" or self._get_forced_tool_name() is not None

    def _should_recover_structured_json(self) -> bool:
        if self._is_forced_tool_choice():
            return False
        structured_outputs = getattr(self.request_for_sampling, "structured_outputs", None)
        if structured_outputs is None:
            return False
        return bool(
            getattr(structured_outputs, "json", None) is not None
            or getattr(structured_outputs, "json_object", None)
        )

    @staticmethod
    def _compose_delta_message(
        reasoning: str | None, content: str | None
    ) -> DeltaMessage | None:
        delta_message = DeltaMessage(reasoning=reasoning, content=content)
        if not delta_message.reasoning and not delta_message.content:
            return None
        return delta_message

    def _coalesce_forced_tool_choice_delta(
        self,
        delta_message: DeltaMessage | None,
        *,
        finish_reason: str | None,
    ) -> DeltaMessage | None:
        if not self._is_forced_tool_choice():
            return delta_message

        combined_finish_text = None
        if finish_reason and delta_message is not None and not self.in_progress_tool_calls:
            combined_finish_text = "".join(
                part
                for part in (
                    self._forced_tool_json_buffer or "",
                    delta_message.reasoning or "",
                    delta_message.content or "",
                )
                if part
            )
            recovered_prefix = self._recover_forced_tool_calls_from_text_with_prefix(
                combined_finish_text
            )
            if recovered_prefix is not None:
                self._forced_tool_json_buffer = None
                self._log_forced_tool_state(
                    "recover_finish_chunk_with_prefix",
                    finish_reason=finish_reason,
                    prefix_preview=recovered_prefix[:160],
                )
                return self._compose_delta_message(
                    recovered_prefix if recovered_prefix else None,
                    None,
                )

        if (
            finish_reason
            and self._forced_tool_json_buffer
            and not self.in_progress_tool_calls
            and self._extract_forced_tool_calls_from_json(self._forced_tool_json_buffer)
        ):
            self._log_forced_tool_state(
                "buffer_parse_at_finish",
                finish_reason=finish_reason,
                buffered_preview=(self._forced_tool_json_buffer or "")[:160],
            )
            self._forced_tool_json_buffer = None
            if (
                delta_message
                and delta_message.content
                and not self._is_control_only_content(delta_message.content)
            ):
                return self._compose_delta_message(None, delta_message.content)
            return None

        if delta_message is None:
            return delta_message

        buffered_fragment: str | None = None
        if delta_message.reasoning:
            content = delta_message.content
            if content and not self._is_control_only_content(content):
                return delta_message
            buffered_fragment = delta_message.reasoning
        elif delta_message.content and not delta_message.tool_calls:
            buffered_fragment = delta_message.content
        else:
            return delta_message

        self._forced_tool_json_buffer = (
            (self._forced_tool_json_buffer or "") + buffered_fragment
        )
        self._log_forced_tool_state(
            "buffer_append",
            finish_reason=finish_reason,
            fragment_len=len(buffered_fragment),
            fragment_preview=buffered_fragment[:160],
        )
        if finish_reason and self._extract_forced_tool_calls_from_json(
            self._forced_tool_json_buffer
        ):
            self._log_forced_tool_state(
                "buffer_parse_on_finish_chunk",
                finish_reason=finish_reason,
                buffered_preview=(self._forced_tool_json_buffer or "")[:160],
            )
            self._forced_tool_json_buffer = None
            return None
        if self._extract_forced_tool_calls_from_json(self._forced_tool_json_buffer):
            self._log_forced_tool_state(
                "buffer_parse_mid_stream",
                finish_reason=finish_reason,
                buffered_preview=(self._forced_tool_json_buffer or "")[:160],
            )
            self._forced_tool_json_buffer = None
            return None
        return None

    def _coalesce_structured_json_delta(
        self,
        delta_message: DeltaMessage | None,
    ) -> DeltaMessage | None:
        if (
            not self._should_recover_structured_json()
            or self._structured_response_completed
            or delta_message is None
            or delta_message.tool_calls
        ):
            return delta_message

        fragment = None
        if delta_message.reasoning:
            fragment = delta_message.reasoning
        elif delta_message.content:
            fragment = delta_message.content
        if not fragment:
            return delta_message

        combined = (self._structured_json_buffer or "") + fragment
        decoded = _decode_forced_tool_json_with_prefix(
            combined,
            prefer_array=False,
            allow_trailing_whitespace_repair=True,
        )
        if decoded is None:
            self._structured_json_buffer = combined
            return delta_message

        _, prefix, parsed_json = decoded
        self._structured_json_buffer = None
        self._structured_response_completed = True
        return self._compose_delta_message(
            prefix if prefix else None,
            parsed_json,
        )

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
        if self._extract_forced_tool_calls_from_json(text):
            return self._compose_delta_message(saved_reasoning, None)

        if self.tool_parser is None:
            return self._compose_delta_message(saved_reasoning, None)

        extracted = self.tool_parser.extract_tool_calls(text, self.request_for_sampling)
        if extracted.tools_called:
            for i, tool_call in enumerate(extracted.tool_calls):
                self._add_tool_call_from_extracted(i, tool_call)
            return self._compose_delta_message(saved_reasoning, None)

        return self._compose_delta_message(saved_reasoning, extracted.content or None)

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

    def _merge_streaming_tool_calls(self, tool_calls: list[DeltaToolCall]) -> None:
        for tool_delta in tool_calls:
            existing = self.in_progress_tool_calls.get(tool_delta.index)
            merged = self._merge_tool_call(existing, tool_delta)
            self.in_progress_tool_calls[tool_delta.index] = merged

    def _dump_in_progress_tool_calls(self) -> list[dict[str, Any]]:
        return [
            tool_call.model_dump(exclude_none=True)
            for _, tool_call in self.in_progress_tool_calls.items()
        ]

    def _emit_tool_calls_choice(self, output: Any) -> dict[str, Any]:
        choice = {
            "index": output.index,
            "delta": {
                "role": "assistant",
                "tool_calls": self._dump_in_progress_tool_calls(),
            },
            "finish_reason": "tool_calls",
            "logprobs": output.logprobs,
        }
        self.in_progress_tool_calls.clear()
        return choice

    @staticmethod
    def _build_choice(output: Any, delta: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": output.index,
            "delta": delta,
            "finish_reason": output.finish_reason,
            "logprobs": output.logprobs,
        }

    def process_output(self, output: Any) -> dict[str, Any] | None:
        delta_token_ids = list(output.token_ids or [])
        # vLLM output_processor already applies stop-token/stop-string trimming
        # to text. Re-detokenizing from token_ids can reintroduce stop markers.
        delta_text = output.text or ""
        self._log_forced_tool_state(
            "process_output_start",
            finish_reason=output.finish_reason,
            delta_text_len=len(delta_text),
            delta_text_preview=delta_text[:160],
            delta_token_ids=len(delta_token_ids),
        )
        self._log_structured_state(
            "process_output_start",
            finish_reason=output.finish_reason,
            delta_text_len=len(delta_text),
            delta_text_preview=delta_text[:160],
            delta_token_ids=len(delta_token_ids),
        )
        delta: dict[str, Any] = {}
        if self._fast_plain_text:
            if delta_text:
                delta = {
                    "role": "assistant",
                    "content": delta_text,
                }
            elif output.finish_reason:
                delta = {}
            else:
                return None
            return self._build_choice(output, delta)

        current_text = self.previous_text + delta_text
        current_token_ids = self.previous_token_ids + delta_token_ids

        delta_message: DeltaMessage | None = DeltaMessage(content=delta_text)

        # ------------------------------------------------------------------
        # Drain the tool-text buffer (populated when </think> and <tool_call>
        # arrived in the same chunk).  The streaming tool parser cannot
        # handle that transition correctly, so we accumulate text here and
        # use the non-streaming extract_tool_calls() once complete.
        # ------------------------------------------------------------------
        if self._tool_text_buffer is not None:
            self._tool_text_buffer += delta_text
            tool_call_end = getattr(self.tool_parser, "tool_call_end_token", None)
            buffer_complete = (
                tool_call_end and tool_call_end in self._tool_text_buffer
            ) or output.finish_reason
            if buffer_complete:
                buffered_text = self._tool_text_buffer
                self._tool_text_buffer = None
                self._log_forced_tool_state(
                    "tool_buffer_complete",
                    finish_reason=output.finish_reason,
                    buffered_len=len(buffered_text),
                    buffered_preview=buffered_text[:160],
                )
                delta_message = self._extract_tool_calls_from_text(buffered_text)
            else:
                # Still accumulating; emit nothing for this chunk.
                self._log_forced_tool_state(
                    "tool_buffer_wait",
                    finish_reason=output.finish_reason,
                    buffered_len=len(self._tool_text_buffer),
                )
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

            # When reasoning ends in this chunk, reset accumulated state.
            # If there is post-reasoning content (e.g. <tool_call> markup),
            # buffer it for non-streaming extraction rather than feeding it
            # to the streaming tool parser which cannot handle the combined
            # reasoning-end + tool-start in a single chunk.
            if self.reasoning_parser.is_reasoning_end_streaming(
                current_token_ids, delta_token_ids
            ):
                self.reasoning_is_done = True
                self._log_structured_state(
                    "reasoning_end_streaming",
                    finish_reason=output.finish_reason,
                    delta_text_preview=delta_text[:160],
                    current_text_preview=current_text[:160],
                )
                saved_reasoning = delta_message.reasoning if delta_message else None
                post_content = (delta_message.content if delta_message else None) or ""

                self.previous_text = ""
                self.previous_token_ids = []
                current_text = ""
                current_token_ids = []

                tool_call_start = getattr(
                    self.tool_parser, "tool_call_start_token", None
                )
                if post_content and tool_call_start and tool_call_start in post_content:
                    # Tool call markup present — buffer for non-streaming
                    # extraction (streaming parser can't handle the combined
                    # reasoning-end + tool-start in a single chunk).
                    self._tool_text_buffer = post_content
                    self._log_forced_tool_state(
                        "reasoning_end_with_tool_buffer",
                        finish_reason=output.finish_reason,
                        saved_reasoning_len=len(saved_reasoning or ""),
                        post_content_len=len(post_content),
                        post_content_preview=post_content[:160],
                    )
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
                    # Plain content (or no content) after reasoning end.
                    delta_message = self._compose_delta_message(
                        reasoning=saved_reasoning,
                        content=post_content if post_content else None,
                    )
            else:
                self._log_structured_state(
                    "reasoning_continues",
                    finish_reason=output.finish_reason,
                    delta_has_reasoning=bool(delta_message and delta_message.reasoning),
                    delta_has_content=bool(delta_message and delta_message.content),
                    delta_reasoning_preview=(
                        (delta_message.reasoning or "")[:160]
                        if delta_message
                        else None
                    ),
                    delta_content_preview=(
                        (delta_message.content or "")[:160]
                        if delta_message
                        else None
                    ),
                )
            if (
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

        delta_message = self._coalesce_forced_tool_choice_delta(
            delta_message,
            finish_reason=output.finish_reason,
        )
        delta_message = self._coalesce_structured_json_delta(delta_message)

        choice = None
        if (
            output.finish_reason
            and delta_message is not None
            and not delta_message.tool_calls
            and not delta_message.content
            and delta_message.reasoning
            and self._extract_forced_tool_calls_from_json(delta_message.reasoning)
        ):
            delta_message = None
        if delta_message is None:
            if self.in_progress_tool_calls:
                choice = self._emit_tool_calls_choice(output)
            elif output.finish_reason:
                choice = self._build_choice(output, {})
            else:
                self._log_forced_tool_state(
                    "suppressed_chunk_no_choice",
                    finish_reason=output.finish_reason,
                )
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
            has_tool_calls = bool(self.in_progress_tool_calls)
            if has_tool_calls:
                delta["tool_calls"] = self._dump_in_progress_tool_calls()
                self.in_progress_tool_calls.clear()
            if len(delta) > 1:
                choice = self._build_choice(output, delta)
                if has_tool_calls and choice:
                    choice["finish_reason"] = "tool_calls"
                elif choice and self._structured_response_completed:
                    choice["finish_reason"] = "stop"
        elif self.in_progress_tool_calls:
            choice = self._emit_tool_calls_choice(output)
        elif output.finish_reason:
            choice = self._build_choice(output, {})
        elif self._structured_response_completed:
            choice = self._build_choice(output, {})
            choice["finish_reason"] = "stop"

        self.previous_text = current_text
        self.previous_token_ids = current_token_ids
        self._log_forced_tool_state(
            "process_output_end",
            finish_reason=output.finish_reason,
            emitted_choice=choice,
        )
        self._log_structured_state(
            "process_output_end",
            finish_reason=output.finish_reason,
            emitted_choice=choice,
            delta_has_reasoning=bool(delta_message and delta_message.reasoning),
            delta_has_content=bool(delta_message and delta_message.content),
        )
        return choice
