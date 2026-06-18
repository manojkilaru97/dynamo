#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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


def _uses_forced_tool_choice(tool_choice: Any) -> bool:
    if tool_choice == "required":
        return True
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        return isinstance(function, dict) and bool(function.get("name"))
    return bool(getattr(getattr(tool_choice, "function", None), "name", None))


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
    if isinstance(request, ChatCompletionRequest):
        request_for_sampling = request
    elif SKIP_REQUEST_VALIDATION:
        # Trusted fast path; caller must provide OpenAI-compatible payload.
        request_for_sampling = ChatCompletionRequest.model_construct(**request)
        if request_for_sampling.tools and any(
            not hasattr(tool, "model_dump") for tool in request_for_sampling.tools
        ):
            request_for_sampling = ChatCompletionRequest.model_validate(request)
        elif request_for_sampling.response_format or request_for_sampling.structured_outputs:
            # Validation materializes response_format into structured_outputs.
            # Without this, guided decoding constraints can be lost before the
            # backend sees the request.
            request_for_sampling = ChatCompletionRequest.model_validate(request)
    else:
        request_for_sampling = ChatCompletionRequest.model_validate(request)

    tool_parser: ToolParser | None = None
    # With enable_auto_tool_choice the model may emit tool calls even when the
    # client did not supply an explicit `tools` list, so we activate the parser
    # whenever the tool_parser_class is available.
    has_tools = bool(request_for_sampling.tools)
    if tool_parser_class and (has_tools or enable_auto_tool_choice):
        if request_for_sampling.tool_choice != "none":
            tool_parser = tool_parser_class(tokenizer)
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
    chat_template_kwargs = dict(request_for_sampling.chat_template_kwargs or {})
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
    renderer,
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
        guided_tool_choice: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.request_for_sampling = request_for_sampling
        self.guided_tool_choice = (
            request_for_sampling.tool_choice
            if guided_tool_choice is None
            else guided_tool_choice
        )
        self.sampling_params = sampling_params
        self.chat_template_kwargs = chat_template_kwargs
        self.tool_parser = tool_parser
        self.reasoning_parser = (
            reasoning_parser_class(
                tokenizer,
                chat_template_kwargs=chat_template_kwargs,
            )
            if reasoning_parser_class
            else None
        )
        self._fast_plain_text = (
            self.tool_parser is None and self.reasoning_parser is None
        )

        self._control_markers = tuple(
            t for t in getattr(tokenizer, "all_special_tokens", ()) if t
        )

        self.previous_text = ""
        self.previous_token_ids: list[int] = []
        self.reasoning_is_done = False
        self.in_progress_tool_calls: dict[int, DeltaToolCall] = {}
        # Buffer for post-reasoning tool text when </think> and <tool_call>
        # arrive in the same chunk.  The streaming tool parser cannot handle
        # this correctly, so we accumulate text here and fall back to the
        # non-streaming extract_tool_calls() once the buffer is complete.
        self._tool_text_buffer: str | None = None
        self._guided_tool_choice_buffer: str = ""

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

    @staticmethod
    def _tool_choice_function_name(tool_choice: Any) -> str | None:
        if isinstance(tool_choice, dict):
            function = tool_choice.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                return name if isinstance(name, str) and name else None
        name = getattr(getattr(tool_choice, "function", None), "name", None)
        return name if isinstance(name, str) and name else None

    @staticmethod
    def _tool_name(tool: Any) -> str | None:
        function = getattr(tool, "function", None)
        name = getattr(function, "name", None)
        if isinstance(name, str) and name:
            return name
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                return name if isinstance(name, str) and name else None
        return None

    def _infer_guided_tool_choice(self) -> Any | None:
        structured_outputs = self.request_for_sampling.structured_outputs
        if not structured_outputs or not self.request_for_sampling.tools:
            return None

        json_schema = getattr(structured_outputs, "json", None)
        if not isinstance(json_schema, dict):
            return None
        if json_schema.get("type") == "array":
            return "required"

        tools = list(self.request_for_sampling.tools or [])
        if len(tools) == 1:
            name = self._tool_name(tools[0])
            if name:
                return {"type": "function", "function": {"name": name}}
        return None

    def _uses_guided_tool_choice_json(self) -> bool:
        tool_choice = self.guided_tool_choice or self._infer_guided_tool_choice()
        return (
            tool_choice == "required"
            or (
                not isinstance(tool_choice, str)
                and self._tool_choice_function_name(tool_choice)
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
            return self._compose_delta_message(saved_reasoning, None)

        return self._compose_delta_message(saved_reasoning, extracted.content or None)

    @staticmethod
    def _json_arguments(arguments: Any) -> str:
        if isinstance(arguments, str):
            return arguments
        return json.dumps(
            arguments if arguments is not None else {},
            separators=(",", ":"),
        )

    def _guided_tool_choice_to_delta(self, text: str) -> DeltaMessage | None:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None

        tool_choice = self.guided_tool_choice or self._infer_guided_tool_choice()
        tool_calls: list[DeltaToolCall] = []

        if tool_choice == "required":
            if not isinstance(value, list):
                return None
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    return None
                name = item.get("name")
                arguments = item.get("parameters", item.get("arguments"))
                if not isinstance(name, str) or not name:
                    return None
                tool_calls.append(
                    DeltaToolCall(
                        index=index,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=name,
                            arguments=self._json_arguments(arguments),
                        ),
                    )
                )
        else:
            name = self._tool_choice_function_name(tool_choice)
            if not isinstance(name, str) or not name:
                return None
            tool_calls.append(
                DeltaToolCall(
                    index=0,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=name,
                        arguments=self._json_arguments(value),
                    ),
                )
            )

        return DeltaMessage(tool_calls=tool_calls)

    def _buffer_guided_tool_choice_content(
        self, delta_message: DeltaMessage, output: Any
    ) -> DeltaMessage | None:
        self._guided_tool_choice_buffer += delta_message.content or ""
        if output.finish_reason:
            buffered_text = self._guided_tool_choice_buffer
            self._guided_tool_choice_buffer = ""
            parsed_delta = self._guided_tool_choice_to_delta(buffered_text)
            if parsed_delta is not None:
                return parsed_delta
            return DeltaMessage(content=buffered_text)
        return self._compose_delta_message(delta_message.reasoning, None)

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
            "finish_reason": output.finish_reason,
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

            # When reasoning ends in this chunk, reset accumulated state.
            # If there is post-reasoning content (e.g. <tool_call> markup),
            # buffer it for non-streaming extraction rather than feeding it
            # to the streaming tool parser which cannot handle the combined
            # reasoning-end + tool-start in a single chunk.
            if self.reasoning_parser.is_reasoning_end_streaming(
                current_token_ids, delta_token_ids
            ):
                self.reasoning_is_done = True
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
            elif (
                delta_message
                and delta_message.content
                and not delta_message.reasoning
                and (self._uses_guided_tool_choice_json() or self._should_parse_tools())
            ):
                # Reasoning parser returned content (not reasoning).
                # The model may have skipped reasoning and gone straight
                # to tool calls (e.g. Mistral [TOOL_CALLS] without
                # [THINK]...[/THINK]).  Let the tool parser decide.
                if self._uses_guided_tool_choice_json():
                    delta_message = self._buffer_guided_tool_choice_content(
                        delta_message, output
                    )
                else:
                    delta_message = self._extract_tool_calls_streaming(
                        current_text=current_text,
                        delta_text=delta_text,
                        current_token_ids=current_token_ids,
                        delta_token_ids=delta_token_ids,
                    )
        else:
            if (
                self._uses_guided_tool_choice_json()
                and delta_message
                and delta_message.content
                and not delta_message.reasoning
            ):
                delta_message = self._buffer_guided_tool_choice_content(
                    delta_message, output
                )
            elif self._should_parse_tools():
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

        if (
            self._uses_guided_tool_choice_json()
            and output.finish_reason
            and self._guided_tool_choice_buffer
            and (delta_message is None or not delta_message.content)
        ):
            buffered_text = self._guided_tool_choice_buffer
            self._guided_tool_choice_buffer = ""
            delta_message = self._guided_tool_choice_to_delta(buffered_text)
            if delta_message is None:
                delta_message = DeltaMessage(content=buffered_text)

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
