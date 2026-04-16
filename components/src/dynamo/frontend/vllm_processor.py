#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

#
# Use vllm for input and output processing
#

import asyncio
import json
import logging
import os
import re
import time
from argparse import Namespace
from collections.abc import AsyncGenerator
from typing import Any

from vllm.config import CacheConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
)
from vllm.inputs.data import TokensPrompt
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tasks import GENERATION_TASKS
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager
from vllm.tool_parsers.utils import get_json_schema_from_tools
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, OutputProcessorOutput

from dynamo._internal import ModelDeploymentCard
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

from .prepost import (
    StreamingPostProcessor,
    _decode_forced_tool_json,
    _decode_forced_tool_json_with_prefix,
    _strip_non_json_prefix,
    preprocess_chat_request,
)
from .utils import PreprocessError, random_uuid

try:
    import xgrammar as xgr
except ImportError:  # pragma: no cover - depends on runtime environment
    xgr = None

logger = logging.getLogger(__name__)


_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "eos": FinishReason.STOP,
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "error": FinishReason.ERROR,
    "cancelled": FinishReason.ABORT,
    "content_filter": FinishReason.STOP,
}


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


def _invalid_request_error(
    message: str,
    *,
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "message": message,
        "type": "invalid_request_error",
    }
    if param is not None:
        error["param"] = param
    if code is not None:
        error["code"] = code
    return {"error": error}


def _validate_grammar_constraint(
    grammar: str,
    tokenizer: TokenizerLike,
    *,
    compiler: Any | None = None,
) -> None:
    if not grammar:
        return
    if compiler is None and xgr is None:
        return

    try:
        active_compiler = compiler
        if active_compiler is None:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
            active_compiler = xgr.GrammarCompiler(tokenizer_info)
        active_compiler.compile_grammar(grammar)
    except Exception as exc:
        raise PreprocessError(
            _invalid_request_error(
                f"Invalid structured_outputs.grammar: {exc}",
                param="structured_outputs.grammar",
                code="invalid_grammar",
            )
        ) from exc


def _structured_outputs_to_guided_decoding(
    structured_outputs: Any | None,
) -> dict[str, Any] | None:
    """Translate vLLM structured outputs into Dynamo's guided_decoding payload."""
    if structured_outputs is None:
        return None

    def _get_field(name: str) -> Any:
        if isinstance(structured_outputs, dict):
            return structured_outputs.get(name)
        return getattr(structured_outputs, name, None)

    guided_decoding: dict[str, Any] = {}
    for field in (
        "json",
        "regex",
        "choice",
        "grammar",
        "json_object",
        "whitespace_pattern",
        "structural_tag",
        "disable_fallback",
        "disable_any_whitespace",
        "disable_additional_properties",
    ):
        value = _get_field(field)
        if field == "json_object" and value is False:
            value = None
        if value is not None:
            guided_decoding[field] = value

    return guided_decoding or None


def _response_format_to_guided_decoding(
    response_format: Any | None,
) -> dict[str, Any] | None:
    if response_format is None:
        return None

    if isinstance(response_format, dict):
        rf_type = response_format.get("type")
        json_schema = response_format.get("json_schema")
    else:
        rf_type = getattr(response_format, "type", None)
        json_schema = getattr(response_format, "json_schema", None)

    if rf_type == "json_object":
        return {"json_object": True}

    if rf_type != "json_schema" or json_schema is None:
        return None

    if isinstance(json_schema, dict):
        schema = json_schema.get("schema") or json_schema.get("json_schema")
    else:
        schema = getattr(json_schema, "schema", None) or getattr(
            json_schema, "json_schema", None
        )
    if schema is None:
        return None
    return {"json": schema}


def _tighten_json_guided_decoding(
    guided_decoding: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if guided_decoding is None:
        return None

    # Forced-tool and structured JSON requests do not benefit from unlimited
    # inter-token whitespace. MiniMax can legally loop on spaces/newlines
    # before the next required JSON token, which makes streamed tool calls
    # look like a hang. Keep explicit caller whitespace settings intact.
    if (
        (guided_decoding.get("json") is not None or guided_decoding.get("json_object"))
        and "disable_any_whitespace" not in guided_decoding
        and "whitespace_pattern" not in guided_decoding
    ):
        guided_decoding = dict(guided_decoding)
        guided_decoding["disable_any_whitespace"] = True

    return guided_decoding


def _tool_choice_to_guided_decoding(
    tool_choice: Any,
    tools: Any | None,
) -> dict[str, Any] | None:
    if tools is None:
        return None
    json_schema: dict[str, Any] | str | None = None
    if tool_choice == "required" and all(isinstance(tool, dict) for tool in tools):
        any_of: list[dict[str, Any]] = []
        defs: dict[str, Any] = {}
        for tool in tools:
            function = tool.get("function", {})
            name = function.get("name")
            if not name:
                continue
            params = function.get("parameters")
            if params is None:
                params = {"type": "object", "properties": {}}
            elif isinstance(params, dict):
                params = dict(params)
            if isinstance(params, dict):
                tool_defs = params.pop("$defs", {})
                if isinstance(tool_defs, dict):
                    defs.update(tool_defs)
            any_of.append(
                {
                    "properties": {
                        "name": {"type": "string", "enum": [name]},
                        "parameters": params,
                    },
                    "required": ["name", "parameters"],
                }
            )
        if any_of:
            json_schema = {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "anyOf": any_of,
                },
            }
            if defs:
                json_schema["$defs"] = defs
    else:
        json_schema = get_json_schema_from_tools(tool_choice, tools)
    if json_schema is None:
        forced_name = _forced_tool_choice_name(tool_choice)
        if forced_name is not None:
            for tool in tools:
                if isinstance(tool, dict):
                    function = tool.get("function", {})
                    if function.get("name") == forced_name:
                        json_schema = function.get("parameters")
                        break
                elif getattr(getattr(tool, "function", None), "name", None) == forced_name:
                    json_schema = getattr(tool.function, "parameters", None)
                    break
    if json_schema is None:
        return None
    return {"json": json_schema}


def _maybe_wrap_guided_decoding_for_minimax_reasoning(
    guided_decoding: dict[str, Any] | None,
    reasoning_parser_class: type[ReasoningParser] | None,
    *,
    source: str | None = None,
) -> dict[str, Any] | None:
    if guided_decoding is None or reasoning_parser_class is None:
        return guided_decoding

    if reasoning_parser_class.__name__ != "MiniMaxM2AppendThinkReasoningParser":
        return guided_decoding

    content: dict[str, Any] | None = None
    if guided_decoding.get("json") is not None:
        content = {
            "type": "json_schema",
            "json_schema": guided_decoding["json"],
        }
    elif guided_decoding.get("json_object"):
        content = {
            "type": "json_schema",
            "json_schema": {"type": "object"},
        }
    elif guided_decoding.get("regex") is not None:
        content = {
            "type": "regex",
            "pattern": guided_decoding["regex"],
        }
    elif guided_decoding.get("grammar") is not None:
        content = {
            "type": "grammar",
            "grammar": guided_decoding["grammar"],
        }
    elif guided_decoding.get("choice") is not None:
        valid_choices = [c for c in guided_decoding["choice"] if c is not None]
        if valid_choices:
            content = {
                "type": "regex",
                "pattern": "|".join(re.escape(str(c)) for c in valid_choices),
            }

    if content is None:
        return guided_decoding

    wrapped = dict(guided_decoding)
    wrapped.pop("json", None)
    wrapped.pop("json_object", None)
    wrapped.pop("regex", None)
    wrapped.pop("grammar", None)
    wrapped.pop("choice", None)
    wrapped["_dynamo_structural_content_type"] = content["type"]
    wrapped["structural_tag"] = json.dumps(
        {
            "type": "sequence",
            "elements": [
                {
                    "type": "tag",
                    # MiniMax M2 emits reasoning first and closes it with
                    # </think>; it does not reliably emit a leading <think>.
                    "begin": "",
                    "content": {"type": "any_text"},
                    "end": "</think>",
                },
                content,
            ],
        },
        ensure_ascii=False,
    )
    logger.info(
        "Wrapped MiniMax guided decoding for reasoning handoff: source=%s content_type=%s",
        source,
        content["type"],
    )
    return wrapped


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


def _vllm_reasoning_parser_name(name: str | None) -> str | None:
    if name == "minimax_append_think":
        return "minimax_m2_append_think"
    return name


def _parse_forced_tool_choice_calls(
    tool_choice: Any,
    raw_text: str | None,
) -> list[dict[str, Any]]:
    raw = _strip_json_fence(raw_text)
    if not raw:
        return []

    parsed = _decode_forced_tool_json(
        raw,
        prefer_array=_forced_tool_choice_name(tool_choice) is None,
        allow_trailing_whitespace_repair=True,
    )
    if parsed is None:
        return []

    forced_name = _forced_tool_choice_name(tool_choice)
    if forced_name is not None:
        if not isinstance(parsed, dict):
            return []
        return [
            {
                "index": 0,
                "type": "function",
                "id": make_tool_call_id(),
                "function": {
                    "name": forced_name,
                    "arguments": json.dumps(parsed, ensure_ascii=False),
                },
            }
        ]

    if tool_choice != "required":
        return []

    if isinstance(parsed, dict) and "name" in parsed and "parameters" in parsed:
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    tool_calls: list[dict[str, Any]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict) or not item.get("name"):
            continue
        tool_calls.append(
            {
                "index": index,
                "type": "function",
                "id": make_tool_call_id(),
                "function": {
                    "name": item["name"],
                    "arguments": json.dumps(
                        item.get("parameters", {}),
                        ensure_ascii=False,
                    ),
                },
            }
        )
    return tool_calls


def _forced_tool_choice_delta_complete(
    tool_choice: Any,
    choice: dict[str, Any],
) -> bool:
    is_forced = tool_choice == "required" or _forced_tool_choice_name(tool_choice)
    if not is_forced:
        return False

    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return False

    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return False

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            return False
        function = tool_call.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            return False
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            return False
        try:
            json.loads(_strip_json_fence(arguments))
        except Exception:
            return False

    return True


def _request_needs_cache_isolation(
    request: dict[str, Any],
    request_for_sampling: Any,
) -> bool:
    tool_choice = request.get("tool_choice", request_for_sampling.tool_choice)
    if tool_choice == "required" or _forced_tool_choice_name(tool_choice):
        return True

    if getattr(request_for_sampling, "structured_outputs", None) is not None:
        return True

    if request.get("structured_outputs") is not None:
        return True

    response_format = request.get("response_format")
    if isinstance(response_format, dict) and response_format:
        return True

    return False


class VllmProcessor:
    def __init__(
        self,
        tokenizer: TokenizerLike,
        input_processor: InputProcessor,
        router: Any,  # Client or KvRouter
        output_processor: OutputProcessor,
        tool_parser_class: type[ToolParser] | None,
        reasoning_parser_class: type[ReasoningParser] | None,
    ):
        self.tokenizer = tokenizer
        self.input_processor = input_processor
        self.router = router
        self.is_kv_router = isinstance(router, KvRouter)
        self.output_processor = output_processor
        self.tool_parser_class = tool_parser_class
        self.reasoning_parser_class = reasoning_parser_class
        self._xgrammar_compiler: Any | None = None

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

    def _get_xgrammar_compiler(self) -> Any | None:
        if xgr is None:
            return None
        if self._xgrammar_compiler is None:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer)
            self._xgrammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        return self._xgrammar_compiler

    def _validate_guided_decoding(
        self,
        guided_decoding: dict[str, Any] | None,
    ) -> None:
        if guided_decoding is None:
            return
        grammar = guided_decoding.get("grammar")
        if isinstance(grammar, str):
            _validate_grammar_constraint(
                grammar,
                self.tokenizer,
                compiler=self._get_xgrammar_compiler(),
            )

    @staticmethod
    def _coalesce_final_forced_tool_choice(
        post: StreamingPostProcessor,
        request: dict[str, Any],
        choice: dict[str, Any],
    ) -> dict[str, Any]:
        tool_choice = request.get("tool_choice", post.request_for_sampling.tool_choice)
        is_forced = tool_choice == "required" or _forced_tool_choice_name(
            tool_choice
        ) is not None
        if not is_forced:
            return choice

        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            return choice

        existing_tool_calls = delta.get("tool_calls")
        if existing_tool_calls:
            setattr(post, "_forced_tool_reasoning_buffer", "")
            setattr(post, "_forced_tool_content_buffer", "")
            return choice

        reasoning = delta.get("reasoning_content")
        if reasoning:
            current = getattr(post, "_forced_tool_reasoning_buffer", "")
            setattr(post, "_forced_tool_reasoning_buffer", current + reasoning)
        content = delta.get("content")
        if isinstance(content, str) and content:
            current = getattr(post, "_forced_tool_content_buffer", "")
            setattr(post, "_forced_tool_content_buffer", current + content)

        if not choice.get("finish_reason"):
            return choice

        buffered_reasoning = getattr(post, "_forced_tool_reasoning_buffer", "")
        buffered_content = getattr(post, "_forced_tool_content_buffer", "")
        buffered_json = getattr(post, "_forced_tool_json_buffer", "") or ""
        combined = "".join(
            part for part in (buffered_reasoning, buffered_content, buffered_json) if part
        )
        prefer_array = _forced_tool_choice_name(tool_choice) is None
        decoded = _decode_forced_tool_json_with_prefix(
            combined,
            prefer_array=prefer_array,
            allow_trailing_whitespace_repair=True,
        )
        if decoded is None:
            return choice
        parsed, prefix, parsed_json = decoded
        parsed_tool_calls = _parse_forced_tool_choice_calls(tool_choice, parsed_json)
        if not parsed_tool_calls:
            if tool_choice == "required" and isinstance(parsed, dict):
                parsed_tool_calls = _parse_forced_tool_choice_calls(
                    tool_choice,
                    json.dumps([parsed], ensure_ascii=False),
                )
            if not parsed_tool_calls:
                return choice

        new_delta: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": parsed_tool_calls,
        }
        prefix = prefix.strip()
        if prefix:
            new_delta["reasoning_content"] = prefix

        choice["delta"] = new_delta
        choice["finish_reason"] = "tool_calls"
        setattr(post, "_forced_tool_reasoning_buffer", "")
        setattr(post, "_forced_tool_content_buffer", "")
        logger.warning(
            "Recovered forced tool choice from buffered response JSON: tool_choice=%r tool_calls=%d",
            tool_choice,
            len(parsed_tool_calls),
        )
        return choice

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

        async for item in self._generator_inner(request):
            yield item

    async def _generator_inner(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        request_id = random_uuid()

        try:
            pre = await preprocess_chat_request(
                request,
                tokenizer=self.tokenizer,
                renderer=self.input_processor.renderer,
                tool_parser_class=self.tool_parser_class,
                reasoning_parser_class=self.reasoning_parser_class,
            )
        except PreprocessError as exc:
            yield exc.error_dict
            return

        request_for_sampling = pre.request_for_sampling
        tool_parser = pre.tool_parser
        chat_template_kwargs = pre.chat_template_kwargs
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

        # This calls update_from_generation_config and update_from_tokenizer on SamplingParams
        prompt_inputs = TokensPrompt(prompt_token_ids=tokens)
        if "multi_modal_data" in engine_prompt:
            prompt_inputs["multi_modal_data"] = engine_prompt["multi_modal_data"]
        if "multi_modal_uuids" in engine_prompt:
            prompt_inputs["multi_modal_uuids"] = engine_prompt["multi_modal_uuids"]
        if request_for_sampling.cache_salt is not None:
            prompt_inputs["cache_salt"] = request_for_sampling.cache_salt
        elif _request_needs_cache_isolation(request, request_for_sampling):
            prompt_inputs["cache_salt"] = request_id
        if request_for_sampling.mm_processor_kwargs is not None:
            prompt_inputs[
                "mm_processor_kwargs"
            ] = request_for_sampling.mm_processor_kwargs

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
        if sp.n != 1:
            logger.error("Unsupported SamplingParams.n=%d, only n=1 is supported", sp.n)
            yield {
                "error": {
                    "message": (
                        f"Unsupported value: 'n={sp.n}'. "
                        "This endpoint currently supports only n=1."
                    ),
                    "type": "invalid_request_error",
                    "param": "n",
                    "code": "unsupported_value",
                }
            }
            return

        dynamo_preproc = {
            "model": request["model"],
            "token_ids": tokens,
            "cache_salt": prompt_inputs.get("cache_salt"),
            "stop_conditions": {
                "max_tokens": sp.max_tokens,
                "stop": sp.stop,
                "stop_token_ids": sp.stop_token_ids,
                "min_tokens": sp.min_tokens,
                "ignore_eos": sp.ignore_eos,
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
        }
        guided_source: str | None = None
        guided_decoding = _structured_outputs_to_guided_decoding(
            request_for_sampling.structured_outputs
        )
        if guided_decoding is not None:
            guided_source = "structured_outputs"
        if guided_decoding is None:
            guided_decoding = _structured_outputs_to_guided_decoding(
                request.get("structured_outputs")
            )
            if guided_decoding is not None:
                guided_source = "structured_outputs"
        if guided_decoding is None:
            guided_decoding = _tool_choice_to_guided_decoding(
                request_for_sampling.tool_choice,
                request_for_sampling.tools,
            )
            if guided_decoding is not None:
                guided_source = "tool_choice"
        if guided_decoding is None:
            guided_decoding = _response_format_to_guided_decoding(
                request.get("response_format")
            )
            if guided_decoding is not None:
                guided_source = "response_format"
        guided_decoding = _tighten_json_guided_decoding(guided_decoding)
        try:
            self._validate_guided_decoding(guided_decoding)
        except PreprocessError as exc:
            yield exc.error_dict
            return
        guided_decoding = _maybe_wrap_guided_decoding_for_minimax_reasoning(
            guided_decoding,
            self.reasoning_parser_class,
            source=guided_source,
        )
        if guided_decoding is not None:
            # Match Dynamo's Rust OpenAI path: guided decoding requests should
            # not read from prefix cache. Cache reuse can strand reasoning-aware
            # structured outputs under concurrent KV routing.
            sampling_params.skip_reading_prefix_cache = True
        external_request_id = request.get("request_id")
        if external_request_id:
            dynamo_preproc["request_id"] = external_request_id
        if guided_decoding:
            dynamo_preproc["sampling_options"]["guided_decoding"] = guided_decoding
            dynamo_preproc["sampling_options"]["skip_reading_prefix_cache"] = True

        post = StreamingPostProcessor(
            tokenizer=self.tokenizer,
            request_for_sampling=request_for_sampling,
            sampling_params=sampling_params,
            prompt_token_ids=tokens,
            tool_parser=tool_parser,
            reasoning_parser_class=self.reasoning_parser_class,
            chat_template_kwargs=chat_template_kwargs,
            structured_decoding_active=guided_decoding is not None,
        )

        async for item in self._generate_and_stream(
            request_id,
            request,
            dynamo_preproc,
            tokens,
            vllm_preproc,
            post,
        ):
            yield item

    async def _generate_and_stream(
        self,
        request_id: str,
        request: dict[str, Any],
        dynamo_preproc: dict[str, Any],
        tokens: list[int],
        vllm_preproc: EngineCoreRequest,
        post: StreamingPostProcessor,
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.output_processor.add_request(vllm_preproc, None)

        try:
            if self.is_kv_router:
                dynamo_stream = await self.router.generate(
                    token_ids=tokens,
                    model=dynamo_preproc["model"],
                    stop_conditions=dynamo_preproc["stop_conditions"],
                    sampling_options=dynamo_preproc["sampling_options"],
                    output_options=dynamo_preproc["output_options"],
                    cache_salt=dynamo_preproc.get("cache_salt"),
                )
            else:
                dynamo_stream = await self.router.generate(
                    dynamo_preproc, annotated=False
                )

            async for dynamo_response in dynamo_stream:
                if self.is_kv_router:
                    engine_response = dynamo_response
                elif hasattr(dynamo_response, "data"):
                    engine_response = dynamo_response.data()
                else:
                    engine_response = dynamo_response

                if engine_response is None or "token_ids" not in engine_response:
                    logger.error("No outputs from engine for request %s", request_id)
                    yield {
                        "error": {
                            "message": f"Invalid engine response for request {request_id}",
                            "type": "internal_error",
                        }
                    }
                    break

                raw_finish_reason = engine_response.get("finish_reason")
                finish_reason = map_finish_reason(raw_finish_reason)
                stop_reason = engine_response.get("stop_reason")

                vllm_response = EngineCoreOutput(
                    request_id=vllm_preproc.request_id,
                    new_token_ids=engine_response["token_ids"],
                    finish_reason=finish_reason,
                    stop_reason=stop_reason,
                )

                vllm_out: OutputProcessorOutput = self.output_processor.process_outputs(
                    [vllm_response]
                )

                if vllm_out.reqs_to_abort:
                    self.output_processor.abort_requests(
                        vllm_out.reqs_to_abort, internal=True
                    )

                choices = []
                terminate_after_yield = False
                if not vllm_out.request_outputs:
                    if raw_finish_reason:
                        synthetic_output = Namespace(
                            index=0,
                            token_ids=[],
                            text="",
                            finish_reason=raw_finish_reason,
                            logprobs=None,
                        )
                        choice = post.process_output(synthetic_output)
                        if choice:
                            choice = self._coalesce_final_forced_tool_choice(
                                post, request, choice
                            )
                            tool_choice = request.get(
                                "tool_choice", post.request_for_sampling.tool_choice
                            )
                            if (
                                not choice.get("finish_reason")
                                and _forced_tool_choice_delta_complete(
                                    tool_choice, choice
                                )
                            ):
                                choice["finish_reason"] = "tool_calls"
                            if choice.get("finish_reason") == "tool_calls":
                                terminate_after_yield = True
                            elif (
                                choice.get("finish_reason") == "stop"
                                and getattr(post, "_structured_response_completed", False)
                            ):
                                terminate_after_yield = True
                            choices.append(choice)
                            if choices:
                                yield {
                                    "id": request_id,
                                    "choices": choices,
                                    "created": int(time.time()),
                                    "model": request["model"],
                                    "object": "chat.completion.chunk",
                                }
                            if terminate_after_yield:
                                return
                    continue
                for output in vllm_out.request_outputs[0].outputs:
                    choice = post.process_output(output)
                    if choice:
                        choice = self._coalesce_final_forced_tool_choice(
                            post, request, choice
                        )
                        tool_choice = request.get(
                            "tool_choice", post.request_for_sampling.tool_choice
                        )
                        if (
                            not choice.get("finish_reason")
                            and _forced_tool_choice_delta_complete(tool_choice, choice)
                        ):
                            choice["finish_reason"] = "tool_calls"
                        if choice.get("finish_reason") == "tool_calls":
                            terminate_after_yield = True
                        elif (
                            choice.get("finish_reason") == "stop"
                            and getattr(post, "_structured_response_completed", False)
                        ):
                            terminate_after_yield = True
                        choices.append(choice)

                if choices:
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

                if terminate_after_yield:
                    logger.warning(
                        "Ending backend stream early after emitting tool_calls for request %s",
                        request_id,
                    )
                    return
        finally:
            if vllm_preproc.request_id in self.output_processor.request_states:
                self.output_processor.abort_requests(
                    [vllm_preproc.request_id], internal=True
                )


class EngineFactory:
    def __init__(
        self,
        runtime: DistributedRuntime,
        router_config: RouterConfig,
        config: FrontendConfig,
        flags: Namespace,
    ):
        if config.preprocess_workers != 0:
            raise RuntimeError(
                "preprocess_workers > 0 is not supported by vllm preprocessor"
            )

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
        loop = asyncio.get_running_loop()

        source_path = mdc.source_path()
        if not os.path.exists(source_path):
            await fetch_model(source_path, ignore_weights=True)

        tokenizer_mode = getattr(self.flags, "tokenizer_mode", None) or "auto"
        config_format = getattr(self.flags, "config_format", None) or "auto"
        load_format = getattr(self.flags, "load_format", None) or "dummy"

        model_config = ModelConfig(
            model=source_path,
            tokenizer_mode=tokenizer_mode,
            config_format=config_format,
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            load_config=LoadConfig(load_format=load_format),
            cache_config=CacheConfig(),
            # scheduler_config=SchedulerConfig(),
        )

        input_processor = InputProcessor(vllm_config)
        tokenizer = input_processor.get_tokenizer()
        output_processor = OutputProcessor(
            tokenizer,
            log_stats=False,
            stream_interval=self.stream_interval,
        )
        logger.info("vLLM OutputProcessor stream_interval=%d", self.stream_interval)

        tool_parser_name = self.flags.tool_call_parser or mdc.runtime_config().get(
            "tool_call_parser"
        )
        if tool_parser_name:
            tool_parser_class = ToolParserManager.get_tool_parser(tool_parser_name)
        else:
            tool_parser_class = None

        reasoning_parser_name = _vllm_reasoning_parser_name(
            self.flags.reasoning_parser or mdc.runtime_config().get("reasoning_parser")
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

        gen = VllmProcessor(
            tokenizer,
            input_processor,
            router,
            output_processor,
            tool_parser_class,
            reasoning_parser_class,
        )

        return PythonAsyncEngine(gen.generator, loop)
