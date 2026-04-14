#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage

import dynamo.frontend.vllm_processor as vllm_processor_module
from dynamo.frontend.prepost import (
    StreamingPostProcessor,
    _decode_forced_tool_json,
    _decode_forced_tool_json_with_prefix,
    _prepare_request,
)
from dynamo.frontend.utils import PreprocessError
from dynamo.frontend.vllm_processor import VllmProcessor, _validate_grammar_constraint


class _DummyToolParser:
    def extract_tool_calls_streaming(self, **kwargs):
        return DeltaMessage(content=kwargs["delta_text"])


class _DummyMiniMaxReasoningParser:
    def __init__(self, tokenizer, chat_template_kwargs=None):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs


_DummyMiniMaxReasoningParser.__name__ = "MiniMaxM2AppendThinkReasoningParser"


class _DummySplitReasoningParser:
    def __init__(self, tokenizer, chat_template_kwargs=None):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs

    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ):
        return DeltaMessage(
            reasoning="reasoning prefix\n",
            content=(
                "{ date_range?, metadata?, tags? }\n"
                "- options: { sort_by?, ascending?, limit?, offset? }\n"
                'I will use "pdf docx".\n'
                '{\n  "query": "pdf docx"\n}'
            ),
        )

    def is_reasoning_end_streaming(self, current_token_ids, delta_token_ids):
        return False


_DummySplitReasoningParser.__name__ = "MiniMaxM2AppendThinkReasoningParser"


class _DummyTokenizer:
    pass


def test_prepare_request_defaults_minimax_enable_thinking_for_structured_outputs():
    request_for_sampling, _, chat_template_kwargs, messages_for_render, chat_params = _prepare_request(
        {
            "model": "minimaxai/minimax-m2.7",
            "messages": [{"role": "user", "content": "Answer yes or no only."}],
            "structured_outputs": {"choice": ["yes", "no"]},
        },
        tokenizer=_DummyTokenizer(),
        tool_parser_class=None,
        reasoning_parser_class=_DummyMiniMaxReasoningParser,
    )

    assert request_for_sampling.chat_template_kwargs == {"enable_thinking": True}
    assert chat_template_kwargs["enable_thinking"] is True
    assert chat_params.chat_template_kwargs["enable_thinking"] is True
    assert messages_for_render[0]["role"] == "system"
    assert "Structured output requirement" in messages_for_render[0]["content"]
    assert '["yes", "no"]' in messages_for_render[0]["content"]


def test_prepare_request_injects_structured_json_hint_for_minimax():
    _, _, _, messages_for_render, _ = _prepare_request(
        {
            "model": "minimaxai/minimax-m2.7",
            "messages": [{"role": "user", "content": "Return a ticket summary object."}],
            "structured_outputs": {
                "json": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["id", "summary"],
                    "additionalProperties": False,
                }
            },
        },
        tokenizer=_DummyTokenizer(),
        tool_parser_class=None,
        reasoning_parser_class=_DummyMiniMaxReasoningParser,
    )

    assert messages_for_render[0]["role"] == "system"
    assert "matches this schema exactly" in messages_for_render[0]["content"]
    assert '"summary"' in messages_for_render[0]["content"]


def test_prepare_request_normalizes_response_format_for_minimax():
    request_for_sampling, _, chat_template_kwargs, messages_for_render, chat_params = (
        _prepare_request(
            {
                "model": "minimaxai/minimax-m2.7",
                "messages": [
                    {"role": "user", "content": "Return a ticket summary object."}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ticket_summary",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "summary": {"type": "string"},
                            },
                            "required": ["id", "summary"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
            tokenizer=_DummyTokenizer(),
            tool_parser_class=None,
            reasoning_parser_class=_DummyMiniMaxReasoningParser,
        )
    )

    assert request_for_sampling.structured_outputs is not None
    assert request_for_sampling.structured_outputs.json is not None
    assert chat_template_kwargs["enable_thinking"] is True
    assert chat_params.chat_template_kwargs["enable_thinking"] is True
    assert messages_for_render[0]["role"] == "system"
    assert '"summary"' in messages_for_render[0]["content"]


def test_prepare_request_respects_explicit_enable_thinking_override():
    request_for_sampling, _, chat_template_kwargs, _, chat_params = _prepare_request(
        {
            "model": "minimaxai/minimax-m2.7",
            "messages": [{"role": "user", "content": "Answer yes or no only."}],
            "structured_outputs": {"choice": ["yes", "no"]},
            "chat_template_kwargs": {"enable_thinking": False},
        },
        tokenizer=_DummyTokenizer(),
        tool_parser_class=None,
        reasoning_parser_class=_DummyMiniMaxReasoningParser,
    )

    assert request_for_sampling.chat_template_kwargs == {"enable_thinking": False}
    assert chat_template_kwargs["enable_thinking"] is False
    assert chat_params.chat_template_kwargs["enable_thinking"] is False


def test_prepare_request_defaults_minimax_enable_thinking_for_forced_tool_choice():
    request_for_sampling, _, chat_template_kwargs, _, chat_params = _prepare_request(
        {
            "model": "minimaxai/minimax-m2.7",
            "messages": [{"role": "user", "content": "Calculate 2+2."}],
            "tool_choice": {"type": "function", "function": {"name": "calculate"}},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                        },
                    },
                }
            ],
        },
        tokenizer=_DummyTokenizer(),
        tool_parser_class=None,
        reasoning_parser_class=_DummyMiniMaxReasoningParser,
    )

    assert request_for_sampling.chat_template_kwargs == {"enable_thinking": True}
    assert chat_template_kwargs["enable_thinking"] is True
    assert chat_params.chat_template_kwargs["enable_thinking"] is True


def test_prepare_request_renders_only_forced_named_tool_in_template():
    request_for_sampling, _, _, _, chat_params = _prepare_request(
        {
            "model": "minimaxai/minimax-m2.7",
            "messages": [{"role": "user", "content": "Calculate 2+2."}],
            "tool_choice": {"type": "function", "function": {"name": "calculate"}},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_documents",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                },
            ],
        },
        tokenizer=_DummyTokenizer(),
        tool_parser_class=None,
        reasoning_parser_class=None,
    )

    assert len(request_for_sampling.tools) == 1
    assert request_for_sampling.tools[0].function.name == "calculate"
    rendered_tools = chat_params.chat_template_kwargs["tools"]
    assert [tool["function"]["name"] for tool in rendered_tools] == ["calculate"]


def _make_output(text: str, finish_reason: str | None = None):
    return SimpleNamespace(
        index=0,
        token_ids=[],
        text=text,
        finish_reason=finish_reason,
        logprobs=None,
    )


def test_forced_tool_choice_emits_tool_calls_from_streamed_content_json():
    request_for_sampling = SimpleNamespace(
        tool_choice={"type": "function", "function": {"name": "calculate"}},
        structured_outputs=None,
    )
    post = StreamingPostProcessor(
        tokenizer=SimpleNamespace(all_special_tokens=()),
        request_for_sampling=request_for_sampling,
        sampling_params=SimpleNamespace(),
        prompt_token_ids=[],
        tool_parser=_DummyToolParser(),
        reasoning_parser_class=None,
        chat_template_kwargs={},
        structured_decoding_active=True,
    )

    assert post.process_output(_make_output('{"expression": "')) is None

    choice = post.process_output(_make_output('2+2"}'))

    assert choice is not None
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["delta"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "calculate"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "expression": "2+2"
    }


def test_forced_tool_choice_parses_first_complete_json_prefix():
    request_for_sampling = SimpleNamespace(
        tool_choice={"type": "function", "function": {"name": "calculate"}},
        structured_outputs=None,
    )
    post = StreamingPostProcessor(
        tokenizer=SimpleNamespace(all_special_tokens=()),
        request_for_sampling=request_for_sampling,
        sampling_params=SimpleNamespace(),
        prompt_token_ids=[],
        tool_parser=_DummyToolParser(),
        reasoning_parser_class=None,
        chat_template_kwargs={},
        structured_decoding_active=True,
    )

    assert post.process_output(_make_output('{"expression": "')) is None

    choice = post.process_output(_make_output('2+2"}{"expression":"repeat"}'))

    assert choice is not None
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["delta"]["tool_calls"]
    assert len(tool_calls) == 1
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "expression": "2+2"
    }


def test_forced_tool_choice_ignores_reasoning_prefix_before_json():
    request_for_sampling = SimpleNamespace(
        tool_choice={"type": "function", "function": {"name": "calculate"}},
        structured_outputs=None,
    )
    post = StreamingPostProcessor(
        tokenizer=SimpleNamespace(all_special_tokens=()),
        request_for_sampling=request_for_sampling,
        sampling_params=SimpleNamespace(),
        prompt_token_ids=[],
        tool_parser=_DummyToolParser(),
        reasoning_parser_class=None,
        chat_template_kwargs={},
        structured_decoding_active=True,
    )

    choice = post.process_output(
        _make_output('Let me reason briefly.</think>{"expression":"2+2"}')
    )

    assert choice is not None
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["delta"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "calculate"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "expression": "2+2"
    }


def test_decode_forced_tool_json_recovers_later_json_candidate():
    parsed = _decode_forced_tool_json(
        'reasoning with {noise not json\n</think>{"expression":"2+2"}',
        prefer_array=False,
        allow_trailing_whitespace_repair=True,
    )

    assert parsed == {"expression": "2+2"}


def test_decode_forced_tool_json_with_prefix_recovers_schema_and_prose_suffix():
    decoded = _decode_forced_tool_json_with_prefix(
        '{ date_range?, metadata?, tags? }\n'
        "- options: { sort_by?, ascending?, limit?, offset? }\n"
        'I will use "pdf docx".\n'
        '{\n  "query": "pdf docx"\n}',
        prefer_array=False,
        allow_trailing_whitespace_repair=True,
    )

    assert decoded is not None
    parsed, prefix, parsed_json = decoded
    assert parsed == {"query": "pdf docx"}
    assert parsed_json == '{\n  "query": "pdf docx"\n}'
    assert prefix.startswith("{ date_range?, metadata?, tags? }")
    assert prefix.endswith('I will use "pdf docx".\n')


def test_final_forced_tool_choice_coalescer_recovers_from_buffered_content_json():
    post = SimpleNamespace(
        request_for_sampling=SimpleNamespace(
            tool_choice={"type": "function", "function": {"name": "search_documents"}}
        ),
        _forced_tool_reasoning_buffer=(
            "The tool cannot search file paths directly.\n"
            "{ date_range?, metadata?, tags? }\n"
            "- options: { sort_by?, ascending?, limit?, offset? }\n"
            'I will use "pdf docx".\n'
        ),
        _forced_tool_content_buffer='{\n  "query": "pdf docx"\n}',
        _forced_tool_json_buffer=None,
    )

    choice = {
        "delta": {
            "role": "assistant",
            "reasoning_content": "",
            "content": "",
        },
        "finish_reason": "stop",
    }

    recovered = VllmProcessor._coalesce_final_forced_tool_choice(
        post,
        {"tool_choice": {"type": "function", "function": {"name": "search_documents"}}},
        choice,
    )

    assert recovered["finish_reason"] == "tool_calls"
    tool_calls = recovered["delta"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "search_documents"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "query": "pdf docx"
    }
    assert "file paths directly" in recovered["delta"]["reasoning_content"]
    assert "date_range?" in recovered["delta"]["reasoning_content"]


def test_forced_tool_choice_recovers_without_tool_parser():
    request_for_sampling = SimpleNamespace(
        tool_choice={"type": "function", "function": {"name": "search_documents"}},
        structured_outputs=None,
    )
    post = StreamingPostProcessor(
        tokenizer=SimpleNamespace(all_special_tokens=()),
        request_for_sampling=request_for_sampling,
        sampling_params=SimpleNamespace(),
        prompt_token_ids=[],
        tool_parser=None,
        reasoning_parser_class=None,
        chat_template_kwargs={},
        structured_decoding_active=True,
    )

    choice = post.process_output(
        _make_output(
            '{ date_range?, metadata?, tags? }\n'
            "- options: { sort_by?, ascending?, limit?, offset? }\n"
            'I will use "pdf docx".\n'
            '{\n  "query": "pdf docx"\n}'
        )
    )

    assert choice is not None
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["delta"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "search_documents"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "query": "pdf docx"
    }


def test_forced_tool_choice_recovers_from_single_finish_chunk_with_reasoning_and_content():
    request_for_sampling = SimpleNamespace(
        tool_choice={"type": "function", "function": {"name": "search_documents"}},
        structured_outputs=None,
    )
    post = StreamingPostProcessor(
        tokenizer=SimpleNamespace(all_special_tokens=()),
        request_for_sampling=request_for_sampling,
        sampling_params=SimpleNamespace(),
        prompt_token_ids=[],
        tool_parser=None,
        reasoning_parser_class=_DummySplitReasoningParser,
        chat_template_kwargs={},
        structured_decoding_active=True,
    )

    helper_post = StreamingPostProcessor(
        tokenizer=SimpleNamespace(all_special_tokens=()),
        request_for_sampling=request_for_sampling,
        sampling_params=SimpleNamespace(),
        prompt_token_ids=[],
        tool_parser=None,
        reasoning_parser_class=_DummySplitReasoningParser,
        chat_template_kwargs={},
        structured_decoding_active=True,
    )

    assert (
        helper_post._recover_forced_tool_calls_from_text_with_prefix(
            "reasoning prefix\n"
            "{ date_range?, metadata?, tags? }\n"
            "- options: { sort_by?, ascending?, limit?, offset? }\n"
            'I will use "pdf docx".\n'
            '{\n  "query": "pdf docx"\n}'
        )
        is not None
    )

    choice = post.process_output(_make_output("ignored", finish_reason="stop"))

    assert choice is not None
    assert choice["finish_reason"] == "tool_calls"
    assert "reasoning prefix" in choice["delta"]["reasoning_content"]
    assert "date_range?" in choice["delta"]["reasoning_content"]
    tool_calls = choice["delta"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "search_documents"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "query": "pdf docx"
    }


def test_forced_tool_choice_repairs_unterminated_json_after_whitespace_tail():
    request_for_sampling = SimpleNamespace(
        tool_choice={"type": "function", "function": {"name": "search_documents"}},
        structured_outputs=None,
    )
    post = StreamingPostProcessor(
        tokenizer=SimpleNamespace(all_special_tokens=()),
        request_for_sampling=request_for_sampling,
        sampling_params=SimpleNamespace(),
        prompt_token_ids=[],
        tool_parser=_DummyToolParser(),
        reasoning_parser_class=None,
        chat_template_kwargs={},
        structured_decoding_active=True,
    )

    assert (
        post.process_output(
            _make_output(
                '{"query":"English text עברית النص العربي"',
            )
        )
        is None
    )

    choice = post.process_output(_make_output(" " * 40))

    assert choice is not None
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["delta"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "search_documents"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "query": "English text עברית النص العربي"
    }


def test_minimax_keeps_reasoning_parser_when_structured_decoding_is_active():
    request_for_sampling = SimpleNamespace(
        tool_choice={"type": "function", "function": {"name": "calculate"}},
        structured_outputs=SimpleNamespace(json={"type": "object"}),
    )

    post = StreamingPostProcessor(
        tokenizer=SimpleNamespace(all_special_tokens=()),
        request_for_sampling=request_for_sampling,
        sampling_params=SimpleNamespace(),
        prompt_token_ids=[],
        tool_parser=_DummyToolParser(),
        reasoning_parser_class=_DummyMiniMaxReasoningParser,
        chat_template_kwargs={"thinking": True},
        structured_decoding_active=True,
    )

    assert post.reasoning_parser is not None


class _FakeGrammarCompiler:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.compiled: list[str] = []

    def compile_grammar(self, grammar: str) -> None:
        self.compiled.append(grammar)
        if self.error is not None:
            raise self.error


def test_validate_grammar_constraint_rejects_invalid_grammar():
    compiler = _FakeGrammarCompiler(
        RuntimeError("EBNF lexer error at line 2, column 10")
    )

    with pytest.raises(PreprocessError) as excinfo:
        _validate_grammar_constraint(
            'root ::= "A"\nws ::= [ \n]*\n',
            tokenizer=SimpleNamespace(),
            compiler=compiler,
        )

    assert compiler.compiled == ['root ::= "A"\nws ::= [ \n]*\n']
    error = excinfo.value.error_dict["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "structured_outputs.grammar"
    assert error["code"] == "invalid_grammar"
    assert "EBNF lexer error" in error["message"]


@pytest.mark.asyncio
async def test_generator_inner_carries_cache_salt_into_dynamo_request(monkeypatch):
    processor = VllmProcessor.__new__(VllmProcessor)
    processor.tokenizer = SimpleNamespace(eos_token_ids=[0])
    processor.input_processor = SimpleNamespace(
        renderer=SimpleNamespace(),
        generation_config_fields={},
        process_inputs=lambda request_id, prompt_inputs, sampling_params, supported_tasks: SimpleNamespace(
            request_id=request_id,
            sampling_params=sampling_params,
        ),
    )
    processor.output_processor = SimpleNamespace()
    processor.tool_parser_class = None
    processor.reasoning_parser_class = None
    processor._xgrammar_compiler = None

    request = {
        "model": "minimaxai/minimax-m2.7",
        "tool_choice": {"type": "function", "function": {"name": "calculate"}},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                    },
                },
            }
        ],
    }
    pre = SimpleNamespace(
        request_for_sampling=SimpleNamespace(
            max_completion_tokens=None,
            max_tokens=16,
            logprobs=None,
            top_logprobs=None,
            cache_salt=None,
            mm_processor_kwargs=None,
            structured_outputs=None,
            tool_choice=request["tool_choice"],
            tools=request["tools"],
            response_format=None,
        ),
        tool_parser=None,
        chat_template_kwargs={},
        engine_prompt={},
        prompt_token_ids=[1, 2, 3],
    )

    async def fake_preprocess_chat_request(*args, **kwargs):
        return pre

    captured: dict[str, object] = {}

    async def fake_generate_and_stream(
        request_id,
        forwarded_request,
        dynamo_preproc,
        tokens,
        vllm_preproc,
        post,
    ):
        captured["request_id"] = request_id
        captured["request"] = forwarded_request
        captured["dynamo_preproc"] = dynamo_preproc
        if False:
            yield {}

    monkeypatch.setattr(
        vllm_processor_module,
        "preprocess_chat_request",
        fake_preprocess_chat_request,
    )
    monkeypatch.setattr(vllm_processor_module, "random_uuid", lambda: "req-cache-salt")
    monkeypatch.setattr(processor, "_generate_and_stream", fake_generate_and_stream)

    results = [item async for item in processor._generator_inner(request)]

    assert results == []
    assert captured["request_id"] == "req-cache-salt"
    dynamo_preproc = captured["dynamo_preproc"]
    assert isinstance(dynamo_preproc, dict)
    assert dynamo_preproc["cache_salt"] == "req-cache-salt"


@pytest.mark.asyncio
async def test_kv_router_generate_receives_cache_salt():
    class _FakeRouter:
        def __init__(self):
            self.kwargs = None

        async def generate(self, **kwargs):
            self.kwargs = kwargs

            async def _empty_stream():
                if False:
                    yield {}

            return _empty_stream()

    router = _FakeRouter()
    processor = VllmProcessor.__new__(VllmProcessor)
    processor.router = router
    processor.is_kv_router = True
    processor.output_processor = SimpleNamespace(add_request=lambda *args, **kwargs: None)

    async for _ in processor._generate_and_stream(
        "req-cache-salt",
        {"model": "minimaxai/minimax-m2.7", "stream": False},
        {
            "model": "minimaxai/minimax-m2.7",
            "token_ids": [1, 2, 3],
            "cache_salt": "req-cache-salt",
            "stop_conditions": {},
            "sampling_options": {},
            "output_options": {},
        },
        [1, 2, 3],
        SimpleNamespace(request_id="req-cache-salt"),
        SimpleNamespace(request_for_sampling=SimpleNamespace(tool_choice="auto")),
    ):
        pass

    assert router.kwargs is not None
    assert router.kwargs["cache_salt"] == "req-cache-salt"


@pytest.mark.asyncio
async def test_generate_and_stream_emits_finish_only_forced_tool_chunk():
    class _FakeRouter:
        async def generate(self, **kwargs):
            async def _stream():
                yield {"token_ids": [], "finish_reason": "stop"}

            return _stream()

    processor = VllmProcessor.__new__(VllmProcessor)
    processor.router = _FakeRouter()
    processor.is_kv_router = True
    processor.output_processor = SimpleNamespace(
        add_request=lambda *args, **kwargs: None,
        process_outputs=lambda outputs: SimpleNamespace(
            request_outputs=[],
            reqs_to_abort=[],
        ),
        abort_requests=lambda *args, **kwargs: None,
        request_states={},
    )

    post = SimpleNamespace(
        request_for_sampling=SimpleNamespace(
            tool_choice={"type": "function", "function": {"name": "search_documents"}}
        ),
        process_output=lambda output: {
            "index": 0,
            "delta": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "type": "function",
                        "id": "call-test",
                        "function": {
                            "name": "search_documents",
                            "arguments": '{"query":"pdf docx"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
        if output.finish_reason == "stop"
        else None,
    )

    results = [
        item
        async for item in processor._generate_and_stream(
            "req-finish-only",
            {
                "model": "minimaxai/minimax-m2.7",
                "stream": True,
                "tool_choice": {"type": "function", "function": {"name": "search_documents"}},
            },
            {
                "model": "minimaxai/minimax-m2.7",
                "token_ids": [],
                "stop_conditions": {},
                "sampling_options": {},
                "output_options": {},
            },
            [],
            SimpleNamespace(request_id="req-finish-only"),
            post,
        )
    ]

    assert len(results) == 1
    choice = results[0]["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["delta"]["tool_calls"][0]["function"]["name"] == "search_documents"
