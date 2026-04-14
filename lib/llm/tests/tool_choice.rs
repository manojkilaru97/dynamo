// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_async_openai::types::{
    ChatCompletionMessageContent, ChatCompletionNamedToolChoice, ChatCompletionRequestMessage,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionToolType,
    CreateChatCompletionRequest, FunctionName, FunctionObject,
};
use dynamo_llm::protocols::common;
use dynamo_llm::protocols::common::llm_backend::BackendOutput;

/// Helper to extract text from ChatCompletionMessageContent
fn get_text(content: &ChatCompletionMessageContent) -> &str {
    match content {
        ChatCompletionMessageContent::Text(text) => text.as_str(),
        ChatCompletionMessageContent::Parts(_) => "",
    }
}
use dynamo_llm::protocols::openai::DeltaGeneratorExt;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

fn create_test_request() -> NvCreateChatCompletionRequest {
    let messages = vec![ChatCompletionRequestMessage::User(
        ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text("test".to_string()),
            name: None,
        },
    )];

    NvCreateChatCompletionRequest {
        inner: CreateChatCompletionRequest {
            model: "test-model".to_string(),
            messages,
            stream: Some(false),
            stream_options: None,
            tools: Some(default_test_tools()),
            ..Default::default()
        },
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
        media_io_kwargs: None,
        request_id: None,
        structured_outputs: None,
        unsupported_fields: Default::default(),
    }
}

fn default_test_tools() -> Vec<ChatCompletionTool> {
    vec![
        test_tool(
            "get_weather",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["location"],
            }),
        ),
        test_tool(
            "search",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            }),
        ),
        test_tool(
            "summarize",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                },
                "required": ["topic"],
            }),
        ),
        test_tool(
            "calculate",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            }),
        ),
        test_tool(
            "search_documents",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "date_range": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                },
                            },
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "ascending": {"type": "boolean"},
                            "limit": {"type": "integer"},
                        },
                    },
                },
                "required": ["query"],
            }),
        ),
    ]
}

fn test_tool(name: &str, parameters: serde_json::Value) -> ChatCompletionTool {
    ChatCompletionTool {
        r#type: ChatCompletionToolType::Function,
        function: FunctionObject {
            name: name.to_string(),
            description: None,
            parameters: Some(parameters),
            strict: None,
        },
    }
}

fn default_parser_tool_definitions() -> Vec<dynamo_parsers::tool_calling::ToolDefinition> {
    default_test_tools()
        .into_iter()
        .map(|tool| dynamo_parsers::tool_calling::ToolDefinition {
            name: tool.function.name,
            parameters: tool.function.parameters,
        })
        .collect()
}

async fn apply_jail_transformation(
    raw_response: dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
) -> dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse {
    use dynamo_llm::protocols::openai::chat_completions::jail::JailedStream;
    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::StreamExt;
    use futures::stream;

    let input_stream = stream::iter(vec![Annotated {
        data: Some(raw_response),
        id: None,
        event: None,
        comment: None,
        error: None,
    }]);

    let mut builder = JailedStream::builder();

    match tool_choice {
        Some(ChatCompletionToolChoiceOption::Named(ref named)) => {
            builder = builder.tool_choice_named(named.function.name.clone());
        }
        Some(ChatCompletionToolChoiceOption::Required) => {
            builder = builder.tool_choice_required();
        }
        _ => {}
    }

    let jail = builder.build();
    let output_stream = jail.apply_with_finish_reason(input_stream);

    tokio::pin!(output_stream);
    output_stream.next().await.unwrap().data.unwrap()
}

async fn apply_jail_transformation_streaming(
    raw_responses: Vec<
        dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
    >,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
) -> Vec<dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse> {
    use dynamo_llm::protocols::openai::chat_completions::jail::JailedStream;
    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::StreamExt;
    use futures::stream;

    let input_stream = stream::iter(raw_responses.into_iter().map(|r| Annotated {
        data: Some(r),
        id: None,
        event: None,
        comment: None,
        error: None,
    }));

    let mut builder = JailedStream::builder();

    match tool_choice {
        Some(ChatCompletionToolChoiceOption::Named(ref named)) => {
            builder = builder.tool_choice_named(named.function.name.clone());
        }
        Some(ChatCompletionToolChoiceOption::Required) => {
            builder = builder.tool_choice_required();
        }
        _ => {}
    }

    let jail = builder.build();
    let output_stream = jail.apply_with_finish_reason(input_stream);

    tokio::pin!(output_stream);
    output_stream
        .filter_map(|ann| async move { ann.data })
        .collect()
        .await
}

fn build_backend_output(text: &str) -> BackendOutput {
    BackendOutput {
        token_ids: vec![],
        tokens: vec![],
        text: Some(text.to_string()),
        cum_log_probs: None,
        log_probs: None,
        top_logprobs: None,
        finish_reason: Some(common::FinishReason::Stop),
        stop_reason: None,
        index: Some(0),
        completion_usage: None,
        disaggregated_params: None,
    }
}

fn build_backend_output_with_finish(
    text: &str,
    finish_reason: Option<common::FinishReason>,
) -> BackendOutput {
    BackendOutput {
        token_ids: vec![],
        tokens: vec![],
        text: Some(text.to_string()),
        cum_log_probs: None,
        log_probs: None,
        top_logprobs: None,
        finish_reason,
        stop_reason: None,
        index: Some(0),
        completion_usage: None,
        disaggregated_params: None,
    }
}

fn move_content_to_reasoning(
    response: &mut dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
) {
    for choice in &mut response.choices {
        choice.delta.reasoning_content = choice.delta.content.take().and_then(|content| {
            if let ChatCompletionMessageContent::Text(text) = content {
                Some(text)
            } else {
                None
            }
        });
    }
}

#[tokio::test]
async fn test_named_tool_choice_parses_json() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "get_weather".to_string(),
            },
        },
    ));
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-1".to_string());
    let backend_output = build_backend_output(r#"{"location":"Paris"}"#);
    let raw_response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    let response = apply_jail_transformation(raw_response, tool_choice).await;
    let choice = &response.choices[0];

    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );
    let delta = &choice.delta;
    assert!(delta.content.is_none() || delta.content.as_ref().map(get_text) == Some(""));
    let tool_calls = delta.tool_calls.as_ref().unwrap();

    assert_eq!(tool_calls.len(), 1);

    let tool_call = &tool_calls[0];
    assert_eq!(tool_call.index, 0);
    assert!(tool_call.id.as_ref().unwrap().starts_with("call-"));
    assert_eq!(tool_call.r#type, Some(ChatCompletionToolType::Function));
    assert_eq!(
        tool_call.function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );
    assert_eq!(
        tool_call.function.as_ref().unwrap().arguments.as_deref(),
        Some(r#"{"location":"Paris"}"#)
    );
}

#[tokio::test]
async fn test_required_tool_choice_parses_json_array() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Required);
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-2".to_string());
    let backend_output = build_backend_output(
        r#"[{"name":"search","parameters":{"query":"rust"}},
            {"name":"summarize","parameters":{"topic":"memory"}}]"#,
    );
    let raw_response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    let response = apply_jail_transformation(raw_response, tool_choice).await;
    let choice = &response.choices[0];

    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ToolCalls)
    );
    let delta = &choice.delta;
    assert!(delta.content.is_none() || delta.content.as_ref().map(get_text) == Some(""));
    let tool_calls = delta.tool_calls.as_ref().unwrap();

    assert_eq!(tool_calls.len(), 2);

    assert_eq!(tool_calls[0].index, 0);
    assert!(tool_calls[0].id.as_ref().unwrap().starts_with("call-"));
    assert_eq!(tool_calls[0].r#type, Some(ChatCompletionToolType::Function));
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("search")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"query":"rust"}"#)
    );

    assert_eq!(tool_calls[1].index, 1);
    assert!(tool_calls[1].id.as_ref().unwrap().starts_with("call-"));
    assert_eq!(tool_calls[1].r#type, Some(ChatCompletionToolType::Function));
    assert_eq!(
        tool_calls[1].function.as_ref().unwrap().name.as_deref(),
        Some("summarize")
    );
    assert_eq!(
        tool_calls[1]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"topic":"memory"}"#)
    );
}

#[tokio::test]
async fn test_named_tool_choice_parses_reasoning_content_json() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "get_weather".to_string(),
            },
        },
    ));
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-rc-1".to_string());
    let backend_output = build_backend_output(r#"{"location":"Paris"}"#);
    let mut raw_response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");
    move_content_to_reasoning(&mut raw_response);

    let response = apply_jail_transformation(raw_response, tool_choice).await;
    let choice = &response.choices[0];

    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );
    assert!(choice.delta.reasoning_content.is_none());

    let tool_calls = choice.delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"location":"Paris"}"#)
    );
}

#[tokio::test]
async fn test_required_tool_choice_parses_reasoning_content_stream() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Required);
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-rc-2".to_string());
    let mut raw_response_1 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            r#"[{"name":"search","parameters":{"query":"rust"}}"#,
            None,
        ))
        .expect("first choice generation");
    move_content_to_reasoning(&mut raw_response_1);

    let mut raw_response_2 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            "]",
            Some(common::FinishReason::Stop),
        ))
        .expect("second choice generation");
    move_content_to_reasoning(&mut raw_response_2);

    let responses =
        apply_jail_transformation_streaming(vec![raw_response_1, raw_response_2], tool_choice)
            .await;

    assert_eq!(responses.len(), 1);

    let choice = &responses[0].choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ToolCalls)
    );
    assert!(choice.delta.reasoning_content.is_none());

    let tool_calls = choice.delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("search")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"query":"rust"}"#)
    );
}

#[tokio::test]
async fn test_named_tool_choice_ignores_reasoning_before_json() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "calculate".to_string(),
            },
        },
    ));
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-rc-3".to_string());
    let mut raw_response_1 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            "I should compute this carefully first.",
            None,
        ))
        .expect("first choice generation");
    move_content_to_reasoning(&mut raw_response_1);

    let raw_response_2 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            r#"{"expression":"15 * 23"}"#,
            Some(common::FinishReason::Stop),
        ))
        .expect("second choice generation");

    let responses =
        apply_jail_transformation_streaming(vec![raw_response_1, raw_response_2], tool_choice)
            .await;

    assert_eq!(responses.len(), 1);

    let choice = &responses[0].choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );
    assert_eq!(
        choice.delta.reasoning_content.as_deref(),
        Some("I should compute this carefully first.")
    );

    let tool_calls = choice.delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("calculate")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"expression":"15 * 23"}"#)
    );
}

#[tokio::test]
async fn test_named_tool_choice_parses_json_after_reasoning_end_marker() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "calculate".to_string(),
            },
        },
    ));
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-rc-3b".to_string());
    let mut raw_response_1 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            "I should compute this carefully first.",
            None,
        ))
        .expect("first choice generation");
    move_content_to_reasoning(&mut raw_response_1);

    let mut raw_response_2 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            r#"</think>{"expression":"15 * 23"}"#,
            Some(common::FinishReason::Stop),
        ))
        .expect("second choice generation");
    move_content_to_reasoning(&mut raw_response_2);

    let responses =
        apply_jail_transformation_streaming(vec![raw_response_1, raw_response_2], tool_choice)
            .await;

    assert_eq!(responses.len(), 1);

    let choice = &responses[0].choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );
    assert_eq!(
        choice.delta.reasoning_content.as_deref(),
        Some("I should compute this carefully first.")
    );

    let tool_calls = choice.delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("calculate")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"expression":"15 * 23"}"#)
    );
}

#[tokio::test]
async fn test_named_tool_choice_parses_inline_reasoning_end_marker_in_content() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "calculate".to_string(),
            },
        },
    ));
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-rc-inline".to_string());
    let raw_response = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            "The user wants me to calculate 2+2.\n</think>{\"expression\":\"2+2\"}",
            Some(common::FinishReason::Stop),
        ))
        .expect("inline reasoning choice generation");

    let responses = apply_jail_transformation_streaming(vec![raw_response], tool_choice).await;

    assert_eq!(responses.len(), 1);

    let choice = &responses[0].choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );
    assert_eq!(
        choice.delta.reasoning_content.as_deref(),
        Some("The user wants me to calculate 2+2.\n")
    );

    let tool_calls = choice.delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("calculate")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"expression":"2+2"}"#)
    );
}

#[tokio::test]
async fn test_named_tool_choice_ignores_reasoning_only_content_before_json() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "calculate".to_string(),
            },
        },
    ));
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-rc-split".to_string());
    let raw_response_1 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            "The user is asking me to calculate 2+2.\n",
            None,
        ))
        .expect("content chunk generation");
    let raw_response_2 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            "</think>{\"expression\":\"2+2\"}",
            Some(common::FinishReason::Stop),
        ))
        .expect("json chunk generation");

    let responses =
        apply_jail_transformation_streaming(vec![raw_response_1, raw_response_2], tool_choice)
            .await;

    assert_eq!(
        responses.len(),
        1,
        "reasoning-only content should stay out of the JSON buffer"
    );

    let choice = &responses[0].choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );
    assert_eq!(
        choice.delta.reasoning_content.as_deref(),
        Some("The user is asking me to calculate 2+2.\n")
    );

    let tool_calls = choice.delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("calculate")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"expression":"2+2"}"#)
    );
}

#[tokio::test]
async fn test_named_tool_choice_recovers_json_suffix_after_schema_prefix() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "search".to_string(),
            },
        },
    ));
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-rc-schema".to_string());
    let raw_response = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            "{date_range?,metadata?,tags?}\n-options:{sort_by?,ascending?,limit?,offset?}\n{\"query\":\"pdf docx\"}",
            Some(common::FinishReason::Stop),
        ))
        .expect("schema-prefixed choice generation");

    let responses = apply_jail_transformation_streaming(vec![raw_response], tool_choice).await;

    assert_eq!(responses.len(), 1);

    let choice = &responses[0].choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );
    assert_eq!(
        choice.delta.reasoning_content.as_deref(),
        Some("{date_range?,metadata?,tags?}\n-options:{sort_by?,ascending?,limit?,offset?}\n")
    );

    let tool_calls = choice.delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("search")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"query":"pdf docx"}"#)
    );
}

#[tokio::test]
async fn test_named_tool_choice_recovers_live_schema_prose_suffix_with_reasoning() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "search_documents".to_string(),
            },
        },
    ));
    request.inner.tool_choice = tool_choice.clone();

    let live_content = "{ date_range?, metadata?, tags? }\n- options: { sort_by?, ascending?, limit?, offset? }\n\nThis doesn't seem to be designed for file path-based searching. The user might be expecting a file system search, but I don't have a tool for that.\n\nI should clarify with the user what they're looking for. The `search_documents` tool seems to be for searching document content/metadata, not for finding files by path patterns.\n\nLet me try using the search_documents function with a query that might match what they're looking for, or I should ask for clarification.\n\nActually, I think the best approach is to try the search_documents function with a reasonable query. The user might want to search for documents in general. Let me try with a query that might work, or I should explain the limitation.\n\nGiven the tools available, I should use search_documents. Let me try a query that might help find these documents. However, the paths provided are very specific file system paths, not really searchable content.\n\nI think I should try the search_documents function and see what results come back, or I should inform the user that I don't have a file system search tool available.\n\nLet me try the search_documents function with a query that might be relevant. I'll use \"pdf docx\" as a query to try to find documents of those types.\n{\n  \"query\": \"pdf docx\"\n}";
    let live_reasoning = "The user wants to search for documents with specific patterns in two different paths:\n1. /home/user/documents/**/*.pdf - This looks like a Unix/Linux path pattern\n2. C:\\Users\\Admin\\Downloads\\*.docx - This looks like a Windows path pattern\n\nHowever, looking at the available tools, I have a `search_documents` function that allows searching documents with filters, but it doesn't take file path patterns directly. It has a `query` parameter for text search and `filters` for metadata, date range, and tags.\n\nThe `search_documents` function signature is:\n- query: string\n- filters:";

    let mut generator = request.response_generator("req-rc-live-schema".to_string());
    let mut raw_response = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            live_content,
            Some(common::FinishReason::Stop),
        ))
        .expect("live schema prose choice generation");
    raw_response.choices[0].delta.reasoning_content = Some(live_reasoning.to_string());

    let responses = apply_jail_transformation_streaming(vec![raw_response], tool_choice).await;

    assert_eq!(responses.len(), 1);

    let choice = &responses[0].choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );

    let tool_calls = choice.delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("search_documents")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some("{\n  \"query\": \"pdf docx\"\n}")
    );

    let reasoning = choice.delta.reasoning_content.as_deref().unwrap();
    assert!(reasoning.contains("The user wants to search for documents with specific patterns"));
    assert!(reasoning.contains("This doesn't seem to be designed for file path-based searching."));
    assert!(reasoning.contains("I'll use \"pdf docx\" as a query"));
}

#[tokio::test]
async fn test_named_tool_choice_prefers_last_valid_json_suffix() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "calculate".to_string(),
            },
        },
    ));
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-rc-last-json".to_string());
    let raw_response = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            "{\"a\":1}{\"b\":2}{\"expression\":\"{\\\"a\\\":1}+{\\\"b\\\":2}\"}",
            Some(common::FinishReason::Stop),
        ))
        .expect("embedded-json choice generation");

    let responses = apply_jail_transformation_streaming(vec![raw_response], tool_choice).await;

    assert_eq!(responses.len(), 1);

    let choice = &responses[0].choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );

    let tool_calls = choice.delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("calculate")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"expression":"{\"a\":1}+{\"b\":2}"}"#)
    );
}

#[tokio::test]
async fn test_named_tool_choice_waits_for_complete_top_level_object() {
    use dynamo_llm::protocols::openai::chat_completions::jail::JailedStream;
    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::StreamExt;
    use futures::stream;

    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "search_documents".to_string(),
            },
        },
    ));

    let mut request = create_test_request();
    request.inner.tool_choice = tool_choice.clone();
    let mut generator = request.response_generator("req-rc-nested-object".to_string());
    let raw_response_1 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            "{\n  \"query\": \"machine learning\",\n  \"filters\": {\n    \"date_range\": {\n      \"from\": \"2023-01-01\",\n      \"to\": \"2023-12-31\"\n    }",
            None,
        ))
        .expect("nested-object first chunk generation");
    let raw_response_2 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            ",\n    \"tags\": [\"AI\", \"research\", \"papers\"]\n  },\n  \"options\": {\n    \"ascending\": false,\n    \"limit\": 10\n  }\n}",
            Some(common::FinishReason::Stop),
        ))
        .expect("nested-object second chunk generation");

    let input_stream = stream::iter(
        vec![raw_response_1, raw_response_2]
            .into_iter()
            .map(|response| Annotated {
                data: Some(response),
                id: None,
                event: None,
                comment: None,
                error: None,
            }),
    );
    let jail = JailedStream::builder()
        .tool_choice_named("search_documents".to_string())
        .tool_definitions(default_parser_tool_definitions())
        .build();
    let output_stream = jail.apply_with_finish_reason(input_stream);
    tokio::pin!(output_stream);
    let responses: Vec<_> = output_stream
        .filter_map(|annotated| async move { annotated.data })
        .collect()
        .await;

    assert_eq!(responses.len(), 1);

    let choice = &responses[0].choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );
    assert!(choice.delta.reasoning_content.is_none());

    let tool_calls = choice.delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("search_documents")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(
            "{\n  \"query\": \"machine learning\",\n  \"filters\": {\n    \"date_range\": {\n      \"from\": \"2023-01-01\",\n      \"to\": \"2023-12-31\"\n    },\n    \"tags\": [\"AI\", \"research\", \"papers\"]\n  },\n  \"options\": {\n    \"ascending\": false,\n    \"limit\": 10\n  }\n}"
        )
    );
}

#[tokio::test]
async fn test_named_tool_choice_ignores_json_literals_inside_reasoning() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "calculate".to_string(),
            },
        },
    ));
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-rc-reasoning-json".to_string());
    let mut raw_response_1 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            r#"The expression contains JSON literals like {"a":1} and {"b":2}."#,
            None,
        ))
        .expect("reasoning choice generation");
    move_content_to_reasoning(&mut raw_response_1);

    let mut raw_response_2 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            r#"</think>{"expression":"{\"a\":1}+{\"b\":2}"}"#,
            Some(common::FinishReason::Stop),
        ))
        .expect("final json choice generation");
    move_content_to_reasoning(&mut raw_response_2);

    let responses =
        apply_jail_transformation_streaming(vec![raw_response_1, raw_response_2], tool_choice)
            .await;

    assert_eq!(responses.len(), 1);

    let choice = &responses[0].choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );
    assert_eq!(
        choice.delta.reasoning_content.as_deref(),
        Some(r#"The expression contains JSON literals like {"a":1} and {"b":2}."#)
    );

    let tool_calls = choice.delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"expression":"{\"a\":1}+{\"b\":2}"}"#)
    );
}

#[tokio::test]
async fn test_required_tool_choice_ignores_reasoning_before_json() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Required);
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-rc-4".to_string());
    let mut raw_response_1 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            "I need to decide which tools to call.",
            None,
        ))
        .expect("first choice generation");
    move_content_to_reasoning(&mut raw_response_1);

    let raw_response_2 = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            r#"[{"name":"calculate","parameters":{"expression":"15 * 23"}}]"#,
            Some(common::FinishReason::Stop),
        ))
        .expect("second choice generation");

    let responses =
        apply_jail_transformation_streaming(vec![raw_response_1, raw_response_2], tool_choice)
            .await;

    assert_eq!(responses.len(), 1);

    let choice = &responses[0].choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ToolCalls)
    );
    assert_eq!(
        choice.delta.reasoning_content.as_deref(),
        Some("I need to decide which tools to call.")
    );

    let tool_calls = choice.delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("calculate")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"expression":"15 * 23"}"#)
    );
}

#[tokio::test]
async fn test_required_tool_choice_recovers_json_suffix_after_prose_prefix() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Required);
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-rc-required-suffix".to_string());
    let raw_response = generator
        .choice_from_postprocessor(build_backend_output_with_finish(
            "I should call the calculator now.\n[{\"name\":\"calculate\",\"parameters\":{\"expression\":\"15 * 23\"}}]",
            Some(common::FinishReason::Stop),
        ))
        .expect("required suffix choice generation");

    let responses = apply_jail_transformation_streaming(vec![raw_response], tool_choice).await;

    assert_eq!(responses.len(), 1);

    let choice = &responses[0].choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ToolCalls)
    );
    assert_eq!(
        choice.delta.reasoning_content.as_deref(),
        Some("I should call the calculator now.\n")
    );

    let tool_calls = choice.delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("calculate")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"expression":"15 * 23"}"#)
    );
}

#[tokio::test]
async fn test_tool_choice_parse_failure_returns_as_content() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Required);
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-3".to_string());
    let backend_output = build_backend_output("not-json");
    let raw_response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    let responses = apply_jail_transformation_streaming(vec![raw_response], tool_choice).await;
    assert_eq!(responses.len(), 1, "parse failure should still emit fallback content");
    let delta = &responses[0].choices[0].delta;

    // Jail stream behavior: if parsing fails, return accumulated content as-is
    // This matches marker-based FC behavior
    assert_eq!(delta.content.as_ref().map(get_text), Some("not-json"));
    assert!(delta.tool_calls.is_none());
}

#[tokio::test]
async fn test_streaming_named_tool_buffers_until_finish() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "get_weather".to_string(),
            },
        },
    ));
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-stream-1".to_string());

    let chunks = [r#"{"location":""#, r#"Paris","unit":""#, r#"celsius"}"#];

    let mut raw_responses = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let backend_output = BackendOutput {
            token_ids: vec![],
            tokens: vec![],
            text: Some(chunk.to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: if i == chunks.len() - 1 {
                Some(common::FinishReason::Stop)
            } else {
                None
            },
            stop_reason: None,
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
        };

        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("streaming chunk");
        raw_responses.push(response);
    }

    let all_responses = apply_jail_transformation_streaming(raw_responses, tool_choice).await;

    // Jail stream buffers content until valid JSON, then emits once
    assert_eq!(all_responses.len(), 1);

    let response = &all_responses[0];
    assert_eq!(
        response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );

    let tool_calls = response.choices[0].delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"location":"Paris","unit":"celsius"}"#)
    );
}

#[tokio::test]
async fn test_streaming_required_tool_parallel() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Required);
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-stream-2".to_string());

    let chunks = [
        r#"[{"name":"search","parameters":{"query":"rust"}},"#,
        r#"{"name":"summarize","parameters":{"topic":"memory"}}]"#,
    ];

    let mut raw_responses = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let backend_output = BackendOutput {
            token_ids: vec![],
            tokens: vec![],
            text: Some(chunk.to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: if i == chunks.len() - 1 {
                Some(common::FinishReason::Stop)
            } else {
                None
            },
            stop_reason: None,
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
        };

        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("streaming chunk");
        raw_responses.push(response);
    }

    let all_responses = apply_jail_transformation_streaming(raw_responses, tool_choice).await;

    // Jail stream buffers until complete JSON array
    assert_eq!(all_responses.len(), 1);

    let response = &all_responses[0];
    assert_eq!(
        response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ToolCalls)
    );

    let tool_calls = response.choices[0].delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 2);

    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("search")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"query":"rust"}"#)
    );

    assert_eq!(
        tool_calls[1].function.as_ref().unwrap().name.as_deref(),
        Some("summarize")
    );
    assert_eq!(
        tool_calls[1]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"topic":"memory"}"#)
    );
}

#[test]
fn test_no_tool_choice_outputs_normal_text() {
    let request = create_test_request();

    let mut generator = request.response_generator("req-stream-4".to_string());

    let backend_output = BackendOutput {
        token_ids: vec![],
        tokens: vec![],
        text: Some("Hello world".to_string()),
        cum_log_probs: None,
        log_probs: None,
        top_logprobs: None,
        finish_reason: None,
        stop_reason: None,
        index: Some(0),
        completion_usage: None,
        disaggregated_params: None,
    };

    let response = generator
        .choice_from_postprocessor(backend_output)
        .expect("normal text");

    assert_eq!(
        response.choices[0].delta.content.as_ref().map(get_text),
        Some("Hello world")
    );
    assert!(response.choices[0].delta.tool_calls.is_none());
}
