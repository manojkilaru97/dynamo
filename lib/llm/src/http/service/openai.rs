// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::Request,
    http::{HeaderMap, HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use base64::Engine as _;
use bytes::Bytes;
use dynamo_async_openai::types::ChatCompletionToolChoiceOption;
use dynamo_runtime::config::env_is_truthy;
use dynamo_runtime::config::environment_names::llm as env_llm;
use dynamo_runtime::config::environment_names::logging as env_logging;
use dynamo_runtime::{
    pipeline::{AsyncEngineContextProvider, Context},
    protocols::annotated::AnnotationsProvider,
};
use futures::{StreamExt, stream};
use serde::{Deserialize, Serialize, Serializer};

use super::{
    RouteDoc,
    disconnect::{ConnectionHandle, create_connection_monitor, monitor_for_disconnects},
    error::HttpError,
    metrics::{
        CancellationLabels, Endpoint, ErrorType, EventConverter,
        process_response_and_observe_metrics,
        process_response_using_event_converter_and_observe_metrics,
    },
    service_v2,
};
use crate::engines::ValidateRequest;
use crate::protocols::openai::chat_completions::aggregator::ChatCompletionAggregator;
use crate::protocols::openai::nvext::apply_header_routing_overrides;
use crate::protocols::openai::{
    chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
        NvCreateChatCompletionStreamResponse,
    },
    completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
    images::{NvCreateImageRequest, NvImagesResponse},
    responses::{NvCreateResponse, NvResponse, ResponseParams, chat_completion_to_response},
    tools,
    videos::{NvCreateVideoRequest, NvVideosResponse},
};
use crate::request_template::RequestTemplate;
use crate::types::Annotated;
use dynamo_runtime::logging::{emit_payload_log, get_distributed_tracing_context};
use tracing::Instrument;

pub const DYNAMO_REQUEST_ID_HEADER: &str = "x-dynamo-request-id";
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Dynamo Annotation for the request ID
pub const ANNOTATION_REQUEST_ID: &str = "request_id";

const VALIDATION_PREFIX: &str = "Validation: ";

// Default axum max body limit without configuring is 2MB: https://docs.rs/axum/latest/axum/extract/struct.DefaultBodyLimit.html
/// Default body limit in bytes (45MB) to support 500k+ token payloads.
/// Can be configured at runtime using the DYN_HTTP_BODY_LIMIT_MB environment variable.
pub(super) fn get_body_limit() -> usize {
    std::env::var(env_llm::DYN_HTTP_BODY_LIMIT_MB)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(45 * 1024 * 1024)
}

/// Tracing target used for payload log records exported to OTEL.
/// Suppressed from console output; visible only in the OTEL log pipeline.
const PAYLOAD_LOG_TARGET: &str = "dynamo_payload";
const PAYLOAD_LOG_FALLBACK_TARGET: &str = "dynamo_llm::http::service::service_v2";
const MAX_PAYLOAD_ACCUMULATE_BYTES: usize = 256 * 1024;
const REDACTED_MM_INPUT: &str = "[redacted-mm-input]";

fn configured_max_output_len() -> Option<u32> {
    std::env::var(env_llm::DYN_MAX_OUTPUT_LEN)
        .ok()
        .and_then(|raw| raw.parse::<u32>().ok())
        .filter(|value| *value > 0)
}

fn cap_output_len_field(field_name: &str, value: &mut Option<u32>, cap: u32) {
    if let Some(current) = *value {
        if current > cap {
            tracing::info!(
                field_name,
                requested = current,
                capped = cap,
                "capped request output length to server policy",
            );
            *value = Some(cap);
        }
    }
}

fn apply_chat_max_output_len_cap(request: &mut NvCreateChatCompletionRequest) {
    apply_chat_max_output_len_cap_with(request, configured_max_output_len());
}

fn apply_chat_max_output_len_cap_with(
    request: &mut NvCreateChatCompletionRequest,
    cap: Option<u32>,
) {
    let Some(cap) = cap else {
        return;
    };

    if request.inner.max_completion_tokens.is_some() {
        cap_output_len_field(
            "max_completion_tokens",
            &mut request.inner.max_completion_tokens,
            cap,
        );
    } else {
        cap_output_len_field("max_tokens", &mut request.inner.max_tokens, cap);
    }
}

fn apply_completion_max_output_len_cap(request: &mut NvCreateCompletionRequest) {
    apply_completion_max_output_len_cap_with(request, configured_max_output_len());
}

fn apply_completion_max_output_len_cap_with(
    request: &mut NvCreateCompletionRequest,
    cap: Option<u32>,
) {
    let Some(cap) = cap else {
        return;
    };
    cap_output_len_field("max_tokens", &mut request.inner.max_tokens, cap);
}

fn apply_responses_max_output_len_cap(request: &mut NvCreateResponse) {
    apply_responses_max_output_len_cap_with(request, configured_max_output_len());
}

fn apply_responses_max_output_len_cap_with(request: &mut NvCreateResponse, cap: Option<u32>) {
    let Some(cap) = cap else {
        return;
    };
    cap_output_len_field(
        "max_output_tokens",
        &mut request.inner.max_output_tokens,
        cap,
    );
}

fn ensure_chat_template_thinking_disabled(obj: &mut serde_json::Map<String, serde_json::Value>) {
    let target_key = if obj.contains_key("chat_template_kwargs") {
        "chat_template_kwargs"
    } else if obj.contains_key("chat_template_args") {
        "chat_template_args"
    } else {
        "chat_template_kwargs"
    };

    if !obj
        .get(target_key)
        .is_some_and(serde_json::Value::is_object)
    {
        obj.insert(
            target_key.to_string(),
            serde_json::Value::Object(serde_json::Map::new()),
        );
    }

    if let Some(args) = obj
        .get_mut(target_key)
        .and_then(serde_json::Value::as_object_mut)
    {
        args.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
        args.insert("thinking".to_string(), serde_json::Value::Bool(false));
    }
}

fn has_explicit_chat_template_thinking_control(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    for key in ["chat_template_kwargs", "chat_template_args"] {
        if let Some(args) = obj.get(key).and_then(serde_json::Value::as_object)
            && (args.contains_key("enable_thinking") || args.contains_key("thinking"))
        {
            return true;
        }
    }
    false
}

fn normalize_chat_compat_payload(payload: &mut serde_json::Value) {
    let has_structured_output = detect_payload_structured_output_kind(payload).is_some();
    let Some(obj) = payload.as_object_mut() else {
        return;
    };

    let reasoning_effort = obj
        .get("reasoning_effort")
        .and_then(serde_json::Value::as_str)
        .map(|value| value.to_ascii_lowercase());

    match reasoning_effort.as_deref() {
        // The public UI sends `none` for "no reasoning". async-openai does not model that
        // enum variant, so translate it into the template control Dynamo already supports.
        Some("none") => {
            obj.remove("reasoning_effort");
            ensure_chat_template_thinking_disabled(obj);
        }
        // Some clients expose a five-level UI. Dynamo/backends only understand OpenAI's
        // low/medium/high values, so `max` should behave as the strongest supported level.
        Some("max") => {
            obj.insert(
                "reasoning_effort".to_string(),
                serde_json::Value::String("high".to_string()),
            );
        }
        _ => {}
    }

    // OpenAI structured output schemas constrain assistant content, not hidden
    // reasoning. For Qwen-style default-thinking models, leaving thinking
    // implicit can put valid JSON into `reasoning_content`, which breaks
    // OpenAI/LangChain structured-output clients. Keep explicit caller intent,
    // but default structured output to non-thinking content mode.
    if has_structured_output && !has_explicit_chat_template_thinking_control(obj) {
        ensure_chat_template_thinking_disabled(obj);
    }
}

fn ensure_chat_response_logprobs_field(response: &mut serde_json::Value) {
    let Some(choices) = response
        .get_mut("choices")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };

    for choice in choices {
        if let Some(choice_obj) = choice.as_object_mut() {
            choice_obj
                .entry("logprobs".to_string())
                .or_insert(serde_json::Value::Null);
        }
    }
}

fn header_map_to_json(headers: &HeaderMap) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for (name, value) in headers {
        let key = name.as_str().to_string();
        let value = value.to_str().unwrap_or_default().to_string();
        if let Some(existing) = out.remove(&key) {
            let updated = match existing {
                serde_json::Value::String(first) => serde_json::Value::Array(vec![
                    serde_json::Value::String(first),
                    serde_json::Value::String(value),
                ]),
                serde_json::Value::Array(mut items) => {
                    items.push(serde_json::Value::String(value));
                    serde_json::Value::Array(items)
                }
                other => other,
            };
            out.insert(key, updated);
        } else {
            out.insert(key, serde_json::Value::String(value));
        }
    }
    serde_json::Value::Object(out)
}

fn error_response_payload(response: &ErrorResponse) -> serde_json::Value {
    serde_json::to_value(&*response.1).unwrap_or_else(|_| {
        serde_json::json!({
            "error": {
                "message": response.1.message.clone(),
                "type": response.1.error_type.clone(),
                "code": response.1.code,
            }
        })
    })
}

fn count_payload_modalities(payload: &serde_json::Value) -> (usize, usize, usize) {
    fn count_content_parts(content: &serde_json::Value, counts: &mut (usize, usize, usize)) {
        let Some(parts) = content.as_array() else {
            return;
        };
        for part in parts {
            let Some(part_type) = part.get("type").and_then(serde_json::Value::as_str) else {
                continue;
            };
            match part_type {
                "image_url" | "input_image" => counts.0 += 1,
                "video_url" => counts.1 += 1,
                "audio_url" | "input_audio" => counts.2 += 1,
                _ => {}
            }
        }
    }

    let mut counts = (0, 0, 0);
    if let Some(messages) = payload
        .get("messages")
        .and_then(serde_json::Value::as_array)
    {
        for message in messages {
            if let Some(content) = message.get("content") {
                count_content_parts(content, &mut counts);
            }
        }
    }
    if let Some(input_items) = payload.get("input").and_then(serde_json::Value::as_array) {
        for item in input_items {
            let item_type = item
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            match item_type {
                "image_url" | "input_image" => {
                    counts.0 += 1;
                    continue;
                }
                "audio_url" | "input_audio" => {
                    counts.2 += 1;
                    continue;
                }
                "video_url" => {
                    counts.1 += 1;
                    continue;
                }
                _ => {}
            }
            if let Some(content) = item.get("content") {
                count_content_parts(content, &mut counts);
            }
        }
    }
    counts
}

fn normalize_payload_tool_choice(payload: &serde_json::Value) -> Option<&'static str> {
    match payload.get("tool_choice") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(choice)) => match choice.as_str() {
            "auto" => Some("auto"),
            "none" => Some("none"),
            "required" => Some("required"),
            _ => Some("other"),
        },
        Some(serde_json::Value::Object(choice)) => {
            if choice
                .get("function")
                .and_then(|function| function.get("name"))
                .is_some()
            {
                Some("named")
            } else {
                choice
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .map(|choice_type| match choice_type {
                        "auto" => "auto",
                        "none" => "none",
                        "required" => "required",
                        "function" => "function",
                        _ => "other",
                    })
                    .or(Some("named"))
            }
        }
        Some(_) => Some("other"),
    }
}

fn detect_payload_structured_output_kind(payload: &serde_json::Value) -> Option<&'static str> {
    if let Some(response_format) = payload
        .get("response_format")
        .and_then(serde_json::Value::as_object)
        && let Some(format_type) = response_format
            .get("type")
            .and_then(serde_json::Value::as_str)
    {
        match format_type {
            "json_schema" => return Some("json_schema"),
            "json_object" => return Some("json_object"),
            "structural_tag" => return Some("structural_tag"),
            _ => {}
        }
    }

    if let Some(structured_outputs) = payload.get("structured_outputs") {
        if let Some(structured_outputs) = structured_outputs.as_object() {
            if structured_outputs.is_empty() {
                return None;
            }
            for key in [
                "json",
                "json_object",
                "json_schema",
                "structural_tag",
                "regex",
                "choice",
                "grammar",
            ] {
                if structured_outputs
                    .get(key)
                    .is_some_and(|value| !value.is_null())
                {
                    return Some(if key == "json" { "json_schema" } else { key });
                }
            }
        }
        return Some("structured_outputs");
    }

    if let Some(text) = payload.get("text").and_then(serde_json::Value::as_object)
        && let Some(format) = text.get("format").and_then(serde_json::Value::as_object)
        && let Some(format_type) = format.get("type").and_then(serde_json::Value::as_str)
    {
        match format_type {
            "json_schema" => return Some("json_schema"),
            "json_object" => return Some("json_object"),
            _ => {}
        }
    }

    None
}

fn add_request_shape_log_attrs(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    payload: &serde_json::Value,
) {
    let (image_count, video_count, audio_count) = count_payload_modalities(payload);
    let tool_count = payload
        .get("tools")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    let tool_choice = normalize_payload_tool_choice(payload);
    let structured_output_kind = detect_payload_structured_output_kind(payload);

    fields.insert("input_image_count".to_string(), image_count.into());
    fields.insert("input_video_count".to_string(), video_count.into());
    fields.insert("input_audio_count".to_string(), audio_count.into());
    fields.insert("input_tool_count".to_string(), tool_count.into());
    fields.insert("has_images".to_string(), (image_count > 0).into());
    fields.insert("has_videos".to_string(), (video_count > 0).into());
    fields.insert("has_audios".to_string(), (audio_count > 0).into());
    fields.insert("has_tools".to_string(), (tool_count > 0).into());
    fields.insert(
        "has_tool_calls_enabled".to_string(),
        (tool_count > 0 && tool_choice != Some("none")).into(),
    );
    fields.insert(
        "has_structured_output".to_string(),
        structured_output_kind.is_some().into(),
    );
    if let Some(tool_choice) = tool_choice {
        fields.insert("tool_choice".to_string(), tool_choice.into());
    }
    if let Some(structured_output_kind) = structured_output_kind {
        fields.insert(
            "structured_output_kind".to_string(),
            structured_output_kind.into(),
        );
    }
}

fn is_multimodal_reference(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.starts_with("file://")
        || lower.contains("data:image/")
        || lower.contains("data:video/")
        || lower.contains("data:audio/")
        || lower.contains(";asset_id,")
}

fn redact_html_src_attrs(input: &str) -> Option<String> {
    let lower = input.to_ascii_lowercase();
    if !lower.contains("src=") || !is_multimodal_reference(input) {
        return None;
    }

    let bytes = input.as_bytes();
    let lower_bytes = lower.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut cursor = 0usize;

    while let Some(rel) = lower_bytes[cursor..]
        .windows(4)
        .position(|window| window == b"src=")
    {
        let src_pos = cursor + rel;
        let quote_pos = src_pos + 4;
        if quote_pos >= bytes.len() || (bytes[quote_pos] != b'\'' && bytes[quote_pos] != b'"') {
            cursor = quote_pos.min(bytes.len());
            continue;
        }
        let quote = bytes[quote_pos];
        let value_start = quote_pos + 1;
        let Some(value_rel_end) = bytes[value_start..].iter().position(|b| *b == quote) else {
            break;
        };
        let value_end = value_start + value_rel_end;
        let value = &input[value_start..value_end];

        out.push_str(&input[cursor..value_start]);
        if is_multimodal_reference(value) {
            out.push_str(REDACTED_MM_INPUT);
        } else {
            out.push_str(value);
        }
        cursor = value_end;
    }

    if cursor == 0 {
        return None;
    }
    out.push_str(&input[cursor..]);
    Some(out)
}

fn redact_string_for_payload_log(value: &str) -> serde_json::Value {
    if let Some(redacted) = redact_html_src_attrs(value) {
        return redacted.into();
    }
    if is_multimodal_reference(value) {
        return REDACTED_MM_INPUT.into();
    }
    value.into()
}

fn redact_media_container_for_payload_log(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::String(_) => REDACTED_MM_INPUT.into(),
        serde_json::Value::Object(map) => {
            let mut redacted = serde_json::Map::with_capacity(map.len());
            for (key, child) in map {
                if key == "url" {
                    redacted.insert(key.clone(), REDACTED_MM_INPUT.into());
                } else {
                    redacted.insert(key.clone(), redact_multimodal_payload_for_logging(child));
                }
            }
            serde_json::Value::Object(redacted)
        }
        _ => redact_multimodal_payload_for_logging(value),
    }
}

fn redact_multimodal_payload_for_logging(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(items) => serde_json::Value::Array(
            items
                .iter()
                .map(redact_multimodal_payload_for_logging)
                .collect(),
        ),
        serde_json::Value::Object(map) => {
            let mut redacted = serde_json::Map::with_capacity(map.len());
            for (key, child) in map {
                let child = match key.as_str() {
                    "image_url" | "video_url" | "audio_url" => {
                        redact_media_container_for_payload_log(child)
                    }
                    "input_audio" => match child {
                        serde_json::Value::Object(audio) => {
                            let mut audio_map = serde_json::Map::with_capacity(audio.len());
                            for (audio_key, audio_value) in audio {
                                if audio_key == "data" {
                                    audio_map.insert(audio_key.clone(), REDACTED_MM_INPUT.into());
                                } else {
                                    audio_map.insert(
                                        audio_key.clone(),
                                        redact_multimodal_payload_for_logging(audio_value),
                                    );
                                }
                            }
                            serde_json::Value::Object(audio_map)
                        }
                        _ => redact_multimodal_payload_for_logging(child),
                    },
                    _ => redact_multimodal_payload_for_logging(child),
                };
                redacted.insert(key.clone(), child);
            }
            serde_json::Value::Object(redacted)
        }
        serde_json::Value::String(s) => redact_string_for_payload_log(s),
        _ => value.clone(),
    }
}

fn normalize_chat_response_payload_for_logging(
    mut payload: serde_json::Value,
    include_empty_tool_calls: bool,
    strip_null_reasoning_content: bool,
) -> serde_json::Value {
    let Some(choices) = payload
        .get_mut("choices")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return payload;
    };

    for choice in choices {
        let Some(message) = choice
            .get_mut("message")
            .and_then(serde_json::Value::as_object_mut)
        else {
            continue;
        };
        if !message.contains_key("content") {
            message.insert("content".to_string(), serde_json::Value::Null);
        }
        if strip_null_reasoning_content
            && message
                .get("reasoning_content")
                .is_some_and(serde_json::Value::is_null)
        {
            message.remove("reasoning_content");
        }
        if include_empty_tool_calls && !message.contains_key("tool_calls") {
            message.insert(
                "tool_calls".to_string(),
                serde_json::Value::Array(Vec::new()),
            );
        }
    }

    payload
}

fn emit_openai_request_log(
    request_id: &str,
    model: &str,
    endpoint: &'static str,
    streaming: bool,
    headers: serde_json::Value,
    payload: serde_json::Value,
) {
    if !log_payloads_enabled() {
        return;
    }

    let mut fields = serde_json::Map::new();
    fields.insert("rid".to_string(), request_id.into());
    fields.insert("request_id".to_string(), request_id.into());
    fields.insert("model".to_string(), model.into());
    fields.insert("endpoint".to_string(), endpoint.into());
    fields.insert("streaming".to_string(), streaming.into());
    fields.insert("headers".to_string(), headers.clone());
    add_request_shape_log_attrs(&mut fields, &payload);
    fields.insert(
        "payload".to_string(),
        redact_multimodal_payload_for_logging(&payload),
    );

    emit_payload_log(
        "openai.request",
        PAYLOAD_LOG_TARGET,
        serde_json::Value::Object(fields),
    );

    if endpoint == "responses" {
        tracing::warn!(
            request_id = request_id,
            "responses request payload logger invoked"
        );
        tracing::info!(
            target: PAYLOAD_LOG_FALLBACK_TARGET,
            rid = request_id,
            request_id = request_id,
            model = model,
            endpoint = endpoint,
            streaming = streaming,
            headers_json = %headers,
            payload_json = %payload,
            "openai.request"
        );
    }
}

fn emit_openai_response_log(
    request_id: &str,
    model: &str,
    endpoint: &'static str,
    streaming: bool,
    status_code: u16,
    payload: serde_json::Value,
) {
    emit_openai_response_log_with_options(
        request_id,
        model,
        endpoint,
        streaming,
        status_code,
        payload,
        false,
        false,
    )
}

fn emit_openai_response_log_with_options(
    request_id: &str,
    model: &str,
    endpoint: &'static str,
    streaming: bool,
    status_code: u16,
    payload: serde_json::Value,
    include_empty_tool_calls: bool,
    strip_null_reasoning_content: bool,
) {
    if !log_payloads_enabled() {
        return;
    }

    let payload = if endpoint == "chat_completions" {
        normalize_chat_response_payload_for_logging(
            payload,
            include_empty_tool_calls,
            strip_null_reasoning_content,
        )
    } else {
        payload
    };

    emit_payload_log(
        "openai.response",
        PAYLOAD_LOG_TARGET,
        serde_json::json!({
            "rid": request_id,
            "request_id": request_id,
            "model": model,
            "endpoint": endpoint,
            "streaming": streaming,
            "status_code": status_code,
            "payload": payload,
        }),
    );

    if endpoint == "responses" {
        tracing::warn!(
            request_id = request_id,
            status_code = status_code,
            "responses response payload logger invoked"
        );
        tracing::info!(
            target: PAYLOAD_LOG_FALLBACK_TARGET,
            rid = request_id,
            request_id = request_id,
            model = model,
            endpoint = endpoint,
            streaming = streaming,
            status_code = status_code,
            payload_json = %payload,
            "openai.response"
        );
    }
}

/// Returns true if OTEL payload logging is enabled via `DYNAMO_LOG_PAYLOADS`.
fn log_payloads_enabled() -> bool {
    env_is_truthy(env_logging::DYNAMO_LOG_PAYLOADS)
}

fn normalize_responses_reasoning(value: &mut serde_json::Value) {
    let Some(root) = value.as_object_mut() else {
        return;
    };
    let Some(reasoning) = root.get_mut("reasoning") else {
        return;
    };
    let Some(reasoning_obj) = reasoning.as_object_mut() else {
        return;
    };

    let effort_is_none = reasoning_obj
        .get("effort")
        .and_then(|value| value.as_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("none"));

    if !effort_is_none {
        return;
    }

    // The Responses API tests use effort=none to mean "disable reasoning".
    // The upstream typed model does not accept "none", so normalize it to
    // omission and carry an internal marker for the chat-completions bridge.
    reasoning_obj.remove("effort");
    if reasoning_obj.is_empty() {
        root.remove("reasoning");
    }
    root.insert(
        "dynamo_disable_reasoning".to_string(),
        serde_json::Value::Bool(true),
    );
}

fn normalize_responses_tools(value: &mut serde_json::Value) {
    let Some(root) = value.as_object_mut() else {
        return;
    };
    let Some(tools) = root.get_mut("tools").and_then(|value| value.as_array_mut()) else {
        return;
    };

    for tool in tools {
        let Some(tool_obj) = tool.as_object_mut() else {
            continue;
        };
        let is_function = tool_obj
            .get("type")
            .and_then(|value| value.as_str())
            .is_some_and(|value| value == "function");
        if !is_function {
            continue;
        }
        let Some(function_value) = tool_obj.remove("function") else {
            continue;
        };
        let Some(function_obj) = function_value.as_object() else {
            tool_obj.insert("function".to_string(), function_value);
            continue;
        };

        for key in ["name", "description", "parameters", "strict"] {
            if let Some(existing) = tool_obj.get(key) {
                if !existing.is_null() {
                    continue;
                }
            }
            if let Some(nested) = function_obj.get(key) {
                tool_obj.insert(key.to_string(), nested.clone());
            }
        }
    }
}

fn normalize_responses_request_json(mut value: serde_json::Value) -> serde_json::Value {
    normalize_responses_reasoning(&mut value);
    normalize_responses_tools(&mut value);
    value
}

pub type ErrorResponse = (StatusCode, Json<ErrorMessage>);

#[derive(Deserialize, Debug)]
pub(crate) struct ErrorMessage {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: u16,
}

impl Serialize for ErrorMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serde_json::json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
            }
        })
        .serialize(serializer)
    }
}

fn map_error_code_to_error_type(code: StatusCode) -> String {
    match code.canonical_reason() {
        Some(reason) => reason.to_string(),
        None => "UnknownError".to_string(),
    }
}

/// Classify error for metrics based on status code and message
fn classify_error_for_metrics(code: StatusCode, message: &str) -> ErrorType {
    match code {
        StatusCode::BAD_REQUEST => {
            // 400
            if message.starts_with("Validation:") {
                ErrorType::Validation
            } else {
                ErrorType::Internal
            }
        }
        StatusCode::NOT_FOUND => ErrorType::NotFound, // 404
        StatusCode::NOT_IMPLEMENTED => ErrorType::NotImplemented, // 501
        StatusCode::TOO_MANY_REQUESTS => ErrorType::Overload, // 429
        StatusCode::SERVICE_UNAVAILABLE => ErrorType::Overload, // 503
        StatusCode::INTERNAL_SERVER_ERROR => ErrorType::Internal, // 500
        _ if code.is_client_error() => ErrorType::Validation, // other 4xx
        _ => ErrorType::Internal,                     // everything else
    }
}

/// Extract ErrorType from ErrorResponse for metrics
fn extract_error_type_from_response(response: &ErrorResponse) -> ErrorType {
    classify_error_for_metrics(response.0, &response.1.message)
}

impl ErrorMessage {
    fn bad_request_from_message<T: Into<String>>(message: T) -> ErrorResponse {
        let code = StatusCode::BAD_REQUEST;
        (
            code,
            Json(ErrorMessage {
                message: message.into(),
                error_type: map_error_code_to_error_type(code),
                code: code.as_u16(),
            }),
        )
    }

    /// Not Found Error
    pub fn model_not_found() -> ErrorResponse {
        let code = StatusCode::NOT_FOUND;
        let error_type = map_error_code_to_error_type(code);
        (
            code,
            Json(ErrorMessage {
                message: "Model not found".to_string(),
                error_type,
                code: code.as_u16(),
            }),
        )
    }

    /// Service Unavailable
    /// This is returned when the service is live, but not ready.
    pub fn _service_unavailable() -> ErrorResponse {
        let code = StatusCode::SERVICE_UNAVAILABLE;
        let error_type = map_error_code_to_error_type(code);
        (
            code,
            Json(ErrorMessage {
                message: "Service is not ready".to_string(),
                error_type,
                code: code.as_u16(),
            }),
        )
    }

    /// Internal Service Error
    /// Return this error when the service encounters an internal error.
    /// We should return a generic message to the client instead of the real error.
    /// Internal Services errors are the result of misconfiguration or bugs in the service.
    pub fn internal_server_error(msg: &str) -> ErrorResponse {
        tracing::error!("Internal server error: {msg}");
        let code = StatusCode::INTERNAL_SERVER_ERROR;
        let error_type = map_error_code_to_error_type(code);
        (
            code,
            Json(ErrorMessage {
                message: msg.to_string(),
                error_type,
                code: code.as_u16(),
            }),
        )
    }

    /// Not Implemented Error
    /// Return this error when the client requests a feature that is not yet implemented.
    /// This should be used for features that are planned but not available.
    pub fn not_implemented_error<T: Display>(msg: T) -> ErrorResponse {
        tracing::error!("Not Implemented error: {msg}");
        let code = StatusCode::NOT_IMPLEMENTED;
        let error_type = map_error_code_to_error_type(code);
        (
            code,
            Json(ErrorMessage {
                message: msg.to_string(),
                error_type,
                code: code.as_u16(),
            }),
        )
    }

    /// The OAI endpoints call an [`dynamo.runtime::engine::AsyncEngine`] which are specialized to return
    /// an [`anyhow::Error`]. This method will convert the [`anyhow::Error`] into an [`HttpError`].
    /// If successful, it will return the [`HttpError`] as an [`ErrorMessage::internal_server_error`]
    /// with the details of the error.
    pub fn from_anyhow(err: anyhow::Error, alt_msg: &str) -> ErrorResponse {
        let err_string = format!("{err:#}");
        if let Some(response) = map_backend_validation_error(&err_string) {
            return response;
        }

        // First check for transient capacity/routing PipelineErrors.
        if let Some(pipeline_err) =
            err.downcast_ref::<dynamo_runtime::pipeline::error::PipelineError>()
            && matches!(
                pipeline_err,
                dynamo_runtime::pipeline::error::PipelineError::ServiceOverloaded(_)
                    | dynamo_runtime::pipeline::error::PipelineError::InstanceUnavailable(_)
            )
        {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorMessage {
                    message: pipeline_err.to_string(),
                    error_type: map_error_code_to_error_type(StatusCode::SERVICE_UNAVAILABLE),
                    code: StatusCode::SERVICE_UNAVAILABLE.as_u16(),
                }),
            );
        }

        // Check for DynamoError with InvalidArgument → HTTP 400
        if let Some(dynamo_err) = err.downcast_ref::<dynamo_runtime::error::DynamoError>()
            && dynamo_err.error_type() == dynamo_runtime::error::ErrorType::InvalidArgument
        {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorMessage {
                    message: dynamo_err.message().to_string(),
                    error_type: map_error_code_to_error_type(StatusCode::BAD_REQUEST),
                    code: StatusCode::BAD_REQUEST.as_u16(),
                }),
            );
        }

        // Then check for HttpError
        match err.downcast::<HttpError>() {
            Ok(http_error) => ErrorMessage::from_http_error(http_error),
            Err(err) => ErrorMessage::internal_server_error(&format!("{alt_msg}: {err:#}")),
        }
    }

    /// Implementers should only be able to throw 400-499 errors.
    pub fn from_http_error(err: HttpError) -> ErrorResponse {
        if err.code < 400 || err.code >= 500 {
            return ErrorMessage::internal_server_error(&err.message);
        }
        match StatusCode::from_u16(err.code) {
            Ok(code) => (
                code,
                Json(ErrorMessage {
                    message: err.message,
                    error_type: map_error_code_to_error_type(code),
                    code: code.as_u16(),
                }),
            ),
            Err(_) => ErrorMessage::internal_server_error(&err.message),
        }
    }
}

impl From<HttpError> for ErrorMessage {
    fn from(err: HttpError) -> Self {
        ErrorMessage {
            message: err.message,
            error_type: map_error_code_to_error_type(
                StatusCode::from_u16(err.code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            ),
            code: err.code,
        }
    }
}

fn map_backend_validation_error(error: &str) -> Option<ErrorResponse> {
    let trimmed = error.trim();

    if trimmed.contains("VLLMValidationError:")
        || trimmed.contains("Failed to convert the grammar from GBNF to Lark:")
        || (trimmed.contains("Failed to apply prompt template:")
            && (trimmed.contains("Unexpected item type")
                || trimmed.contains("invalid operation: Unexpected")
                || trimmed.contains("unsupported")))
    {
        let detail = trimmed
            .split_once(": ")
            .map(|(_, rhs)| rhs.to_string())
            .unwrap_or_else(|| trimmed.to_string());
        return Some(ErrorMessage::bad_request_from_message(detail));
    }

    None
}

// Problem: Currently we are using JSON from axum as the request validator. Whenever there is an invalid JSON, it will return a 422.
// But all the downstream apps that relies on openai based APIs, expects to get 400 for all these cases otherwise they fail badly
// Solution: Intercept the response from handlers and convert ANY 422 status codes to 400 with the actual error message.
pub async fn smart_json_error_middleware(request: Request<Body>, next: Next) -> Response {
    let response = next.run(request).await;

    if response.status() == StatusCode::UNPROCESSABLE_ENTITY {
        let (_parts, body) = response.into_parts();
        let body_bytes = axum::body::to_bytes(body, get_body_limit())
            .await
            .unwrap_or_default();
        let error_message = String::from_utf8_lossy(&body_bytes).to_string();
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorMessage {
                message: error_message,
                error_type: map_error_code_to_error_type(StatusCode::BAD_REQUEST),
                code: StatusCode::BAD_REQUEST.as_u16(),
            }),
        )
            .into_response()
    } else {
        // Pass through if it is not a 422
        response
    }
}

/// Get the request ID from a primary source, or next from the headers, or lastly create a new one if not present
// TODO: Similar function exists in lib/llm/src/grpc/service/openai.rs but with different signature and simpler logic
pub(super) fn get_or_create_request_id(primary: Option<&str>, headers: &HeaderMap) -> String {
    // Try to get request id from trace context
    if let Some(trace_context) = get_distributed_tracing_context()
        && let Some(x_dynamo_request_id) = trace_context.x_dynamo_request_id
    {
        return x_dynamo_request_id;
    }

    // Try to get the request ID from the primary source
    if let Some(primary) = primary
        && !primary.trim().is_empty()
    {
        return primary.to_string();
    }

    // Try to get the request ID header as a string slice
    let request_id_opt = headers
        .get(DYNAMO_REQUEST_ID_HEADER)
        .or_else(|| headers.get(REQUEST_ID_HEADER))
        .and_then(|h| h.to_str().ok());

    if let Some(request_id) = request_id_opt
        && !request_id.trim().is_empty()
    {
        return request_id.to_string();
    }

    uuid::Uuid::new_v4().to_string()
}

/// OpenAI Completions Request Handler
///
/// This method will handle the incoming request for the `/v1/completions endpoint`. The endpoint is a "source"
/// for an [`super::OpenAICompletionsStreamingEngine`] and will return a stream of
/// responses which will be forward to the client.
///
/// Note: For all requests, streaming or non-streaming, we always call the engine with streaming enabled. For
/// non-streaming requests, we will fold the stream into a single response as part of this handler.
async fn handler_completions(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(mut request): Json<NvCreateCompletionRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    let request_id = get_or_create_request_id(request.inner.user.as_deref(), &headers);
    let streaming = request.inner.stream.unwrap_or(false);
    let raw_model = request.inner.model.clone();

    emit_openai_request_log(
        &request_id,
        &raw_model,
        "completions",
        streaming,
        header_map_to_json(&headers),
        serde_json::to_value(&request).unwrap_or(serde_json::Value::Null),
    );

    request.nvext = apply_header_routing_overrides(request.nvext.take(), &headers);

    // create the context for the request
    let cancellation_labels = CancellationLabels {
        model: request.inner.model.clone(),
        endpoint: Endpoint::Completions.to_string(),
        request_type: if streaming { "stream" } else { "unary" }.to_string(),
    };
    let request = Context::with_id(request, request_id.clone());
    let context = request.context();

    // create the connection handles
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        context.clone(),
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    // possibly long running task
    // if this returns a streaming response, the stream handle will be armed and captured by the response stream
    let response =
        match tokio::spawn(completions(state, request, stream_handle).in_current_span()).await {
            Ok(response) => response,
            Err(e) => {
                let err_response = ErrorMessage::internal_server_error(&format!(
                    "Failed to await chat completions task: {:?}",
                    e,
                ));
                emit_openai_response_log(
                    &request_id,
                    &raw_model,
                    "completions",
                    streaming,
                    err_response.0.as_u16(),
                    error_response_payload(&err_response),
                );
                return Err(err_response);
            }
        };

    // if we got here, then we will return a response and the potentially long running task has completed successfully
    // without need to be cancelled.
    connection_handle.disarm();

    if let Err(err_response) = &response {
        emit_openai_response_log(
            &request_id,
            &raw_model,
            "completions",
            streaming,
            err_response.0.as_u16(),
            error_response_payload(err_response),
        );
    }

    response
}

#[tracing::instrument(skip_all)]
async fn completions(
    state: Arc<service_v2::State>,
    mut request: Context<NvCreateCompletionRequest>,
    stream_handle: ConnectionHandle,
) -> Result<Response, ErrorResponse> {
    use crate::protocols::openai::completions::get_prompt_batch_size;

    // return a 503 if the service is not ready
    check_ready(&state)?;

    apply_completion_max_output_len_cap(&mut request);

    // Validate stream_options is only used when streaming (NVBug 5662680)
    validate_completion_stream_options(&request)?;

    validate_completion_fields_generic(&request)?;

    // Detect batch prompts
    let batch_size = get_prompt_batch_size(&request.inner.prompt);
    let n = request.inner.n.unwrap_or(1);

    // If single prompt or single-element batch, use original flow
    if batch_size == 1 {
        return completions_single(state, request, stream_handle).await;
    }

    // Batch processing: handle multiple prompts
    completions_batch(state, request, stream_handle, batch_size, n).await
}

/// Handle single prompt completions (original logic)
#[tracing::instrument(skip_all)]
async fn completions_single(
    state: Arc<service_v2::State>,
    mut request: Context<NvCreateCompletionRequest>,
    stream_handle: ConnectionHandle,
) -> Result<Response, ErrorResponse> {
    let request_id = request.id().to_string();

    // todo - decide on default
    let streaming = request.inner.stream.unwrap_or(false);

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let requested_model = request.inner.model.clone();
    let model = state.manager().resolve_canonical_name(&requested_model);
    request.inner.model = model.clone();
    let metrics_model = model.clone();
    tracing::info!(
        requested_model = %requested_model,
        canonical_model = %model,
        "Resolved chat request model"
    );

    // Create inflight_guard early to ensure all errors are counted
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &metrics_model,
        Endpoint::Completions,
        streaming,
    );

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state
        .metrics_clone()
        .create_http_queue_guard(&metrics_model);

    // todo - error handling should be more robust
    let (engine, parsing_options) = state
        .manager()
        .get_completions_engine_with_parsing(&model)
        .map_err(|_| {
            let err_response = ErrorMessage::model_not_found();
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    let mut response_collector = state
        .metrics_clone()
        .create_response_collector(&metrics_model);

    // prepare to process any annotations
    let annotations = request.annotations();

    // issue the generate call on the engine
    let stream = engine.generate(request).await.map_err(|e| {
        let err_response = ErrorMessage::from_anyhow(e, "Failed to generate completions");
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

    // capture the context to cancel the stream if the client disconnects
    let ctx = stream.context();

    let annotations = annotations.map_or(Vec::new(), |annotations| {
        annotations
            .iter()
            .filter_map(|annotation| {
                if annotation == ANNOTATION_REQUEST_ID {
                    Annotated::<NvCreateCompletionResponse>::from_annotation(
                        ANNOTATION_REQUEST_ID,
                        &request_id,
                    )
                    .ok()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    });

    // apply any annotations to the front of the stream
    let stream = stream::iter(annotations).chain(stream);

    if streaming {
        // For streaming, we'll drop the http_queue_guard on the first token
        let mut http_queue_guard = Some(http_queue_guard);

        // Payload log accumulators for streaming completions.
        let log_payloads = log_payloads_enabled();
        let mut payload_text_bufs: HashMap<u32, String> = HashMap::new();
        let request_id_for_log = request_id.clone();
        let model_for_log = model.clone();

        let stream = stream
            .map(move |response| {
                // Accumulate text content for payload logging before response is consumed.
                let mut is_final = false;
                if log_payloads {
                    if let Some(data) = &response.data {
                        // data.inner is CreateCompletionResponse (NvCreateCompletionResponse
                        // wraps it with #[serde(flatten)] but no Deref)
                        for choice in &data.inner.choices {
                            let buf = payload_text_bufs.entry(choice.index).or_default();
                            if buf.len() < MAX_PAYLOAD_ACCUMULATE_BYTES {
                                buf.push_str(&choice.text);
                            }
                            if choice.finish_reason.is_some() {
                                is_final = true;
                            }
                        }
                    }
                }

                // Calls observe_response() on each token
                let sse_result = process_response_using_event_converter_and_observe_metrics(
                    EventConverter::from(response),
                    &mut response_collector,
                    &mut http_queue_guard,
                );

                // Emit assembled payload log on the final chunk.
                if is_final {
                    let choices_json: Vec<serde_json::Value> = payload_text_bufs
                        .iter()
                        .map(|(idx, text)| serde_json::json!({ "index": idx, "text": text }))
                        .collect();
                    emit_openai_response_log(
                        &request_id_for_log,
                        &model_for_log,
                        "completions",
                        true,
                        StatusCode::OK.as_u16(),
                        serde_json::json!({ "choices": choices_json }),
                    );
                }

                sse_result
            })
            .filter_map(|result| {
                use futures::future;
                // Transpose Result<Option<T>> -> Option<Result<T>>
                future::ready(result.transpose())
            });
        let stream = monitor_for_disconnects(stream, ctx, inflight_guard, stream_handle);

        let mut sse_stream = Sse::new(stream);

        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        let mut response = sse_stream.into_response();
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );

        Ok(response)
    } else {
        // Tap the stream to collect metrics for non-streaming requests without altering items
        let mut http_queue_guard = Some(http_queue_guard);
        let stream = stream.inspect(move |response| {
            // Calls observe_response() on each token - drops http_queue_guard on first token
            process_response_and_observe_metrics(
                response,
                &mut response_collector,
                &mut http_queue_guard,
            );
        });

        let mut response =
            NvCreateCompletionResponse::from_annotated_stream(stream, parsing_options)
                .await
                .map_err(|e| {
                    tracing::error!(
                        "Failed to fold completions stream for {}: {:?}",
                        request_id,
                        e
                    );
                    let err_response = ErrorMessage::internal_server_error(&format!(
                        "Failed to fold completions stream for {}: {:?}",
                        request_id, e
                    ));
                    inflight_guard.mark_error(extract_error_type_from_response(&err_response));
                    err_response
                })?;
        response.inner.model = model.clone();

        emit_openai_response_log(
            &request_id,
            &model,
            "completions",
            false,
            StatusCode::OK.as_u16(),
            serde_json::to_value(&response).unwrap_or(serde_json::Value::Null),
        );

        inflight_guard.mark_ok();
        // If the engine context was killed (client disconnect), the response was
        // assembled but never delivered. Override to cancelled.
        if ctx.is_killed() {
            inflight_guard.mark_error(ErrorType::Cancelled);
        }
        Ok(Json(response).into_response())
    }
}

/// Handle batch prompt completions (multiple prompts with n choices each)
#[tracing::instrument(skip_all)]
async fn completions_batch(
    state: Arc<service_v2::State>,
    mut request: Context<NvCreateCompletionRequest>,
    stream_handle: ConnectionHandle,
    batch_size: usize,
    n: u8,
) -> Result<Response, ErrorResponse> {
    use crate::protocols::openai::completions::extract_single_prompt;
    use futures::stream::{self, StreamExt};

    let request_id = request.id().to_string();
    let streaming = request.inner.stream.unwrap_or(false);
    let model = state.manager().resolve_canonical_name(&request.inner.model);
    request.inner.model = model.clone();
    let metrics_model = model.clone();

    // Create inflight_guard early to ensure all errors are counted
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &metrics_model,
        Endpoint::Completions,
        streaming,
    );

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state
        .metrics_clone()
        .create_http_queue_guard(&metrics_model);

    let (engine, parsing_options) = state
        .manager()
        .get_completions_engine_with_parsing(&model)
        .map_err(|_| {
            let err_response = ErrorMessage::model_not_found();
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    let mut response_collector = state
        .metrics_clone()
        .create_response_collector(&metrics_model);

    // prepare to process any annotations
    let annotations = request.annotations();

    // Generate streams for each prompt in the batch
    let mut all_streams = Vec::new();
    let mut first_ctx = None;

    for prompt_idx in 0..batch_size {
        // Extract single prompt at this index
        let single_prompt = extract_single_prompt(&request.inner.prompt, prompt_idx);

        // Create a new request with this single prompt
        let mut single_request = request.content().clone();
        single_request.inner.prompt = single_prompt;

        // Generate unique request_id for each prompt: original_id-{prompt_idx}
        let unique_request_id = format!("{}-{}", request.id(), prompt_idx);
        let single_request_context = Context::with_id(single_request, unique_request_id);

        // Generate stream for this prompt
        let stream = engine.generate(single_request_context).await.map_err(|e| {
            let err_response = ErrorMessage::from_anyhow(e, "Failed to generate completions");
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

        // Capture context from first stream
        if first_ctx.is_none() {
            first_ctx = Some(stream.context());
        }

        // Remap choice indices: choice.index += prompt_idx * n
        let prompt_idx_u32 = prompt_idx as u32;
        let n_u32 = n as u32;
        let remapped_stream = stream.map(move |mut response| {
            if let Some(ref mut data) = response.data {
                for choice in &mut data.inner.choices {
                    choice.index += prompt_idx_u32 * n_u32;
                }
            }
            response
        });

        all_streams.push(remapped_stream);
    }

    // Merge all streams
    let merged_stream = stream::select_all(all_streams);

    // capture the context to cancel the stream if the client disconnects
    let ctx = first_ctx.expect("At least one stream should be generated");

    let annotations_vec = annotations.map_or(Vec::new(), |annotations| {
        annotations
            .iter()
            .filter_map(|annotation| {
                if annotation == ANNOTATION_REQUEST_ID {
                    Annotated::<NvCreateCompletionResponse>::from_annotation(
                        ANNOTATION_REQUEST_ID,
                        &request_id,
                    )
                    .ok()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    });

    // apply any annotations to the front of the stream
    let merged_stream = stream::iter(annotations_vec).chain(merged_stream);

    if streaming {
        // For streaming, we'll drop the http_queue_guard on the first token
        let mut http_queue_guard = Some(http_queue_guard);
        let stream = merged_stream
            .map(move |response| {
                // Calls observe_response() on each token
                process_response_using_event_converter_and_observe_metrics(
                    EventConverter::from(response),
                    &mut response_collector,
                    &mut http_queue_guard,
                )
            })
            .filter_map(|result| {
                use futures::future;
                // Transpose Result<Option<T>> -> Option<Result<T>>
                future::ready(result.transpose())
            });
        let stream = monitor_for_disconnects(stream, ctx, inflight_guard, stream_handle);

        let mut sse_stream = Sse::new(stream);

        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        let mut response = sse_stream.into_response();
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );

        Ok(response)
    } else {
        // Tap the stream to collect metrics for non-streaming requests without altering items
        let mut http_queue_guard = Some(http_queue_guard);
        let stream = merged_stream.inspect(move |response| {
            // Calls observe_response() on each token - drops http_queue_guard on first token
            process_response_and_observe_metrics(
                response,
                &mut response_collector,
                &mut http_queue_guard,
            );
        });

        let mut response =
            NvCreateCompletionResponse::from_annotated_stream(stream, parsing_options)
                .await
                .map_err(|e| {
                    tracing::error!(
                        "Failed to fold completions stream for {}: {:?}",
                        request_id,
                        e
                    );
                    let err_response = ErrorMessage::internal_server_error(&format!(
                        "Failed to fold completions stream for {}: {:?}",
                        request_id, e
                    ));
                    inflight_guard.mark_error(extract_error_type_from_response(&err_response));
                    err_response
                })?;
        response.inner.model = model.clone();

        inflight_guard.mark_ok();
        // If the engine context was killed (client disconnect), the response was
        // assembled but never delivered. Override to cancelled.
        if ctx.is_killed() {
            inflight_guard.mark_error(ErrorType::Cancelled);
        }
        Ok(Json(response).into_response())
    }
}

#[tracing::instrument(skip_all)]
async fn embeddings(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateEmbeddingRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    let request_id = get_or_create_request_id(request.inner.user.as_deref(), &headers);
    let mut request = Context::with_id(request, request_id.clone());
    let request_id = request.id().to_string();

    // Embeddings are typically not streamed, so we default to non-streaming
    let streaming = false;

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = state.manager().resolve_canonical_name(&request.inner.model);
    request.inner.model = model.clone();
    let metrics_model = model.clone();

    // Create inflight_guard early to ensure all errors are counted
    let mut inflight = state.metrics_clone().create_inflight_guard(
        &metrics_model,
        Endpoint::Embeddings,
        streaming,
    );

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state
        .metrics_clone()
        .create_http_queue_guard(&metrics_model);

    // todo - error handling should be more robust
    let engine = state.manager().get_embeddings_engine(&model).map_err(|_| {
        let err_response = ErrorMessage::model_not_found();
        inflight.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

    let mut response_collector = state
        .metrics_clone()
        .create_response_collector(&metrics_model);

    // issue the generate call on the engine
    let stream = engine.generate(request).await.map_err(|e| {
        let err_response = ErrorMessage::from_anyhow(e, "Failed to generate embeddings");
        inflight.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

    // Process stream to collect metrics and drop http_queue_guard on first token
    let mut http_queue_guard = Some(http_queue_guard);
    let stream = stream.inspect(move |response| {
        // Calls observe_response() on each token - drops http_queue_guard on first token
        process_response_and_observe_metrics(
            response,
            &mut response_collector,
            &mut http_queue_guard,
        );
    });

    // Embeddings are typically returned as a single response (non-streaming)
    // so we fold the stream into a single response
    let response = NvCreateEmbeddingResponse::from_annotated_stream(stream)
        .await
        .map_err(|e| {
            tracing::error!(
                "Failed to fold embeddings stream for {}: {:?}",
                request_id,
                e
            );
            let err_response =
                ErrorMessage::internal_server_error("Failed to fold embeddings stream");
            inflight.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    inflight.mark_ok();
    Ok(Json(response).into_response())
}

async fn handler_chat_completions(
    State((state, template)): State<(Arc<service_v2::State>, Option<RequestTemplate>)>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    let payload_value =
        serde_json::from_slice::<serde_json::Value>(&body).unwrap_or(serde_json::Value::Null);
    let payload_obj = payload_value.as_object();
    let request_id = get_or_create_request_id(
        payload_obj
            .and_then(|obj| obj.get("user"))
            .and_then(serde_json::Value::as_str),
        &headers,
    );
    let streaming = payload_obj
        .and_then(|obj| obj.get("stream"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let raw_model = payload_obj
        .and_then(|obj| obj.get("model"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();

    emit_openai_request_log(
        &request_id,
        &raw_model,
        "chat_completions",
        streaming,
        header_map_to_json(&headers),
        payload_value.clone(),
    );

    let mut backend_payload_value = payload_value.clone();
    normalize_chat_compat_payload(&mut backend_payload_value);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_value(backend_payload_value)
        .map_err(|e| {
            let err_response = ErrorMessage::from_http_error(HttpError {
                code: 400,
                message: format!("Failed to deserialize the JSON body into the target type: {e}"),
            });
            emit_openai_response_log(
                &request_id,
                &raw_model,
                "chat_completions",
                streaming,
                err_response.0.as_u16(),
                error_response_payload(&err_response),
            );
            err_response
        })?;

    request.nvext = apply_header_routing_overrides(request.nvext.take(), &headers);

    // create the context for the request
    let cancellation_labels = CancellationLabels {
        model: request.inner.model.clone(),
        endpoint: Endpoint::ChatCompletions.to_string(),
        request_type: if streaming { "stream" } else { "unary" }.to_string(),
    };
    let request = Context::with_id(request, request_id.clone());
    let context = request.context();

    // create the connection handles
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        context.clone(),
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    let response = match tokio::spawn(
        chat_completions(state, template, request, stream_handle).in_current_span(),
    )
    .await
    {
        Ok(response) => response,
        Err(e) => {
            let err_response = ErrorMessage::internal_server_error(&format!(
                "Failed to await chat completions task: {:?}",
                e,
            ));
            emit_openai_response_log(
                &request_id,
                &raw_model,
                "chat_completions",
                streaming,
                err_response.0.as_u16(),
                error_response_payload(&err_response),
            );
            return Err(err_response);
        }
    };

    // if we got here, then we will return a response and the potentially long running task has completed successfully
    // without need to be cancelled.
    connection_handle.disarm();

    if let Err(err_response) = &response {
        emit_openai_response_log(
            &request_id,
            &raw_model,
            "chat_completions",
            streaming,
            err_response.0.as_u16(),
            error_response_payload(err_response),
        );
    }

    response
}

/// Checks if an Annotated event represents a backend error and extracts error information.
/// Returns Some((message, status_code)) if it's an error, None otherwise.
fn extract_backend_error_if_present<T: serde::Serialize>(
    event: &Annotated<T>,
) -> Option<(String, StatusCode)> {
    #[derive(serde::Deserialize)]
    struct ErrorPayload {
        message: Option<String>,
        code: Option<u16>,
    }

    // Check if event type is "error" (from postprocessor when FinishReason::Error is encountered)
    if let Some(event_type) = &event.event
        && event_type == "error"
    {
        // Extract error string: prefer DynamoError field, fallback to legacy comment.
        // Use message() instead of to_string() for DynamoError to avoid prefixing
        // the ErrorType (e.g., "Unknown: {...}"), which would break JSON parsing.
        let error_str = if let Some(ref dynamo_err) = event.error {
            let mut parts = Vec::new();
            let mut current: Option<&dyn std::error::Error> = Some(dynamo_err);
            while let Some(e) = current {
                if let Some(de) = e.downcast_ref::<dynamo_runtime::error::DynamoError>() {
                    parts.push(de.message().to_string());
                } else {
                    parts.push(e.to_string());
                }
                current = e.source();
            }
            parts.join(", ")
        } else {
            event
                .comment
                .as_ref()
                .map(|c| c.join(", "))
                .unwrap_or_else(|| "Unknown error".to_string())
        };

        // Try to parse as error JSON to extract status code
        if let Ok(error_payload) = serde_json::from_str::<ErrorPayload>(&error_str) {
            let code = error_payload
                .code
                .and_then(|c| StatusCode::from_u16(c).ok())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let message = error_payload.message.unwrap_or(error_str);
            return Some((message, code));
        }

        return Some((error_str, StatusCode::INTERNAL_SERVER_ERROR));
    }

    // Check if the data payload itself contains an error structure with code >= 400
    if let Some(data) = &event.data
        && let Ok(json_value) = serde_json::to_value(data)
        && let Ok(error_payload) = serde_json::from_value::<ErrorPayload>(json_value.clone())
        && let Some(code_num) = error_payload.code
        && code_num >= 400
    {
        let code = StatusCode::from_u16(code_num).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let message = error_payload
            .message
            .unwrap_or_else(|| json_value.to_string());
        return Some((message, code));
    }

    // Check if comment contains error information (without event: error)
    if let Some(comments) = &event.comment
        && !comments.is_empty()
    {
        let comment_str = comments.join(", ");

        // Try to parse comment as error JSON with code >= 400
        if let Ok(error_payload) = serde_json::from_str::<ErrorPayload>(&comment_str)
            && let Some(code_num) = error_payload.code
            && code_num >= 400
        {
            let code = StatusCode::from_u16(code_num).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let message = error_payload.message.unwrap_or(comment_str);
            return Some((message, code));
        }

        // Comments present with no data AND no event type indicates error
        // (events with event types like "request_id" or "event.dynamo.test.sentinel" are annotations)
        if event.data.is_none() && event.event.is_none() {
            return Some((comment_str, StatusCode::INTERNAL_SERVER_ERROR));
        }
    }

    None
}

/// Checks if the first event in the stream is a backend error.
/// Returns Err(ErrorResponse) if error detected, Ok(stream) otherwise.
pub(super) async fn check_for_backend_error(
    mut stream: impl futures::Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>
    + Send
    + Unpin
    + 'static,
) -> Result<
    impl futures::Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send,
    ErrorResponse,
> {
    use futures::stream::StreamExt;

    // Peek at the first event
    if let Some(first_event) = stream.next().await {
        // Check if it's an error event
        if let Some((error_msg, status_code)) = extract_backend_error_if_present(&first_event) {
            let (error_msg, status_code) = normalize_backend_error(error_msg, status_code);
            return Err((
                status_code,
                Json(ErrorMessage {
                    message: error_msg,
                    error_type: map_error_code_to_error_type(status_code),
                    code: status_code.as_u16(),
                }),
            ));
        }

        // Not an error - reconstruct stream with first event
        let reconstructed_stream = futures::stream::iter(vec![first_event]).chain(stream);
        Ok(reconstructed_stream)
    } else {
        // Empty stream - this shouldn't happen but handle gracefully
        Ok(futures::stream::iter(vec![]).chain(stream))
    }
}

fn normalize_backend_error(error_msg: String, status_code: StatusCode) -> (String, StatusCode) {
    if status_code == StatusCode::INTERNAL_SERVER_ERROR
        && (error_msg.contains("Grammar error:")
            || error_msg.contains("Failed to convert the grammar from GBNF to Lark:"))
    {
        return (error_msg, StatusCode::BAD_REQUEST);
    }

    (error_msg, status_code)
}

/// Serialize `payload` and wrap it as an SSE event with the given name.
fn make_dispatch_event(
    event_name: &str,
    payload: &impl serde::Serialize,
) -> Option<Result<Event, axum::Error>> {
    match serde_json::to_string(payload) {
        Ok(json) => Some(Ok(Event::default().event(event_name).data(json))),
        Err(e) => {
            tracing::warn!("streaming_{event_name}: failed to serialize: {e}");
            None
        }
    }
}

/// Emits early `event: tool_call_dispatch` SSE events for any complete tool calls found in a
/// streaming response chunk, when `DYN_ENABLE_STREAMING_TOOL_DISPATCH` is enabled.
///
/// Dynamo backends emit each tool call as a single complete chunk (id + name + arguments
/// all present), so we can dispatch immediately upon seeing the chunk rather than waiting
/// for `finish_reason="tool_calls"` to arrive. Each event payload includes `choice_index`
/// for correct disambiguation when `n > 1`.
fn streaming_tool_dispatch_events(
    response: &crate::types::Annotated<NvCreateChatCompletionStreamResponse>,
    dispatched_ids: &mut HashSet<String>,
) -> Vec<Result<Event, axum::Error>> {
    let Some(data) = &response.data else {
        return vec![];
    };

    let mut events = vec![];
    for choice in &data.choices {
        let Some(tool_calls) = &choice.delta.tool_calls else {
            continue;
        };
        for chunk in tool_calls {
            // Only dispatch when the tool call is fully formed (id + name + arguments)
            let has_name_and_args = chunk
                .function
                .as_ref()
                .is_some_and(|f| f.name.is_some() && f.arguments.is_some());

            if let (true, Some(id)) = (has_name_and_args, &chunk.id) {
                // Skip already-dispatched tool calls (dedup guard, matches
                // the stopped/done flags in Anthropic/Responses converters).
                if !dispatched_ids.insert(id.clone()) {
                    continue;
                }
                let payload = serde_json::json!({
                    "choice_index": choice.index,
                    "tool_call": chunk,
                });
                events.extend(make_dispatch_event("tool_call_dispatch", &payload));
            }
        }
    }
    events
}

/// Accumulates reasoning tokens and emits a single `event: reasoning_dispatch` SSE event
/// when the complete reasoning block has been decoded (i.e. when `reasoning_content`
/// transitions from `Some(token)` to `None`), matching the UX of `tool_call_dispatch`.
///
/// The buffer is maintained across chunks by the caller (captured in the flat_map closure).
/// Flushing also occurs when `finish_reason` is set, to handle max_tokens during reasoning.
fn accumulate_reasoning_dispatch(
    response: &crate::types::Annotated<NvCreateChatCompletionStreamResponse>,
    buffers: &mut HashMap<u32, String>,
) -> Vec<Result<Event, axum::Error>> {
    let Some(data) = &response.data else {
        return vec![];
    };

    let mut events = vec![];
    for choice in &data.choices {
        let buffer = buffers.entry(choice.index).or_default();
        let has_reasoning = choice
            .delta
            .reasoning_content
            .as_ref()
            .is_some_and(|r| !r.is_empty());

        if has_reasoning {
            buffer.push_str(choice.delta.reasoning_content.as_ref().unwrap());
        }

        // Emit when reasoning transitions to None OR when the stream ends (finish_reason).
        if !buffer.is_empty() && (!has_reasoning || choice.finish_reason.is_some()) {
            let payload = serde_json::json!({
                "index": choice.index,
                "reasoning_content": buffer.as_str(),
            });
            events.extend(make_dispatch_event("reasoning_dispatch", &payload));
            buffer.clear();
        }
    }
    events
}

/// Normalize streaming finish_reason for backends that emit complete tool calls
/// in one chunk and then send a separate terminal `stop` chunk.
fn normalize_stream_tool_call_finish_reason(
    response: &mut Annotated<NvCreateChatCompletionStreamResponse>,
    choices_with_tool_calls: &mut HashSet<u32>,
) {
    let Some(data) = response.data.as_mut() else {
        return;
    };

    for choice in &mut data.choices {
        if choice
            .delta
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| !tool_calls.is_empty())
        {
            choices_with_tool_calls.insert(choice.index);
        }

        if choices_with_tool_calls.contains(&choice.index)
            && choice.finish_reason == Some(dynamo_async_openai::types::FinishReason::Stop)
        {
            choice.finish_reason = Some(dynamo_async_openai::types::FinishReason::ToolCalls);
        }
    }
}

/// Returns `true` when any streaming chat choice carries a `finish_reason`.
fn is_final_chat_payload_chunk(
    response: &Annotated<NvCreateChatCompletionStreamResponse>,
    expect_usage_chunk: bool,
) -> bool {
    let Some(data) = &response.data else {
        return false;
    };

    if expect_usage_chunk {
        return data.usage.is_some();
    }

    data.choices
        .iter()
        .any(|choice| choice.finish_reason.is_some())
}

/// OpenAI Chat Completions Request Handler
///
/// This method will handle the incoming request for the /v1/chat/completions endpoint. The endpoint is a "source"
/// for an [`super::OpenAIChatCompletionsStreamingEngine`] and will return a stream of responses which will be
/// forward to the client.
///
/// Note: For all requests, streaming or non-streaming, we always call the engine with streaming enabled. For
/// non-streaming requests, we will fold the stream into a single response as part of this handler.
async fn chat_completions(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    mut request: Context<NvCreateChatCompletionRequest>,
    mut stream_handle: ConnectionHandle,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    let request_id = request.id().to_string();

    // Determine streaming mode early
    // todo - decide on default
    let streaming = request.inner.stream.unwrap_or(false);

    // Apply template values first to resolve the model before creating metrics guards
    if let Some(template) = template {
        if request.inner.model.is_empty() {
            request.inner.model = template.model.clone();
        }
        if request.inner.temperature.unwrap_or(0.0) == 0.0 {
            request.inner.temperature = Some(template.temperature);
        }
        if request.inner.max_completion_tokens.unwrap_or(0) == 0 {
            request.inner.max_completion_tokens = Some(template.max_completion_tokens);
        }
    }
    apply_chat_max_output_len_cap(&mut request);

    // Capture the resolved model after template application for metrics and engine lookup
    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    // todo - determine the proper error code for when a request model is not present
    let requested_model = request.inner.model.clone();
    let model = state.manager().resolve_canonical_name(&requested_model);
    request.inner.model = model.clone();
    let metrics_model = model.clone();
    tracing::info!(
        requested_model = %requested_model,
        canonical_model = %model,
        "Resolved chat request model"
    );

    tracing::trace!("Received chat completions request: {:?}", request.content());

    // Create inflight_guard early to ensure all errors (including validation) are counted
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &metrics_model,
        Endpoint::ChatCompletions,
        streaming,
    );

    // Handle unsupported fields - if Some(resp) is returned by
    // validate_chat_completion_unsupported_fields,
    // then a field was used that is unsupported. We will log an error message
    // and early return a 501 NOT_IMPLEMENTED status code. Otherwise, proceeed.
    if let Err(err_response) = validate_chat_completion_unsupported_fields(&request) {
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        return Err(err_response);
    }

    // Handle required fields like messages shouldn't be empty.
    if let Err(err_response) = validate_chat_completion_required_fields(&request) {
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        return Err(err_response);
    }

    // Validate stream_options is only used when streaming (NVBug 5662680)
    if let Err(err_response) = validate_chat_completion_stream_options(&request) {
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        return Err(err_response);
    }

    // Handle Rest of Validation Errors
    if let Err(err_response) = validate_chat_completion_fields_generic(&request) {
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        return Err(err_response);
    }

    state
        .metrics_clone()
        .observe_chat_request_shape(request.content(), streaming);

    // Create HTTP queue guard after template resolution so labels are correct
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    tracing::trace!("Getting chat completions engine for model: {}", model);

    let (engine, parsing_options) = state
        .manager()
        .get_chat_completions_engine_with_parsing(&model)
        .map_err(|_| {
            let err_response = ErrorMessage::model_not_found();
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    let mut response_collector = state
        .metrics_clone()
        .create_response_collector(&metrics_model);
    let include_empty_tool_calls_in_stream_log = request
        .inner
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty());
    let request_wants_logprobs =
        request.inner.logprobs.unwrap_or(false) || request.inner.top_logprobs.unwrap_or(0) > 0;

    // The preprocessor forces usage emission for chat-completion streams so clients always get
    // a final usage-only chunk. Match the actual emitted stream contract here rather than the
    // raw incoming request shape, otherwise payload logging will finalize one chunk too early and
    // drop `usage` from the logged response.
    let expect_usage_chunk = true;

    let annotations = request.annotations();

    // issue the generate call on the engine
    let stream = engine.generate(request).await.map_err(|e| {
        let err_response = ErrorMessage::from_anyhow(e, "Failed to generate completions");
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

    // capture the context to cancel the stream if the client disconnects
    let ctx = stream.context();

    // prepare any requested annotations
    let annotations = annotations.map_or(Vec::new(), |annotations| {
        annotations
            .iter()
            .filter_map(|annotation| {
                if annotation == ANNOTATION_REQUEST_ID {
                    Annotated::from_annotation(ANNOTATION_REQUEST_ID, &request_id).ok()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    });

    // apply any annotations to the front of the stream
    let stream = stream::iter(annotations).chain(stream);

    // todo - tap the stream and propagate request level metrics
    // note - we might do this as part of the post processing set to make it more generic

    if streaming {
        // For streaming responses, we return HTTP 200 immediately without checking for errors.
        // Once HTTP 200 OK is sent, we cannot change the status code, so any backend errors
        // must be delivered as SSE events with `event: error` in the stream (handled by
        // EventConverter and monitor_for_disconnects). This is standard SSE behavior.
        stream_handle.arm(); // allows the system to detect client disconnects and cancel the LLM generation

        let mut http_queue_guard = Some(http_queue_guard);
        let tool_dispatch_enabled = state.streaming_tool_dispatch_enabled();
        let reasoning_dispatch_enabled = state.streaming_reasoning_dispatch_enabled();
        let mut reasoning_buffer: HashMap<u32, String> = HashMap::new();
        let mut dispatched_tool_ids: HashSet<String> = HashSet::new();
        let mut choices_with_tool_calls: HashSet<u32> = HashSet::new();

        let log_payloads = log_payloads_enabled();
        let parsing_options_for_log = parsing_options.clone();
        let mut payload_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = Vec::new();
        let request_id_for_log = request_id.clone();
        let model_for_log = model.clone();
        let include_empty_tool_calls_in_stream_log = include_empty_tool_calls_in_stream_log;

        // flat_map lets us optionally prepend extra SSE events before each regular chunk:
        //   - `event: tool_call_dispatch`  — complete tool call detected early (tool dispatch)
        //   - `event: reasoning_dispatch`  — complete reasoning block (emitted once)
        // When both flags are off the flat_map is equivalent to the original map + filter_map.
        let stream = stream.flat_map(move |mut response| {
            if let Some(data) = response.data.as_mut() {
                data.model = model_for_log.clone();
            }
            normalize_stream_tool_call_finish_reason(&mut response, &mut choices_with_tool_calls);

            // Extract side-channel events before the response is consumed by EventConverter.
            let mut events: Vec<Result<Event, axum::Error>> = vec![];
            if tool_dispatch_enabled {
                events.extend(streaming_tool_dispatch_events(
                    &response,
                    &mut dispatched_tool_ids,
                ));
            }
            if reasoning_dispatch_enabled {
                events.extend(accumulate_reasoning_dispatch(
                    &response,
                    &mut reasoning_buffer,
                ));
            }

            let is_final = if log_payloads {
                payload_chunks.push(response.clone());
                is_final_chat_payload_chunk(&response, expect_usage_chunk)
            } else {
                false
            };

            // Convert to SSE event (this consumes the response).
            // EventConverter will detect `event: "error"` and convert to SSE error events.
            let sse_result = process_response_using_event_converter_and_observe_metrics(
                EventConverter::from(response),
                &mut response_collector,
                &mut http_queue_guard,
            );

            if is_final {
                let response_chunks = std::mem::take(&mut payload_chunks);
                let request_id_for_log = request_id_for_log.clone();
                let model_for_log = model_for_log.clone();
                let parsing_options_for_log = parsing_options_for_log.clone();
                tokio::spawn(async move {
                    let stream = stream::iter(response_chunks);
                    if let Ok(final_response) =
                        NvCreateChatCompletionResponse::from_annotated_stream(
                            stream,
                            parsing_options_for_log,
                        )
                        .await
                    {
                        emit_openai_response_log_with_options(
                            &request_id_for_log,
                            &model_for_log,
                            "chat_completions",
                            true,
                            StatusCode::OK.as_u16(),
                            serde_json::to_value(&final_response)
                                .unwrap_or(serde_json::Value::Null),
                            include_empty_tool_calls_in_stream_log,
                            true,
                        );
                    }
                });
            }

            // Side-channel events come first, then the regular data event.
            match sse_result {
                Ok(Some(ev)) => events.push(Ok(ev)),
                Ok(None) => {}
                Err(e) => events.push(Err(e)),
            }
            stream::iter(events)
        });
        let stream = monitor_for_disconnects(stream, ctx, inflight_guard, stream_handle);

        let mut sse_stream = Sse::new(stream);

        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        let mut response = sse_stream.into_response();
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );

        Ok(response)
    } else {
        // Check first event for backend errors before aggregating (non-streaming only)
        let stream_with_check =
            check_for_backend_error(stream)
                .await
                .map_err(|error_response| {
                    tracing::error!(request_id, "Backend error detected: {:?}", error_response);
                    inflight_guard.mark_error(extract_error_type_from_response(&error_response));
                    error_response
                })?;

        let mut http_queue_guard = Some(http_queue_guard);
        let stream = stream_with_check.inspect(move |response| {
            // Calls observe_response() on each token - drops http_queue_guard on first token
            process_response_and_observe_metrics(
                response,
                &mut response_collector,
                &mut http_queue_guard,
            );
        });

        let mut response =
            NvCreateChatCompletionResponse::from_annotated_stream(stream, parsing_options.clone())
                .await
                .map_err(|e| {
                    tracing::error!(
                        request_id,
                        "Failed to parse chat completion response: {:?}",
                        e
                    );
                    let err_response = map_stream_parse_error_to_response(
                        "Failed to parse chat completion response",
                        &e.to_string(),
                    );
                    inflight_guard.mark_error(extract_error_type_from_response(&err_response));
                    err_response
                })?;
        response.model = model.clone();

        let mut response_payload =
            serde_json::to_value(&response).unwrap_or(serde_json::Value::Null);
        if request_wants_logprobs {
            ensure_chat_response_logprobs_field(&mut response_payload);
        }

        emit_openai_response_log(
            &request_id,
            &model,
            "chat_completions",
            false,
            StatusCode::OK.as_u16(),
            response_payload.clone(),
        );

        inflight_guard.mark_ok();
        // If the engine context was killed (client disconnect), the response was
        // assembled but never delivered. Override to cancelled.
        if ctx.is_killed() {
            inflight_guard.mark_error(ErrorType::Cancelled);
        }
        Ok(Json(response_payload).into_response())
    }
}

/// Checks for unsupported fields in the request.
/// Returns Some(response) if unsupported fields are present.
#[allow(deprecated)]
pub fn validate_chat_completion_unsupported_fields(
    request: &NvCreateChatCompletionRequest,
) -> Result<(), ErrorResponse> {
    let inner = &request.inner;

    if inner.function_call.is_some() {
        return Err(ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string()
                + "`function_call` is deprecated. Please migrate to use `tool_choice` instead.",
        ));
    }

    if inner.functions.is_some() {
        return Err(ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string()
                + "`functions` is deprecated. Please migrate to use `tools` instead.",
        ));
    }

    Ok(())
}

/// Validates that required fields are present and valid in the chat completion request
pub fn validate_chat_completion_required_fields(
    request: &NvCreateChatCompletionRequest,
) -> Result<(), ErrorResponse> {
    let inner = &request.inner;

    if inner.messages.is_empty() {
        return Err(ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string()
                + "The 'messages' field cannot be empty. At least one message is required.",
        }));
    }

    Ok(())
}

/// Validates that stream_options is only used when stream=true for chat completions (NVBug 5662680)
pub fn validate_chat_completion_stream_options(
    request: &NvCreateChatCompletionRequest,
) -> Result<(), ErrorResponse> {
    let inner = &request.inner;
    let streaming = inner.stream.unwrap_or(false);
    if !streaming && inner.stream_options.is_some() {
        return Err(ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string()
                + "The 'stream_options' field is only allowed when 'stream' is set to true.",
        }));
    }
    Ok(())
}

/// Validates a chat completion request and returns an error response if validation fails.
///
/// This function calls the `validate` method implemented for `NvCreateChatCompletionRequest`.
/// If validation fails, it maps the error into an OpenAI-compatible error response.
pub fn validate_chat_completion_fields_generic(
    request: &NvCreateChatCompletionRequest,
) -> Result<(), ErrorResponse> {
    if request
        .inner
        .tool_choice
        .as_ref()
        .is_some_and(|choice| !matches!(choice, ChatCompletionToolChoiceOption::None))
        && !request
            .inner
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
    {
        return Err(ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string()
                + "When using `tool_choice`, `tools` must be set.",
        }));
    }

    if let Some(tool_choice) = request.inner.tool_choice.as_ref()
        && let Err(err) =
            tools::get_json_schema_from_tools(Some(tool_choice), request.inner.tools.as_deref())
    {
        return Err(ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string() + &err.to_string(),
        }));
    }

    if let Some(top_p) = request.inner.top_p
        && !(0.0 < top_p && top_p <= 1.0)
    {
        return Err(ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string()
                + &format!("`top_p` must be in (0, 1], got {top_p}."),
        }));
    }

    request.validate().map_err(|e| {
        ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string() + &e.to_string(),
        })
    })
}

/// Validates that stream_options is only used when stream=true for completions (NVBug 5662680)
pub fn validate_completion_stream_options(
    request: &NvCreateCompletionRequest,
) -> Result<(), ErrorResponse> {
    let inner = &request.inner;
    let streaming = inner.stream.unwrap_or(false);
    if !streaming && inner.stream_options.is_some() {
        return Err(ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string()
                + "The 'stream_options' field is only allowed when 'stream' is set to true.",
        }));
    }
    Ok(())
}

/// Validates a completion request and returns an error response if validation fails.
///
/// This function calls the `validate` method implemented for `NvCreateCompletionRequest`.
/// If validation fails, it maps the error into an OpenAI-compatible error response.
pub fn validate_completion_fields_generic(
    request: &NvCreateCompletionRequest,
) -> Result<(), ErrorResponse> {
    request.validate().map_err(|e| {
        ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string() + &e.to_string(),
        })
    })
}

fn map_stream_parse_error_to_response(context: &str, error: &str) -> ErrorResponse {
    if error.contains("Grammar error:")
        || error.contains("Failed to convert the grammar from GBNF to Lark:")
    {
        return ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: error.to_string(),
        });
    }

    ErrorMessage::internal_server_error(&format!("{context}: {error}"))
}

/// OpenAI Responses Request Handler
///
/// This method will handle the incoming request for the /v1/responses endpoint.
async fn handler_responses(
    State((state, template)): State<(Arc<service_v2::State>, Option<RequestTemplate>)>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    let request_json =
        serde_json::from_slice::<serde_json::Value>(&body).unwrap_or(serde_json::Value::Null);
    let payload_obj = request_json.as_object();
    let request_id = get_or_create_request_id(
        payload_obj
            .and_then(|obj| obj.get("request_id"))
            .and_then(serde_json::Value::as_str),
        &headers,
    );
    let streaming = payload_obj
        .and_then(|obj| obj.get("stream"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let raw_model = payload_obj
        .and_then(|obj| obj.get("model"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();

    emit_openai_request_log(
        &request_id,
        &raw_model,
        "responses",
        streaming,
        header_map_to_json(&headers),
        request_json.clone(),
    );

    let normalized_request_json = normalize_responses_request_json(request_json);
    let mut request: NvCreateResponse =
        serde_json::from_value(normalized_request_json).map_err(|err| {
            let err_response = ErrorMessage::bad_request_from_message(format!(
                "Failed to deserialize the JSON body into the target type: {err}"
            ));
            emit_openai_response_log(
                &request_id,
                &raw_model,
                "responses",
                streaming,
                err_response.0.as_u16(),
                error_response_payload(&err_response),
            );
            err_response
        })?;

    request.nvext = apply_header_routing_overrides(request.nvext.take(), &headers);

    // create the context for the request
    let cancellation_labels = CancellationLabels {
        model: request.inner.model.clone().unwrap_or_default(),
        endpoint: Endpoint::Responses.to_string(),
        request_type: if streaming { "stream" } else { "unary" }.to_string(),
    };
    let request = Context::with_id(request, request_id.clone());
    let context = request.context();

    // create the connection handles
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        context.clone(),
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    let response =
        tokio::spawn(responses(state, template, request, stream_handle).in_current_span())
            .await
            .map_err(|e| {
                let err_response = ErrorMessage::internal_server_error(&format!(
                    "Failed to await responses task: {:?}",
                    e,
                ));
                emit_openai_response_log(
                    &request_id,
                    &raw_model,
                    "responses",
                    streaming,
                    err_response.0.as_u16(),
                    error_response_payload(&err_response),
                );
                err_response
            })?;

    // if we got here, then we will return a response and the potentially long running task has completed successfully
    // without need to be cancelled.
    connection_handle.disarm();

    if let Err(err_response) = &response {
        emit_openai_response_log(
            &request_id,
            &raw_model,
            "responses",
            streaming,
            err_response.0.as_u16(),
            error_response_payload(err_response),
        );
    }

    response
}

#[tracing::instrument(level = "debug", skip_all, fields(request_id = %request.id()))]
async fn responses(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    mut request: Context<NvCreateResponse>,
    mut stream_handle: ConnectionHandle,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    // Apply template values if present, with sensible defaults for the Responses API.
    // Unlike chat completions where backends may have their own defaults, the Responses API
    // should provide a generous default to avoid truncated responses (especially with
    // reasoning models that emit <think> tokens).
    const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;

    if let Some(template) = template {
        if request.inner.model.as_deref().unwrap_or("").is_empty() {
            request.inner.model = Some(template.model.clone());
        }
        if request.inner.temperature.is_none() {
            request.inner.temperature = Some(template.temperature);
        }
        if request.inner.max_output_tokens.is_none() {
            request.inner.max_output_tokens = Some(template.max_completion_tokens);
        }
    } else if request.inner.max_output_tokens.is_none() {
        request.inner.max_output_tokens = Some(DEFAULT_MAX_OUTPUT_TOKENS);
    }
    apply_responses_max_output_len_cap(&mut request);
    tracing::trace!("Received responses request: {:?}", request.inner);

    let model = state
        .manager()
        .resolve_canonical_name(request.inner.model.as_deref().unwrap_or_default());
    request.inner.model = Some(model.clone());
    let metrics_model = model.clone();
    let streaming = request.inner.stream.unwrap_or(false);
    let request_id = request.id().to_string();

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state
        .metrics_clone()
        .create_http_queue_guard(&metrics_model);
    let mut inflight_guard =
        state
            .metrics_clone()
            .create_inflight_guard(&metrics_model, Endpoint::Responses, streaming);

    // Handle unsupported fields - if Some(resp) is returned by validate_unsupported_fields,
    // then a field was used that is unsupported. We will log an error message
    // and early return a 501 NOT_IMPLEMENTED status code.
    if let Some(resp) = validate_response_unsupported_fields(&request) {
        inflight_guard.mark_error(ErrorType::NotImplemented);
        return Ok(resp.into_response());
    }

    // Extract request parameters before into_parts() consumes the request.
    // These are echoed back in the Response object per the OpenAI spec.
    let response_params = ResponseParams {
        model: request.inner.model.clone(),
        temperature: request.inner.temperature,
        top_p: request.inner.top_p,
        max_output_tokens: request.inner.max_output_tokens,
        store: request.inner.store,
        tools: request.inner.tools.clone(),
        tool_choice: request.inner.tool_choice.clone(),
        instructions: request.inner.instructions.clone(),
        reasoning: request.inner.reasoning.clone(),
        text: request.inner.text.clone(),
        service_tier: request.inner.service_tier,
        include: request.inner.include.clone(),
        truncation: request.inner.truncation,
    };
    let (orig_request, context) = request.into_parts();

    let mut chat_request: NvCreateChatCompletionRequest =
        orig_request.try_into().map_err(|e: anyhow::Error| {
            tracing::error!(
                request_id,
                error = %e,
                "Failed to convert NvCreateResponse to NvCreateChatCompletionRequest",
            );
            let err_response = ErrorMessage::not_implemented_error(
                VALIDATION_PREFIX.to_string()
                    + "Failed to convert responses request: "
                    + &e.to_string(),
            );
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    // Always use internal streaming for aggregation.
    // Set stream_options.include_usage so the backend sends token counts in the final chunk.
    chat_request.inner.stream = Some(true);
    chat_request.inner.stream_options =
        Some(dynamo_async_openai::types::ChatCompletionStreamOptions {
            include_usage: true,
            continuous_usage_stats: false,
        });

    state
        .metrics_clone()
        .observe_chat_request_shape(&chat_request, streaming);

    let request = context.map(|mut _req| chat_request);

    tracing::trace!("Getting chat completions engine for model: {}", model);

    let (engine, parsing_options) = state
        .manager()
        .get_chat_completions_engine_with_parsing(&model)
        .map_err(|_| {
            let err_response = ErrorMessage::model_not_found();
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    let mut response_collector = state
        .metrics_clone()
        .create_response_collector(&metrics_model);

    tracing::trace!("Issuing generate call for responses");

    // issue the generate call on the engine
    let engine_stream = engine.generate(request).await.map_err(|e| {
        let err_response = ErrorMessage::from_anyhow(e, "Failed to generate completions");
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

    // Capture the context to cancel the stream if the client disconnects
    let ctx = engine_stream.context();

    if streaming {
        // For streaming responses, we return HTTP 200 immediately without checking for errors.
        // Once HTTP 200 OK is sent, we cannot change the status code, so any backend errors
        // must be delivered as SSE events in the stream. This is standard SSE behavior.
        stream_handle.arm(); // allows the system to detect client disconnects and cancel the LLM generation

        // Streaming path: convert chat completion stream chunks to Responses API SSE events.
        // The engine yields Annotated<NvCreateChatCompletionStreamResponse>. We extract the
        // inner stream response data and convert it to Responses API events.
        use crate::protocols::openai::responses::stream_converter::ResponseStreamConverter;
        use std::sync::atomic::{AtomicBool, Ordering};

        let mut converter = ResponseStreamConverter::new(model.clone(), response_params);
        let start_events = converter.emit_start_events();

        // Use std::sync::Mutex (not tokio) since process_chunk/emit_end_events are
        // synchronous -- no .await while lock is held. Avoids async lock overhead per token.
        let converter = std::sync::Arc::new(std::sync::Mutex::new(converter));
        let converter_end = converter.clone();
        let request_id_for_log = request_id.clone();
        let model_for_log = model.clone();

        // Track whether the backend sent an error event during the stream.
        // Shared between event_stream (writer) and done_stream (reader).
        let saw_error = std::sync::Arc::new(AtomicBool::new(false));
        let saw_error_end = saw_error.clone();

        let mut http_queue_guard = Some(http_queue_guard);

        // Process each annotated chunk: extract the stream response data, convert to events
        let event_stream = engine_stream
            .inspect(move |response| {
                process_response_and_observe_metrics(
                    response,
                    &mut response_collector,
                    &mut http_queue_guard,
                );
            })
            .filter_map(move |annotated_chunk| {
                let converter = converter.clone();
                let saw_error = saw_error.clone();
                async move {
                    // Check for backend error before extracting data.
                    // Error events have data: None and event: Some("error").
                    if annotated_chunk.data.is_none() {
                        if annotated_chunk.event.as_deref() == Some("error") {
                            saw_error.store(true, Ordering::Release);
                        }
                        return None;
                    }
                    let stream_resp = annotated_chunk.data?;
                    let mut conv = converter.lock().expect("converter lock poisoned");
                    let events = conv.process_chunk(&stream_resp);
                    Some(stream::iter(events))
                }
            })
            .flatten();

        // Chain: start_events -> chunk_events -> end_events
        let start_stream = stream::iter(start_events);

        let done_stream = stream::once(async move {
            let mut conv = converter_end.lock().expect("converter lock poisoned");
            let end_events = if saw_error_end.load(Ordering::Acquire) {
                emit_openai_response_log(
                    &request_id_for_log,
                    &model_for_log,
                    "responses",
                    true,
                    StatusCode::OK.as_u16(),
                    serde_json::to_value(conv.failed_response()).unwrap_or(serde_json::Value::Null),
                );
                conv.emit_error_events()
            } else {
                emit_openai_response_log(
                    &request_id_for_log,
                    &model_for_log,
                    "responses",
                    true,
                    StatusCode::OK.as_u16(),
                    serde_json::to_value(conv.completed_response())
                        .unwrap_or(serde_json::Value::Null),
                );
                conv.emit_end_events()
            };
            stream::iter(end_events)
        })
        .flatten();

        let full_stream = start_stream.chain(event_stream).chain(done_stream);

        let full_stream = full_stream.map(|result| result.map_err(axum::Error::new));

        // Wrap with disconnect monitoring: detects client disconnects, cancels generation,
        // and defers inflight_guard.mark_ok() until the stream completes.
        let stream = monitor_for_disconnects(full_stream, ctx, inflight_guard, stream_handle);

        let mut sse_stream = Sse::new(stream);
        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        let mut response = sse_stream.into_response();
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );

        Ok(response)
    } else {
        // Non-streaming path: aggregate stream into single response

        // Check first event for backend errors before aggregating (non-streaming only)
        let stream_with_check =
            check_for_backend_error(engine_stream)
                .await
                .map_err(|error_response| {
                    tracing::error!(request_id, "Backend error detected: {:?}", error_response);
                    inflight_guard.mark_error(extract_error_type_from_response(&error_response));
                    error_response
                })?;

        let mut http_queue_guard = Some(http_queue_guard);
        let stream = stream_with_check.inspect(move |response| {
            process_response_and_observe_metrics(
                response,
                &mut response_collector,
                &mut http_queue_guard,
            );
        });

        let response =
            NvCreateChatCompletionResponse::from_annotated_stream(stream, parsing_options.clone())
                .await
                .map_err(|e| {
                    tracing::error!(request_id, "Failed to fold responses stream: {:?}", e);
                    let err_response = ErrorMessage::internal_server_error(&format!(
                        "Failed to fold responses stream: {}",
                        e
                    ));
                    inflight_guard.mark_error(extract_error_type_from_response(&err_response));
                    err_response
                })?;

        // Convert NvCreateChatCompletionResponse --> NvResponse
        let response: NvResponse = chat_completion_to_response(response, &response_params)
            .map_err(|e| {
                tracing::error!(
                    request_id,
                    "Failed to convert NvCreateChatCompletionResponse to NvResponse: {:?}",
                    e
                );
                let err_response =
                    ErrorMessage::internal_server_error("Failed to convert internal response");
                inflight_guard.mark_error(extract_error_type_from_response(&err_response));
                err_response
            })?;

        inflight_guard.mark_ok();
        // If the engine context was killed (client disconnect), the response was
        // assembled but never delivered. Override to cancelled.
        if ctx.is_killed() {
            inflight_guard.mark_error(ErrorType::Cancelled);
        }

        emit_openai_response_log(
            &request_id,
            &model,
            "responses",
            false,
            StatusCode::OK.as_u16(),
            serde_json::to_value(&response).unwrap_or(serde_json::Value::Null),
        );

        Ok(Json(response).into_response())
    }
}

/// Checks for unsupported fields in the request.
/// Returns Some(response) if unsupported fields are present.
pub fn validate_response_unsupported_fields(
    request: &NvCreateResponse,
) -> Option<impl IntoResponse> {
    let inner = &request.inner;

    if inner.background == Some(true) {
        return Some(ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string() + "`background: true` is not supported.",
        ));
    }
    if inner.previous_response_id.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string() + "`previous_response_id` is not supported.",
        ));
    }
    if inner.prompt.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string() + "`prompt` is not supported.",
        ));
    }
    if inner.store == Some(true) {
        return Some(ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string() + "`store: true` is not supported.",
        ));
    }
    None
}

// todo - abstract this to the top level lib.rs to be reused
// todo - move the service_observer to its own state/arc
fn check_ready(state: &Arc<service_v2::State>) -> Result<(), ErrorResponse> {
    super::health::check_frontend_ready(state).map_err(|message| {
        let code = StatusCode::SERVICE_UNAVAILABLE;
        (
            code,
            Json(ErrorMessage {
                message,
                error_type: map_error_code_to_error_type(code),
                code: code.as_u16(),
            }),
        )
    })
}

/// openai compatible format
/// Example:
/// {
///  "object": "list",
///  "data": [
///    {
///      "id": "model-id-0",
///      "object": "model",
///      "created": 1686935002,
///      "owned_by": "organization-owner"
///    },
///    ]
/// }
async fn list_models_openai(
    State(state): State<Arc<service_v2::State>>,
) -> Result<Response, ErrorResponse> {
    check_ready(&state)?;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut data = Vec::new();

    let models: HashSet<String> = state.manager().model_display_names();
    for model_name in models {
        data.push(ModelListing {
            id: model_name.clone(),
            object: "model", // Per OpenAI spec, this should be "model"
            created,
            owned_by: "nvidia".to_string(),
        });
    }

    let out = ListModelOpenAI {
        object: "list",
        data,
    };
    Ok(Json(out).into_response())
}

#[derive(Serialize)]
struct ListModelOpenAI {
    object: &'static str, // always "list"
    data: Vec<ModelListing>,
}

#[derive(Serialize)]
struct ModelListing {
    id: String,
    object: &'static str, // always "model" per OpenAI spec
    created: u64,         // Seconds since epoch
    owned_by: String,
}

/// Create an Axum [`Router`] for the OpenAI API Completions endpoint
/// If not path is provided, the default path is `/v1/completions`
pub fn completions_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/completions".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(handler_completions))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

/// Create an Axum [`Router`] for the OpenAI API Chat Completions endpoint
/// If not path is provided, the default path is `/v1/chat/completions`
pub fn chat_completions_router(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/chat/completions".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(handler_chat_completions))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state((state, template));
    (vec![doc], router)
}

/// Create an Axum [`Router`] for the OpenAI API Embeddings endpoint
/// If not path is provided, the default path is `/v1/embeddings`
pub fn embeddings_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/embeddings".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(embeddings))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

/// List Models
pub fn list_models_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    // Standard OpenAI compatible list models endpoint
    let openai_path = path.unwrap_or("/v1/models".to_string());
    let doc_for_openai = RouteDoc::new(axum::http::Method::GET, &openai_path);

    let router = Router::new()
        .route(&openai_path, get(list_models_openai))
        .with_state(state);

    (vec![doc_for_openai], router)
}

/// Create an Axum [`Router`] for the OpenAI API Responses endpoint
/// If not path is provided, the default path is `/v1/responses`
pub fn responses_router(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/responses".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(handler_responses))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state((state, template));
    (vec![doc], router)
}

async fn images(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateImageRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    let request_id = get_or_create_request_id(request.inner.user.as_deref(), &headers);
    let request = Context::with_id(request, request_id.clone());
    let request_id = request.id().to_string();

    // Images are typically not streamed, so we default to non-streaming
    let streaming = false;

    // Get the model name from the request (diffusion model)
    let model = request
        .inner
        .model
        .as_ref()
        .map(|m| match m {
            dynamo_async_openai::types::ImageModel::DallE2 => "dall-e-2".to_string(),
            dynamo_async_openai::types::ImageModel::DallE3 => "dall-e-3".to_string(),
            dynamo_async_openai::types::ImageModel::Other(s) => s.clone(),
        })
        .unwrap_or_else(|| "diffusion".to_string());

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    // Get the image generation engine
    let engine = state
        .manager()
        .get_images_engine(&model)
        .map_err(|_| ErrorMessage::model_not_found())?;

    // this will increment the inflight gauge for the model
    let mut inflight =
        state
            .metrics_clone()
            .create_inflight_guard(&model, Endpoint::Images, streaming);

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    // Issue the generate call on the engine
    // Note: This uses ServerStreamingEngine for internal routing/distribution,
    // NOT for client-facing SSE streaming. The stream is immediately folded into
    // a single response below.
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorMessage::from_anyhow(e, "Failed to generate images"))?;

    // Process stream to collect metrics and drop http_queue_guard on first response
    let mut http_queue_guard = Some(http_queue_guard);
    let stream = stream.inspect(move |response| {
        // Calls observe_response() on each item - drops http_queue_guard on first item
        process_response_and_observe_metrics(
            response,
            &mut response_collector,
            &mut http_queue_guard,
        );
    });

    // Images are returned as a single response (non-streaming to client)
    // Fold the internal stream into a single response
    let response = NvImagesResponse::from_annotated_stream(stream)
        .await
        .map_err(|e| {
            tracing::error!("Failed to fold images stream for {}: {:?}", request_id, e);
            ErrorMessage::internal_server_error("Failed to fold images stream")
        })?;

    inflight.mark_ok();
    Ok(Json(response).into_response())
}

/// Create an Axum [`Router`] for the OpenAI API Images endpoint
/// If not path is provided, the default path is `/v1/images/generations`
pub fn images_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/images/generations".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(images))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

async fn videos(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateVideoRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    let request_id = get_or_create_request_id(request.user.as_deref(), &headers);
    let request = Context::with_id(request, request_id.clone());
    let request_id = request.id().to_string();

    // Videos are typically not streamed, so we default to non-streaming
    let streaming = false;

    // Get the model name from the request (video generation model)
    let model = request.model.clone();

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    // Get the video generation engine
    let engine = state
        .manager()
        .get_videos_engine(&model)
        .map_err(|_| ErrorMessage::model_not_found())?;

    // this will increment the inflight gauge for the model
    let mut inflight =
        state
            .metrics_clone()
            .create_inflight_guard(&model, Endpoint::Videos, streaming);

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    // issue the generate call on the engine
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorMessage::from_anyhow(e, "Failed to generate videos"))?;

    // Process stream to collect metrics and drop http_queue_guard on first token
    let mut http_queue_guard = Some(http_queue_guard);
    let stream = stream.inspect(move |response| {
        // Calls observe_response() on each token - drops http_queue_guard on first token
        process_response_and_observe_metrics(
            response,
            &mut response_collector,
            &mut http_queue_guard,
        );
    });

    // Videos are typically returned as a single response (non-streaming)
    // so we fold the stream into a single response
    let response = NvVideosResponse::from_annotated_stream(stream)
        .await
        .map_err(|e| {
            tracing::error!("Failed to fold videos stream for {}: {:?}", request_id, e);
            ErrorMessage::internal_server_error("Failed to fold videos stream")
        })?;

    inflight.mark_ok();
    Ok(Json(response).into_response())
}

/// [EXPERIMENTAL] MJPEG streaming handler for `/v1/videos/stream`.
///
/// The backend is expected to yield one [`NvVideosResponse`] per frame, carrying a
/// JPEG-encoded frame as `data[0].b64_json`. This handler decodes each frame and
/// writes it as an MJPEG multipart boundary so the client receives a live
/// `multipart/x-mixed-replace` stream viewable directly in a browser `<img>` tag
/// or via `ffplay http://.../v1/videos/stream`.
async fn video_stream(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateVideoRequest>,
) -> Result<Response, ErrorResponse> {
    check_ready(&state)?;

    let request_id = get_or_create_request_id(request.user.as_deref(), &headers);
    let request = Context::with_id(request, request_id.clone());
    let model = request.model.clone();

    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    let engine = state
        .manager()
        .get_videos_engine(&model)
        .map_err(|_| ErrorMessage::model_not_found())?;

    let mut inflight = state
        .metrics_clone()
        .create_inflight_guard(&model, Endpoint::Videos, true);

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorMessage::from_anyhow(e, "Failed to start video stream"))?;

    // Capture the context to cancel the stream if the client disconnects.
    let ctx = stream.context();

    // Create connection monitor. The connection_handle is disarmed immediately because
    // video_stream returns the streaming body directly (graceful handler exit).
    // The stream_handle is armed below and lives inside the monitored stream so that
    // a client disconnect (body drop) signals the engine context to cancel.
    let (mut connection_handle, mut stream_handle) = create_connection_monitor(
        ctx.clone(),
        Some(state.metrics_clone()),
        CancellationLabels {
            model: model.clone(),
            endpoint: Endpoint::Videos.to_string(),
            request_type: "stream".to_string(),
        },
    )
    .await;
    connection_handle.disarm();

    let mut http_queue_guard = Some(http_queue_guard);
    let stream = stream.inspect(move |response| {
        process_response_and_observe_metrics(
            response,
            &mut response_collector,
            &mut http_queue_guard,
        );
    });

    // Map each annotated NvVideosResponse to an MJPEG boundary chunk.
    // The backend yields one response per frame with the JPEG in data[0].b64_json.
    let mjpeg_stream = stream.filter_map(|annotated| async move {
        let ann = match annotated.ok() {
            Ok(a) => a,
            Err(e) => {
                tracing::error!("Video stream error: {e}");
                return None;
            }
        };
        let response = ann.data?;
        let frame = response.data.into_iter().next()?;
        let b64 = frame.b64_json?;
        let jpeg_bytes = match base64::prelude::BASE64_STANDARD.decode(&b64) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("Failed to decode frame base64: {e}");
                return None;
            }
        };
        let header = format!(
            "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
            jpeg_bytes.len()
        );
        let mut chunk = Vec::with_capacity(header.len() + jpeg_bytes.len() + 2);
        chunk.extend_from_slice(header.as_bytes());
        chunk.extend_from_slice(&jpeg_bytes);
        chunk.extend_from_slice(b"\r\n");
        Some(Ok::<Bytes, std::convert::Infallible>(Bytes::from(chunk)))
    });

    // Arm the stream handle and monitor for client disconnects or context cancellation.
    // inflight.mark_ok() is deferred until the stream ends naturally. If the stream is
    // dropped early (client disconnect), the armed stream_handle signals the connection
    // monitor, which cancels the engine context.
    stream_handle.arm();
    let monitored_stream = async_stream::stream! {
        tokio::pin!(mjpeg_stream);
        loop {
            tokio::select! {
                frame = mjpeg_stream.next() => {
                    match frame {
                        Some(item) => yield item,
                        None => {
                            // Stream ended naturally: mark inflight OK and disarm the handle.
                            inflight.mark_ok();
                            stream_handle.disarm();
                            break;
                        }
                    }
                }
                _ = ctx.stopped() => {
                    tracing::trace!("Context stopped; breaking MJPEG stream");
                    break;
                }
            }
        }
    };

    axum::http::Response::builder()
        .status(axum::http::StatusCode::OK)
        .header(
            axum::http::header::CONTENT_TYPE,
            "multipart/x-mixed-replace; boundary=frame",
        )
        .body(Body::from_stream(monitored_stream))
        .map(|r| r.into_response())
        .map_err(|e| {
            ErrorMessage::internal_server_error(&format!("Failed to build MJPEG response: {e}"))
        })
}

/// Create an Axum [`Router`] for the OpenAI API Videos endpoint
/// If no path is provided, the default path is `/v1/videos`
///
/// Two routes are registered:
/// - `POST /v1/videos`        — non-streaming, returns a single JSON response
/// - `POST /v1/videos/stream` — MJPEG streaming via `multipart/x-mixed-replace`
pub fn videos_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/videos".to_string());
    let stream_path = format!("{}/stream", path);
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let stream_doc = RouteDoc::new(axum::http::Method::POST, &stream_path);
    let router = Router::new()
        .route(&path, post(videos))
        .route(&stream_path, post(video_stream))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc, stream_doc], router)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::discovery::ModelManagerError;
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use crate::protocols::openai::common_ext::CommonExt;
    use crate::protocols::openai::completions::NvCreateCompletionRequest;
    use crate::protocols::openai::responses::NvCreateResponse;
    use dynamo_async_openai::types::responses::{CreateResponse, Input, PromptConfig};
    use dynamo_async_openai::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
        CreateCompletionRequest,
    };

    #[test]
    fn test_apply_chat_max_output_len_cap_prefers_max_completion_tokens() {
        let mut request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                max_completion_tokens: Some(2048),
                max_tokens: Some(4096),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };

        apply_chat_max_output_len_cap_with(&mut request, Some(1024));
        assert_eq!(request.inner.max_completion_tokens, Some(1024));
        assert_eq!(request.inner.max_tokens, Some(4096));
    }

    #[test]
    fn test_apply_completion_max_output_len_cap() {
        let mut request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".to_string().into(),
                max_tokens: Some(8192),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };

        apply_completion_max_output_len_cap_with(&mut request, Some(2048));
        assert_eq!(request.inner.max_tokens, Some(2048));
    }

    #[test]
    fn test_apply_responses_max_output_len_cap() {
        let mut request = NvCreateResponse {
            inner: CreateResponse {
                model: Some("test-model".to_string()),
                input: Input::Text("Hello".to_string()),
                max_output_tokens: Some(5000),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            request_id: None,
        };

        apply_responses_max_output_len_cap_with(&mut request, Some(1024));
        assert_eq!(request.inner.max_output_tokens, Some(1024));
    }

    const BACKUP_ERROR_MESSAGE: &str = "Failed to generate completions";

    fn http_error_from_engine(code: u16) -> Result<(), anyhow::Error> {
        Err(HttpError {
            code,
            message: "custom error message".to_string(),
        })?
    }

    fn other_error_from_engine() -> Result<(), anyhow::Error> {
        Err(ModelManagerError::ModelNotFound("foo".to_string()))?
    }

    fn make_base_request() -> NvCreateResponse {
        NvCreateResponse {
            inner: CreateResponse {
                input: Input::Text("hello".into()),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
        }
    }

    #[test]
    fn test_http_error_response_from_anyhow() {
        let err = http_error_from_engine(400).unwrap_err();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::BAD_REQUEST);
        assert_eq!(response.1.message, "custom error message");
    }

    #[test]
    fn test_error_response_from_anyhow_out_of_range() {
        let err = http_error_from_engine(399).unwrap_err();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.1.message, "custom error message");

        let err = http_error_from_engine(500).unwrap_err();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.1.message, "custom error message");

        let err = http_error_from_engine(501).unwrap_err();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.1.message, "custom error message");
    }

    #[test]
    fn test_other_error_response_from_anyhow() {
        let err = other_error_from_engine().unwrap_err();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            response.1.message,
            format!(
                "{}: {}",
                BACKUP_ERROR_MESSAGE,
                other_error_from_engine().unwrap_err()
            )
        );
    }

    #[test]
    fn test_service_overloaded_error_response_from_anyhow() {
        use dynamo_runtime::pipeline::error::PipelineError;

        let err: anyhow::Error = PipelineError::ServiceOverloaded(
            "All workers are busy, please retry later".to_string(),
        )
        .into();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.1.message,
            "Service temporarily unavailable: All workers are busy, please retry later"
        );
    }

    #[test]
    fn test_instance_unavailable_error_response_from_anyhow() {
        use dynamo_runtime::pipeline::error::PipelineError;

        let err: anyhow::Error = PipelineError::InstanceUnavailable(
            "instance_id=123 not found for endpoint dynamo/backend/generate".to_string(),
        )
        .into();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.1.message,
            "Service temporarily unavailable: instance_id=123 not found for endpoint dynamo/backend/generate"
        );
    }

    #[test]
    fn test_validate_unsupported_fields_accepts_clean_request() {
        let request = make_base_request();
        let result = validate_response_unsupported_fields(&request);
        assert!(result.is_none());
    }

    #[test]
    fn test_validate_unsupported_fields_accepts_parallel_tool_calls() {
        let mut request = make_base_request();
        request.inner.parallel_tool_calls = Some(true);
        let result = validate_response_unsupported_fields(&request);
        assert!(result.is_none(), "parallel_tool_calls should be supported");
    }

    fn chat_stream_chunk(
        index: u32,
        tool_calls: Option<Vec<dynamo_async_openai::types::ChatCompletionMessageToolCallChunk>>,
        finish_reason: Option<dynamo_async_openai::types::FinishReason>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        Annotated {
            data: Some(NvCreateChatCompletionStreamResponse {
                id: "chatcmpl-test".to_string(),
                choices: vec![dynamo_async_openai::types::ChatChoiceStream {
                    index,
                    delta: dynamo_async_openai::types::ChatCompletionStreamResponseDelta {
                        content: None,
                        function_call: None,
                        tool_calls,
                        role: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason,
                    stop_reason: None,
                    logprobs: None,
                }],
                created: 0,
                model: "test-model".to_string(),
                system_fingerprint: None,
                object: "chat.completion.chunk".to_string(),
                service_tier: None,
                usage: None,
                nvext: None,
            }),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    fn test_tool_call_chunk() -> dynamo_async_openai::types::ChatCompletionMessageToolCallChunk {
        dynamo_async_openai::types::ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call_1".to_string()),
            r#type: Some(dynamo_async_openai::types::ChatCompletionToolType::Function),
            function: Some(dynamo_async_openai::types::FunctionCallStream {
                name: Some("search".to_string()),
                arguments: Some(r#"{"query":"qwen"}"#.to_string()),
            }),
        }
    }

    #[test]
    fn test_normalize_stream_tool_call_finish_reason_rewrites_later_stop() {
        let mut seen_tool_calls = HashSet::new();

        let mut tool_chunk = chat_stream_chunk(0, Some(vec![test_tool_call_chunk()]), None);
        normalize_stream_tool_call_finish_reason(&mut tool_chunk, &mut seen_tool_calls);

        let mut final_chunk = chat_stream_chunk(
            0,
            None,
            Some(dynamo_async_openai::types::FinishReason::Stop),
        );
        normalize_stream_tool_call_finish_reason(&mut final_chunk, &mut seen_tool_calls);

        let choice = &final_chunk.data.as_ref().unwrap().choices[0];
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::FinishReason::ToolCalls)
        );
    }

    #[test]
    fn test_normalize_stream_tool_call_finish_reason_preserves_plain_stop() {
        let mut seen_tool_calls = HashSet::new();
        let mut final_chunk = chat_stream_chunk(
            0,
            None,
            Some(dynamo_async_openai::types::FinishReason::Stop),
        );

        normalize_stream_tool_call_finish_reason(&mut final_chunk, &mut seen_tool_calls);

        let choice = &final_chunk.data.as_ref().unwrap().choices[0];
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::FinishReason::Stop)
        );
    }

    #[test]
    fn test_validate_unsupported_fields_detects_flags() {
        #[allow(clippy::type_complexity)]
        let unsupported_cases: Vec<(&str, Box<dyn FnOnce(&mut CreateResponse)>)> = vec![
            ("background", Box::new(|r| r.background = Some(true))),
            (
                "previous_response_id",
                Box::new(|r| r.previous_response_id = Some("prev-id".into())),
            ),
            (
                "prompt",
                Box::new(|r| {
                    r.prompt = Some(PromptConfig {
                        id: "template-id".into(),
                        version: None,
                        variables: None,
                    })
                }),
            ),
            ("store", Box::new(|r| r.store = Some(true))),
        ];

        for (field, set_field) in unsupported_cases {
            let mut req = make_base_request();
            (set_field)(&mut req.inner);
            let result = validate_response_unsupported_fields(&req);
            assert!(result.is_some(), "Expected rejection for `{field}`");
        }
    }

    #[test]
    fn test_validate_chat_completion_required_fields_empty_messages() {
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![],
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_required_fields(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!(
                    "{VALIDATION_PREFIX}The 'messages' field cannot be empty. At least one message is required."
                )
            );
        }
    }

    #[test]
    fn test_validate_chat_completion_required_fields_with_messages() {
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_required_fields(&request);
        assert!(result.is_ok());
    }

    #[test]
    // Test for all Bad Requests Example for Chat Completion
    // 1. Echo:  Should be a boolean : Not Done
    // 2. Frequency Penalty: Should be a float between -2.0 and 2.0 : Done
    // 3. logprobs: Done
    // 4. Model Format: Should be a string : Not Done
    // 5. Prompt or Messages Validation
    // 6. Max Tokens: Should be a positive integer
    // 7. Presence Penalty: Should be a float between -2.0 and 2.0 : Done
    // 8. Stop : Should be a string or an array of strings : Not Done
    // 9. Invalid or Out of range temperature: Done
    // 10.Invalid or out of range top_p: Done
    // 11. Repetition Penalty: Should be a float between 0.0 and 2.0 : Done
    // 12. Logprobs: Should be a positive integer between 0 and 5 : Done
    // invalid or non existing user : Only empty string is not allowed validation is there. How can we check non-extisting user ?
    // Unknown fields : Done (rejected via extra_fields catch-all)
    // guided_whitespace_pattern null or invalid : Not Done
    // "response_format": { "type": "invalid_format" } : Not Done
    // "logit_bias": { "invalid_token": "not_a_number" }, : Partial Validation is already there
    fn test_bad_base_request_for_completion() {
        // Frequency Penalty: Should be a float between -2.0 and 2.0
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                frequency_penalty: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };

        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Frequency penalty must be between -2 and 2, got -3")
            );
        }

        // Presence Penalty: Should be a float between -2.0 and 2.0
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                presence_penalty: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Presence penalty must be between -2 and 2, got -3")
            );
        }

        // Temperature: Should be a float between 0.0 and 2.0
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                temperature: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Temperature must be between 0 and 2, got -3")
            );
        }

        // Top P: Should be a float between 0.0 and 1.0
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                top_p: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Top_p must be between 0 and 1, got -3")
            );
        }

        // Repetition Penalty: Should be a float between 0.0 and 2.0
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                ..Default::default()
            },
            common: CommonExt::builder()
                .repetition_penalty(-3.0)
                .build()
                .unwrap(),
            nvext: None,
            metadata: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Repetition penalty must be between 0 and 2, got -3")
            );
        }

        // Logprobs: Should be a positive integer between 0 and 5
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                logprobs: Some(6),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Logprobs must be between 0 and 5, got 6")
            );
        }
    }

    #[test]
    fn test_chat_completion_rejects_unknown_named_tool_choice() {
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                tools: Some(vec![dynamo_async_openai::types::ChatCompletionTool {
                    r#type: dynamo_async_openai::types::ChatCompletionToolType::Function,
                    function: dynamo_async_openai::types::FunctionObject {
                        name: "xxyyzz".to_string(),
                        description: Some("xxyyzz two numbers".to_string()),
                        parameters: Some(serde_json::json!({
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"}
                            },
                            "required": ["a", "b"]
                        })),
                        strict: None,
                    },
                }]),
                tool_choice: Some(ChatCompletionToolChoiceOption::Named(
                    dynamo_async_openai::types::ChatCompletionNamedToolChoice {
                        r#type: dynamo_async_openai::types::ChatCompletionToolType::Function,
                        function: dynamo_async_openai::types::FunctionName {
                            name: "zzyyxx".to_string(),
                        },
                    },
                )),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };

        let result = validate_chat_completion_fields_generic(&request);

        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert!(
                error_response
                    .1
                    .message
                    .contains("tool `zzyyxx` not found in the tools list")
            );
        }
    }

    #[test]
    fn test_metadata_field_nested() {
        use serde_json::json;

        // Test metadata field with nested object
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: json!({
                "user": {"id": 1, "name": "user-1"},
                "session": {"id": "session-1", "timestamp": 1640995200}
            })
            .into(),
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };

        let result = validate_completion_fields_generic(&request);
        assert!(result.is_ok());

        // Verify metadata is accessible
        assert!(request.metadata.is_some());
        assert_eq!(request.metadata.as_ref().unwrap()["user"]["id"], 1);
    }

    #[test]
    fn test_bad_base_request_for_chatcompletion() {
        // Frequency Penalty: Should be a float between -2.0 and 2.0
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                frequency_penalty: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };

        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Frequency penalty must be between -2 and 2, got -3")
            );
        }

        // Presence Penalty: Should be a float between -2.0 and 2.0
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                presence_penalty: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Presence penalty must be between -2 and 2, got -3")
            );
        }

        // Temperature: Should be a float between 0.0 and 2.0
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                temperature: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Temperature must be between 0 and 2, got -3")
            );
        }

        // Top P: Should be a float between 0.0 and 1.0
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                top_p: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Top_p must be between 0 and 1, got -3")
            );
        }

        // Repetition Penalty: Should be a float between 0.0 and 2.0
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                ..Default::default()
            },
            common: CommonExt::builder()
                .repetition_penalty(-3.0)
                .build()
                .unwrap(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Repetition penalty must be between 0 and 2, got -3")
            );
        }

        // Top Logprobs: Should be a positive integer between 0 and 20
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                top_logprobs: Some(25),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            structured_outputs: None,
            request_id: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Top_logprobs must be between 0 and 20, got 25")
            );
        }
    }

    #[test]
    fn test_chat_completions_unknown_fields_rejected() {
        // Test that known unsupported fields are rejected and all shown in error message
        let json = r#"{
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model",
            "add_special_tokens": true,
            "documents": ["doc1"],
            "chat_template": "custom"
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json).unwrap();

        // Verify all unsupported fields were captured
        assert!(
            request
                .unsupported_fields
                .contains_key("add_special_tokens")
        );
        assert!(request.unsupported_fields.contains_key("documents"));
        assert!(request.unsupported_fields.contains_key("chat_template"));

        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            let msg = &error_response.1.message;
            assert!(msg.contains("Unsupported parameter"));
            // Verify all fields appear in the error message
            assert!(msg.contains("add_special_tokens"));
            assert!(msg.contains("documents"));
            assert!(msg.contains("chat_template"));
        }
    }

    #[test]
    fn test_completions_unsupported_fields_rejected() {
        // Test that known unsupported fields are rejected and shown in error message.
        // Note: response_format is now a supported field and should NOT appear here.
        let json = r#"{
            "model": "test-model",
            "prompt": "Hello",
            "add_special_tokens": true
        }"#;

        let request: NvCreateCompletionRequest = serde_json::from_str(json).unwrap();

        // Verify unsupported field was captured
        assert!(
            request
                .unsupported_fields
                .contains_key("add_special_tokens")
        );
        // response_format is now supported — it should NOT be in unsupported_fields
        assert!(!request.unsupported_fields.contains_key("response_format"));

        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            let msg = &error_response.1.message;
            assert!(msg.contains("Unsupported parameter"));
            // Verify field appears in error message
            assert!(msg.contains("add_special_tokens"));
        }
    }

    #[test]
    fn test_completions_response_format_is_supported() {
        // Test that response_format is accepted as a supported field (not rejected)
        let json = r#"{
            "model": "test-model",
            "prompt": "Hello",
            "response_format": {"type": "json_object"}
        }"#;

        let request: NvCreateCompletionRequest = serde_json::from_str(json).unwrap();

        // response_format should be parsed into the inner field, not unsupported_fields
        assert!(!request.unsupported_fields.contains_key("response_format"));
        assert!(request.inner.response_format.is_some());

        // And validation should pass (no unsupported fields)
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_structured_chat_defaults_to_non_thinking_template() {
        let mut payload = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}
                }
            }
        });

        normalize_chat_compat_payload(&mut payload);
        let args = payload
            .get("chat_template_kwargs")
            .and_then(serde_json::Value::as_object)
            .expect("chat_template_kwargs");

        assert_eq!(
            args.get("enable_thinking"),
            Some(&serde_json::Value::Bool(false))
        );
        assert_eq!(args.get("thinking"), Some(&serde_json::Value::Bool(false)));
    }

    #[test]
    fn test_structured_chat_preserves_explicit_thinking_template() {
        let mut payload = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "chat_template_kwargs": {"enable_thinking": true},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}
                }
            }
        });

        normalize_chat_compat_payload(&mut payload);
        let args = payload
            .get("chat_template_kwargs")
            .and_then(serde_json::Value::as_object)
            .expect("chat_template_kwargs");

        assert_eq!(
            args.get("enable_thinking"),
            Some(&serde_json::Value::Bool(true))
        );
        assert!(args.get("thinking").is_none());
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_error_event() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use futures::stream;

        // Create an error event
        let error_event = Annotated::<NvCreateChatCompletionStreamResponse> {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: Some(vec!["Backend service unavailable".to_string()]),
            error: None,
        };

        let test_stream = stream::iter(vec![error_event]);
        let result = check_for_backend_error(test_stream).await;

        // Should return an error
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(error_response.1.message, "Backend service unavailable");
        }
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_json_error_and_code() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use futures::stream;

        // Create an error event with JSON payload containing error code in comment
        let error_json =
            r#"{"message":"prompt > max_seq_len","type":"Internal Server Error","code":500}"#;
        let error_event = Annotated::<NvCreateChatCompletionStreamResponse> {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: Some(vec![error_json.to_string()]),
            error: None,
        };

        let test_stream = stream::iter(vec![error_event]);
        let result = check_for_backend_error(test_stream).await;

        // Should return an error with correct status code extracted from JSON
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(error_response.1.message, "prompt > max_seq_len");
            assert_eq!(error_response.1.code, 500);
        }
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_normal_event() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use dynamo_async_openai::types::CreateChatCompletionStreamResponse;
        use futures::stream::{self, StreamExt};

        // Create a normal data event
        let normal_event = Annotated::<NvCreateChatCompletionStreamResponse> {
            data: Some(CreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![],
                created: 0,
                model: "test-model".to_string(),
                system_fingerprint: None,
                object: "chat.completion.chunk".to_string(),
                service_tier: None,
                usage: None,
                nvext: None,
            }),
            id: Some("msg-1".to_string()),
            event: None,
            comment: None,
            error: None,
        };

        let test_stream = stream::iter(vec![normal_event.clone()]);
        let result = check_for_backend_error(test_stream).await;

        // Should return Ok with the stream
        assert!(result.is_ok());
        let mut returned_stream = result.unwrap();

        // Verify we can read the event back from the stream
        let first = returned_stream.next().await;
        assert!(first.is_some());
        let first_event = first.unwrap();
        assert_eq!(first_event.id, Some("msg-1".to_string()));
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_empty_stream() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use futures::stream::{self, StreamExt};

        // Create an empty stream
        let test_stream =
            stream::iter::<Vec<Annotated<NvCreateChatCompletionStreamResponse>>>(vec![]);
        let result = check_for_backend_error(test_stream).await;

        // Should return Ok with an empty stream
        assert!(result.is_ok());
        let mut returned_stream = result.unwrap();

        // Verify stream is empty
        let first = returned_stream.next().await;
        assert!(first.is_none());
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_comment_but_no_event_type() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use futures::stream;

        // Create an event with comment but no event type and no data (error indicator)
        let error_event = Annotated::<NvCreateChatCompletionStreamResponse> {
            data: None,
            id: None,
            event: None,
            comment: Some(vec!["Connection timeout".to_string()]),
            error: None,
        };

        let test_stream = stream::iter(vec![error_event]);
        let result = check_for_backend_error(test_stream).await;

        // Should return an error based on is_backend_error_event logic
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(error_response.1.message, "Connection timeout");
        }
    }

    #[test]
    fn test_classify_error_for_metrics_validation() {
        // 400 with "Validation:" prefix to validation
        let error_type =
            classify_error_for_metrics(StatusCode::BAD_REQUEST, "Validation: Invalid parameter");
        assert_eq!(error_type, ErrorType::Validation);

        // 400 WITHOUT "Validation:" to internal (fallback)
        let error_type = classify_error_for_metrics(StatusCode::BAD_REQUEST, "Some other error");
        assert_eq!(error_type, ErrorType::Internal);
    }

    #[test]
    fn test_classify_error_for_metrics_status_codes() {
        assert_eq!(
            classify_error_for_metrics(StatusCode::NOT_FOUND, "Model not found"),
            ErrorType::NotFound
        );
        assert_eq!(
            classify_error_for_metrics(StatusCode::NOT_IMPLEMENTED, "Feature not supported"),
            ErrorType::NotImplemented
        );
        assert_eq!(
            classify_error_for_metrics(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded"),
            ErrorType::Overload
        );
        assert_eq!(
            classify_error_for_metrics(StatusCode::SERVICE_UNAVAILABLE, "Overloaded"),
            ErrorType::Overload
        );
        assert_eq!(
            classify_error_for_metrics(StatusCode::INTERNAL_SERVER_ERROR, "Panic"),
            ErrorType::Internal
        );
    }

    #[test]
    fn test_classify_error_for_metrics_client_errors() {
        // Other 4xx errors should be classified as validation
        assert_eq!(
            classify_error_for_metrics(StatusCode::UNAUTHORIZED, "Unauthorized"),
            ErrorType::Validation
        );
        assert_eq!(
            classify_error_for_metrics(StatusCode::FORBIDDEN, "Forbidden"),
            ErrorType::Validation
        );
    }

    #[test]
    fn test_extract_error_type_from_response_validation() {
        let response = ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: "Validation: bad input".to_string(),
        });
        assert_eq!(
            extract_error_type_from_response(&response),
            ErrorType::Validation
        );
    }

    #[test]
    fn test_extract_error_type_from_response_not_found() {
        let response = ErrorMessage::model_not_found();
        assert_eq!(
            extract_error_type_from_response(&response),
            ErrorType::NotFound
        );
    }

    #[test]
    fn test_extract_error_type_from_response_internal() {
        let response = ErrorMessage::internal_server_error("Something went wrong");
        assert_eq!(
            extract_error_type_from_response(&response),
            ErrorType::Internal
        );
    }

    #[test]
    fn test_extract_error_type_from_response_not_implemented() {
        let response = ErrorMessage::not_implemented_error("Feature not available");
        assert_eq!(
            extract_error_type_from_response(&response),
            ErrorType::NotImplemented
        );
    }
}
