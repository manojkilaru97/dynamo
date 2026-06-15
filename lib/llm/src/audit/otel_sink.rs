// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OTLP/HTTP audit sink for OpenAI payload logs.
//!
//! This emits protobuf OTLP directly instead of the Rust logging SDK because
//! the payload tests require structured attributes and nested null values.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Mutex,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use async_trait::async_trait;
use dynamo_runtime::config::environment_names::logging::otlp as env_otlp;
use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
use opentelemetry_proto::tonic::common::v1::{
    AnyValue, ArrayValue, InstrumentationScope, KeyValue, KeyValueList, any_value,
};
use opentelemetry_proto::tonic::logs::v1::{LogRecord, ResourceLogs, ScopeLogs, SeverityNumber};
use opentelemetry_proto::tonic::resource::v1::Resource;
use prost014::Message;
use serde_json::{Value, json};

use super::config::AuditPolicy;
use super::handle::{AuditEventType, AuditRecord};
use super::sink::AuditSink;

const DEFAULT_OTLP_HTTP_LOGS_ENDPOINT: &str = "http://localhost:4318/v1/logs";
const DEFAULT_OTLP_HTTP_TIMEOUT_MS: u64 = 5_000;
const DEFAULT_SERVICE_NAME: &str = "dynamo";
const AUDIT_ENDPOINT_CHAT_COMPLETION: &str = "openai.chat_completion";
const AUDIT_SCOPE: &str = "dynamo.payload";
const SUPPRESS_PAYLOAD_NCA_IDS_ENV: &str = "VLLM_SUPPRESS_PAYLOAD_NCA_IDS";
const PAYLOAD_SUPPRESSION_REASON_NCA_ID: &str = "nca_id";
const NCA_HEADER_NAMES: [&str; 2] = ["nvcf-ncaid", "nvcf-nca-id"];
const MAX_TRACKED_SUPPRESSED_REQUESTS: usize = 32768;

pub struct OtelSink {
    client: reqwest::Client,
    endpoint: String,
    service_name: String,
    max_payload_bytes: usize,
    suppressed_nca_ids: HashSet<String>,
    suppressed_requests: Mutex<SuppressedRequestTracker>,
}

#[derive(Default)]
struct SuppressedRequestTracker {
    by_request_id: HashMap<String, String>,
    order: VecDeque<String>,
}

impl SuppressedRequestTracker {
    fn insert(&mut self, request_id: String, nca_id: String) {
        if self
            .by_request_id
            .insert(request_id.clone(), nca_id)
            .is_none()
        {
            self.order.push_back(request_id);
        }
        while self.by_request_id.len() > MAX_TRACKED_SUPPRESSED_REQUESTS {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.by_request_id.remove(&oldest);
        }
    }

    fn get(&self, request_id: &str) -> Option<String> {
        self.by_request_id.get(request_id).cloned()
    }
}

impl OtelSink {
    pub async fn from_policy(policy: &AuditPolicy) -> Result<Self> {
        let protocol = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_LOGS_PROTOCOL)
            .or_else(|_| std::env::var(env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL))
            .unwrap_or_else(|_| "http/protobuf".to_string());
        if !matches!(
            protocol.trim().to_ascii_lowercase().as_str(),
            "http" | "http/protobuf" | "http/proto"
        ) {
            tracing::warn!(
                protocol,
                "audit otel: only OTLP/HTTP protobuf is supported by the direct payload sink"
            );
        }

        let endpoint = resolve_logs_endpoint();
        let service_name = std::env::var(env_otlp::OTEL_SERVICE_NAME)
            .unwrap_or_else(|_| DEFAULT_SERVICE_NAME.to_string());
        let timeout = resolve_export_timeout();

        Ok(Self {
            client: reqwest::Client::builder().timeout(timeout).build()?,
            endpoint,
            service_name,
            max_payload_bytes: policy.otel_max_payload_bytes,
            suppressed_nca_ids: suppressed_nca_ids_from_env(),
            suppressed_requests: Mutex::new(SuppressedRequestTracker::default()),
        })
    }

    fn payload_value(&self, rec: &AuditRecord) -> Value {
        let suppressed_nca_id = self.suppressed_nca_id_for_record(rec);
        let payload = match rec.event_type {
            AuditEventType::Request => rec
                .raw_request
                .as_ref()
                .map(|payload| payload.as_ref().clone())
                .or_else(|| {
                    rec.request
                        .as_ref()
                        .and_then(|request| serde_json::to_value(request.as_ref()).ok())
                }),
            AuditEventType::Response => rec
                .raw_response
                .as_ref()
                .map(|payload| payload.as_ref().clone())
                .or_else(|| {
                    rec.response
                        .as_ref()
                        .and_then(|response| serde_json::to_value(response.as_ref()).ok())
                }),
        };

        let Some(mut payload) = payload else {
            if let Some(nca_id) = suppressed_nca_id {
                return base_suppressed_payload(&nca_id);
            }
            return json!({
                "audit_complete": false,
                "audit_drop_reason": "payload_serialize_failed",
            });
        };
        if let Some(nca_id) = suppressed_nca_id {
            return match rec.event_type {
                AuditEventType::Request => build_suppressed_request_payload(&payload, &nca_id),
                AuditEventType::Response => build_suppressed_response_payload(&payload, &nca_id),
            };
        }
        redact_multimodal_payload(&mut payload);

        match serde_json::to_vec(&payload) {
            Ok(bytes) if bytes.len() <= self.max_payload_bytes => payload,
            Ok(bytes) => json!({
                "audit_complete": false,
                "audit_drop_reason": format!(
                    "otel_payload_too_large:max_bytes={}:actual_bytes={}",
                    self.max_payload_bytes,
                    bytes.len()
                ),
            }),
            Err(_) => json!({
                "audit_complete": false,
                "audit_drop_reason": "payload_size_check_failed",
            }),
        }
    }

    fn suppressed_nca_id_for_record(&self, rec: &AuditRecord) -> Option<String> {
        match rec.event_type {
            AuditEventType::Request => {
                let nca_id = self.suppressed_nca_id(rec.headers.as_deref())?;
                self.track_suppressed_request(&rec.request_id, &nca_id);
                Some(nca_id)
            }
            AuditEventType::Response => self
                .suppressed_nca_id(rec.headers.as_deref())
                .or_else(|| self.tracked_suppressed_request(&rec.request_id)),
        }
    }

    fn suppressed_nca_id(&self, headers: Option<&Value>) -> Option<String> {
        if self.suppressed_nca_ids.is_empty() {
            return None;
        }
        let nca_id = resolve_nca_id(headers)?;
        if self.suppressed_nca_ids.contains(&nca_id) {
            Some(nca_id)
        } else {
            None
        }
    }

    fn track_suppressed_request(&self, request_id: &str, nca_id: &str) {
        if request_id.trim().is_empty() {
            return;
        }
        if let Ok(mut tracker) = self.suppressed_requests.lock() {
            tracker.insert(request_id.to_string(), nca_id.to_string());
        }
    }

    fn tracked_suppressed_request(&self, request_id: &str) -> Option<String> {
        if request_id.trim().is_empty() {
            return None;
        }
        self.suppressed_requests
            .lock()
            .ok()
            .and_then(|tracker| tracker.get(request_id))
    }

    fn build_request(&self, rec: &AuditRecord) -> ExportLogsServiceRequest {
        let event_type = match rec.event_type {
            AuditEventType::Request => "request",
            AuditEventType::Response => "response",
        };
        let body = match rec.event_type {
            AuditEventType::Request => "openai.request",
            AuditEventType::Response => "openai.response",
        };

        let payload = self.payload_value(rec);
        let payload_suppressed = payload
            .get("payload_suppressed")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let mut attrs = vec![
            kv("rid", string_value(rec.request_id.clone())),
            kv("request_id", string_value(rec.request_id.clone())),
            kv("event_type", string_value(event_type)),
            kv("endpoint", string_value(AUDIT_ENDPOINT_CHAT_COMPLETION)),
            kv("model", string_value(rec.model.clone())),
            kv("streaming", bool_value(rec.requested_streaming)),
            kv("audit_complete", bool_value(true)),
            kv("payload", json_to_any(payload.clone())),
        ];
        if payload_suppressed {
            attrs.push(kv("payload_suppressed", bool_value(true)));
            attrs.push(kv(
                "suppression_reason",
                string_value(PAYLOAD_SUPPRESSION_REASON_NCA_ID),
            ));
            if let Some(nca_id) = payload.get("nca_id").and_then(Value::as_str) {
                attrs.push(kv("nca_id", string_value(nca_id.to_string())));
            }
        }
        if let Some(headers) = &rec.headers {
            attrs.push(kv("headers", json_to_any(headers.as_ref().clone())));
        }
        if rec.event_type == AuditEventType::Request {
            add_request_shape_attrs(&mut attrs, &payload);
        }

        let now = unix_nanos();
        ExportLogsServiceRequest {
            resource_logs: vec![ResourceLogs {
                resource: Some(Resource {
                    attributes: vec![kv("service.name", string_value(self.service_name.clone()))],
                    dropped_attributes_count: 0,
                    entity_refs: Vec::new(),
                }),
                scope_logs: vec![ScopeLogs {
                    scope: Some(InstrumentationScope {
                        name: AUDIT_SCOPE.to_string(),
                        version: String::new(),
                        attributes: Vec::new(),
                        dropped_attributes_count: 0,
                    }),
                    log_records: vec![LogRecord {
                        time_unix_nano: now,
                        observed_time_unix_nano: now,
                        severity_number: SeverityNumber::Info as i32,
                        severity_text: "INFO".to_string(),
                        body: Some(string_value(body)),
                        attributes: attrs,
                        dropped_attributes_count: 0,
                        flags: 0,
                        trace_id: Vec::new(),
                        span_id: Vec::new(),
                        event_name: String::new(),
                    }],
                    schema_url: String::new(),
                }],
                schema_url: String::new(),
            }],
        }
    }
}

fn resolve_logs_endpoint() -> String {
    if let Ok(endpoint) = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_LOGS_ENDPOINT)
        && !endpoint.trim().is_empty()
    {
        return endpoint;
    }

    if let Ok(endpoint) = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_ENDPOINT)
        && !endpoint.trim().is_empty()
    {
        return append_signal_endpoint(&endpoint, "logs");
    }

    if let Ok(endpoint) = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_TRACES_ENDPOINT)
        && !endpoint.trim().is_empty()
    {
        return endpoint
            .strip_suffix("/v1/traces")
            .map(|base| format!("{base}/v1/logs"))
            .unwrap_or(endpoint);
    }

    DEFAULT_OTLP_HTTP_LOGS_ENDPOINT.to_string()
}

fn append_signal_endpoint(endpoint: &str, signal: &str) -> String {
    let endpoint = endpoint.trim_end_matches('/');
    if endpoint.ends_with("/v1/traces") {
        endpoint
            .strip_suffix("/v1/traces")
            .map(|base| format!("{base}/v1/{signal}"))
            .unwrap_or_else(|| endpoint.to_string())
    } else if endpoint.ends_with("/v1/logs") || endpoint.ends_with("/v1/metrics") {
        endpoint.to_string()
    } else {
        format!("{endpoint}/v1/{signal}")
    }
}

fn resolve_export_timeout() -> Duration {
    std::env::var(env_otlp::OTEL_EXPORTER_OTLP_LOGS_TIMEOUT)
        .or_else(|_| std::env::var(env_otlp::OTEL_EXPORTER_OTLP_TIMEOUT))
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|millis| *millis > 0)
        .map(Duration::from_millis)
        .unwrap_or_else(|| Duration::from_millis(DEFAULT_OTLP_HTTP_TIMEOUT_MS))
}

#[async_trait]
impl AuditSink for OtelSink {
    fn name(&self) -> &'static str {
        "otel"
    }

    async fn emit(&self, rec: &AuditRecord) {
        let request = self.build_request(rec);
        let mut body = Vec::with_capacity(request.encoded_len());
        if let Err(err) = request.encode(&mut body) {
            tracing::warn!(target: "dynamo_llm::audit", "audit otel: encode failed: {err}");
            return;
        }

        let result = self
            .client
            .post(&self.endpoint)
            .header(reqwest::header::CONTENT_TYPE, "application/x-protobuf")
            .body(body)
            .send()
            .await
            .and_then(|response| response.error_for_status());
        if let Err(err) = result {
            tracing::warn!(target: "dynamo_llm::audit", "audit otel: export failed: {err}");
        }
    }
}

fn unix_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64
}

fn kv(key: impl Into<String>, value: AnyValue) -> KeyValue {
    KeyValue {
        key: key.into(),
        value: Some(value),
    }
}

fn string_value(value: impl Into<String>) -> AnyValue {
    AnyValue {
        value: Some(any_value::Value::StringValue(value.into())),
    }
}

fn bool_value(value: bool) -> AnyValue {
    AnyValue {
        value: Some(any_value::Value::BoolValue(value)),
    }
}

fn int_value(value: usize) -> AnyValue {
    AnyValue {
        value: Some(any_value::Value::IntValue(
            i64::try_from(value).unwrap_or(i64::MAX),
        )),
    }
}

fn add_request_shape_attrs(attrs: &mut Vec<KeyValue>, payload: &Value) {
    let (image_count, video_count, audio_count) = count_modalities(payload);
    let tool_count = payload
        .get("tools")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let tool_choice = normalize_tool_choice(payload.get("tool_choice"));
    let structured_output_kind = structured_output_kind(payload);

    attrs.push(kv("input_image_count", int_value(image_count)));
    attrs.push(kv("input_video_count", int_value(video_count)));
    attrs.push(kv("input_audio_count", int_value(audio_count)));
    attrs.push(kv("input_tool_count", int_value(tool_count)));
    attrs.push(kv("has_images", bool_value(image_count > 0)));
    attrs.push(kv("has_videos", bool_value(video_count > 0)));
    attrs.push(kv("has_audios", bool_value(audio_count > 0)));
    attrs.push(kv("has_tools", bool_value(tool_count > 0)));
    attrs.push(kv(
        "has_tool_calls_enabled",
        bool_value(tool_count > 0 && tool_choice.as_deref() != Some("none")),
    ));
    attrs.push(kv(
        "has_structured_output",
        bool_value(structured_output_kind.is_some()),
    ));
    if let Some(tool_choice) = tool_choice {
        attrs.push(kv("tool_choice", string_value(tool_choice)));
    }
    if let Some(kind) = structured_output_kind {
        attrs.push(kv("structured_output_kind", string_value(kind)));
    }
}

fn normalize_tool_choice(value: Option<&Value>) -> Option<String> {
    match value {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.clone()),
        Some(Value::Object(object)) => {
            if object
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .is_some()
            {
                Some("named".to_string())
            } else {
                object
                    .get("type")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| Some("named".to_string()))
            }
        }
        Some(value) => Some(value.to_string()),
    }
}

fn structured_output_kind(payload: &Value) -> Option<String> {
    if let Some(response_format) = payload.get("response_format").and_then(Value::as_object)
        && let Some(format_type) = response_format.get("type").and_then(Value::as_str)
        && matches!(
            format_type,
            "json_schema" | "json_object" | "structural_tag"
        )
    {
        return Some(format_type.to_string());
    }

    if let Some(structured_outputs) = payload.get("structured_outputs") {
        let Some(object) = structured_outputs.as_object() else {
            return Some("structured_outputs".to_string());
        };
        if object.is_empty() {
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
            if object.get(key).is_some_and(|value| !value.is_null()) {
                return Some(if key == "json" { "json_schema" } else { key }.to_string());
            }
        }
        return Some("structured_outputs".to_string());
    }

    payload
        .get("text")
        .and_then(Value::as_object)
        .and_then(|text| text.get("format"))
        .and_then(Value::as_object)
        .and_then(|format| format.get("type"))
        .and_then(Value::as_str)
        .filter(|format_type| matches!(*format_type, "json_schema" | "json_object"))
        .map(str::to_string)
}

fn count_modalities(payload: &Value) -> (usize, usize, usize) {
    let mut image_count = 0;
    let mut video_count = 0;
    let mut audio_count = 0;

    if let Some(messages) = payload.get("messages").and_then(Value::as_array) {
        for message in messages {
            if let Some(content) = message.get("content").and_then(Value::as_array) {
                for part in content {
                    count_content_part(part, &mut image_count, &mut video_count, &mut audio_count);
                }
            }
        }
    }

    if let Some(input_items) = payload.get("input").and_then(Value::as_array) {
        for item in input_items {
            count_content_part(item, &mut image_count, &mut video_count, &mut audio_count);
            if let Some(content) = item.get("content").and_then(Value::as_array) {
                for part in content {
                    count_content_part(part, &mut image_count, &mut video_count, &mut audio_count);
                }
            }
        }
    }

    (image_count, video_count, audio_count)
}

fn count_content_part(
    part: &Value,
    image_count: &mut usize,
    video_count: &mut usize,
    audio_count: &mut usize,
) {
    match part.get("type").and_then(Value::as_str) {
        Some("input_image" | "image_url") => *image_count += 1,
        Some("video_url") => *video_count += 1,
        Some("input_audio" | "audio_url") => *audio_count += 1,
        _ => {}
    }
}

fn suppressed_nca_ids_from_env() -> HashSet<String> {
    std::env::var(SUPPRESS_PAYLOAD_NCA_IDS_ENV)
        .unwrap_or_default()
        .split(',')
        .filter_map(|part| normalize_nca_id(part).filter(|nca_id| !nca_id.is_empty()))
        .collect()
}

fn resolve_nca_id(headers: Option<&Value>) -> Option<String> {
    let object = headers?.as_object()?;
    for header_name in NCA_HEADER_NAMES {
        if let Some(value) = get_ci_object_value(object, header_name)
            && let Some(nca_id) = normalize_nca_id_from_value(value)
            && !nca_id.is_empty()
        {
            return Some(nca_id);
        }
    }
    None
}

fn get_ci_object_value<'a>(
    object: &'a serde_json::Map<String, Value>,
    needle: &str,
) -> Option<&'a Value> {
    object.get(needle).or_else(|| {
        object
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(needle))
            .map(|(_, value)| value)
    })
}

fn normalize_nca_id_from_value(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => normalize_nca_id(value),
        Value::Array(values) => values.iter().find_map(normalize_nca_id_from_value),
        value => normalize_nca_id(value.to_string()),
    }
}

fn normalize_nca_id(value: impl AsRef<str>) -> Option<String> {
    let value = value.as_ref().trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn base_suppressed_payload(nca_id: &str) -> Value {
    json!({
        "payload_suppressed": true,
        "suppression_reason": PAYLOAD_SUPPRESSION_REASON_NCA_ID,
        "nca_id": nca_id,
    })
}

fn build_suppressed_request_payload(payload: &Value, nca_id: &str) -> Value {
    const SAFE_REQUEST_FIELDS: [&str; 26] = [
        "best_of",
        "echo",
        "frequency_penalty",
        "ignore_eos",
        "include_reasoning",
        "max_completion_tokens",
        "max_output_tokens",
        "max_tokens",
        "min_p",
        "min_tokens",
        "model",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning_budget",
        "reasoning_effort",
        "repetition_penalty",
        "seed",
        "service_tier",
        "stream",
        "temperature",
        "top_k",
        "top_logprobs",
        "top_p",
        "truncate_prompt_tokens",
        "use_beam_search",
    ];

    let mut out = base_suppressed_payload(nca_id);
    let Some(out_object) = out.as_object_mut() else {
        return out;
    };
    let Some(payload_object) = payload.as_object() else {
        return out;
    };
    for key in SAFE_REQUEST_FIELDS {
        if let Some(value) = payload_object.get(key) {
            out_object.insert(key.to_string(), safe_value(value));
        }
    }
    if let Some(reasoning_effort) = payload_object
        .get("reasoning")
        .and_then(Value::as_object)
        .and_then(|reasoning| reasoning.get("effort"))
    {
        out_object.insert("reasoning_effort".to_string(), safe_value(reasoning_effort));
    }
    out
}

fn build_suppressed_response_payload(payload: &Value, nca_id: &str) -> Value {
    let mut out = base_suppressed_payload(nca_id);
    let Some(out_object) = out.as_object_mut() else {
        return out;
    };
    let Some(payload_object) = payload.as_object() else {
        return out;
    };

    for key in [
        "created", "id", "model", "object", "status", "stream", "usage",
    ] {
        if let Some(value) = payload_object.get(key) {
            out_object.insert(key.to_string(), safe_value(value));
        }
    }
    if let Some(choices) = payload_object.get("choices").and_then(Value::as_array) {
        let finish_reasons: Vec<Value> = choices
            .iter()
            .filter_map(|choice| choice.get("finish_reason"))
            .filter(|value| !value.is_null())
            .map(safe_value)
            .collect();
        if let Some(finish_reason) = finish_reasons.first() {
            out_object.insert("finish_reason".to_string(), finish_reason.clone());
        }
        if !finish_reasons.is_empty() {
            out_object.insert("finish_reasons".to_string(), Value::Array(finish_reasons));
        }
    }
    if let Some(error) = payload_object.get("error").and_then(Value::as_object) {
        let mut safe_error = serde_json::Map::new();
        for key in ["type", "code"] {
            if let Some(value) = error.get(key) {
                safe_error.insert(key.to_string(), safe_value(value));
            }
        }
        if !safe_error.is_empty() {
            out_object.insert("error".to_string(), Value::Object(safe_error));
        }
    }
    out
}

fn safe_value(value: &Value) -> Value {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => value.clone(),
        Value::Array(values) => Value::Array(values.iter().map(safe_value).collect()),
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), safe_value(value)))
                .collect(),
        ),
    }
}

fn redact_multimodal_payload(value: &mut Value) {
    match value {
        Value::String(text) => redact_multimodal_string(text),
        Value::Array(values) => {
            for value in values {
                redact_multimodal_payload(value);
            }
        }
        Value::Object(object) => {
            match object.get("type").and_then(Value::as_str) {
                Some("image_url") => redact_media_field(object, "image_url", "url"),
                Some("video_url") => redact_media_field(object, "video_url", "url"),
                Some("audio_url") => redact_media_field(object, "audio_url", "url"),
                Some("input_audio") => redact_media_field(object, "input_audio", "data"),
                Some("input_image") => redact_media_field(object, "image_url", "url"),
                _ => {}
            }
            for value in object.values_mut() {
                redact_multimodal_payload(value);
            }
        }
        _ => {}
    }
}

fn redact_media_field(
    object: &mut serde_json::Map<String, Value>,
    field_name: &str,
    scalar_name: &str,
) {
    let Some(field) = object.get_mut(field_name) else {
        return;
    };
    match field {
        Value::String(value) => *value = "[redacted-mm-input]".to_string(),
        Value::Object(media) => {
            if media.contains_key(scalar_name) {
                media.insert(
                    scalar_name.to_string(),
                    Value::String("[redacted-mm-input]".to_string()),
                );
            }
        }
        _ => {}
    }
}

fn redact_multimodal_string(text: &mut String) {
    if is_inline_media_uri(text) {
        *text = "[redacted-mm-input]".to_string();
        return;
    }

    for tag in ["img", "image", "video", "audio"] {
        let mut search_from = 0;
        loop {
            let Some(rel_tag_start) = text[search_from..].find(&format!("<{tag}")) else {
                break;
            };
            let tag_start = search_from + rel_tag_start;
            let Some(rel_tag_end) = text[tag_start..].find('>') else {
                break;
            };
            let tag_end = tag_start + rel_tag_end;
            let tag_text = &text[tag_start..=tag_end];
            let Some(rel_src) = tag_text.find("src=\"") else {
                search_from = tag_end + 1;
                continue;
            };
            let src_start = tag_start + rel_src + "src=\"".len();
            let Some(rel_src_end) = text[src_start..=tag_end].find('"') else {
                break;
            };
            let src_end = src_start + rel_src_end;
            if is_inline_media_uri(&text[src_start..src_end]) {
                text.replace_range(src_start..src_end, "[redacted-mm-input]");
                search_from = src_start + "[redacted-mm-input]".len();
            } else {
                search_from = tag_end + 1;
            }
        }
    }
}

fn is_inline_media_uri(text: &str) -> bool {
    let text = text.trim_start().to_ascii_lowercase();
    text.starts_with("data:") && text.contains(";base64,")
}

fn json_to_any(value: Value) -> AnyValue {
    match value {
        Value::Null => AnyValue { value: None },
        Value::Bool(value) => AnyValue {
            value: Some(any_value::Value::BoolValue(value)),
        },
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                AnyValue {
                    value: Some(any_value::Value::IntValue(value)),
                }
            } else if let Some(value) = value.as_u64().and_then(|v| i64::try_from(v).ok()) {
                AnyValue {
                    value: Some(any_value::Value::IntValue(value)),
                }
            } else {
                AnyValue {
                    value: Some(any_value::Value::DoubleValue(
                        value.as_f64().unwrap_or_default(),
                    )),
                }
            }
        }
        Value::String(value) => string_value(value),
        Value::Array(values) => AnyValue {
            value: Some(any_value::Value::ArrayValue(ArrayValue {
                values: values.into_iter().map(json_to_any).collect(),
            })),
        },
        Value::Object(values) => AnyValue {
            value: Some(any_value::Value::KvlistValue(KeyValueList {
                values: values
                    .into_iter()
                    .map(|(key, value)| kv(key, json_to_any(value)))
                    .collect(),
            })),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionResponse;
    use std::{
        collections::HashSet,
        sync::{Arc, Mutex},
    };

    fn test_sink_with_suppressed_nca(nca_id: &str) -> OtelSink {
        OtelSink {
            client: reqwest::Client::new(),
            endpoint: "http://127.0.0.1:4318/v1/logs".to_string(),
            service_name: "test".to_string(),
            max_payload_bytes: 1024 * 1024,
            suppressed_nca_ids: HashSet::from([nca_id.to_string()]),
            suppressed_requests: Mutex::new(SuppressedRequestTracker::default()),
        }
    }

    fn test_response(content: &str) -> NvCreateChatCompletionResponse {
        serde_json::from_value(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }))
        .unwrap()
    }

    #[test]
    fn response_headers_are_sufficient_for_nca_payload_suppression() {
        let nca_id = "suppressed-nca";
        let sink = test_sink_with_suppressed_nca(nca_id);
        let rec = AuditRecord {
            schema_version: 1,
            event_type: AuditEventType::Response,
            request_id: "rid-1".to_string(),
            requested_streaming: false,
            model: "test-model".to_string(),
            request: None,
            raw_request: None,
            response: Some(Arc::new(test_response("SECRET_RESPONSE_MARKER"))),
            raw_response: None,
            headers: Some(Arc::new(json!({"nvcf-ncaid": nca_id}))),
        };

        let payload = sink.payload_value(&rec);

        assert_eq!(payload["payload_suppressed"], true);
        assert_eq!(payload["nca_id"], nca_id);
        assert_eq!(payload.get("choices"), None);
        assert!(!payload.to_string().contains("SECRET_RESPONSE_MARKER"));
    }
}
