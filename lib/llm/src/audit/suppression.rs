// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use serde_json::{Value, json};

use super::handle::{AuditEventType, AuditRecord};

const NCA_HEADER_NAMES: [&str; 2] = ["nvcf-ncaid", "nvcf-nca-id"];

pub(super) fn normalize_nca_id(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let unwrapped = value
        .strip_prefix("nca-")
        .and_then(|value| value.strip_suffix("-nca"))
        .unwrap_or(value);
    (!unwrapped.is_empty()).then(|| unwrapped.to_string())
}

fn resolve_nca_id(headers: Option<&Value>) -> Option<String> {
    let headers = headers?.as_object()?;
    for name in NCA_HEADER_NAMES {
        let value = headers.get(name).or_else(|| {
            headers
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, value)| value)
        });
        let Some(value) = value else {
            continue;
        };
        let value = match value {
            Value::String(value) => value.as_str(),
            Value::Array(values) => {
                let Some(value) = values.iter().find_map(Value::as_str) else {
                    continue;
                };
                value
            }
            _ => continue,
        };
        if let Some(nca_id) = normalize_nca_id(value) {
            return Some(nca_id);
        }
    }
    None
}

pub(super) fn suppress_record(rec: &mut AuditRecord, suppressed_nca_ids: &HashSet<String>) {
    if suppressed_nca_ids.is_empty() {
        return;
    }
    let Some(nca_id) = resolve_nca_id(rec.headers.as_deref()) else {
        return;
    };
    if !suppressed_nca_ids.contains(&nca_id) {
        return;
    }

    let marker = Arc::new(json!({
        "payload_suppressed": true,
        "suppression_reason": "nca_id",
        "nca_id": nca_id,
    }));
    rec.request = None;
    rec.raw_request = None;
    rec.response = None;
    rec.raw_response = None;
    match rec.event_type {
        AuditEventType::Request => rec.raw_request = Some(marker),
        AuditEventType::Response => rec.raw_response = Some(marker),
    }
    rec.headers = Some(Arc::new(json!({"nvcf-ncaid": nca_id})));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
    };

    fn record(event_type: AuditEventType, nca_id: &str) -> AuditRecord {
        AuditRecord {
            schema_version: 1,
            event_type,
            request_id: "req-1".to_string(),
            requested_streaming: false,
            model: "test-model".to_string(),
            request: Some(Arc::new(
                serde_json::from_value::<NvCreateChatCompletionRequest>(json!({
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "SECRET_REQUEST"}],
                }))
                .unwrap(),
            )),
            raw_request: Some(Arc::new(json!({"messages": ["SECRET_RAW_REQUEST"]}))),
            response: Some(Arc::new(
                serde_json::from_value::<NvCreateChatCompletionResponse>(json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "SECRET_RESPONSE"},
                        "finish_reason": "stop"
                    }]
                }))
                .unwrap(),
            )),
            raw_response: Some(Arc::new(json!({"output": "SECRET_RAW_RESPONSE"}))),
            headers: Some(Arc::new(json!({
                "authorization": "Bearer SECRET_TOKEN",
                "nvcf-ncaid": nca_id,
                "x-user-header": "SECRET_HEADER",
            }))),
        }
    }

    #[test]
    fn suppressed_nca_removes_every_payload_and_non_nca_header() {
        for event_type in [AuditEventType::Request, AuditEventType::Response] {
            let mut rec = record(event_type, "nca-customer-1-nca");
            suppress_record(&mut rec, &HashSet::from(["customer-1".to_string()]));

            let serialized = serde_json::to_string(&rec).unwrap();
            assert!(serialized.contains("payload_suppressed"));
            assert!(serialized.contains("customer-1"));
            for forbidden in [
                "SECRET_REQUEST",
                "SECRET_RAW_REQUEST",
                "SECRET_RESPONSE",
                "SECRET_RAW_RESPONSE",
                "SECRET_TOKEN",
                "SECRET_HEADER",
            ] {
                assert!(!serialized.contains(forbidden), "leaked {forbidden}");
            }
        }
    }

    #[test]
    fn unsuppressed_nca_preserves_payload_and_all_headers() {
        let mut rec = record(AuditEventType::Request, "control-customer");
        suppress_record(
            &mut rec,
            &HashSet::from(["suppressed-customer".to_string()]),
        );

        let serialized = serde_json::to_string(&rec).unwrap();
        assert!(serialized.contains("SECRET_RAW_REQUEST"));
        assert!(serialized.contains("SECRET_TOKEN"));
        assert!(serialized.contains("SECRET_HEADER"));
        assert!(!serialized.contains("payload_suppressed"));
    }
}
