// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{bus, config};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AuditEventType {
    Request,
    Response,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AuditRecord {
    pub schema_version: u32,
    pub event_type: AuditEventType,
    pub request_id: String,
    pub requested_streaming: bool,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<Arc<NvCreateChatCompletionRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_request: Option<Arc<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Arc<NvCreateChatCompletionResponse>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<Arc<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<Arc<serde_json::Value>>,
}

#[derive(Clone)]
pub struct AuditHandle {
    requested_streaming: bool,
    request_id: String,
    model: String,
    headers: Option<Arc<serde_json::Value>>,
}

impl AuditHandle {
    pub fn streaming(&self) -> bool {
        self.requested_streaming
    }

    pub fn with_headers(mut self, headers: Option<Arc<serde_json::Value>>) -> Self {
        self.headers = headers;
        self
    }

    pub fn emit_request(
        &self,
        request: Option<Arc<NvCreateChatCompletionRequest>>,
        raw_request: Option<Arc<serde_json::Value>>,
        headers: Option<Arc<serde_json::Value>>,
    ) {
        let rec = AuditRecord {
            schema_version: 1,
            event_type: AuditEventType::Request,
            request_id: self.request_id.clone(),
            requested_streaming: self.requested_streaming,
            model: self.model.clone(),
            request,
            raw_request,
            response: None,
            raw_response: None,
            headers: headers.or_else(|| self.headers.clone()),
        };
        bus::publish(rec);
    }

    pub fn emit_response(self, response: Arc<NvCreateChatCompletionResponse>) {
        let rec = AuditRecord {
            schema_version: 1,
            event_type: AuditEventType::Response,
            request_id: self.request_id,
            requested_streaming: self.requested_streaming,
            model: self.model,
            request: None,
            raw_request: None,
            response: Some(response),
            raw_response: None,
            headers: self.headers,
        };
        bus::publish(rec);
    }

    pub fn emit_raw_response(self, response: Arc<serde_json::Value>) {
        let rec = AuditRecord {
            schema_version: 1,
            event_type: AuditEventType::Response,
            request_id: self.request_id,
            requested_streaming: self.requested_streaming,
            model: self.model,
            request: None,
            raw_request: None,
            response: None,
            raw_response: Some(response),
            headers: self.headers,
        };
        bus::publish(rec);
    }
}

pub fn create_handle(req: &NvCreateChatCompletionRequest, request_id: &str) -> Option<AuditHandle> {
    let policy = config::policy();
    create_handle_with_config(req, request_id, policy.enabled, policy.force_logging)
}

fn create_handle_with_config(
    req: &NvCreateChatCompletionRequest,
    request_id: &str,
    enabled: bool,
    force_logging: bool,
) -> Option<AuditHandle> {
    if !enabled {
        return None;
    }
    // If force_logging is enabled, ignore the store flag
    if !force_logging && !req.inner.store.unwrap_or(false) {
        return None;
    }
    let requested_streaming = req.inner.stream.unwrap_or(false);
    let model = req.inner.model.clone();

    Some(AuditHandle {
        requested_streaming,
        request_id: request_id.to_string(),
        model,
        headers: None,
    })
}

pub fn emit_raw_request_response(
    request_id: &str,
    model: String,
    requested_streaming: bool,
    raw_request: Option<Arc<serde_json::Value>>,
    headers: Option<Arc<serde_json::Value>>,
    raw_response: Arc<serde_json::Value>,
) {
    if !should_emit_raw_request_response(raw_request.as_deref()) {
        return;
    }

    bus::publish(AuditRecord {
        schema_version: 1,
        event_type: AuditEventType::Request,
        request_id: request_id.to_string(),
        requested_streaming,
        model: model.clone(),
        request: None,
        raw_request,
        response: None,
        raw_response: None,
        headers: headers.clone(),
    });
    bus::publish(AuditRecord {
        schema_version: 1,
        event_type: AuditEventType::Response,
        request_id: request_id.to_string(),
        requested_streaming,
        model,
        request: None,
        raw_request: None,
        response: None,
        raw_response: Some(raw_response),
        headers,
    });
}

pub fn should_emit_raw_request_response(raw_request: Option<&serde_json::Value>) -> bool {
    let policy = config::policy();
    if !policy.enabled {
        return false;
    }
    if policy.force_logging {
        return true;
    }
    raw_request
        .and_then(|request| request.get("store"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn create_test_request(model: &str, store: bool) -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "store": store
        });
        serde_json::from_value(json).expect("Failed to create test request")
    }

    fn create_test_request_with_nvext() -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "store": true,
            "nvext": {
                "agent_hints": {
                    "priority": 5
                }
            }
        });
        serde_json::from_value(json).expect("Failed to create test request")
    }

    fn create_test_response(content: &str) -> NvCreateChatCompletionResponse {
        let json = serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }]
        });
        serde_json::from_value(json).expect("Failed to create test response")
    }

    /// Test that DYN_AUDIT_FORCE_LOGGING=true bypasses store=false
    /// When force logging is enabled, audit handle should be created even when store=false
    #[test]
    fn test_force_logging_bypasses_store() {
        let request = create_test_request("test-model", false);
        let handle = create_handle_with_config(&request, "test-id", true, true);

        assert!(
            handle.is_some(),
            "force logging should create a handle even with store=false"
        );
    }

    #[test]
    fn audit_record_serializes_nvext_and_response_content() {
        let record = AuditRecord {
            schema_version: 1,
            event_type: AuditEventType::Response,
            request_id: "req-123".to_string(),
            requested_streaming: true,
            model: "test-model".to_string(),
            request: Some(Arc::new(create_test_request_with_nvext())),
            raw_request: None,
            response: Some(Arc::new(create_test_response("final answer"))),
            raw_response: None,
            headers: None,
        };

        let value = serde_json::to_value(record).unwrap();

        assert_eq!(value["request"]["nvext"]["agent_hints"]["priority"], 5);
        assert_eq!(
            value["response"]["choices"][0]["message"]["content"],
            "final answer"
        );
    }
}
