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
}

impl AuditHandle {
    pub fn streaming(&self) -> bool {
        self.requested_streaming
    }

    pub fn emit_request(
        &self,
        request: Arc<NvCreateChatCompletionRequest>,
        raw_request: Option<Arc<serde_json::Value>>,
        headers: Option<Arc<serde_json::Value>>,
    ) {
        let rec = AuditRecord {
            schema_version: 1,
            event_type: AuditEventType::Request,
            request_id: self.request_id.clone(),
            requested_streaming: self.requested_streaming,
            model: self.model.clone(),
            request: Some(request),
            raw_request,
            response: None,
            raw_response: None,
            headers,
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
            headers: None,
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
            headers: None,
        };
        bus::publish(rec);
    }
}

pub fn create_handle(req: &NvCreateChatCompletionRequest, request_id: &str) -> Option<AuditHandle> {
    let policy = config::policy();
    if !policy.enabled {
        return None;
    }
    // If force_logging is enabled, ignore the store flag
    if !policy.force_logging && !req.inner.store.unwrap_or(false) {
        return None;
    }
    let requested_streaming = req.inner.stream.unwrap_or(false);
    let model = req.inner.model.clone();

    Some(AuditHandle {
        requested_streaming,
        request_id: request_id.to_string(),
        model,
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
    let policy = config::policy();
    if !policy.enabled {
        return;
    }

    let requested_store = raw_request
        .as_ref()
        .and_then(|request| request.get("store"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if !policy.force_logging && !requested_store {
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
        headers,
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
        headers: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use temp_env::with_vars;

    fn create_test_request(model: &str, store: bool) -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "store": store
        });
        serde_json::from_value(json).expect("Failed to create test request")
    }

    /// Test that DYN_AUDIT_FORCE_LOGGING=true bypasses store=false
    /// When force logging is enabled, audit handle should be created even when store=false
    #[test]
    fn test_force_logging_bypasses_store() {
        with_vars(
            [
                ("DYN_AUDIT_SINKS", Some("stderr")),
                ("DYN_AUDIT_FORCE_LOGGING", Some("true")),
            ],
            || {
                // Create request with store=false
                let request = create_test_request("test-model", false);
                let handle = create_handle(&request, "test-id");

                assert!(
                    handle.is_some(),
                    "When DYN_AUDIT_FORCE_LOGGING=true, handle should be created even with store=false"
                );
            },
        );
    }
}
