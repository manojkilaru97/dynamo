// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use dynamo_runtime::config::environment_names::llm::audit as env_audit;

const DEFAULT_CAPACITY: usize = 1024;
const DEFAULT_OTEL_MAX_PAYLOAD_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct AuditPolicy {
    pub enabled: bool,
    pub force_logging: bool,
    pub capacity: usize,
    pub sinks: Vec<String>,
    pub otel_max_payload_bytes: usize,
}

static POLICY: OnceLock<AuditPolicy> = OnceLock::new();

fn parse_sink_names(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Audit is enabled if we have at least one sink
fn init_from_env() -> AuditPolicy {
    let sinks = std::env::var(env_audit::DYN_AUDIT_SINKS)
        .ok()
        .map(|v| parse_sink_names(&v))
        .unwrap_or_default();
    let capacity = std::env::var(env_audit::DYN_AUDIT_CAPACITY)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_CAPACITY);
    let otel_max_payload_bytes = std::env::var(env_audit::DYN_AUDIT_OTEL_MAX_PAYLOAD_BYTES)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_OTEL_MAX_PAYLOAD_BYTES);

    AuditPolicy {
        enabled: !sinks.is_empty(),
        force_logging: std::env::var(env_audit::DYN_AUDIT_FORCE_LOGGING)
            .ok()
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false),
        capacity,
        sinks,
        otel_max_payload_bytes,
    }
}

pub fn policy() -> &'static AuditPolicy {
    POLICY.get_or_init(init_from_env)
}
