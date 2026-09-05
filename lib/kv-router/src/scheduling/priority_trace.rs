// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::protocols::WorkerWithDpRank;

static EVENT_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static NEXT_QUEUE_ID: AtomicU64 = AtomicU64::new(1);
static START: LazyLock<Instant> = LazyLock::new(Instant::now);

pub(super) fn enabled() -> bool {
    dynamo_truthy::env_is_truthy("DYN_PRIORITY_TRACE")
}

fn next_sequence() -> Option<u64> {
    if !enabled() {
        return None;
    }
    let limit = std::env::var("DYN_PRIORITY_TRACE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(200_000)
        .min(1_000_000);
    let sequence = EVENT_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
    (sequence <= limit).then_some(sequence)
}

pub(super) fn next_queue_id() -> u64 {
    NEXT_QUEUE_ID.fetch_add(1, Ordering::Relaxed)
}

fn format_candidates(candidates: &[WorkerWithDpRank]) -> (String, bool) {
    const MAX_CANDIDATES: usize = 64;
    let value = candidates
        .iter()
        .take(MAX_CANDIDATES)
        .map(|worker| format!("{}:{}", worker.worker_id, worker.dp_rank))
        .collect::<Vec<_>>()
        .join(",");
    (value, candidates.len() > MAX_CANDIDATES)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_router_event(
    stage: &'static str,
    queue_id: u64,
    request_id: &str,
    policy_class: &str,
    queue_policy: &str,
    enqueue_sequence: Option<u64>,
    priority_jump: f64,
    strict_priority: u32,
    policy_score: Option<f64>,
    wait_ms: Option<u64>,
    candidates: &[WorkerWithDpRank],
    selected_worker: Option<WorkerWithDpRank>,
) {
    let Some(sequence) = next_sequence() else {
        return;
    };
    let (eligible_candidates, candidates_truncated) = format_candidates(candidates);
    tracing::info!(
        target: "dynamo_priority",
        schema = "dynamo.priority.v1",
        stage,
        sequence,
        monotonic_ns = START.elapsed().as_nanos() as u64,
        process_id = std::process::id(),
        queue_id,
        request_id,
        policy_class,
        queue_policy,
        enqueue_sequence,
        priority_jump,
        strict_priority,
        policy_score,
        wait_ms,
        eligible_candidates,
        eligible_candidate_count = candidates.len(),
        candidates_truncated,
        selected_worker_id = selected_worker.map(|worker| worker.worker_id),
        selected_dp_rank = selected_worker.map(|worker| worker.dp_rank),
        "priority_trace"
    );
}
