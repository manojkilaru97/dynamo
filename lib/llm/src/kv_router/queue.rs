// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

use super::WorkerSelector;
use super::protocols::WorkerWithDpRank;
use super::scheduler::{SchedulingRequest, SchedulingResponse};
use super::sequence::{ActiveSequencesMulti, SequenceRequest};
use crate::discovery::RuntimeConfigWatch;

/// Large default for max_num_batched_tokens when not configured (effectively disables queueing for that worker)
const DEFAULT_MAX_BATCHED_TOKENS: u64 = 10_000_000;

/// Entry in the priority queue, ordered by effective arrival time (lower = higher priority).
/// Effective arrival = elapsed time since queue start minus `priority_jump`.
struct QueueEntry {
    effective_offset: Duration,
    request: SchedulingRequest,
    enqueued_at: Instant,
}

impl Eq for QueueEntry {}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.effective_offset == other.effective_offset
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; reverse so lower effective_offset = higher priority
        other.effective_offset.cmp(&self.effective_offset)
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Queue that gates scheduling requests behind a capacity check.
/// When all workers exceed `threshold_frac` utilisation the request is parked in `pending`.
/// When capacity frees up (`update()`), pending requests are scheduled in priority order.
/// If queueing is disabled (threshold_frac is None), requests are scheduled immediately.
pub struct SchedulerQueue {
    pending: Mutex<BinaryHeap<QueueEntry>>,
    pending_count: AtomicUsize,
    slots: Arc<ActiveSequencesMulti>,
    workers_with_configs: RuntimeConfigWatch,
    /// Cached threshold fraction; None disables token-threshold queueing, but request-slot
    /// saturation still applies when workers publish max_num_seqs.
    threshold_frac: Option<f64>,
    /// Maximum number of queued requests to allow per eligible worker.
    max_pending_per_worker: Option<usize>,
    /// Maximum time a request may remain queued before being failed.
    max_queue_wait: Option<Duration>,
    /// Reference instant for computing arrival offsets.
    start_time: Instant,
    block_size: u32,
    selector: Box<dyn WorkerSelector + Send + Sync>,
}

impl SchedulerQueue {
    pub fn new(
        slots: Arc<ActiveSequencesMulti>,
        workers_with_configs: RuntimeConfigWatch,
        threshold_frac: Option<f64>,
        max_pending_per_worker: Option<usize>,
        max_queue_wait: Option<Duration>,
        block_size: u32,
        selector: Box<dyn WorkerSelector + Send + Sync>,
    ) -> Self {
        if let Some(frac) = threshold_frac {
            tracing::info!("Router queue enabled with token threshold fraction {frac}");
        }
        if let Some(limit) = max_pending_per_worker {
            tracing::info!(
                "Router queue depth limited to {limit} pending requests per eligible worker"
            );
        }
        if let Some(timeout) = max_queue_wait {
            tracing::info!(
                timeout_ms = timeout.as_millis() as u64,
                "Router queue wait timeout enabled"
            );
        }
        Self {
            pending: Mutex::new(BinaryHeap::new()),
            pending_count: AtomicUsize::new(0),
            slots,
            workers_with_configs,
            threshold_frac,
            max_pending_per_worker,
            max_queue_wait,
            start_time: Instant::now(),
            block_size,
            selector,
        }
    }

    /// Build a QueueEntry for a request, computing its effective arrival offset.
    fn make_entry(&self, request: SchedulingRequest) -> QueueEntry {
        let arrival_offset = self.start_time.elapsed();
        let jump = Duration::from_secs_f64(request.priority_jump.max(0.0));
        let effective_offset = arrival_offset.saturating_sub(jump);
        QueueEntry {
            effective_offset,
            request,
            enqueued_at: Instant::now(),
        }
    }

    /// Enqueue a new request.
    /// If any eligible worker has capacity, schedule immediately.
    /// Otherwise park in the pending heap, subject to the configured queue bound.
    pub async fn enqueue(&self, request: SchedulingRequest) {
        if self.request_must_wait(&request) {
            if let Some(limit) = self.current_pending_limit(request.allowed_worker_ids.as_ref()) {
                let pending = self.pending_count();
                if pending >= limit {
                    let mut request = request;
                    request.respond(Err(super::scheduler::KvSchedulerError::QueueFull {
                        pending,
                        limit,
                    }));
                    return;
                }
            }

            tracing::debug!("all eligible workers busy, queueing request");
            let entry = self.make_entry(request);
            self.pending.lock().await.push(entry);
            self.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
            return;
        }

        self.schedule(request).await;
    }

    /// Called on prefill_complete/free. Drains pending requests while workers have capacity.
    /// Each scheduled request updates active_tokens via add_request, so the busy check
    /// sees fresh state on the next iteration.
    pub async fn update(&self) {
        let expired = {
            let mut heap = self.pending.lock().await;
            self.refresh_pending_locked(&mut heap)
        };
        self.fail_expired(expired);

        loop {
            let Some(entry) = ({
                let mut heap = self.pending.lock().await;
                heap.pop().map(|entry| {
                    self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
                    entry
                })
            }) else {
                break;
            };

            if self.is_expired(&entry) {
                self.fail_expired(vec![(
                    entry.request,
                    self.max_queue_wait
                        .expect("expired entry requires max_queue_wait")
                        .as_millis() as u64,
                )]);
                continue;
            }

            if self.request_must_wait(&entry.request) {
                let mut heap = self.pending.lock().await;
                heap.push(entry);
                self.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
                break;
            }

            tracing::debug!("scheduling request from pending queue");
            self.schedule(entry.request).await;
        }
    }

    /// Run the full scheduling pipeline for a single request:
    /// compute potential load → select worker → respond → book via add_request.
    async fn schedule(&self, mut request: SchedulingRequest) {
        let saturated_workers = self.saturated_workers(request.allowed_worker_ids.as_ref());
        request.disallowed_workers = if saturated_workers.is_empty() {
            None
        } else {
            Some(saturated_workers)
        };

        let (decode_blocks, prefill_tokens) = self.slots.potential_blocks_and_tokens(
            request.token_seq.clone(),
            request.isl_tokens,
            request.overlaps.clone(),
        );
        request.decode_blocks = decode_blocks;
        request.prefill_tokens = prefill_tokens;

        let selection = {
            let workers = self.workers_with_configs.borrow();
            self.selector
                .select_worker(&workers, &request, self.block_size)
        };

        let selection = match selection {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("scheduling failed: {e}");
                request.respond(Err(e));
                return;
            }
        };

        request.respond(Ok(SchedulingResponse {
            best_worker: selection.worker,
            overlap_blocks: selection.overlap_blocks,
        }));

        if !request.update_states {
            return;
        }

        let Some(request_id) = request.maybe_request_id else {
            tracing::error!("No request_id provided to add_request to the slot tracker");
            return;
        };

        if let Err(e) = self
            .slots
            .add_request(SequenceRequest {
                request_id: request_id.clone(),
                token_sequence: request.token_seq,
                isl: request.isl_tokens,
                overlap: selection.overlap_blocks,
                expected_output_tokens: None,
                worker: selection.worker,
                lora_name: request.lora_name.clone(),
            })
            .await
        {
            tracing::warn!("Failed to add request {request_id}: {e}");
        }
    }

    /// Number of requests currently parked in the pending queue (lock-free).
    pub fn pending_count(&self) -> usize {
        self.pending_count.load(AtomicOrdering::Relaxed)
    }

    fn request_must_wait(&self, request: &SchedulingRequest) -> bool {
        self.all_workers_busy(request.allowed_worker_ids.as_ref())
    }

    /// Check if all eligible workers are busy based on max_num_seqs or token threshold.
    /// Returns true only if ALL eligible workers exceed capacity.
    fn all_workers_busy(&self, allowed: Option<&HashSet<WorkerId>>) -> bool {
        let active_tokens = self.slots.active_tokens();
        let active_requests = self.slots.active_requests();
        let configs = self.workers_with_configs.borrow();

        let mut checked_any = false;
        for (&worker_id, config) in configs.iter() {
            if let Some(ids) = allowed
                && !ids.contains(&worker_id)
            {
                continue;
            }

            let dp_size = config.data_parallel_size;
            let dp_start = config.data_parallel_start_rank;

            for dp_rank in dp_start..dp_start + dp_size {
                checked_any = true;
                let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                if !self.worker_is_saturated(worker, config, &active_tokens, &active_requests) {
                    return false;
                }
            }
        }
        checked_any
    }

    fn saturated_workers(&self, allowed: Option<&HashSet<WorkerId>>) -> HashSet<WorkerWithDpRank> {
        let active_tokens = self.slots.active_tokens();
        let active_requests = self.slots.active_requests();
        let configs = self.workers_with_configs.borrow();
        let mut saturated = HashSet::new();

        for (&worker_id, config) in configs.iter() {
            if let Some(ids) = allowed
                && !ids.contains(&worker_id)
            {
                continue;
            }

            let dp_size = config.data_parallel_size;
            let dp_start = config.data_parallel_start_rank;
            for dp_rank in dp_start..dp_start + dp_size {
                let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                if self.worker_is_saturated(worker, config, &active_tokens, &active_requests) {
                    saturated.insert(worker);
                }
            }
        }

        saturated
    }

    fn worker_is_saturated(
        &self,
        worker: WorkerWithDpRank,
        config: &crate::local_model::runtime_config::ModelRuntimeConfig,
        active_tokens: &HashMap<WorkerWithDpRank, usize>,
        active_requests: &HashMap<WorkerWithDpRank, usize>,
    ) -> bool {
        if let Some(max_num_seqs) = config.max_num_seqs {
            let active = active_requests.get(&worker).copied().unwrap_or(0) as u64;
            if active >= max_num_seqs {
                return true;
            }
        }

        let Some(threshold) = self.threshold_frac else {
            return false;
        };

        let max_batched = config
            .max_num_batched_tokens
            .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS);
        let tokens = active_tokens.get(&worker).copied().unwrap_or(0);
        (tokens as f64) > threshold * (max_batched as f64)
    }

    fn current_pending_limit(&self, allowed: Option<&HashSet<WorkerId>>) -> Option<usize> {
        let per_worker_limit = self.max_pending_per_worker?;
        let configs = self.workers_with_configs.borrow();
        let mut eligible_workers = 0usize;

        for (&worker_id, config) in configs.iter() {
            if let Some(ids) = allowed
                && !ids.contains(&worker_id)
            {
                continue;
            }
            eligible_workers += config.data_parallel_size as usize;
        }

        (eligible_workers > 0).then_some(per_worker_limit.saturating_mul(eligible_workers))
    }

    fn is_expired(&self, entry: &QueueEntry) -> bool {
        let Some(limit) = self.max_queue_wait else {
            return false;
        };
        entry.enqueued_at.elapsed() >= limit
    }

    fn refresh_pending_locked(
        &self,
        heap: &mut BinaryHeap<QueueEntry>,
    ) -> Vec<(SchedulingRequest, u64)> {
        let pending = std::mem::take(heap).into_vec();
        let limit_ms = self.max_queue_wait.map(|limit| limit.as_millis() as u64);

        let mut fresh = Vec::with_capacity(pending.len());
        let mut expired = Vec::new();
        for entry in pending {
            if self.is_expired(&entry) {
                expired.push((entry.request, limit_ms.unwrap_or_default()));
                continue;
            }
            fresh.push(entry);
        }

        *heap = BinaryHeap::from(fresh);
        self.pending_count
            .store(heap.len(), AtomicOrdering::Relaxed);
        expired
    }

    fn fail_expired(&self, expired: Vec<(SchedulingRequest, u64)>) {
        for (mut request, limit_ms) in expired {
            request.respond(Err(super::scheduler::KvSchedulerError::QueueWaitTimeout {
                waited_ms: limit_ms,
                limit_ms,
            }));
        }
    }
}
