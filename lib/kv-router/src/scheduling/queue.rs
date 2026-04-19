// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tokio::sync::watch;

use super::policy::{FcfsPolicy, SchedulingPolicy};
use super::selector::WorkerSelector;
use super::types::{KvSchedulerError, SchedulingRequest, SchedulingResponse};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};
use crate::sequences::{ActiveSequencesMultiWorker, SequencePublisher, SequenceRequest};

/// Large default for max_num_batched_tokens when not configured (effectively disables queueing for that worker)
pub const DEFAULT_MAX_BATCHED_TOKENS: u64 = 10_000_000;

/// Entry in the priority queue, ordered by key (higher key = higher priority).
struct QueueEntry<K: Ord + Eq> {
    key: K,
    request: SchedulingRequest,
    enqueued_at: Instant,
}

impl<K: Ord + Eq> Eq for QueueEntry<K> {}

impl<K: Ord + Eq> PartialEq for QueueEntry<K> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<K: Ord + Eq> Ord for QueueEntry<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl<K: Ord + Eq> PartialOrd for QueueEntry<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Queue that gates scheduling requests behind capacity checks.
/// Requests are queued when all eligible workers are saturated by either:
/// - token load (`router_queue_threshold * max_num_batched_tokens`), or
/// - request-slot load (`max_num_seqs`)
///
/// The queue also supports an optional bounded depth and queue wait timeout.
pub struct SchedulerQueue<
    P: SequencePublisher,
    C: WorkerConfigLike,
    S: SchedulingPolicy = FcfsPolicy,
> {
    pending: Mutex<BinaryHeap<QueueEntry<S::Key>>>,
    /// Number of requests currently parked in the pending queue.
    /// Incremented after push, decremented after pop. Lock-free reads via `Relaxed` load.
    pending_count: AtomicUsize,
    slots: Arc<ActiveSequencesMultiWorker<P>>,
    workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
    /// Cached threshold fraction; None disables token-threshold queueing but hard request-slot
    /// saturation still applies.
    threshold_frac: Option<f64>,
    /// Maximum number of queued requests to allow per eligible worker. None disables queue-depth
    /// limiting.
    max_pending_per_worker: Option<usize>,
    /// Maximum time a request may remain queued before being failed. None disables queue age
    /// enforcement.
    max_queue_wait: Option<Duration>,
    /// Reference instant for computing arrival offsets.
    start_time: Instant,
    block_size: u32,
    selector: Box<dyn WorkerSelector<C> + Send + Sync>,
    policy: S,
}

impl<P: SequencePublisher + 'static, C: WorkerConfigLike, S: SchedulingPolicy>
    SchedulerQueue<P, C, S>
{
    pub fn new(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        threshold_frac: Option<f64>,
        max_pending_per_worker: Option<usize>,
        max_queue_wait: Option<Duration>,
        block_size: u32,
        selector: Box<dyn WorkerSelector<C> + Send + Sync>,
        policy: S,
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
            policy,
        }
    }

    /// Register externally-provided workers in the slot tracker.
    ///
    /// Looks up DP rank/size from the discovery watch channel; defaults to
    /// `(0, 1)` for workers not yet known to discovery.
    pub fn register_workers(&self, worker_ids: &std::collections::HashSet<u64>) {
        let discovery_workers = self.workers_with_configs.borrow();
        let dp_range: std::collections::HashMap<u64, (u32, u32)> = worker_ids
            .iter()
            .map(|&id| {
                let (dp_start, dp_size) = discovery_workers
                    .get(&id)
                    .map(|runtime_config| {
                        (
                            runtime_config.data_parallel_start_rank(),
                            runtime_config.data_parallel_size(),
                        )
                    })
                    .unwrap_or((0, 1));
                (id, (dp_start, dp_size))
            })
            .collect();
        self.slots.register_external_workers(&dp_range);
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
                    request.respond(Err(KvSchedulerError::QueueFull { pending, limit }));
                    return;
                }
            }

            tracing::debug!("all eligible workers busy, queueing request");
            let arrival_offset = self.start_time.elapsed();
            let key = self.policy.enqueue_key(arrival_offset, &request);
            self.pending.lock().await.push(QueueEntry {
                key,
                request,
                enqueued_at: Instant::now(),
            });
            self.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
            return;
        }

        self.schedule(request).await;
    }

    /// Called on prefill_complete/free and by the periodic scheduler loop.
    /// Drains timed-out queued requests, then schedules queued work while the head request can be
    /// admitted.
    pub async fn update(&self) {
        let expired = {
            let mut heap = self.pending.lock().await;
            self.refresh_pending_locked(&mut heap)
        };
        self.fail_expired(expired);

        loop {
            let maybe_entry = {
                let mut heap = self.pending.lock().await;
                heap.pop().map(|entry| {
                    self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
                    entry
                })
            };

            let Some(entry) = maybe_entry else {
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
    /// compute potential load -> select worker -> respond -> book via add_request.
    async fn schedule(&self, mut request: SchedulingRequest) {
        let saturated_workers = self.saturated_workers(request.allowed_worker_ids.as_ref());
        request.disallowed_workers = if saturated_workers.is_empty() {
            None
        } else {
            Some(saturated_workers)
        };

        let (decode_blocks, prefill_tokens) = self.slots.potential_blocks_and_tokens(
            request.token_seq.as_deref(),
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
                expected_output_tokens: request.expected_output_tokens,
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
            let dp_size = config.data_parallel_size();
            let dp_start_rank = config.data_parallel_start_rank();

            for dp_rank in dp_start_rank..dp_start_rank + dp_size {
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
            let dp_size = config.data_parallel_size();
            let dp_start_rank = config.data_parallel_start_rank();
            for dp_rank in dp_start_rank..dp_start_rank + dp_size {
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
        config: &C,
        active_tokens: &HashMap<WorkerWithDpRank, usize>,
        active_requests: &HashMap<WorkerWithDpRank, usize>,
    ) -> bool {
        if let Some(max_num_seqs) = config.max_num_seqs() {
            let active = active_requests.get(&worker).copied().unwrap_or(0) as u64;
            if active >= max_num_seqs {
                return true;
            }
        }

        let Some(threshold) = self.threshold_frac else {
            return false;
        };

        let max_batched = config
            .max_num_batched_tokens()
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
            eligible_workers += config.data_parallel_size() as usize;
        }

        (eligible_workers > 0).then_some(per_worker_limit.saturating_mul(eligible_workers))
    }

    fn is_expired(&self, entry: &QueueEntry<S::Key>) -> bool {
        let Some(limit) = self.max_queue_wait else {
            return false;
        };
        entry.enqueued_at.elapsed() >= limit
    }

    fn refresh_pending_locked(
        &self,
        heap: &mut BinaryHeap<QueueEntry<S::Key>>,
    ) -> Vec<(SchedulingRequest, u64)> {
        let pending = std::mem::take(heap).into_vec();
        let arrival_offset = self.start_time.elapsed();
        let limit_ms = self.max_queue_wait.map(|limit| limit.as_millis() as u64);

        let mut fresh = Vec::with_capacity(pending.len());
        let mut expired = Vec::new();
        for entry in pending {
            if self.is_expired(&entry) {
                expired.push((entry.request, limit_ms.unwrap_or_default()));
                continue;
            }

            let key = if S::DYNAMIC {
                self.policy
                    .rekey(arrival_offset, &entry.key, &entry.request)
            } else {
                entry.key
            };
            fresh.push(QueueEntry {
                key,
                request: entry.request,
                enqueued_at: entry.enqueued_at,
            });
        }

        *heap = BinaryHeap::from(fresh);
        self.pending_count
            .store(heap.len(), AtomicOrdering::Relaxed);
        expired
    }

    fn fail_expired(&self, expired: Vec<(SchedulingRequest, u64)>) {
        for (mut request, limit_ms) in expired {
            request.respond(Err(KvSchedulerError::QueueWaitTimeout {
                waited_ms: limit_ms,
                limit_ms,
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::sync::watch;

    use super::*;
    use crate::protocols::OverlapScores;
    use crate::selector::DefaultWorkerSelector;
    use crate::sequences::ActiveSequencesMultiWorker;
    use crate::test_utils::{NoopSequencePublisher, SimpleWorkerConfig};

    fn make_queue(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
        max_num_seqs: Option<u64>,
        max_pending_per_worker: Option<usize>,
        max_queue_wait: Option<Duration>,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));

        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        for id in 0..num_workers as u64 {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_seqs,
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        let (cfg_tx, cfg_rx) = watch::channel(configs);
        std::mem::forget(cfg_tx);

        let selector = Box::new(DefaultWorkerSelector::new(None, "test"));
        let queue = Arc::new(SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            threshold_frac,
            max_pending_per_worker,
            max_queue_wait,
            block_size,
            selector,
            FcfsPolicy,
        ));

        (queue, slots)
    }

    fn make_request(
        request_id: &str,
        isl_tokens: usize,
    ) -> (
        SchedulingRequest,
        tokio::sync::oneshot::Receiver<
            Result<SchedulingResponse, crate::scheduling::types::KvSchedulerError>,
        >,
    ) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let req = SchedulingRequest {
            maybe_request_id: Some(request_id.to_string()),
            token_seq: None,
            isl_tokens,
            overlaps: OverlapScores::default(),
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
            router_config_override: None,
            update_states: true,
            lora_name: None,
            priority_jump: 0.0,
            expected_output_tokens: None,
            allowed_worker_ids: None,
            disallowed_workers: None,
            resp_tx: Some(tx),
        };
        (req, rx)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_flood() {
        let block_size = 16;
        let isl = 512;
        let num_workers = 4;
        let num_tasks = 25;

        let (queue, slots) = make_queue(num_workers, block_size, isl, None, None, None, None);

        let mut handles = Vec::new();
        for i in 0..num_tasks {
            let queue = Arc::clone(&queue);
            let slots = Arc::clone(&slots);
            handles.push(tokio::spawn(async move {
                let req_id = format!("req-{i}");
                let (req, rx) = make_request(&req_id, isl);
                queue.enqueue(req).await;
                let resp = rx.await.expect("oneshot dropped");
                let resp = resp.expect("scheduling failed");
                assert!(resp.best_worker.worker_id < num_workers as u64);

                slots.mark_prefill_completed(&req_id).await.unwrap();
                slots.free(&req_id).await.unwrap();
                queue.update().await;
            }));
        }

        for h in handles {
            h.await.expect("task panicked");
        }

        let active = slots.active_tokens();
        for (worker, tokens) in &active {
            assert_eq!(
                *tokens, 0,
                "worker {worker:?} still has {tokens} active tokens"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_queueing_under_pressure() {
        let block_size = 16;
        let isl = 512;
        let num_workers = 2;
        let num_requests = 10;

        let (queue, slots) = make_queue(
            num_workers,
            block_size,
            isl,
            Some(0.0),
            None,
            Some(32),
            Some(Duration::from_secs(30)),
        );

        let mut receivers = Vec::new();
        let mut req_ids = Vec::new();

        for i in 0..num_requests {
            let req_id = format!("pressure-{i}");
            let (req, rx) = make_request(&req_id, isl);
            queue.enqueue(req).await;
            receivers.push(rx);
            req_ids.push(req_id);
        }

        for _ in 0..num_requests {
            queue.update().await;
            for rid in &req_ids {
                let _ = slots.mark_prefill_completed(rid).await;
                let _ = slots.free(rid).await;
            }
        }
        queue.update().await;

        let mut ok_count = 0;
        for mut rx in receivers {
            if let Ok(result) = rx.try_recv() {
                result.expect("scheduling returned error");
                ok_count += 1;
            }
        }
        assert_eq!(ok_count, num_requests, "not all requests were scheduled");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_pending_count() {
        let block_size = 16;
        let isl = 512;
        let num_workers = 1;

        let (queue, slots) = make_queue(
            num_workers,
            block_size,
            isl,
            Some(0.0),
            None,
            Some(32),
            Some(Duration::from_secs(30)),
        );
        assert_eq!(queue.pending_count(), 0);

        let (req1, rx1) = make_request("req-1", isl);
        queue.enqueue(req1).await;
        let _resp1 = rx1.await.unwrap().unwrap();
        assert_eq!(queue.pending_count(), 0);

        let (req2, _rx2) = make_request("req-2", isl);
        queue.enqueue(req2).await;
        assert_eq!(queue.pending_count(), 1);

        let (req3, _rx3) = make_request("req-3", isl);
        queue.enqueue(req3).await;
        assert_eq!(queue.pending_count(), 2);

        slots
            .mark_prefill_completed(&"req-1".to_string())
            .await
            .unwrap();
        slots.free(&"req-1".to_string()).await.unwrap();
        queue.update().await;

        assert!(
            queue.pending_count() < 2,
            "pending_count should decrease after free+update, got {}",
            queue.pending_count()
        );

        let _ = slots.mark_prefill_completed(&"req-2".to_string()).await;
        let _ = slots.free(&"req-2".to_string()).await;
        queue.update().await;
        let _ = slots.mark_prefill_completed(&"req-3".to_string()).await;
        let _ = slots.free(&"req-3".to_string()).await;
        queue.update().await;

        assert_eq!(queue.pending_count(), 0, "all requests should be drained");
    }

    #[tokio::test]
    async fn test_no_workers_returns_error() {
        let (queue, _slots) = make_queue(0, 16, 512, None, None, None, None);

        let (req, rx) = make_request("lonely-req", 512);
        queue.enqueue(req).await;

        let resp = rx.await.expect("oneshot dropped");
        assert!(
            matches!(
                resp,
                Err(crate::scheduling::types::KvSchedulerError::NoEndpoints)
            ),
            "expected NoEndpoints, got {resp:?}"
        );
    }

    #[tokio::test]
    async fn test_max_num_seqs_forces_queueing_even_without_token_threshold() {
        let (queue, _slots) = make_queue(
            1,
            16,
            512,
            None,
            Some(1),
            Some(32),
            Some(Duration::from_secs(30)),
        );

        let (req1, rx1) = make_request("slot-1", 512);
        queue.enqueue(req1).await;
        let _ = rx1
            .await
            .expect("oneshot dropped")
            .expect("schedule failed");
        assert_eq!(queue.pending_count(), 0);

        let (req2, _rx2) = make_request("slot-2", 512);
        queue.enqueue(req2).await;
        assert_eq!(
            queue.pending_count(),
            1,
            "second request should wait behind max_num_seqs"
        );
    }

    #[tokio::test]
    async fn test_queue_full_returns_error() {
        let (queue, _slots) = make_queue(
            1,
            16,
            512,
            None,
            Some(1),
            Some(1),
            Some(Duration::from_secs(30)),
        );

        let (req1, rx1) = make_request("full-1", 512);
        queue.enqueue(req1).await;
        let _ = rx1
            .await
            .expect("oneshot dropped")
            .expect("schedule failed");

        let (req2, _rx2) = make_request("full-2", 512);
        queue.enqueue(req2).await;
        assert_eq!(queue.pending_count(), 1);

        let (req3, rx3) = make_request("full-3", 512);
        queue.enqueue(req3).await;
        let resp = rx3.await.expect("oneshot dropped");
        assert!(matches!(
            resp,
            Err(KvSchedulerError::QueueFull {
                pending: 1,
                limit: 1
            })
        ));
    }

    #[tokio::test]
    async fn test_queue_wait_timeout_returns_error() {
        let (queue, _slots) = make_queue(
            1,
            16,
            512,
            None,
            Some(1),
            Some(32),
            Some(Duration::from_millis(20)),
        );

        let (req1, rx1) = make_request("timeout-1", 512);
        queue.enqueue(req1).await;
        let _ = rx1
            .await
            .expect("oneshot dropped")
            .expect("schedule failed");

        let (req2, rx2) = make_request("timeout-2", 512);
        queue.enqueue(req2).await;
        assert_eq!(queue.pending_count(), 1);

        tokio::time::sleep(Duration::from_millis(40)).await;
        queue.update().await;

        let resp = rx2.await.expect("oneshot dropped");
        assert!(matches!(
            resp,
            Err(KvSchedulerError::QueueWaitTimeout { limit_ms: 20, .. })
        ));
        assert_eq!(queue.pending_count(), 0);
    }
}
