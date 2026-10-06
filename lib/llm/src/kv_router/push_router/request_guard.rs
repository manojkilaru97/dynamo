// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use dynamo_kv_router::{
    protocols::{
        BlockExtraInfo, BlockHashOptions, compute_block_hash_for_seq, compute_next_seq_hash,
    },
    scheduling::{RequestLifecycleLease, RequestProgressUpdater},
};
use dynamo_runtime::{
    metrics::frontend_perf::{STAGE_DISPATCH, StageGuard},
    protocols::annotated::Annotated,
};
use prometheus::IntCounter;

use crate::{
    kv_router::{
        KvRouter,
        cache_history::{CacheHistory, CacheHistoryStats},
        metrics::RouterRequestMetrics,
    },
    preprocessor::PreprocessedRequest,
    protocols::common::{
        llm_backend::LLMEngineOutput,
        timing::{RequestPhase, RequestTracker},
    },
};

/// Router-side cached-prefix estimate captured for one tracked routing attempt: the prompt
/// length, the best cached prefix among eligible workers, and the cached prefix on the
/// selected worker (all raw tokens, every router-visible tier including CPU offload).
#[derive(Clone, Copy, Debug)]
pub(super) struct RouteObservation {
    pub(super) prompt_tokens: u64,
    pub(super) best_router_tokens: u64,
    pub(super) selected_router_tokens: u64,
}

struct KvHitTracking {
    prompt_tokens: u64,
    /// Taken by the first valid worker report, so an attempt counts at most once.
    reused_tokens: Option<IntCounter>,
}

/// Cache-hit report the worker attaches to its final chunk (`engine_data.kv_cache_hit`).
#[derive(serde::Deserialize)]
struct WorkerCacheHitReport {
    prompt_tokens: u64,
    reused_tokens: u64,
}

/// Reused tokens from a worker report, accepted only when it describes the routed prompt.
fn worker_cache_hit_tokens(prompt_tokens: u64, value: &serde_json::Value) -> Option<u64> {
    let report = <WorkerCacheHitReport as serde::Deserialize>::deserialize(value).ok()?;
    if report.prompt_tokens != prompt_tokens {
        return None;
    }
    Some(report.reused_tokens)
}

/// Opt-in F1 tracking for one routing attempt: which prompt blocks the router's bounded
/// history had already seen at selection, plus the generated blocks to add on completion.
pub(super) struct CacheHistoryTracking {
    history: Arc<CacheHistory>,
    prompt_hashes: Vec<u64>,
    output_hashes: Vec<u64>,
    prompt_tokens: u64,
    previously_seen_tokens: u64,
}

impl CacheHistoryTracking {
    pub(super) fn new(
        history: Arc<CacheHistory>,
        prompt_hashes: Vec<u64>,
        prompt_tokens: u64,
    ) -> Self {
        let previously_seen_tokens = history.previously_computed_tokens(&prompt_hashes);
        Self {
            history,
            prompt_hashes,
            output_hashes: Vec::new(),
            prompt_tokens,
            previously_seen_tokens,
        }
    }
}

struct CacheHistoryFinalization {
    prompt_tokens: u64,
    previously_seen_tokens: u64,
    retained: Option<CacheHistoryStats>,
}

/// Finalize F1 tracking exactly once. Only a completed attempt teaches the history.
fn finalize_cache_history(
    tracking: &mut Option<CacheHistoryTracking>,
    record_completed: bool,
) -> Option<CacheHistoryFinalization> {
    let tracking = tracking.take()?;
    let retained = record_completed
        .then(|| {
            tracking.history.record_completed(
                tracking
                    .prompt_hashes
                    .iter()
                    .copied()
                    .chain(tracking.output_hashes.iter().copied()),
            )
        })
        .flatten();
    Some(CacheHistoryFinalization {
        prompt_tokens: tracking.prompt_tokens,
        previously_seen_tokens: tracking.previously_seen_tokens,
        retained,
    })
}

#[derive(Clone)]
struct OutputHashBranch {
    tail: Vec<u32>,
    parent_hash: Option<u64>,
    first_mm_info: Option<BlockExtraInfo>,
}

/// Incrementally extends the canonical sequence-hash chain used for prompt routing over
/// generated tokens, retaining only each choice's unfinished block.
struct CanonicalOutputTracker {
    template: OutputHashBranch,
    branches: HashMap<u32, OutputHashBranch>,
    block_size: u32,
    lora_name: Option<String>,
    cache_namespace: Option<String>,
    is_eagle: bool,
}

impl CanonicalOutputTracker {
    fn new(
        request: &PreprocessedRequest,
        block_size: u32,
        is_eagle: bool,
        parent_hash: Option<u64>,
    ) -> Self {
        let (tokens, mm_infos) = request.block_mm_routing_info();
        let routing = request.routing.as_ref();
        Self::from_parts(
            tokens,
            mm_infos,
            block_size,
            is_eagle,
            routing.and_then(|routing| routing.lora_name.clone()),
            routing.and_then(|routing| routing.cache_namespace.clone()),
            parent_hash,
        )
    }

    fn from_parts(
        tokens: &[u32],
        mm_infos: Option<&[Option<BlockExtraInfo>]>,
        block_size: u32,
        is_eagle: bool,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        parent_hash: Option<u64>,
    ) -> Self {
        let stride = block_size as usize;
        let complete_blocks = if stride == 0 {
            0
        } else if is_eagle {
            tokens.len().saturating_sub(1) / stride
        } else {
            tokens.len() / stride
        };
        let tail_start = complete_blocks.saturating_mul(stride).min(tokens.len());
        Self {
            template: OutputHashBranch {
                tail: tokens[tail_start..].to_vec(),
                parent_hash,
                first_mm_info: mm_infos
                    .and_then(|infos| infos.get(complete_blocks))
                    .cloned()
                    .flatten(),
            },
            branches: HashMap::new(),
            block_size,
            lora_name,
            cache_namespace,
            is_eagle,
        }
    }

    /// Append one choice's streamed tokens and return the sequence hashes of the blocks they
    /// complete.
    fn observe(&mut self, index: u32, token_ids: &[u32], completed: &mut Vec<u64>) {
        if token_ids.is_empty() || self.block_size == 0 {
            return;
        }
        let stride = self.block_size as usize;
        let window_size = if self.is_eagle { stride + 1 } else { stride };
        // The newest sampled token is visible before the engine feeds it back, so a normal
        // block needs one token beyond its window to have KV. Eagle's window already ends
        // with that lookahead token.
        let materialization_size = window_size + usize::from(!self.is_eagle);
        let branch = self
            .branches
            .entry(index)
            .or_insert_with(|| self.template.clone());
        branch.tail.extend_from_slice(token_ids);
        let mut consumed = 0;
        while branch.tail.len().saturating_sub(consumed) >= materialization_size {
            let mm_info = branch.first_mm_info.clone().map(Some);
            let mm_infos = mm_info.as_ref().map(std::slice::from_ref);
            let Some(local_hash) = compute_block_hash_for_seq(
                &branch.tail[consumed..consumed + window_size],
                self.block_size,
                BlockHashOptions {
                    block_mm_infos: mm_infos,
                    lora_name: self.lora_name.as_deref(),
                    cache_namespace: self.cache_namespace.as_deref(),
                    is_eagle: Some(self.is_eagle),
                },
            )
            .into_iter()
            .next() else {
                break;
            };
            let sequence_hash = branch.parent_hash.map_or(local_hash.0, |parent| {
                compute_next_seq_hash(parent, local_hash)
            });
            completed.push(sequence_hash);
            branch.parent_hash = Some(sequence_hash);
            consumed += stride;
            branch.first_mm_info = None;
        }
        if consumed > 0 {
            branch.tail.drain(..consumed);
        }
    }
}

/// Owns scheduler cleanup after a worker is selected.
///
/// `KvPushRouter` installs this through [`RequestGuard`] immediately after
/// selection and before backend dispatch. The lifecycle lease moves directly
/// from the scheduling response into this guard and remains responsible for
/// cleanup until the request ends.
struct RequestCleanup {
    chooser: Arc<KvRouter>,
    context_id: String,
    scheduler_tracked: bool,
    lifecycle: Option<(RequestProgressUpdater, RequestLifecycleLease)>,
    freed: bool,
}

impl RequestCleanup {
    fn new(
        chooser: Arc<KvRouter>,
        context_id: String,
        scheduler_tracked: bool,
        lifecycle: Option<(RequestProgressUpdater, RequestLifecycleLease)>,
    ) -> Self {
        debug_assert!(lifecycle.is_none() || scheduler_tracked);
        Self {
            chooser,
            context_id,
            scheduler_tracked,
            lifecycle,
            freed: false,
        }
    }

    async fn finish(&mut self) {
        if let Some((_progress, lease)) = self.lifecycle.take() {
            // RequestLifecycleLease reports the terminal outcome to the scheduler actor.
            // The scheduler actor remains the sole owner of booking and queue cleanup.
            drop(lease);
        } else if self.scheduler_tracked
            && let Err(error) = self.chooser.free(&self.context_id).await
        {
            tracing::warn!(
                request_id = %self.context_id,
                %error,
                "Failed to free request"
            );
        }
        self.freed = true;
    }
}

impl Drop for RequestCleanup {
    fn drop(&mut self) {
        if self.freed || !self.scheduler_tracked || self.lifecycle.is_some() {
            return;
        }

        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                request_id = %self.context_id,
                "No tokio runtime for request cleanup"
            );
            return;
        };

        let chooser = self.chooser.clone();
        let context_id = self.context_id.clone();
        handle.spawn(async move {
            let result = chooser.free(&context_id).await;
            if let Err(error) = result {
                tracing::warn!(
                    request_id = %context_id,
                    %error,
                    "Failed to free request from drop guard"
                );
            }
        });
    }
}

/// Owns request-scoped timing and metrics state.
struct RequestObservability {
    tracker: Option<Arc<RequestTracker>>,
    request_metrics: Arc<RouterRequestMetrics>,
    cumulative_osl: usize,
    authoritative_context_tokens: Option<usize>,
    metrics_recorded: bool,
    first_token_recorded: bool,
    dispatch_guard: Option<StageGuard>,
    dispatched: bool,
}

impl RequestObservability {
    fn new(
        tracker: Option<Arc<RequestTracker>>,
        request_metrics: Arc<RouterRequestMetrics>,
    ) -> Self {
        Self {
            tracker,
            request_metrics,
            cumulative_osl: 0,
            authoritative_context_tokens: None,
            metrics_recorded: false,
            first_token_recorded: false,
            dispatch_guard: None,
            dispatched: false,
        }
    }

    fn request_metrics(&self) -> &RouterRequestMetrics {
        &self.request_metrics
    }

    fn start_dispatch(&mut self, phase_label: &str) {
        self.dispatch_guard = Some(StageGuard::new(STAGE_DISPATCH, phase_label));
    }

    fn record_prefill_start(&self) {
        if let Some(tracker) = &self.tracker {
            tracker.record_prefill_start();
        }
    }

    fn mark_dispatched(&mut self) {
        self.dispatched = true;
    }

    fn observe_response(&mut self) {
        // Taking the guard ends dispatch latency exactly once; later responses see None.
        self.dispatch_guard.take();
    }

    fn observe_tokens(&mut self, new_tokens: usize) {
        if !self.first_token_recorded && new_tokens > 0 {
            if let Some(tracker) = &self.tracker {
                tracker.record_first_token();
                if tracker.phase() == RequestPhase::Decode {
                    tracker.record_decode_first_token();
                }
                if let Some(ttft) = tracker.ttft_ms() {
                    self.request_metrics
                        .time_to_first_token_seconds
                        .observe(ttft / 1000.0);
                }
            }
            self.first_token_recorded = true;
        }

        self.cumulative_osl += new_tokens;
    }

    fn cumulative_osl(&self) -> usize {
        self.cumulative_osl
    }

    fn observe_context_tokens(&mut self, context_tokens: usize) {
        self.authoritative_context_tokens = Some(
            self.authoritative_context_tokens
                .map_or(context_tokens, |observed| observed.max(context_tokens)),
        );
    }

    fn context_tokens(&self, initial_context_tokens: usize) -> usize {
        self.authoritative_context_tokens
            .unwrap_or_else(|| initial_context_tokens.saturating_add(self.cumulative_osl))
    }

    fn observe_output_block_boundary(&self) {
        let Some(tracker) = &self.tracker else {
            return;
        };

        // Refresh finish time at block boundaries so the streaming ITL sample stays current.
        tracker.record_osl(self.cumulative_osl);
        tracker.record_finish();
        if let Some(avg_itl) = tracker.avg_itl_ms() {
            self.request_metrics
                .inter_token_latency_seconds
                .observe(avg_itl / 1000.0);
        }
    }

    fn record_metrics(&mut self) {
        // A failed dispatch never reached the backend and must not count as a request.
        if self.metrics_recorded || !self.dispatched {
            return;
        }
        self.metrics_recorded = true;

        if let Some(tracker) = &self.tracker {
            tracker.record_finish();
            tracker.record_osl(self.cumulative_osl);
            if let Some(latency) = tracker.kv_transfer_estimated_latency_secs() {
                self.request_metrics
                    .kv_transfer_estimated_latency_seconds
                    .observe(latency);
            }
        }
        if self.cumulative_osl > 0 {
            self.request_metrics
                .output_sequence_tokens
                .observe(self.cumulative_osl as f64);
        }
        self.request_metrics.requests_total.inc();
    }
}

struct OutputBlockUpdate {
    decay_fraction: Option<f64>,
}

/// Tracks when streamed output grows into a new scheduler accounting block.
struct OutputBlockTracker {
    track_output_blocks: bool,
    track_request_progress: bool,
    current_total_blocks: usize,
    isl_tokens: usize,
    block_size: usize,
    expected_output_tokens: Option<u32>,
}

impl OutputBlockTracker {
    fn new(
        track_output_blocks: bool,
        track_request_progress: bool,
        isl_tokens: usize,
        block_size: usize,
        expected_output_tokens: Option<u32>,
    ) -> Self {
        Self {
            track_output_blocks,
            track_request_progress,
            current_total_blocks: isl_tokens.div_ceil(block_size),
            isl_tokens,
            block_size,
            expected_output_tokens,
        }
    }

    fn observe(&mut self, cumulative_osl: usize) -> Option<OutputBlockUpdate> {
        if !self.track_output_blocks && !self.track_request_progress {
            return None;
        }

        let new_total_blocks = (self.isl_tokens + cumulative_osl).div_ceil(self.block_size);
        if new_total_blocks <= self.current_total_blocks {
            return None;
        }

        // Advance before returning so a failed scheduler update preserves existing no-retry behavior.
        self.current_total_blocks = new_total_blocks;
        let decay_fraction = self
            .expected_output_tokens
            .map(|expected| (1.0 - cumulative_osl as f64 / expected.max(1) as f64).max(0.0));
        Some(OutputBlockUpdate { decay_fraction })
    }
}

/// Coordinates scheduler cleanup, observability, and streamed load tracking.
///
/// Session-affinity lifetime is separate: `AffinityAcquire` and
/// `AffinityLease` own binding commit, release, and invalidation.
pub(super) struct RequestGuard {
    cleanup: RequestCleanup,
    observability: RequestObservability,
    output_blocks: OutputBlockTracker,
    prefill_marked: bool,
    kv_hit: Option<KvHitTracking>,
    cache_history: Option<CacheHistoryTracking>,
    output_hashes: Option<CanonicalOutputTracker>,
}

impl RequestGuard {
    pub(super) fn new(
        chooser: Arc<KvRouter>,
        request_metrics: Arc<RouterRequestMetrics>,
        context_id: String,
        request: &PreprocessedRequest,
        scheduler_tracked: bool,
        lifecycle: Option<(RequestProgressUpdater, RequestLifecycleLease)>,
        kv_route: Option<RouteObservation>,
    ) -> Self {
        // Snapshot request-scoped inputs now so the guard can outlive the
        // PreprocessedRequest after it is moved into backend dispatch.
        let block_size = chooser.block_size() as usize;
        let isl_tokens = request.token_ids.len();
        let expected_output_tokens = request
            .routing
            .as_ref()
            .and_then(|routing| routing.expected_output_tokens);
        let track_output_blocks =
            scheduler_tracked && chooser.kv_router_config().router_track_output_blocks;
        let track_request_progress = lifecycle.is_some();
        if scheduler_tracked {
            request_metrics.requests_started_total().inc();
        }
        let kv_hit = kv_route.map(|route| KvHitTracking {
            prompt_tokens: route.prompt_tokens,
            reused_tokens: Some(request_metrics.observe_kv_route_estimate(
                request.phase(),
                &request.model,
                route.best_router_tokens,
                route.selected_router_tokens,
            )),
        });

        Self {
            cleanup: RequestCleanup::new(chooser, context_id, scheduler_tracked, lifecycle),
            observability: RequestObservability::new(request.tracker.clone(), request_metrics),
            output_blocks: OutputBlockTracker::new(
                track_output_blocks,
                track_request_progress,
                isl_tokens,
                block_size,
                expected_output_tokens,
            ),
            prefill_marked: false,
            kv_hit,
            cache_history: None,
            output_hashes: None,
        }
    }

    /// Start opt-in F1 tracking for this attempt. Generated blocks extend the prompt's
    /// canonical hash chain so a later turn that replays them also counts as seen.
    pub(super) fn track_cache_history(
        &mut self,
        tracking: CacheHistoryTracking,
        request: &PreprocessedRequest,
        block_size: u32,
        is_eagle: bool,
    ) {
        self.observability
            .request_metrics()
            .observe_cache_history_input(tracking.prompt_tokens);
        let parent_hash = tracking.prompt_hashes.last().copied();
        self.output_hashes = Some(CanonicalOutputTracker::new(
            request,
            block_size,
            is_eagle,
            parent_hash,
        ));
        self.cache_history = Some(tracking);
    }

    fn finish_cache_history(&mut self, record_completed: bool) {
        let Some(finalization) = finalize_cache_history(&mut self.cache_history, record_completed)
        else {
            return;
        };
        self.output_hashes = None;
        let metrics = self.observability.request_metrics();
        if record_completed {
            metrics.observe_cache_history_complete(
                finalization.prompt_tokens,
                finalization.previously_seen_tokens,
            );
        } else {
            metrics.observe_cache_history_incomplete();
        }
        if let Some(stats) = finalization.retained {
            metrics.set_cache_history_retained(stats);
        }
    }

    pub(super) fn request_metrics(&self) -> &RouterRequestMetrics {
        self.observability.request_metrics()
    }

    pub(super) fn start_dispatch(&mut self, phase_label: &str) {
        self.observability.start_dispatch(phase_label);
    }

    pub(super) fn record_prefill_start(&self) {
        self.observability.record_prefill_start();
    }

    pub(super) async fn mark_dispatched(&mut self) {
        if let Some((_progress, lease)) = self.cleanup.lifecycle.as_mut() {
            // Backend dispatch already succeeded. Record that fact synchronously so
            // lease cleanup still reports Dispatched before the terminal event if it
            // overtakes the actor command.
            lease.mark_dispatched().await;
        }
        self.observability.mark_dispatched();
    }

    pub(super) async fn on_item(&mut self, item: &Annotated<LLMEngineOutput>) {
        self.observability.observe_response();

        if let Some(usage) = item
            .data
            .as_ref()
            .and_then(|data| data.completion_usage.as_ref())
        {
            self.observability
                .observe_context_tokens(usage.total_tokens as usize);
        }

        if !self.prefill_marked {
            let has_tokens = item
                .data
                .as_ref()
                .is_some_and(|data| !data.token_ids.is_empty());
            if has_tokens {
                if self.cleanup.scheduler_tracked
                    && let Err(error) = self
                        .cleanup
                        .chooser
                        .mark_prefill_completed(&self.cleanup.context_id)
                        .await
                {
                    tracing::warn!(
                        request_id = %self.cleanup.context_id,
                        %error,
                        "Failed to mark prefill completed"
                    );
                }
                self.prefill_marked = true;
            }
        }

        let new_tokens = item.data.as_ref().map_or(0, |data| data.token_ids.len());
        self.observability.observe_tokens(new_tokens);
        self.capture_kv_worker_hit(item);
        if let (Some(history), Some(tracker), Some(data)) = (
            self.cache_history.as_mut(),
            self.output_hashes.as_mut(),
            item.data.as_ref(),
        ) {
            tracker.observe(
                data.index.unwrap_or(0),
                &data.token_ids,
                &mut history.output_hashes,
            );
        }
        let cumulative_osl = self.observability.cumulative_osl();
        let Some(update) = self.output_blocks.observe(cumulative_osl) else {
            return;
        };

        if let Some((progress, _lease)) = &self.cleanup.lifecycle {
            progress.update_context_tokens(
                self.output_blocks.isl_tokens.saturating_add(cumulative_osl),
            );
        }
        if !self.output_blocks.track_output_blocks {
            return;
        }

        if let Err(error) = self
            .cleanup
            .chooser
            .add_output_block(&self.cleanup.context_id, update.decay_fraction)
        {
            tracing::warn!(
                request_id = %self.cleanup.context_id,
                %error,
                "Failed to add output block"
            );
        }

        self.observability.observe_output_block_boundary();
    }

    pub(super) async fn finish(&mut self) {
        // Metrics must observe the completed request before cleanup releases its state.
        self.finish_cache_history(true);
        self.observability.record_metrics();
        self.mark_completed_terminal();
        self.cleanup.finish().await;
    }

    pub(super) fn mark_completed_terminal(&mut self) {
        let context_tokens = self
            .observability
            .context_tokens(self.output_blocks.isl_tokens);
        if let Some((progress, _lease)) = &self.cleanup.lifecycle {
            progress.update_context_tokens(context_tokens);
        }
        if let Some((_progress, lease)) = self.cleanup.lifecycle.as_mut() {
            lease.mark_completed(context_tokens);
        }
    }

    pub(super) async fn abort(&mut self) {
        self.finish_cache_history(false);
        self.cleanup.finish().await;
    }

    /// Count a worker report once, even if the stream subsequently fails or is cancelled.
    fn capture_kv_worker_hit(&mut self, item: &Annotated<LLMEngineOutput>) {
        let Some(kv) = self.kv_hit.as_mut() else {
            return;
        };
        if kv.reused_tokens.is_none() {
            return;
        }
        let Some(reused) = item
            .data
            .as_ref()
            .and_then(|data| data.engine_data.as_ref())
            .and_then(|data| data.get("kv_cache_hit"))
            .and_then(|value| worker_cache_hit_tokens(kv.prompt_tokens, value))
        else {
            return;
        };
        if let Some(counter) = kv.reused_tokens.take() {
            counter.inc_by(reused);
        }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.finish_cache_history(false);
        // RequestCleanup drops immediately afterward and performs resource cleanup.
        self.observability.record_metrics();
    }
}

#[cfg(test)]
mod kv_cache_hit_tests {
    use super::*;

    fn report(reused: u64) -> serde_json::Value {
        serde_json::json!({
            "prompt_tokens": 100,
            "reused_tokens": reused,
        })
    }

    #[test]
    fn worker_values_may_exceed_router_estimate_and_prompt_length() {
        assert_eq!(worker_cache_hit_tokens(100, &report(135)), Some(135));
    }

    #[test]
    fn missing_invalid_or_mismatched_reports_are_rejected() {
        for value in [
            serde_json::json!({}),
            serde_json::json!({"prompt_tokens": 100}),
            serde_json::json!({"prompt_tokens": 99, "reused_tokens": 70}),
            serde_json::json!({"prompt_tokens": 100, "reused_tokens": -1}),
            serde_json::json!({"prompt_tokens": 100, "reused_tokens": null}),
        ] {
            assert_eq!(worker_cache_hit_tokens(100, &value), None);
        }
    }

    #[test]
    fn zero_reports_and_additive_extensions_are_accepted() {
        let mut value = report(0);
        value["tiers"] = serde_json::json!({"device": 0});
        value["lookup_tokens"] = 0.into();
        assert_eq!(worker_cache_hit_tokens(100, &value), Some(0));
        assert_eq!(
            worker_cache_hit_tokens(100, &report(u64::MAX)),
            Some(u64::MAX)
        );
    }
}

#[cfg(test)]
mod cache_history_tests {
    use super::*;
    use dynamo_kv_router::protocols::compute_seq_hash_for_block;

    fn direct_sequence_hashes(tokens: &[u32], block_size: u32, is_eagle: bool) -> Vec<u64> {
        let local_hashes = compute_block_hash_for_seq(
            tokens,
            block_size,
            BlockHashOptions {
                is_eagle: Some(is_eagle),
                ..Default::default()
            },
        );
        compute_seq_hash_for_block(&local_hashes)
    }

    fn tracker(prompt: &[u32], block_size: u32, is_eagle: bool) -> CanonicalOutputTracker {
        let parent = direct_sequence_hashes(prompt, block_size, is_eagle)
            .last()
            .copied();
        CanonicalOutputTracker::from_parts(prompt, None, block_size, is_eagle, None, None, parent)
    }

    #[test]
    fn completed_request_observes_once_and_records_membership() {
        let history = Arc::new(CacheHistory::with_capacity(4, 8));
        let mut tracking = Some(CacheHistoryTracking::new(history.clone(), vec![10], 8));

        let first = finalize_cache_history(&mut tracking, true).unwrap();
        assert_eq!(first.prompt_tokens, 8);
        assert_eq!(first.previously_seen_tokens, 0);
        assert!(first.retained.is_some());
        assert_eq!(history.previously_computed_tokens(&[10]), 8);
        assert!(finalize_cache_history(&mut tracking, true).is_none());
    }

    #[test]
    fn aborted_request_observes_once_without_recording_membership() {
        let history = Arc::new(CacheHistory::with_capacity(4, 8));
        let mut tracking = Some(CacheHistoryTracking::new(history.clone(), vec![10], 8));

        let first = finalize_cache_history(&mut tracking, false).unwrap();
        assert_eq!(first.prompt_tokens, 8);
        assert_eq!(first.previously_seen_tokens, 0);
        assert!(first.retained.is_none());
        assert_eq!(history.previously_computed_tokens(&[10]), 0);
        assert!(finalize_cache_history(&mut tracking, false).is_none());
    }

    #[test]
    fn previously_seen_prefix_is_measured_at_selection() {
        let history = Arc::new(CacheHistory::with_capacity(8, 16));
        history.record_completed([10, 20].into_iter());
        let tracking = CacheHistoryTracking::new(history, vec![10, 20, 30], 50);
        assert_eq!(tracking.previously_seen_tokens, 32);
    }

    #[test]
    fn streamed_chunks_complete_prompt_tail_and_extend_canonical_chain() {
        let mut tracker = tracker(&[1, 2, 3], 4, false);
        let mut completed = Vec::new();
        tracker.observe(0, &[4, 5], &mut completed);
        assert_eq!(completed.len(), 1);
        tracker.observe(0, &[6, 7, 8, 9], &mut completed);
        assert_eq!(
            completed,
            direct_sequence_hashes(&[1, 2, 3, 4, 5, 6, 7, 8], 4, false)
        );
    }

    #[test]
    fn generated_blocks_continue_an_aligned_prompt_chain() {
        let prompt = [1, 2, 3, 4];
        let mut tracker = tracker(&prompt, 4, false);
        let mut completed = Vec::new();
        // The newest sampled token has no KV yet, so a block needs one extra token.
        tracker.observe(0, &[5, 6, 7, 8], &mut completed);
        assert!(completed.is_empty());
        tracker.observe(0, &[9], &mut completed);
        assert_eq!(
            completed,
            direct_sequence_hashes(&[1, 2, 3, 4, 5, 6, 7, 8], 4, false)[1..]
        );
    }

    #[test]
    fn eagle_windows_match_routing_hashes() {
        let prompt = [1, 2, 3, 4, 5];
        let mut tracker = tracker(&prompt, 4, true);
        let mut completed = Vec::new();
        tracker.observe(0, &[6, 7, 8, 9], &mut completed);
        assert_eq!(
            completed,
            direct_sequence_hashes(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 4, true)[1..]
        );
    }

    #[test]
    fn multiple_choices_keep_independent_tails() {
        let prompt = [1, 2, 3, 4];
        let mut tracker = tracker(&prompt, 4, false);
        let mut completed = Vec::new();
        tracker.observe(0, &[5, 6, 7, 8, 9], &mut completed);
        tracker.observe(1, &[15, 16, 17, 18, 19], &mut completed);
        assert_eq!(
            completed,
            vec![
                direct_sequence_hashes(&[1, 2, 3, 4, 5, 6, 7, 8], 4, false)[1],
                direct_sequence_hashes(&[1, 2, 3, 4, 15, 16, 17, 18], 4, false)[1],
            ]
        );
    }

    #[test]
    fn incomplete_output_tail_is_not_materialized() {
        let mut tracker = tracker(&[1], 4, false);
        let mut completed = Vec::new();
        tracker.observe(0, &[2, 3], &mut completed);
        tracker.observe(0, &[], &mut completed);
        assert!(completed.is_empty());
    }
}
