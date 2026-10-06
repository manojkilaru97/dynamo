// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(any(test, feature = "bench"))]
use std::sync::Arc;
use std::{cell::Cell, collections::HashMap, sync::LazyLock};

#[cfg(any(test, feature = "bench"))]
use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use super::config::KvRouterConfig;
use super::filter::{RoutingEligibility, WorkerEligibilityError};
use super::policy_config::{
    RouterPolicyConfigError, WorkerSelectionPolicyKind, WorkerSelectionStage,
};
use super::two_tier_cost_fn::{self, TwoTierCostFn, TwoTierRow};
use super::types::{KvSchedulerError, SchedulingRequest};
use crate::protocols::complete_block_count;
use crate::protocols::{
    RoutingDecisionCandidate, RoutingDecisionTrace, TwoTierDecisionTrace, WorkerConfigLike,
    WorkerId, WorkerSelectionResult, WorkerWithDpRank,
};

/// A trait that users can implement to define custom selection logic.
///
/// Generic over `C` so that the scheduling layer does not depend on a concrete config type.
pub trait WorkerSelector<C: WorkerConfigLike> {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError>;
}

/// Opt-in routing-decision traces (upstream ai-dynamo/dynamo#14109). Disabled by default:
/// collecting the candidate table is diagnostic-only and adds no work when off.
pub const DYN_ROUTER_DECISION_TRACE_ENABLED: &str = "DYN_ROUTER_DECISION_TRACE_ENABLED";
/// Fraction of requests traced when enabled, in `[0, 1]`; unset means 1.0, invalid means 0.0.
pub const DYN_ROUTER_DECISION_TRACE_SAMPLE_RATE: &str = "DYN_ROUTER_DECISION_TRACE_SAMPLE_RATE";
const DECISION_TRACE_SCHEMA: &str = "dynamo.router.decision.v1";

/// Process-wide decision-trace setting, read once: `None` when disabled, otherwise the
/// sample rate. Selectors copy it at construction.
static ROUTER_DECISION_TRACE: LazyLock<Option<f64>> = LazyLock::new(|| {
    dynamo_truthy::env_is_truthy(DYN_ROUTER_DECISION_TRACE_ENABLED).then(|| {
        parse_decision_trace_sample_rate(
            std::env::var_os(DYN_ROUTER_DECISION_TRACE_SAMPLE_RATE)
                .map(|value| value.to_string_lossy().into_owned()),
        )
    })
});

fn parse_decision_trace_sample_rate(value: Option<String>) -> f64 {
    let Some(value) = value else {
        return 1.0;
    };
    match value.trim().parse::<f64>() {
        Ok(rate) if rate.is_finite() && (0.0..=1.0).contains(&rate) => rate,
        _ => {
            tracing::warn!(
                value = %value,
                "Ignoring invalid {DYN_ROUTER_DECISION_TRACE_SAMPLE_RATE}; expected 0 through 1"
            );
            0.0
        }
    }
}

/// Deterministic per-request sampling, so every hop of one request agrees.
fn sampled_request(request_id: &str, sample_rate: f64) -> bool {
    if sample_rate <= 0.0 {
        return false;
    }
    if sample_rate >= 1.0 {
        return true;
    }
    let sample = xxhash_rust::xxh3::xxh3_64(request_id.as_bytes()) as f64 / u64::MAX as f64;
    sample < sample_rate
}

/// Whether this selection should carry a decision trace. Pinned selections have no choice
/// to explain.
#[inline]
fn should_trace_decision(request: &SchedulingRequest, sample_rate: f64) -> bool {
    request.pinned_worker.is_none()
        && request
            .mode
            .request_id()
            .is_some_and(|request_id| sampled_request(request_id, sample_rate))
}

/// Mark the max-overlap candidate and assemble the trace around the policy-specific rows.
/// `policy_overlap` is the overlap the selecting policy ranked on, so `max_overlap` and the
/// avoidable prefill describe that policy's own view.
#[allow(clippy::too_many_arguments)]
fn finish_decision_trace(
    mut candidates: Vec<RoutingDecisionCandidate>,
    policy_overlap: fn(&RoutingDecisionCandidate) -> f64,
    config: &KvRouterConfig,
    worker_type: &'static str,
    policy: &str,
    selection_reason: String,
    request: &SchedulingRequest,
    block_size: u32,
    weights: LogitWeights,
    temperature: f64,
    selected: WorkerWithDpRank,
    two_tier: Option<TwoTierDecisionTrace>,
) -> Option<Box<RoutingDecisionTrace>> {
    candidates.sort_unstable_by_key(|candidate| (candidate.worker_id, candidate.dp_rank));
    let selected_candidate = candidates.iter().find(|candidate| candidate.selected)?;
    let selected_overlap = policy_overlap(selected_candidate);
    // On a tie the selected worker is the max-overlap worker, so ties never look like a loss.
    let max_overlap = candidates
        .iter()
        .max_by(|left, right| policy_overlap(left).total_cmp(&policy_overlap(right)))
        .filter(|max| policy_overlap(max) > selected_overlap)
        .unwrap_or(selected_candidate);
    let (max_overlap_worker_id, max_overlap_dp_rank, max_overlap_blocks) = (
        max_overlap.worker_id,
        max_overlap.dp_rank,
        policy_overlap(max_overlap),
    );
    for candidate in &mut candidates {
        candidate.max_overlap = candidate.worker_id == max_overlap_worker_id
            && candidate.dp_rank == max_overlap_dp_rank;
    }
    Some(Box::new(RoutingDecisionTrace {
        schema: DECISION_TRACE_SCHEMA.to_string(),
        worker_type: worker_type.to_owned(),
        policy: policy.to_owned(),
        selection_reason,
        candidate_scope: "eligible_workers_only".to_string(),
        block_size,
        request_blocks: request.request_blocks(block_size),
        track_prefill_tokens: request.track_prefill_tokens,
        selected_worker_id: selected.worker_id,
        selected_dp_rank: selected.dp_rank,
        max_overlap_worker_id,
        max_overlap_dp_rank,
        avoidable_prefill_token_equivalents: (max_overlap_blocks - selected_overlap)
            * block_size as f64,
        overlap_score_credit: weights.overlap_score_credit,
        overlap_score_credit_decay: weights.overlap_score_credit_decay,
        prefill_load_scale: weights.prefill_load_scale,
        host_cache_hit_weight: config.host_cache_hit_weight,
        disk_cache_hit_weight: config.disk_cache_hit_weight,
        shared_cache_multiplier: weights.shared_cache_multiplier,
        decode_active_request_weight: config.decode_active_request_weight,
        router_temperature: temperature,
        candidates,
        two_tier,
    }))
}

/// Best raw router-visible cached prefix among the eligible workers a selection visits
/// (cache-reuse funnel stage F2), tracked in the scan the selector already performs.
///
/// Only tracked requests record it; query-only selections report `None` so the router does
/// not count them. Raw means unweighted device + host-pinned (CPU offload) + disk blocks.
struct RawCacheReuse<'a> {
    request: &'a SchedulingRequest,
    block_size: u32,
    max: Cell<Option<usize>>,
}

impl<'a> RawCacheReuse<'a> {
    fn new(request: &'a SchedulingRequest, block_size: u32) -> Self {
        Self {
            request,
            block_size,
            max: Cell::new(request.mode.is_tracked().then_some(0)),
        }
    }

    /// Count an eligible worker rank toward the best eligible cached prefix.
    #[inline]
    fn track(&self, worker: WorkerWithDpRank) {
        if let Some(current) = self.max.get() {
            let raw = self.request.raw_cached_tokens_for(worker, self.block_size);
            self.max.set(Some(current.max(raw)));
        }
    }

    /// As [`Self::track`] for a caller that already summed the worker's raw tier blocks.
    #[inline]
    fn track_raw_blocks(&self, worker: WorkerWithDpRank, raw_blocks: usize) {
        if let Some(current) = self.max.get() {
            let raw = raw_blocks.saturating_mul(self.block_size as usize);
            debug_assert_eq!(
                raw,
                self.request.raw_cached_tokens_for(worker, self.block_size)
            );
            self.max.set(Some(current.max(raw)));
        }
    }

    /// As [`Self::track`] for a caller that already read the device and host-pinned tiers,
    /// so only the disk tier is looked up again.
    #[inline]
    fn track_with_device_and_host(
        &self,
        worker: WorkerWithDpRank,
        device_blocks: usize,
        host_blocks: usize,
    ) {
        if let Some(current) = self.max.get() {
            let disk_blocks = self
                .request
                .overlap
                .tier_overlap_blocks
                .disk
                .get(&worker)
                .copied()
                .unwrap_or(0);
            let raw = device_blocks
                .saturating_add(host_blocks)
                .saturating_add(disk_blocks)
                .saturating_mul(self.block_size as usize);
            debug_assert_eq!(
                raw,
                self.request.raw_cached_tokens_for(worker, self.block_size)
            );
            self.max.set(Some(current.max(raw)));
        }
    }

    /// `(max_raw_cached_tokens, selected_raw_cached_tokens)` for the chosen worker.
    fn finish(&self, selected: WorkerWithDpRank) -> (Option<usize>, Option<usize>) {
        let max = self.max.get();
        let selected_raw = max.map(|_| {
            self.request
                .raw_cached_tokens_for(selected, self.block_size)
        });
        (max, selected_raw)
    }
}

/// Helper function for softmax sampling.
/// Returns the selected worker and its logit.
fn softmax_sample(
    logits: &FxHashMap<WorkerWithDpRank, f64>,
    temperature: f64,
) -> (WorkerWithDpRank, f64) {
    softmax_sample_with_sample(logits, temperature, fastrand::f64())
}

fn softmax_sample_with_sample(
    logits: &FxHashMap<WorkerWithDpRank, f64>,
    temperature: f64,
    sample: f64,
) -> (WorkerWithDpRank, f64) {
    assert!(!logits.is_empty(), "Empty logits for softmax sampling");

    if temperature == 0.0 {
        let (worker, logit) = logits
            .iter()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .expect("logits non-empty");
        return (*worker, *logit);
    }

    let entries: Vec<(WorkerWithDpRank, f64)> = logits.iter().map(|(w, l)| (*w, *l)).collect();
    softmax_sample_entries(entries, temperature, sample)
}

fn softmax_sample_entries(
    entries: Vec<(WorkerWithDpRank, f64)>,
    temperature: f64,
    sample: f64,
) -> (WorkerWithDpRank, f64) {
    assert!(!entries.is_empty(), "Empty logits for softmax sampling");

    let (min_val, max_val) = entries
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (_, v)| {
            (lo.min(*v), hi.max(*v))
        });

    let mut probs = if min_val == max_val {
        vec![1.0 / entries.len() as f64; entries.len()]
    } else {
        // Negate logits and rescale to [−1/temperature, 0] for numerical stability
        // before softmax. Subtracting the max (which maps to min_val) keeps exp() inputs ≤ 0.
        let scale = -1.0 / ((max_val - min_val) * temperature);
        let max_scaled = min_val * scale;
        entries
            .iter()
            .map(|(_, v)| (v * scale - max_scaled).exp())
            .collect::<Vec<f64>>()
    };

    let sum: f64 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);

    let mut cumsum = 0.0;
    for (i, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if sample <= cumsum {
            return entries[i];
        }
    }

    *entries.last().unwrap()
}

/// Default implementation matching the Python _cost_function.
#[derive(Debug, Clone)]
pub struct DefaultWorkerSelector {
    pub kv_router_config: KvRouterConfig,
    pub worker_type: &'static str,
    /// Worker-selection policy chosen by `router_policy_config`'s `worker_selection` section.
    /// `None` keeps the built-in additive cost function.
    worker_selection_policy: Option<SelectedWorkerPolicy>,
    /// Routing-decision trace sample rate, or `None` when tracing is off (the default).
    decision_trace_sample_rate: Option<f64>,
    #[cfg(any(test, feature = "bench"))]
    deterministic_rng: Option<Arc<Mutex<fastrand::Rng>>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SelectedWorkerPolicy {
    /// `is_eagle` selects Eagle's shifted token windows when counting a request's complete blocks.
    TwoTierCostFn {
        policy: TwoTierCostFn,
        is_eagle: bool,
    },
}

#[derive(Debug, Clone, Copy)]
struct LogitWeights {
    overlap_score_credit: f64,
    overlap_score_credit_decay: f64,
    prefill_load_scale: f64,
    shared_cache_multiplier: f64,
}

/// Intermediates of the built-in cost for one worker rank (see `worker_score_parts`).
#[derive(Debug, Clone, Copy)]
struct WorkerScoreParts {
    effective_overlap_blocks: f64,
    device_overlap_blocks: f64,
    host_overlap_blocks: f64,
    disk_overlap_blocks: f64,
    shared_beyond: u32,
    raw_prefill_blocks: f64,
    active_prefill_tokens: usize,
    overlap_credit_decay: f64,
    effective_overlap_score_credit: f64,
    overlap_credit_blocks: f64,
    decode_cost_blocks: f64,
    active_requests: usize,
    active_request_cost_blocks: f64,
    adjusted_prefill_blocks: f64,
    prefill_cost_blocks: f64,
    decode_overlap_formula: bool,
    logit: f64,
    /// Unweighted device + host-pinned + disk blocks (the raw funnel prefix), from the same
    /// tier lookups as the cost.
    raw_cached_blocks: usize,
}

impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>, worker_type: &'static str) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
            worker_type,
            worker_selection_policy: None,
            decision_trace_sample_rate: *ROUTER_DECISION_TRACE,
            #[cfg(any(test, feature = "bench"))]
            deterministic_rng: None,
        }
    }

    /// Build the selector for one worker pool, applying the worker-selection policy that
    /// `router_policy_config` selects for `stage` (upstream `worker_selection.<stage>`).
    ///
    /// `is_eagle` must match the router's block hashing, so the policy's cache ratio counts the
    /// same complete blocks the indexer matches.
    pub fn for_stage(
        kv_router_config: Option<KvRouterConfig>,
        worker_type: &'static str,
        stage: WorkerSelectionStage,
        is_eagle: bool,
    ) -> Result<Self, RouterPolicyConfigError> {
        let mut selector = Self::new(kv_router_config, worker_type);
        selector.worker_selection_policy = selector
            .kv_router_config
            .worker_selection_policy(stage)?
            .map(|kind| match kind {
                WorkerSelectionPolicyKind::TwoTierCostFn(parameters) => {
                    SelectedWorkerPolicy::TwoTierCostFn {
                        policy: TwoTierCostFn::new(
                            parameters,
                            selector.kv_router_config.host_cache_hit_weight,
                        ),
                        is_eagle,
                    }
                }
            });
        if selector.worker_selection_policy.is_none()
            && selector
                .kv_router_config
                .worker_selection_policy_with_env(stage, |_| None)
                .ok()
                .flatten()
                .is_some()
        {
            tracing::warn!(
                worker_type,
                stage = stage.as_str(),
                "worker-selection policy override selects the built-in selector over router_policy_config"
            );
        }
        if let Some(SelectedWorkerPolicy::TwoTierCostFn { policy, .. }) =
            selector.worker_selection_policy
        {
            tracing::info!(
                worker_type,
                stage = stage.as_str(),
                policy = two_tier_cost_fn::POLICY_TYPE,
                cache_threshold = policy.parameters.cache_threshold,
                balance_abs_threshold = policy.parameters.balance_abs_threshold,
                balance_rel_threshold = policy.parameters.balance_rel_threshold,
                host_cache_weight = policy.host_cache_weight,
                "Using worker-selection policy"
            );
        }
        Ok(selector)
    }

    /// Override the process-wide decision-trace setting (`None` disables tracing).
    #[cfg(test)]
    fn with_decision_trace_sample_rate(mut self, sample_rate: Option<f64>) -> Self {
        self.decision_trace_sample_rate = sample_rate;
        self
    }

    /// The shipped policy type this selector runs, or `None` for the built-in cost function.
    pub fn worker_selection_policy_type(&self) -> Option<&'static str> {
        self.worker_selection_policy.map(|policy| match policy {
            SelectedWorkerPolicy::TwoTierCostFn { .. } => two_tier_cost_fn::POLICY_TYPE,
        })
    }

    /// Select among eligible workers with the two-tier cost function.
    ///
    /// One pass over the eligible ranks, with no allocation: device overlap is the indexed device
    /// prefix (0.0 when absent, as upstream supplies it), host overlap the host-pinned
    /// continuation, and load the router's active-request count.
    #[allow(clippy::too_many_arguments)]
    fn select_two_tier<C: WorkerConfigLike>(
        &self,
        policy: &TwoTierCostFn,
        is_eagle: bool,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        block_size: u32,
        weights: LogitWeights,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        let tiers = &request.overlap.tier_overlap_blocks;
        let mut candidates = 0usize;
        let mut accumulator = policy.accumulator::<WorkerWithDpRank>();
        let raw_reuse = RawCacheReuse::new(request, block_size);
        eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
            candidates += 1;
            let device_blocks = tiers.device.get(&worker).copied().unwrap_or(0);
            let host_blocks = tiers.host_pinned.get(&worker).copied().unwrap_or(0);
            raw_reuse.track_with_device_and_host(worker, device_blocks, host_blocks);
            accumulator.push(
                worker,
                TwoTierRow {
                    device_overlap_blocks: device_blocks as f64,
                    host_overlap_blocks: host_blocks as f64,
                    active_requests: request.worker_load_for(worker).active_requests,
                },
            );
        });

        let request_blocks = request.request_blocks(block_size);
        let matchable_blocks =
            complete_block_count(request.isl_tokens, block_size, is_eagle) as u64;
        let Some((worker, selected, decision)) = accumulator.finish(matchable_blocks) else {
            return Err(KvSchedulerError::NoEndpoints);
        };
        let effective_overlap_blocks = request.effective_overlap_blocks_for(worker);
        let total_kv_blocks = workers
            .get(&worker.worker_id)
            .and_then(|cfg| cfg.total_kv_blocks());
        tracing::info!(
            router_mode = "kv",
            request_id = request.mode.request_id().unwrap_or("-"),
            worker_id = worker.worker_id,
            worker_type = %self.worker_type,
            dp_rank = ?worker.dp_rank,
            policy = two_tier_cost_fn::POLICY_TYPE,
            tier = decision.as_str(),
            candidates,
            request_blocks,
            matchable_blocks,
            device_blocks = selected.device_overlap_blocks,
            host_pinned_blocks = selected.host_overlap_blocks,
            active_requests = selected.active_requests,
            effective_cached_blocks = effective_overlap_blocks,
            total_kv_blocks = ?total_kv_blocks,
            "Selected worker"
        );

        let (max_raw_cached_tokens, selected_raw_cached_tokens) = raw_reuse.finish(worker);
        let decision_trace = self
            .decision_trace_sample_rate
            .is_some_and(|rate| should_trace_decision(request, rate))
            .then(|| {
                self.two_tier_decision_trace(
                    policy,
                    workers,
                    request,
                    eligibility,
                    block_size,
                    weights,
                    matchable_blocks,
                    decision.as_str(),
                    worker,
                )
            })
            .flatten();
        Ok(WorkerSelectionResult {
            worker,
            required_blocks: request_blocks,
            effective_overlap_blocks,
            cached_tokens: request.effective_cached_tokens_for(worker),
            max_raw_cached_tokens,
            selected_raw_cached_tokens,
            potential_decode_blocks: request
                .potential_decode_blocks_after_admission(worker, block_size),
            decision_trace,
        })
    }

    /// Candidate table for a sampled two-tier selection. Built only after selection, from the
    /// same tier inputs the policy ranked.
    #[allow(clippy::too_many_arguments)]
    fn two_tier_decision_trace<C: WorkerConfigLike>(
        &self,
        policy: &TwoTierCostFn,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        block_size: u32,
        weights: LogitWeights,
        matchable_blocks: u64,
        tier: &str,
        selected: WorkerWithDpRank,
    ) -> Option<Box<RoutingDecisionTrace>> {
        let tiers = &request.overlap.tier_overlap_blocks;
        let mut candidates = Vec::new();
        eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
            let row = TwoTierRow {
                device_overlap_blocks: tiers.device.get(&worker).copied().unwrap_or(0) as f64,
                host_overlap_blocks: tiers.host_pinned.get(&worker).copied().unwrap_or(0) as f64,
                active_requests: request.worker_load_for(worker).active_requests,
            };
            let load = request.worker_load_for(worker);
            candidates.push(RoutingDecisionCandidate {
                worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                eligible: true,
                selected: worker == selected,
                max_overlap: false,
                total_cost_blocks: 0.0,
                effective_overlap_blocks: request.effective_overlap_blocks_for(worker),
                device_overlap_blocks: row.device_overlap_blocks,
                host_overlap_blocks: row.host_overlap_blocks,
                disk_overlap_blocks: tiers.disk.get(&worker).copied().unwrap_or(0) as f64,
                shared_beyond_device_blocks: 0,
                raw_prefill_blocks: 0.0,
                active_prefill_tokens: load.active_prefill_tokens,
                prefill_cost_blocks: 0.0,
                decode_cost_blocks: load.potential_decode_blocks() as f64,
                active_requests: row.active_requests,
                active_request_cost_blocks: 0.0,
                overlap_credit_blocks: 0.0,
                overlap_credit_decay: 1.0,
                effective_overlap_score_credit: 0.0,
                adjusted_prefill_blocks: 0.0,
                base_score_blocks: 0.0,
                preferred_taint_multiplier: None,
                decode_overlap_formula: false,
                raw_cached_tokens: request.raw_cached_tokens_for(worker, block_size),
                two_tier_overlap_blocks: Some(policy.effective_overlap_blocks(&row)),
            });
        });
        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(self.kv_router_config.router_temperature);
        finish_decision_trace(
            candidates,
            |candidate| {
                candidate
                    .two_tier_overlap_blocks
                    .unwrap_or(candidate.effective_overlap_blocks)
            },
            &self.kv_router_config,
            self.worker_type,
            two_tier_cost_fn::POLICY_TYPE,
            format!("two_tier_{tier}"),
            request,
            block_size,
            weights,
            temperature,
            selected,
            Some(TwoTierDecisionTrace {
                tier: tier.to_string(),
                cache_threshold: policy.parameters.cache_threshold,
                balance_abs_threshold: policy.parameters.balance_abs_threshold,
                balance_rel_threshold: policy.parameters.balance_rel_threshold,
                host_cache_weight: policy.host_cache_weight,
                matchable_blocks,
            }),
        )
    }

    /// Candidate table for a sampled built-in selection: every eligible rank's score parts,
    /// computed by the same function the selection used.
    #[allow(clippy::too_many_arguments)]
    fn default_decision_trace<C: WorkerConfigLike>(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        block_size: u32,
        min_active_prefill_tokens: usize,
        weights: LogitWeights,
        temperature: f64,
        selected: WorkerWithDpRank,
    ) -> Option<Box<RoutingDecisionTrace>> {
        let mut candidates = Vec::new();
        eligibility.for_each_eligible_worker_rank(workers, |worker, config| {
            let parts = self.worker_score_parts(
                request,
                worker,
                block_size,
                min_active_prefill_tokens,
                weights,
            );
            let preferred_taint_multiplier = request
                .routing_constraints
                .preferred_taint_multiplier(config.taints());
            candidates.push(RoutingDecisionCandidate {
                worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                eligible: true,
                selected: worker == selected,
                max_overlap: false,
                total_cost_blocks: preferred_taint_multiplier
                    .map_or(parts.logit, |multiplier| parts.logit * multiplier),
                effective_overlap_blocks: parts.effective_overlap_blocks,
                device_overlap_blocks: parts.device_overlap_blocks,
                host_overlap_blocks: parts.host_overlap_blocks,
                disk_overlap_blocks: parts.disk_overlap_blocks,
                shared_beyond_device_blocks: parts.shared_beyond,
                raw_prefill_blocks: parts.raw_prefill_blocks,
                active_prefill_tokens: parts.active_prefill_tokens,
                prefill_cost_blocks: parts.prefill_cost_blocks,
                decode_cost_blocks: parts.decode_cost_blocks,
                active_requests: parts.active_requests,
                active_request_cost_blocks: parts.active_request_cost_blocks,
                overlap_credit_blocks: parts.overlap_credit_blocks,
                overlap_credit_decay: parts.overlap_credit_decay,
                effective_overlap_score_credit: parts.effective_overlap_score_credit,
                adjusted_prefill_blocks: parts.adjusted_prefill_blocks,
                base_score_blocks: parts.logit,
                preferred_taint_multiplier,
                decode_overlap_formula: parts.decode_overlap_formula,
                raw_cached_tokens: request.raw_cached_tokens_for(worker, block_size),
                two_tier_overlap_blocks: None,
            });
        });
        finish_decision_trace(
            candidates,
            |candidate| candidate.effective_overlap_blocks,
            &self.kv_router_config,
            self.worker_type,
            "default",
            if temperature == 0.0 {
                "minimum_cost".to_string()
            } else {
                "temperature_sample".to_string()
            },
            request,
            block_size,
            weights,
            temperature,
            selected,
            None,
        )
    }

    #[cfg(any(test, feature = "bench"))]
    pub fn new_seeded(
        kv_router_config: Option<KvRouterConfig>,
        worker_type: &'static str,
        seed: u64,
    ) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
            worker_type,
            worker_selection_policy: None,
            decision_trace_sample_rate: *ROUTER_DECISION_TRACE,
            deterministic_rng: Some(Arc::new(Mutex::new(fastrand::Rng::with_seed(seed)))),
        }
    }

    /// Every intermediate of the built-in additive cost for one worker rank. `worker_logit`
    /// returns `logit`; decision traces report the parts, so the two cannot diverge.
    #[inline(always)]
    fn worker_score_parts(
        &self,
        request: &SchedulingRequest,
        worker: WorkerWithDpRank,
        block_size: u32,
        min_active_prefill_tokens: usize,
        weights: LogitWeights,
    ) -> WorkerScoreParts {
        let block_size_f64 = block_size as f64;
        let effective_overlap_blocks = request.effective_overlap_blocks_for(worker);
        let has_tier_overlap_blocks = !request.overlap.tier_overlap_blocks.device.is_empty()
            || !request.overlap.tier_overlap_blocks.host_pinned.is_empty()
            || !request.overlap.tier_overlap_blocks.disk.is_empty();
        let device_tier_blocks = request
            .overlap
            .tier_overlap_blocks
            .device
            .get(&worker)
            .copied();
        let device_overlap_blocks = device_tier_blocks
            .map(|blocks| blocks as f64)
            .unwrap_or_else(|| {
                if has_tier_overlap_blocks {
                    0.0
                } else {
                    effective_overlap_blocks
                }
            });
        // `shared_cache_hits::hits_beyond` expects an integer block count, so
        // use the unweighted device prefix depth for this comparison.
        let device_overlap_blocks_u32 = device_overlap_blocks.round().max(0.0) as u32;
        let worker_load = request.worker_loads.get(&worker).copied();
        let raw_prefill_tokens = if request.track_prefill_tokens {
            match worker_load {
                Some(load) => {
                    let cached_tokens = request.effective_cached_tokens_for(worker);
                    // Preserve the legacy operation order when overlap exceeds the prompt.
                    let uncached_tokens = super::prefill_load::effective_prefill_tokens(
                        request.isl_tokens,
                        cached_tokens,
                    );
                    let projected_tokens = load.active_prefill_tokens + uncached_tokens;
                    projected_tokens.saturating_add(cached_tokens)
                }
                None => request.isl_tokens,
            }
        } else {
            0
        } as f64;

        let host_tier_blocks = request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .get(&worker)
            .copied()
            .unwrap_or(0);
        let disk_tier_blocks = request
            .overlap
            .tier_overlap_blocks
            .disk
            .get(&worker)
            .copied()
            .unwrap_or(0);
        let host_overlap_blocks = host_tier_blocks as f64;
        let disk_overlap_blocks = disk_tier_blocks as f64;
        let raw_cached_blocks = device_tier_blocks
            .unwrap_or(0)
            .saturating_add(host_tier_blocks)
            .saturating_add(disk_tier_blocks);

        // Credit shared cache hits beyond this worker's device prefix.
        let (shared_overlap_blocks, shared_beyond) =
            if let Some(ref shared_hits) = request.shared_cache_hits {
                let beyond = shared_hits.hits_beyond(device_overlap_blocks_u32);
                (weights.shared_cache_multiplier * (beyond as f64), beyond)
            } else {
                (0.0, 0)
            };

        let raw_prefill_blocks = raw_prefill_tokens / block_size_f64;
        let active_prefill_tokens = worker_load.unwrap_or_default().active_prefill_tokens;
        // Normalize backlog above the least-loaded eligible worker by this request's
        // size. The rational decay softly trades cache locality for prefill balance,
        // while leaving workers at the load floor with their full device credit.
        let overlap_credit_decay =
            if request.track_prefill_tokens && weights.overlap_score_credit_decay > 0.0 {
                let excess_active_prefill_blocks =
                    active_prefill_tokens.saturating_sub(min_active_prefill_tokens) as f64
                        / block_size_f64;
                let normalized_prefill_load =
                    excess_active_prefill_blocks / request.request_blocks(block_size) as f64;
                1.0 / (1.0 + weights.overlap_score_credit_decay * normalized_prefill_load)
            } else {
                1.0
            };
        let effective_overlap_score_credit = weights.overlap_score_credit * overlap_credit_decay;
        let overlap_credit_blocks = effective_overlap_score_credit * device_overlap_blocks
            + self.kv_router_config.host_cache_hit_weight * host_overlap_blocks
            + self.kv_router_config.disk_cache_hit_weight * disk_overlap_blocks
            + shared_overlap_blocks;
        let worker_load = worker_load.unwrap_or_default();
        let decode_cost_blocks = worker_load.potential_decode_blocks() as f64;
        let active_requests = worker_load.active_requests;
        let active_request_cost_blocks =
            self.kv_router_config.decode_active_request_weight * active_requests as f64;

        // Decode routers normally force `overlap_score_credit=0` through the
        // per-request override, which preserves load-only disagg routing. When
        // conditional disagg leaves a positive overlap credit in place, prefer
        // cache-hot decode workers while still charging decode backlog.
        let decode_overlap_formula = self.worker_type == "decode"
            && !request.track_prefill_tokens
            && weights.overlap_score_credit > 0.0;
        let (adjusted_prefill_blocks, prefill_cost_blocks, logit) = if decode_overlap_formula {
            // Clamp at zero because downstream taint multipliers assume non-negative scores.
            // This loses ordering between workers whose overlap fully offsets decode load, but
            // avoids inverting taint preference among negative-score workers.
            let overlap_adjusted_decode_blocks =
                (decode_cost_blocks - overlap_credit_blocks).max(0.0);
            (
                0.0,
                0.0,
                overlap_adjusted_decode_blocks + active_request_cost_blocks,
            )
        } else {
            let adjusted_prefill_blocks = (raw_prefill_blocks - overlap_credit_blocks).max(0.0);
            let prefill_cost_blocks = weights.prefill_load_scale * adjusted_prefill_blocks;
            (
                adjusted_prefill_blocks,
                prefill_cost_blocks,
                prefill_cost_blocks + decode_cost_blocks + active_request_cost_blocks,
            )
        };

        WorkerScoreParts {
            effective_overlap_blocks,
            device_overlap_blocks,
            host_overlap_blocks,
            disk_overlap_blocks,
            shared_beyond,
            raw_prefill_blocks,
            active_prefill_tokens,
            overlap_credit_decay,
            effective_overlap_score_credit,
            overlap_credit_blocks,
            decode_cost_blocks,
            active_requests,
            active_request_cost_blocks,
            adjusted_prefill_blocks,
            prefill_cost_blocks,
            decode_overlap_formula,
            logit,
            raw_cached_blocks,
        }
    }

    fn worker_logit(
        &self,
        request: &SchedulingRequest,
        worker: WorkerWithDpRank,
        block_size: u32,
        min_active_prefill_tokens: usize,
        weights: LogitWeights,
        formula_name: &'static str,
    ) -> f64 {
        self.logged_worker_score(
            request,
            worker,
            block_size,
            min_active_prefill_tokens,
            weights,
            formula_name,
        )
        .logit
    }

    /// Score one worker rank and emit the formula debug row; returns every score part.
    fn logged_worker_score(
        &self,
        request: &SchedulingRequest,
        worker: WorkerWithDpRank,
        block_size: u32,
        min_active_prefill_tokens: usize,
        weights: LogitWeights,
        formula_name: &'static str,
    ) -> WorkerScoreParts {
        let parts = self.worker_score_parts(
            request,
            worker,
            block_size,
            min_active_prefill_tokens,
            weights,
        );
        let WorkerScoreParts {
            effective_overlap_blocks,
            shared_beyond,
            raw_prefill_blocks,
            overlap_credit_decay,
            overlap_credit_blocks,
            decode_cost_blocks,
            active_request_cost_blocks,
            adjusted_prefill_blocks,
            decode_overlap_formula,
            logit,
            ..
        } = parts;

        if decode_overlap_formula {
            // Stamped for the same reason as the two rows below: this row is emitted from the
            // `SchedulerQueueActor` task, so the logging layer cannot attach request identity to
            // it.
            tracing::debug!(
                request_id = request.mode.request_id().unwrap_or("-"),
                worker_type = self.worker_type,
                "{formula_name} for worker_id={} dp_rank={:?} with {effective_overlap_blocks:.2} effective cached blocks: {logit:.3} \
                 = max(0, decode_blocks - overlap_credit_blocks) + active_request_cost_blocks \
                 = max(0, {decode_cost_blocks:.3} - {overlap_credit_blocks:.3}) + {active_request_cost_blocks:.3}",
                worker.worker_id,
                worker.dp_rank,
            );
            return parts;
        }

        // These rows are emitted from the `SchedulerQueueActor` task, which `scheduling::queue`
        // spawns without the caller's request span, so the logging layer cannot attach
        // `x_request_id`/`trace_id` to them. Stamp the identity the row needs to be self-joining:
        // `request_id` is the same value `[ROUTING] Best` logs, and `worker_type` separates the
        // prefill-pool and decode-pool decisions that interleave into one log. Both are evaluated
        // inside the macro so they cost nothing when DEBUG is disabled.
        if shared_beyond > 0 {
            tracing::debug!(
                request_id = request.mode.request_id().unwrap_or("-"),
                worker_type = self.worker_type,
                "{formula_name} for worker_id={} dp_rank={:?} with {effective_overlap_blocks:.2} effective cached blocks, \
                 {shared_beyond} shared blocks beyond device (multiplier={shared_cache_multiplier:.2}): {logit:.3} \
                 = prefill_load_scale * adjusted_prefill_blocks + decode_blocks + active_request_cost_blocks \
                 = {prefill_load_scale:.3} * {adjusted_prefill_blocks:.3} + {decode_cost_blocks:.3} + {active_request_cost_blocks:.3} \
                 (raw_prefill_blocks: {raw_prefill_blocks:.3}, overlap_credit_blocks: {overlap_credit_blocks:.3}, \
                 overlap_credit_decay: {overlap_credit_decay:.3})",
                worker.worker_id,
                worker.dp_rank,
                shared_cache_multiplier = weights.shared_cache_multiplier,
                prefill_load_scale = weights.prefill_load_scale
            );
        } else {
            tracing::debug!(
                request_id = request.mode.request_id().unwrap_or("-"),
                worker_type = self.worker_type,
                "{formula_name} for worker_id={} dp_rank={:?} with {effective_overlap_blocks:.2} effective cached blocks: {logit:.3} \
                 = prefill_load_scale * adjusted_prefill_blocks + decode_blocks + active_request_cost_blocks \
                 = {prefill_load_scale:.3} * {adjusted_prefill_blocks:.3} + {decode_cost_blocks:.3} + {active_request_cost_blocks:.3} \
                 (raw_prefill_blocks: {raw_prefill_blocks:.3}, overlap_credit_blocks: {overlap_credit_blocks:.3}, \
                 overlap_credit_decay: {overlap_credit_decay:.3})",
                worker.worker_id,
                worker.dp_rank,
                prefill_load_scale = weights.prefill_load_scale
            );
        }

        parts
    }
}

impl<C: WorkerConfigLike> WorkerSelector<C> for DefaultWorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        assert!(request.isl_tokens > 0);
        eligibility.validate_pinned_worker_allowed()?;

        let pinned_worker = eligibility.pinned_worker();

        if pinned_worker.is_none()
            && !eligibility.has_eligible_worker(
                workers
                    .iter()
                    .map(|(&worker_id, config)| (worker_id, config)),
            )
        {
            if eligibility.has_eligible_worker_ignoring_overload(
                workers
                    .iter()
                    .map(|(&worker_id, config)| (worker_id, config)),
            ) {
                return Err(KvSchedulerError::AllEligibleWorkersOverloaded);
            }

            return Err(KvSchedulerError::NoEndpoints);
        }

        let request_blocks = request.request_blocks(block_size);
        // Borrowed, never allocated; bridges the winner row to `[ROUTING] Best`.
        let request_id = request.mode.request_id().unwrap_or("-");

        let weights = LogitWeights {
            overlap_score_credit: request
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.overlap_score_credit)
                .unwrap_or(self.kv_router_config.overlap_score_credit),
            overlap_score_credit_decay: self.kv_router_config.overlap_score_credit_decay,
            prefill_load_scale: request
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.prefill_load_scale)
                .unwrap_or(self.kv_router_config.prefill_load_scale),
            shared_cache_multiplier: request
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.shared_cache_multiplier)
                .unwrap_or(self.kv_router_config.shared_cache_multiplier),
        };

        if let Some(worker) = pinned_worker {
            match eligibility.validate_worker_rank(workers, worker) {
                Ok(_) => {}
                Err(WorkerEligibilityError::WorkerOverloaded { .. }) => {
                    return Err(KvSchedulerError::PinnedWorkerOverloaded {
                        worker_id: worker.worker_id,
                    });
                }
                Err(_) => return Err(KvSchedulerError::NoEndpoints),
            }

            let min_active_prefill_tokens = request.worker_load_for(worker).active_prefill_tokens;
            let logit = self.worker_logit(
                request,
                worker,
                block_size,
                min_active_prefill_tokens,
                weights,
                "Pinned formula",
            );
            let effective_overlap_blocks = request.effective_overlap_blocks_for(worker);
            let cached_tokens = request.effective_cached_tokens_for(worker);

            tracing::info!(
                request_id,
                "Selected pinned worker: worker_type={}, worker_id={} dp_rank={:?}, logit: {:.3}, effective cached blocks: {:.2}",
                self.worker_type,
                worker.worker_id,
                worker.dp_rank,
                logit,
                effective_overlap_blocks,
            );

            // A pin narrows the eligible set to the pinned rank, so the best eligible
            // cached prefix is the pinned worker's own.
            let raw_reuse = RawCacheReuse::new(request, block_size);
            raw_reuse.track(worker);
            let (max_raw_cached_tokens, selected_raw_cached_tokens) = raw_reuse.finish(worker);
            return Ok(WorkerSelectionResult {
                worker,
                required_blocks: request_blocks,
                effective_overlap_blocks,
                cached_tokens,
                max_raw_cached_tokens,
                selected_raw_cached_tokens,
                potential_decode_blocks: request
                    .potential_decode_blocks_after_admission(worker, block_size),
                decision_trace: None,
            });
        }

        if let Some(SelectedWorkerPolicy::TwoTierCostFn { policy, is_eagle }) =
            &self.worker_selection_policy
        {
            return self.select_two_tier(
                policy,
                *is_eagle,
                workers,
                request,
                eligibility,
                block_size,
                weights,
            );
        }

        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(self.kv_router_config.router_temperature);
        let min_active_prefill_tokens =
            if request.track_prefill_tokens && weights.overlap_score_credit_decay > 0.0 {
                let mut minimum = usize::MAX;
                eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
                    minimum = minimum.min(request.worker_load_for(worker).active_prefill_tokens);
                });
                debug_assert_ne!(minimum, usize::MAX);
                minimum
            } else {
                0
            };
        // Every scan below scores each eligible rank through `get_score`, so tracking the
        // raw cached prefix there needs no extra pass. `max` is idempotent if a rank repeats.
        let raw_reuse = RawCacheReuse::new(request, block_size);
        let get_score = |worker: WorkerWithDpRank| -> f64 {
            let parts = self.logged_worker_score(
                request,
                worker,
                block_size,
                min_active_prefill_tokens,
                weights,
                "Formula",
            );
            raw_reuse.track_raw_blocks(worker, parts.raw_cached_blocks);
            let base_score = parts.logit;
            let Some(config) = workers.get(&worker.worker_id) else {
                return base_score;
            };
            match request
                .routing_constraints
                .preferred_taint_multiplier(config.taints())
            {
                // NOTE: This multiplicative bias assumes a non-negative score. Negative
                // overlap scores expose its pre-existing sign sensitivity; keep it for now.
                Some(multiplier) => base_score * multiplier,
                None => base_score,
            }
        };

        #[cfg(any(test, feature = "bench"))]
        let deterministic_choice = self.deterministic_rng.as_ref().map(|rng| {
            let mut candidates = Vec::new();
            eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
                candidates.push(worker);
            });
            candidates.sort_unstable_by_key(|worker| (worker.worker_id, worker.dp_rank));

            let mut rng = rng.lock();
            if temperature == 0.0 {
                let mut best_worker = None;
                let mut best_logit = f64::INFINITY;
                let mut tie_count = 0usize;
                for worker in candidates {
                    let score = get_score(worker);
                    if score < best_logit {
                        best_worker = Some(worker);
                        best_logit = score;
                        tie_count = 1;
                        continue;
                    }

                    if score == best_logit {
                        tie_count += 1;
                        if rng.usize(0..tie_count) == 0 {
                            best_worker = Some(worker);
                        }
                    }
                }
                return (
                    best_worker.expect("eligible worker rank non-empty"),
                    best_logit,
                );
            }

            let entries = candidates
                .into_iter()
                .map(|worker| (worker, get_score(worker)))
                .collect();
            softmax_sample_entries(entries, temperature, rng.f64())
        });

        let random_choice = || {
            if temperature == 0.0 {
                let mut best_worker = None;
                let mut best_logit = f64::INFINITY;
                let mut tie_count = 0usize;
                eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
                    let score = get_score(worker);
                    if score < best_logit {
                        best_worker = Some(worker);
                        best_logit = score;
                        tie_count = 1;
                        return;
                    }

                    if score == best_logit {
                        tie_count += 1;
                        // Reservoir sampling keeps tied minima uniform without collecting workers.
                        if fastrand::usize(0..tie_count) == 0 {
                            best_worker = Some(worker);
                        }
                    }
                });

                return (
                    best_worker.expect("eligible worker rank non-empty"),
                    best_logit,
                );
            }

            let mut worker_logits = FxHashMap::default();
            eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
                let score = get_score(worker);
                worker_logits.insert(worker, score);
            });

            softmax_sample(&worker_logits, temperature)
        };
        #[cfg(any(test, feature = "bench"))]
        let (best_worker, best_logit) = deterministic_choice.unwrap_or_else(random_choice);
        #[cfg(not(any(test, feature = "bench")))]
        let (best_worker, best_logit) = random_choice();

        let decision_trace = self
            .decision_trace_sample_rate
            .is_some_and(|rate| should_trace_decision(request, rate))
            .then(|| {
                self.default_decision_trace(
                    workers,
                    request,
                    eligibility,
                    block_size,
                    min_active_prefill_tokens,
                    weights,
                    temperature,
                    best_worker,
                )
            })
            .flatten();

        let best_host_pinned_overlap_blocks = request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .get(&best_worker)
            .copied()
            .unwrap_or(0);
        let best_disk_overlap_blocks = request
            .overlap
            .tier_overlap_blocks
            .disk
            .get(&best_worker)
            .copied()
            .unwrap_or(0);

        if self.worker_type == "decode" {
            let effective_overlap_blocks = request.effective_overlap_blocks_for(best_worker);
            let cached_tokens = request.effective_cached_tokens_for(best_worker);
            tracing::info!(
                router_mode = "kv",
                request_id,
                worker_id = best_worker.worker_id,
                worker_type = %self.worker_type,
                dp_rank = ?best_worker.dp_rank,
                logit = best_logit,
                host_pinned_blocks = best_host_pinned_overlap_blocks,
                disk_blocks = best_disk_overlap_blocks,
                "Selected worker"
            );

            let (max_raw_cached_tokens, selected_raw_cached_tokens) = raw_reuse.finish(best_worker);
            return Ok(WorkerSelectionResult {
                worker: best_worker,
                required_blocks: request_blocks,
                effective_overlap_blocks,
                cached_tokens,
                max_raw_cached_tokens,
                selected_raw_cached_tokens,
                potential_decode_blocks: request
                    .potential_decode_blocks_after_admission(best_worker, block_size),
                decision_trace,
            });
        }

        let best_overlap = request.effective_overlap_blocks_for(best_worker);
        let best_cached_tokens = request.effective_cached_tokens_for(best_worker);

        let total_kv_blocks = workers
            .get(&best_worker.worker_id)
            .and_then(|cfg| cfg.total_kv_blocks());

        tracing::info!(
            router_mode = "kv",
            request_id,
            worker_id = best_worker.worker_id,
            worker_type = %self.worker_type,
            dp_rank = ?best_worker.dp_rank,
            logit = best_logit,
            effective_cached_blocks = best_overlap,
            host_pinned_blocks = best_host_pinned_overlap_blocks,
            disk_blocks = best_disk_overlap_blocks,
            total_kv_blocks = ?total_kv_blocks,
            "Selected worker"
        );

        let (max_raw_cached_tokens, selected_raw_cached_tokens) = raw_reuse.finish(best_worker);
        Ok(WorkerSelectionResult {
            worker: best_worker,
            required_blocks: request_blocks,
            effective_overlap_blocks: best_overlap,
            cached_tokens: best_cached_tokens,
            max_raw_cached_tokens,
            selected_raw_cached_tokens,
            potential_decode_blocks: request
                .potential_decode_blocks_after_admission(best_worker, block_size),
            decision_trace,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::protocols::{SharedCacheHits, WorkerConfigLike};
    use crate::scheduling::{OverlapSignals, ScheduleMode};

    #[derive(Clone, Default)]
    struct TaintedWorkerConfig {
        taints: HashSet<String>,
    }

    impl WorkerConfigLike for TaintedWorkerConfig {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            None
        }

        fn taints(&self) -> &HashSet<String> {
            &self.taints
        }
    }

    fn base_request(isl_tokens: usize) -> SchedulingRequest {
        SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        }
    }

    fn worker_loads_with_active_decode(
        decode_blocks: FxHashMap<WorkerWithDpRank, usize>,
    ) -> FxHashMap<WorkerWithDpRank, crate::sequences::WorkerLoadProjection> {
        decode_blocks
            .into_iter()
            .map(|(worker, active_decode_blocks)| {
                (
                    worker,
                    crate::sequences::WorkerLoadProjection {
                        active_decode_blocks,
                        ..Default::default()
                    },
                )
            })
            .collect()
    }

    #[test]
    fn test_softmax_sample_single_key() {
        let mut logits = FxHashMap::default();
        let worker = WorkerWithDpRank::from_worker_id(42);
        for (logit, temperature) in [
            (0.5, 0.1),
            (0.5, 1.0),
            (0.5, 10.0),
            (-100.0, 1.0),
            (100.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ] {
            logits.clear();
            logits.insert(worker, logit);

            let result = softmax_sample(&logits, temperature);
            assert_eq!(result.0, worker, "Should return the only available worker");
            assert_eq!(result.1, logit, "Should return the selected worker's logit");
        }
    }

    #[test]
    fn test_softmax_sample_zero_temperature() {
        let mut logits = FxHashMap::default();
        let worker1 = WorkerWithDpRank::from_worker_id(1);
        let worker2 = WorkerWithDpRank::from_worker_id(2);
        let worker3 = WorkerWithDpRank::from_worker_id(3);
        let worker4 = WorkerWithDpRank::from_worker_id(4);
        logits.insert(worker1, 5.0);
        logits.insert(worker2, 3.0);
        logits.insert(worker3, 7.0);
        logits.insert(worker4, 3.5);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(
            result.0, worker2,
            "Should return worker with smallest logit when temperature is 0"
        );
        assert_eq!(
            result.1, 3.0,
            "Should return the smallest logit when temperature is 0"
        );

        logits.clear();
        let worker5 = WorkerWithDpRank::from_worker_id(5);
        let worker6 = WorkerWithDpRank::from_worker_id(6);
        logits.insert(worker1, 5.0);
        logits.insert(worker2, 3.0);
        logits.insert(worker5, 3.0);
        logits.insert(worker6, 7.0);

        let result = softmax_sample(&logits, 0.0);
        assert!(
            result.0 == worker2 || result.0 == worker5,
            "Should return one of the workers tied for the smallest logit"
        );
        assert_eq!(result.1, 3.0, "Should return the tied minimum logit");

        logits.clear();
        let worker10 = WorkerWithDpRank::from_worker_id(10);
        let worker20 = WorkerWithDpRank::from_worker_id(20);
        let worker30 = WorkerWithDpRank::from_worker_id(30);
        logits.insert(worker10, -1.0);
        logits.insert(worker20, -5.0);
        logits.insert(worker30, 0.0);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(
            result.0, worker20,
            "Should handle negative logits correctly"
        );
        assert_eq!(result.1, -5.0, "Should return the minimum negative logit");
    }

    #[test]
    fn test_softmax_sample_with_sample_returns_selected_logit() {
        let worker1 = WorkerWithDpRank::from_worker_id(1);
        let worker2 = WorkerWithDpRank::from_worker_id(2);
        let worker3 = WorkerWithDpRank::from_worker_id(3);

        let logits = FxHashMap::from_iter([(worker1, 0.0), (worker2, 3.0), (worker3, 9.0)]);
        let entries: Vec<_> = logits
            .iter()
            .map(|(worker, logit)| (*worker, *logit))
            .collect();
        let values: Vec<_> = entries.iter().map(|(_, logit)| *logit).collect();

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let temperature = 1.0;
        let range = max_val - min_val;
        let scaled: Vec<f64> = values.iter().map(|&v| -(v / range) / temperature).collect();
        let max_scaled = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut probabilities: Vec<f64> = scaled.iter().map(|&v| (v - max_scaled).exp()).collect();
        let sum: f64 = probabilities.iter().sum();
        probabilities.iter_mut().for_each(|p| *p /= sum);

        let target_idx = entries
            .iter()
            .position(|(_, logit)| *logit > min_val)
            .expect("expected at least one non-minimum logit");
        let cumsum_before: f64 = probabilities.iter().take(target_idx).sum();
        let sample = cumsum_before + probabilities[target_idx] / 2.0;

        let result = softmax_sample_with_sample(&logits, temperature, sample);
        assert_eq!(result, entries[target_idx]);
    }

    #[test]
    fn test_default_selector_randomizes_zero_temperature_ties() {
        use crate::test_utils::SimpleWorkerConfig;

        let config = KvRouterConfig {
            router_temperature: 0.0,
            ..Default::default()
        };
        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let workers = HashMap::from([
            (10, SimpleWorkerConfig::default()),
            (20, SimpleWorkerConfig::default()),
            (30, SimpleWorkerConfig::default()),
        ]);
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        };
        let mut selected = [false; 3];

        for _ in 0..120 {
            let result = selector
                .select_worker(&workers, &request, request.eligibility(), 16)
                .unwrap();
            match result.worker.worker_id {
                10 => selected[0] = true,
                20 => selected[1] = true,
                30 => selected[2] = true,
                worker_id => panic!("unexpected worker id: {worker_id}"),
            }
        }

        let selected_count = selected.into_iter().filter(|seen| *seen).count();
        assert!(
            selected_count > 1,
            "zero-temperature tie-breaking should not always select the same worker"
        );
    }

    #[test]
    fn seeded_selector_is_stable_for_ties_and_temperature_sampling() {
        use crate::test_utils::SimpleWorkerConfig;

        for temperature in [0.0, 0.7] {
            let config = KvRouterConfig {
                router_temperature: temperature,
                ..Default::default()
            };
            let first = DefaultWorkerSelector::new_seeded(Some(config.clone()), "test", 42);
            let second = DefaultWorkerSelector::new_seeded(Some(config), "test", 42);
            let first_workers = HashMap::from([
                (30, SimpleWorkerConfig::default()),
                (10, SimpleWorkerConfig::default()),
                (20, SimpleWorkerConfig::default()),
            ]);
            let second_workers = HashMap::from([
                (20, SimpleWorkerConfig::default()),
                (30, SimpleWorkerConfig::default()),
                (10, SimpleWorkerConfig::default()),
            ]);
            let request = base_request(16);

            let first_sequence = (0..64)
                .map(|_| {
                    first
                        .select_worker(&first_workers, &request, request.eligibility(), 16)
                        .unwrap()
                        .worker
                })
                .collect::<Vec<_>>();
            let second_sequence = (0..64)
                .map(|_| {
                    second
                        .select_worker(&second_workers, &request, request.eligibility(), 16)
                        .unwrap()
                        .worker
                })
                .collect::<Vec<_>>();

            assert_eq!(first_sequence, second_sequence);
        }
    }

    #[test]
    fn test_overloaded_high_overlap_worker_is_skipped() {
        use crate::test_utils::SimpleWorkerConfig;

        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit: 1.0,
                router_temperature: 0.0,
                ..Default::default()
            }),
            "test",
        );
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);
        let worker0 = WorkerWithDpRank::from_worker_id(0);
        let mut request = base_request(64);
        request
            .overlap
            .effective_overlap_blocks
            .insert(worker0, 4.0);
        request.overlap.effective_cached_tokens.insert(worker0, 64);

        let overloaded_worker_ids = HashSet::from([0]);
        let result = selector
            .select_worker(
                &workers,
                &request,
                request.eligibility_with_overloaded(Some(&overloaded_worker_ids)),
                16,
            )
            .unwrap();

        assert_eq!(result.worker.worker_id, 1);
    }

    #[test]
    fn test_all_eligible_workers_overloaded_returns_overload_error() {
        use crate::test_utils::SimpleWorkerConfig;

        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);
        let request = base_request(16);
        let overloaded_worker_ids = HashSet::from([0, 1]);

        let result = selector.select_worker(
            &workers,
            &request,
            request.eligibility_with_overloaded(Some(&overloaded_worker_ids)),
            16,
        );

        assert!(matches!(
            result,
            Err(KvSchedulerError::AllEligibleWorkersOverloaded)
        ));
    }

    #[test]
    fn test_overloaded_pinned_worker_is_not_rerouted() {
        use crate::test_utils::SimpleWorkerConfig;

        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);
        let mut request = base_request(16);
        request.pinned_worker = Some(WorkerWithDpRank::from_worker_id(0));
        let overloaded_worker_ids = HashSet::from([0]);

        let result = selector.select_worker(
            &workers,
            &request,
            request.eligibility_with_overloaded(Some(&overloaded_worker_ids)),
            16,
        );

        assert!(matches!(
            result,
            Err(KvSchedulerError::PinnedWorkerOverloaded { worker_id: 0 })
        ));
    }

    #[test]
    fn test_required_taints_return_no_endpoints_when_no_worker_matches() {
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let workers = HashMap::from([(
            10,
            TaintedWorkerConfig {
                taints: HashSet::from(["mdc-a".to_string()]),
            },
        )]);
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::from(["mdc-b".to_string()]),
                preferred_taints: HashMap::new(),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector.select_worker(&workers, &request, request.eligibility(), 16);
        assert!(matches!(result, Err(KvSchedulerError::NoEndpoints)));
    }

    #[test]
    fn test_required_taints_filter_out_incompatible_workers() {
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let workers = HashMap::from([
            (
                10,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-a".to_string()]),
                },
            ),
            (
                20,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-b".to_string()]),
                },
            ),
        ]);
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::from(["mdc-b".to_string()]),
                preferred_taints: HashMap::new(),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), 16)
            .unwrap();
        assert_eq!(result.worker.worker_id, 20);
    }

    #[test]
    fn test_required_taints_switch_matching_worker_sets_by_label() {
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let name_a = "mdc-a".to_string();
        let name_b = "mdc-b".to_string();
        let name_c = "mdc-c".to_string();
        let taint_a = TaintedWorkerConfig {
            taints: HashSet::from([name_a.clone()]),
        };
        let taint_b = TaintedWorkerConfig {
            taints: HashSet::from([name_b.clone()]),
        };
        let taint_c = TaintedWorkerConfig {
            taints: HashSet::from([name_c.clone()]),
        };
        let workers = HashMap::from([
            (10, taint_a.clone()),
            (11, taint_a),
            (20, taint_b.clone()),
            (21, taint_b),
            (30, taint_c.clone()),
            (31, taint_c),
        ]);

        for (required_taint, expected_worker_id, noisy_worker_id) in [
            (name_a, 10_u64, 11_u64),
            (name_b, 20_u64, 21_u64),
            (name_c, 30_u64, 31_u64),
        ] {
            let mut decode_blocks = FxHashMap::default();
            decode_blocks.insert(WorkerWithDpRank::from_worker_id(expected_worker_id), 0);
            decode_blocks.insert(WorkerWithDpRank::from_worker_id(noisy_worker_id), 400_000);

            let request = SchedulingRequest {
                mode: ScheduleMode::QueryOnly {
                    request_id: Some("test".into()),
                },
                token_seq: None,
                isl_tokens: 16,
                overlap: OverlapSignals {
                    tier_overlap_blocks: Default::default(),
                    effective_overlap_blocks: HashMap::default(),
                    effective_cached_tokens: HashMap::default(),
                },
                worker_loads: worker_loads_with_active_decode(decode_blocks),
                track_prefill_tokens: true,
                router_config_override: None,
                lora_name: None,
                priority_jump: 0.0,
                strict_priority: 0,
                policy_class: None,
                session_id: None,
                expected_output_tokens: None,
                pinned_worker: None,
                allowed_worker_ids: None,
                routing_constraints: crate::protocols::RoutingConstraints {
                    required_taints: HashSet::from([required_taint.clone()]),
                    preferred_taints: HashMap::new(),
                },
                shared_cache_hits: None,
                resp_tx: None,
            };

            let result = selector
                .select_worker(&workers, &request, request.eligibility(), 16)
                .unwrap();
            assert_eq!(
                result.worker.worker_id, expected_worker_id,
                "required taint {required_taint} should route only within its compatible worker set"
            );
        }
    }

    #[test]
    fn test_preferred_taints_prefer_matching_worker() {
        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                router_temperature: 0.0,
                ..Default::default()
            }),
            "test",
        );
        let workers = HashMap::from([
            (
                10,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-a".to_string()]),
                },
            ),
            (
                20,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-b".to_string()]),
                },
            ),
        ]);
        let mut decode_blocks = FxHashMap::default();
        decode_blocks.insert(WorkerWithDpRank::from_worker_id(10), 100);
        decode_blocks.insert(WorkerWithDpRank::from_worker_id(20), 90);

        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::new(),
                preferred_taints: HashMap::from([("mdc-a".to_string(), 0.85)]),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), 16)
            .unwrap();
        assert_eq!(result.worker.worker_id, 10);
    }

    #[test]
    fn test_negative_preferred_taints_avoid_matching_worker() {
        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                router_temperature: 0.0,
                ..Default::default()
            }),
            "test",
        );
        let workers = HashMap::from([
            (
                10,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-a".to_string()]),
                },
            ),
            (
                20,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-b".to_string()]),
                },
            ),
        ]);
        let mut decode_blocks = FxHashMap::default();
        decode_blocks.insert(WorkerWithDpRank::from_worker_id(10), 90);
        decode_blocks.insert(WorkerWithDpRank::from_worker_id(20), 100);

        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::new(),
                preferred_taints: HashMap::from([("mdc-a".to_string(), -0.25)]),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), 16)
            .unwrap();
        assert_eq!(result.worker.worker_id, 20);
    }

    /// Test the scoring formula with shared cache hits.
    ///
    /// Request [A, B, C, D], shared_cache_multiplier=0.5, block_size=1
    /// - Worker 0: device=[A,B] (overlap=2), shared has [A,B,C,D] -> shared_beyond=2
    ///   adjusted_prefill = isl - 2 - 0.5*2 = 4-2-1 = 1, logit = 1.0 * 1 + 0 = 1.0
    /// - Worker 1: device=[] (overlap=0), shared has [A,B,C,D] -> shared_beyond=4
    ///   adjusted_prefill = isl - 0.5*4 = 4-2 = 2, logit = 1.0 * 2 + 0 = 2.0
    ///
    /// Worker 0 has lower logit (less work), so it wins.
    #[test]
    fn test_shared_cache_hits_scoring() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 1u32;
        let isl = 4usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);

        let mut effective_overlap_blocks = HashMap::new();
        effective_overlap_blocks.insert(worker0, 2.0);
        // worker1 has 0 overlap (not in map)

        let mut effective_cached_tokens = HashMap::new();
        effective_cached_tokens.insert(worker0, 2);

        let mut tier_overlap_blocks = crate::scheduling::TierOverlapBlocks::default();
        tier_overlap_blocks.device.insert(worker0, 2);

        #[allow(clippy::single_range_in_vec_init)]
        let shared_hits = SharedCacheHits::from_ranges(vec![0..4]);

        let config = KvRouterConfig {
            overlap_score_credit: 1.0,
            shared_cache_multiplier: 0.5,
            router_temperature: 0.0,
            ..Default::default()
        };

        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());
        workers.insert(1, SimpleWorkerConfig::default());

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: isl,
            overlap: OverlapSignals {
                tier_overlap_blocks,
                effective_overlap_blocks,
                effective_cached_tokens,
            },
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: Some(shared_hits),
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), block_size)
            .unwrap();

        // Worker 0 should win: logit 1.0 < 2.0
        assert_eq!(
            result.worker, worker0,
            "Worker 0 should be selected (lower logit due to device and shared cache)"
        );
    }

    #[test]
    fn test_prefill_load_scale_applies_after_overlap_credits() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let isl = 64usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);
        let worker1 = WorkerWithDpRank::from_worker_id(1);

        let mut effective_cached_tokens = HashMap::new();
        effective_cached_tokens.insert(worker0, 32);

        let mut tier_overlap_blocks = crate::scheduling::TierOverlapBlocks::default();
        tier_overlap_blocks.device.insert(worker0, 2);

        let config = KvRouterConfig {
            overlap_score_credit: 1.0,
            prefill_load_scale: 2.0,
            router_temperature: 0.0,
            ..Default::default()
        };

        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());
        workers.insert(1, SimpleWorkerConfig::default());

        let mut decode_blocks = FxHashMap::default();
        decode_blocks.insert(worker0, 3);
        decode_blocks.insert(worker1, 0);

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: isl,
            overlap: OverlapSignals {
                tier_overlap_blocks,
                effective_overlap_blocks: HashMap::new(),
                effective_cached_tokens,
            },
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), block_size)
            .unwrap();

        assert_eq!(
            result.worker, worker0,
            "prefill load scale should apply before adding decode block load"
        );
    }

    #[test]
    fn test_overlap_credit_above_one_can_prefer_colocated_worker() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let warm_worker = WorkerWithDpRank::from_worker_id(0);
        let cold_worker = WorkerWithDpRank::from_worker_id(1);
        let workers = HashMap::from([
            (warm_worker.worker_id, SimpleWorkerConfig::default()),
            (cold_worker.worker_id, SimpleWorkerConfig::default()),
        ]);

        let mut request = base_request(128);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(warm_worker, 4);
        request
            .overlap
            .effective_cached_tokens
            .insert(warm_worker, 64);
        request.worker_loads.insert(
            warm_worker,
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 5,
                ..Default::default()
            },
        );

        let normal_credit = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit: 1.0,
                ..Default::default()
            }),
            "test",
        );
        let amplified_credit = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit: 1.5,
                ..Default::default()
            }),
            "test",
        );

        assert_eq!(
            normal_credit
                .select_worker(&workers, &request, request.eligibility(), block_size)
                .unwrap()
                .worker,
            cold_worker
        );
        assert_eq!(
            amplified_credit
                .select_worker(&workers, &request, request.eligibility(), block_size)
                .unwrap()
                .worker,
            warm_worker
        );
    }

    #[test]
    fn test_worker_logit_clamps_non_decode_overlap_credit() {
        let worker = WorkerWithDpRank::from_worker_id(0);
        let mut request = base_request(64);
        request.overlap.effective_cached_tokens.insert(worker, 96);
        request.overlap.tier_overlap_blocks.device.insert(worker, 6);
        request.worker_loads.insert(
            worker,
            crate::sequences::WorkerLoadProjection {
                active_prefill_tokens: 16,
                active_decode_blocks: 2,
                active_requests: 0,
                additional_active_blocks: 3,
            },
        );
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 2.0,
            shared_cache_multiplier: 0.0,
        };

        assert_eq!(
            selector.worker_logit(&request, worker, 16, 0, weights, "test"),
            7.0
        );

        request.track_prefill_tokens = false;
        assert_eq!(
            selector.worker_logit(&request, worker, 16, 0, weights, "test"),
            5.0
        );
    }

    #[test]
    fn test_worker_logit_can_charge_active_requests() {
        let worker = WorkerWithDpRank::from_worker_id(0);
        let mut request = base_request(0);
        request.worker_loads.insert(
            worker,
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 100,
                active_requests: 4,
                ..Default::default()
            },
        );
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 1.0,
            shared_cache_multiplier: 0.0,
        };
        let default = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let weighted = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                decode_active_request_weight: 32.0,
                ..Default::default()
            }),
            "test",
        );

        assert_eq!(
            default.worker_logit(&request, worker, 16, 0, weights, "test"),
            100.0
        );
        assert_eq!(
            weighted.worker_logit(&request, worker, 16, 0, weights, "test"),
            228.0
        );
    }

    #[test]
    fn test_decode_worker_logit_credits_overlap_without_prefill_tracking() {
        let worker = WorkerWithDpRank::from_worker_id(0);
        let mut request = base_request(64);
        request.track_prefill_tokens = false;
        request.overlap.tier_overlap_blocks.device.insert(worker, 3);
        request.worker_loads.insert(
            worker,
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 10,
                ..Default::default()
            },
        );
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "decode");
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 1.0,
            shared_cache_multiplier: 0.0,
        };

        assert_eq!(
            selector.worker_logit(&request, worker, 16, 0, weights, "test"),
            7.0
        );
    }

    #[test]
    fn test_overlap_credit_decay_can_prefer_less_loaded_cold_worker() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let warm_worker = WorkerWithDpRank::from_worker_id(0);
        let cold_worker = WorkerWithDpRank::from_worker_id(1);
        let workers = HashMap::from([
            (warm_worker.worker_id, SimpleWorkerConfig::default()),
            (cold_worker.worker_id, SimpleWorkerConfig::default()),
        ]);

        let mut request = base_request(64);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(warm_worker, 4);
        request
            .overlap
            .effective_cached_tokens
            .insert(warm_worker, 64);
        request.worker_loads.insert(
            warm_worker,
            crate::sequences::WorkerLoadProjection {
                active_prefill_tokens: 48,
                ..Default::default()
            },
        );

        let no_decay = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit_decay: 0.0,
                ..Default::default()
            }),
            "test",
        );
        let with_decay = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit_decay: 1.0,
                ..Default::default()
            }),
            "test",
        );

        assert_eq!(
            no_decay
                .select_worker(&workers, &request, request.eligibility(), block_size)
                .unwrap()
                .worker,
            warm_worker
        );
        assert_eq!(
            with_decay
                .select_worker(&workers, &request, request.eligibility(), block_size)
                .unwrap()
                .worker,
            cold_worker
        );
    }

    #[test]
    fn test_effective_overlap_falls_back_when_tier_blocks_are_absent() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let isl = 64usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);
        let worker1 = WorkerWithDpRank::from_worker_id(1);

        let mut effective_overlap_blocks = HashMap::new();
        effective_overlap_blocks.insert(worker0, 4.0);

        let config = KvRouterConfig {
            overlap_score_credit: 1.0,
            router_temperature: 0.0,
            ..Default::default()
        };

        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());
        workers.insert(1, SimpleWorkerConfig::default());

        let mut decode_blocks = FxHashMap::default();
        decode_blocks.insert(worker0, 1);
        decode_blocks.insert(worker1, 0);

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: isl,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks,
                effective_cached_tokens: HashMap::new(),
            },
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), block_size)
            .unwrap();

        assert_eq!(
            result.worker, worker0,
            "effective overlap should still credit older callers without tier maps"
        );
    }

    /// Without shared cache hits, the scoring should be unchanged.
    #[test]
    fn test_no_shared_cache_unchanged() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let isl = 64usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);

        let mut effective_overlap_blocks = HashMap::new();
        effective_overlap_blocks.insert(worker0, 2.0);

        let config = KvRouterConfig::default();
        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: isl,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks,
                effective_cached_tokens: HashMap::new(),
            },
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), block_size)
            .unwrap();

        assert_eq!(result.worker, worker0);
    }

    fn two_tier_selector(yaml: &str, config: KvRouterConfig) -> DefaultWorkerSelector {
        two_tier_selector_with(yaml, config, false)
    }

    fn two_tier_selector_with(
        yaml: &str,
        config: KvRouterConfig,
        is_eagle: bool,
    ) -> DefaultWorkerSelector {
        use std::io::Write;

        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(file.path().display().to_string()),
            ..config
        };
        // The policy document is parsed and cached during construction, so the file may go.
        DefaultWorkerSelector::for_stage(
            Some(config),
            "decode",
            WorkerSelectionStage::Aggregated,
            is_eagle,
        )
        .unwrap()
    }

    const TWO_TIER_YAML: &str = r#"
worker_selection:
  aggregated: dynamo-two-tier-cost-fn
  instances:
    - name: dynamo-two-tier-cost-fn
      type: dynamo-two-tier-cost-fn
"#;

    /// Request of ten 16-token blocks over workers given as
    /// `(worker_id, device_blocks, host_blocks, active_requests)`.
    fn two_tier_request(workers: &[(u64, usize, usize, usize)]) -> SchedulingRequest {
        two_tier_request_isl(160, workers)
    }

    /// As [`two_tier_request`] with an explicit prompt length in tokens (16-token blocks).
    fn two_tier_request_isl(
        isl_tokens: usize,
        workers: &[(u64, usize, usize, usize)],
    ) -> SchedulingRequest {
        let mut request = base_request(isl_tokens);
        for &(id, device, host, active) in workers {
            let worker = WorkerWithDpRank::from_worker_id(id);
            request
                .overlap
                .tier_overlap_blocks
                .device
                .insert(worker, device);
            request
                .overlap
                .tier_overlap_blocks
                .host_pinned
                .insert(worker, host);
            request.overlap.effective_overlap_blocks.insert(
                worker,
                device as f64 + KvRouterConfig::default().host_cache_hit_weight * host as f64,
            );
            request.worker_loads.insert(
                worker,
                crate::sequences::WorkerLoadProjection {
                    active_requests: active,
                    ..Default::default()
                },
            );
        }
        request
    }

    fn select_ids(
        selector: &DefaultWorkerSelector,
        request: &SchedulingRequest,
        ids: &[u64],
    ) -> WorkerSelectionResult {
        let workers: HashMap<_, _> = ids
            .iter()
            .map(|&id| (id, TaintedWorkerConfig::default()))
            .collect();
        selector
            .select_worker(&workers, request, request.eligibility(), 16)
            .unwrap()
    }

    #[test]
    fn stage_without_worker_selection_keeps_builtin_selector() {
        let selector = DefaultWorkerSelector::for_stage(
            None,
            "decode",
            WorkerSelectionStage::Aggregated,
            false,
        )
        .unwrap();
        assert_eq!(selector.worker_selection_policy_type(), None);

        // A document that selects the policy only for prefill leaves aggregated on the built-in.
        let prefill_only = TWO_TIER_YAML.replace("aggregated:", "prefill:");
        let selector = two_tier_selector(&prefill_only, KvRouterConfig::default());
        assert_eq!(selector.worker_selection_policy_type(), None);
    }

    #[test]
    fn invalid_worker_selection_fails_selector_construction() {
        use std::io::Write;

        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(
            TWO_TIER_YAML
                .replace("type: dynamo-two-tier-cost-fn", "type: nope")
                .as_bytes(),
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(file.path().display().to_string()),
            ..Default::default()
        };
        assert!(config.validate().is_err());
        assert!(
            DefaultWorkerSelector::for_stage(
                Some(config),
                "decode",
                WorkerSelectionStage::Aggregated,
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn two_tier_policy_takes_cache_tier_where_builtin_balances_load() {
        let selector = two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default());
        assert_eq!(
            selector.worker_selection_policy_type(),
            Some(two_tier_cost_fn::POLICY_TYPE)
        );
        // Worker 2 holds 6/10 blocks on device with 4 active requests; worker 1 is idle.
        let request = two_tier_request(&[(1, 0, 0, 0), (2, 6, 0, 4)]);
        let result = select_ids(&selector, &request, &[1, 2]);
        assert_eq!(result.worker.worker_id, 2);
        assert_eq!(result.required_blocks, 10);
        assert_eq!(result.effective_overlap_blocks, 6.0);
    }

    #[test]
    fn two_tier_cache_ratio_uses_complete_blocks() {
        let selector = two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default());
        let pick = |isl, workers: &[(u64, usize, usize, usize)]| {
            let request = two_tier_request_isl(isl, workers);
            select_ids(&selector, &request, &[1, 2]).worker.worker_id
        };
        // 1 complete block + 1-token tail: a full hit is 1/1, not 1/2 = 0.5 (rejected upstream).
        assert_eq!(pick(17, &[(1, 0, 0, 0), (2, 1, 0, 4)]), 2);
        // 5 complete + 3-token tail: 3 device blocks are 3/5 = 0.6, not 3/6 = 0.5.
        assert_eq!(pick(83, &[(1, 0, 0, 0), (2, 3, 0, 4)]), 2);
        // Same with CPU blocks: 4 host blocks at 0.75 are 3.0 effective, 3/5 = 0.6.
        assert_eq!(pick(83, &[(1, 0, 0, 0), (2, 0, 4, 4)]), 2);
        // Exact boundary on the complete count stays strict: 10 complete + tail, 2 device +
        // 4 host * 0.75 = 5.0 effective = 5/10 = 0.5, so load decides.
        assert_eq!(pick(165, &[(1, 0, 0, 0), (2, 2, 4, 4)]), 1);
        // Below one complete block nothing can match; least-loaded wins.
        assert_eq!(pick(15, &[(1, 0, 0, 0), (2, 0, 0, 4)]), 1);
        // The ceiled count still sizes admission.
        let request = two_tier_request_isl(83, &[(1, 0, 0, 0), (2, 3, 0, 4)]);
        assert_eq!(select_ids(&selector, &request, &[1, 2]).required_blocks, 6);
    }

    #[test]
    fn two_tier_cache_ratio_counts_eagle_windows() {
        // Eagle hashes (isl - 1) / block_size complete blocks: 160 tokens are 9 blocks, so 5
        // device blocks are 5/9 > 0.5; without Eagle they are 5/10 = 0.5 and load decides.
        let request = two_tier_request_isl(160, &[(1, 0, 0, 0), (2, 5, 0, 4)]);
        let eagle = two_tier_selector_with(TWO_TIER_YAML, KvRouterConfig::default(), true);
        assert_eq!(select_ids(&eagle, &request, &[1, 2]).worker.worker_id, 2);
        let plain = two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default());
        assert_eq!(select_ids(&plain, &request, &[1, 2]).worker.worker_id, 1);
    }

    #[test]
    fn two_tier_policy_does_not_receive_effective_overlap_as_device_overlap() {
        // Equivalent of upstream custom_policy_does_not_receive_effective_overlap_as_device_overlap:
        // with no tier maps, worker 2's six effective blocks are not device blocks, so the idle
        // worker 1 wins instead of the cache tier picking worker 2.
        let selector = two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default());
        let mut request = base_request(160);
        let (cold, warm) = (
            WorkerWithDpRank::from_worker_id(1),
            WorkerWithDpRank::from_worker_id(2),
        );
        request.overlap.effective_overlap_blocks.insert(warm, 6.0);
        request.overlap.effective_cached_tokens.insert(warm, 96);
        for (worker, active_requests) in [(cold, 0), (warm, 4)] {
            request.worker_loads.insert(
                worker,
                crate::sequences::WorkerLoadProjection {
                    active_requests,
                    ..Default::default()
                },
            );
        }
        assert!(request.overlap.tier_overlap_blocks.device.is_empty());
        assert_eq!(select_ids(&selector, &request, &[1, 2]).worker, cold);
    }

    #[test]
    fn two_tier_policy_load_tier_overrides_cache() {
        let selector = two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default());
        let request = two_tier_request(&[(1, 0, 0, 0), (2, 10, 0, 40)]);
        assert_eq!(select_ids(&selector, &request, &[1, 2]).worker.worker_id, 1);
    }

    #[test]
    fn two_tier_policy_counts_cpu_tier_at_router_host_weight() {
        // Worker 2 has 8/10 blocks only in CPU offload: 8 * 0.75 = 6.0 effective blocks.
        let request = two_tier_request(&[(1, 0, 0, 0), (2, 0, 8, 4)]);
        let selector = two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default());
        assert_eq!(select_ids(&selector, &request, &[1, 2]).worker.worker_id, 2);

        // DYN_ROUTER_HOST_CACHE_HIT_WEIGHT=0.5 lands in host_cache_hit_weight: 8 * 0.5 = 4.0,
        // below the threshold, so load decides.
        let selector = two_tier_selector(
            TWO_TIER_YAML,
            KvRouterConfig {
                host_cache_hit_weight: 0.5,
                ..Default::default()
            },
        );
        assert_eq!(select_ids(&selector, &request, &[1, 2]).worker.worker_id, 1);

        // An instance override wins over the router weight.
        let overridden =
            format!("{TWO_TIER_YAML}      parameters:\n        host_cache_weight: 1.0\n");
        let selector = two_tier_selector(
            &overridden,
            KvRouterConfig {
                host_cache_hit_weight: 0.0,
                ..Default::default()
            },
        );
        assert_eq!(select_ids(&selector, &request, &[1, 2]).worker.worker_id, 2);
    }

    #[test]
    fn two_tier_policy_reads_host_pinned_lower_tier_matches() {
        // Feed the policy from the indexer's tiered matches, as the scheduler queue does: worker 2
        // holds no device prefix but an 8-block HostPinned continuation (the CPU tier that lazy
        // offload stores, completed by the zmq_wire lower-tier fill, populate).
        use crate::indexer::{LowerTierMatchDetails, MatchDetails, TieredMatchDetails};
        use crate::protocols::{OverlapScores, StorageTier};
        use crate::scheduling::OverlapAnalysis;

        let cold = WorkerWithDpRank::from_worker_id(1);
        let cpu = WorkerWithDpRank::from_worker_id(2);
        let mut host = LowerTierMatchDetails::default();
        host.hits.insert(cpu, 8);
        let tiered = TieredMatchDetails {
            device: MatchDetails {
                overlap_scores: OverlapScores::new(),
                last_matched_hashes: Default::default(),
            },
            lower_tier: std::collections::HashMap::from([(StorageTier::HostPinned, host)]),
        };
        let config = KvRouterConfig::default();
        let mut request = base_request(160);
        request.overlap = OverlapAnalysis::new(&config, 16, &tiered).signals();
        assert_eq!(request.overlap.tier_overlap_blocks.host_pinned[&cpu], 8);
        for (worker, active_requests) in [(cold, 0), (cpu, 4)] {
            request.worker_loads.insert(
                worker,
                crate::sequences::WorkerLoadProjection {
                    active_requests,
                    ..Default::default()
                },
            );
        }

        let selector = two_tier_selector(TWO_TIER_YAML, config);
        let result = select_ids(&selector, &request, &[1, 2]);
        assert_eq!(result.worker, cpu);
        assert_eq!(result.effective_overlap_blocks, 6.0);
    }

    #[test]
    fn two_tier_policy_skips_overloaded_workers() {
        let selector = two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default());
        let request = two_tier_request(&[(1, 0, 0, 0), (2, 10, 0, 0)]);
        let workers = HashMap::from([
            (1, TaintedWorkerConfig::default()),
            (2, TaintedWorkerConfig::default()),
        ]);
        let overloaded = HashSet::from([2]);
        let result = selector
            .select_worker(
                &workers,
                &request,
                request.eligibility_with_overloaded(Some(&overloaded)),
                16,
            )
            .unwrap();
        assert_eq!(result.worker.worker_id, 1);

        let overloaded = HashSet::from([1, 2]);
        assert!(matches!(
            selector.select_worker(
                &workers,
                &request,
                request.eligibility_with_overloaded(Some(&overloaded)),
                16,
            ),
            Err(KvSchedulerError::AllEligibleWorkersOverloaded)
        ));
    }

    #[test]
    fn two_tier_policy_honours_pinned_worker() {
        let selector = two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default());
        let mut request = two_tier_request(&[(1, 0, 0, 0), (2, 10, 0, 0)]);
        request.pinned_worker = Some(WorkerWithDpRank::from_worker_id(1));
        assert_eq!(select_ids(&selector, &request, &[1, 2]).worker.worker_id, 1);
    }

    // ---- Cache-reuse funnel (F2/F3): raw router-visible cached prefix ----

    fn tracked(mut request: SchedulingRequest) -> SchedulingRequest {
        request.mode = ScheduleMode::Tracked {
            request_id: "tracked".into(),
        };
        request
    }

    #[test]
    fn raw_cached_tokens_add_every_tier_without_weights() {
        let mut request = base_request(128);
        let worker = WorkerWithDpRank::from_worker_id(1);
        request.overlap.tier_overlap_blocks.device.insert(worker, 2);
        request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .insert(worker, 3);
        request.overlap.tier_overlap_blocks.disk.insert(worker, 1);
        request.overlap.effective_cached_tokens.insert(worker, 7);
        assert_eq!(request.raw_cached_tokens_for(worker, 16), 96);
    }

    #[test]
    fn raw_overlap_is_specific_to_the_selected_dp_rank() {
        let mut request = base_request(128);
        let selected = WorkerWithDpRank::new(1, 0);
        let other_rank = WorkerWithDpRank::new(1, 1);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(selected, 1);
        request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .insert(selected, 2);
        request.overlap.tier_overlap_blocks.disk.insert(selected, 1);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(other_rank, 7);
        assert_eq!(request.raw_cached_tokens_for(selected, 16), 64);
        assert_eq!(request.raw_cached_tokens_for(other_rank, 16), 112);
    }

    #[test]
    fn builtin_selector_reports_raw_cache_reuse_for_tracked_requests() {
        // Worker 1 holds the longer raw prefix (2 device + 4 CPU blocks = 96 tokens) but is
        // loaded; worker 2 holds 3 device blocks (48 tokens) and is idle, so it wins.
        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                router_temperature: 0.0,
                ..Default::default()
            }),
            "decode",
        );
        let mut request = tracked(two_tier_request(&[(1, 2, 4, 0), (2, 3, 0, 0)]));
        request.worker_loads.insert(
            WorkerWithDpRank::from_worker_id(1),
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 1_000,
                ..Default::default()
            },
        );
        let result = select_ids(&selector, &request, &[1, 2]);
        assert_eq!(result.worker.worker_id, 2);
        assert_eq!(result.max_raw_cached_tokens, Some(96));
        assert_eq!(result.selected_raw_cached_tokens, Some(48));
    }

    #[test]
    fn two_tier_policy_counts_cpu_tier_in_raw_cache_reuse() {
        // Load tier: worker 1 (8 device + 2 CPU blocks) is far busier, so the policy takes the
        // idle cold worker 2; the best eligible raw prefix still counts worker 1's CPU blocks.
        let selector = two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default());
        let request = tracked(two_tier_request(&[(1, 8, 2, 100), (2, 0, 0, 0)]));
        let result = select_ids(&selector, &request, &[1, 2]);
        assert_eq!(result.worker.worker_id, 2);
        assert_eq!(result.max_raw_cached_tokens, Some(160));
        assert_eq!(result.selected_raw_cached_tokens, Some(0));

        // Cache tier: a CPU-only prefix wins over nothing and counts at full raw size.
        let request = tracked(two_tier_request(&[(1, 0, 8, 0), (2, 0, 0, 0)]));
        let result = select_ids(&selector, &request, &[1, 2]);
        assert_eq!(result.worker.worker_id, 1);
        assert_eq!(result.max_raw_cached_tokens, Some(128));
        assert_eq!(result.selected_raw_cached_tokens, Some(128));
    }

    #[test]
    fn query_only_requests_skip_raw_cache_reuse() {
        let selector = two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default());
        let request = two_tier_request(&[(1, 4, 0, 0)]);
        let result = select_ids(&selector, &request, &[1]);
        assert_eq!(result.max_raw_cached_tokens, None);
        assert_eq!(result.selected_raw_cached_tokens, None);

        let builtin = DefaultWorkerSelector::new(None, "decode");
        let result = select_ids(&builtin, &request, &[1]);
        assert_eq!(result.max_raw_cached_tokens, None);
        assert_eq!(result.selected_raw_cached_tokens, None);
    }

    #[test]
    fn pinned_worker_is_its_own_best_eligible_prefix() {
        for selector in [
            DefaultWorkerSelector::new(None, "decode"),
            two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default()),
        ] {
            let mut request = tracked(two_tier_request(&[(1, 1, 1, 0), (2, 9, 0, 0)]));
            request.pinned_worker = Some(WorkerWithDpRank::from_worker_id(1));
            let result = select_ids(&selector, &request, &[1, 2]);
            assert_eq!(result.worker.worker_id, 1);
            assert_eq!(result.max_raw_cached_tokens, Some(32));
            assert_eq!(result.selected_raw_cached_tokens, Some(32));
        }
    }

    #[test]
    fn ineligible_workers_do_not_count_toward_best_eligible_prefix() {
        for selector in [
            DefaultWorkerSelector::new(None, "decode"),
            two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default()),
        ] {
            let request = tracked(two_tier_request(&[(1, 9, 0, 0), (2, 1, 0, 0)]));
            let workers = HashMap::from([
                (1, TaintedWorkerConfig::default()),
                (2, TaintedWorkerConfig::default()),
            ]);
            let overloaded = HashSet::from([1]);
            let result = selector
                .select_worker(
                    &workers,
                    &request,
                    request.eligibility_with_overloaded(Some(&overloaded)),
                    16,
                )
                .unwrap();
            assert_eq!(result.worker.worker_id, 2);
            // The overloaded worker's 144 cached tokens are not available reuse.
            assert_eq!(result.max_raw_cached_tokens, Some(16));
            assert_eq!(result.selected_raw_cached_tokens, Some(16));
        }
    }

    // ---- Opt-in routing-decision traces (upstream #14109) ----

    fn weights_for(selector: &DefaultWorkerSelector) -> LogitWeights {
        LogitWeights {
            overlap_score_credit: selector.kv_router_config.overlap_score_credit,
            overlap_score_credit_decay: selector.kv_router_config.overlap_score_credit_decay,
            prefill_load_scale: selector.kv_router_config.prefill_load_scale,
            shared_cache_multiplier: selector.kv_router_config.shared_cache_multiplier,
        }
    }

    #[test]
    fn decision_trace_sampling_is_deterministic_and_bounded() {
        assert!(!sampled_request("req", 0.0));
        assert!(sampled_request("req", 1.0));
        let sampled = (0..10_000)
            .filter(|index| sampled_request(&format!("req-{index}"), 0.1))
            .count();
        assert!((800..1200).contains(&sampled), "{sampled}");
        assert_eq!(
            sampled_request("stable", 0.5),
            sampled_request("stable", 0.5)
        );
        assert_eq!(parse_decision_trace_sample_rate(None), 1.0);
        assert_eq!(parse_decision_trace_sample_rate(Some("0.25".into())), 0.25);
        for invalid in ["-0.1", "1.5", "NaN", "abc"] {
            assert_eq!(parse_decision_trace_sample_rate(Some(invalid.into())), 0.0);
        }
    }

    #[test]
    fn pinned_or_anonymous_selections_are_not_traced() {
        let mut request = tracked(base_request(64));
        assert!(should_trace_decision(&request, 1.0));
        request.pinned_worker = Some(WorkerWithDpRank::from_worker_id(1));
        assert!(!should_trace_decision(&request, 1.0));
        let mut anonymous = base_request(64);
        anonymous.mode = ScheduleMode::QueryOnly { request_id: None };
        assert!(!should_trace_decision(&anonymous, 1.0));
    }

    fn select_traced(
        selector: &DefaultWorkerSelector,
        request: &SchedulingRequest,
        ids: &[u64],
    ) -> (WorkerSelectionResult, RoutingDecisionTrace) {
        let mut result = select_ids(selector, request, ids);
        let trace = *result
            .decision_trace
            .take()
            .expect("an enabled, fully sampled selector traces the decision");
        (result, trace)
    }

    #[test]
    fn default_decision_trace_scores_match_selection() {
        // A positive credit decay makes the min-active-prefill floor matter; worker 2 carries
        // prefill backlog so the floor is nonzero for the others.
        let config = KvRouterConfig {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.5,
            router_temperature: 0.0,
            ..Default::default()
        };
        let selector = DefaultWorkerSelector::new(Some(config), "test")
            .with_decision_trace_sample_rate(Some(1.0));
        let mut request = tracked(two_tier_request_isl(
            128,
            &[(0, 4, 0, 0), (1, 0, 2, 0), (2, 1, 0, 0)],
        ));
        for (id, decode, prefill) in [(0, 10, 32), (1, 1, 16), (2, 3, 400)] {
            request.worker_loads.insert(
                WorkerWithDpRank::from_worker_id(id),
                crate::sequences::WorkerLoadProjection {
                    active_decode_blocks: decode,
                    active_prefill_tokens: prefill,
                    ..Default::default()
                },
            );
        }
        request.router_config_override = Some(crate::config::RouterConfigOverride {
            prefill_load_scale: Some(2.0),
            ..Default::default()
        });
        let (selected, trace) = select_traced(&selector, &request, &[0, 1, 2]);
        assert_eq!(trace.policy, "default");
        assert_eq!(trace.selection_reason, "minimum_cost");
        assert_eq!(trace.prefill_load_scale, 2.0);
        assert_eq!(trace.selected_worker_id, selected.worker.worker_id);
        assert_eq!(trace.max_overlap_worker_id, 0);
        assert!(trace.two_tier.is_none());
        assert_eq!(trace.candidates.len(), 3);
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.5,
            prefill_load_scale: 2.0,
            shared_cache_multiplier: selector.kv_router_config.shared_cache_multiplier,
        };
        let best_cost = trace
            .candidates
            .iter()
            .map(|candidate| candidate.total_cost_blocks)
            .fold(f64::INFINITY, f64::min);
        for candidate in &trace.candidates {
            let worker = WorkerWithDpRank::new(candidate.worker_id, candidate.dp_rank);
            // Floor = least active prefill tokens among eligible workers = 16.
            let expected = selector.worker_logit(&request, worker, 16, 16, weights, "test");
            assert_eq!(candidate.total_cost_blocks, expected);
            assert_eq!(candidate.selected, worker == selected.worker);
            assert_eq!(
                candidate.raw_cached_tokens,
                request.raw_cached_tokens_for(worker, 16)
            );
        }
        let chosen = trace.candidates.iter().find(|c| c.selected).unwrap();
        assert_eq!(chosen.total_cost_blocks, best_cost);
        let selected_overlap = chosen.effective_overlap_blocks;
        assert_eq!(
            trace.avoidable_prefill_token_equivalents,
            (4.0 - selected_overlap) * 16.0
        );
        let json = serde_json::to_value(&trace).unwrap();
        assert!(json.get("two_tier").is_none());
        assert!(
            json["candidates"][0]
                .get("two_tier_overlap_blocks")
                .is_none()
        );
    }

    #[test]
    fn default_decision_trace_applies_preferred_taint_multiplier() {
        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                router_temperature: 0.0,
                ..Default::default()
            }),
            "test",
        )
        .with_decision_trace_sample_rate(Some(1.0));
        let mut request = tracked(two_tier_request(&[(1, 2, 0, 0), (2, 0, 0, 0)]));
        request.routing_constraints = crate::protocols::RoutingConstraints {
            preferred_taints: HashMap::from([("fast".to_string(), 0.5)]),
            ..Default::default()
        };
        let workers = HashMap::from([
            (
                1,
                TaintedWorkerConfig {
                    taints: HashSet::from(["fast".to_string()]),
                },
            ),
            (2, TaintedWorkerConfig::default()),
        ]);
        let mut result = selector
            .select_worker(&workers, &request, request.eligibility(), 16)
            .unwrap();
        let trace = result.decision_trace.take().unwrap();
        let tainted = trace.candidates.iter().find(|c| c.worker_id == 1).unwrap();
        let multiplier = tainted.preferred_taint_multiplier.unwrap();
        assert!(multiplier < 1.0);
        assert_eq!(
            tainted.total_cost_blocks,
            tainted.base_score_blocks * multiplier
        );
        let plain = trace.candidates.iter().find(|c| c.worker_id == 2).unwrap();
        assert_eq!(plain.preferred_taint_multiplier, Some(1.0));
    }

    #[test]
    fn two_tier_decision_trace_reports_the_deciding_tier() {
        // Load tier: worker 1 holds the prefix but is far busier.
        let selector = two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default())
            .with_decision_trace_sample_rate(Some(1.0));
        let request = tracked(two_tier_request(&[(1, 8, 2, 100), (2, 0, 0, 0)]));
        let (selected, trace) = select_traced(&selector, &request, &[1, 2]);
        assert_eq!(selected.worker.worker_id, 2);
        assert_eq!(trace.policy, two_tier_cost_fn::POLICY_TYPE);
        assert_eq!(trace.selection_reason, "two_tier_load");
        assert_eq!(trace.two_tier.as_ref().unwrap().tier, "load");
        assert_eq!(trace.two_tier.as_ref().unwrap().matchable_blocks, 10);
        assert_eq!(trace.max_overlap_worker_id, 1);
        assert_eq!(trace.candidates[0].raw_cached_tokens, 160);
        assert_eq!(trace.candidates[0].active_requests, 100);

        // Cache tier with a CPU-held prefix.
        let request = tracked(two_tier_request(&[(1, 2, 8, 0), (2, 4, 0, 1)]));
        let (selected, trace) = select_traced(&selector, &request, &[1, 2]);
        assert_eq!(selected.worker.worker_id, 1);
        assert_eq!(trace.selection_reason, "two_tier_cache");
        let first = &trace.candidates[0];
        assert!(first.selected && first.max_overlap);
        assert_eq!(first.host_overlap_blocks, 8.0);
        assert_eq!(first.raw_cached_tokens, 160);
        let host_weight = trace.two_tier.as_ref().unwrap().host_cache_weight;
        assert_eq!(first.two_tier_overlap_blocks, Some(2.0 + host_weight * 8.0));
        assert_eq!(trace.avoidable_prefill_token_equivalents, 0.0);
    }

    #[test]
    fn two_tier_decision_trace_ranks_on_the_policy_cpu_weight() {
        // The policy weighs CPU blocks at 1.0 while the router's estimate uses 0.75: the policy
        // picks B (8 CPU blocks over A's 7 device blocks), and the trace must agree.
        let yaml = r#"
worker_selection:
  aggregated: dynamo-two-tier-cost-fn
  instances:
    - name: dynamo-two-tier-cost-fn
      type: dynamo-two-tier-cost-fn
      parameters:
        host_cache_weight: 1.0
"#;
        let selector = two_tier_selector(yaml, KvRouterConfig::default())
            .with_decision_trace_sample_rate(Some(1.0));
        let request = tracked(two_tier_request(&[(1, 7, 0, 0), (2, 0, 8, 0)]));
        let (selected, trace) = select_traced(&selector, &request, &[1, 2]);
        assert_eq!(selected.worker.worker_id, 2);
        assert_eq!(selected.max_raw_cached_tokens, Some(128));
        assert_eq!(trace.max_overlap_worker_id, 2);
        assert_eq!(trace.avoidable_prefill_token_equivalents, 0.0);
        assert!(
            trace
                .candidates
                .iter()
                .find(|c| c.worker_id == 2)
                .unwrap()
                .max_overlap
        );
    }

    #[test]
    fn decision_traces_respect_enablement_sampling_and_pins() {
        let request = tracked(two_tier_request(&[(1, 4, 0, 0), (2, 0, 0, 0)]));
        for selector in [
            DefaultWorkerSelector::new(None, "decode"),
            two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default()),
        ] {
            let disabled = selector.clone().with_decision_trace_sample_rate(None);
            assert!(
                select_ids(&disabled, &request, &[1, 2])
                    .decision_trace
                    .is_none()
            );
            let unsampled = selector.clone().with_decision_trace_sample_rate(Some(0.0));
            assert!(
                select_ids(&unsampled, &request, &[1, 2])
                    .decision_trace
                    .is_none()
            );
            let enabled = selector.with_decision_trace_sample_rate(Some(1.0));
            assert!(
                select_ids(&enabled, &request, &[1, 2])
                    .decision_trace
                    .is_some()
            );
            let mut pinned = tracked(two_tier_request(&[(1, 4, 0, 0), (2, 0, 0, 0)]));
            pinned.pinned_worker = Some(WorkerWithDpRank::from_worker_id(1));
            assert!(
                select_ids(&enabled, &pinned, &[1, 2])
                    .decision_trace
                    .is_none()
            );
        }
    }

    #[test]
    fn out_of_range_dp_rank_is_not_an_eligible_prefix() {
        // Worker 1 advertises one DP rank (0); a stale overlap entry for rank 1 must not count.
        let workers = HashMap::from([
            (1, TaintedWorkerConfig::default()),
            (2, TaintedWorkerConfig::default()),
        ]);
        for selector in [
            DefaultWorkerSelector::new(None, "decode"),
            two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default()),
        ] {
            let selector = selector.with_decision_trace_sample_rate(Some(1.0));
            let mut request = tracked(two_tier_request(&[(1, 2, 0, 0), (2, 1, 0, 0)]));
            let stale = WorkerWithDpRank::new(1, 1);
            request.overlap.tier_overlap_blocks.device.insert(stale, 9);
            request.overlap.effective_overlap_blocks.insert(stale, 9.0);
            let mut result = selector
                .select_worker(&workers, &request, request.eligibility(), 16)
                .unwrap();
            assert_eq!(result.max_raw_cached_tokens, Some(32));
            assert_eq!(
                result.selected_raw_cached_tokens,
                Some(request.raw_cached_tokens_for(result.worker, 16))
            );
            let trace = result.decision_trace.take().unwrap();
            assert!(
                trace
                    .candidates
                    .iter()
                    .all(|candidate| candidate.dp_rank == 0)
            );
            assert_eq!(trace.candidates.len(), 2);
        }
    }

    #[test]
    fn decision_trace_tie_marks_the_selected_worker_as_max_overlap() {
        let selector = two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default())
            .with_decision_trace_sample_rate(Some(1.0));
        let request = tracked(two_tier_request(&[(1, 0, 0, 1), (2, 0, 0, 0)]));
        let (selected, trace) = select_traced(&selector, &request, &[1, 2]);
        assert_eq!(selected.worker.worker_id, 2);
        assert_eq!(trace.max_overlap_worker_id, 2);
        assert_eq!(trace.avoidable_prefill_token_equivalents, 0.0);
    }

    #[test]
    fn decision_tracing_is_off_by_default() {
        // The test runner does not set the flag, so a default selector never traces.
        if std::env::var_os(DYN_ROUTER_DECISION_TRACE_ENABLED).is_none() {
            assert_eq!(*ROUTER_DECISION_TRACE, None);
            let selector = DefaultWorkerSelector::new(None, "decode");
            assert_eq!(selector.decision_trace_sample_rate, None);
        }
    }

    /// Hot-path cost of the always-on F2/F3 tracking: identical selections differing only in
    /// tracked versus query-only mode. Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_raw_cache_reuse_tracking_overhead() {
        use std::hint::black_box;
        use std::time::Instant;

        let iterations = 200_000;
        for worker_count in [2_u64, 16, 64, 256] {
            let rows: Vec<_> = (0..worker_count)
                .map(|id| (id, (id % 7) as usize, (id % 5) as usize, (id % 11) as usize))
                .collect();
            let ids: Vec<u64> = (0..worker_count).collect();
            let workers: HashMap<_, _> = ids
                .iter()
                .map(|&id| (id, TaintedWorkerConfig::default()))
                .collect();
            for (label, selector) in [
                (
                    "builtin",
                    DefaultWorkerSelector::new(
                        Some(KvRouterConfig {
                            router_temperature: 0.0,
                            ..Default::default()
                        }),
                        "decode",
                    ),
                ),
                (
                    "two_tier",
                    two_tier_selector(TWO_TIER_YAML, KvRouterConfig::default()),
                ),
            ] {
                let query_only = two_tier_request(&rows);
                let tracked = tracked(two_tier_request(&rows));
                let mut timings = [0.0_f64; 2];
                for round in 0..6 {
                    // Alternate the order so drift does not favor either mode.
                    let order = if round % 2 == 0 {
                        [(0, &query_only), (1, &tracked)]
                    } else {
                        [(1, &tracked), (0, &query_only)]
                    };
                    for (slot, request) in order {
                        let start = Instant::now();
                        for _ in 0..iterations {
                            black_box(
                                selector
                                    .select_worker(&workers, request, request.eligibility(), 16)
                                    .unwrap(),
                            );
                        }
                        if round > 0 {
                            timings[slot] +=
                                start.elapsed().as_nanos() as f64 / iterations as f64 / 5.0;
                        }
                    }
                }
                println!(
                    "BENCH selector={label} workers={worker_count} untracked_ns={:.1} tracked_ns={:.1} delta_ns={:.1}",
                    timings[0],
                    timings[1],
                    timings[1] - timings[0]
                );
            }
        }
    }
}
