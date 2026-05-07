// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for the KV router.
//!
//! This module centralizes all router-side Prometheus metric definitions:
//!
//! - [`WorkerLoadMetrics`]: Per-worker active decode blocks and prefill tokens gauges.
//!   Registered on the frontend's own `prometheus::Registry` (default port 8000).
//!   Populated by `KvWorkerMonitor` in the frontend when receiving ActiveLoad events.
//!   - Frontend (aggregated and disaggregated): available on default port 8000
//!   - Standalone router (`python -m dynamo.router`): not created (frontend-only)
//!
//! - [`RoutingOverheadMetrics`]: Per-request routing phase latency histograms.
//!   Registered on the frontend's own `prometheus::Registry` (default port 8000).
//!   Populated by `KvPushRouter` in the frontend during routing decisions.
//!   - Frontend (aggregated and disaggregated): available on default port 8000
//!   - Standalone router: not created (frontend-only)
//!
//! - [`RouterRequestMetrics`]: Per-request aggregate histograms (TTFT, ITL, tokens, KV hit rate).
//!   Registered on the DRT `MetricsRegistry` hierarchy via `Component::metrics()`.
//!   Eagerly created so they appear as zeros before any requests arrive.
//!   Populated by `KvPushRouter::generate()` and its `RequestGuard` as it observes
//!   the streaming response (TTFT on first token, ITL per output block,
//!   ISL/OSL/kv_hit_rate at routing and completion).
//!   - Frontend, non-KV modes (direct/random/round-robin): always zero (registered
//!     on default port 8000, but never populated since KvPushRouter is not used)
//!   - Frontend, KV mode (aggregated and disaggregated): available on default port
//!     8000 via the `drt_metrics` bridge, populated per-request
//!   - Standalone router (`python -m dynamo.router`): available on `DYN_SYSTEM_PORT`
//!     when set (default is `-1`, disabled), populated per-request
//!
//! The standalone router does not create `WorkerLoadMetrics` or
//! `RoutingOverheadMetrics` (those are frontend-only). It only exposes
//! `RouterRequestMetrics` and standard DRT transport metrics
//! (`dynamo_component_inflight_requests`, `dynamo_component_requests_total`, etc.)
//! via the system status server when `DYN_SYSTEM_PORT` is explicitly set.
//!
//! See also: `docs/observability/metrics.md` (Router Metrics section).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, OnceLock, RwLock};
use std::time::Duration;

use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::metrics::prometheus_names::{
    frontend_service, labels, name_prefix, router_request, routing_overhead,
};

/// Build a router metric name: `"router_" + frontend_service_suffix`.
fn router_metric(suffix: &str) -> String {
    format!("{}{}", router_request::METRIC_PREFIX, suffix)
}
use dynamo_runtime::traits::DistributedRuntimeProvider;
use prometheus::{GaugeVec, HistogramOpts, IntCounterVec, IntGaugeVec, Opts};

use crate::http::service::metrics::generate_log_buckets;
use crate::local_model::runtime_config::ModelRuntimeConfig;

use super::protocols::{WorkerId, WorkerWithDpRank};

/// Exponential buckets for routing overhead histograms:
/// from 0.0001 ms (0.1 µs) to ~13.1 ms, factor 2, 18 steps.
fn overhead_buckets() -> Vec<f64> {
    prometheus::exponential_buckets(0.0001, 2.0, 18).expect("exponential buckets should not fail")
}

// ---------------------------------------------------------------------------
// Worker load metrics (gauges)
// ---------------------------------------------------------------------------

/// Per-worker active load gauges, published by `ActiveSequencesMultiWorker`
/// and cleaned up by `KvWorkerMonitor` when workers disappear.
pub struct WorkerLoadMetrics {
    pub active_decode_blocks: IntGaugeVec,
    pub active_prefill_tokens: IntGaugeVec,
    pub request_active_slots: IntGaugeVec,
    pub num_requests_waiting: IntGaugeVec,
    pub request_total_slots: IntGaugeVec,
}

impl WorkerLoadMetrics {
    pub fn observe(
        &self,
        worker_id: u64,
        dp_rank: u32,
        worker_type: &str,
        active_blocks: usize,
        active_tokens: usize,
        request_active_slots: Option<u64>,
        num_requests_waiting: Option<u64>,
        request_total_slots: Option<u64>,
    ) {
        let worker_id_str = worker_id.to_string();
        let dp_rank_str = dp_rank.to_string();
        let labels = &[worker_id_str.as_str(), dp_rank_str.as_str(), worker_type];
        self.active_decode_blocks
            .with_label_values(labels)
            .set(active_blocks as i64);
        self.active_prefill_tokens
            .with_label_values(labels)
            .set(active_tokens as i64);
        if let Some(active) = request_active_slots {
            self.request_active_slots
                .with_label_values(labels)
                .set(active as i64);
        }
        if let Some(waiting) = num_requests_waiting {
            self.num_requests_waiting
                .with_label_values(labels)
                .set(waiting as i64);
        }
        if let Some(total) = request_total_slots {
            self.request_total_slots
                .with_label_values(labels)
                .set(total as i64);
        }
    }
}

pub static WORKER_LOAD_METRICS: LazyLock<WorkerLoadMetrics> = LazyLock::new(|| WorkerLoadMetrics {
    active_decode_blocks: IntGaugeVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_ACTIVE_DECODE_BLOCKS
            ),
            "Active KV cache decode blocks per worker",
        ),
        &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
    )
    .expect("Failed to create worker_active_decode_blocks gauge"),
    active_prefill_tokens: IntGaugeVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_ACTIVE_PREFILL_TOKENS
            ),
            "Active prefill tokens queued per worker",
        ),
        &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
    )
    .expect("Failed to create worker_active_prefill_tokens gauge"),
    request_active_slots: IntGaugeVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_REQUEST_ACTIVE_SLOTS
            ),
            "Active backend request slots per worker",
        ),
        &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
    )
    .expect("Failed to create worker_request_active_slots gauge"),
    num_requests_waiting: IntGaugeVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_NUM_REQUESTS_WAITING
            ),
            "Backend requests waiting in the worker scheduler queue",
        ),
        &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
    )
    .expect("Failed to create worker_num_requests_waiting gauge"),
    request_total_slots: IntGaugeVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_REQUEST_TOTAL_SLOTS
            ),
            "Backend request slot capacity per worker",
        ),
        &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
    )
    .expect("Failed to create worker_request_total_slots gauge"),
});

/// Register the worker load gauges with the given Prometheus registry.
/// Called during frontend HTTP service setup (`service_v2.rs`), served on port 8000.
pub fn register_worker_load_metrics(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    let m = &*WORKER_LOAD_METRICS;
    registry.register(Box::new(m.active_decode_blocks.clone()))?;
    registry.register(Box::new(m.active_prefill_tokens.clone()))?;
    registry.register(Box::new(m.request_active_slots.clone()))?;
    registry.register(Box::new(m.num_requests_waiting.clone()))?;
    registry.register(Box::new(m.request_total_slots.clone()))?;
    Ok(())
}

pub static ROUTER_QUEUE_PENDING_REQUESTS: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("{}_router_pending_requests", name_prefix::FRONTEND),
            "Pending requests parked in the KV router scheduler queue",
        ),
        &[labels::WORKER_TYPE],
    )
    .expect("Failed to create router_pending_requests gauge")
});

pub static ROUTER_RUNTIME_CONFIG_WORKERS: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("{}_router_runtime_config_workers", name_prefix::FRONTEND),
            "Worker counts observed by KV router runtime-config discovery",
        ),
        &["source"],
    )
    .expect("Failed to create router_runtime_config_workers gauge")
});

pub static ROUTER_RUNTIME_CONFIG_DIVERGENCE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("{}_router_runtime_config_divergence", name_prefix::FRONTEND),
            "Whether KV router Endpoint and EndpointModels discovery disagree",
        ),
        &["direction"],
    )
    .expect("Failed to create router_runtime_config_divergence gauge")
});

pub struct RouterDpHealthMetrics {
    pub eligible: IntGaugeVec,
    pub last_eligible_unix_seconds: IntGaugeVec,
    pub selected_total: IntCounterVec,
    pub last_selected_unix_seconds: IntGaugeVec,
    pub candidate_logit: GaugeVec,
}

static ROUTER_DP_LAST_SELECTED: LazyLock<RwLock<HashMap<(WorkerWithDpRank, String), i64>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static ROUTER_DP_FIRST_ELIGIBLE: LazyLock<RwLock<HashMap<(WorkerWithDpRank, String), i64>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

impl RouterDpHealthMetrics {
    pub fn observe_selection_snapshot(
        &self,
        workers: &HashMap<WorkerId, ModelRuntimeConfig>,
        allowed_worker_ids: Option<&HashSet<WorkerId>>,
        candidate_logits: &HashMap<WorkerWithDpRank, f64>,
        worker_type: &str,
    ) {
        let now = current_unix_seconds();
        let mut first_eligible = ROUTER_DP_FIRST_ELIGIBLE.write().ok();
        for (worker_id, config) in workers {
            if allowed_worker_ids.is_some_and(|ids| !ids.contains(worker_id)) {
                continue;
            }

            for dp_rank in config.data_parallel_start_rank
                ..config.data_parallel_start_rank + config.data_parallel_size
            {
                let worker = WorkerWithDpRank::new(*worker_id, dp_rank);
                let worker_id = worker.worker_id.to_string();
                let dp_rank = worker.dp_rank.to_string();
                let labels = &[worker_id.as_str(), dp_rank.as_str(), worker_type];
                if let Some(logit) = candidate_logits.get(&worker) {
                    self.eligible.with_label_values(labels).set(1);
                    self.last_eligible_unix_seconds
                        .with_label_values(labels)
                        .set(now);
                    self.candidate_logit.with_label_values(labels).set(*logit);
                    if let Some(first_eligible) = first_eligible.as_mut() {
                        first_eligible
                            .entry((worker, worker_type.to_string()))
                            .or_insert(now);
                    }
                } else {
                    self.eligible.with_label_values(labels).set(0);
                    if let Some(first_eligible) = first_eligible.as_mut() {
                        first_eligible.remove(&(worker, worker_type.to_string()));
                    }
                }
            }
        }
    }

    pub fn observe_selected(&self, worker: WorkerWithDpRank, worker_type: &str) {
        let worker_id = worker.worker_id.to_string();
        let dp_rank = worker.dp_rank.to_string();
        let labels = &[worker_id.as_str(), dp_rank.as_str(), worker_type];
        let now = current_unix_seconds();
        self.selected_total.with_label_values(labels).inc();
        self.last_selected_unix_seconds
            .with_label_values(labels)
            .set(now);
        if let Ok(mut last_selected) = ROUTER_DP_LAST_SELECTED.write() {
            last_selected.insert((worker, worker_type.to_string()), now);
        }
    }

    pub fn oldest_stale_workers<'a>(
        &self,
        workers: impl Iterator<Item = &'a WorkerWithDpRank>,
        worker_type: &'static str,
        stale_secs: u64,
    ) -> Vec<WorkerWithDpRank> {
        let now = current_unix_seconds();
        let Ok(last_selected) = ROUTER_DP_LAST_SELECTED.read() else {
            return Vec::new();
        };
        let Ok(first_eligible) = ROUTER_DP_FIRST_ELIGIBLE.read() else {
            return Vec::new();
        };
        let mut max_age = None;
        let mut stale_workers = Vec::new();
        for worker in workers {
            let key = (*worker, worker_type.to_string());
            let age = if let Some(last) = last_selected.get(&key) {
                now.saturating_sub(*last)
            } else if let Some(first) = first_eligible.get(&key) {
                now.saturating_sub(*first)
            } else {
                continue;
            };
            if age < stale_secs as i64 {
                continue;
            }

            match max_age {
                Some(current) if age < current => {}
                Some(current) if age == current => stale_workers.push(*worker),
                _ => {
                    max_age = Some(age);
                    stale_workers.clear();
                    stale_workers.push(*worker);
                }
            }
        }
        stale_workers
    }

    #[cfg(test)]
    pub fn oldest_stale_workers_at<'a>(
        workers: impl Iterator<Item = &'a WorkerWithDpRank>,
        worker_type: &'static str,
        stale_secs: u64,
        now: i64,
    ) -> Vec<WorkerWithDpRank> {
        let Ok(last_selected) = ROUTER_DP_LAST_SELECTED.read() else {
            return Vec::new();
        };
        let Ok(first_eligible) = ROUTER_DP_FIRST_ELIGIBLE.read() else {
            return Vec::new();
        };
        let mut max_age = None;
        let mut stale_workers = Vec::new();
        for worker in workers {
            let key = (*worker, worker_type.to_string());
            let age = if let Some(last) = last_selected.get(&key) {
                now.saturating_sub(*last)
            } else if let Some(first) = first_eligible.get(&key) {
                now.saturating_sub(*first)
            } else {
                continue;
            };
            if age < stale_secs as i64 {
                continue;
            }

            match max_age {
                Some(current) if age < current => {}
                Some(current) if age == current => stale_workers.push(*worker),
                _ => {
                    max_age = Some(age);
                    stale_workers.clear();
                    stale_workers.push(*worker);
                }
            }
        }
        stale_workers
    }

    #[cfg(test)]
    pub fn clear_stale_tracking_for_test() {
        if let Ok(mut last_selected) = ROUTER_DP_LAST_SELECTED.write() {
            last_selected.clear();
        }
        if let Ok(mut first_eligible) = ROUTER_DP_FIRST_ELIGIBLE.write() {
            first_eligible.clear();
        }
    }

    #[cfg(test)]
    pub fn mark_first_eligible_for_test(worker: WorkerWithDpRank, worker_type: &str, at: i64) {
        if let Ok(mut first_eligible) = ROUTER_DP_FIRST_ELIGIBLE.write() {
            first_eligible.insert((worker, worker_type.to_string()), at);
        }
    }

    #[cfg(test)]
    pub fn mark_last_selected_for_test(worker: WorkerWithDpRank, worker_type: &str, at: i64) {
        if let Ok(mut last_selected) = ROUTER_DP_LAST_SELECTED.write() {
            last_selected.insert((worker, worker_type.to_string()), at);
        }
    }

    pub fn remove(&self, worker_id: u64, dp_rank: u32, worker_type: &str) {
        let worker_id_str = worker_id.to_string();
        let dp_rank_str = dp_rank.to_string();
        let labels = &[worker_id_str.as_str(), dp_rank_str.as_str(), worker_type];
        let _ = self.eligible.remove_label_values(labels);
        let _ = self.last_eligible_unix_seconds.remove_label_values(labels);
        let _ = self.selected_total.remove_label_values(labels);
        let _ = self.last_selected_unix_seconds.remove_label_values(labels);
        let _ = self.candidate_logit.remove_label_values(labels);
        if let Ok(mut last_selected) = ROUTER_DP_LAST_SELECTED.write() {
            last_selected.remove(&(
                WorkerWithDpRank::new(worker_id, dp_rank),
                worker_type.to_string(),
            ));
        }
        if let Ok(mut first_eligible) = ROUTER_DP_FIRST_ELIGIBLE.write() {
            first_eligible.remove(&(
                WorkerWithDpRank::new(worker_id, dp_rank),
                worker_type.to_string(),
            ));
        }
    }
}

fn current_unix_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

pub static ROUTER_DP_HEALTH_METRICS: LazyLock<RouterDpHealthMetrics> =
    LazyLock::new(|| RouterDpHealthMetrics {
        eligible: IntGaugeVec::new(
            Opts::new(
                format!("{}_router_worker_dp_eligible", name_prefix::FRONTEND),
                "Worker/DP pairs eligible for KV router selection",
            ),
            &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
        )
        .expect("Failed to create router_worker_dp_eligible gauge"),
        last_eligible_unix_seconds: IntGaugeVec::new(
            Opts::new(
                format!(
                    "{}_router_worker_dp_last_eligible_unix_seconds",
                    name_prefix::FRONTEND
                ),
                "Unix timestamp when a worker/DP pair was last eligible for KV router selection",
            ),
            &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
        )
        .expect("Failed to create router_worker_dp_last_eligible_unix_seconds gauge"),
        selected_total: IntCounterVec::new(
            Opts::new(
                format!("{}_router_worker_dp_selected_total", name_prefix::FRONTEND),
                "Total KV router selections by worker/DP pair",
            ),
            &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
        )
        .expect("Failed to create router_worker_dp_selected_total counter"),
        last_selected_unix_seconds: IntGaugeVec::new(
            Opts::new(
                format!(
                    "{}_router_worker_dp_last_selected_unix_seconds",
                    name_prefix::FRONTEND
                ),
                "Unix timestamp when a worker/DP pair was last selected by the KV router",
            ),
            &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
        )
        .expect("Failed to create router_worker_dp_last_selected_unix_seconds gauge"),
        candidate_logit: GaugeVec::new(
            Opts::new(
                format!("{}_router_worker_dp_candidate_logit", name_prefix::FRONTEND),
                "Latest KV router candidate logit by worker/DP pair; lower is preferred",
            ),
            &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
        )
        .expect("Failed to create router_worker_dp_candidate_logit gauge"),
    });

pub fn observe_runtime_config_discovery(
    endpoint_workers: usize,
    model_card_workers: usize,
    joined_workers: usize,
    endpoint_only_workers: usize,
    model_card_only_workers: usize,
) {
    ROUTER_RUNTIME_CONFIG_WORKERS
        .with_label_values(&["endpoint"])
        .set(endpoint_workers as i64);
    ROUTER_RUNTIME_CONFIG_WORKERS
        .with_label_values(&["model_card"])
        .set(model_card_workers as i64);
    ROUTER_RUNTIME_CONFIG_WORKERS
        .with_label_values(&["joined"])
        .set(joined_workers as i64);
    ROUTER_RUNTIME_CONFIG_DIVERGENCE
        .with_label_values(&["endpoint_only"])
        .set(endpoint_only_workers as i64);
    ROUTER_RUNTIME_CONFIG_DIVERGENCE
        .with_label_values(&["model_card_only"])
        .set(model_card_only_workers as i64);
}

pub fn register_router_queue_metrics(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    registry.register(Box::new(ROUTER_QUEUE_PENDING_REQUESTS.clone()))?;
    registry.register(Box::new(ROUTER_RUNTIME_CONFIG_WORKERS.clone()))?;
    registry.register(Box::new(ROUTER_RUNTIME_CONFIG_DIVERGENCE.clone()))?;
    let dp = &*ROUTER_DP_HEALTH_METRICS;
    registry.register(Box::new(dp.eligible.clone()))?;
    registry.register(Box::new(dp.last_eligible_unix_seconds.clone()))?;
    registry.register(Box::new(dp.selected_total.clone()))?;
    registry.register(Box::new(dp.last_selected_unix_seconds.clone()))?;
    registry.register(Box::new(dp.candidate_logit.clone()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Routing overhead metrics (histograms)
// ---------------------------------------------------------------------------

/// Per-request routing phase latency histograms (milliseconds).
pub struct RoutingOverheadMetrics {
    pub block_hashing: prometheus::Histogram,
    pub indexer_find_matches: prometheus::Histogram,
    pub seq_hashing: prometheus::Histogram,
    pub scheduling: prometheus::Histogram,
    pub total: prometheus::Histogram,
}

static ROUTING_OVERHEAD_METRICS: OnceLock<Arc<RoutingOverheadMetrics>> = OnceLock::new();

impl RoutingOverheadMetrics {
    /// Register routing overhead histograms with the given registry and store for later use.
    /// Metric names: `dynamo_router_overhead_*` with const label `router_id=instance_id`.
    /// Called during frontend HTTP service setup (`service_v2.rs`), so these metrics
    /// are served on the frontend's own port (default 8000). Not available in the
    /// standalone router, which has no frontend HTTP server.
    pub fn register(
        registry: &prometheus::Registry,
        instance_id: u64,
    ) -> Result<(), prometheus::Error> {
        let m = ROUTING_OVERHEAD_METRICS.get_or_init(|| {
            let buckets = overhead_buckets();
            let router_id = instance_id.to_string();
            let make = |suffix: &str, help: &str| {
                let name = format!("{}_{}", name_prefix::ROUTER, suffix);
                prometheus::Histogram::with_opts(
                    HistogramOpts::new(name, help)
                        .const_label(labels::ROUTER_ID, &router_id)
                        .buckets(buckets.clone()),
                )
            };
            let block_hashing = make(
                routing_overhead::BLOCK_HASHING_MS,
                "Time spent computing block hashes in milliseconds",
            )
            .expect("overhead_block_hashing_ms");
            let indexer_find_matches = make(
                routing_overhead::INDEXER_FIND_MATCHES_MS,
                "Time spent in indexer find_matches in milliseconds",
            )
            .expect("overhead_indexer_find_matches_ms");
            let seq_hashing = make(
                routing_overhead::SEQ_HASHING_MS,
                "Time spent computing sequence hashes in milliseconds",
            )
            .expect("overhead_seq_hashing_ms");
            let scheduling = make(
                routing_overhead::SCHEDULING_MS,
                "Time spent in scheduler worker selection in milliseconds",
            )
            .expect("overhead_scheduling_ms");
            let total = make(
                routing_overhead::TOTAL_MS,
                "Total routing overhead per request in milliseconds",
            )
            .expect("overhead_total_ms");
            Arc::new(Self {
                block_hashing,
                indexer_find_matches,
                seq_hashing,
                scheduling,
                total,
            })
        });
        registry.register(Box::new(m.block_hashing.clone()))?;
        registry.register(Box::new(m.indexer_find_matches.clone()))?;
        registry.register(Box::new(m.seq_hashing.clone()))?;
        registry.register(Box::new(m.scheduling.clone()))?;
        registry.register(Box::new(m.total.clone()))?;
        Ok(())
    }

    /// Returns the registered metrics if `register()` was called earlier.
    pub fn get() -> Option<Arc<Self>> {
        ROUTING_OVERHEAD_METRICS.get().cloned()
    }

    /// Observe routing overhead timings in milliseconds.
    pub fn observe(
        &self,
        hash_elapsed: Duration,
        find_matches_elapsed: Duration,
        seq_hash_elapsed: Duration,
        total_elapsed: Duration,
    ) {
        self.block_hashing
            .observe(hash_elapsed.as_secs_f64() * 1000.0);
        self.indexer_find_matches.observe(
            find_matches_elapsed
                .saturating_sub(hash_elapsed)
                .as_secs_f64()
                * 1000.0,
        );
        self.seq_hashing.observe(
            seq_hash_elapsed
                .saturating_sub(find_matches_elapsed)
                .as_secs_f64()
                * 1000.0,
        );
        self.scheduling
            .observe(total_elapsed.saturating_sub(seq_hash_elapsed).as_secs_f64() * 1000.0);
        self.total.observe(total_elapsed.as_secs_f64() * 1000.0);
    }
}

// ---------------------------------------------------------------------------
// Router request metrics (dynamo_component_router_* via MetricsHierarchy)
// ---------------------------------------------------------------------------

/// Aggregate per-request metrics observed at the router level.
///
/// Component-scoped via `from_component()` to get automatic `dynamo_component_` prefix,
/// `dynamo_namespace`/`dynamo_component`/`dynamo_endpoint` labels, and registration
/// with the DRT `MetricsRegistry` hierarchy.
///
/// # Scrapeability
///
/// - **Frontend, non-KV modes**: Always zero (registered but never populated).
/// - **Frontend, KV mode (aggregated and disaggregated)**: Available on the
///   frontend's `/metrics` endpoint (default port 8000) via the `drt_metrics`
///   bridge, populated per-request.
/// - **Standalone router** (`python -m dynamo.router`): Available on the system
///   status server when `DYN_SYSTEM_PORT` is set, populated per-request.
///
/// # When these metrics are created
///
/// Eagerly in `KvPushRouter::new()`, so they appear as zeros before any requests.
/// Both the frontend pipeline and the standalone router (via Python bindings)
/// create a `KvPushRouter`, so both get these metrics registered automatically.
///
/// # Why component-scoped
///
/// These metrics MUST be registered through the Component hierarchy (not a standalone
/// registry). In hierarchical planner deployments, the frontend's router is the global
/// entry point, but each worker pool has its own local router (e.g. prefill pool,
/// decode pool). Component-scoped metrics let each local router emit metrics with
/// distinct `dynamo_component` labels, so pools can be monitored and scaled
/// independently.
pub struct RouterRequestMetrics {
    pub requests_total: prometheus::IntCounter,
    pub time_to_first_token_seconds: prometheus::Histogram,
    pub inter_token_latency_seconds: prometheus::Histogram,
    pub input_sequence_tokens: prometheus::Histogram,
    pub output_sequence_tokens: prometheus::Histogram,
    pub kv_hit_rate: prometheus::Histogram,
}

static ROUTER_REQUEST_METRICS: OnceLock<Arc<RouterRequestMetrics>> = OnceLock::new();

impl RouterRequestMetrics {
    /// Create from a Component, memoized in a static OnceLock.
    /// Uses the MetricsHierarchy API which auto-prepends `dynamo_component_`,
    /// injects hierarchy labels, and registers with the DRT `MetricsRegistry`.
    /// Also adds `router_id` (discovery instance_id) to distinguish router instances.
    ///
    /// Called eagerly by `KvPushRouter::new()` so metrics appear as zeros at startup.
    pub fn from_component(component: &Component) -> Arc<Self> {
        ROUTER_REQUEST_METRICS
            .get_or_init(|| {
                let instance_id = component.drt().discovery().instance_id();
                let router_id = instance_id.to_string();
                let extra_labels: &[(&str, &str)] = &[(labels::ROUTER_ID, &router_id)];

                let metrics = component.metrics();
                let requests_total = metrics
                    .create_intcounter(
                        &router_metric(frontend_service::REQUESTS_TOTAL),
                        "Total number of requests processed by the router",
                        extra_labels,
                    )
                    .expect("failed to create router_requests_total");
                let time_to_first_token_seconds = metrics
                    .create_histogram(
                        &router_metric(frontend_service::TIME_TO_FIRST_TOKEN_SECONDS),
                        "Time to first token observed at the router",
                        extra_labels,
                        Some(generate_log_buckets(0.001, 480.0, 18)),
                    )
                    .expect("failed to create router_time_to_first_token_seconds");
                let inter_token_latency_seconds = metrics
                    .create_histogram(
                        &router_metric(frontend_service::INTER_TOKEN_LATENCY_SECONDS),
                        "Average inter-token latency observed at the router",
                        extra_labels,
                        Some(generate_log_buckets(0.001, 2.0, 13)),
                    )
                    .expect("failed to create router_inter_token_latency_seconds");
                let input_sequence_tokens = metrics
                    .create_histogram(
                        &router_metric(frontend_service::INPUT_SEQUENCE_TOKENS),
                        "Input sequence length in tokens observed at the router",
                        extra_labels,
                        Some(generate_log_buckets(50.0, 128000.0, 12)),
                    )
                    .expect("failed to create router_input_sequence_tokens");
                let output_sequence_tokens = metrics
                    .create_histogram(
                        &router_metric(frontend_service::OUTPUT_SEQUENCE_TOKENS),
                        "Output sequence length in tokens observed at the router",
                        extra_labels,
                        Some(generate_log_buckets(50.0, 32000.0, 10)),
                    )
                    .expect("failed to create router_output_sequence_tokens");
                let kv_hit_rate = metrics
                    .create_histogram(
                        &router_metric(frontend_service::KV_HIT_RATE),
                        "Predicted KV cache hit rate at routing time (0.0-1.0)",
                        extra_labels,
                        Some(prometheus::linear_buckets(0.0, 0.05, 21).unwrap()),
                    )
                    .expect("failed to create router_kv_hit_rate");
                Arc::new(Self {
                    requests_total,
                    time_to_first_token_seconds,
                    inter_token_latency_seconds,
                    input_sequence_tokens,
                    output_sequence_tokens,
                    kv_hit_rate,
                })
            })
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::{Encoder, TextEncoder};

    fn gather_pef(registry: &prometheus::Registry) -> String {
        let encoder = TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&registry.gather(), &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }

    #[test]
    fn test_worker_load_metrics_pef() {
        let registry = prometheus::Registry::new();
        let metrics = WorkerLoadMetrics {
            active_decode_blocks: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::FRONTEND,
                        frontend_service::WORKER_ACTIVE_DECODE_BLOCKS
                    ),
                    "Active KV cache decode blocks per worker",
                ),
                &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
            )
            .unwrap(),
            active_prefill_tokens: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::FRONTEND,
                        frontend_service::WORKER_ACTIVE_PREFILL_TOKENS
                    ),
                    "Active prefill tokens queued per worker",
                ),
                &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
            )
            .unwrap(),
            request_active_slots: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::FRONTEND,
                        frontend_service::WORKER_REQUEST_ACTIVE_SLOTS
                    ),
                    "Active backend request slots per worker",
                ),
                &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
            )
            .unwrap(),
            num_requests_waiting: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::FRONTEND,
                        frontend_service::WORKER_NUM_REQUESTS_WAITING
                    ),
                    "Backend requests waiting in the worker scheduler queue",
                ),
                &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
            )
            .unwrap(),
            request_total_slots: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::FRONTEND,
                        frontend_service::WORKER_REQUEST_TOTAL_SLOTS
                    ),
                    "Backend request slot capacity per worker",
                ),
                &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
            )
            .unwrap(),
        };
        registry
            .register(Box::new(metrics.active_decode_blocks.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.active_prefill_tokens.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.request_active_slots.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.num_requests_waiting.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.request_total_slots.clone()))
            .unwrap();

        metrics.observe(123, 0, "decode", 42, 100, Some(7), Some(3), Some(8));

        let output = gather_pef(&registry);
        let expected = "\
# HELP dynamo_frontend_worker_active_decode_blocks Active KV cache decode blocks per worker
# TYPE dynamo_frontend_worker_active_decode_blocks gauge
dynamo_frontend_worker_active_decode_blocks{dp_rank=\"0\",worker_id=\"123\",worker_type=\"decode\"} 42
# HELP dynamo_frontend_worker_active_prefill_tokens Active prefill tokens queued per worker
# TYPE dynamo_frontend_worker_active_prefill_tokens gauge
dynamo_frontend_worker_active_prefill_tokens{dp_rank=\"0\",worker_id=\"123\",worker_type=\"decode\"} 100
# HELP dynamo_frontend_worker_num_requests_waiting Backend requests waiting in the worker scheduler queue
# TYPE dynamo_frontend_worker_num_requests_waiting gauge
dynamo_frontend_worker_num_requests_waiting{dp_rank=\"0\",worker_id=\"123\",worker_type=\"decode\"} 3
# HELP dynamo_frontend_worker_request_active_slots Active backend request slots per worker
# TYPE dynamo_frontend_worker_request_active_slots gauge
dynamo_frontend_worker_request_active_slots{dp_rank=\"0\",worker_id=\"123\",worker_type=\"decode\"} 7
# HELP dynamo_frontend_worker_request_total_slots Backend request slot capacity per worker
# TYPE dynamo_frontend_worker_request_total_slots gauge
dynamo_frontend_worker_request_total_slots{dp_rank=\"0\",worker_id=\"123\",worker_type=\"decode\"} 8
";
        assert_eq!(
            output, expected,
            "\nActual PEF:\n{output}\nExpected PEF:\n{expected}"
        );
    }

    #[test]
    fn router_dp_stale_selection_returns_oldest_candidates() {
        RouterDpHealthMetrics::clear_stale_tracking_for_test();

        let worker_0 = WorkerWithDpRank::new(7, 0);
        let worker_1 = WorkerWithDpRank::new(7, 1);
        let worker_2 = WorkerWithDpRank::new(7, 2);
        RouterDpHealthMetrics::mark_first_eligible_for_test(worker_0, "decode", 0);
        RouterDpHealthMetrics::mark_first_eligible_for_test(worker_1, "decode", 20);
        RouterDpHealthMetrics::mark_first_eligible_for_test(worker_2, "decode", 0);
        RouterDpHealthMetrics::mark_last_selected_for_test(worker_2, "decode", 90);

        let workers = [worker_0, worker_1, worker_2];
        let stale =
            RouterDpHealthMetrics::oldest_stale_workers_at(workers.iter(), "decode", 100, 200);

        assert_eq!(stale, vec![worker_0]);
        RouterDpHealthMetrics::clear_stale_tracking_for_test();
    }

    #[test]
    fn router_dp_stale_selection_includes_tied_oldest_never_selected_candidates() {
        RouterDpHealthMetrics::clear_stale_tracking_for_test();

        let worker_0 = WorkerWithDpRank::new(7, 0);
        let worker_1 = WorkerWithDpRank::new(7, 1);
        let worker_2 = WorkerWithDpRank::new(7, 2);
        RouterDpHealthMetrics::mark_first_eligible_for_test(worker_0, "decode", 0);
        RouterDpHealthMetrics::mark_first_eligible_for_test(worker_1, "decode", 0);
        RouterDpHealthMetrics::mark_first_eligible_for_test(worker_2, "decode", 120);

        let workers = [worker_0, worker_1, worker_2];
        let stale =
            RouterDpHealthMetrics::oldest_stale_workers_at(workers.iter(), "decode", 100, 200);

        assert_eq!(stale.len(), 2);
        assert!(stale.contains(&worker_0));
        assert!(stale.contains(&worker_1));
        RouterDpHealthMetrics::clear_stale_tracking_for_test();
    }

    #[test]
    fn test_routing_overhead_metric_names_pef() {
        // Verify the overhead constants produce valid histogram names when
        // combined with dynamo_router_ prefix.
        let registry = prometheus::Registry::new();
        let buckets = overhead_buckets();
        let prefix = name_prefix::ROUTER;
        let name = format!("{}_{}", prefix, routing_overhead::TOTAL_MS);
        let total = prometheus::Histogram::with_opts(
            prometheus::HistogramOpts::new(
                name,
                "Total routing overhead per request in milliseconds",
            )
            .buckets(buckets),
        )
        .unwrap();
        registry.register(Box::new(total.clone())).unwrap();
        total.observe(1.5);

        let output = gather_pef(&registry);
        assert!(
            output.contains("# HELP dynamo_router_overhead_total_ms"),
            "PEF missing HELP for routing overhead metric"
        );
        assert!(
            output.contains("# TYPE dynamo_router_overhead_total_ms histogram"),
            "PEF missing TYPE for routing overhead metric"
        );
        assert!(
            output.contains("dynamo_router_overhead_total_ms_count 1"),
            "PEF missing observation count"
        );
    }

    #[test]
    fn test_routing_overhead_saturating_sub() {
        let buckets = prometheus::exponential_buckets(0.0001, 2.0, 18).unwrap();
        let make = |name: &str| {
            prometheus::Histogram::with_opts(
                prometheus::HistogramOpts::new(name, "test").buckets(buckets.clone()),
            )
            .unwrap()
        };
        let metrics = RoutingOverheadMetrics {
            block_hashing: make("test_block_hashing_ms"),
            indexer_find_matches: make("test_find_matches_ms"),
            seq_hashing: make("test_seq_hashing_ms"),
            scheduling: make("test_scheduling_ms"),
            total: make("test_total_ms"),
        };

        // Out-of-order durations: each phase < previous (would panic without saturating_sub)
        metrics.observe(
            Duration::from_millis(10),
            Duration::from_millis(5),
            Duration::from_millis(3),
            Duration::from_millis(1),
        );
        // Reaching here without panic confirms saturating_sub works
    }
}
