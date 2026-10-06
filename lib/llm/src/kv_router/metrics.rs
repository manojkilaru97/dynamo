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
//! - [`KvPublisherMetrics`]: Worker-local KV event publisher and ZMQ relay counters.
//!   Registered on the DRT `MetricsRegistry` hierarchy via `Component::metrics()`.
//!   Populated by `KvEventPublisher` and the ZMQ listener when engines publish KV
//!   events.
//!
//! The standalone router does not create `WorkerLoadMetrics` or
//! `RoutingOverheadMetrics` (those are frontend-only). It only exposes
//! `RouterRequestMetrics` and standard DRT transport metrics
//! (`dynamo_component_inflight_requests`, `dynamo_component_requests_total`, etc.)
//! via the system status server when `DYN_SYSTEM_PORT` is explicitly set.
//!
//! See also: `docs/observability/metrics.md` (Router Metrics section).

use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Duration;

use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::metrics::prometheus_names::{
    frontend_service, kv_publisher, labels, name_prefix, router, router_request, routing_overhead,
};

/// Build a router metric name: `"router_" + frontend_service_suffix`.
fn router_metric(suffix: &str) -> String {
    format!("{}{}", router_request::METRIC_PREFIX, suffix)
}
use dynamo_runtime::traits::DistributedRuntimeProvider;
use prometheus::{HistogramOpts, IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts};

use crate::http::service::metrics::generate_log_buckets;
use crate::kv_router::cache_history::{self, CacheHistoryStats};
use crate::protocols::common::timing::RequestPhase;

pub(crate) const ROUTER_WORKER_ID_LABEL: &str = "router_worker_id";
const TARGET_NAMESPACE_LABEL: &str = "target_namespace";
const TARGET_COMPONENT_LABEL: &str = "target_component";
const TARGET_ENDPOINT_LABEL: &str = "target_endpoint";

/// Buckets for CPU-bound compute phases (block hashing, sequence hashing).
fn compute_overhead_buckets() -> Vec<f64> {
    prometheus::exponential_buckets(0.001, 2.0, 15).unwrap()
}

/// Buckets for async phases (indexer find_matches, scheduling, total).
fn async_overhead_buckets() -> Vec<f64> {
    prometheus::exponential_buckets(0.01, 3.0, 17).unwrap()
}

// ---------------------------------------------------------------------------
// KV publisher metrics
// ---------------------------------------------------------------------------

/// Metrics for the KV publisher, created via the MetricsHierarchy API.
/// This provides automatic `dynamo_namespace`, `dynamo_component`, and other
/// hierarchy labels for free.
pub(crate) struct KvPublisherMetrics {
    /// Total number of raw events dropped by engines before reaching publisher.
    pub engines_dropped_events_total: IntCounter,
    /// Total number of decoded ZMQ KV events by relay stage and event type.
    pub zmq_events_total: IntCounterVec,
    /// Total number of ZMQ KV events filtered before conversion.
    pub zmq_filtered_events_total: IntCounterVec,
    /// Total number of ZMQ KV events dropped due to conversion issues.
    pub zmq_conversion_issues_total: IntCounterVec,
    /// Total number of suspicious-but-forwarded ZMQ KV events.
    pub zmq_suspicious_events_total: IntCounterVec,
}

static KV_PUBLISHER_METRICS: OnceLock<Arc<KvPublisherMetrics>> = OnceLock::new();

impl KvPublisherMetrics {
    /// Create from a Component, memoized in a static OnceLock.
    /// Uses the MetricsHierarchy API which auto-prepends `dynamo_component_`,
    /// injects hierarchy labels (including `worker_id`), and registers with the
    /// DRT `MetricsRegistry`.
    pub fn from_component(component: &Component) -> Arc<Self> {
        KV_PUBLISHER_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                let engines_dropped_events_total = metrics
                    .create_intcounter(
                        kv_publisher::ENGINES_DROPPED_EVENTS_TOTAL,
                        "Total number of raw events dropped by engines before reaching publisher (detected via event_id gaps)",
                        &[],
                    )
                    .expect("failed to create kv_publisher_engines_dropped_events_total");
                let zmq_events_total = metrics
                    .create_intcountervec(
                        kv_publisher::ZMQ_EVENTS_TOTAL,
                        "Total number of ZMQ KV events seen by the relay",
                        &["stage", "event_type"],
                        &[],
                    )
                    .expect("failed to create kv_publisher_zmq_events_total");
                let zmq_filtered_events_total = metrics
                    .create_intcountervec(
                        kv_publisher::ZMQ_FILTERED_EVENTS_TOTAL,
                        "Total number of ZMQ KV events filtered before conversion",
                        &["event_type", "reason"],
                        &[],
                    )
                    .expect("failed to create kv_publisher_zmq_filtered_events_total");
                let zmq_conversion_issues_total = metrics
                    .create_intcountervec(
                        kv_publisher::ZMQ_CONVERSION_ISSUES_TOTAL,
                        "Total number of ZMQ KV events dropped due to conversion issues",
                        &["event_type", "reason"],
                        &[],
                    )
                    .expect("failed to create kv_publisher_zmq_conversion_issues_total");
                let zmq_suspicious_events_total = metrics
                    .create_intcountervec(
                        kv_publisher::ZMQ_SUSPICIOUS_EVENTS_TOTAL,
                        "Total number of suspicious-but-forwarded ZMQ KV events",
                        &["event_type", "reason"],
                        &[],
                    )
                    .expect("failed to create kv_publisher_zmq_suspicious_events_total");

                Arc::new(Self {
                    engines_dropped_events_total,
                    zmq_events_total,
                    zmq_filtered_events_total,
                    zmq_conversion_issues_total,
                    zmq_suspicious_events_total,
                })
            })
            .clone()
    }

    /// Increment the engines dropped events counter by the given amount.
    pub fn increment_engines_dropped_events(&self, count: u64) {
        self.engines_dropped_events_total.inc_by(count);
    }

    pub fn increment_zmq_event(&self, stage: &'static str, event_type: &'static str) {
        self.zmq_events_total
            .with_label_values(&[stage, event_type])
            .inc();
    }

    pub fn increment_zmq_filtered_event(&self, event_type: &'static str, reason: &'static str) {
        self.zmq_filtered_events_total
            .with_label_values(&[event_type, reason])
            .inc();
    }

    pub fn increment_zmq_conversion_issue(&self, event_type: &'static str, reason: &'static str) {
        self.zmq_conversion_issues_total
            .with_label_values(&[event_type, reason])
            .inc();
    }

    pub fn increment_zmq_suspicious_event(&self, event_type: &'static str, reason: &'static str) {
        self.zmq_suspicious_events_total
            .with_label_values(&[event_type, reason])
            .inc();
    }
}

pub(crate) fn kv_publisher_metrics() -> Option<Arc<KvPublisherMetrics>> {
    KV_PUBLISHER_METRICS.get().cloned()
}

// ---------------------------------------------------------------------------
// Direct-ZMQ KV ingress metrics
// ---------------------------------------------------------------------------

pub(crate) struct KvZmqIngressMetrics {
    sources: IntGaugeVec,
    batches_total: IntCounter,
    lifecycle_total: IntCounterVec,
}

static KV_ZMQ_INGRESS_METRICS: OnceLock<Arc<KvZmqIngressMetrics>> = OnceLock::new();

impl KvZmqIngressMetrics {
    pub(crate) fn from_component(component: &Component) -> Arc<Self> {
        KV_ZMQ_INGRESS_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                let sources = metrics
                    .create_intgaugevec(
                        "router_kv_zmq_ingress_sources",
                        "Number of direct-ZMQ KV ingress sources by lifecycle state",
                        &["state"],
                        &[],
                    )
                    .expect("failed to create router_kv_zmq_ingress_sources gauge");
                let batches_total = metrics
                    .create_intcounter(
                        "router_kv_zmq_ingress_batches_total",
                        "Total direct-ZMQ KV ingress batches handed to WorkerQueryClient",
                        &[],
                    )
                    .expect("failed to create router_kv_zmq_ingress_batches_total counter");
                let lifecycle_total = metrics
                    .create_intcountervec(
                        "router_kv_zmq_ingress_lifecycle_total",
                        "Total direct-ZMQ KV ingress lifecycle transitions and errors",
                        &["action"],
                        &[],
                    )
                    .expect("failed to create router_kv_zmq_ingress_lifecycle_total counter");
                Arc::new(Self {
                    sources,
                    batches_total,
                    lifecycle_total,
                })
            })
            .clone()
    }

    pub(crate) fn increment_sources(&self, state: &'static str) {
        self.sources.with_label_values(&[state]).inc();
    }

    pub(crate) fn decrement_sources(&self, state: &'static str) {
        self.sources.with_label_values(&[state]).dec();
    }

    pub(crate) fn increment_batch(&self) {
        self.batches_total.inc();
    }

    pub(crate) fn increment_lifecycle(&self, action: &'static str) {
        self.lifecycle_total.with_label_values(&[action]).inc();
    }
}

// ---------------------------------------------------------------------------
// Direct-ZMQ active-sequence ingress metrics
// ---------------------------------------------------------------------------

pub(crate) struct ActiveSequenceZmqIngressMetrics {
    tracked_sources: IntGauge,
    sequence_gaps_total: IntCounter,
    out_of_order_total: IntCounter,
    reconnects_total: IntCounter,
    replacements_total: IntCounter,
    envelope_decode_errors_total: IntCounter,
    payload_decode_errors_total: IntCounter,
    identity_errors_total: IntCounter,
    forced_aborts_total: IntCounter,
}

static ACTIVE_SEQUENCE_ZMQ_INGRESS_METRICS: OnceLock<Arc<ActiveSequenceZmqIngressMetrics>> =
    OnceLock::new();

impl ActiveSequenceZmqIngressMetrics {
    pub(crate) fn from_component(component: &Component) -> Arc<Self> {
        ACTIVE_SEQUENCE_ZMQ_INGRESS_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                let counter = |name, help| {
                    metrics
                        .create_intcounter(name, help, &[])
                        .unwrap_or_else(|error| panic!("failed to create {name}: {error}"))
                };
                Arc::new(Self {
                    tracked_sources: metrics
                        .create_intgauge(
                            "router_active_sequence_zmq_ingress_sources",
                            "Number of tracked direct-ZMQ active-sequence sources",
                            &[],
                        )
                        .expect("failed to create router_active_sequence_zmq_ingress_sources"),
                    sequence_gaps_total: counter(
                        "router_active_sequence_zmq_ingress_sequence_gaps_total",
                        "Missing direct-ZMQ active-sequence envelopes inferred from sequence gaps",
                    ),
                    out_of_order_total: counter(
                        "router_active_sequence_zmq_ingress_out_of_order_total",
                        "Non-increasing direct-ZMQ active-sequence envelope sequences",
                    ),
                    reconnects_total: counter(
                        "router_active_sequence_zmq_ingress_reconnects_total",
                        "Direct-ZMQ active-sequence source reconnect attempts",
                    ),
                    replacements_total: counter(
                        "router_active_sequence_zmq_ingress_replacements_total",
                        "Direct-ZMQ active-sequence source task replacements",
                    ),
                    envelope_decode_errors_total: counter(
                        "router_active_sequence_zmq_ingress_envelope_decode_errors_total",
                        "Direct-ZMQ active-sequence envelope decode errors",
                    ),
                    payload_decode_errors_total: counter(
                        "router_active_sequence_zmq_ingress_payload_decode_errors_total",
                        "Direct-ZMQ active-sequence payload decode errors",
                    ),
                    identity_errors_total: counter(
                        "router_active_sequence_zmq_ingress_identity_errors_total",
                        "Direct-ZMQ active-sequence frame and envelope identity mismatches",
                    ),
                    forced_aborts_total: counter(
                        "router_active_sequence_zmq_ingress_forced_aborts_total",
                        "Direct-ZMQ active-sequence source tasks aborted after join timeout",
                    ),
                })
            })
            .clone()
    }

    pub(crate) fn source_started(&self) {
        self.tracked_sources.inc();
    }

    pub(crate) fn source_stopped(&self) {
        self.tracked_sources.dec();
    }

    pub(crate) fn record_gap(&self, missing: u64) {
        self.sequence_gaps_total.inc_by(missing);
    }

    pub(crate) fn record_out_of_order(&self) {
        self.out_of_order_total.inc();
    }

    pub(crate) fn record_reconnect(&self) {
        self.reconnects_total.inc();
    }

    pub(crate) fn record_replacement(&self) {
        self.replacements_total.inc();
    }

    pub(crate) fn record_envelope_decode_error(&self) {
        self.envelope_decode_errors_total.inc();
    }

    pub(crate) fn record_payload_decode_error(&self) {
        self.payload_decode_errors_total.inc();
    }

    pub(crate) fn record_identity_error(&self) {
        self.identity_errors_total.inc();
    }

    pub(crate) fn record_forced_abort(&self) {
        self.forced_aborts_total.inc();
    }
}

// ---------------------------------------------------------------------------
// Router worker status metrics (component-scoped gauges)
// ---------------------------------------------------------------------------

/// Component-scoped router gauges for worker discovery.
pub(crate) struct RouterWorkerStatusMetrics {
    pub registered: IntGaugeVec,
    pub kv_event_source_mismatch_workers: IntGaugeVec,
}

static ROUTER_WORKER_STATUS_METRICS: OnceLock<Arc<RouterWorkerStatusMetrics>> = OnceLock::new();

impl RouterWorkerStatusMetrics {
    /// Create component-scoped gauges for standalone router observability.
    ///
    /// The `MetricsHierarchy` injects labels such as `dynamo_namespace` and
    /// `dynamo_component`. It reserves `worker_id` for the metric producer, so
    /// the backend worker ID discovered by the router uses `router_worker_id`.
    pub fn from_component(component: &Component) -> Arc<Self> {
        ROUTER_WORKER_STATUS_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                let registered = metrics
                    .create_intgaugevec(
                        router::WORKER_REGISTERED,
                        "Whether the router currently has this worker/dp_rank registered (1 = registered)",
                        &[ROUTER_WORKER_ID_LABEL, labels::DP_RANK, labels::WORKER_TYPE],
                        &[],
                    )
                    .expect("failed to create router_worker_registered gauge");
                let kv_event_source_mismatch_workers = metrics
                    .create_intgaugevec(
                        router::KV_EVENT_SOURCE_MISMATCH_WORKERS,
                        "Number of serving workers with missing or ambiguous KV sources, explicitly disabled KV event publishing, or without an expected RecoveryTarget",
                        &[
                            labels::MODEL,
                            labels::WORKER_TYPE,
                            TARGET_NAMESPACE_LABEL,
                            TARGET_COMPONENT_LABEL,
                            TARGET_ENDPOINT_LABEL,
                        ],
                        &[],
                    )
                    .expect("failed to create router_kv_event_source_mismatch_workers gauge");

                Arc::new(Self {
                    registered,
                    kv_event_source_mismatch_workers,
                })
            })
            .clone()
    }

    pub fn set_registered(&self, worker_id: u64, dp_rank: u32, worker_type: &str) {
        let worker_id = worker_id.to_string();
        let dp_rank = dp_rank.to_string();
        let labels = &[worker_id.as_str(), dp_rank.as_str(), worker_type];
        self.registered.with_label_values(labels).set(1);
    }

    pub fn remove_worker(&self, worker_id: u64, dp_rank: u32, worker_type: &str) {
        let worker_id = worker_id.to_string();
        let dp_rank = dp_rank.to_string();
        let labels = &[worker_id.as_str(), dp_rank.as_str(), worker_type];
        let _ = self.registered.remove_label_values(labels);
    }

    pub fn set_kv_event_source_mismatch_workers(
        &self,
        model: &str,
        worker_type: &str,
        target_namespace: &str,
        target_component: &str,
        target_endpoint: &str,
        count: usize,
    ) {
        self.kv_event_source_mismatch_workers
            .with_label_values(&[
                model,
                worker_type,
                target_namespace,
                target_component,
                target_endpoint,
            ])
            .set(count as i64);
    }
}

// ---------------------------------------------------------------------------
// Worker load metrics (gauges)
// ---------------------------------------------------------------------------

/// Per-worker active load gauges, published by `ActiveSequencesMultiWorker`
/// and cleaned up by `KvWorkerMonitor` when workers disappear.
pub struct WorkerLoadMetrics {
    pub active_decode_blocks: IntGaugeVec,
    pub active_prefill_tokens: IntGaugeVec,
}

impl WorkerLoadMetrics {
    pub fn observe(
        &self,
        worker_id: u64,
        dp_rank: u32,
        worker_type: &str,
        active_blocks: usize,
        active_tokens: usize,
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
});

/// Register the worker load gauges with the given Prometheus registry.
/// Called during frontend HTTP service setup (`service_v2.rs`), served on port 8000.
pub fn register_worker_load_metrics(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    let m = &*WORKER_LOAD_METRICS;
    registry.register(Box::new(m.active_decode_blocks.clone()))?;
    registry.register(Box::new(m.active_prefill_tokens.clone()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Router queue metrics (gauge)
// ---------------------------------------------------------------------------

/// Gauge tracking the number of requests pending in the router's scheduler queue.
/// Labeled by `worker_type` ("prefill" or "decode") to distinguish queues in
/// disaggregated mode. At most 2 label combinations.
pub struct RouterQueueMetrics {
    pub pending_requests: IntGaugeVec,
    pub pending_isl_tokens: IntGaugeVec,
    pub pending_cached_tokens: IntGaugeVec,
    pub backpressure_total: IntCounterVec,
}

#[derive(Clone)]
pub struct RouterQueueMetricHandles {
    pub pending_requests: IntGauge,
    pub pending_isl_tokens: IntGauge,
    pub pending_cached_tokens: IntGauge,
    pub request_limit_rejections: IntCounter,
    pub raw_isl_limit_rejections: IntCounter,
    pub cached_token_limit_rejections: IntCounter,
}

pub static ROUTER_QUEUE_METRICS: LazyLock<RouterQueueMetrics> =
    LazyLock::new(|| RouterQueueMetrics {
        pending_requests: IntGaugeVec::new(
            Opts::new(
                format!(
                    "{}_{}",
                    name_prefix::FRONTEND,
                    frontend_service::ROUTER_QUEUE_PENDING_REQUESTS
                ),
                "Number of requests pending in the router scheduler queue",
            ),
            &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
        )
        .expect("Failed to create router_queue_pending_requests gauge"),
        pending_isl_tokens: IntGaugeVec::new(
            Opts::new(
                format!("{}_router_queue_pending_isl_tokens", name_prefix::FRONTEND),
                "Sum of isl_tokens for requests pending in the router scheduler queue",
            ),
            &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
        )
        .expect("Failed to create router_queue_pending_isl_tokens gauge"),
        pending_cached_tokens: IntGaugeVec::new(
            Opts::new(
                format!(
                    "{}_router_queue_pending_cached_tokens",
                    name_prefix::FRONTEND
                ),
                "Estimated cached tokens for requests pending in the router scheduler queue",
            ),
            &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
        )
        .expect("Failed to create router_queue_pending_cached_tokens gauge"),
        backpressure_total: IntCounterVec::new(
            Opts::new(
                format!("{}_router_queue_backpressure_total", name_prefix::FRONTEND),
                "Total number of router scheduler queue backpressure rejections",
            ),
            &[labels::MODEL, labels::WORKER_TYPE, "policy_class", "reason"],
        )
        .expect("Failed to create router_queue_backpressure_total counter"),
    });

impl RouterQueueMetrics {
    pub fn handles(
        &self,
        model: &str,
        worker_type: &str,
        policy_class: &str,
    ) -> RouterQueueMetricHandles {
        let queue_labels = [model, worker_type, policy_class];
        let rejection = |reason| {
            self.backpressure_total
                .with_label_values(&[model, worker_type, policy_class, reason])
        };
        RouterQueueMetricHandles {
            pending_requests: self.pending_requests.with_label_values(&queue_labels),
            pending_isl_tokens: self.pending_isl_tokens.with_label_values(&queue_labels),
            pending_cached_tokens: self.pending_cached_tokens.with_label_values(&queue_labels),
            request_limit_rejections: rejection("request_limit"),
            raw_isl_limit_rejections: rejection("raw_isl_token_limit"),
            cached_token_limit_rejections: rejection("cached_token_limit"),
        }
    }
}

/// Register the router queue gauge with the given Prometheus registry.
/// Called during frontend HTTP service setup (`service_v2.rs`), served on port 8000.
pub fn register_router_queue_metrics(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    let m = &*ROUTER_QUEUE_METRICS;
    registry.register(Box::new(m.pending_requests.clone()))?;
    registry.register(Box::new(m.pending_isl_tokens.clone()))?;
    registry.register(Box::new(m.pending_cached_tokens.clone()))?;
    registry.register(Box::new(m.backpressure_total.clone()))?;
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
    pub shared_cache_query: prometheus::Histogram,
    pub shared_cache_errors_total: prometheus::IntCounter,
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
            let compute_buckets = compute_overhead_buckets();
            let async_buckets = async_overhead_buckets();
            let router_id = instance_id.to_string();
            let make = |suffix: &str, help: &str, buckets: Vec<f64>| {
                let name = format!("{}_{}", name_prefix::ROUTER, suffix);
                prometheus::Histogram::with_opts(
                    HistogramOpts::new(name, help)
                        .const_label(labels::ROUTER_ID, &router_id)
                        .buckets(buckets),
                )
            };
            let block_hashing = make(
                routing_overhead::BLOCK_HASHING_MS,
                "Time spent computing block hashes in milliseconds",
                compute_buckets.clone(),
            )
            .expect("overhead_block_hashing_ms");
            let indexer_find_matches = make(
                routing_overhead::INDEXER_FIND_MATCHES_MS,
                "Time spent in indexer find_matches in milliseconds",
                async_buckets.clone(),
            )
            .expect("overhead_indexer_find_matches_ms");
            let seq_hashing = make(
                routing_overhead::SEQ_HASHING_MS,
                "Time spent computing sequence hashes in milliseconds",
                compute_buckets,
            )
            .expect("overhead_seq_hashing_ms");
            let scheduling = make(
                routing_overhead::SCHEDULING_MS,
                "Time spent in scheduler worker selection in milliseconds",
                async_buckets.clone(),
            )
            .expect("overhead_scheduling_ms");
            let total = make(
                routing_overhead::TOTAL_MS,
                "Total routing overhead per request in milliseconds",
                async_buckets.clone(),
            )
            .expect("overhead_total_ms");
            let shared_cache_query = make(
                routing_overhead::SHARED_CACHE_QUERY_MS,
                "Time spent querying the shared KV cache in milliseconds",
                async_buckets,
            )
            .expect("overhead_shared_cache_query_ms");
            let shared_cache_errors_total = {
                let name = format!(
                    "{}_{}",
                    name_prefix::ROUTER,
                    routing_overhead::SHARED_CACHE_ERRORS_TOTAL
                );
                prometheus::IntCounter::with_opts(
                    Opts::new(name, "Total shared cache query errors")
                        .const_label(labels::ROUTER_ID, &router_id),
                )
                .expect("shared_cache_errors_total")
            };
            Arc::new(Self {
                block_hashing,
                indexer_find_matches,
                seq_hashing,
                scheduling,
                total,
                shared_cache_query,
                shared_cache_errors_total,
            })
        });
        registry.register(Box::new(m.block_hashing.clone()))?;
        registry.register(Box::new(m.indexer_find_matches.clone()))?;
        registry.register(Box::new(m.seq_hashing.clone()))?;
        registry.register(Box::new(m.scheduling.clone()))?;
        registry.register(Box::new(m.total.clone()))?;
        registry.register(Box::new(m.shared_cache_query.clone()))?;
        registry.register(Box::new(m.shared_cache_errors_total.clone()))?;
        Ok(())
    }

    /// Returns the registered metrics if `register()` was called earlier.
    pub fn get() -> Option<Arc<Self>> {
        ROUTING_OVERHEAD_METRICS.get().cloned()
    }

    /// Observe routing overhead timings in milliseconds.
    ///
    /// `indexer_duration` and `shared_cache_duration` are independent wall-clock times
    /// measured inside the `tokio::join!` block. They run in parallel, so
    /// `find_matches_elapsed >= max(indexer_duration, shared_cache_duration)`.
    pub fn observe(
        &self,
        hash_elapsed: Duration,
        seq_hash_elapsed: Duration,
        indexer_duration: Duration,
        shared_cache_duration: Option<Duration>,
        find_matches_elapsed: Duration,
        total_elapsed: Duration,
    ) {
        self.block_hashing
            .observe(hash_elapsed.as_secs_f64() * 1000.0);
        self.seq_hashing
            .observe(seq_hash_elapsed.saturating_sub(hash_elapsed).as_secs_f64() * 1000.0);
        self.indexer_find_matches
            .observe(indexer_duration.as_secs_f64() * 1000.0);
        if let Some(sc_duration) = shared_cache_duration {
            self.shared_cache_query
                .observe(sc_duration.as_secs_f64() * 1000.0);
        }
        self.scheduling.observe(
            total_elapsed
                .saturating_sub(find_matches_elapsed)
                .as_secs_f64()
                * 1000.0,
        );
        self.total.observe(total_elapsed.as_secs_f64() * 1000.0);
    }

    /// Increment the shared cache error counter.
    pub fn inc_shared_cache_errors(&self) {
        self.shared_cache_errors_total.inc();
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
/// registry). In global planner deployments, the frontend's router is the global
/// entry point, but each worker pool has its own local router (e.g. prefill pool,
/// decode pool). Component-scoped metrics let each local router emit metrics with
/// distinct `dynamo_component` labels, so pools can be monitored and scaled
/// independently.
pub struct RouterRequestMetrics {
    pub requests_total: prometheus::IntCounter,
    pub time_to_first_token_seconds: prometheus::Histogram,
    pub inter_token_latency_seconds: prometheus::Histogram,
    /// Prompt tokens per tracked routing attempt (funnel F0 is its `_sum`); labels `phase`
    /// (aggregated | prefill | decode) and `model`. Query-only selections are excluded.
    pub input_sequence_tokens: prometheus::HistogramVec,
    pub output_sequence_tokens: prometheus::Histogram,
    pub kv_hit_rate: prometheus::Histogram,
    pub kv_transfer_estimated_latency_seconds: prometheus::Histogram,
    pub shared_cache_hit_rate: prometheus::Histogram,
    pub shared_cache_beyond_blocks: prometheus::Histogram,
    /// Raw cached prefix tokens on the best eligible worker at selection (funnel F2), one
    /// observation per tracked attempt; labels `phase` and `model`.
    pub kv_best_eligible_cached_prefix_tokens: IntCounterVec,
    /// Raw cached prefix tokens on the selected worker and DP rank at selection (funnel F3);
    /// same labels.
    pub kv_selected_cached_prefix_tokens: IntCounterVec,
    /// Worker-reported reused tokens, counted once per attempt (funnel F4); same labels.
    pub kv_worker_reused_tokens: IntCounterVec,
    /// Opt-in F1 history series; `None` unless `DYN_ROUTER_CACHE_REUSE_HISTORY` is set.
    pub(crate) cache_history: Option<CacheHistoryMetrics>,
}

#[derive(Clone)]
pub(crate) struct CacheHistoryMetrics {
    observation_input_tokens_total: IntCounter,
    f0_tokens_total: IntCounter,
    f1_tokens_total: IntCounter,
    complete_observations_total: IntCounter,
    incomplete_observations_total: IntCounter,
    retained_entries: IntGauge,
    represented_tokens: IntGauge,
    estimated_retained_bytes: IntGauge,
    capacity_entries: IntGauge,
    capacity_bytes: IntGauge,
}

/// Label carrying the routing phase on the cache-reuse funnel series.
pub(crate) const KV_PHASE_LABEL: &str = "phase";

static ROUTER_REQUEST_METRICS: OnceLock<Arc<RouterRequestMetrics>> = OnceLock::new();
static ROUTER_REQUESTS_STARTED_TOTAL: OnceLock<prometheus::IntCounter> = OnceLock::new();

impl RouterRequestMetrics {
    /// Returns the registered metrics if `from_component()` was called earlier.
    pub fn get() -> Option<Arc<Self>> {
        ROUTER_REQUEST_METRICS.get().cloned()
    }

    /// Total requests admitted by the router scheduler.
    pub fn requests_started_total(&self) -> &prometheus::IntCounter {
        ROUTER_REQUESTS_STARTED_TOTAL
            .get()
            .expect("router request metrics must be initialized")
    }

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

                let requests_started_total = component
                    .metrics()
                    .create_intcounter(
                        &router_metric(frontend_service::REQUESTS_STARTED_TOTAL),
                        "Total number of requests admitted by the router scheduler",
                        extra_labels,
                    )
                    .expect("failed to create router_requests_started_total");
                assert!(
                    ROUTER_REQUESTS_STARTED_TOTAL
                        .set(requests_started_total)
                        .is_ok(),
                    "router_requests_started_total already initialized"
                );
                Arc::new(Self::build(component, extra_labels))
            })
            .clone()
    }

    fn build<H: MetricsHierarchy>(hierarchy: &H, extra_labels: &[(&str, &str)]) -> Self {
        let metrics = hierarchy.metrics();
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
            .create_histogramvec(
                &router_metric(frontend_service::INPUT_SEQUENCE_TOKENS),
                "Input sequence length in tokens observed at the router",
                &[KV_PHASE_LABEL, labels::MODEL],
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
        let kv_transfer_estimated_latency_seconds = metrics
            .create_histogram(
                &router_metric(frontend_service::KV_TRANSFER_ESTIMATED_LATENCY_SECONDS),
                "Upper-bound estimation of KV cache transfer latency in disaggregated serving (prefill_complete to first_token)",
                extra_labels,
                Some(generate_log_buckets(0.001, 10.0, 15)),
            )
            .expect("failed to create router_kv_transfer_estimated_latency_seconds");
        let shared_cache_hit_rate = metrics
            .create_histogram(
                &router_metric(frontend_service::SHARED_CACHE_HIT_RATE),
                "Fraction of request blocks found in the shared KV cache (0.0-1.0)",
                extra_labels,
                Some(prometheus::linear_buckets(0.0, 0.05, 21).unwrap()),
            )
            .expect("failed to create router_shared_cache_hit_rate");
        let shared_cache_beyond_blocks = metrics
            .create_histogram(
                &router_metric(frontend_service::SHARED_CACHE_BEYOND_BLOCKS),
                "Shared cache blocks beyond device overlap for the selected worker",
                extra_labels,
                Some(prometheus::exponential_buckets(1.0, 2.0, 12).unwrap()),
            )
            .expect("failed to create router_shared_cache_beyond_blocks");
        let kv_tokens_counter = |name: &str, help: &str| {
            metrics
                .create_intcountervec(
                    &router_metric(name),
                    help,
                    &[KV_PHASE_LABEL, labels::MODEL],
                    extra_labels,
                )
                .expect("failed to create router KV token counter")
        };
        let kv_best_eligible_cached_prefix_tokens = kv_tokens_counter(
            frontend_service::KV_BEST_ELIGIBLE_CACHED_PREFIX_TOKENS_TOTAL,
            "Raw cached prefix tokens on the best eligible worker at selection time",
        );
        let kv_selected_cached_prefix_tokens = kv_tokens_counter(
            frontend_service::KV_SELECTED_CACHED_PREFIX_TOKENS_TOTAL,
            "Raw cached prefix tokens on the selected worker and DP rank",
        );
        let kv_worker_reused_tokens = kv_tokens_counter(
            frontend_service::KV_WORKER_REUSED_TOKENS_TOTAL,
            "Worker-reported reused tokens per routing attempt",
        );
        let cache_history = cache_history::enabled()
            .then(|| Self::build_cache_history_metrics(hierarchy, extra_labels));
        Self {
            requests_total,
            time_to_first_token_seconds,
            inter_token_latency_seconds,
            input_sequence_tokens,
            output_sequence_tokens,
            kv_hit_rate,
            kv_transfer_estimated_latency_seconds,
            shared_cache_hit_rate,
            shared_cache_beyond_blocks,
            kv_best_eligible_cached_prefix_tokens,
            kv_selected_cached_prefix_tokens,
            kv_worker_reused_tokens,
            cache_history,
        }
    }

    fn build_cache_history_metrics<H: MetricsHierarchy>(
        hierarchy: &H,
        extra_labels: &[(&str, &str)],
    ) -> CacheHistoryMetrics {
        let metrics = hierarchy.metrics();
        let observation_input_tokens_total = metrics
            .create_intcounter(
                &router_metric(frontend_service::CACHE_LOSS_OBSERVATION_INPUT_TOKENS_TOTAL),
                "Input tokens for cache-reuse observations started by the router",
                extra_labels,
            )
            .expect("failed to create router_cache_loss_observation_input_tokens_total");
        let funnel_tokens_total = metrics
            .create_intcountervec(
                &router_metric(frontend_service::CACHE_LOSS_FUNNEL_TOKENS_TOTAL),
                "Raw token observations at each cache-reuse funnel stage",
                &["stage"],
                extra_labels,
            )
            .expect("failed to create router_cache_loss_funnel_tokens_total");
        let f0_tokens_total = funnel_tokens_total.with_label_values(&["f0"]);
        let f1_tokens_total = metrics
            .create_intcounter(
                &router_metric(frontend_service::KV_HISTORY_CACHED_PREFIX_TOKENS_TOTAL),
                "Prompt-prefix tokens previously seen in this router's bounded history",
                extra_labels,
            )
            .expect("failed to create router_kv_history_cached_prefix_tokens_total");
        let observations_total = metrics
            .create_intcountervec(
                &router_metric(frontend_service::CACHE_LOSS_OBSERVATIONS_TOTAL),
                "Cache-reuse observations by completion status",
                &["result"],
                extra_labels,
            )
            .expect("failed to create router_cache_loss_observations_total");
        let complete_observations_total = observations_total.with_label_values(&["complete"]);
        let incomplete_observations_total = observations_total.with_label_values(&["incomplete"]);
        let gauge = |suffix, help| {
            metrics
                .create_intgauge(&router_metric(suffix), help, extra_labels)
                .unwrap_or_else(|error| panic!("failed to create {suffix}: {error}"))
        };
        CacheHistoryMetrics {
            observation_input_tokens_total,
            f0_tokens_total,
            f1_tokens_total,
            complete_observations_total,
            incomplete_observations_total,
            retained_entries: gauge(
                frontend_service::CACHE_LOSS_HISTORY_UNIQUE_HASHES,
                "Distinct canonical block hashes retained by the last cache-history-enabled router to update this process-global gauge",
            ),
            represented_tokens: gauge(
                frontend_service::CACHE_LOSS_HISTORY_REPRESENTED_TOKENS,
                "Tokens represented by the last cache-history-enabled router to update this process-global gauge",
            ),
            estimated_retained_bytes: gauge(
                frontend_service::CACHE_LOSS_HISTORY_ESTIMATED_BYTES,
                "Estimated retained bytes for the last cache-history-enabled router to update this process-global gauge",
            ),
            capacity_entries: gauge(
                frontend_service::CACHE_LOSS_HISTORY_CAPACITY_BLOCKS,
                "Per-router distinct-block capacity; process-global gauge reports the last initialized cache-history-enabled router",
            ),
            capacity_bytes: gauge(
                frontend_service::CACHE_LOSS_HISTORY_CAPACITY_BYTES,
                "Per-router byte budget; process-global gauge reports the last initialized cache-history-enabled router",
            ),
        }
    }

    /// `(f0, f1, complete, incomplete, retained_entries)` of the F1 series, when enabled.
    #[cfg(test)]
    pub(crate) fn cache_history_values_for_test(&self) -> Option<(u64, u64, u64, u64, i64)> {
        self.cache_history.as_ref().map(|metrics| {
            (
                metrics.f0_tokens_total.get(),
                metrics.f1_tokens_total.get(),
                metrics.complete_observations_total.get(),
                metrics.incomplete_observations_total.get(),
                metrics.retained_entries.get(),
            )
        })
    }

    pub(crate) fn observe_cache_history_input(&self, prompt_tokens: u64) {
        let Some(metrics) = &self.cache_history else {
            return;
        };
        metrics.observation_input_tokens_total.inc_by(prompt_tokens);
    }

    pub(crate) fn observe_cache_history_complete(
        &self,
        prompt_tokens: u64,
        previously_seen_tokens: u64,
    ) {
        let Some(metrics) = &self.cache_history else {
            return;
        };
        metrics.f0_tokens_total.inc_by(prompt_tokens);
        metrics.f1_tokens_total.inc_by(previously_seen_tokens);
        metrics.complete_observations_total.inc();
    }

    pub(crate) fn observe_cache_history_incomplete(&self) {
        if let Some(metrics) = &self.cache_history {
            metrics.incomplete_observations_total.inc();
        }
    }

    pub(crate) fn set_cache_history_capacity(&self, stats: CacheHistoryStats) {
        let Some(metrics) = &self.cache_history else {
            return;
        };
        let gauge = |value: usize| i64::try_from(value).unwrap_or(i64::MAX);
        metrics.capacity_entries.set(gauge(stats.capacity_entries));
        metrics.capacity_bytes.set(gauge(stats.capacity_bytes));
    }

    pub(crate) fn set_cache_history_retained(&self, stats: CacheHistoryStats) {
        let Some(metrics) = &self.cache_history else {
            return;
        };
        let gauge = |value: usize| i64::try_from(value).unwrap_or(i64::MAX);
        metrics.retained_entries.set(gauge(stats.retained_entries));
        metrics
            .represented_tokens
            .set(i64::try_from(stats.represented_tokens).unwrap_or(i64::MAX));
        metrics
            .estimated_retained_bytes
            .set(gauge(stats.estimated_retained_bytes));
    }

    /// Fresh metrics on an isolated hierarchy, for tests that inspect exported series.
    #[cfg(test)]
    pub(crate) fn for_test<H: MetricsHierarchy>(hierarchy: &H) -> Arc<Self> {
        Arc::new(Self::build(hierarchy, &[]))
    }

    /// Record the prompt length of one tracked routing attempt (funnel F0).
    pub fn observe_input_sequence_tokens(&self, phase: RequestPhase, model: &str, tokens: usize) {
        self.input_sequence_tokens
            .with_label_values(&[phase.as_str(), model])
            .observe(tokens as f64);
    }

    /// Record the router's estimates for one tracked attempt at selection time and
    /// return its worker-reuse counter, already exported at zero for backends that
    /// never report.
    pub fn observe_kv_route_estimate(
        &self,
        phase: RequestPhase,
        model: &str,
        best_tokens: u64,
        selected_tokens: u64,
    ) -> IntCounter {
        let labels = &[phase.as_str(), model];
        self.kv_best_eligible_cached_prefix_tokens
            .with_label_values(labels)
            .inc_by(best_tokens);
        self.kv_selected_cached_prefix_tokens
            .with_label_values(labels)
            .inc_by(selected_tokens);
        self.kv_worker_reused_tokens.with_label_values(labels)
    }
}

pub struct RemoteIndexerMetrics {
    pub query_failures_total: prometheus::IntCounter,
    pub write_failures_total: prometheus::IntCounter,
}

static REMOTE_INDEXER_METRICS: OnceLock<Arc<RemoteIndexerMetrics>> = OnceLock::new();

impl RemoteIndexerMetrics {
    pub fn from_component(component: &Component) -> Arc<Self> {
        REMOTE_INDEXER_METRICS
            .get_or_init(|| {
                let instance_id = component.drt().discovery().instance_id();
                let router_id = instance_id.to_string();
                let extra_labels: &[(&str, &str)] = &[(labels::ROUTER_ID, &router_id)];

                let metrics = component.metrics();
                let query_failures_total = metrics
                    .create_intcounter(
                        router::REMOTE_INDEXER_QUERY_FAILURES_TOTAL,
                        "Total number of remote indexer overlap queries that failed",
                        extra_labels,
                    )
                    .expect("failed to create router_remote_indexer_query_failures_total");
                let write_failures_total = metrics
                    .create_intcounter(
                        router::REMOTE_INDEXER_WRITE_FAILURES_TOTAL,
                        "Total number of remote indexer routing-decision writes that failed",
                        extra_labels,
                    )
                    .expect("failed to create router_remote_indexer_write_failures_total");

                Arc::new(Self {
                    query_failures_total,
                    write_failures_total,
                })
            })
            .clone()
    }

    pub fn increment_query_failures(&self) {
        self.query_failures_total.inc();
    }

    pub fn increment_write_failures(&self) {
        self.write_failures_total.inc();
    }
}

#[cfg(test)]
pub(crate) mod test_hierarchy {
    use dynamo_runtime::metrics::{MetricsHierarchy, MetricsRegistry};

    /// A root metrics hierarchy with its own registry and no runtime, for exporting
    /// router metrics in isolation.
    #[derive(Default)]
    pub(crate) struct IsolatedHierarchy {
        pub(crate) registry: MetricsRegistry,
    }

    impl MetricsHierarchy for IsolatedHierarchy {
        fn basename(&self) -> String {
            String::new()
        }

        fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
            Vec::new()
        }

        fn get_metrics_registry(&self) -> &MetricsRegistry {
            &self.registry
        }
    }

    impl IsolatedHierarchy {
        pub(crate) fn exposition(&self) -> String {
            self.registry.prometheus_expfmt_combined().unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::{Encoder, TextEncoder};

    #[tokio::test]
    async fn router_request_metrics_register_with_component_labels() {
        use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};

        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let component = distributed
            .namespace("funnel-labels".to_string())
            .unwrap()
            .component("frontend".to_string())
            .unwrap();
        let metrics = RouterRequestMetrics::build(&component, &[(labels::ROUTER_ID, "291")]);
        metrics
            .observe_kv_route_estimate(RequestPhase::Prefill, "m", 96, 64)
            .inc_by(72);
        metrics.observe_input_sequence_tokens(RequestPhase::Prefill, "m", 100);
        let exposition = component
            .get_metrics_registry()
            .prometheus_expfmt_combined()
            .unwrap();
        for name in [
            "input_sequence_tokens_sum",
            "kv_best_eligible_cached_prefix_tokens_total",
            "kv_selected_cached_prefix_tokens_total",
            "kv_worker_reused_tokens_total",
        ] {
            let line = exposition
                .lines()
                .find(|line| line.starts_with(&format!("dynamo_component_router_{name}{{")))
                .unwrap_or_else(|| panic!("missing {name} in:\n{exposition}"));
            for label in [
                "dynamo_namespace=\"funnel_labels\"",
                "dynamo_component=\"frontend\"",
                "router_id=\"291\"",
                "phase=\"prefill\"",
                "model=\"m\"",
            ] {
                assert!(line.contains(label), "missing {label} on {line}");
            }
        }
        drop(distributed);
        runtime.shutdown();
    }

    #[test]
    fn kv_estimates_and_input_tokens_have_matching_labels() {
        let hierarchy = test_hierarchy::IsolatedHierarchy::default();
        let metrics = RouterRequestMetrics::for_test(&hierarchy);
        for (phase, model, input, best) in [
            (RequestPhase::Prefill, "m", 100, 96),
            (RequestPhase::Decode, "m", 200, 150),
            (RequestPhase::Prefill, "other", 300, 250),
        ] {
            metrics.observe_input_sequence_tokens(phase, model, input);
            metrics.observe_kv_route_estimate(phase, model, best as u64, best as u64);
            let output = hierarchy.exposition();
            let phase = phase.as_str();
            assert!(output.contains(&format!("router_input_sequence_tokens_sum{{model=\"{model}\",phase=\"{phase}\"}} {input}\n")), "{output}");
            assert!(output.contains(&format!("router_kv_best_eligible_cached_prefix_tokens_total{{model=\"{model}\",phase=\"{phase}\"}} {best}\n")), "{output}");
            assert!(output.contains(&format!("router_kv_selected_cached_prefix_tokens_total{{model=\"{model}\",phase=\"{phase}\"}} {best}\n")), "{output}");
        }
    }

    /// Per-attempt cost of the always-on funnel series (F0 histogram + F2/F3 counters + the
    /// F4 handle) against the previous unlabelled F0 histogram. Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_kv_funnel_metric_observation() {
        use std::hint::black_box;
        use std::time::Instant;

        let hierarchy = test_hierarchy::IsolatedHierarchy::default();
        let metrics = RouterRequestMetrics::for_test(&hierarchy);
        let legacy = prometheus::Histogram::with_opts(
            HistogramOpts::new("legacy_isl", "legacy")
                .buckets(generate_log_buckets(50.0, 128000.0, 12)),
        )
        .unwrap();
        let model = "private/nvidia/nemotron-3.5-super-120b-a12b";
        let iterations = 2_000_000;
        for _ in 0..3 {
            let start = Instant::now();
            for index in 0..iterations {
                legacy.observe(black_box(index as f64));
            }
            let legacy_ns = start.elapsed().as_nanos() as f64 / iterations as f64;
            let start = Instant::now();
            for index in 0..iterations {
                metrics.observe_input_sequence_tokens(
                    RequestPhase::Aggregated,
                    model,
                    black_box(index),
                );
                black_box(metrics.observe_kv_route_estimate(
                    RequestPhase::Aggregated,
                    model,
                    black_box(index as u64),
                    black_box(index as u64),
                ));
            }
            let funnel_ns = start.elapsed().as_nanos() as f64 / iterations as f64;
            println!(
                "BENCH funnel_metrics legacy_f0_ns={legacy_ns:.1} funnel_f0_f2_f3_f4_ns={funnel_ns:.1} delta_ns={:.1}",
                funnel_ns - legacy_ns
            );
        }
    }

    #[test]
    fn missing_worker_reports_export_zero_reused_tokens() {
        let hierarchy = test_hierarchy::IsolatedHierarchy::default();
        let metrics = RouterRequestMetrics::for_test(&hierarchy);
        let reused = metrics.observe_kv_route_estimate(RequestPhase::Aggregated, "m", 96, 64);
        let output = hierarchy.exposition();
        assert!(
            output.contains("kv_worker_reused_tokens_total{model=\"m\",phase=\"aggregated\"} 0"),
            "{output}"
        );
        reused.inc_by(72);
        let output = hierarchy.exposition();
        assert!(
            output.contains("kv_worker_reused_tokens_total{model=\"m\",phase=\"aggregated\"} 72"),
            "{output}"
        );
        assert!(
            output.contains("dynamo_component_router_kv_best_eligible_cached_prefix_tokens_total"),
            "{output}"
        );
    }

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
        };
        registry
            .register(Box::new(metrics.active_decode_blocks.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.active_prefill_tokens.clone()))
            .unwrap();

        metrics.observe(123, 0, "decode", 42, 100);

        let output = gather_pef(&registry);
        let expected = "\
# HELP dynamo_frontend_worker_active_decode_blocks Active KV cache decode blocks per worker
# TYPE dynamo_frontend_worker_active_decode_blocks gauge
dynamo_frontend_worker_active_decode_blocks{dp_rank=\"0\",worker_id=\"123\",worker_type=\"decode\"} 42
# HELP dynamo_frontend_worker_active_prefill_tokens Active prefill tokens queued per worker
# TYPE dynamo_frontend_worker_active_prefill_tokens gauge
dynamo_frontend_worker_active_prefill_tokens{dp_rank=\"0\",worker_id=\"123\",worker_type=\"decode\"} 100
";
        assert_eq!(
            output, expected,
            "\nActual PEF:\n{output}\nExpected PEF:\n{expected}"
        );
    }

    #[test]
    fn test_router_worker_status_metrics_pef() {
        let registry = prometheus::Registry::new();
        let metrics = RouterWorkerStatusMetrics {
            registered: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::COMPONENT,
                        router::WORKER_REGISTERED
                    ),
                    "Whether the router currently has this worker/dp_rank registered (1 = registered)",
                ),
                &[ROUTER_WORKER_ID_LABEL, labels::DP_RANK, labels::WORKER_TYPE],
            )
            .unwrap(),
            kv_event_source_mismatch_workers: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::COMPONENT,
                        router::KV_EVENT_SOURCE_MISMATCH_WORKERS
                    ),
                    "Number of workers expected to publish KV events but missing worker-local KV indexer query endpoints",
                ),
                &[
                    labels::MODEL,
                    labels::WORKER_TYPE,
                    TARGET_NAMESPACE_LABEL,
                    TARGET_COMPONENT_LABEL,
                    TARGET_ENDPOINT_LABEL,
                ],
            )
            .unwrap(),
        };
        registry
            .register(Box::new(metrics.registered.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.kv_event_source_mismatch_workers.clone()))
            .unwrap();

        metrics.set_registered(123, 0, "decode");
        metrics.set_kv_event_source_mismatch_workers(
            "model-a", "decode", "ns-a", "decode", "generate", 2,
        );
        metrics.set_kv_event_source_mismatch_workers(
            "model-a", "prefill", "ns-a", "prefill", "generate", 0,
        );

        let output = gather_pef(&registry);
        assert!(
            output.contains(
                "dynamo_component_router_worker_registered{dp_rank=\"0\",router_worker_id=\"123\",worker_type=\"decode\"} 1"
            ),
            "\nActual PEF:\n{output}"
        );
        assert!(
            output.contains(
                "dynamo_component_router_kv_event_source_mismatch_workers{model=\"model-a\",target_component=\"decode\",target_endpoint=\"generate\",target_namespace=\"ns-a\",worker_type=\"decode\"} 2"
            ),
            "\nActual PEF:\n{output}"
        );
        assert!(
            output.contains(
                "dynamo_component_router_kv_event_source_mismatch_workers{model=\"model-a\",target_component=\"prefill\",target_endpoint=\"generate\",target_namespace=\"ns-a\",worker_type=\"prefill\"} 0"
            ),
            "\nActual PEF:\n{output}"
        );

        metrics.remove_worker(123, 0, "decode");
        let output = gather_pef(&registry);
        assert!(
            !output.contains("router_worker_id=\"123\""),
            "\nActual PEF after remove:\n{output}"
        );
    }

    #[test]
    fn test_router_queue_metrics_pef() {
        let registry = prometheus::Registry::new();
        let metrics = RouterQueueMetrics {
            pending_requests: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::FRONTEND,
                        frontend_service::ROUTER_QUEUE_PENDING_REQUESTS
                    ),
                    "Number of requests pending in the router scheduler queue",
                ),
                &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
            )
            .unwrap(),
            pending_isl_tokens: IntGaugeVec::new(
                Opts::new(
                    format!("{}_router_queue_pending_isl_tokens", name_prefix::FRONTEND),
                    "Sum of isl_tokens for requests pending in the router scheduler queue",
                ),
                &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
            )
            .unwrap(),
            pending_cached_tokens: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_router_queue_pending_cached_tokens",
                        name_prefix::FRONTEND
                    ),
                    "Estimated cached tokens for requests pending in the router scheduler queue",
                ),
                &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
            )
            .unwrap(),
            backpressure_total: IntCounterVec::new(
                Opts::new(
                    format!("{}_router_queue_backpressure_total", name_prefix::FRONTEND),
                    "Total number of router scheduler queue backpressure rejections",
                ),
                &[labels::MODEL, labels::WORKER_TYPE, "policy_class", "reason"],
            )
            .unwrap(),
        };
        registry
            .register(Box::new(metrics.pending_requests.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.pending_isl_tokens.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.pending_cached_tokens.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.backpressure_total.clone()))
            .unwrap();

        let handles = metrics.handles("model", "decode", "default");
        handles.pending_requests.set(5);
        handles.pending_isl_tokens.set(1024);
        handles.pending_cached_tokens.set(512);

        let output = gather_pef(&registry);
        let expected = "\
# HELP dynamo_frontend_router_queue_backpressure_total Total number of router scheduler queue backpressure rejections
# TYPE dynamo_frontend_router_queue_backpressure_total counter
dynamo_frontend_router_queue_backpressure_total{model=\"model\",policy_class=\"default\",reason=\"cached_token_limit\",worker_type=\"decode\"} 0
dynamo_frontend_router_queue_backpressure_total{model=\"model\",policy_class=\"default\",reason=\"raw_isl_token_limit\",worker_type=\"decode\"} 0
dynamo_frontend_router_queue_backpressure_total{model=\"model\",policy_class=\"default\",reason=\"request_limit\",worker_type=\"decode\"} 0
# HELP dynamo_frontend_router_queue_pending_cached_tokens Estimated cached tokens for requests pending in the router scheduler queue
# TYPE dynamo_frontend_router_queue_pending_cached_tokens gauge
dynamo_frontend_router_queue_pending_cached_tokens{model=\"model\",policy_class=\"default\",worker_type=\"decode\"} 512
# HELP dynamo_frontend_router_queue_pending_isl_tokens Sum of isl_tokens for requests pending in the router scheduler queue
# TYPE dynamo_frontend_router_queue_pending_isl_tokens gauge
dynamo_frontend_router_queue_pending_isl_tokens{model=\"model\",policy_class=\"default\",worker_type=\"decode\"} 1024
# HELP dynamo_frontend_router_queue_pending_requests Number of requests pending in the router scheduler queue
# TYPE dynamo_frontend_router_queue_pending_requests gauge
dynamo_frontend_router_queue_pending_requests{model=\"model\",policy_class=\"default\",worker_type=\"decode\"} 5
";
        assert_eq!(
            output, expected,
            "\nActual PEF:\n{output}\nExpected PEF:\n{expected}"
        );
    }

    #[test]
    fn test_routing_overhead_metric_names_pef() {
        // Verify the overhead constants produce valid histogram names when
        // combined with dynamo_router_ prefix.
        let registry = prometheus::Registry::new();
        let buckets = async_overhead_buckets();
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
            shared_cache_query: make("test_shared_cache_query_ms"),
            shared_cache_errors_total: prometheus::IntCounter::new(
                "test_shared_cache_errors_total",
                "test",
            )
            .unwrap(),
        };

        // Out-of-order cumulative durations: each phase < previous (would panic without saturating_sub)
        metrics.observe(
            Duration::from_millis(10),
            Duration::from_millis(5),
            Duration::from_millis(4),
            None,
            Duration::from_millis(3),
            Duration::from_millis(1),
        );
        // Reaching here without panic confirms saturating_sub works
    }

    #[test]
    fn test_kv_transfer_estimated_latency_metric_pef() {
        // Verify the metric name is correctly composed from the constant
        // and produces valid PEF when observed.
        let registry = prometheus::Registry::new();
        let name = format!(
            "{}{}",
            router_request::METRIC_PREFIX,
            frontend_service::KV_TRANSFER_ESTIMATED_LATENCY_SECONDS,
        );
        let buckets = generate_log_buckets(0.001, 10.0, 15);
        let histogram = prometheus::Histogram::with_opts(
            prometheus::HistogramOpts::new(
                &name,
                "Upper-bound estimation of KV cache transfer latency in disaggregated serving (prefill_complete to first_token)",
            )
            .buckets(buckets),
        )
        .unwrap();
        registry.register(Box::new(histogram.clone())).unwrap();

        // Observe a 5ms latency
        histogram.observe(0.005);

        let output = gather_pef(&registry);
        assert!(
            output.contains("# HELP router_kv_transfer_estimated_latency_seconds"),
            "PEF missing HELP line. Got:\n{output}"
        );
        assert!(
            output.contains("# TYPE router_kv_transfer_estimated_latency_seconds histogram"),
            "PEF missing TYPE line. Got:\n{output}"
        );
        assert!(
            output.contains("router_kv_transfer_estimated_latency_seconds_count 1"),
            "PEF missing observation count. Got:\n{output}"
        );
        assert!(
            output.contains("router_kv_transfer_estimated_latency_seconds_sum 0.005"),
            "PEF missing observation sum. Got:\n{output}"
        );
    }
}
