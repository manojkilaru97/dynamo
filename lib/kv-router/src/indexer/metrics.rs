// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::time::{Duration, Instant};

#[cfg(feature = "runtime-protocols")]
use std::sync::Arc;
use std::sync::Mutex;
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
use std::sync::OnceLock;

use dashmap::DashMap;
#[cfg(feature = "runtime-protocols")]
use dynamo_runtime::component::Component;
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
use dynamo_runtime::metrics::MetricsHierarchy;
#[cfg(feature = "metrics")]
use prometheus::{GaugeVec, IntCounterVec, Opts};

use crate::protocols::{KvCacheEventData, KvCacheEventError, WorkerId};

/// Metrics for the KV Indexer.
#[cfg_attr(not(feature = "metrics"), derive(Default))]
pub struct KvIndexerMetrics {
    #[cfg(feature = "metrics")]
    pub kv_cache_events_applied: IntCounterVec,
    #[cfg(feature = "metrics")]
    pub kv_cache_events_applied_by_worker: IntCounterVec,
    #[cfg(feature = "metrics")]
    pub kv_cache_events_dropped_total: IntCounterVec,
    #[cfg(feature = "metrics")]
    pub kv_cache_worker_quarantines_total: IntCounterVec,
    #[cfg(feature = "metrics")]
    pub kv_cache_worker_quarantined: GaugeVec,

    quarantine_policy: KvMissQuarantinePolicy,
    worker_states: DashMap<WorkerId, WorkerQuarantineState>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KvMissQuarantinePolicy {
    pub miss_threshold: Option<u64>,
    pub window: Duration,
    pub cooldown: Duration,
}

impl KvMissQuarantinePolicy {
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn is_enabled(&self) -> bool {
        self.miss_threshold.is_some()
    }
}

#[derive(Debug, Default)]
struct WorkerQuarantineState {
    recent_miss_timestamps: Mutex<VecDeque<Instant>>,
    quarantined_until: Mutex<Option<Instant>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerKvEventAction {
    Apply,
    DropQuarantined,
    NewlyQuarantined,
}

/// Metric status labels.
pub const METRIC_STATUS_OK: &str = "ok";
pub const METRIC_STATUS_PARENT_NOT_FOUND: &str = "parent_block_not_found";
pub const METRIC_STATUS_BLOCK_NOT_FOUND: &str = "block_not_found";
pub const METRIC_STATUS_INVALID_BLOCK: &str = "invalid_block";
pub const METRIC_STATUS_DROPPED_QUARANTINED: &str = "dropped_quarantined";

/// Metric event labels.
pub const METRIC_EVENT_STORED: &str = "stored";
pub const METRIC_EVENT_REMOVED: &str = "removed";
pub const METRIC_EVENT_CLEARED: &str = "cleared";

/// Metric reason labels.
pub const METRIC_REASON_KV_MISS_SPIKE: &str = "kv_miss_spike";

#[cfg(feature = "metrics")]
const KV_CACHE_EVENTS_APPLIED_SUFFIX: &str = "kv_cache_events_applied";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENTS_APPLIED_NAME: &str = "dynamo_kvrouter_kv_cache_events_applied";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENTS_APPLIED_BY_WORKER_SUFFIX: &str = "kv_cache_events_applied_by_worker_total";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENTS_APPLIED_BY_WORKER_NAME: &str =
    "dynamo_kvrouter_kv_cache_events_applied_by_worker_total";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENTS_DROPPED_SUFFIX: &str = "kv_cache_events_dropped_total";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENTS_DROPPED_NAME: &str = "dynamo_kvrouter_kv_cache_events_dropped_total";
#[cfg(feature = "metrics")]
const KV_CACHE_WORKER_QUARANTINES_SUFFIX: &str = "kv_cache_worker_quarantines_total";
#[cfg(feature = "metrics")]
const KV_CACHE_WORKER_QUARANTINES_NAME: &str = "dynamo_kvrouter_kv_cache_worker_quarantines_total";
#[cfg(feature = "metrics")]
const KV_CACHE_WORKER_QUARANTINED_SUFFIX: &str = "kv_cache_worker_quarantined";
#[cfg(feature = "metrics")]
const KV_CACHE_WORKER_QUARANTINED_NAME: &str = "dynamo_kvrouter_kv_cache_worker_quarantined";

#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
static KV_INDEXER_METRICS: OnceLock<Arc<KvIndexerMetrics>> = OnceLock::new();

impl KvIndexerMetrics {
    #[cfg(feature = "metrics")]
    fn new(
        kv_cache_events_applied: IntCounterVec,
        kv_cache_events_applied_by_worker: IntCounterVec,
        kv_cache_events_dropped_total: IntCounterVec,
        kv_cache_worker_quarantines_total: IntCounterVec,
        kv_cache_worker_quarantined: GaugeVec,
        quarantine_policy: KvMissQuarantinePolicy,
    ) -> Self {
        Self {
            kv_cache_events_applied,
            kv_cache_events_applied_by_worker,
            kv_cache_events_dropped_total,
            kv_cache_worker_quarantines_total,
            kv_cache_worker_quarantined,
            quarantine_policy,
            worker_states: DashMap::new(),
        }
    }

    #[cfg(feature = "runtime-protocols")]
    pub fn from_component_with_quarantine_policy(
        component: &Component,
        quarantine_policy: KvMissQuarantinePolicy,
    ) -> Arc<Self> {
        #[cfg(feature = "metrics")]
        {
            KV_INDEXER_METRICS
                .get_or_init(|| {
                    let metrics = || -> Result<Arc<Self>, anyhow::Error> {
                        let kv_cache_events_applied = component.metrics().create_intcountervec(
                            KV_CACHE_EVENTS_APPLIED_SUFFIX,
                            "Total number of KV cache events applied to index",
                            &["event_type", "status"],
                            &[],
                        )?;
                        let kv_cache_events_applied_by_worker = component.metrics().create_intcountervec(
                            KV_CACHE_EVENTS_APPLIED_BY_WORKER_SUFFIX,
                            "Total number of KV cache events applied to index, partitioned by worker",
                            &["worker_id", "event_type", "status"],
                            &[],
                        )?;
                        let kv_cache_events_dropped_total = component.metrics().create_intcountervec(
                            KV_CACHE_EVENTS_DROPPED_SUFFIX,
                            "Total number of KV cache events dropped before index application",
                            &["worker_id", "event_type", "reason"],
                            &[],
                        )?;
                        let kv_cache_worker_quarantines_total = component.metrics().create_intcountervec(
                            KV_CACHE_WORKER_QUARANTINES_SUFFIX,
                            "Total number of worker KV overlap quarantines triggered by router index misses",
                            &["worker_id", "reason"],
                            &[],
                        )?;
                        let kv_cache_worker_quarantined = component.metrics().create_gaugevec(
                            KV_CACHE_WORKER_QUARANTINED_SUFFIX,
                            "Whether a worker's KV overlap state is currently quarantined (1=yes, 0=no)",
                            &["worker_id"],
                            &[],
                        )?;
                        Ok(Arc::new(Self::new(
                            kv_cache_events_applied,
                            kv_cache_events_applied_by_worker,
                            kv_cache_events_dropped_total,
                            kv_cache_worker_quarantines_total,
                            kv_cache_worker_quarantined,
                            quarantine_policy,
                        )))
                    };

                    match metrics() {
                        Ok(metrics) => metrics,
                        Err(e) => {
                            tracing::warn!("Failed to create kv indexer metrics from component: {}. Using unregistered metrics as fallback.", e);
                            Arc::new(Self::new_unregistered_with_quarantine_policy(
                                quarantine_policy,
                            ))
                        }
                    }
                })
                .clone()
        }

        #[cfg(not(feature = "metrics"))]
        {
            let _ = component;
            Arc::new(Self::new_unregistered_with_quarantine_policy(
                quarantine_policy,
            ))
        }
    }

    #[cfg(feature = "runtime-protocols")]
    pub fn from_component(component: &Component) -> Arc<Self> {
        Self::from_component_with_quarantine_policy(component, KvMissQuarantinePolicy::disabled())
    }

    #[cfg(feature = "metrics")]
    pub fn new_unregistered_with_quarantine_policy(
        quarantine_policy: KvMissQuarantinePolicy,
    ) -> Self {
        Self {
            kv_cache_events_applied: IntCounterVec::new(
                Opts::new(
                    KV_CACHE_EVENTS_APPLIED_NAME,
                    "Total number of KV cache events applied to index",
                ),
                &["event_type", "status"],
            )
            .unwrap(),
            kv_cache_events_applied_by_worker: IntCounterVec::new(
                Opts::new(
                    KV_CACHE_EVENTS_APPLIED_BY_WORKER_NAME,
                    "Total number of KV cache events applied to index, partitioned by worker",
                ),
                &["worker_id", "event_type", "status"],
            )
            .unwrap(),
            kv_cache_events_dropped_total: IntCounterVec::new(
                Opts::new(
                    KV_CACHE_EVENTS_DROPPED_NAME,
                    "Total number of KV cache events dropped before index application",
                ),
                &["worker_id", "event_type", "reason"],
            )
            .unwrap(),
            kv_cache_worker_quarantines_total: IntCounterVec::new(
                Opts::new(
                    KV_CACHE_WORKER_QUARANTINES_NAME,
                    "Total number of worker KV overlap quarantines triggered by router index misses",
                ),
                &["worker_id", "reason"],
            )
            .unwrap(),
            kv_cache_worker_quarantined: GaugeVec::new(
                Opts::new(
                    KV_CACHE_WORKER_QUARANTINED_NAME,
                    "Whether a worker's KV overlap state is currently quarantined (1=yes, 0=no)",
                ),
                &["worker_id"],
            )
            .unwrap(),
            quarantine_policy,
            worker_states: DashMap::new(),
        }
    }

    #[cfg(feature = "metrics")]
    pub fn new_unregistered() -> Self {
        Self::new_unregistered_with_quarantine_policy(KvMissQuarantinePolicy::disabled())
    }

    #[cfg(not(feature = "metrics"))]
    pub fn new_unregistered_with_quarantine_policy(
        quarantine_policy: KvMissQuarantinePolicy,
    ) -> Self {
        Self {
            quarantine_policy,
            worker_states: DashMap::new(),
        }
    }

    #[cfg(not(feature = "metrics"))]
    pub fn new_unregistered() -> Self {
        Self::default()
    }

    pub fn get_event_type(event_data: &KvCacheEventData) -> &'static str {
        match event_data {
            KvCacheEventData::Stored(_) => METRIC_EVENT_STORED,
            KvCacheEventData::Removed(_) => METRIC_EVENT_REMOVED,
            KvCacheEventData::Cleared => METRIC_EVENT_CLEARED,
        }
    }

    fn get_status_label(result: &Result<(), KvCacheEventError>) -> &'static str {
        match result {
            Ok(_) => METRIC_STATUS_OK,
            Err(KvCacheEventError::ParentBlockNotFound) => METRIC_STATUS_PARENT_NOT_FOUND,
            Err(KvCacheEventError::BlockNotFound) => METRIC_STATUS_BLOCK_NOT_FOUND,
            Err(KvCacheEventError::InvalidBlockSequence) => METRIC_STATUS_INVALID_BLOCK,
        }
    }

    pub fn should_drop_event(&self, worker_id: WorkerId) -> bool {
        if !self.quarantine_policy.is_enabled() {
            return false;
        }
        let Some(state) = self.worker_states.get(&worker_id) else {
            return false;
        };
        let now = Instant::now();
        let mut quarantined_until = state.quarantined_until.lock().expect("mutex poisoned");
        if let Some(until) = *quarantined_until {
            if until > now {
                return true;
            }
            *quarantined_until = None;
            #[cfg(feature = "metrics")]
            self.kv_cache_worker_quarantined
                .with_label_values(&[worker_id.to_string().as_str()])
                .set(0.0);
        }
        false
    }

    pub fn record_dropped_event(
        &self,
        worker_id: WorkerId,
        event_type: &'static str,
        reason: &'static str,
    ) {
        #[cfg(feature = "metrics")]
        {
            let worker_id_label = worker_id.to_string();
            self.kv_cache_events_dropped_total
                .with_label_values(&[worker_id_label.as_str(), event_type, reason])
                .inc();
        }
        #[cfg(not(feature = "metrics"))]
        let _ = (worker_id, event_type, reason);
    }

    pub fn record_event_applied(
        &self,
        worker_id: WorkerId,
        event_type: &'static str,
        result: Result<(), KvCacheEventError>,
    ) -> WorkerKvEventAction {
        let status = Self::get_status_label(&result);
        #[cfg(not(feature = "metrics"))]
        let _ = event_type;
        #[cfg(feature = "metrics")]
        {
            let worker_id_label = worker_id.to_string();
            self.kv_cache_events_applied
                .with_label_values(&[event_type, status])
                .inc_by(1);
            self.kv_cache_events_applied_by_worker
                .with_label_values(&[worker_id_label.as_str(), event_type, status])
                .inc_by(1);
        }

        if result.is_ok() {
            return WorkerKvEventAction::Apply;
        }
        if !self.quarantine_policy.is_enabled() {
            return WorkerKvEventAction::Apply;
        }
        if status != METRIC_STATUS_PARENT_NOT_FOUND && status != METRIC_STATUS_BLOCK_NOT_FOUND {
            return WorkerKvEventAction::Apply;
        }

        let threshold = self
            .quarantine_policy
            .miss_threshold
            .expect("checked above");
        let now = Instant::now();
        let state = self.worker_states.entry(worker_id).or_default();

        {
            let mut quarantined_until = state.quarantined_until.lock().expect("mutex poisoned");
            if let Some(until) = *quarantined_until {
                if until > now {
                    return WorkerKvEventAction::DropQuarantined;
                }
                *quarantined_until = None;
                #[cfg(feature = "metrics")]
                self.kv_cache_worker_quarantined
                    .with_label_values(&[worker_id.to_string().as_str()])
                    .set(0.0);
            }
        }

        let mut misses = state.recent_miss_timestamps.lock().expect("mutex poisoned");
        misses.push_back(now);
        while let Some(ts) = misses.front().copied() {
            if now.duration_since(ts) <= self.quarantine_policy.window {
                break;
            }
            misses.pop_front();
        }

        if misses.len() < threshold as usize {
            return WorkerKvEventAction::Apply;
        }

        misses.clear();
        {
            let mut quarantined_until = state.quarantined_until.lock().expect("mutex poisoned");
            *quarantined_until = Some(now + self.quarantine_policy.cooldown);
        }
        #[cfg(feature = "metrics")]
        {
            let worker_id_label = worker_id.to_string();
            self.kv_cache_worker_quarantined
                .with_label_values(&[worker_id_label.as_str()])
                .set(1.0);
            self.kv_cache_worker_quarantines_total
                .with_label_values(&[worker_id_label.as_str(), METRIC_REASON_KV_MISS_SPIKE])
                .inc();
        }
        WorkerKvEventAction::NewlyQuarantined
    }

    pub fn increment_event_applied(
        &self,
        event_type: &'static str,
        result: Result<(), KvCacheEventError>,
    ) {
        let _ = self.record_event_applied(0, event_type, result);
    }
}
