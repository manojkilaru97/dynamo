// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! System health monitoring and health check management

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};
use tokio::sync::mpsc;

use crate::component;
use crate::config::HealthStatus;
use crate::metrics::{MetricsHierarchy, prometheus_names::distributed_runtime};

/// Health check target containing instance info and payload
#[derive(Clone, Debug)]
pub struct HealthCheckTarget {
    pub instance: component::Instance,
    pub payload: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct RealTrafficHealthConfig {
    pub window: Duration,
    pub min_samples: usize,
    pub failure_threshold: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestOutcome {
    Success,
    Failure,
    Overloaded,
}

impl RequestOutcome {
    fn is_failure(self) -> bool {
        matches!(self, Self::Failure)
    }

    fn is_eligible_for_failure_ratio(self) -> bool {
        !matches!(self, Self::Overloaded)
    }

    fn label(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::Overloaded => "overloaded",
        }
    }
}

#[derive(Clone, Debug)]
struct RealTrafficWindow {
    events: VecDeque<(Instant, RequestOutcome)>,
    status: HealthStatus,
}

impl Default for RealTrafficWindow {
    fn default() -> Self {
        Self {
            events: VecDeque::new(),
            status: HealthStatus::Ready,
        }
    }
}

impl RealTrafficWindow {
    fn record(&mut self, now: Instant, outcome: RequestOutcome, config: &RealTrafficHealthConfig) {
        self.prune(now, config.window);
        self.events.push_back((now, outcome));
        self.status = self.compute_status(config);
    }

    fn prune(&mut self, now: Instant, window: Duration) {
        while let Some((ts, _)) = self.events.front() {
            if now.saturating_duration_since(*ts) <= window {
                break;
            }
            self.events.pop_front();
        }
    }

    fn sample_counts(&self) -> (usize, usize) {
        let mut total = 0usize;
        let mut failures = 0usize;

        for (_, outcome) in &self.events {
            if !outcome.is_eligible_for_failure_ratio() {
                continue;
            }
            total += 1;
            if outcome.is_failure() {
                failures += 1;
            }
        }

        (total, failures)
    }

    fn outcome_counts(&self) -> (usize, usize, usize) {
        let mut successes = 0usize;
        let mut failures = 0usize;
        let mut overloaded = 0usize;

        for (_, outcome) in &self.events {
            match outcome {
                RequestOutcome::Success => successes += 1,
                RequestOutcome::Failure => failures += 1,
                RequestOutcome::Overloaded => overloaded += 1,
            }
        }

        (successes, failures, overloaded)
    }

    fn compute_status(&self, config: &RealTrafficHealthConfig) -> HealthStatus {
        let (total, failures) = self.sample_counts();
        if total < config.min_samples {
            return HealthStatus::Ready;
        }

        let failure_ratio = failures as f64 / total as f64;
        if failure_ratio >= config.failure_threshold {
            HealthStatus::NotReady
        } else {
            HealthStatus::Ready
        }
    }

    fn status_at(&mut self, now: Instant, config: &RealTrafficHealthConfig) -> HealthStatus {
        self.prune(now, config.window);
        self.status = self.compute_status(config);
        self.status.clone()
    }

    fn has_success_within(&mut self, now: Instant, within: Duration) -> bool {
        self.prune(now, within);
        self.events.iter().rev().any(|(ts, outcome)| {
            matches!(outcome, RequestOutcome::Success)
                && now.saturating_duration_since(*ts) <= within
        })
    }

    fn reason_at(&mut self, now: Instant, config: &RealTrafficHealthConfig) -> Option<String> {
        self.prune(now, config.window);
        let (total, failures) = self.sample_counts();
        if total < config.min_samples {
            return None;
        }

        let failure_ratio = failures as f64 / total as f64;
        if failure_ratio >= config.failure_threshold {
            Some(format!(
                "real traffic failure ratio {:.2} >= {:.2} over last {}s (failures={}, total={})",
                failure_ratio,
                config.failure_threshold,
                config.window.as_secs(),
                failures,
                total
            ))
        } else {
            None
        }
    }
}

/// Current Health Status
/// If use_endpoint_health_status is set then
/// initialize the endpoint_health hashmap to the
/// starting health status
#[derive(Clone)]
pub struct SystemHealth {
    system_health: HealthStatus,
    endpoint_health: Arc<std::sync::RwLock<HashMap<String, HealthStatus>>>,
    endpoint_real_traffic_health: Arc<std::sync::RwLock<HashMap<String, RealTrafficWindow>>>,
    real_traffic_health_config: RealTrafficHealthConfig,
    /// Maps endpoint subject to health check target (instance + payload)
    health_check_targets: Arc<std::sync::RwLock<HashMap<String, HealthCheckTarget>>>,
    /// Maps endpoint subject to its specific health check notifier
    health_check_notifiers: Arc<std::sync::RwLock<HashMap<String, Arc<tokio::sync::Notify>>>>,
    /// Channel for new endpoint registrations
    /// This solves the race condition where HealthCheckManager starts before endpoints are registered
    /// Using a channel ensures no registrations are lost.
    new_endpoint_tx: mpsc::UnboundedSender<String>,
    new_endpoint_rx: Arc<parking_lot::Mutex<Option<mpsc::UnboundedReceiver<String>>>>,
    use_endpoint_health_status: Vec<String>,
    health_path: String,
    live_path: String,
    endpoint_last_health_check_success: Arc<std::sync::RwLock<HashMap<String, Instant>>>,
    start_time: Instant,
    uptime_gauge: OnceLock<prometheus::Gauge>,
    overall_ready_gauge: OnceLock<prometheus::Gauge>,
    endpoint_health_status_gauge: OnceLock<prometheus::GaugeVec>,
    endpoint_real_traffic_failure_ratio_gauge: OnceLock<prometheus::GaugeVec>,
    endpoint_real_traffic_samples_gauge: OnceLock<prometheus::GaugeVec>,
    endpoint_real_traffic_outcome_samples_gauge: OnceLock<prometheus::GaugeVec>,
    health_check_last_success_age_seconds_gauge: OnceLock<prometheus::GaugeVec>,
    health_check_requests_started_total: OnceLock<prometheus::IntCounterVec>,
    health_check_requests_completed_total: OnceLock<prometheus::IntCounterVec>,
    health_check_last_duration_seconds_gauge: OnceLock<prometheus::GaugeVec>,
    system_status_requests_total: OnceLock<prometheus::IntCounterVec>,
}

impl SystemHealth {
    pub fn new(
        starting_health_status: HealthStatus,
        use_endpoint_health_status: Vec<String>,
        health_path: String,
        live_path: String,
        real_traffic_health_config: RealTrafficHealthConfig,
    ) -> Self {
        let mut endpoint_health = HashMap::new();
        let mut endpoint_real_traffic_health = HashMap::new();
        for endpoint in &use_endpoint_health_status {
            endpoint_health.insert(endpoint.clone(), starting_health_status.clone());
            endpoint_real_traffic_health.insert(endpoint.clone(), RealTrafficWindow::default());
        }

        let (tx, rx) = mpsc::unbounded_channel();

        SystemHealth {
            system_health: starting_health_status,
            endpoint_health: Arc::new(std::sync::RwLock::new(endpoint_health)),
            endpoint_real_traffic_health: Arc::new(std::sync::RwLock::new(
                endpoint_real_traffic_health,
            )),
            real_traffic_health_config,
            health_check_targets: Arc::new(std::sync::RwLock::new(HashMap::new())),
            health_check_notifiers: Arc::new(std::sync::RwLock::new(HashMap::new())),
            new_endpoint_tx: tx,
            new_endpoint_rx: Arc::new(parking_lot::Mutex::new(Some(rx))),
            use_endpoint_health_status,
            health_path,
            live_path,
            endpoint_last_health_check_success: Arc::new(std::sync::RwLock::new(HashMap::new())),
            start_time: Instant::now(),
            uptime_gauge: OnceLock::new(),
            overall_ready_gauge: OnceLock::new(),
            endpoint_health_status_gauge: OnceLock::new(),
            endpoint_real_traffic_failure_ratio_gauge: OnceLock::new(),
            endpoint_real_traffic_samples_gauge: OnceLock::new(),
            endpoint_real_traffic_outcome_samples_gauge: OnceLock::new(),
            health_check_last_success_age_seconds_gauge: OnceLock::new(),
            health_check_requests_started_total: OnceLock::new(),
            health_check_requests_completed_total: OnceLock::new(),
            health_check_last_duration_seconds_gauge: OnceLock::new(),
            system_status_requests_total: OnceLock::new(),
        }
    }

    pub fn set_health_status(&mut self, status: HealthStatus) {
        self.system_health = status;
    }

    pub fn set_endpoint_health_status(&self, endpoint: &str, status: HealthStatus) {
        let mut endpoint_health = self.endpoint_health.write().unwrap();
        endpoint_health.insert(endpoint.to_string(), status);
    }

    pub fn record_health_check_success(&self, endpoint: &str) {
        self.endpoint_last_health_check_success
            .write()
            .unwrap()
            .insert(endpoint.to_string(), Instant::now());
    }

    pub fn record_health_check_request_started(&self, endpoint: &str, trigger: &str) {
        if let Some(counter) = self.health_check_requests_started_total.get() {
            counter.with_label_values(&[endpoint, trigger]).inc();
        }
    }

    pub fn record_health_check_request_completed(
        &self,
        endpoint: &str,
        trigger: &str,
        result: &str,
        elapsed: Duration,
    ) {
        if let Some(counter) = self.health_check_requests_completed_total.get() {
            counter
                .with_label_values(&[endpoint, trigger, result])
                .inc();
        }
        if let Some(gauge) = self.health_check_last_duration_seconds_gauge.get() {
            gauge
                .with_label_values(&[endpoint, trigger, result])
                .set(elapsed.as_secs_f64());
        }
    }

    pub fn record_system_status_request(&self, route: &str, status_code: u16) {
        if let Some(counter) = self.system_status_requests_total.get() {
            let status = status_code.to_string();
            counter.with_label_values(&[route, status.as_str()]).inc();
        }
    }

    pub fn record_endpoint_request_result(&self, endpoint: &str, success: bool) {
        let outcome = if success {
            RequestOutcome::Success
        } else {
            RequestOutcome::Failure
        };
        self.record_endpoint_request_outcome(endpoint, outcome);
    }

    pub fn record_endpoint_request_overload(&self, endpoint: &str) {
        self.record_endpoint_request_outcome(endpoint, RequestOutcome::Overloaded);
    }

    fn record_endpoint_request_outcome(&self, endpoint: &str, outcome: RequestOutcome) {
        let now = Instant::now();
        let mut windows = self.endpoint_real_traffic_health.write().unwrap();
        let window = windows.entry(endpoint.to_string()).or_default();
        window.record(now, outcome, &self.real_traffic_health_config);
    }

    pub fn get_endpoint_real_traffic_health_status(&self, endpoint: &str) -> Option<HealthStatus> {
        let now = Instant::now();
        let mut endpoint_real_traffic_health = self.endpoint_real_traffic_health.write().unwrap();
        endpoint_real_traffic_health
            .get_mut(endpoint)
            .map(|window| window.status_at(now, &self.real_traffic_health_config))
    }

    pub fn real_traffic_window(&self) -> Duration {
        self.real_traffic_health_config.window
    }

    pub fn has_recent_endpoint_success(&self, endpoint: &str, within: Duration) -> bool {
        let now = Instant::now();
        let mut endpoint_real_traffic_health = self.endpoint_real_traffic_health.write().unwrap();
        endpoint_real_traffic_health
            .get_mut(endpoint)
            .is_some_and(|window| window.has_success_within(now, within))
    }

    /// Returns the overall health status and endpoint health statuses
    /// System health is determined by ALL endpoints that have registered health checks
    pub fn get_health_status(&self) -> (bool, HashMap<String, String>) {
        let (healthy, endpoints, _) = self.get_health_status_with_reasons();
        (healthy, endpoints)
    }

    pub fn get_health_status_with_reasons(
        &self,
    ) -> (bool, HashMap<String, String>, HashMap<String, String>) {
        let now = Instant::now();
        let health_check_targets = self.health_check_targets.read().unwrap();
        let endpoint_health = self.endpoint_health.read().unwrap();
        let mut endpoint_real_traffic_health = self.endpoint_real_traffic_health.write().unwrap();
        let mut endpoints: HashMap<String, String> = HashMap::new();
        let mut reasons: HashMap<String, String> = HashMap::new();

        for endpoint in endpoint_health.keys() {
            let base_ready = endpoint_health
                .get(endpoint)
                .is_some_and(|status| *status == HealthStatus::Ready);
            let real_reason = endpoint_real_traffic_health
                .get_mut(endpoint)
                .and_then(|window| window.reason_at(now, &self.real_traffic_health_config));
            let real_ready = real_reason.is_none();
            let status = if base_ready && real_ready {
                "ready"
            } else {
                if !base_ready {
                    let reason = if health_check_targets.contains_key(endpoint) {
                        "active health check marked endpoint not ready".to_string()
                    } else {
                        "endpoint registration is not ready".to_string()
                    };
                    reasons.insert(endpoint.clone(), reason);
                } else if let Some(reason) = real_reason {
                    reasons.insert(endpoint.clone(), reason);
                }
                "notready"
            };
            endpoints.insert(endpoint.clone(), status.to_string());
        }

        let endpoint_is_ready = |endpoint: &str,
                                 endpoint_health: &HashMap<String, HealthStatus>,
                                 endpoint_real_traffic_health: &mut HashMap<
            String,
            RealTrafficWindow,
        >| {
            let base_ready = endpoint_health
                .get(endpoint)
                .is_some_and(|status| *status == HealthStatus::Ready);
            let real_ready = endpoint_real_traffic_health
                .get_mut(endpoint)
                .is_none_or(|window| {
                    window.status_at(now, &self.real_traffic_health_config) == HealthStatus::Ready
                });
            base_ready && real_ready
        };

        let healthy = if !self.use_endpoint_health_status.is_empty() {
            self.use_endpoint_health_status.iter().all(|endpoint| {
                endpoint_is_ready(
                    endpoint,
                    &endpoint_health,
                    &mut endpoint_real_traffic_health,
                )
            })
        } else if !health_check_targets.is_empty() {
            health_check_targets.keys().all(|endpoint_subject| {
                endpoint_is_ready(
                    endpoint_subject,
                    &endpoint_health,
                    &mut endpoint_real_traffic_health,
                )
            })
        } else {
            self.system_health == HealthStatus::Ready
        };

        (healthy, endpoints, reasons)
    }

    /// Register a health check target for an endpoint
    pub fn register_health_check_target(
        &self,
        endpoint_subject: &str,
        instance: component::Instance,
        payload: serde_json::Value,
    ) {
        let key = endpoint_subject.to_owned();

        let inserted = {
            let mut targets = self.health_check_targets.write().unwrap();
            match targets.entry(key.clone()) {
                std::collections::hash_map::Entry::Occupied(_) => false,
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(HealthCheckTarget { instance, payload });
                    true
                }
            }
        };

        if !inserted {
            tracing::warn!(
                "Attempted to re-register health check for endpoint '{}'; ignoring.",
                key
            );
            return;
        }

        {
            let mut notifiers = self.health_check_notifiers.write().unwrap();
            notifiers
                .entry(key.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Notify::new()));
        }

        {
            let mut endpoint_health = self.endpoint_health.write().unwrap();
            endpoint_health
                .entry(key.clone())
                .or_insert(HealthStatus::NotReady);
        }

        {
            let mut endpoint_real_traffic_health =
                self.endpoint_real_traffic_health.write().unwrap();
            endpoint_real_traffic_health
                .entry(key.clone())
                .or_insert_with(RealTrafficWindow::default);
        }

        if let Err(e) = self.new_endpoint_tx.send(key.clone()) {
            tracing::error!(
                "Failed to send endpoint '{}' registration to health check manager: {}. \
                 Health checks will not be performed for this endpoint.",
                key,
                e
            );
        }
    }

    pub fn get_health_check_targets(&self) -> Vec<(String, HealthCheckTarget)> {
        let targets = self.health_check_targets.read().unwrap();
        targets
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    pub fn has_health_check_targets(&self) -> bool {
        let targets = self.health_check_targets.read().unwrap();
        !targets.is_empty()
    }

    pub fn get_health_check_endpoints(&self) -> Vec<String> {
        let targets = self.health_check_targets.read().unwrap();
        targets.keys().cloned().collect()
    }

    pub fn get_health_check_target(&self, endpoint: &str) -> Option<HealthCheckTarget> {
        let targets = self.health_check_targets.read().unwrap();
        targets.get(endpoint).cloned()
    }

    pub fn get_endpoint_health_status(&self, endpoint: &str) -> Option<HealthStatus> {
        let endpoint_health = self.endpoint_health.read().unwrap();
        endpoint_health.get(endpoint).cloned()
    }

    pub fn get_endpoint_health_check_notifier(
        &self,
        endpoint_subject: &str,
    ) -> Option<Arc<tokio::sync::Notify>> {
        let notifiers = self.health_check_notifiers.read().unwrap();
        notifiers.get(endpoint_subject).cloned()
    }

    pub fn take_new_endpoint_receiver(&self) -> Option<mpsc::UnboundedReceiver<String>> {
        self.new_endpoint_rx.lock().take()
    }

    pub fn initialize_uptime_gauge<T: MetricsHierarchy>(&self, registry: &T) -> anyhow::Result<()> {
        let gauge = registry.metrics().create_gauge(
            distributed_runtime::UPTIME_SECONDS,
            "Total uptime of the DistributedRuntime in seconds",
            &[],
        )?;
        self.uptime_gauge
            .set(gauge)
            .map_err(|_| anyhow::anyhow!("uptime_gauge already initialized"))?;
        Ok(())
    }

    pub fn initialize_health_observability_metrics<T: MetricsHierarchy>(
        &self,
        registry: &T,
    ) -> anyhow::Result<()> {
        let overall_ready_gauge = registry.metrics().create_gauge(
            distributed_runtime::OVERALL_READY,
            "Current overall readiness state of the DistributedRuntime (1=ready, 0=not ready)",
            &[],
        )?;
        self.overall_ready_gauge
            .set(overall_ready_gauge)
            .map_err(|_| anyhow::anyhow!("overall_ready_gauge already initialized"))?;

        let endpoint_health_status_gauge = registry.metrics().create_gaugevec(
            distributed_runtime::ENDPOINT_HEALTH_STATUS,
            "Current per-endpoint health state by status kind (1=ready, 0=not ready)",
            &["endpoint", "status_kind"],
            &[],
        )?;
        self.endpoint_health_status_gauge
            .set(endpoint_health_status_gauge)
            .map_err(|_| anyhow::anyhow!("endpoint_health_status_gauge already initialized"))?;

        let endpoint_real_traffic_failure_ratio_gauge = registry.metrics().create_gaugevec(
            distributed_runtime::ENDPOINT_REAL_TRAFFIC_FAILURE_RATIO,
            "Current rolling real-traffic failure ratio per endpoint",
            &["endpoint"],
            &[],
        )?;
        self.endpoint_real_traffic_failure_ratio_gauge
            .set(endpoint_real_traffic_failure_ratio_gauge)
            .map_err(|_| {
                anyhow::anyhow!("endpoint_real_traffic_failure_ratio_gauge already initialized")
            })?;

        let endpoint_real_traffic_samples_gauge = registry.metrics().create_gaugevec(
            distributed_runtime::ENDPOINT_REAL_TRAFFIC_SAMPLES,
            "Current rolling real-traffic sample count per endpoint used by the health failure ratio",
            &["endpoint"],
            &[],
        )?;
        self.endpoint_real_traffic_samples_gauge
            .set(endpoint_real_traffic_samples_gauge)
            .map_err(|_| {
                anyhow::anyhow!("endpoint_real_traffic_samples_gauge already initialized")
            })?;

        let endpoint_real_traffic_outcome_samples_gauge = registry.metrics().create_gaugevec(
            distributed_runtime::ENDPOINT_REAL_TRAFFIC_OUTCOME_SAMPLES,
            "Current rolling real-traffic sample count per endpoint broken out by outcome",
            &["endpoint", "outcome"],
            &[],
        )?;
        self.endpoint_real_traffic_outcome_samples_gauge
            .set(endpoint_real_traffic_outcome_samples_gauge)
            .map_err(|_| {
                anyhow::anyhow!("endpoint_real_traffic_outcome_samples_gauge already initialized")
            })?;

        let health_check_last_success_age_seconds_gauge = registry.metrics().create_gaugevec(
            distributed_runtime::HEALTH_CHECK_LAST_SUCCESS_AGE_SECONDS,
            "Seconds since the last successful active health check per endpoint (-1 if never)",
            &["endpoint"],
            &[],
        )?;
        self.health_check_last_success_age_seconds_gauge
            .set(health_check_last_success_age_seconds_gauge)
            .map_err(|_| {
                anyhow::anyhow!("health_check_last_success_age_seconds_gauge already initialized")
            })?;

        let health_check_requests_started_total = registry.metrics().create_intcountervec(
            distributed_runtime::HEALTH_CHECK_REQUESTS_STARTED_TOTAL,
            "Total active health check requests started",
            &["endpoint", "trigger"],
            &[],
        )?;
        self.health_check_requests_started_total
            .set(health_check_requests_started_total)
            .map_err(|_| {
                anyhow::anyhow!("health_check_requests_started_total already initialized")
            })?;

        let health_check_requests_completed_total = registry.metrics().create_intcountervec(
            distributed_runtime::HEALTH_CHECK_REQUESTS_COMPLETED_TOTAL,
            "Total active health check requests completed by result",
            &["endpoint", "trigger", "result"],
            &[],
        )?;
        self.health_check_requests_completed_total
            .set(health_check_requests_completed_total)
            .map_err(|_| {
                anyhow::anyhow!("health_check_requests_completed_total already initialized")
            })?;

        let health_check_last_duration_seconds_gauge = registry.metrics().create_gaugevec(
            distributed_runtime::HEALTH_CHECK_LAST_DURATION_SECONDS,
            "Last observed active health check duration in seconds",
            &["endpoint", "trigger", "result"],
            &[],
        )?;
        self.health_check_last_duration_seconds_gauge
            .set(health_check_last_duration_seconds_gauge)
            .map_err(|_| {
                anyhow::anyhow!("health_check_last_duration_seconds_gauge already initialized")
            })?;

        let system_status_requests_total = registry.metrics().create_intcountervec(
            distributed_runtime::SYSTEM_STATUS_REQUESTS_TOTAL,
            "Total system status endpoint requests by route and HTTP status",
            &["route", "status"],
            &[],
        )?;
        self.system_status_requests_total
            .set(system_status_requests_total)
            .map_err(|_| anyhow::anyhow!("system_status_requests_total already initialized"))?;

        Ok(())
    }

    pub fn update_health_observability_gauges(&self) {
        let now = Instant::now();
        let (healthy, _, _) = self.get_health_status_with_reasons();

        if let Some(gauge) = self.overall_ready_gauge.get() {
            gauge.set(if healthy { 1.0 } else { 0.0 });
        }

        let endpoint_health = self.endpoint_health.read().unwrap();
        let mut endpoint_real_traffic_health = self.endpoint_real_traffic_health.write().unwrap();
        let endpoint_last_health_check_success =
            self.endpoint_last_health_check_success.read().unwrap();

        for endpoint in endpoint_health.keys() {
            let base_ready = endpoint_health
                .get(endpoint)
                .is_some_and(|status| *status == HealthStatus::Ready);

            let (
                real_ready,
                failure_ratio,
                total_samples,
                success_samples,
                failure_samples,
                overloaded_samples,
            ) = endpoint_real_traffic_health
                .get_mut(endpoint)
                .map(|window| {
                    let status = window.status_at(now, &self.real_traffic_health_config);
                    let (total, failures) = window.sample_counts();
                    let (success_samples, failure_samples, overloaded_samples) =
                        window.outcome_counts();
                    let failure_ratio = if total == 0 {
                        0.0
                    } else {
                        failures as f64 / total as f64
                    };
                    (
                        status == HealthStatus::Ready,
                        failure_ratio,
                        total as f64,
                        success_samples as f64,
                        failure_samples as f64,
                        overloaded_samples as f64,
                    )
                })
                .unwrap_or((true, 0.0, 0.0, 0.0, 0.0, 0.0));

            let effective_ready = base_ready && real_ready;

            if let Some(gauge) = self.endpoint_health_status_gauge.get() {
                gauge
                    .with_label_values(&[endpoint.as_str(), "base"])
                    .set(if base_ready { 1.0 } else { 0.0 });
                gauge
                    .with_label_values(&[endpoint.as_str(), "real_traffic"])
                    .set(if real_ready { 1.0 } else { 0.0 });
                gauge
                    .with_label_values(&[endpoint.as_str(), "effective"])
                    .set(if effective_ready { 1.0 } else { 0.0 });
            }

            if let Some(gauge) = self.endpoint_real_traffic_failure_ratio_gauge.get() {
                gauge.with_label_values(&[endpoint]).set(failure_ratio);
            }

            if let Some(gauge) = self.endpoint_real_traffic_samples_gauge.get() {
                gauge.with_label_values(&[endpoint]).set(total_samples);
            }

            if let Some(gauge) = self.endpoint_real_traffic_outcome_samples_gauge.get() {
                let success_labels = [endpoint.as_str(), RequestOutcome::Success.label()];
                let failure_labels = [endpoint.as_str(), RequestOutcome::Failure.label()];
                let overloaded_labels = [endpoint.as_str(), RequestOutcome::Overloaded.label()];
                gauge
                    .with_label_values(&success_labels)
                    .set(success_samples);
                gauge
                    .with_label_values(&failure_labels)
                    .set(failure_samples);
                gauge
                    .with_label_values(&overloaded_labels)
                    .set(overloaded_samples);
            }

            if let Some(gauge) = self.health_check_last_success_age_seconds_gauge.get() {
                let age_seconds = endpoint_last_health_check_success
                    .get(endpoint)
                    .map(|last| now.saturating_duration_since(*last).as_secs_f64())
                    .unwrap_or(-1.0);
                gauge.with_label_values(&[endpoint]).set(age_seconds);
            }
        }
    }
    pub fn uptime(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    pub fn update_uptime_gauge(&self) {
        if let Some(gauge) = self.uptime_gauge.get() {
            gauge.set(self.uptime().as_secs_f64());
        }
    }

    pub fn health_path(&self) -> &str {
        &self.health_path
    }

    pub fn live_path(&self) -> &str {
        &self.live_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_system_health(config: RealTrafficHealthConfig) -> SystemHealth {
        SystemHealth::new(
            HealthStatus::Ready,
            vec!["generate".to_string()],
            "/health".to_string(),
            "/live".to_string(),
            config,
        )
    }

    #[test]
    fn test_recent_endpoint_success_tracks_successes() {
        let system_health = make_system_health(RealTrafficHealthConfig {
            window: Duration::from_secs(600),
            min_samples: 5,
            failure_threshold: 0.8,
        });

        assert!(!system_health.has_recent_endpoint_success("generate", Duration::from_secs(600)));
        system_health.record_endpoint_request_result("generate", false);
        assert!(!system_health.has_recent_endpoint_success("generate", Duration::from_secs(600)));
        system_health.record_endpoint_request_result("generate", true);
        assert!(system_health.has_recent_endpoint_success("generate", Duration::from_secs(600)));
    }

    #[test]
    fn test_real_traffic_health_requires_min_samples() {
        let system_health = make_system_health(RealTrafficHealthConfig {
            window: Duration::from_secs(600),
            min_samples: 3,
            failure_threshold: 0.8,
        });

        system_health.record_endpoint_request_result("generate", false);
        system_health.record_endpoint_request_result("generate", false);

        assert_eq!(
            system_health.get_endpoint_real_traffic_health_status("generate"),
            Some(HealthStatus::Ready)
        );
    }

    #[test]
    fn test_real_traffic_health_fences_after_threshold() {
        let system_health = make_system_health(RealTrafficHealthConfig {
            window: Duration::from_secs(600),
            min_samples: 5,
            failure_threshold: 0.8,
        });

        for _ in 0..4 {
            system_health.record_endpoint_request_result("generate", false);
        }
        system_health.record_endpoint_request_result("generate", true);

        assert_eq!(
            system_health.get_endpoint_real_traffic_health_status("generate"),
            Some(HealthStatus::NotReady)
        );
        assert!(!system_health.get_health_status().0);
    }

    #[test]
    fn test_real_traffic_health_recovers_with_successes() {
        let system_health = make_system_health(RealTrafficHealthConfig {
            window: Duration::from_secs(600),
            min_samples: 5,
            failure_threshold: 0.8,
        });

        for _ in 0..4 {
            system_health.record_endpoint_request_result("generate", false);
        }
        system_health.record_endpoint_request_result("generate", true);
        assert_eq!(
            system_health.get_endpoint_real_traffic_health_status("generate"),
            Some(HealthStatus::NotReady)
        );

        for _ in 0..5 {
            system_health.record_endpoint_request_result("generate", true);
        }

        assert_eq!(
            system_health.get_endpoint_real_traffic_health_status("generate"),
            Some(HealthStatus::Ready)
        );
        assert!(system_health.get_health_status().0);
    }

    #[test]
    fn test_real_traffic_health_prunes_old_failures() {
        let config = RealTrafficHealthConfig {
            window: Duration::from_secs(600),
            min_samples: 1,
            failure_threshold: 0.8,
        };
        let mut window = RealTrafficWindow::default();
        let now = Instant::now();

        window.record(
            now - Duration::from_secs(601),
            RequestOutcome::Failure,
            &config,
        );

        assert_eq!(window.status_at(now, &config), HealthStatus::Ready);
    }

    #[test]
    fn test_real_traffic_health_reason_is_reported() {
        let system_health = make_system_health(RealTrafficHealthConfig {
            window: Duration::from_secs(600),
            min_samples: 5,
            failure_threshold: 0.8,
        });

        for _ in 0..4 {
            system_health.record_endpoint_request_result("generate", false);
        }
        system_health.record_endpoint_request_result("generate", true);

        let (healthy, _endpoints, reasons) = system_health.get_health_status_with_reasons();
        assert!(!healthy);
        assert!(
            reasons
                .get("generate")
                .expect("missing generate reason")
                .contains("real traffic failure ratio")
        );
    }

    #[test]
    fn test_real_traffic_overload_does_not_count_as_failure() {
        let system_health = make_system_health(RealTrafficHealthConfig {
            window: Duration::from_secs(600),
            min_samples: 3,
            failure_threshold: 0.8,
        });

        system_health.record_endpoint_request_overload("generate");
        system_health.record_endpoint_request_result("generate", true);
        system_health.record_endpoint_request_result("generate", true);
        system_health.record_endpoint_request_result("generate", true);

        assert_eq!(
            system_health.get_endpoint_real_traffic_health_status("generate"),
            Some(HealthStatus::Ready)
        );
    }
}
