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
    health_check_enabled: bool,
    health_path: String,
    live_path: String,
    start_time: Instant,
    uptime_gauge: OnceLock<prometheus::Gauge>,
}

impl SystemHealth {
    pub fn new(
        starting_health_status: HealthStatus,
        use_endpoint_health_status: Vec<String>,
        health_check_enabled: bool,
        health_path: String,
        live_path: String,
        real_traffic_health_config: RealTrafficHealthConfig,
    ) -> Self {
        // Force NotReady when canary is enabled — canary verifies before marking Ready.
        let initial_endpoint_status = if health_check_enabled {
            HealthStatus::NotReady
        } else {
            starting_health_status.clone()
        };
        let mut endpoint_health = HashMap::new();
        let mut endpoint_real_traffic_health = HashMap::new();
        for endpoint in &use_endpoint_health_status {
            endpoint_health.insert(endpoint.clone(), initial_endpoint_status.clone());
            endpoint_real_traffic_health.insert(endpoint.clone(), RealTrafficWindow::default());
        }

        // Create the channel for endpoint registration notifications
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
            health_check_enabled,
            health_path,
            live_path,
            start_time: Instant::now(),
            uptime_gauge: OnceLock::new(),
        }
    }

    pub fn health_check_enabled(&self) -> bool {
        self.health_check_enabled
    }

    /// Signal endpoint transport registration. Sets Ready when canary is disabled;
    /// no-op when canary is enabled (canary will set Ready after verification).
    pub fn set_endpoint_registered(&self, endpoint: &str) {
        if !self.health_check_enabled {
            self.set_endpoint_health_status(endpoint, HealthStatus::Ready);
        }
    }

    pub fn set_health_status(&mut self, status: HealthStatus) {
        self.system_health = status;
    }

    pub fn set_endpoint_health_status(&self, endpoint: &str, status: HealthStatus) {
        let mut endpoint_health = self.endpoint_health.write().unwrap();
        endpoint_health.insert(endpoint.to_string(), status);
    }

    /// Returns the overall health status and endpoint health statuses
    /// System health is determined by ALL endpoints that have registered health checks
    pub fn get_health_status(&self) -> (bool, HashMap<String, String>) {
        let health_check_targets = self.health_check_targets.read().unwrap();
        let endpoint_health = self.endpoint_health.read().unwrap();
        let mut endpoints: HashMap<String, String> = HashMap::new();

        for (endpoint, status) in endpoint_health.iter() {
            endpoints.insert(
                endpoint.clone(),
                if *status == HealthStatus::Ready {
                    "ready".to_string()
                } else {
                    "notready".to_string()
                },
            );
        }

        let healthy = if !self.use_endpoint_health_status.is_empty() {
            self.use_endpoint_health_status.iter().all(|endpoint| {
                endpoint_health
                    .get(endpoint)
                    .is_some_and(|status| *status == HealthStatus::Ready)
            })
        } else {
            // If we have registered health check targets, use them to determine health
            if !health_check_targets.is_empty() {
                health_check_targets
                    .iter()
                    .all(|(endpoint_subject, _target)| {
                        endpoint_health
                            .get(endpoint_subject)
                            .is_some_and(|status| *status == HealthStatus::Ready)
                    })
            } else {
                // No health check targets registered, use simple system health
                self.system_health == HealthStatus::Ready
            }
        };

        (healthy, endpoints)
    }

    /// Register a health check target for an endpoint
    pub fn register_health_check_target(
        &self,
        endpoint_subject: &str,
        instance: component::Instance,
        payload: serde_json::Value,
    ) {
        let key = endpoint_subject.to_owned();

        // Atomically check+insert under a single write lock to avoid races.
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

        // Create and store a unique notifier for this endpoint (idempotent).
        {
            let mut notifiers = self.health_check_notifiers.write().unwrap();
            notifiers
                .entry(key.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Notify::new()));
        }

        // Initialize endpoint health status conservatively to NotReady.
        {
            let mut endpoint_health = self.endpoint_health.write().unwrap();
            endpoint_health
                .entry(key.clone())
                .or_insert(HealthStatus::NotReady);
        }
        {
            let mut real_traffic = self.endpoint_real_traffic_health.write().unwrap();
            real_traffic
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

    /// Get all health check targets
    pub fn get_health_check_targets(&self) -> Vec<(String, HealthCheckTarget)> {
        let targets = self.health_check_targets.read().unwrap();
        targets
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Check if any health check targets are registered
    pub fn has_health_check_targets(&self) -> bool {
        let targets = self.health_check_targets.read().unwrap();
        !targets.is_empty()
    }

    /// Get list of endpoints with health check targets
    pub fn get_health_check_endpoints(&self) -> Vec<String> {
        let targets = self.health_check_targets.read().unwrap();
        targets.keys().cloned().collect()
    }

    /// Get health check target for a specific endpoint
    pub fn get_health_check_target(&self, endpoint: &str) -> Option<HealthCheckTarget> {
        let targets = self.health_check_targets.read().unwrap();
        targets.get(endpoint).cloned()
    }

    /// Get the endpoint health status (Ready/NotReady)
    pub fn get_endpoint_health_status(&self, endpoint: &str) -> Option<HealthStatus> {
        let endpoint_health = self.endpoint_health.read().unwrap();
        endpoint_health.get(endpoint).cloned()
    }

    pub fn real_traffic_window(&self) -> Duration {
        self.real_traffic_health_config.window
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
        windows
            .entry(endpoint.to_string())
            .or_insert_with(RealTrafficWindow::default)
            .record(now, outcome, &self.real_traffic_health_config);
    }

    pub fn get_endpoint_real_traffic_health_status(&self, endpoint: &str) -> Option<HealthStatus> {
        let now = Instant::now();
        let mut windows = self.endpoint_real_traffic_health.write().unwrap();
        windows
            .get_mut(endpoint)
            .map(|window| window.status_at(now, &self.real_traffic_health_config))
    }

    pub fn has_recent_endpoint_success(&self, endpoint: &str, within: Duration) -> bool {
        let now = Instant::now();
        let mut windows = self.endpoint_real_traffic_health.write().unwrap();
        windows
            .get_mut(endpoint)
            .map(|window| window.has_success_within(now, within))
            .unwrap_or(false)
    }

    /// Get the endpoint-specific health check notifier
    pub fn get_endpoint_health_check_notifier(
        &self,
        endpoint_subject: &str,
    ) -> Option<Arc<tokio::sync::Notify>> {
        let notifiers = self.health_check_notifiers.read().unwrap();
        notifiers.get(endpoint_subject).cloned()
    }

    /// Take the receiver for new endpoint registrations (can only be called once)
    /// This is used by HealthCheckManager to receive notifications of new endpoints
    pub fn take_new_endpoint_receiver(&self) -> Option<mpsc::UnboundedReceiver<String>> {
        self.new_endpoint_rx.lock().take()
    }

    /// Initialize the uptime gauge using the provided metrics registry
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

    /// Get the current uptime as a Duration
    pub fn uptime(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Update the uptime gauge with the current uptime value
    pub fn update_uptime_gauge(&self) {
        if let Some(gauge) = self.uptime_gauge.get() {
            gauge.set(self.uptime().as_secs_f64());
        }
    }

    /// Get the health check path
    pub fn health_path(&self) -> &str {
        &self.health_path
    }

    /// Get the liveness check path
    pub fn live_path(&self) -> &str {
        &self.live_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_system_health(min_samples: usize, failure_threshold: f64) -> SystemHealth {
        SystemHealth::new(
            HealthStatus::Ready,
            vec!["worker".to_string()],
            false,
            "/health".to_string(),
            "/live".to_string(),
            RealTrafficHealthConfig {
                window: Duration::from_secs(60),
                min_samples,
                failure_threshold,
            },
        )
    }

    #[test]
    fn real_traffic_health_fails_open_until_min_samples() {
        let health = test_system_health(3, 0.8);

        health.record_endpoint_request_result("worker", false);
        health.record_endpoint_request_result("worker", false);

        assert_eq!(
            health.get_endpoint_real_traffic_health_status("worker"),
            Some(HealthStatus::Ready)
        );
    }

    #[test]
    fn real_traffic_health_threshold_is_inclusive() {
        let health = test_system_health(5, 0.8);

        for _ in 0..4 {
            health.record_endpoint_request_result("worker", false);
        }
        health.record_endpoint_request_result("worker", true);

        assert_eq!(
            health.get_endpoint_real_traffic_health_status("worker"),
            Some(HealthStatus::NotReady)
        );
    }

    #[test]
    fn real_traffic_health_ignores_overloads() {
        let health = test_system_health(3, 0.8);

        for _ in 0..10 {
            health.record_endpoint_request_overload("worker");
        }

        assert_eq!(
            health.get_endpoint_real_traffic_health_status("worker"),
            Some(HealthStatus::Ready)
        );
    }

    #[test]
    fn real_traffic_health_tracks_recent_success() {
        let health = test_system_health(1, 0.8);

        assert!(!health.has_recent_endpoint_success("worker", Duration::from_secs(60)));
        health.record_endpoint_request_result("worker", true);

        assert!(health.has_recent_endpoint_success("worker", Duration::from_secs(60)));
    }
}
