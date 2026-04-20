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

#[derive(Clone, Debug)]
struct RealTrafficWindow {
    events: VecDeque<(Instant, bool)>,
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
    fn record(&mut self, now: Instant, success: bool, config: &RealTrafficHealthConfig) {
        self.prune(now, config.window);
        self.events.push_back((now, success));
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
        let total = self.events.len();
        let failures = self.events.iter().filter(|(_, success)| !*success).count();
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
        self.events
            .iter()
            .rev()
            .any(|(ts, success)| *success && now.saturating_duration_since(*ts) <= within)
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
    start_time: Instant,
    uptime_gauge: OnceLock<prometheus::Gauge>,
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
            start_time: Instant::now(),
            uptime_gauge: OnceLock::new(),
        }
    }

    pub fn set_health_status(&mut self, status: HealthStatus) {
        self.system_health = status;
    }

    pub fn set_endpoint_health_status(&self, endpoint: &str, status: HealthStatus) {
        let mut endpoint_health = self.endpoint_health.write().unwrap();
        endpoint_health.insert(endpoint.to_string(), status);
    }

    pub fn record_endpoint_request_result(&self, endpoint: &str, success: bool) {
        let now = Instant::now();
        let mut windows = self.endpoint_real_traffic_health.write().unwrap();
        let window = windows.entry(endpoint.to_string()).or_default();
        window.record(now, success, &self.real_traffic_health_config);
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
                                 endpoint_real_traffic_health: &mut HashMap<String, RealTrafficWindow>| {
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
                endpoint_is_ready(endpoint, &endpoint_health, &mut endpoint_real_traffic_health)
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
            let mut endpoint_real_traffic_health = self.endpoint_real_traffic_health.write().unwrap();
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

        window.record(now - Duration::from_secs(601), false, &config);

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
        assert!(reasons
            .get("generate")
            .expect("missing generate reason")
            .contains("real traffic failure ratio"));
    }
}
