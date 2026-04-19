// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::component::{Client, Component, Endpoint, Instance};
use crate::config::HealthStatus;
use crate::pipeline::PushRouter;
use crate::pipeline::{AsyncEngine, Context, ManyOut, SingleIn};
use crate::protocols::annotated::Annotated;
use crate::protocols::maybe_error::MaybeError;
use crate::{DistributedRuntime, SystemHealth};
use futures::StreamExt;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Configuration for health check behavior
pub struct HealthCheckConfig {
    /// Wait time before sending canary health checks (when no activity)
    pub canary_wait_time: Duration,
    /// Timeout for health check requests
    pub request_timeout: Duration,
    /// How long a worker can go without a fully successful health check before it is fenced
    pub success_ttl: Duration,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            canary_wait_time: Duration::from_secs(crate::config::DEFAULT_CANARY_WAIT_TIME_SECS),
            request_timeout: Duration::from_secs(
                crate::config::DEFAULT_HEALTH_CHECK_REQUEST_TIMEOUT_SECS,
            ),
            success_ttl: Duration::from_secs(crate::config::DEFAULT_HEALTH_CHECK_SUCCESS_TTL_SECS),
        }
    }
}

// Type alias for the router cache to improve readability
// Maps endpoint subject -> router and payload
type RouterCache =
    Arc<Mutex<HashMap<String, Arc<PushRouter<serde_json::Value, Annotated<serde_json::Value>>>>>>;

fn should_keep_endpoint_ready_after_failure(
    last_success: Option<Instant>,
    now: Instant,
    success_ttl: Duration,
) -> bool {
    last_success
        .map(|last_success| now.saturating_duration_since(last_success) < success_ttl)
        .unwrap_or(false)
}

async fn consume_health_check_stream<S, T>(mut response_stream: S) -> anyhow::Result<usize>
where
    S: futures::Stream<Item = T> + Unpin,
    T: MaybeError,
{
    let mut response_count = 0usize;

    while let Some(response) = response_stream.next().await {
        response_count += 1;
        if let Some(error) = response.err() {
            return Err(anyhow::anyhow!(
                "Health check returned an error response after {} response item(s): {}",
                response_count,
                error
            ));
        }
    }

    if response_count == 0 {
        Err(anyhow::anyhow!("Health check got no response"))
    } else {
        Ok(response_count)
    }
}

/// Health check manager that monitors endpoint health
pub struct HealthCheckManager {
    drt: DistributedRuntime,
    config: HealthCheckConfig,
    /// Cache of PushRouters and payloads for each endpoint
    router_cache: RouterCache,
    /// Last time an endpoint completed a full health check successfully
    last_success: Arc<Mutex<HashMap<String, Instant>>>,
    /// Track per-endpoint health check tasks
    /// Maps: endpoint_subject -> task_handle
    endpoint_tasks: Arc<Mutex<HashMap<String, JoinHandle<()>>>>,
}

impl HealthCheckManager {
    pub fn new(drt: DistributedRuntime, config: HealthCheckConfig) -> Self {
        Self {
            drt,
            config,
            router_cache: Arc::new(Mutex::new(HashMap::new())),
            last_success: Arc::new(Mutex::new(HashMap::new())),
            endpoint_tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn last_success(&self, endpoint_subject: &str) -> Option<Instant> {
        self.last_success.lock().get(endpoint_subject).copied()
    }

    fn should_probe_on_activity(&self, endpoint_subject: &str) -> bool {
        !should_keep_endpoint_ready_after_failure(
            self.last_success(endpoint_subject),
            Instant::now(),
            self.config.success_ttl,
        )
    }

    fn mark_endpoint_ready(&self, endpoint_subject: &str) {
        self.last_success
            .lock()
            .insert(endpoint_subject.to_string(), Instant::now());
        self.drt
            .system_health()
            .lock()
            .set_endpoint_health_status(endpoint_subject, HealthStatus::Ready);
    }

    fn mark_endpoint_not_ready_if_stale(&self, endpoint_subject: &str, failure: &str) {
        let now = Instant::now();
        let last_success = self.last_success(endpoint_subject);

        if should_keep_endpoint_ready_after_failure(last_success, now, self.config.success_ttl) {
            if let Some(last_success) = last_success {
                warn!(
                    "Health check failed for {} but last full success was {:?} ago (< {:?}); keeping endpoint ready. Failure: {}",
                    endpoint_subject,
                    now.saturating_duration_since(last_success),
                    self.config.success_ttl,
                    failure
                );
            }
            return;
        }

        warn!(
            "Marking {} as not ready after health check failure and stale success window {:?}. Failure: {}",
            endpoint_subject, self.config.success_ttl, failure
        );
        self.drt
            .system_health()
            .lock()
            .set_endpoint_health_status(endpoint_subject, HealthStatus::NotReady);
    }

    /// Get or create a PushRouter for an endpoint
    async fn get_or_create_router(
        &self,
        cache_key: &str,
        endpoint: Endpoint,
    ) -> anyhow::Result<Arc<PushRouter<serde_json::Value, Annotated<serde_json::Value>>>> {
        let cache_key = cache_key.to_string();

        // Check cache first
        {
            let cache = self.router_cache.lock();
            if let Some(router) = cache.get(&cache_key) {
                return Ok(router.clone());
            }
        }

        // Create a client that discovers instances dynamically for this endpoint
        let client = Client::new(endpoint).await?;

        // Create PushRouter - it will use direct routing when we call direct()
        let router: Arc<PushRouter<serde_json::Value, Annotated<serde_json::Value>>> = Arc::new(
            PushRouter::from_client(
                client,
                crate::pipeline::RouterMode::RoundRobin, // Default mode, we'll use direct() explicitly
            )
            .await?,
        );

        // Cache it
        self.router_cache.lock().insert(cache_key, router.clone());

        Ok(router)
    }

    /// Start the health check manager by spawning per-endpoint monitoring tasks
    pub async fn start(self: Arc<Self>) -> anyhow::Result<()> {
        // Get all registered endpoints at startup
        let targets = self.drt.system_health().lock().get_health_check_targets();

        info!(
            "Starting health check tasks for {} endpoints with canary_wait_time: {:?}",
            targets.len(),
            self.config.canary_wait_time
        );

        // Spawn a health check task for each registered endpoint
        for (endpoint_subject, _target) in targets {
            self.spawn_endpoint_health_check_task(endpoint_subject);
        }

        // CRITICAL: Spawn a task to monitor for NEW endpoints registered after startup
        // This uses a channel-based approach to guarantee no lost notifications
        // Will return an error if the receiver has already been taken
        self.spawn_new_endpoint_monitor().await?;

        info!("HealthCheckManager started successfully with channel-based endpoint discovery");
        Ok(())
    }

    /// Spawn a dedicated health check task for a specific endpoint
    fn spawn_endpoint_health_check_task(self: &Arc<Self>, endpoint_subject: String) {
        let manager = self.clone();
        let canary_wait = self.config.canary_wait_time;
        let endpoint_subject_clone = endpoint_subject.clone();

        // Get the endpoint-specific notifier
        let notifier = self
            .drt
            .system_health()
            .lock()
            .get_endpoint_health_check_notifier(&endpoint_subject)
            .expect("Notifier should exist for registered endpoint");

        let task = tokio::spawn(async move {
            let endpoint_subject = endpoint_subject_clone;
            info!("Health check task started for: {}", endpoint_subject);

            loop {
                // Wait for either timeout or activity notification
                tokio::select! {
                    _ = tokio::time::sleep(canary_wait) => {
                        // Timeout - send health check for this specific endpoint
                        debug!("Canary timer expired for {}, sending health check", endpoint_subject);

                        // Get the health check payload for this endpoint
                        let target = manager.drt.system_health().lock().get_health_check_target(&endpoint_subject);

                        if let Some(target) = target {
                            if let Err(e) = manager.send_health_check_request(&endpoint_subject, &target.payload).await {
                                error!("Failed to send health check for {}: {}", endpoint_subject, e);
                            }
                        } else {
                            // This should never happen - targets are registered at startup and never removed
                            error!(
                                "CRITICAL: Health check target for {} disappeared unexpectedly! This indicates a bug. Stopping health check task.",
                                endpoint_subject
                            );
                            break;
                        }
                    }

                    _ = notifier.notified() => {
                        if manager.should_probe_on_activity(&endpoint_subject) {
                            debug!(
                                "Activity detected for {} with stale health-check success; sending immediate health check",
                                endpoint_subject
                            );

                            let target = manager.drt.system_health().lock().get_health_check_target(&endpoint_subject);

                            if let Some(target) = target {
                                if let Err(e) = manager.send_health_check_request(&endpoint_subject, &target.payload).await {
                                    error!("Failed to send activity-triggered health check for {}: {}", endpoint_subject, e);
                                }
                            } else {
                                error!(
                                    "CRITICAL: Health check target for {} disappeared unexpectedly during activity-triggered probe!",
                                    endpoint_subject
                                );
                                break;
                            }
                        } else {
                            debug!("Activity detected for {}, resetting health check timer", endpoint_subject);
                        }
                    }
                }
            }

            info!("Health check task for {} exiting", endpoint_subject);
        });

        // Store the task handle
        self.endpoint_tasks
            .lock()
            .insert(endpoint_subject.clone(), task);

        info!(
            "Spawned health check task for endpoint: {}",
            endpoint_subject
        );
    }

    /// Spawn a task to monitor for newly registered endpoints
    /// Returns an error if duplicate endpoints are detected, indicating a bug in the system
    async fn spawn_new_endpoint_monitor(self: &Arc<Self>) -> anyhow::Result<()> {
        let manager = self.clone();

        // Get the receiver (can only be taken once)
        let mut rx = manager
            .drt
            .system_health()
            .lock()
            .take_new_endpoint_receiver()
            .ok_or_else(|| {
                anyhow::anyhow!("Endpoint receiver already taken - this should only be called once")
            })?;

        tokio::spawn(async move {
            info!("Starting dynamic endpoint discovery monitor with channel-based notifications");

            while let Some(endpoint_subject) = rx.recv().await {
                debug!(
                    "Received endpoint registration via channel: {}",
                    endpoint_subject
                );

                let already_exists = {
                    let tasks = manager.endpoint_tasks.lock();
                    tasks.contains_key(&endpoint_subject)
                };

                if already_exists {
                    error!(
                        "CRITICAL: Received registration for endpoint '{}' that already has a health check task!",
                        endpoint_subject
                    );
                    break;
                }

                info!(
                    "Spawning health check task for new endpoint: {}",
                    endpoint_subject
                );
                manager.spawn_endpoint_health_check_task(endpoint_subject);
            }

            info!("Endpoint discovery monitor exiting - no new endpoints will be monitored!");
        });

        info!("Dynamic endpoint discovery monitor started");
        Ok(())
    }

    /// Send a health check request through AsyncEngine
    async fn send_health_check_request(
        &self,
        endpoint_subject: &str,
        payload: &serde_json::Value,
    ) -> anyhow::Result<()> {
        let target = self
            .drt
            .system_health()
            .lock()
            .get_health_check_target(endpoint_subject)
            .ok_or_else(|| {
                anyhow::anyhow!("No health check target found for {}", endpoint_subject)
            })?;

        debug!(
            "Sending health check to {} (instance_id: {})",
            endpoint_subject, target.instance.instance_id
        );

        // Create the Endpoint directly from the Instance info
        let namespace = self.drt.namespace(&target.instance.namespace)?;
        let component = namespace.component(&target.instance.component)?;
        let endpoint = component.endpoint(&target.instance.endpoint);

        // Get or create router for this endpoint
        let router = match self.get_or_create_router(endpoint_subject, endpoint).await {
            Ok(router) => router,
            Err(e) => {
                self.mark_endpoint_not_ready_if_stale(endpoint_subject, &e.to_string());
                return Err(e);
            }
        };

        // Wait for watch stream to discover instances before checking
        // This ensures the router's client has populated its instance list
        // from etcd before we attempt to send the health check request.
        // Without this, the first health check can fail due to a race condition
        // where the watch stream hasn't completed its initial discovery yet.
        match tokio::time::timeout(
            Duration::from_secs(10), // 10 second timeout for discovery
            router.client.wait_for_instances(),
        )
        .await
        {
            Ok(Ok(instances)) => {
                debug!(
                    "Health check for {}: watch stream ready, found {} instance(s)",
                    endpoint_subject,
                    instances.len()
                );
            }
            Ok(Err(e)) => {
                let err = anyhow::anyhow!(
                    "Failed to discover instances for {} during health check: {}",
                    endpoint_subject,
                    e
                );
                self.mark_endpoint_not_ready_if_stale(endpoint_subject, &err.to_string());
                return Err(err);
            }
            Err(_) => {
                let err = anyhow::anyhow!(
                    "Timeout waiting for instance discovery for {} during health check",
                    endpoint_subject
                );
                self.mark_endpoint_not_ready_if_stale(endpoint_subject, &err.to_string());
                return Err(err);
            }
        }

        // Create the request context
        let request: SingleIn<serde_json::Value> = Context::new(payload.clone());
        let instance_id = target.instance.instance_id;
        let timeout = self.config.request_timeout;

        let result = tokio::time::timeout(timeout, async {
            match router.direct(request, instance_id).await {
                Ok(response_stream) => consume_health_check_stream(response_stream).await,
                Err(e) => Err(anyhow::anyhow!(
                    "Health check request failed for {}: {}",
                    endpoint_subject,
                    e
                )),
            }
        })
        .await;

        match result {
            Ok(Ok(response_count)) => {
                debug!(
                    "Health check completed successfully for {} after consuming {} response item(s)",
                    endpoint_subject, response_count
                );
                self.mark_endpoint_ready(endpoint_subject);
                Ok(())
            }
            Ok(Err(e)) => {
                self.mark_endpoint_not_ready_if_stale(endpoint_subject, &e.to_string());
                Err(e)
            }
            Err(_) => {
                let err = anyhow::anyhow!(
                    "Health check timed out for {} after {:?}",
                    endpoint_subject,
                    timeout
                );
                self.mark_endpoint_not_ready_if_stale(endpoint_subject, &err.to_string());
                Err(err)
            }
        }
    }
}

/// Start health check manager for the distributed runtime
pub async fn start_health_check_manager(
    drt: DistributedRuntime,
    config: Option<HealthCheckConfig>,
) -> anyhow::Result<()> {
    let config = config.unwrap_or_default();
    let manager = Arc::new(HealthCheckManager::new(drt, config));

    // Start the health check manager (this spawns per-endpoint tasks internally)
    manager.start().await?;

    Ok(())
}

/// Get health check status for all endpoints
pub async fn get_health_check_status(
    drt: &DistributedRuntime,
) -> anyhow::Result<serde_json::Value> {
    // Get endpoints list from SystemHealth
    let endpoint_subjects: Vec<String> = drt.system_health().lock().get_health_check_endpoints();

    let mut endpoint_statuses = HashMap::new();

    // Check each endpoint's health status
    {
        let system_health = drt.system_health();
        let system_health_lock = system_health.lock();
        for endpoint_subject in &endpoint_subjects {
            let health_status = system_health_lock
                .get_endpoint_health_status(endpoint_subject)
                .unwrap_or(HealthStatus::NotReady);

            let is_healthy = matches!(health_status, HealthStatus::Ready);

            endpoint_statuses.insert(
                endpoint_subject.clone(),
                serde_json::json!({
                    "healthy": is_healthy,
                    "status": format!("{:?}", health_status),
                }),
            );
        }
    }

    let overall_healthy = endpoint_statuses
        .values()
        .all(|v| v["healthy"].as_bool().unwrap_or(false));

    Ok(serde_json::json!({
        "status": if overall_healthy { "ready" } else { "notready" },
        "endpoints_checked": endpoint_subjects.len(),
        "endpoint_statuses": endpoint_statuses,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[tokio::test]
    async fn test_consume_health_check_stream_requires_full_completion() {
        let response_stream = stream::iter(vec![
            Annotated::from_data(serde_json::json!({"chunk": 1})),
            Annotated::from_data(serde_json::json!({"chunk": 2})),
        ]);

        let response_count = consume_health_check_stream(response_stream).await.unwrap();
        assert_eq!(response_count, 2);
    }

    #[tokio::test]
    async fn test_consume_health_check_stream_fails_if_error_arrives_after_first_chunk() {
        let response_stream = stream::iter(vec![
            Annotated::from_data(serde_json::json!({"chunk": 1})),
            Annotated::from_error("late stream failure".to_string()),
        ]);

        let err = consume_health_check_stream(response_stream)
            .await
            .expect_err("stream should fail when any later chunk is an error");
        assert!(err.to_string().contains("late stream failure"));
    }

    #[tokio::test]
    async fn test_consume_health_check_stream_rejects_empty_stream() {
        let response_stream = stream::iter(Vec::<Annotated<serde_json::Value>>::new());

        let err = consume_health_check_stream(response_stream)
            .await
            .expect_err("empty streams should be unhealthy");
        assert!(err.to_string().contains("no response"));
    }

    #[test]
    fn test_should_keep_endpoint_ready_after_failure_only_with_recent_success() {
        let now = Instant::now();
        let success_ttl = Duration::from_secs(120);

        assert!(should_keep_endpoint_ready_after_failure(
            Some(now - Duration::from_secs(60)),
            now,
            success_ttl,
        ));
        assert!(!should_keep_endpoint_ready_after_failure(
            Some(now - Duration::from_secs(121)),
            now,
            success_ttl,
        ));
        assert!(!should_keep_endpoint_ready_after_failure(
            None,
            now,
            success_ttl
        ));
    }
}

// ===============================
// Integration Tests (require DRT)
// ===============================
#[cfg(all(test, feature = "integration"))]
mod integration_tests {
    use super::*;
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn test_initialization() {
        let drt = create_test_drt_async().await;

        let canary_wait_time = Duration::from_secs(5);
        let request_timeout = Duration::from_secs(3);

        let config = HealthCheckConfig {
            canary_wait_time,
            request_timeout,
            success_ttl: Duration::from_secs(120),
        };

        let manager = HealthCheckManager::new(drt.clone(), config);

        assert_eq!(manager.config.canary_wait_time, canary_wait_time);
        assert_eq!(manager.config.request_timeout, request_timeout);
    }

    #[tokio::test]
    async fn test_payload_registration() {
        let drt = create_test_drt_async().await;

        let endpoint = "test.endpoint";
        let payload = serde_json::json!({
            "prompt": "test",
            "_health_check": true
        });

        drt.system_health().lock().register_health_check_target(
            endpoint,
            crate::component::Instance {
                component: "test_component".to_string(),
                endpoint: "test_endpoint".to_string(),
                namespace: "test_namespace".to_string(),
                instance_id: 12345,
                transport: crate::component::TransportType::Nats(endpoint.to_string()),
            },
            payload.clone(),
        );

        let retrieved = drt
            .system_health()
            .lock()
            .get_health_check_target(endpoint)
            .map(|t| t.payload);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), payload);

        // Verify endpoint appears in the list
        let endpoints = drt.system_health().lock().get_health_check_endpoints();
        assert!(endpoints.contains(&endpoint.to_string()));
    }

    #[tokio::test]
    async fn test_spawn_per_endpoint_tasks() {
        let drt = create_test_drt_async().await;

        for i in 0..3 {
            let endpoint = format!("test.endpoint.{}", i);
            let payload = serde_json::json!({
                "prompt": format!("test{}", i),
                "_health_check": true
            });
            drt.system_health().lock().register_health_check_target(
                &endpoint,
                crate::component::Instance {
                    component: "test_component".to_string(),
                    endpoint: format!("test_endpoint_{}", i),
                    namespace: "test_namespace".to_string(),
                    instance_id: i,
                    transport: crate::component::TransportType::Nats(endpoint.clone()),
                },
                payload,
            );
        }

        let config = HealthCheckConfig {
            canary_wait_time: Duration::from_secs(5),
            request_timeout: Duration::from_secs(1),
            success_ttl: Duration::from_secs(120),
        };

        let manager = Arc::new(HealthCheckManager::new(drt.clone(), config));
        manager.clone().start().await.unwrap();

        // Verify all endpoints have their own health check tasks
        let tasks = manager.endpoint_tasks.lock();
        // Should have 3 tasks (one for each endpoint)
        assert_eq!(tasks.len(), 3);
        // Check that all endpoints are represented in tasks
        let endpoints: Vec<String> = tasks.keys().cloned().collect();
        assert!(endpoints.contains(&"test.endpoint.0".to_string()));
        assert!(endpoints.contains(&"test.endpoint.1".to_string()));
        assert!(endpoints.contains(&"test.endpoint.2".to_string()));
    }

    #[tokio::test]
    async fn test_endpoint_health_check_notifier_created() {
        let drt = create_test_drt_async().await;

        let endpoint = "test.endpoint.notifier";
        let payload = serde_json::json!({
            "prompt": "test",
            "_health_check": true
        });

        // Register the endpoint
        drt.system_health().lock().register_health_check_target(
            endpoint,
            crate::component::Instance {
                component: "test_component".to_string(),
                endpoint: "test_endpoint_notifier".to_string(),
                namespace: "test_namespace".to_string(),
                instance_id: 999,
                transport: crate::component::TransportType::Nats(endpoint.to_string()),
            },
            payload.clone(),
        );

        // Verify that a notifier was created for this endpoint
        let notifier = drt
            .system_health()
            .lock()
            .get_endpoint_health_check_notifier(endpoint);

        assert!(
            notifier.is_some(),
            "Endpoint should have a notifier created"
        );

        // Verify we can notify it without panicking
        if let Some(notifier) = notifier {
            notifier.notify_one();
        }

        // Initially, the endpoint should be Ready (default after registration)
        let status = drt
            .system_health()
            .lock()
            .get_endpoint_health_status(endpoint);
        assert_eq!(status, Some(HealthStatus::NotReady));
    }
}
