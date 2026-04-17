// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{RouteDoc, service_v2};
use axum::{Json, Router, http::Method, http::StatusCode, response::IntoResponse, routing::get};
use crate::endpoint_type::EndpointType;
use dynamo_runtime::instances::list_all_instances;
use serde::Serialize;
use serde_json::json;
use std::sync::Arc;

#[derive(Serialize)]
struct EndpointReadiness {
    endpoint: String,
    enabled: bool,
    ready: bool,
    models: Vec<String>,
    message: String,
}

#[derive(Serialize)]
struct ReadinessReport {
    status: &'static str,
    ready: bool,
    message: String,
    endpoints: Vec<EndpointReadiness>,
}

pub fn health_check_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let health_path = path.unwrap_or_else(|| "/health".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &health_path)];

    let router = Router::new()
        .route(&health_path, get(health_handler))
        .with_state(state);

    (docs, router)
}

pub fn live_check_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let live_path = path.unwrap_or_else(|| "/live".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &live_path)];

    let router = Router::new()
        .route(&live_path, get(live_handler))
        .with_state(state);

    (docs, router)
}

pub fn ready_check_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let ready_path = path.unwrap_or_else(|| "/ready".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &ready_path)];

    let router = Router::new()
        .route(&ready_path, get(ready_handler))
        .with_state(state);

    (docs, router)
}

async fn live_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    // Check if the http service is being cancelled/shutdown
    if state.is_cancelled() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "shutting_down",
                "message": "Service is shutting down"
            })),
        );
    }

    (
        StatusCode::OK,
        Json(json!({
            "status": "live",
            "message": "Service is live"
        })),
    )
}

pub fn check_frontend_ready(state: &service_v2::State) -> Result<(), String> {
    let report = build_readiness_report(state);
    if report.ready {
        Ok(())
    } else {
        Err(report.message)
    }
}

fn models_for_endpoint(state: &service_v2::State, endpoint_type: EndpointType) -> Vec<String> {
    let mut models = match endpoint_type {
        EndpointType::Chat => state.manager().list_chat_completions_models(),
        EndpointType::Completion => state.manager().list_completions_models(),
        EndpointType::Embedding => state.manager().list_embeddings_models(),
        EndpointType::Images => state.manager().list_images_models(),
        EndpointType::Videos => state.manager().list_videos_models(),
        EndpointType::Responses | EndpointType::AnthropicMessages => {
            state.manager().list_chat_completions_models()
        }
        EndpointType::Audios => Vec::new(),
    };
    models.sort();
    models.dedup();
    models
}

fn build_readiness_report(state: &service_v2::State) -> ReadinessReport {
    if state.is_cancelled() {
        return ReadinessReport {
            status: "not_ready",
            ready: false,
            message: "Service is shutting down".to_string(),
            endpoints: Vec::new(),
        };
    }

    let checked_endpoints = [
        EndpointType::Chat,
        EndpointType::Completion,
        EndpointType::Embedding,
        EndpointType::Images,
        EndpointType::Videos,
        EndpointType::Responses,
        EndpointType::AnthropicMessages,
    ];

    let endpoints: Vec<EndpointReadiness> = checked_endpoints
        .into_iter()
        .map(|endpoint_type| {
            let enabled = state.endpoint_enabled(endpoint_type);
            let models = if enabled {
                models_for_endpoint(state, endpoint_type)
            } else {
                Vec::new()
            };
            let ready = enabled && !models.is_empty();
            let message = if !enabled {
                "endpoint disabled".to_string()
            } else if ready {
                "routeable models available".to_string()
            } else {
                "endpoint enabled but no routeable models registered".to_string()
            };

            EndpointReadiness {
                endpoint: endpoint_type.as_str().to_string(),
                enabled,
                ready,
                models,
                message,
            }
        })
        .collect();

    let enabled_count = endpoints.iter().filter(|entry| entry.enabled).count();
    let text_endpoints = ["chat", "completion", "responses", "anthropic_messages"];
    let text_enabled: Vec<&EndpointReadiness> = endpoints
        .iter()
        .filter(|entry| entry.enabled && text_endpoints.contains(&entry.endpoint.as_str()))
        .collect();
    let text_ready = text_enabled.iter().any(|entry| entry.ready);
    let any_ready = endpoints.iter().any(|entry| entry.enabled && entry.ready);

    let (ready, message) = if enabled_count == 0 {
        (
            false,
            "No frontend endpoints are enabled for this service".to_string(),
        )
    } else if !text_enabled.is_empty() && !text_ready {
        let names: Vec<&str> = text_enabled
            .iter()
            .map(|entry| entry.endpoint.as_str())
            .collect();
        (
            false,
            format!(
                "Frontend is not ready: no routeable models registered for enabled text endpoints [{}]",
                names.join(", ")
            ),
        )
    } else if text_ready || any_ready {
        (
            true,
            "Frontend is ready: at least one enabled serving endpoint has a routeable model"
                .to_string(),
        )
    } else {
        (
            false,
            "Frontend is not ready: no routeable models registered for any enabled endpoint"
                .to_string(),
        )
    };

    ReadinessReport {
        status: if ready { "ready" } else { "not_ready" },
        ready,
        message,
        endpoints,
    }
}

async fn health_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    let instances = match list_all_instances(state.discovery()).await {
        Ok(instances) => instances,
        Err(err) => {
            tracing::warn!(%err, "Failed to fetch instances from discovery");
            vec![]
        }
    };
    let mut endpoints: Vec<String> = instances
        .iter()
        .map(|instance| instance.endpoint_id().as_url())
        .collect();
    endpoints.sort();
    endpoints.dedup();
    let readiness = build_readiness_report(&state);
    let status_code = if readiness.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status_code,
        Json(json!({
            "status": readiness.status,
            "ready": readiness.ready,
            "message": readiness.message,
            "frontend_endpoints": readiness.endpoints,
            "discovery_endpoints": endpoints,
            "instances": instances
        })),
    )
}

async fn ready_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    let readiness = build_readiness_report(&state);
    let status_code = if readiness.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status_code, Json(json!(readiness)))
}
