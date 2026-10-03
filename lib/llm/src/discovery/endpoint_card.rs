// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::{DiscoveryEvent, DiscoveryQuery};
use dynamo_runtime::prelude::DistributedRuntimeProvider;

use crate::model_card::ModelDeploymentCard;

/// Wait for a worker on `endpoint` to publish its `ModelDeploymentCard`.
///
/// Uses the watch-based discovery API so the wait is event-driven (no polling)
/// and returns as soon as the first card is observed. Existing registrations
/// are delivered as `Added` events at the start of the stream, so callers do
/// not need to issue a separate `list` first.
///
/// Returns `Ok(Some(card))` once a card is observed, or `Ok(None)` if `timeout`
/// elapses, the supplied `cancel_token` fires, or the discovery stream ends
/// without ever delivering a deserializable model card. When `cancel_token` is
/// `None`, the runtime's primary token is used so the wait aborts on shutdown.
pub async fn wait_for_endpoint_model_card(
    endpoint: &Endpoint,
    timeout: Duration,
    cancel_token: Option<CancellationToken>,
) -> Result<Option<ModelDeploymentCard>> {
    wait_for_endpoint_model_card_where(endpoint, timeout, cancel_token, |_| true).await
}

/// [`wait_for_endpoint_model_card`], returning only a card that satisfies `accept`.
///
/// Cards that fail `accept` (e.g. a legacy card without a worker role) are skipped and the wait
/// continues until an accepted card arrives or the same `timeout` / `cancel_token` ends it.
pub async fn wait_for_endpoint_model_card_where(
    endpoint: &Endpoint,
    timeout: Duration,
    cancel_token: Option<CancellationToken>,
    accept: impl Fn(&ModelDeploymentCard) -> bool,
) -> Result<Option<ModelDeploymentCard>> {
    let cancel_token = cancel_token.unwrap_or_else(|| endpoint.drt().primary_token());
    let eid = endpoint.id();
    let query = DiscoveryQuery::EndpointModels {
        namespace: eid.namespace,
        component: eid.component,
        endpoint: eid.name,
    };

    let mut stream = endpoint
        .drt()
        .discovery()
        .list_and_watch(query, Some(cancel_token.clone()))
        .await?;

    let find_card = async {
        while let Some(event) = stream.next().await {
            match event {
                Ok(DiscoveryEvent::Added(instance)) => {
                    let card = match instance.deserialize_model::<ModelDeploymentCard>() {
                        Ok(card) => card,
                        Err(error) => {
                            tracing::warn!(
                                %error,
                                discovery_instance = ?instance.id(),
                                "Failed to deserialize model card while waiting for endpoint registration; continuing"
                            );
                            continue;
                        }
                    };
                    if !accept(&card) {
                        tracing::debug!(
                            discovery_instance = ?instance.id(),
                            "Skipping a model card that does not satisfy the wait; continuing"
                        );
                        continue;
                    }
                    return Some(card);
                }
                Ok(DiscoveryEvent::Removed(_)) => {}
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        "Discovery event error while waiting for endpoint model card; continuing"
                    );
                }
            }
        }
        None
    };

    Ok(tokio::select! {
        card = find_card => card,
        _ = tokio::time::sleep(timeout) => None,
        _ = cancel_token.cancelled() => None,
    })
}

#[cfg(test)]
mod tests {
    use dynamo_runtime::discovery::DiscoverySpec;
    use dynamo_runtime::distributed::DistributedConfig;
    use dynamo_runtime::{DistributedRuntime, Runtime};

    use super::*;
    use crate::worker_type::WorkerType;

    /// A standalone router that needs the worker role (worker_selection) or model name waits for
    /// a card registered after it starts, instead of resolving from an empty snapshot.
    #[tokio::test]
    async fn waits_for_a_typed_card_registered_after_the_router_starts() {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("delayed-card".to_string())
            .unwrap()
            .component("decode".to_string())
            .unwrap()
            .endpoint("generate");
        let eid = endpoint.id();

        let snapshot = drt
            .discovery()
            .list(DiscoveryQuery::EndpointModels {
                namespace: eid.namespace.clone(),
                component: eid.component.clone(),
                endpoint: eid.name.clone(),
            })
            .await
            .unwrap();
        assert!(snapshot.is_empty(), "nothing is registered yet");

        let waiter = {
            let endpoint = endpoint.clone();
            tokio::spawn(async move {
                wait_for_endpoint_model_card(&endpoint, Duration::from_secs(30), None).await
            })
        };
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            !waiter.is_finished(),
            "the wait must block until a card registers"
        );

        let mut card = ModelDeploymentCard::with_name_only("delayed-model");
        card.worker_type = Some(WorkerType::Decode);
        let _registration = drt
            .discovery()
            .register(
                DiscoverySpec::from_model(eid.namespace, eid.component, eid.name, &card).unwrap(),
            )
            .await
            .unwrap();

        let found = tokio::time::timeout(Duration::from_secs(10), waiter)
            .await
            .expect("the wait must return once the card registers")
            .unwrap()
            .unwrap()
            .expect("card");
        assert_eq!(found.worker_type, Some(WorkerType::Decode));
        assert_eq!(found.display_name, "delayed-model");

        drt.shutdown();
        runtime.shutdown();
    }

    /// An untyped (legacy) card already registered does not end a wait that needs a worker role;
    /// the typed card registered later does.
    #[tokio::test]
    async fn typed_card_wait_skips_an_untyped_card() {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("untyped-then-typed".to_string())
            .unwrap()
            .component("decode".to_string())
            .unwrap()
            .endpoint("generate");
        let eid = endpoint.id();
        // Distinct suffixes give two cards of the same model separate keys on one endpoint.
        let register = |card: ModelDeploymentCard, suffix: &'static str| {
            let drt = drt.clone();
            let eid = eid.clone();
            async move {
                drt.discovery()
                    .register(
                        DiscoverySpec::from_model_with_suffix(
                            eid.namespace,
                            eid.component,
                            eid.name,
                            &card,
                            Some(suffix.to_string()),
                        )
                        .unwrap(),
                    )
                    .await
                    .unwrap()
            }
        };

        let untyped = ModelDeploymentCard::with_name_only("shared-model");
        assert!(untyped.worker_type.is_none());
        let _untyped_registration = register(untyped, "legacy").await;

        let waiter = {
            let endpoint = endpoint.clone();
            tokio::spawn(async move {
                wait_for_endpoint_model_card_where(
                    &endpoint,
                    Duration::from_secs(30),
                    None,
                    |card| card.worker_type.is_some(),
                )
                .await
            })
        };
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            !waiter.is_finished(),
            "an untyped card must not end the wait"
        );

        let mut typed = ModelDeploymentCard::with_name_only("shared-model");
        typed.worker_type = Some(WorkerType::Decode);
        let _typed_registration = register(typed, "typed").await;

        let found = tokio::time::timeout(Duration::from_secs(10), waiter)
            .await
            .expect("the typed card must end the wait")
            .unwrap()
            .unwrap()
            .expect("card");
        assert_eq!(found.worker_type, Some(WorkerType::Decode));

        drt.shutdown();
        runtime.shutdown();
    }
}
