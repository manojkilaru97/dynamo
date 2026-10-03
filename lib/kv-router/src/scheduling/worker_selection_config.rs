// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! YAML schema for process-wide worker-selection policy instances.
//!
//! Mirrors upstream `worker_selection` (ai-dynamo/dynamo#12536) so the same router-policy
//! document selects a shipped policy on this branch and on upstream builds. This branch does not
//! carry the upstream plugin registry; the policy types it links are resolved in
//! [`super::two_tier_cost_fn`] and rejected at startup when unknown.

use std::collections::HashMap;

use serde::Deserialize;

use super::policy_config::{RouterPolicyConfigError, validate_identifier};
use super::two_tier_cost_fn::{self, TwoTierCostFnParameters};

/// Worker pool a selector instance serves, matching upstream `WorkerType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerSelectionStage {
    /// Full-request workers (aggregated serving).
    Aggregated,
    Prefill,
    Decode,
    Encode,
}

impl WorkerSelectionStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Aggregated => "aggregated",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Encode => "encode",
        }
    }
}

/// A resolved, linked worker-selection policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkerSelectionPolicyKind {
    TwoTierCostFn(TwoTierCostFnParameters),
}

/// Process-wide worker-selection configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerSelectionConfig {
    aggregated: Option<String>,
    prefill: Option<String>,
    decode: Option<String>,
    encode: Option<String>,
    instances: HashMap<String, WorkerSelectionInstance>,
}

impl WorkerSelectionConfig {
    /// The selected instance for a full-request worker pool.
    pub fn aggregated_instance(&self) -> Option<&str> {
        self.aggregated.as_deref()
    }

    pub fn prefill_instance(&self) -> Option<&str> {
        self.prefill.as_deref()
    }

    pub fn decode_instance(&self) -> Option<&str> {
        self.decode.as_deref()
    }

    pub fn encode_instance(&self) -> Option<&str> {
        self.encode.as_deref()
    }

    /// The instance name selected for one stage; `None` and `default` both mean the built-in
    /// selector.
    pub fn instance_name_for(&self, stage: WorkerSelectionStage) -> Option<&str> {
        let selected = match stage {
            WorkerSelectionStage::Aggregated => self.aggregated_instance(),
            WorkerSelectionStage::Prefill => self.prefill_instance(),
            WorkerSelectionStage::Decode => self.decode_instance(),
            WorkerSelectionStage::Encode => self.encode_instance(),
        };
        selected.filter(|name| *name != "default")
    }

    /// The linked policy selected for one stage, or `None` for the built-in selector.
    pub fn policy_for(&self, stage: WorkerSelectionStage) -> Option<WorkerSelectionPolicyKind> {
        self.instance_name_for(stage)
            .and_then(|name| self.instances.get(name))
            .map(|instance| instance.kind)
    }

    /// Look up one named instance.
    pub fn instance(&self, name: &str) -> Option<&WorkerSelectionInstance> {
        self.instances.get(name)
    }

    /// Return configured instance names in stable order for diagnostics.
    pub fn instance_names(&self) -> Vec<String> {
        let mut names = self.instances.keys().cloned().collect::<Vec<_>>();
        names.sort_unstable();
        names
    }
}

/// One named, parameterized worker-selection policy instance.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerSelectionInstance {
    policy_type: String,
    parameters: serde_yaml::Value,
    kind: WorkerSelectionPolicyKind,
}

impl WorkerSelectionInstance {
    /// The policy type, e.g. `dynamo-two-tier-cost-fn`.
    pub fn policy_type(&self) -> &str {
        &self.policy_type
    }

    /// YAML parameters as written.
    pub fn parameters(&self) -> &serde_yaml::Value {
        &self.parameters
    }

    /// The resolved policy.
    pub fn kind(&self) -> WorkerSelectionPolicyKind {
        self.kind
    }
}

/// Policy types linked into this build, in diagnostic order.
pub const LINKED_POLICY_TYPES: &[&str] = &[two_tier_cost_fn::POLICY_TYPE];

fn resolve_kind(
    name: &str,
    policy_type: &str,
    parameters: &serde_yaml::Value,
) -> Result<WorkerSelectionPolicyKind, RouterPolicyConfigError> {
    match policy_type {
        two_tier_cost_fn::POLICY_TYPE => TwoTierCostFnParameters::from_yaml(parameters)
            .map(WorkerSelectionPolicyKind::TwoTierCostFn)
            .map_err(|error| {
                RouterPolicyConfigError::Validation(format!(
                    "worker_selection instance {name:?} of type {policy_type:?}: {error}"
                ))
            }),
        other => Err(RouterPolicyConfigError::Validation(format!(
            "worker_selection instance {name:?} has unknown policy type {other:?}; this build links: {}",
            LINKED_POLICY_TYPES.join(", ")
        ))),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RawWorkerSelectionConfig {
    aggregated: Option<String>,
    prefill: Option<String>,
    decode: Option<String>,
    encode: Option<String>,
    #[serde(default)]
    instances: Vec<RawWorkerSelectionInstance>,
}

impl RawWorkerSelectionConfig {
    pub(super) fn resolve(self) -> Result<WorkerSelectionConfig, RouterPolicyConfigError> {
        if self.instances.is_empty()
            && self.aggregated.is_none()
            && self.prefill.is_none()
            && self.decode.is_none()
            && self.encode.is_none()
        {
            return Err(RouterPolicyConfigError::Validation(
                "worker_selection must define an instance or an aggregated, prefill, decode, or encode selection"
                    .to_string(),
            ));
        }

        let mut instances = HashMap::with_capacity(self.instances.len());
        for raw in self.instances {
            validate_identifier(&raw.name, "instance", "worker_selection")?;
            if raw.name == "default" {
                return Err(RouterPolicyConfigError::Validation(
                    "worker_selection instance name 'default' is reserved for Dynamo's built-in selector".to_string(),
                ));
            }
            validate_identifier(&raw.policy_type, "policy type", "worker_selection")?;
            if !matches!(raw.parameters, serde_yaml::Value::Mapping(_)) {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "worker_selection instance {:?} parameters must be a YAML mapping",
                    raw.name
                )));
            }
            let kind = resolve_kind(&raw.name, &raw.policy_type, &raw.parameters)?;
            let instance = WorkerSelectionInstance {
                policy_type: raw.policy_type,
                parameters: raw.parameters,
                kind,
            };
            if instances.insert(raw.name.clone(), instance).is_some() {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "worker_selection contains duplicate instance {:?}",
                    raw.name
                )));
            }
        }

        for (stage, selected) in [
            ("aggregated", self.aggregated.as_deref()),
            ("prefill", self.prefill.as_deref()),
            ("decode", self.decode.as_deref()),
            ("encode", self.encode.as_deref()),
        ] {
            if let Some(selected) = selected
                && selected != "default"
                && !instances.contains_key(selected)
            {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "worker_selection {stage} {selected:?} does not name a configured instance"
                )));
            }
        }

        Ok(WorkerSelectionConfig {
            aggregated: self.aggregated,
            prefill: self.prefill,
            decode: self.decode,
            encode: self.encode,
            instances,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWorkerSelectionInstance {
    name: String,
    #[serde(rename = "type")]
    policy_type: String,
    #[serde(default = "empty_parameters")]
    parameters: serde_yaml::Value,
}

fn empty_parameters() -> serde_yaml::Value {
    serde_yaml::Value::Mapping(Default::default())
}
