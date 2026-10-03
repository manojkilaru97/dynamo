// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Two-tier worker-selection cost function.
//!
//! Ported from upstream `lib/router-plugins/builtin/src/two_tier_cost_fn.rs`
//! (ai-dynamo/dynamo#14498). This branch has no
//! upstream plugin registry, so the policy is linked directly into the KV router and selected by
//! the same `worker_selection` router-policy YAML.
//!
//! Dynamo's built-in selector folds cache overlap and load into one additive cost. This policy
//! instead ranks on two tiers, taking the first that applies. For each eligible worker it reads
//! device-KV overlap and active-request count, then:
//!
//! 1. Load tier: if active-request spread exceeds `balance_abs_threshold` and the largest count
//!    exceeds `balance_rel_threshold` times the smallest, select the least-loaded worker.
//! 2. Cache tier: otherwise, if the largest device-KV overlap is strictly greater than
//!    `cache_threshold` of the request's block count, select the least-loaded worker holding that
//!    maximum overlap.
//! 3. Otherwise, select the least-loaded worker.
//!
//! Both load gates must hold to take step 1, so load displaces cache affinity only when the
//! imbalance is both large in absolute terms and disproportionate.
//!
//! The thresholds default to `experimental/sgl-router`'s `cache_aware_zmq` values, so an instance
//! with no `parameters` mapping reproduces it exactly.
//!
//! Ties between equally ranked workers resolve on candidate row order, which the host leaves
//! unspecified. This matches the ported implementation; note that Dynamo's built-in selector
//! instead samples uniformly among ties.

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-two-tier-cost-fn";

/// Keep these equal to `experimental/sgl-router`'s `cache_aware_zmq` defaults, so an instance
/// with no `parameters` mapping reproduces that policy exactly.
const DEFAULT_CACHE_THRESHOLD: f64 = 0.5;
const DEFAULT_BALANCE_ABS_THRESHOLD: usize = 32;
const DEFAULT_BALANCE_REL_THRESHOLD: f64 = 1.1;

/// Tunables for [`POLICY_TYPE`], named after their `sgl-router` counterparts.
///
/// Every field is optional and keeps the upstream default when omitted. Unknown keys are rejected
/// at startup rather than ignored, so a misremembered name fails loudly.
#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TwoTierCostFnParameters {
    /// Fraction of the request's blocks that must be device-resident on the best worker before the
    /// cache tier applies. Compared strictly.
    pub cache_threshold: f64,
    /// Minimum active-request spread before the load tier applies.
    pub balance_abs_threshold: usize,
    /// Minimum ratio of largest to smallest active-request count before the load tier applies.
    pub balance_rel_threshold: f64,
}

impl Default for TwoTierCostFnParameters {
    fn default() -> Self {
        Self {
            cache_threshold: DEFAULT_CACHE_THRESHOLD,
            balance_abs_threshold: DEFAULT_BALANCE_ABS_THRESHOLD,
            balance_rel_threshold: DEFAULT_BALANCE_REL_THRESHOLD,
        }
    }
}

impl TwoTierCostFnParameters {
    /// Deserialize and validate an instance `parameters` mapping.
    pub fn from_yaml(parameters: &serde_yaml::Value) -> Result<Self, String> {
        let parameters: Self = serde_yaml::from_value(parameters.clone())
            .map_err(|error| format!("invalid parameters: {error}"))?;
        parameters.validate()?;
        Ok(parameters)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.cache_threshold.is_finite() || !(0.0..=1.0).contains(&self.cache_threshold) {
            return Err("cache_threshold must be a finite number between 0.0 and 1.0".to_string());
        }
        if !self.balance_rel_threshold.is_finite() || self.balance_rel_threshold < 1.0 {
            return Err(
                "balance_rel_threshold must be a finite number greater than or equal to 1.0"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// A configured two-tier policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoTierCostFn {
    pub parameters: TwoTierCostFnParameters,
}

impl TwoTierCostFn {
    pub fn new(parameters: TwoTierCostFnParameters) -> Self {
        Self { parameters }
    }
}

/// One eligible worker's inputs to the two-tier decision.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TwoTierRow {
    pub device_overlap_blocks: f64,
    pub active_requests: usize,
}

/// Which tier decided the selection, for routing logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoTierDecision {
    Load,
    Cache,
    LeastLoaded,
}

impl TwoTierDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::Cache => "cache",
            Self::LeastLoaded => "least_loaded",
        }
    }
}

fn least_loaded(rows: &[TwoTierRow], candidates: impl Iterator<Item = usize>) -> Option<usize> {
    candidates.min_by_key(|&row| rows[row].active_requests)
}

impl TwoTierCostFn {
    /// Return the selected row and the tier that decided it, or `None` for an empty table.
    pub fn select_row(
        &self,
        rows: &[TwoTierRow],
        request_blocks: u64,
    ) -> Option<(usize, TwoTierDecision)> {
        let parameters = &self.parameters;
        let min_load = rows.iter().map(|row| row.active_requests).min()?;
        let max_load = rows.iter().map(|row| row.active_requests).max()?;
        if max_load.saturating_sub(min_load) > parameters.balance_abs_threshold
            && (max_load as f64) > parameters.balance_rel_threshold * (min_load as f64)
        {
            return least_loaded(rows, 0..rows.len()).map(|row| (row, TwoTierDecision::Load));
        }

        let max_overlap = rows
            .iter()
            .map(|row| row.device_overlap_blocks)
            .max_by(f64::total_cmp)?;
        let cache_ratio = if request_blocks == 0 {
            0.0
        } else {
            max_overlap / request_blocks as f64
        };
        if cache_ratio > parameters.cache_threshold {
            return least_loaded(
                rows,
                rows.iter()
                    .enumerate()
                    .filter_map(|(index, row)| {
                        (row.device_overlap_blocks == max_overlap).then_some(index)
                    }),
            )
            .map(|row| (row, TwoTierDecision::Cache));
        }

        least_loaded(rows, 0..rows.len()).map(|row| (row, TwoTierDecision::LeastLoaded))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ten blocks, so five overlapping blocks sit exactly on the 0.5 threshold.
    const TEN_BLOCKS: u64 = 10;
    fn policy(parameters: TwoTierCostFnParameters) -> TwoTierCostFn {
        TwoTierCostFn::new(parameters)
    }

    /// Select among two workers given as `(device_blocks, active_requests)`; returns 0 for A and
    /// 1 for B.
    fn select_with(parameters: TwoTierCostFnParameters, rows: [(usize, usize); 2]) -> usize {
        let rows = rows.map(|(device, active)| TwoTierRow {
            device_overlap_blocks: device as f64,
            active_requests: active,
        });
        policy(parameters).select_row(&rows, TEN_BLOCKS).unwrap().0
    }

    fn select(rows: [(usize, usize); 2]) -> usize {
        select_with(TwoTierCostFnParameters::default(), rows)
    }

    const A: usize = 0;
    const B: usize = 1;

    #[test]
    fn cache_tier_outranks_a_less_loaded_worker() {
        // Six of ten blocks is 0.6, above the 0.5 threshold, so B wins despite carrying more load.
        assert_eq!(select([(0, 0), (6, 4)]), B);
    }

    #[test]
    fn cache_tier_threshold_is_strict() {
        // Five of ten blocks is exactly 0.5, so the comparison fails and load decides.
        assert_eq!(select([(0, 0), (5, 4)]), A);
    }

    #[test]
    fn parameters_override_the_upstream_defaults() {
        // Three of ten blocks is 0.3: below the 0.5 default, above a tuned 0.2 threshold.
        let rows = [(0, 0), (3, 4)];
        assert_eq!(select(rows), A);

        let tuned = TwoTierCostFnParameters {
            cache_threshold: 0.2,
            ..TwoTierCostFnParameters::default()
        };
        assert_eq!(select_with(tuned, rows), B);
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let cache = |v| {
            TwoTierCostFnParameters {
                cache_threshold: v,
                ..Default::default()
            }
            .validate()
        };
        let ratio = |v| {
            TwoTierCostFnParameters {
                balance_rel_threshold: v,
                ..Default::default()
            }
            .validate()
        };

        assert!(cache(-0.1).is_err() && cache(1.1).is_err() && cache(f64::NAN).is_err());
        assert!(ratio(0.9).is_err() && ratio(f64::NAN).is_err());
        assert!(TwoTierCostFnParameters::default().validate().is_ok());
    }

    #[test]
    fn load_tier_needs_both_gates() {
        // Spread 40 > 32 and 40 > 1.1 * 0: the load tier fires and ignores B's full overlap.
        assert_eq!(select([(0, 0), (10, 40)]), A);
        // Spread 64 > 32, but 704 is not > 1.1 * 640, so the cache tier still decides. This pair
        // straddles the ratio boundary: 705 would clear it and take the load tier.
        assert_eq!(select([(0, 640), (10, 704)]), B);
        assert_eq!(select([(0, 640), (10, 705)]), A);
    }

    #[test]
    fn parameters_parse_from_yaml_and_reject_unknown_keys() {
        let value: serde_yaml::Value = serde_yaml::from_str("cache_threshold: 0.3\n").unwrap();
        let parsed = TwoTierCostFnParameters::from_yaml(&value).unwrap();
        assert_eq!(parsed.cache_threshold, 0.3);
        assert_eq!(parsed.balance_abs_threshold, DEFAULT_BALANCE_ABS_THRESHOLD);

        let empty = serde_yaml::Value::Mapping(Default::default());
        assert_eq!(
            TwoTierCostFnParameters::from_yaml(&empty).unwrap(),
            TwoTierCostFnParameters::default()
        );

        let typo: serde_yaml::Value = serde_yaml::from_str("cache_affinity_threshold: 0.3").unwrap();
        let error = TwoTierCostFnParameters::from_yaml(&typo).unwrap_err();
        assert!(error.contains("cache_affinity_threshold"), "{error}");

        let bad: serde_yaml::Value = serde_yaml::from_str("cache_threshold: 2.0").unwrap();
        assert!(TwoTierCostFnParameters::from_yaml(&bad).is_err());
    }

    #[test]
    fn empty_table_selects_nothing() {
        assert!(
            policy(TwoTierCostFnParameters::default())
                .select_row(&[], TEN_BLOCKS)
                .is_none()
        );
    }

    #[test]
    fn decision_reports_the_tier() {
        let p = policy(TwoTierCostFnParameters::default());
        let row = |device: usize, active| TwoTierRow {
            device_overlap_blocks: device as f64,
            active_requests: active,
        };
        assert_eq!(
            p.select_row(&[row(0, 0), row(10, 40)], TEN_BLOCKS),
            Some((0, TwoTierDecision::Load))
        );
        assert_eq!(
            p.select_row(&[row(0, 0), row(6, 4)], TEN_BLOCKS),
            Some((1, TwoTierDecision::Cache))
        );
        assert_eq!(
            p.select_row(&[row(0, 3), row(2, 1)], TEN_BLOCKS),
            Some((1, TwoTierDecision::LeastLoaded))
        );
    }
}
