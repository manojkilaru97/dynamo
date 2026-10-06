// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Two-tier worker-selection cost function.
//!
//! Ported from upstream `lib/router-plugins/builtin/src/two_tier_cost_fn.rs`
//! (ai-dynamo/dynamo#14498, with the CPU-prefix extension from 3c438f32c). This branch has no
//! upstream plugin registry, so the policy is linked directly into the KV router and selected by
//! the same `worker_selection` router-policy YAML.
//!
//! Dynamo's built-in selector folds cache overlap and load into one additive cost. This policy
//! instead ranks on two tiers, taking the first that applies. For each eligible worker it reads
//! KV overlap and active-request count, then:
//!
//! 1. Load tier: if active-request spread exceeds `balance_abs_threshold` and the largest count
//!    exceeds `balance_rel_threshold` times the smallest, select the least-loaded worker.
//! 2. Cache tier: otherwise, if the largest *effective* KV overlap is strictly greater than
//!    `cache_threshold` of the request's matchable (complete) block count, select the
//!    least-loaded worker holding that maximum overlap. Effective overlap is device-resident blocks plus host-pinned (CPU offload)
//!    blocks scaled by `host_cache_weight`, so a worker holding the prefix in CPU can win the cache
//!    tier over one holding nothing, while still losing to an equal device-resident hit.
//! 3. Otherwise, select the least-loaded worker.
//!
//! Both load gates must hold to take step 1, so load displaces cache affinity only when the
//! imbalance is both large in absolute terms and disproportionate.
//!
//! The thresholds default to `experimental/sgl-router`'s `cache_aware_zmq` values, so an instance
//! with no `parameters` mapping reproduces it exactly.
//!
//! `host_cache_weight` defaults to [`KvRouterConfig::host_cache_hit_weight`], which
//! `DYN_ROUTER_HOST_CACHE_HIT_WEIGHT` sets (0.75 by default) and which the built-in selector
//! already applies to the same quantity. Set the weight to 0.0 to restore device-only ranking.
//!
//! Host overlap comes from `TierOverlapBlocks::host_pinned`, the host-pinned lower-tier indexer's
//! continuation beyond the device prefix. On this branch that indexer is fed by vLLM's
//! `BlockStored(medium=CPU)` events, including the token-less lazy-offload stores the ZMQ wire
//! layer completes from device identities (`zmq_wire`, `DYN_KV_ROUTER_FILL_LOWER_TIER`). Device
//! and host overlap are therefore disjoint prefix measurements, so they add rather than max.
//!
//! Deviation from upstream: the cache-tier ratio divides by the request's *complete* block count
//! (`isl / block_size`, or `(isl - 1) / block_size` for Eagle), the blocks the indexer can match,
//! instead of upstream's `isl.div_ceil(block_size)`. Overlap only ever counts complete blocks, so
//! with the ceiled denominator a full prefix hit on a prompt with a partial tail block can sit at
//! exactly the threshold and be rejected; with this branch's 4336-token blocks that disabled the
//! cache tier for every 4.3k-8.7k-token prompt. The ceiled count still sizes admission.
//!
//! Absent device overlap counts as 0.0, as upstream supplies it: the policy never substitutes the
//! weighted effective overlap for the device column.
//!
//! Ties between equally ranked workers resolve on candidate row order, which the host leaves
//! unspecified. This matches the ported implementation; note that Dynamo's built-in selector
//! instead samples uniformly among ties.
//!
//! [`KvRouterConfig::host_cache_hit_weight`]: super::config::KvRouterConfig::host_cache_hit_weight

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
    /// Fraction of the request's blocks that must be reusable on the best worker before the cache
    /// tier applies. Compared strictly.
    pub cache_threshold: f64,
    /// Minimum active-request spread before the load tier applies.
    pub balance_abs_threshold: usize,
    /// Minimum ratio of largest to smallest active-request count before the load tier applies.
    pub balance_rel_threshold: f64,
    /// Weight applied to host-pinned (CPU offload) overlap when ranking cache affinity.
    ///
    /// `None` inherits `KvRouterConfig::host_cache_hit_weight`, i.e. whatever
    /// `DYN_ROUTER_HOST_CACHE_HIT_WEIGHT` is set to.
    pub host_cache_weight: Option<f64>,
}

impl Default for TwoTierCostFnParameters {
    fn default() -> Self {
        Self {
            cache_threshold: DEFAULT_CACHE_THRESHOLD,
            balance_abs_threshold: DEFAULT_BALANCE_ABS_THRESHOLD,
            balance_rel_threshold: DEFAULT_BALANCE_REL_THRESHOLD,
            host_cache_weight: None,
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
        if let Some(weight) = self.host_cache_weight
            && (!weight.is_finite() || weight < 0.0)
        {
            return Err(
                "host_cache_weight must be a finite number greater than or equal to 0.0"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// A two-tier policy with its host-cache weight resolved once at construction: the instance
/// override when given, else the router config's `host_cache_hit_weight`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoTierCostFn {
    pub parameters: TwoTierCostFnParameters,
    pub host_cache_weight: f64,
}

impl TwoTierCostFn {
    pub fn new(parameters: TwoTierCostFnParameters, router_host_cache_hit_weight: f64) -> Self {
        Self {
            parameters,
            host_cache_weight: parameters
                .host_cache_weight
                .unwrap_or(router_host_cache_hit_weight),
        }
    }
}

/// One eligible worker's inputs to the two-tier decision.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TwoTierRow {
    pub device_overlap_blocks: f64,
    pub host_overlap_blocks: f64,
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

/// Blocks this worker can reuse, counting CPU-offloaded blocks at `host_cache_weight`.
fn effective_overlap(row: &TwoTierRow, host_cache_weight: f64) -> f64 {
    row.device_overlap_blocks + host_cache_weight * row.host_overlap_blocks
}

impl TwoTierCostFn {
    /// Return the selected row and the tier that decided it, or `None` for an empty table.
    ///
    /// `matchable_blocks` is the request's complete block count, the denominator of the cache
    /// ratio. This slice form is the reference for [`TwoTierAccumulator`], which the selector uses
    /// to decide in one allocation-free pass.
    pub fn select_row(
        &self,
        rows: &[TwoTierRow],
        matchable_blocks: u64,
    ) -> Option<(usize, TwoTierDecision)> {
        let parameters = &self.parameters;
        let min_load = rows.iter().map(|row| row.active_requests).min()?;
        let max_load = rows.iter().map(|row| row.active_requests).max()?;
        if self.load_tier_applies(min_load, max_load) {
            return least_loaded(rows, 0..rows.len()).map(|row| (row, TwoTierDecision::Load));
        }

        let max_overlap = rows
            .iter()
            .map(|row| effective_overlap(row, self.host_cache_weight))
            .max_by(f64::total_cmp)?;
        if self.cache_ratio(max_overlap, matchable_blocks) > parameters.cache_threshold {
            return least_loaded(
                rows,
                rows.iter().enumerate().filter_map(|(index, row)| {
                    // Recomputed identically to `max_overlap`, so the equality is exact, not a
                    // tolerance comparison on independently derived floats.
                    (effective_overlap(row, self.host_cache_weight) == max_overlap).then_some(index)
                }),
            )
            .map(|row| (row, TwoTierDecision::Cache));
        }

        least_loaded(rows, 0..rows.len()).map(|row| (row, TwoTierDecision::LeastLoaded))
    }

    fn cache_ratio(&self, max_overlap: f64, matchable_blocks: u64) -> f64 {
        if matchable_blocks == 0 {
            0.0
        } else {
            max_overlap / matchable_blocks as f64
        }
    }

    fn load_tier_applies(&self, min_load: usize, max_load: usize) -> bool {
        max_load.saturating_sub(min_load) > self.parameters.balance_abs_threshold
            && (max_load as f64) > self.parameters.balance_rel_threshold * (min_load as f64)
    }

    /// The overlap this policy ranks on for one row: device blocks plus host-pinned blocks at
    /// `host_cache_weight`.
    pub fn effective_overlap_blocks(&self, row: &TwoTierRow) -> f64 {
        effective_overlap(row, self.host_cache_weight)
    }

    /// Start a single-pass decision over candidates fed through [`TwoTierAccumulator::push`].
    pub fn accumulator<T: Copy>(&self) -> TwoTierAccumulator<'_, T> {
        TwoTierAccumulator {
            policy: self,
            min_load: usize::MAX,
            max_load: 0,
            least_loaded: None,
            cache_best: None,
        }
    }
}

/// Allocation-free, single-pass form of [`TwoTierCostFn::select_row`].
///
/// Candidates are pushed in row order; the result is identical to `select_row` over the same rows,
/// including first-row tie resolution: the least-loaded row is the first with the minimum count,
/// and the cache winner is the first least-loaded row among those at the maximum effective overlap.
pub struct TwoTierAccumulator<'a, T> {
    policy: &'a TwoTierCostFn,
    min_load: usize,
    max_load: usize,
    least_loaded: Option<(T, TwoTierRow)>,
    cache_best: Option<(T, TwoTierRow, f64)>,
}

impl<T: Copy> TwoTierAccumulator<'_, T> {
    pub fn push(&mut self, item: T, row: TwoTierRow) {
        let load = row.active_requests;
        self.min_load = self.min_load.min(load);
        self.max_load = self.max_load.max(load);
        if self
            .least_loaded
            .is_none_or(|(_, best)| load < best.active_requests)
        {
            self.least_loaded = Some((item, row));
        }
        let overlap = effective_overlap(&row, self.policy.host_cache_weight);
        let replace = match self.cache_best {
            None => true,
            Some((_, best, best_overlap)) => {
                overlap > best_overlap || (overlap == best_overlap && load < best.active_requests)
            }
        };
        if replace {
            self.cache_best = Some((item, row, overlap));
        }
    }

    /// Return the selected candidate, its row and the deciding tier, or `None` if nothing was
    /// pushed.
    pub fn finish(self, matchable_blocks: u64) -> Option<(T, TwoTierRow, TwoTierDecision)> {
        let (least_item, least_row) = self.least_loaded?;
        if self.policy.load_tier_applies(self.min_load, self.max_load) {
            return Some((least_item, least_row, TwoTierDecision::Load));
        }
        if let Some((item, row, overlap)) = self.cache_best
            && self.policy.cache_ratio(overlap, matchable_blocks)
                > self.policy.parameters.cache_threshold
        {
            return Some((item, row, TwoTierDecision::Cache));
        }
        Some((least_item, least_row, TwoTierDecision::LeastLoaded))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ten blocks, so five overlapping blocks sit exactly on the 0.5 threshold.
    const TEN_BLOCKS: u64 = 10;
    /// Upstream `KvRouterConfig::default().host_cache_hit_weight`.
    const DEFAULT_HOST_WEIGHT: f64 = 0.75;

    fn policy(parameters: TwoTierCostFnParameters) -> TwoTierCostFn {
        TwoTierCostFn::new(parameters, DEFAULT_HOST_WEIGHT)
    }

    /// Select among two workers given as `(device_blocks, host_blocks, active_requests)`;
    /// returns 0 for A and 1 for B.
    fn select_tiers(
        parameters: TwoTierCostFnParameters,
        rows: [(usize, usize, usize); 2],
    ) -> usize {
        let rows = rows.map(|(device, host, active)| TwoTierRow {
            device_overlap_blocks: device as f64,
            host_overlap_blocks: host as f64,
            active_requests: active,
        });
        policy(parameters).select_row(&rows, TEN_BLOCKS).unwrap().0
    }

    fn select(rows: [(usize, usize); 2]) -> usize {
        select_tiers(
            TwoTierCostFnParameters::default(),
            rows.map(|(device, active)| (device, 0, active)),
        )
    }

    const A: usize = 0;
    const B: usize = 1;

    #[test]
    fn default_host_weight_matches_router_default() {
        assert_eq!(
            super::super::config::KvRouterConfig::default().host_cache_hit_weight,
            DEFAULT_HOST_WEIGHT
        );
    }

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
        let rows = [(0, 0, 0), (3, 0, 4)];
        assert_eq!(select_tiers(TwoTierCostFnParameters::default(), rows), A);

        let tuned = TwoTierCostFnParameters {
            cache_threshold: 0.2,
            ..TwoTierCostFnParameters::default()
        };
        assert_eq!(select_tiers(tuned, rows), B);
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
    fn host_overlap_can_win_the_cache_tier() {
        // B holds nothing on device but eight of ten blocks in CPU offload. At the inherited
        // weight of 0.75 that is an effective 6.0 blocks, a ratio of 0.6 above the 0.5 threshold,
        // so B wins despite carrying more load.
        assert_eq!(
            select_tiers(TwoTierCostFnParameters::default(), [(0, 0, 0), (0, 8, 4)]),
            B
        );
    }

    #[test]
    fn host_cache_weight_zero_restores_device_only_ranking() {
        let device_only = TwoTierCostFnParameters {
            host_cache_weight: Some(0.0),
            ..TwoTierCostFnParameters::default()
        };
        assert_eq!(select_tiers(device_only, [(0, 0, 0), (0, 8, 4)]), A);
    }

    #[test]
    fn device_blocks_outrank_the_same_count_of_host_blocks() {
        // A's six device blocks score 6.0 against B's six host blocks at 4.5, so A wins even
        // though B is idle and A carries four requests.
        assert_eq!(
            select_tiers(TwoTierCostFnParameters::default(), [(6, 0, 4), (0, 6, 0)]),
            A
        );
    }

    #[test]
    fn device_and_host_overlap_add() {
        // A: 3 device + 4 host * 0.75 = 6.0 (ratio 0.6) beats B's 5 device blocks even though
        // neither tier alone clears the threshold on A.
        assert_eq!(
            select_tiers(TwoTierCostFnParameters::default(), [(3, 4, 4), (5, 0, 0)]),
            A
        );
    }

    #[test]
    fn rejects_negative_host_cache_weight() {
        let weight = |v| {
            TwoTierCostFnParameters {
                host_cache_weight: Some(v),
                ..Default::default()
            }
            .validate()
        };
        assert!(weight(-0.1).is_err());
        assert!(weight(f64::NAN).is_err());
        assert!(weight(0.0).is_ok());
        assert!(weight(1.0).is_ok());
    }

    #[test]
    fn instance_override_beats_router_host_weight() {
        let parameters = TwoTierCostFnParameters {
            host_cache_weight: Some(0.25),
            ..Default::default()
        };
        assert_eq!(TwoTierCostFn::new(parameters, 0.75).host_cache_weight, 0.25);
        assert_eq!(
            TwoTierCostFn::new(TwoTierCostFnParameters::default(), 0.5).host_cache_weight,
            0.5
        );
    }

    #[test]
    fn parameters_parse_from_yaml_and_reject_unknown_keys() {
        let value: serde_yaml::Value =
            serde_yaml::from_str("cache_threshold: 0.3\nhost_cache_weight: 0.5\n").unwrap();
        let parsed = TwoTierCostFnParameters::from_yaml(&value).unwrap();
        assert_eq!(parsed.cache_threshold, 0.3);
        assert_eq!(parsed.host_cache_weight, Some(0.5));
        assert_eq!(parsed.balance_abs_threshold, DEFAULT_BALANCE_ABS_THRESHOLD);

        let empty = serde_yaml::Value::Mapping(Default::default());
        assert_eq!(
            TwoTierCostFnParameters::from_yaml(&empty).unwrap(),
            TwoTierCostFnParameters::default()
        );

        let typo: serde_yaml::Value =
            serde_yaml::from_str("cache_affinity_threshold: 0.3").unwrap();
        let error = TwoTierCostFnParameters::from_yaml(&typo).unwrap_err();
        assert!(error.contains("cache_affinity_threshold"), "{error}");

        let bad: serde_yaml::Value = serde_yaml::from_str("cache_threshold: 2.0").unwrap();
        assert!(TwoTierCostFnParameters::from_yaml(&bad).is_err());
    }

    #[test]
    fn accumulator_matches_select_row() {
        // Deterministic LCG so the comparison covers ties, both tiers and the load gate without a
        // test-only RNG dependency.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = |bound: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % bound
        };
        for weight in [0.0, 0.75, 1.0] {
            let parameters = TwoTierCostFnParameters {
                host_cache_weight: Some(weight),
                ..Default::default()
            };
            let p = policy(parameters);
            for _ in 0..2000 {
                let len = 1 + next(8) as usize;
                let spread = [4, 40, 800][next(3) as usize];
                let rows: Vec<TwoTierRow> = (0..len)
                    .map(|_| TwoTierRow {
                        device_overlap_blocks: next(6) as f64,
                        host_overlap_blocks: next(6) as f64,
                        active_requests: next(spread) as usize,
                    })
                    .collect();
                let blocks = next(12);
                let expected = p.select_row(&rows, blocks);
                let mut acc = p.accumulator::<usize>();
                for (index, row) in rows.iter().enumerate() {
                    acc.push(index, *row);
                }
                let actual = acc.finish(blocks).map(|(index, row, decision)| {
                    assert_eq!(row, rows[index]);
                    (index, decision)
                });
                assert_eq!(actual, expected, "rows={rows:?} blocks={blocks}");
            }
        }
        assert!(
            policy(TwoTierCostFnParameters::default())
                .accumulator::<usize>()
                .finish(10)
                .is_none()
        );
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
            host_overlap_blocks: 0.0,
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
