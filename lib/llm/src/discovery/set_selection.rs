// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-set selection policy for models served by more than one WorkerSet.
//!
//! The default keeps the historical behavior: a weighted random pick proportional to
//! worker count. Each WorkerSet has its own KV router and indexer, so a random pick sends
//! consecutive turns of one conversation to different sets and discards their prefix cache.
//!
//! `DYN_WORKER_SET_SELECTION=affinity` instead maps a request's affinity key (derived from
//! the conversation's leading messages) to a set with weighted rendezvous hashing over the sets'
//! unique storage keys. Every frontend makes the same choice for the same conversation
//! without shared state, and the long-run traffic split still follows the set weights.
//!
//! A decayed per-frontend share tracker keeps every set inside
//! `fair ± slack·min(fair, 1 − fair)` of recent affinity decisions. When the hashed set is
//! above its band (or another set has fallen below its band), the request spills to the
//! next-ranked rendezvous set that is below its fair share. The spill target depends only on
//! the key and the candidate sets, so frontends agree and a conversation's spilled turns land
//! on one set.
//!
//! `DYN_WORKER_SET_WEIGHTS=suffix=weight,...` scales the per-worker weight of every set whose
//! namespace ends with `suffix` (for example `tp4=2,tp2=1` to weight sets by GPUs). Entries
//! must lie in `(0, 1e6]`; anything else is ignored with a warning.

use std::collections::HashMap;
use std::io;
use std::sync::OnceLock;

use parking_lot::Mutex;
use rand::Rng;
use serde::Serialize;
use xxhash_rust::xxh3::{Xxh3, xxh3_64_with_seed};

const MODE_ENV: &str = "DYN_WORKER_SET_SELECTION";
const SLACK_ENV: &str = "DYN_WORKER_SET_AFFINITY_SLACK";
const WEIGHTS_ENV: &str = "DYN_WORKER_SET_WEIGHTS";
const DEFAULT_SLACK: f64 = 0.25;
/// Largest accepted per-worker weight. Keeps `worker_count × weight` and the sums over sets
/// finite, so weighted picks never see an infinite range.
pub const MAX_PER_WORKER_WEIGHT: f64 = 1e6;
/// Decisions over which the share tracker averages (exponential decay horizon).
const SHARE_WINDOW: f64 = 1000.0;
/// Decayed decisions required before the share guard may override the hashed set.
const SHARE_MIN_SAMPLES: f64 = 50.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetSelectionMode {
    Random,
    Affinity,
}

#[derive(Debug, Clone)]
pub struct SetSelectionConfig {
    pub mode: SetSelectionMode,
    pub slack: f64,
    pub weights: Vec<(String, f64)>,
}

impl Default for SetSelectionConfig {
    fn default() -> Self {
        Self {
            mode: SetSelectionMode::Random,
            slack: DEFAULT_SLACK,
            weights: Vec::new(),
        }
    }
}

impl SetSelectionConfig {
    pub fn global() -> &'static SetSelectionConfig {
        static CONFIG: OnceLock<SetSelectionConfig> = OnceLock::new();
        CONFIG.get_or_init(|| {
            let config = Self::parse(
                std::env::var(MODE_ENV).ok().as_deref(),
                std::env::var(SLACK_ENV).ok().as_deref(),
                std::env::var(WEIGHTS_ENV).ok().as_deref(),
            );
            if config.mode != SetSelectionMode::Random || !config.weights.is_empty() {
                tracing::info!(
                    mode = ?config.mode,
                    slack = config.slack,
                    weights = ?config.weights,
                    "Worker-set selection configured"
                );
            }
            config
        })
    }

    pub fn parse(mode: Option<&str>, slack: Option<&str>, weights: Option<&str>) -> Self {
        let mut config = Self::default();
        match mode.map(str::trim).filter(|m| !m.is_empty()) {
            None | Some("random") => {}
            Some("affinity") => config.mode = SetSelectionMode::Affinity,
            Some(other) => {
                tracing::warn!(value = other, "Unknown {MODE_ENV}; using random");
            }
        }
        if let Some(raw) = slack.map(str::trim).filter(|s| !s.is_empty()) {
            match raw.parse::<f64>() {
                Ok(v) if v.is_finite() && v >= 0.0 => config.slack = v,
                _ => tracing::warn!(value = raw, "Invalid {SLACK_ENV}; using {DEFAULT_SLACK}"),
            }
        }
        if let Some(raw) = weights {
            for item in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let parsed = item
                    .split_once('=')
                    .and_then(|(suffix, w)| Some((suffix.trim(), w.trim().parse::<f64>().ok()?)))
                    .filter(|(suffix, w)| {
                        !suffix.is_empty()
                            && w.is_finite()
                            && *w > 0.0
                            && *w <= MAX_PER_WORKER_WEIGHT
                    });
                match parsed {
                    Some((suffix, w)) => config.weights.push((suffix.to_string(), w)),
                    None => tracing::warn!(value = item, "Invalid {WEIGHTS_ENV} entry; ignored"),
                }
            }
        }
        config
    }

    /// Selection weight of a set: worker count times the matching per-worker weight.
    pub fn set_weight(&self, namespace: &str, worker_count: usize) -> f64 {
        let per_worker = self
            .weights
            .iter()
            .filter(|(suffix, _)| namespace.ends_with(suffix.as_str()))
            .max_by_key(|(suffix, _)| suffix.len())
            .map_or(1.0, |(_, w)| *w);
        worker_count as f64 * per_worker
    }
}

/// Why a set was chosen (metric label).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetChoiceReason {
    Affinity,
    ShareCapFallback,
    Random,
}

impl SetChoiceReason {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Affinity => "affinity",
            Self::ShareCapFallback => "share_cap_fallback",
            Self::Random => "random",
        }
    }
}

/// Rendezvous score of one candidate: lower wins. `None` for non-positive weights.
fn rendezvous_score(key: u64, name: &str, weight: f64) -> Option<f64> {
    if !(weight > 0.0) {
        return None;
    }
    let h = xxh3_64_with_seed(name.as_bytes(), key);
    // Map to (0, 1]: never 0 so ln() stays finite.
    let u = ((h >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    Some(-u.ln() / weight)
}

/// Candidates with a positive weight, best rendezvous score first. Ties (only possible
/// with equal names or infinite weights) break by name, so the order never depends on
/// the order of `candidates`.
pub fn rendezvous_ranking(key: u64, candidates: &[(&str, f64)]) -> Vec<usize> {
    let mut scored: Vec<(f64, &str, usize)> = candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, (name, weight))| {
            rendezvous_score(key, name, *weight).map(|score| (score, *name, idx))
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored.into_iter().map(|(_, _, idx)| idx).collect()
}

/// Weighted rendezvous hashing: candidate `i` wins with probability `w_i / sum(w)` over
/// keys, and removing a candidate only moves the keys it owned.
pub fn rendezvous_pick(key: u64, candidates: &[(&str, f64)]) -> Option<usize> {
    let mut best: Option<(usize, f64, &str)> = None;
    for (idx, (name, weight)) in candidates.iter().enumerate() {
        let Some(score) = rendezvous_score(key, name, *weight) else {
            continue;
        };
        if best.is_none_or(|(_, s, n)| score < s || (score == s && *name < n)) {
            best = Some((idx, score, name));
        }
    }
    best.map(|(idx, _, _)| idx)
}

/// Pick an index with probability proportional to its weight. Non-positive and non-finite
/// weights are skipped; returns `None` when no weight is usable.
pub fn weighted_random_pick(weights: &[f64]) -> Option<usize> {
    let usable = |w: f64| w > 0.0 && w.is_finite();
    let total: f64 = weights.iter().copied().filter(|w| usable(*w)).sum();
    if !(total > 0.0) || !total.is_finite() {
        return None;
    }
    let mut pick = rand::rng().random_range(0.0..total);
    let mut last = None;
    for (idx, w) in weights.iter().copied().enumerate() {
        if !usable(w) {
            continue;
        }
        if pick < w {
            return Some(idx);
        }
        pick -= w;
        last = Some(idx);
    }
    last
}

/// Exponentially decayed count of recent affinity decisions per WorkerSet key.
#[derive(Debug, Default)]
pub struct ShareTracker {
    counts: Mutex<HashMap<String, f64>>,
}

/// Half-width of a set's allowed share band around its fair share.
fn share_band(fair: f64, slack: f64) -> f64 {
    slack * fair.min(1.0 - fair)
}

impl ShareTracker {
    /// Choose among `candidates` (unique set key, weight) for `key`.
    ///
    /// The rendezvous winner is chosen unless it is above `fair + band` of recent
    /// decisions, or it is above its fair share while another set is below `fair − band`.
    /// Then the request spills to the best-ranked other set that is below its fair share.
    /// The result is labelled [`SetChoiceReason::ShareCapFallback`] only when the chosen
    /// set differs from the rendezvous winner.
    pub fn choose(
        &self,
        key: u64,
        candidates: &[(&str, f64)],
        slack: f64,
    ) -> Option<(usize, SetChoiceReason)> {
        let ranking = rendezvous_ranking(key, candidates);
        let preferred = *ranking.first()?;
        let total_weight: f64 = ranking.iter().map(|&i| candidates[i].1).sum();
        let mut counts = self.counts.lock();
        let observed: f64 = ranking
            .iter()
            .map(|&i| counts.get(candidates[i].0).copied().unwrap_or(0.0))
            .sum();
        let mut choice = (preferred, SetChoiceReason::Affinity);
        if observed >= SHARE_MIN_SAMPLES && total_weight > 0.0 && total_weight.is_finite() {
            let share = |i: usize| counts.get(candidates[i].0).copied().unwrap_or(0.0) / observed;
            let fair = |i: usize| candidates[i].1 / total_weight;
            let preferred_share = share(preferred);
            let preferred_fair = fair(preferred);
            let over_cap = preferred_share > preferred_fair + share_band(preferred_fair, slack);
            let other_starved = preferred_share > preferred_fair
                && ranking[1..]
                    .iter()
                    .any(|&i| share(i) < fair(i) - share_band(fair(i), slack));
            if (over_cap || other_starved)
                && let Some(&spill) = ranking[1..].iter().find(|&&i| share(i) < fair(i))
            {
                choice = (spill, SetChoiceReason::ShareCapFallback);
            }
        }
        let decay = 1.0 - 1.0 / SHARE_WINDOW;
        for v in counts.values_mut() {
            *v *= decay;
        }
        counts.retain(|_, v| *v > 1e-3);
        *counts
            .entry(candidates[choice.0].0.to_string())
            .or_insert(0.0) += 1.0;
        Some(choice)
    }

    #[cfg(test)]
    pub(crate) fn share(&self, name: &str) -> f64 {
        let counts = self.counts.lock();
        let total: f64 = counts.values().sum();
        counts.get(name).copied().unwrap_or(0.0) / total.max(f64::MIN_POSITIVE)
    }

    /// Decayed number of recorded decisions (test hook).
    #[cfg(test)]
    pub(crate) fn observed_total(&self) -> f64 {
        self.counts.lock().values().sum()
    }
}

struct HashWriter(Xxh3);

impl io::Write for HashWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

const AFFINITY_SEED: u64 = 0x5e75_e1ec_7a1f_f1a7;

/// Affinity key for a chat conversation: a hash of every message up to and including
/// the first user message. Later turns of the same conversation repeat that prefix, so
/// they map to the same key. Returns `None` when there is no user message.
pub fn chat_affinity_key<M: Serialize>(
    messages: &[M],
    is_user: impl Fn(&M) -> bool,
) -> Option<u64> {
    let end = messages.iter().position(is_user)?;
    let mut writer = HashWriter(Xxh3::with_seed(AFFINITY_SEED));
    for message in &messages[..=end] {
        serde_json::to_writer(&mut writer, message).ok()?;
        io::Write::write_all(&mut writer, b"\x1e").ok()?;
    }
    Some(writer.0.digest())
}

/// Process-global switch read by the HTTP layer: compute keys only when they are used.
pub fn affinity_enabled() -> bool {
    SetSelectionConfig::global().mode == SetSelectionMode::Affinity
}

#[cfg(test)]
mod tests {
    use super::*;

    const TP4: &str = "ns-tp4";
    const TP2: &str = "ns-tp2";
    const SETS: [(&str, f64); 2] = [(TP4, 30.0), (TP2, 60.0)];

    #[test]
    fn parse_defaults_to_random() {
        let c = SetSelectionConfig::parse(None, None, None);
        assert_eq!(c.mode, SetSelectionMode::Random);
        assert_eq!(c.slack, DEFAULT_SLACK);
        assert!(c.weights.is_empty());
        let c = SetSelectionConfig::parse(Some("bogus"), Some("-1"), Some("tp4=x,=2,tp2=0"));
        assert_eq!(c.mode, SetSelectionMode::Random);
        assert_eq!(c.slack, DEFAULT_SLACK);
        assert!(c.weights.is_empty());
    }

    #[test]
    fn parse_affinity_and_weights() {
        let c = SetSelectionConfig::parse(Some("affinity"), Some("0.5"), Some("tp4=2, tp2=1"));
        assert_eq!(c.mode, SetSelectionMode::Affinity);
        assert_eq!(c.slack, 0.5);
        assert_eq!(c.set_weight("dynamo-ns-tp4", 30), 60.0);
        assert_eq!(c.set_weight("dynamo-ns-tp2", 60), 60.0);
        assert_eq!(c.set_weight("other", 7), 7.0);
    }

    /// A3: an oversized per-worker weight used to overflow `worker_count × weight` to
    /// infinity, and `random_range(0.0..inf)` panics on the request path.
    #[test]
    fn parse_rejects_weights_that_could_overflow() {
        let c = SetSelectionConfig::parse(None, None, Some("tp4=1e308,tp2=1e6,x=1e6001"));
        assert_eq!(c.weights, vec![("tp2".to_string(), MAX_PER_WORKER_WEIGHT)]);
        assert_eq!(c.set_weight("dynamo-tp4", 30), 30.0);
        let weights = [
            c.set_weight("dynamo-tp4", 30),
            c.set_weight("dynamo-tp2", 60),
        ];
        assert!(weights.iter().all(|w| w.is_finite()));
        for _ in 0..100 {
            assert!(weighted_random_pick(&weights).is_some());
        }
    }

    #[test]
    fn weighted_random_pick_survives_non_finite_weights() {
        for _ in 0..1000 {
            assert_eq!(weighted_random_pick(&[f64::INFINITY, 1.0]), Some(1));
            assert_eq!(weighted_random_pick(&[f64::NAN, 0.0, 2.0]), Some(2));
            assert_eq!(weighted_random_pick(&[1e308, 1e308]), None);
        }
        assert_eq!(weighted_random_pick(&[f64::INFINITY]), None);
        assert_eq!(weighted_random_pick(&[]), None);
    }

    #[test]
    fn rendezvous_is_deterministic_and_weighted() {
        let mut tp4 = 0usize;
        let n = 60_000u64;
        for key in 0..n {
            let a = rendezvous_pick(key, &SETS).unwrap();
            assert_eq!(a, rendezvous_pick(key, &SETS).unwrap());
            assert_eq!(a, rendezvous_ranking(key, &SETS)[0]);
            if a == 0 {
                tp4 += 1;
            }
        }
        let share = tp4 as f64 / n as f64;
        assert!((share - 1.0 / 3.0).abs() < 0.01, "share {share}");
    }

    #[test]
    fn rendezvous_removal_only_moves_owned_keys() {
        let three = [("a", 1.0), ("b", 1.0), ("c", 1.0)];
        let two = [("a", 1.0), ("b", 1.0)];
        for key in 0..10_000u64 {
            let before = three[rendezvous_pick(key, &three).unwrap()].0;
            let after = two[rendezvous_pick(key, &two).unwrap()].0;
            if before != "c" {
                assert_eq!(before, after);
            }
        }
    }

    /// D1: candidates are unique set keys; the winner must not depend on candidate order.
    #[test]
    fn rendezvous_winner_is_independent_of_candidate_order() {
        let fwd = [("ns:set-a", 1.0), ("ns:set-b", 1.0)];
        let rev = [("ns:set-b", 1.0), ("ns:set-a", 1.0)];
        let mut a_wins = 0;
        for key in 0..4000u64 {
            let x = fwd[rendezvous_pick(key, &fwd).unwrap()].0;
            assert_eq!(x, rev[rendezvous_pick(key, &rev).unwrap()].0);
            let rf: Vec<&str> = rendezvous_ranking(key, &fwd)
                .into_iter()
                .map(|i| fwd[i].0)
                .collect();
            let rr: Vec<&str> = rendezvous_ranking(key, &rev)
                .into_iter()
                .map(|i| rev[i].0)
                .collect();
            assert_eq!(rf, rr);
            a_wins += usize::from(x == "ns:set-a");
        }
        assert!((1600..2400).contains(&a_wins), "a_wins {a_wins}");
        // Exact ties (identical names) still resolve by name, not position.
        let tie = [("same", f64::INFINITY), ("other", f64::INFINITY)];
        assert_eq!(rendezvous_pick(1, &tie), Some(1));
        let tie_rev = [("other", f64::INFINITY), ("same", f64::INFINITY)];
        assert_eq!(rendezvous_pick(1, &tie_rev), Some(0));
    }

    fn hot_key_preferring(name: &str) -> u64 {
        (0..u64::MAX)
            .find(|k| SETS[rendezvous_pick(*k, &SETS).unwrap()].0 == name)
            .unwrap()
    }

    /// Run `n` decisions of one hot key and return (fallbacks, spill targets seen).
    fn run_hot_key(tracker: &ShareTracker, key: u64, n: usize) -> usize {
        let preferred = rendezvous_pick(key, &SETS).unwrap();
        let spill = rendezvous_ranking(key, &SETS)[1];
        let mut fallbacks = 0;
        for _ in 0..n {
            let (idx, reason) = tracker.choose(key, &SETS, DEFAULT_SLACK).unwrap();
            match reason {
                SetChoiceReason::ShareCapFallback => {
                    assert_ne!(idx, preferred, "a fallback must change the set");
                    assert_eq!(idx, spill, "spill target must be deterministic");
                    fallbacks += 1;
                }
                SetChoiceReason::Affinity => assert_eq!(idx, preferred),
                SetChoiceReason::Random => panic!("affinity choice labelled random"),
            }
        }
        fallbacks
    }

    /// A1: a hot key that prefers the larger set used to squeeze the smaller set to half
    /// its fair share (the cap was one-sided and the redraw included the preferred set).
    #[test]
    fn share_guard_protects_smaller_set() {
        let tracker = ShareTracker::default();
        let fallbacks = run_hot_key(&tracker, hot_key_preferring(TP2), 5000);
        assert!(fallbacks > 0);
        let floor = (1.0 / 3.0) * (1.0 - DEFAULT_SLACK) - 0.03;
        let tp4 = tracker.share(TP4);
        assert!(tp4 >= floor, "tp4 share {tp4} below {floor}");
    }

    #[test]
    fn share_guard_caps_a_hot_key_preferring_smaller_set() {
        let tracker = ShareTracker::default();
        let fallbacks = run_hot_key(&tracker, hot_key_preferring(TP4), 5000);
        assert!(fallbacks > 0);
        let tp4 = tracker.share(TP4);
        let cap = 1.0 / 3.0 + share_band(1.0 / 3.0, DEFAULT_SLACK) + 0.03;
        assert!(tp4 <= cap, "tp4 share {tp4} above {cap}");
        let floor = (2.0 / 3.0) * (1.0 - DEFAULT_SLACK) - 0.03;
        let tp2 = tracker.share(TP2);
        assert!(tp2 >= floor, "tp2 share {tp2} below {floor}");
    }

    #[test]
    fn share_guard_keeps_affinity_for_balanced_keys() {
        let tracker = ShareTracker::default();
        let mut fallbacks = 0;
        for key in 0..20_000u64 {
            let mixed = key.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let (idx, reason) = tracker.choose(mixed, &SETS, DEFAULT_SLACK).unwrap();
            if reason == SetChoiceReason::ShareCapFallback {
                assert_ne!(Some(idx), rendezvous_pick(mixed, &SETS));
                fallbacks += 1;
            } else {
                assert_eq!(Some(idx), rendezvous_pick(mixed, &SETS));
            }
        }
        assert!(fallbacks < 200, "fallbacks {fallbacks}");
    }

    /// Spilling with three sets goes to the best-ranked set below its fair share.
    #[test]
    fn share_guard_spills_to_an_underserved_set() {
        let sets = [("a", 1.0), ("b", 1.0), ("c", 1.0)];
        let tracker = ShareTracker::default();
        let key = 42;
        let ranking = rendezvous_ranking(key, &sets);
        for _ in 0..5000 {
            let (idx, reason) = tracker.choose(key, &sets, DEFAULT_SLACK).unwrap();
            if reason == SetChoiceReason::ShareCapFallback {
                assert_ne!(idx, ranking[0]);
            }
        }
        let cap = 1.0 / 3.0 + share_band(1.0 / 3.0, DEFAULT_SLACK) + 0.03;
        assert!(tracker.share(sets[ranking[0]].0) <= cap);
    }

    #[derive(Serialize)]
    struct Msg {
        role: &'static str,
        content: String,
    }

    fn msg(role: &'static str, content: &str) -> Msg {
        Msg {
            role,
            content: content.to_string(),
        }
    }

    #[test]
    fn chat_key_is_stable_across_turns() {
        let is_user = |m: &Msg| m.role == "user";
        let turn1 = vec![msg("system", "sys"), msg("user", "task A")];
        let turn2 = vec![
            msg("system", "sys"),
            msg("user", "task A"),
            msg("assistant", "ok"),
            msg("user", "more"),
        ];
        let other = vec![msg("system", "sys"), msg("user", "task B")];
        let k1 = chat_affinity_key(&turn1, is_user).unwrap();
        assert_eq!(k1, chat_affinity_key(&turn2, is_user).unwrap());
        assert_ne!(k1, chat_affinity_key(&other, is_user).unwrap());
        assert!(chat_affinity_key(&[msg("system", "sys")], is_user).is_none());
    }
}
