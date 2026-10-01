// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-set selection policy for models served by more than one WorkerSet.
//!
//! The default keeps the historical behavior: a weighted random pick proportional to
//! worker count. Each WorkerSet has its own KV router and indexer, so a random pick sends
//! consecutive turns of one conversation to different sets and discards their prefix cache.
//!
//! `DYN_WORKER_SET_SELECTION=affinity` instead maps a request's affinity key (derived from
//! the conversation's leading messages) to a set with weighted rendezvous hashing. Every
//! frontend makes the same choice for the same conversation without shared state, and the
//! long-run traffic split still follows the set weights. A decayed per-frontend share
//! tracker falls back to the weighted random pick whenever the hashed set already received
//! more than `(1 + slack)` times its weight share, so one hot key cannot pin a set.
//!
//! `DYN_WORKER_SET_WEIGHTS=suffix=weight,...` scales the per-worker weight of every set whose
//! namespace ends with `suffix` (for example `tp4=2,tp2=1` to weight sets by GPUs).

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
                    .filter(|(suffix, w)| !suffix.is_empty() && w.is_finite() && *w > 0.0);
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

/// Weighted rendezvous hashing: candidate `i` wins with probability `w_i / sum(w)` over
/// keys, and removing a candidate only moves the keys it owned.
pub fn rendezvous_pick(key: u64, candidates: &[(&str, f64)]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (idx, (name, weight)) in candidates.iter().enumerate() {
        if !(*weight > 0.0) {
            continue;
        }
        let h = xxh3_64_with_seed(name.as_bytes(), key);
        // Map to (0, 1]: never 0 so ln() stays finite.
        let u = ((h >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let score = -u.ln() / weight;
        if best.is_none_or(|(_, s)| score < s) {
            best = Some((idx, score));
        }
    }
    best.map(|(idx, _)| idx)
}

pub fn weighted_random_pick(weights: &[f64]) -> Option<usize> {
    let total: f64 = weights.iter().filter(|w| **w > 0.0).sum();
    if !(total > 0.0) {
        return None;
    }
    let mut pick = rand::rng().random_range(0.0..total);
    let mut last = None;
    for (idx, w) in weights.iter().enumerate() {
        if !(*w > 0.0) {
            continue;
        }
        if pick < *w {
            return Some(idx);
        }
        pick -= *w;
        last = Some(idx);
    }
    last
}

/// Exponentially decayed count of recent decisions per namespace.
#[derive(Debug, Default)]
pub struct ShareTracker {
    counts: Mutex<HashMap<String, f64>>,
}

impl ShareTracker {
    /// Choose among `candidates` for `key`, guarding against one set exceeding
    /// `(1 + slack)` times its weight share of recent decisions.
    pub fn choose(
        &self,
        key: u64,
        candidates: &[(&str, f64)],
        slack: f64,
    ) -> Option<(usize, SetChoiceReason)> {
        let preferred = rendezvous_pick(key, candidates)?;
        let total_weight: f64 = candidates.iter().map(|(_, w)| w.max(0.0)).sum();
        let mut counts = self.counts.lock();
        let observed: f64 = candidates
            .iter()
            .map(|(name, _)| counts.get(*name).copied().unwrap_or(0.0))
            .sum();
        let mut choice = (preferred, SetChoiceReason::Affinity);
        if observed >= SHARE_MIN_SAMPLES && total_weight > 0.0 {
            let (name, weight) = candidates[preferred];
            let share = counts.get(name).copied().unwrap_or(0.0) / observed;
            let cap = (weight / total_weight) * (1.0 + slack);
            if share > cap {
                let weights: Vec<f64> = candidates.iter().map(|(_, w)| *w).collect();
                if let Some(idx) = weighted_random_pick(&weights) {
                    choice = (idx, SetChoiceReason::ShareCapFallback);
                }
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
    fn share(&self, name: &str) -> f64 {
        let counts = self.counts.lock();
        let total: f64 = counts.values().sum();
        counts.get(name).copied().unwrap_or(0.0) / total.max(f64::MIN_POSITIVE)
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

    #[test]
    fn rendezvous_is_deterministic_and_weighted() {
        let cands = [("ns-tp4", 30.0), ("ns-tp2", 60.0)];
        let mut tp4 = 0usize;
        let n = 60_000u64;
        for key in 0..n {
            let a = rendezvous_pick(key, &cands).unwrap();
            assert_eq!(a, rendezvous_pick(key, &cands).unwrap());
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

    #[test]
    fn share_guard_caps_a_hot_key() {
        let tracker = ShareTracker::default();
        let cands = [("ns-tp4", 30.0), ("ns-tp2", 60.0)];
        let hot = (0..u64::MAX)
            .find(|k| rendezvous_pick(*k, &cands) == Some(0))
            .unwrap();
        let mut fallbacks = 0;
        for _ in 0..5000 {
            let (_, reason) = tracker.choose(hot, &cands, 0.25).unwrap();
            if reason == SetChoiceReason::ShareCapFallback {
                fallbacks += 1;
            }
        }
        assert!(fallbacks > 0);
        let share = tracker.share("ns-tp4");
        assert!(share < 1.0 / 3.0 * 1.25 + 0.03, "share {share}");
    }

    #[test]
    fn share_guard_keeps_affinity_for_balanced_keys() {
        let tracker = ShareTracker::default();
        let cands = [("ns-tp4", 30.0), ("ns-tp2", 60.0)];
        let mut fallbacks = 0;
        for key in 0..20_000u64 {
            let mixed = key.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let (idx, reason) = tracker.choose(mixed, &cands, 0.25).unwrap();
            if reason == SetChoiceReason::ShareCapFallback {
                fallbacks += 1;
            } else {
                assert_eq!(Some(idx), rendezvous_pick(mixed, &cands));
            }
        }
        assert!(fallbacks < 200, "fallbacks {fallbacks}");
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
