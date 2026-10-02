// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-set selection policy for models served by more than one WorkerSet.
//!
//! The default keeps the historical behavior: a weighted random pick proportional to
//! worker count. Each WorkerSet has its own KV router and indexer, so a random pick sends
//! consecutive turns of one conversation to different sets and discards their prefix cache.
//!
//! `DYN_WORKER_SET_SELECTION=affinity` instead maps a request's affinity key (see
//! [`chat_request_affinity_key`]) to a set with weighted rendezvous hashing over the sets'
//! unique storage keys. Every frontend makes the same choice for the same conversation
//! without shared state, and the long-run split follows the set weights.
//!
//! Each frontend also keeps decayed, size-weighted load statistics (charged by
//! [`request_charge`], a byte-count proxy for prompt size) to keep every set within
//! `fair ± slack·min(fair, 1 − fair)` of recent affinity load:
//!
//! 1. **Sticky spill.** When the load whose rendezvous winner is set `P` exceeds `P`'s fair
//!    share, the frontend derives a spill fraction `p` (zero until demand exceeds the middle
//!    of the upper band by more than two standard deviations of its sampling noise, at most
//!    `slack`) and moves exactly the keys whose [`spill_point`] is below `p`
//!    to their second rendezvous choice. Whether a key spills, and where to, is a function
//!    of the key, the candidate set keys and `p`; `p` depends only on the aggregate load
//!    mix, so frontends with similar traffic agree and a conversation's turns stay on one
//!    set while `p` is stable.
//! 2. **Heavy-key overflow.** Load that one key concentrates cannot be spread by moving
//!    whole keys. Each frontend keeps a small decayed table of its heaviest keys; a key that
//!    carries at least 1% of recent load is redirected to the best-ranked set below its fair
//!    share while its chosen set is above its band. A spilled heavy key goes back to its
//!    rendezvous winner (`share_cap_revert`); a heavy key native to the over-band set moves
//!    (`share_cap_overflow`) only when the overshoot exceeds the load other sets' keys
//!    spilled into it. No other key is ever redirected, so ordinary conversations change
//!    set only through the sticky spill. This is the only frontend-local decision.
//!
//! Charges are clamped to 8× the decayed mean charge, and the guard stays idle until it
//! has seen about 200 decisions, so one huge request or a freshly started frontend cannot
//! swing the statistics.
//!
//! "Fair" is the decayed average of each set's weight share over the same window, so a
//! worker-count change shifts the targets as gradually as the observed load and only the
//! keys rendezvous hashing itself moves change sets.
//!
//! `DYN_WORKER_SET_WEIGHTS=suffix=weight,...` scales the per-worker weight of every set whose
//! namespace ends with `suffix` (for example `tp4=2,tp2=1` to weight sets by GPUs). Entries
//! must lie in `[1e-6, 1e6]`; anything else is ignored with a warning.
//! `DYN_WORKER_SET_AFFINITY_SLACK` (default 0.25) is clamped to at least 0.05.

use std::collections::HashMap;
use std::io;
use std::sync::OnceLock;

use parking_lot::Mutex;
use rand::Rng;
use xxhash_rust::xxh3::{Xxh3, xxh3_64_with_seed};

use async_openai::types::chat::ChatCompletionRequestDeveloperMessageContentPart;
use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestAssistantMessageContentPart,
    ChatCompletionRequestDeveloperMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestSystemMessageContentPart,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestToolMessageContentPart,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ReasoningContent,
};

use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

const MODE_ENV: &str = "DYN_WORKER_SET_SELECTION";
const SLACK_ENV: &str = "DYN_WORKER_SET_AFFINITY_SLACK";
const WEIGHTS_ENV: &str = "DYN_WORKER_SET_WEIGHTS";
const DEFAULT_SLACK: f64 = 0.25;
/// Smallest accepted slack. A zero band spills about half of all keys whenever load is
/// merely at fair share, which defeats affinity.
pub const MIN_SLACK: f64 = 0.05;
/// Largest accepted per-worker weight. Keeps `worker_count × weight` and the sums over sets
/// finite, so weighted picks never see an infinite range.
pub const MAX_PER_WORKER_WEIGHT: f64 = 1e6;
/// Smallest accepted per-worker weight. Rendezvous scoring also normalizes weights, but a
/// denormal weight is almost certainly a typo.
pub const MIN_PER_WORKER_WEIGHT: f64 = 1e-6;
/// Decisions over which the share tracker averages (exponential decay horizon).
const SHARE_WINDOW: f64 = 1000.0;
/// Decayed decisions required before the share guard may override the hashed set.
const SHARE_MIN_SAMPLES: f64 = 200.0;
/// A recorded charge is clamped to this multiple of the decayed mean charge.
const CHARGE_CLAMP_FACTOR: f64 = 8.0;
/// Standard deviations of sampling noise in the demand share tolerated before spilling.
const SPILL_NOISE_SIGMAS: f64 = 2.0;
/// Size of the per-frontend heavy-key table.
const HEAVY_KEYS: usize = 8;
/// Share of recent load at which a key in the table counts as heavy and may be redirected
/// by the overflow step.
const HEAVY_MIN_SHARE: f64 = 0.01;
/// Upper bound on one request's charge (bytes), so one pathological request cannot
/// dominate the window.
pub const MAX_REQUEST_CHARGE: f64 = 64.0 * 1024.0 * 1024.0;
/// Nominal charge for a non-text content part (image, audio, video reference).
const NON_TEXT_PART_CHARGE: usize = 1024;

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
                Ok(v) if v.is_finite() && v >= MIN_SLACK => config.slack = v,
                Ok(v) if v.is_finite() && v >= 0.0 => {
                    tracing::warn!(
                        value = raw,
                        "{SLACK_ENV} below {MIN_SLACK}; using {MIN_SLACK}"
                    );
                    config.slack = MIN_SLACK;
                }
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
                            && (MIN_PER_WORKER_WEIGHT..=MAX_PER_WORKER_WEIGHT).contains(w)
                    });
                match parsed {
                    Some((suffix, w)) => config.weights.push((suffix.to_string(), w)),
                    None => tracing::warn!(
                        value = item,
                        "Invalid {WEIGHTS_ENV} entry (weight must be in \
                         [{MIN_PER_WORKER_WEIGHT}, {MAX_PER_WORKER_WEIGHT}]); ignored"
                    ),
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
    /// The rendezvous winner.
    Affinity,
    /// A key-deterministic sticky spill to the key's second rendezvous choice.
    ShareCapFallback,
    /// A heavy key redirected away from its native set while that set is above its band.
    ShareCapOverflow,
    /// A spilled heavy key sent back to its rendezvous winner because the spill target is
    /// above its band.
    ShareCapRevert,
    /// No affinity key (or affinity disabled): weighted random pick.
    Random,
}

impl SetChoiceReason {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Affinity => "affinity",
            Self::ShareCapFallback => "share_cap_fallback",
            Self::ShareCapOverflow => "share_cap_overflow",
            Self::ShareCapRevert => "share_cap_revert",
            Self::Random => "random",
        }
    }
}

/// A request's affinity key and its load charge (see [`request_charge`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetAffinity {
    pub key: u64,
    pub charge: f64,
}

impl SetAffinity {
    /// `charge` is clamped to `[1, MAX_REQUEST_CHARGE]`; non-finite charges count as 1.
    pub fn new(key: u64, charge: f64) -> Self {
        let charge = if charge.is_finite() {
            charge.clamp(1.0, MAX_REQUEST_CHARGE)
        } else {
            1.0
        };
        Self { key, charge }
    }
}

/// Largest usable (positive, finite) candidate weight, used to normalize scores so that
/// scaling every weight by a common factor never changes a choice.
fn max_usable_weight(candidates: &[(&str, f64)]) -> Option<f64> {
    candidates
        .iter()
        .map(|(_, w)| *w)
        .filter(|w| w.is_finite() && *w > 0.0)
        .reduce(f64::max)
}

/// Rendezvous score of one candidate with a weight already divided by the largest weight
/// (so in `(0, 1]`): lower wins.
fn rendezvous_score(key: u64, name: &str, normalized_weight: f64) -> f64 {
    let h = xxh3_64_with_seed(name.as_bytes(), key);
    // Map to (0, 1]: never 0 so ln() stays finite.
    let u = ((h >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    -u.ln() / normalized_weight
}

/// Candidates with a positive finite weight, best rendezvous score first. Ties (only
/// possible with identical names) break by name, so the order never depends on the order
/// of `candidates`.
pub fn rendezvous_ranking(key: u64, candidates: &[(&str, f64)]) -> Vec<usize> {
    let Some(max) = max_usable_weight(candidates) else {
        return Vec::new();
    };
    let mut scored: Vec<(f64, &str, usize)> = candidates
        .iter()
        .enumerate()
        .filter(|(_, (_, w))| w.is_finite() && *w > 0.0)
        .map(|(idx, (name, w))| (rendezvous_score(key, name, *w / max), *name, idx))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored.into_iter().map(|(_, _, idx)| idx).collect()
}

/// Weighted rendezvous hashing: candidate `i` wins with probability `w_i / sum(w)` over
/// keys, and removing a candidate only moves the keys it owned.
pub fn rendezvous_pick(key: u64, candidates: &[(&str, f64)]) -> Option<usize> {
    rendezvous_ranking(key, candidates).first().copied()
}

/// Secondary position of a key in `[0, 1)`, independent of its rendezvous scores. A key
/// spills when its spill point is below the current spill fraction, so at a given fraction
/// every frontend spills the same keys.
pub fn spill_point(key: u64) -> f64 {
    const SPILL_SEED: u64 = 0x5b11_15ee_d0c4_a9e3;
    let h = xxh3_64_with_seed(&key.to_le_bytes(), SPILL_SEED);
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// Pick an index with probability proportional to its weight. Non-positive and non-finite
/// weights are skipped; returns `None` when no weight is usable.
pub fn weighted_random_pick(weights: &[f64]) -> Option<usize> {
    let usable = |w: f64| w > 0.0 && w.is_finite();
    let total: f64 = weights.iter().copied().filter(|w| usable(*w)).sum();
    if !total.is_finite() || total <= 0.0 {
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

/// Half-width of a set's allowed share band around its fair share.
fn share_band(fair: f64, slack: f64) -> f64 {
    slack * fair.min(1.0 - fair)
}

/// Fraction of a set's keys to spill so that, if load were spread evenly over keys, its
/// share would come down to the middle of its upper band plus `noise` (the sampling
/// uncertainty of the demand share, so that random fluctuations of a balanced workload do
/// not spill keys back and forth). Zero until the demand share exceeds that point; never
/// above `slack` (whole-key moves cannot fix load concentrated in a few keys, and a larger
/// fraction would only displace unrelated conversations; the heavy-key overflow step
/// handles that case).
fn spill_fraction(demand_share: f64, fair: f64, band: f64, slack: f64, noise: f64) -> f64 {
    let target = fair + band / 2.0 + noise;
    if demand_share <= target {
        return 0.0;
    }
    (1.0 - target / demand_share).clamp(0.0, slack)
}

/// Decayed, charge-weighted statistics per WorkerSet key.
#[derive(Debug, Default)]
struct ShareState {
    /// Decayed number of decisions (sample gate).
    decisions: f64,
    /// Decayed sum of recorded (clamped) charges; `charge_sum / decisions` is the mean.
    charge_sum: f64,
    /// Decayed sum of squared charges, for the effective sample size.
    charge_sq_sum: f64,
    /// Load times each set's weight share at decision time: the decayed "fair" load.
    expected: HashMap<String, f64>,
    /// Load whose rendezvous winner was the set.
    demand: HashMap<String, f64>,
    /// Load actually sent to the set.
    placed: HashMap<String, f64>,
    /// Load sent to the set by keys whose rendezvous winner is another set.
    spilled_in: HashMap<String, f64>,
    /// Decayed load of the heaviest recent keys (at most [`HEAVY_KEYS`] entries).
    heavy: Vec<(u64, f64)>,
}

impl ShareState {
    fn stat(map: &HashMap<String, f64>, name: &str) -> f64 {
        map.get(name).copied().unwrap_or(0.0)
    }

    /// Whether `key` carries at least [`HEAVY_MIN_SHARE`] of recent load.
    fn is_heavy(&self, key: u64) -> bool {
        self.heavy
            .iter()
            .any(|&(k, load)| k == key && load >= HEAVY_MIN_SHARE * self.charge_sum)
    }

    /// Sampling standard deviation of a charge-weighted share `share`, from the effective
    /// number of samples `charge_sum² / charge_sq_sum` in the window.
    fn share_noise(&self, share: f64) -> f64 {
        if self.charge_sq_sum <= 0.0 {
            return 0.0;
        }
        let effective = self.charge_sum * self.charge_sum / self.charge_sq_sum;
        SPILL_NOISE_SIGMAS * (share * (1.0 - share) / effective.max(1.0)).sqrt()
    }

    /// `charge` clamped to [`CHARGE_CLAMP_FACTOR`] times the decayed mean charge, so one
    /// very large request cannot swing the statistics.
    fn clamp_charge(&self, charge: f64) -> f64 {
        if self.decisions >= 1.0 && self.charge_sum > 0.0 {
            charge.min(CHARGE_CLAMP_FACTOR * self.charge_sum / self.decisions)
        } else {
            charge
        }
    }

    /// The decision for `key`; pure in the state.
    fn decide(
        &self,
        key: u64,
        candidates: &[(&str, f64)],
        ranking: &[usize],
        slack: f64,
    ) -> (usize, SetChoiceReason) {
        let preferred = ranking[0];
        let mut choice = (preferred, SetChoiceReason::Affinity);
        if ranking.len() < 2 || self.decisions < SHARE_MIN_SAMPLES {
            return choice;
        }
        let name = |i: usize| candidates[i].0;
        let total =
            |map: &HashMap<String, f64>| ranking.iter().map(|&i| Self::stat(map, name(i))).sum();
        let (expected, demand, placed): (f64, f64, f64) = (
            total(&self.expected),
            total(&self.demand),
            total(&self.placed),
        );
        if expected <= 0.0 || demand <= 0.0 || placed <= 0.0 {
            return choice;
        }
        let fair = |i: usize| Self::stat(&self.expected, name(i)) / expected;
        let band = |i: usize| share_band(fair(i), slack);

        // 1. Sticky spill, decided by the key's spill point.
        let demand_share = Self::stat(&self.demand, name(preferred)) / demand;
        let p = spill_fraction(
            demand_share,
            fair(preferred),
            band(preferred),
            slack,
            self.share_noise(demand_share),
        );
        if spill_point(key) < p {
            choice = (ranking[1], SetChoiceReason::ShareCapFallback);
        }

        // 2. Heavy-key overflow: only a key carrying a large share of recent load is ever
        //    redirected, and only while its chosen set is above its band. A key native to
        //    that set stays unless the overshoot exceeds what other sets' keys spilled in.
        if !self.is_heavy(key) {
            return choice;
        }
        let share = |i: usize| Self::stat(&self.placed, name(i)) / placed;
        let chosen = choice.0;
        let overshoot = share(chosen) - (fair(chosen) + band(chosen));
        let spilled_in = Self::stat(&self.spilled_in, name(chosen)) / placed;
        let redirect = overshoot > 0.0 && (chosen != preferred || overshoot > spilled_in);
        if redirect && let Some(&alt) = ranking.iter().find(|&&i| i != chosen && share(i) < fair(i))
        {
            let reason = if alt == preferred {
                SetChoiceReason::ShareCapRevert
            } else {
                SetChoiceReason::ShareCapOverflow
            };
            choice = (alt, reason);
        }
        choice
    }

    fn record(
        &mut self,
        key: u64,
        candidates: &[(&str, f64)],
        ranking: &[usize],
        chosen: usize,
        charge: f64,
    ) {
        let charge = self.clamp_charge(charge);
        let decay = 1.0 - 1.0 / SHARE_WINDOW;
        self.decisions = self.decisions * decay + 1.0;
        self.charge_sum = self.charge_sum * decay + charge;
        self.charge_sq_sum = self.charge_sq_sum * decay * decay + charge * charge;
        for map in [
            &mut self.expected,
            &mut self.demand,
            &mut self.placed,
            &mut self.spilled_in,
        ] {
            for v in map.values_mut() {
                *v *= decay;
            }
            map.retain(|_, v| *v > 1e-6);
        }
        let total_weight: f64 = ranking.iter().map(|&i| candidates[i].1).sum();
        for &i in ranking {
            *self
                .expected
                .entry(candidates[i].0.to_string())
                .or_insert(0.0) += charge * candidates[i].1 / total_weight;
        }
        *self
            .demand
            .entry(candidates[ranking[0]].0.to_string())
            .or_insert(0.0) += charge;
        *self
            .placed
            .entry(candidates[chosen].0.to_string())
            .or_insert(0.0) += charge;
        if chosen != ranking[0] {
            *self
                .spilled_in
                .entry(candidates[chosen].0.to_string())
                .or_insert(0.0) += charge;
        }
        self.record_heavy(key, charge, decay);
    }

    /// Decayed top-K by load: a present key accumulates; otherwise the key replaces the
    /// lightest entry when its charge exceeds that entry's load.
    fn record_heavy(&mut self, key: u64, charge: f64, decay: f64) {
        for entry in &mut self.heavy {
            entry.1 *= decay;
        }
        if let Some(entry) = self.heavy.iter_mut().find(|(k, _)| *k == key) {
            entry.1 += charge;
        } else if self.heavy.len() < HEAVY_KEYS {
            self.heavy.push((key, charge));
        } else if let Some(lightest) = self
            .heavy
            .iter_mut()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .filter(|(_, load)| *load < charge)
        {
            *lightest = (key, charge);
        }
    }
}

/// Per-frontend share guard for affinity decisions (see the module docs).
#[derive(Debug, Default)]
pub struct ShareTracker {
    state: Mutex<ShareState>,
}

impl ShareTracker {
    /// Choose among `candidates` (unique set key, weight) for `affinity` and record the
    /// decision. Every reason other than [`SetChoiceReason::Affinity`] and
    /// [`SetChoiceReason::ShareCapRevert`] means the chosen set differs from the
    /// rendezvous winner.
    pub fn choose(
        &self,
        affinity: SetAffinity,
        candidates: &[(&str, f64)],
        slack: f64,
    ) -> Option<(usize, SetChoiceReason)> {
        let ranking = rendezvous_ranking(affinity.key, candidates);
        if ranking.is_empty() {
            return None;
        }
        let mut state = self.state.lock();
        let choice = state.decide(affinity.key, candidates, &ranking, slack);
        state.record(
            affinity.key,
            candidates,
            &ranking,
            choice.0,
            affinity.charge,
        );
        Some(choice)
    }

    /// The decision `choose` would make now, without recording it (test hook).
    #[cfg(test)]
    pub(crate) fn peek(
        &self,
        key: u64,
        candidates: &[(&str, f64)],
        slack: f64,
    ) -> (usize, SetChoiceReason) {
        let ranking = rendezvous_ranking(key, candidates);
        self.state.lock().decide(key, candidates, &ranking, slack)
    }

    /// Share of charged load placed on `name` (test hook).
    #[cfg(test)]
    pub(crate) fn share(&self, name: &str) -> f64 {
        let state = self.state.lock();
        let total: f64 = state.placed.values().sum();
        ShareState::stat(&state.placed, name) / total.max(f64::MIN_POSITIVE)
    }

    /// Current sticky-spill fraction for set `idx` of `candidates` (test hook).
    #[cfg(test)]
    pub(crate) fn spill_fraction_of(
        &self,
        idx: usize,
        candidates: &[(&str, f64)],
        slack: f64,
    ) -> f64 {
        let state = self.state.lock();
        let total = |map: &HashMap<String, f64>| -> f64 {
            candidates
                .iter()
                .map(|(n, _)| ShareState::stat(map, n))
                .sum()
        };
        let fair = ShareState::stat(&state.expected, candidates[idx].0) / total(&state.expected);
        let demand = ShareState::stat(&state.demand, candidates[idx].0) / total(&state.demand);
        spill_fraction(
            demand,
            fair,
            share_band(fair, slack),
            slack,
            state.share_noise(demand),
        )
    }

    /// Whether `key` currently counts as heavy (test hook).
    #[cfg(test)]
    pub(crate) fn is_heavy(&self, key: u64) -> bool {
        self.state.lock().is_heavy(key)
    }

    /// Decayed number of recorded decisions (test hook).
    #[cfg(test)]
    pub(crate) fn observed_total(&self) -> f64 {
        self.state.lock().decisions
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

/// Domain tags so an explicit key can never collide with a message-prefix hash.
const PROMPT_CACHE_KEY_TAG: &[u8] = b"prompt_cache_key\0";
const SESSION_AFFINITY_TAG: &[u8] = b"session_affinity\0";
const MESSAGES_TAG: &[u8] = b"messages/v3\0";

fn explicit_affinity_key(tag: &[u8], value: &str) -> u64 {
    let mut hasher = Xxh3::with_seed(AFFINITY_SEED);
    hasher.update(tag);
    hasher.update(value.as_bytes());
    hasher.digest()
}

/// Serialized JSON length of `value`, without allocating the serialization.
fn json_len<T: serde::Serialize>(value: &T) -> Option<usize> {
    struct Count(usize);
    impl io::Write for Count {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0 += buf.len();
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut count = Count(0);
    serde_json::to_writer(&mut count, value).ok()?;
    Some(count.0)
}

/// Text pieces of system/developer content (parts are concatenated, as both the chat
/// templates and the Responses conversion do).
fn system_text(content: &ChatCompletionRequestSystemMessageContent) -> impl Iterator<Item = &str> {
    let (text, parts) = match content {
        ChatCompletionRequestSystemMessageContent::Text(t) => (Some(t.as_str()), None),
        ChatCompletionRequestSystemMessageContent::Array(parts) => (None, Some(parts)),
    };
    text.into_iter()
        .chain(parts.into_iter().flatten().map(|p| match p {
            ChatCompletionRequestSystemMessageContentPart::Text(t) => t.text.as_str(),
        }))
}

fn developer_text(
    content: &ChatCompletionRequestDeveloperMessageContent,
) -> impl Iterator<Item = &str> {
    let (text, parts) = match content {
        ChatCompletionRequestDeveloperMessageContent::Text(t) => (Some(t.as_str()), None),
        ChatCompletionRequestDeveloperMessageContent::Array(parts) => (None, Some(parts)),
    };
    text.into_iter()
        .chain(parts.into_iter().flatten().map(|p| match p {
            ChatCompletionRequestDeveloperMessageContentPart::Text(t) => t.text.as_str(),
        }))
}

/// Streams the canonical opening into the hasher. Each record is a one-byte tag followed by
/// a little-endian `u64` length and that many bytes, so the encoding is unambiguous and
/// text is never copied.
struct OpeningHasher(HashWriter);

impl OpeningHasher {
    fn update(&mut self, bytes: &[u8]) {
        self.0.0.update(bytes);
    }

    fn tag(&mut self, tag: u8) {
        self.update(&[tag]);
    }

    /// A text record whose bytes are `pieces` joined by `separator`.
    fn text(&mut self, tag: u8, pieces: &[&str], separator: &str) {
        let len = pieces.iter().map(|p| p.len()).sum::<usize>()
            + separator.len() * pieces.len().saturating_sub(1);
        self.tag(tag);
        self.update(&(len as u64).to_le_bytes());
        for (i, piece) in pieces.iter().enumerate() {
            if i > 0 {
                self.update(separator.as_bytes());
            }
            self.update(piece.as_bytes());
        }
    }

    /// A non-text part, identified by its JSON serialization (streamed, not buffered).
    fn part<T: serde::Serialize>(&mut self, part: &T) -> Option<()> {
        let len = json_len(part)?;
        self.tag(b'P');
        self.update(&(len as u64).to_le_bytes());
        serde_json::to_writer(&mut self.0, part).ok()
    }

    fn name(&mut self, name: Option<&String>) {
        if let Some(name) = name {
            self.text(b'N', &[name], "");
        }
    }

    fn user(&mut self, content: &ChatCompletionRequestUserMessageContent) -> Option<()> {
        match content {
            ChatCompletionRequestUserMessageContent::Text(t) => self.text(b'T', &[t], ""),
            ChatCompletionRequestUserMessageContent::Array(parts) => {
                let mut run: Vec<&str> = Vec::new();
                let mut wrote = false;
                for part in parts {
                    if let ChatCompletionRequestUserMessageContentPart::Text(t) = part {
                        run.push(&t.text);
                        continue;
                    }
                    if !run.is_empty() {
                        self.text(b'T', &run, "");
                        run.clear();
                    }
                    self.part(part)?;
                    wrote = true;
                }
                if !run.is_empty() || !wrote {
                    self.text(b'T', &run, "");
                }
            }
        }
        Some(())
    }
}

/// Affinity key for a conversation from its messages: a hash of every message before the
/// first assistant, tool or function message (the client-authored opening).
///
/// Every later turn of the same conversation repeats that opening, so turns map to the same
/// key, while two sessions that share boilerplate leading user items (for example Codex's
/// AGENTS.md and `<environment_context>`) still differ by their task.
///
/// The opening is canonicalized so that Chat Completions and converted Responses requests
/// agree: the leading run of system/developer messages is merged into one text joined by
/// `"\n\n"` (exactly what the Responses conversion does), developer counts as system, and
/// each run of adjacent text parts is concatenated, so a string and the equivalent text-part
/// array hash alike. Text is streamed into the hasher without copying the opening.
///
/// Returns `None` when the opening holds no user message (system-only requests, or a
/// conversation that starts with an assistant message): those carry nothing
/// conversation-specific, so they take the weighted random pick instead of collapsing
/// onto one hot key.
///
/// TODO(D6): non-text parts are hashed verbatim, so a follow-up turn that re-sends an image
/// as a UUID-only reference (instead of URL + UUID) changes the key. Canonicalize image
/// parts to their UUID before enabling affinity for a multimodal model (Super 3.5 is a VLM).
pub fn messages_affinity_key(messages: &[ChatCompletionRequestMessage]) -> Option<u64> {
    use ChatCompletionRequestMessage as M;
    let end = messages
        .iter()
        .position(|m| matches!(m, M::Assistant(_) | M::Tool(_) | M::Function(_)))
        .unwrap_or(messages.len());
    let opening = &messages[..end];
    if !opening.iter().any(|m| matches!(m, M::User(_))) {
        return None;
    }
    let leading = opening
        .iter()
        .take_while(|m| matches!(m, M::System(_) | M::Developer(_)))
        .count();

    let mut hasher = OpeningHasher(HashWriter(Xxh3::with_seed(AFFINITY_SEED)));
    hasher.update(MESSAGES_TAG);
    if leading > 0 {
        // Same text as the Responses conversion's merged leading system message.
        let mut pieces: Vec<&str> = Vec::new();
        for (i, message) in opening[..leading].iter().enumerate() {
            if i > 0 {
                pieces.push("\n\n");
            }
            match message {
                M::System(m) => pieces.extend(system_text(&m.content)),
                M::Developer(m) => pieces.extend(developer_text(&m.content)),
                _ => {}
            }
        }
        hasher.text(b'S', &pieces, "");
        hasher.tag(0x1e);
    }
    for message in &opening[leading..] {
        match message {
            M::System(m) => {
                hasher.tag(b's');
                hasher.name(m.name.as_ref());
                hasher.text(b'T', &system_text(&m.content).collect::<Vec<_>>(), "");
            }
            M::Developer(m) => {
                hasher.tag(b's');
                hasher.name(m.name.as_ref());
                hasher.text(b'T', &developer_text(&m.content).collect::<Vec<_>>(), "");
            }
            M::User(m) => {
                hasher.tag(b'u');
                hasher.name(m.name.as_ref());
                hasher.user(&m.content)?;
            }
            // The opening ends before the first of these.
            M::Assistant(_) | M::Tool(_) | M::Function(_) => {}
        }
        hasher.tag(0x1e);
    }
    Some(hasher.0.0.digest())
}

/// Affinity key for a chat-shaped request (Chat Completions, or Responses after
/// conversion), in priority order:
///
/// 1. `prompt_cache_key` (Responses field, or the Chat Completions extra-body field),
/// 2. the session affinity header ([`SessionAffinityId`]),
/// 3. [`messages_affinity_key`] over the messages.
///
/// Blank explicit keys are ignored.
///
/// [`SessionAffinityId`]: crate::protocols::common::extensions::SessionAffinityId
pub fn chat_request_affinity_key(
    messages: &[ChatCompletionRequestMessage],
    prompt_cache_key: Option<&str>,
    session_affinity: Option<&str>,
) -> Option<u64> {
    let non_blank = |s: &&str| !s.trim().is_empty();
    if let Some(key) = prompt_cache_key.filter(non_blank) {
        return Some(explicit_affinity_key(PROMPT_CACHE_KEY_TAG, key));
    }
    if let Some(session) = session_affinity.filter(non_blank) {
        return Some(explicit_affinity_key(SESSION_AFFINITY_TAG, session));
    }
    messages_affinity_key(messages)
}

/// Cheap pre-tokenization size proxy for a request: the UTF-8 bytes of all message text
/// (content, reasoning, refusals, tool-call names and arguments), plus a nominal charge per
/// non-text part. Linear in the number of messages and parts; no allocation or
/// serialization.
pub fn request_charge(messages: &[ChatCompletionRequestMessage]) -> f64 {
    use ChatCompletionRequestMessage as M;
    let user_part = |p: &ChatCompletionRequestUserMessageContentPart| match p {
        ChatCompletionRequestUserMessageContentPart::Text(t) => t.text.len(),
        _ => NON_TEXT_PART_CHARGE,
    };
    let tool_part = |p: &ChatCompletionRequestToolMessageContentPart| match p {
        ChatCompletionRequestToolMessageContentPart::Text(t) => t.text.len(),
        _ => NON_TEXT_PART_CHARGE,
    };
    let bytes: usize = messages
        .iter()
        .map(|message| match message {
            M::System(m) => system_text(&m.content).map(str::len).sum(),
            M::Developer(m) => developer_text(&m.content).map(str::len).sum(),
            M::User(m) => match &m.content {
                ChatCompletionRequestUserMessageContent::Text(t) => t.len(),
                ChatCompletionRequestUserMessageContent::Array(parts) => {
                    parts.iter().map(user_part).sum()
                }
            },
            M::Assistant(m) => {
                let content = match &m.content {
                    Some(ChatCompletionRequestAssistantMessageContent::Text(t)) => t.len(),
                    Some(ChatCompletionRequestAssistantMessageContent::Array(parts)) => parts
                        .iter()
                        .map(|p| match p {
                            ChatCompletionRequestAssistantMessageContentPart::Text(t) => {
                                t.text.len()
                            }
                            ChatCompletionRequestAssistantMessageContentPart::Refusal(r) => {
                                r.refusal.len()
                            }
                        })
                        .sum(),
                    None => 0,
                };
                let reasoning = match &m.reasoning_content {
                    Some(ReasoningContent::Text(t)) => t.len(),
                    Some(ReasoningContent::Segments(s)) => s.iter().map(String::len).sum(),
                    None => 0,
                };
                let tools: usize = m
                    .tool_calls
                    .iter()
                    .flatten()
                    .map(|c| c.function.name.len() + c.function.arguments.len())
                    .sum();
                content + reasoning + tools + m.refusal.as_ref().map_or(0, String::len)
            }
            M::Tool(m) => match &m.content {
                ChatCompletionRequestToolMessageContent::Text(t) => t.len(),
                ChatCompletionRequestToolMessageContent::Array(parts) => {
                    parts.iter().map(tool_part).sum()
                }
            },
            M::Function(m) => m.content.as_ref().map_or(0, String::len),
        })
        .fold(0usize, usize::saturating_add);
    (bytes as f64).clamp(1.0, MAX_REQUEST_CHARGE)
}

/// `prompt_cache_key` sent as a Chat Completions extra-body field (the pinned protocol
/// type predates it, so it lands in `unsupported_fields`).
pub fn chat_prompt_cache_key(request: &NvCreateChatCompletionRequest) -> Option<&str> {
    request
        .unsupported_fields
        .get("prompt_cache_key")
        .and_then(serde_json::Value::as_str)
}

/// Process-global switch read by the HTTP layer: compute keys only when they are used.
pub fn affinity_enabled() -> bool {
    SetSelectionConfig::global().mode == SetSelectionMode::Affinity
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::responses::NvCreateResponse;
    use serde_json::json;

    const TP4: &str = "ns-tp4";
    const TP2: &str = "ns-tp2";
    const SETS: [(&str, f64); 2] = [(TP4, 30.0), (TP2, 60.0)];

    fn unit(key: u64) -> SetAffinity {
        SetAffinity::new(key, 1.0)
    }

    fn mixed(key: u64) -> u64 {
        key.wrapping_mul(0x9e37_79b9_7f4a_7c15)
    }

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

    /// R2-5: a near-zero band spills about half of all keys at fair load.
    #[test]
    fn parse_clamps_small_slack() {
        for raw in ["0", "0.0", "0.01", "0.0499"] {
            let c = SetSelectionConfig::parse(Some("affinity"), Some(raw), None);
            assert_eq!(c.slack, MIN_SLACK, "slack {raw}");
        }
        let c = SetSelectionConfig::parse(Some("affinity"), Some("0.05"), None);
        assert_eq!(c.slack, 0.05);
        let c = SetSelectionConfig::parse(Some("affinity"), Some("NaN"), None);
        assert_eq!(c.slack, DEFAULT_SLACK);
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

    /// R2-4: denormal per-worker weights are rejected, so both sets fall back to the
    /// default per-worker weight (worker count alone).
    #[test]
    fn parse_rejects_tiny_weights() {
        let c = SetSelectionConfig::parse(None, None, Some("tp4=5e-324,tp2=5e-324,x=1e-7"));
        assert!(c.weights.is_empty());
        let c = SetSelectionConfig::parse(None, None, Some("tp4=1e-6"));
        assert_eq!(c.weights, vec![("tp4".to_string(), MIN_PER_WORKER_WEIGHT)]);
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

    fn tp4_share(candidates: &[(&str, f64)], n: u64) -> f64 {
        let hits = (0..n)
            .filter(|k| rendezvous_pick(mixed(*k), candidates) == Some(0))
            .count();
        hits as f64 / n as f64
    }

    #[test]
    fn rendezvous_is_deterministic_and_weighted() {
        for key in 0..1000u64 {
            let a = rendezvous_pick(key, &SETS).unwrap();
            assert_eq!(a, rendezvous_pick(key, &SETS).unwrap());
            assert_eq!(a, rendezvous_ranking(key, &SETS)[0]);
        }
        let share = tp4_share(&SETS, 60_000);
        assert!((share - 1.0 / 3.0).abs() < 0.01, "share {share}");
    }

    /// R2-4: denormal weights used to make every score infinite, leaving the name tie-break
    /// to pick the set. Scores are normalized by the largest weight, so a common scale
    /// never changes a choice.
    #[test]
    fn rendezvous_is_scale_invariant() {
        let tiny = [(TP4, 30.0 * 5e-324), (TP2, 60.0 * 5e-324)];
        let share = tp4_share(&tiny, 60_000);
        assert!((share - 1.0 / 3.0).abs() < 0.01, "share {share}");
        let equal_tiny = [(TP4, 5e-324), (TP2, 5e-324)];
        let share = tp4_share(&equal_tiny, 60_000);
        assert!((share - 0.5).abs() < 0.01, "share {share}");
        for scale in [1e-300, 1e-6, 0.37, 3.0, 1e6, 1e300] {
            let scaled = [(TP4, 30.0 * scale), (TP2, 60.0 * scale)];
            for key in 0..5000u64 {
                assert_eq!(
                    rendezvous_ranking(mixed(key), &scaled),
                    rendezvous_ranking(mixed(key), &SETS),
                    "scale {scale}"
                );
            }
        }
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

    /// D1: candidates are unique set keys; the ranking must not depend on candidate order.
    #[test]
    fn rendezvous_winner_is_independent_of_candidate_order() {
        let fwd = [("ns:set-a", 1.0), ("ns:set-b", 1.0)];
        let rev = [("ns:set-b", 1.0), ("ns:set-a", 1.0)];
        let mut a_wins = 0;
        for key in 0..4000u64 {
            let rf: Vec<&str> = rendezvous_ranking(key, &fwd)
                .into_iter()
                .map(|i| fwd[i].0)
                .collect();
            let rr: Vec<&str> = rendezvous_ranking(key, &rev)
                .into_iter()
                .map(|i| rev[i].0)
                .collect();
            assert_eq!(rf, rr);
            a_wins += usize::from(rf[0] == "ns:set-a");
        }
        assert!((1600..2400).contains(&a_wins), "a_wins {a_wins}");
    }

    /// R2-7: a worker-count change only moves the keys rendezvous hashing must move
    /// (60 → 30 TP2 workers moves 1/3 − 1/2 = 1/6 of keys to TP4), and the original
    /// mapping is restored exactly when the count returns.
    #[test]
    fn worker_count_change_moves_only_the_rebalanced_keys() {
        let full = [(TP4, 30.0), (TP2, 60.0)];
        let halved = [(TP4, 30.0), (TP2, 30.0)];
        let n = 60_000u64;
        let mut moved = 0;
        for key in 0..n {
            let key = mixed(key);
            let before = rendezvous_pick(key, &full).unwrap();
            let during = rendezvous_pick(key, &halved).unwrap();
            if before != during {
                assert_eq!((before, during), (1, 0), "only TP2 → TP4 moves");
                moved += 1;
            }
            assert_eq!(rendezvous_pick(key, &full).unwrap(), before);
        }
        let moved = moved as f64 / n as f64;
        assert!((moved - 1.0 / 6.0).abs() < 0.01, "moved {moved}");
    }

    /// R2-7: the share guard adds no spills of its own across a worker-count change,
    /// because fair shares are averaged over the same window as the observed load.
    #[test]
    fn worker_count_change_causes_no_guard_transient() {
        let tracker = ShareTracker::default();
        let full = [(TP4, 30.0), (TP2, 60.0)];
        let halved = [(TP4, 30.0), (TP2, 30.0)];
        let mut key = 0u64;
        let mut run = |cands: &[(&str, f64)], n: usize| {
            let mut overrides = 0;
            for _ in 0..n {
                key += 1;
                let (idx, reason) = tracker
                    .choose(unit(mixed(key)), cands, DEFAULT_SLACK)
                    .unwrap();
                if reason != SetChoiceReason::Affinity {
                    overrides += 1;
                } else {
                    assert_eq!(Some(idx), rendezvous_pick(mixed(key), cands));
                }
            }
            overrides
        };
        assert!(run(&full, 5000) < 50);
        assert!(run(&halved, 5000) < 50, "60 → 30 transient");
        assert!(run(&full, 5000) < 50, "30 → 60 transient");
    }

    fn hot_key_preferring(name: &str) -> u64 {
        (0..u64::MAX)
            .find(|k| SETS[rendezvous_pick(*k, &SETS).unwrap()].0 == name)
            .unwrap()
    }

    /// Run `n` decisions of one hot key and return how many left the preferred set.
    fn run_hot_key(tracker: &ShareTracker, key: u64, n: usize) -> usize {
        let ranking = rendezvous_ranking(key, &SETS);
        let mut moved = 0;
        for _ in 0..n {
            let (idx, reason) = tracker.choose(unit(key), &SETS, DEFAULT_SLACK).unwrap();
            match reason {
                SetChoiceReason::ShareCapFallback | SetChoiceReason::ShareCapOverflow => {
                    assert_ne!(idx, ranking[0], "a fallback must change the set");
                    assert_eq!(
                        idx, ranking[1],
                        "spill target is the next rendezvous choice"
                    );
                    moved += 1;
                }
                SetChoiceReason::Affinity | SetChoiceReason::ShareCapRevert => {
                    assert_eq!(idx, ranking[0])
                }
                SetChoiceReason::Random => panic!("affinity choice labelled random"),
            }
        }
        moved
    }

    /// A1: a hot key that prefers the larger set must not squeeze the smaller set.
    #[test]
    fn share_guard_protects_smaller_set() {
        let tracker = ShareTracker::default();
        let moved = run_hot_key(&tracker, hot_key_preferring(TP2), 5000);
        assert!(moved > 0);
        let floor = (1.0 / 3.0) * (1.0 - DEFAULT_SLACK) - 0.03;
        let tp4 = tracker.share(TP4);
        assert!(tp4 >= floor, "tp4 share {tp4} below {floor}");
    }

    #[test]
    fn share_guard_caps_a_hot_key_preferring_smaller_set() {
        let tracker = ShareTracker::default();
        let moved = run_hot_key(&tracker, hot_key_preferring(TP4), 5000);
        assert!(moved > 0);
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
        let mut overrides = 0;
        for key in 0..20_000u64 {
            let key = mixed(key);
            let (idx, reason) = tracker.choose(unit(key), &SETS, DEFAULT_SLACK).unwrap();
            if reason == SetChoiceReason::Affinity {
                assert_eq!(Some(idx), rendezvous_pick(key, &SETS));
            } else {
                assert_ne!(Some(idx), rendezvous_pick(key, &SETS));
                overrides += 1;
            }
        }
        assert!(overrides < 200, "overrides {overrides}");
    }

    /// R2-2: load is charged by request size, so a key whose requests are 10x larger
    /// cannot push its set past the band by request count alone.
    #[test]
    fn share_guard_balances_charged_load() {
        for heavy_pref in [TP4, TP2] {
            let tracker = ShareTracker::default();
            let heavy = hot_key_preferring(heavy_pref);
            for i in 0..20_000u64 {
                let affinity = if i % 10 == 0 {
                    SetAffinity::new(heavy, 10.0)
                } else {
                    unit(mixed(i))
                };
                tracker.choose(affinity, &SETS, DEFAULT_SLACK).unwrap();
            }
            for (name, fair) in [(TP4, 1.0 / 3.0), (TP2, 2.0 / 3.0)] {
                let band = share_band(fair, DEFAULT_SLACK);
                let share = tracker.share(name);
                assert!(
                    (fair - band - 0.03..=fair + band + 0.03).contains(&share),
                    "heavy key prefers {heavy_pref}: {name} charged share {share}"
                );
            }
        }
    }

    /// Spilling with three sets goes to a set other than the rendezvous winner.
    #[test]
    fn share_guard_spills_to_an_underserved_set() {
        let sets = [("a", 1.0), ("b", 1.0), ("c", 1.0)];
        let tracker = ShareTracker::default();
        let key = 42;
        let ranking = rendezvous_ranking(key, &sets);
        for _ in 0..5000 {
            let (idx, reason) = tracker.choose(unit(key), &sets, DEFAULT_SLACK).unwrap();
            match reason {
                SetChoiceReason::Affinity | SetChoiceReason::ShareCapRevert => {
                    assert_eq!(idx, ranking[0])
                }
                _ => assert_ne!(idx, ranking[0]),
            }
        }
        let cap = 1.0 / 3.0 + share_band(1.0 / 3.0, DEFAULT_SLACK) + 0.03;
        assert!(tracker.share(sets[ranking[0]].0) <= cap);
    }

    // -- R3-1 / R3-2 / R3-3: heavy-key-only overflow, charge clamp, honest labels --

    /// A key preferring `name` whose spill point satisfies `pred`.
    fn key_preferring(name: &str, pred: impl Fn(f64) -> bool) -> u64 {
        (0..u64::MAX)
            .map(mixed)
            .find(|k| SETS[rendezvous_pick(*k, &SETS).unwrap()].0 == name && pred(spill_point(*k)))
            .unwrap()
    }

    /// Feed `n` decisions where 3 of every 10 requests are `hot` and the rest are distinct
    /// balanced keys; return the decisions of the non-hot keys.
    fn run_with_hot_key(
        tracker: &ShareTracker,
        hot: u64,
        n: u64,
    ) -> Vec<(u64, usize, SetChoiceReason)> {
        let mut others = Vec::new();
        for i in 0..n {
            if i % 10 < 3 {
                tracker.choose(unit(hot), &SETS, DEFAULT_SLACK).unwrap();
            } else {
                let key = mixed(i + 1_000_000);
                let (idx, reason) = tracker.choose(unit(key), &SETS, DEFAULT_SLACK).unwrap();
                others.push((key, idx, reason));
            }
        }
        others
    }

    /// R3-1: a heavy key that sticky-spills onto TP4 pushes TP4 over its band. Only the
    /// spilled heavy key is sent back (labelled `share_cap_revert`); TP4's own
    /// conversations keep their set.
    #[test]
    fn spilled_heavy_key_does_not_displace_native_keys() {
        let tracker = ShareTracker::default();
        let hot = key_preferring(TP2, |x| x < 0.03);
        let others = run_with_hot_key(&tracker, hot, 20_000);
        assert!(tracker.is_heavy(hot));
        assert!(tracker.spill_fraction_of(1, &SETS, DEFAULT_SLACK) > 0.03);
        assert_ne!(
            tracker.peek(hot, &SETS, DEFAULT_SLACK).1,
            SetChoiceReason::Affinity
        );
        for (key, idx, reason) in others {
            let preferred = rendezvous_pick(key, &SETS).unwrap();
            assert!(
                !matches!(
                    reason,
                    SetChoiceReason::ShareCapOverflow | SetChoiceReason::ShareCapRevert
                ),
                "ordinary key redirected by overflow"
            );
            if SETS[preferred].0 == TP4 {
                assert_eq!((idx, reason), (preferred, SetChoiceReason::Affinity));
            }
        }
        let tp4 = tracker.share(TP4);
        let cap = 1.0 / 3.0 + share_band(1.0 / 3.0, DEFAULT_SLACK) + 0.03;
        assert!(tp4 <= cap, "tp4 share {tp4}");
    }

    /// R3-1: a 30% hot key preferring TP4 is the only key redirected, and TP4 stays in band.
    #[test]
    fn hot_key_preferring_tp4_keeps_tp4_within_band() {
        for spilled in [false, true] {
            let tracker = ShareTracker::default();
            let hot = if spilled {
                key_preferring(TP4, |x| x < 0.1)
            } else {
                key_preferring(TP4, |x| x > 0.5)
            };
            let others = run_with_hot_key(&tracker, hot, 20_000);
            for (_, _, reason) in others {
                assert!(!matches!(
                    reason,
                    SetChoiceReason::ShareCapOverflow | SetChoiceReason::ShareCapRevert
                ));
            }
            for (name, fair) in [(TP4, 1.0 / 3.0), (TP2, 2.0 / 3.0)] {
                let band = share_band(fair, DEFAULT_SLACK);
                let share = tracker.share(name);
                assert!(
                    (fair - band - 0.03..=fair + band + 0.03).contains(&share),
                    "spilled {spilled}: {name} share {share}"
                );
            }
        }
    }

    /// R3-1 / R3-2: one request near the maximum charge is clamped to 8x the mean, is not
    /// a heavy key, and changes no other key's set.
    #[test]
    fn single_huge_request_moves_no_other_key() {
        let tracker = ShareTracker::default();
        for i in 0..5000u64 {
            tracker
                .choose(unit(mixed(i)), &SETS, DEFAULT_SLACK)
                .unwrap();
        }
        let probes: Vec<u64> = (0..2000u64).map(|i| mixed(i + 7_000_000)).collect();
        let before: Vec<_> = probes
            .iter()
            .map(|k| tracker.peek(*k, &SETS, DEFAULT_SLACK))
            .collect();
        let huge = mixed(42_000_000);
        tracker
            .choose(
                SetAffinity::new(huge, MAX_REQUEST_CHARGE),
                &SETS,
                DEFAULT_SLACK,
            )
            .unwrap();
        assert!(!tracker.is_heavy(huge));
        let after: Vec<_> = probes
            .iter()
            .map(|k| tracker.peek(*k, &SETS, DEFAULT_SLACK))
            .collect();
        assert_eq!(before, after);
        for i in 0..2000u64 {
            let key = mixed(i + 8_000_000);
            let (idx, reason) = tracker.choose(unit(key), &SETS, DEFAULT_SLACK).unwrap();
            assert_eq!(
                (Some(idx), reason),
                (rendezvous_pick(key, &SETS), SetChoiceReason::Affinity)
            );
        }
    }

    #[test]
    fn charges_are_clamped_to_a_multiple_of_the_mean() {
        let tracker = ShareTracker::default();
        for i in 0..1000u64 {
            tracker
                .choose(SetAffinity::new(mixed(i), 100.0), &SETS, DEFAULT_SLACK)
                .unwrap();
        }
        let state = tracker.state.lock();
        assert!((state.clamp_charge(1e9) - CHARGE_CLAMP_FACTOR * 100.0).abs() < 1.0);
        assert_eq!(state.clamp_charge(50.0), 50.0);
        assert_eq!(ShareState::default().clamp_charge(1e9), 1e9);
    }

    /// The guard stays idle while a fresh frontend has few samples.
    #[test]
    fn guard_waits_for_warm_up() {
        let tracker = ShareTracker::default();
        let hot = hot_key_preferring(TP4);
        for _ in 0..(SHARE_MIN_SAMPLES as usize) {
            let (idx, reason) = tracker.choose(unit(hot), &SETS, DEFAULT_SLACK).unwrap();
            assert_eq!((idx, reason), (0, SetChoiceReason::Affinity));
        }
    }

    /// Deterministic splitmix64 stream for the simulation tests.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn uniform(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (u1, u2) = (self.uniform(), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// Fraction of follow-up turns that change set when `frontends` independent trackers
    /// serve 3000 balanced conversations of 10 turns, with per-conversation lognormal(σ)
    /// size and context growing by turn.
    fn follow_up_set_changes(sigma: f64, frontends: usize, seed: u64) -> f64 {
        let mut rng = Rng(seed);
        let trackers: Vec<ShareTracker> = (0..frontends).map(|_| ShareTracker::default()).collect();
        let conversations: Vec<(u64, f64)> = (0..3000u64)
            .map(|c| (mixed(c + seed), 2000.0 * (sigma * rng.normal()).exp()))
            .collect();
        let mut last = vec![usize::MAX; conversations.len()];
        let mut order: Vec<usize> = (0..conversations.len()).collect();
        let (mut changes, mut follow_ups) = (0usize, 0usize);
        for turn in 0..10u32 {
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            for &c in &order {
                let (key, scale) = conversations[c];
                let frontend = (rng.next_u64() % frontends as u64) as usize;
                let charge = scale * f64::from(turn + 1);
                let (idx, _) = trackers[frontend]
                    .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                    .unwrap();
                if turn > 0 {
                    follow_ups += 1;
                    changes += usize::from(idx != last[c]);
                }
                last[c] = idx;
            }
        }
        changes as f64 / follow_ups as f64
    }

    /// R3-2: heavy-tailed request sizes must not make independent frontends bounce
    /// balanced conversations between sets.
    #[test]
    fn lognormal_charges_keep_conversations_on_their_set() {
        for (sigma, frontends) in [(1.5, 10), (2.0, 10), (1.5, 30), (2.0, 30)] {
            let rate = follow_up_set_changes(sigma, frontends, 0x5eed);
            assert!(
                rate < 0.005,
                "sigma {sigma}: {:.3}% of follow-ups changed set",
                rate * 100.0
            );
        }
    }

    /// The noise allowance does not disable the guard: a real skew under heavy-tailed sizes
    /// (TP2-preferring conversations twice as large, demand ≈ 0.8) is still corrected.
    #[test]
    fn lognormal_skew_is_still_corrected() {
        for sigma in [1.5, 2.0] {
            let mut rng = Rng(0xabc);
            let tracker = ShareTracker::default();
            for i in 0..30_000u64 {
                let key = mixed(i);
                let skew = if rendezvous_pick(key, &SETS) == Some(1) {
                    2.0
                } else {
                    1.0
                };
                let charge = 2000.0 * (sigma * rng.normal()).exp() * skew;
                tracker
                    .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                    .unwrap();
            }
            assert!(tracker.spill_fraction_of(1, &SETS, DEFAULT_SLACK) > 0.0);
            let tp2 = tracker.share(TP2);
            let cap = 2.0 / 3.0 + share_band(2.0 / 3.0, DEFAULT_SLACK) + 0.03;
            assert!(tp2 <= cap, "sigma {sigma}: tp2 share {tp2}");
        }
    }

    // -- R2-1: key-deterministic sticky spill --

    /// A tracker whose decayed statistics are given directly: fair weight split, demand
    /// split, and placed split (fractions of `total` load).
    fn tracker_with(demand_tp2: f64, placed_tp2: f64) -> ShareTracker {
        let total = 1000.0;
        let map = |tp2: f64| {
            HashMap::from([
                (TP4.to_string(), total * (1.0 - tp2)),
                (TP2.to_string(), total * tp2),
            ])
        };
        ShareTracker {
            state: Mutex::new(ShareState {
                decisions: total,
                charge_sum: total,
                // Effectively noiseless statistics, so `p` is exactly the demand formula.
                charge_sq_sum: 1.0,
                expected: map(2.0 / 3.0),
                demand: map(demand_tp2),
                placed: map(placed_tp2),
                ..Default::default()
            }),
        }
    }

    #[test]
    fn spill_is_a_pure_function_of_key_at_fixed_fraction() {
        // Demand 0.8 on TP2 against fair 2/3: p = 1 − (2/3 + band/2) / 0.8 ≈ 0.115, and
        // the placed split is inside the band, so no overflow.
        let tracker = tracker_with(0.8, 0.70);
        let p = tracker.spill_fraction_of(1, &SETS, DEFAULT_SLACK);
        assert!((0.10..0.13).contains(&p), "p {p}");
        let mut spilled = 0;
        let mut tp2_keys = 0;
        for key in 0..20_000u64 {
            let key = mixed(key);
            let ranking = rendezvous_ranking(key, &SETS);
            let expected = if SETS[ranking[0]].0 == TP2 && spill_point(key) < p {
                (ranking[1], SetChoiceReason::ShareCapFallback)
            } else {
                (ranking[0], SetChoiceReason::Affinity)
            };
            for _ in 0..3 {
                assert_eq!(tracker.peek(key, &SETS, DEFAULT_SLACK), expected);
            }
            if SETS[ranking[0]].0 == TP2 {
                tp2_keys += 1;
                spilled += usize::from(expected.0 != ranking[0]);
            }
        }
        let fraction = spilled as f64 / tp2_keys as f64;
        assert!((fraction - p).abs() < 0.01, "spilled {fraction} vs p {p}");
    }

    #[test]
    fn trackers_with_different_histories_agree_outside_the_spill_gap() {
        let a = tracker_with(0.80, 0.70);
        let b = tracker_with(0.78, 0.71);
        let pa = a.spill_fraction_of(1, &SETS, DEFAULT_SLACK);
        let pb = b.spill_fraction_of(1, &SETS, DEFAULT_SLACK);
        let (lo, hi) = (pa.min(pb), pa.max(pb));
        assert!(hi > lo);
        let mut disagreements = 0;
        for key in 0..20_000u64 {
            let key = mixed(key);
            let (ca, cb) = (
                a.peek(key, &SETS, DEFAULT_SLACK),
                b.peek(key, &SETS, DEFAULT_SLACK),
            );
            let x = spill_point(key);
            if x < lo || x >= hi {
                assert_eq!(ca, cb, "key {key} at spill point {x}");
            } else if ca != cb {
                disagreements += 1;
            }
        }
        // Only TP2-preferring keys inside the gap may differ.
        assert!(disagreements as f64 <= 20_000.0 * (hi - lo) * 0.7 + 50.0);
    }

    /// Two live trackers fed different request streams from the same skewed workload
    /// converge to nearly the same spill fraction and agree on almost every key.
    #[test]
    fn live_trackers_converge_and_agree() {
        let feed = |tracker: &ShareTracker, salt: u64| {
            for i in 0..30_000u64 {
                let key = mixed(i ^ salt);
                // TP2-preferring conversations are twice as large: TP2 demand ≈ 0.8.
                let charge = if rendezvous_pick(key, &SETS) == Some(1) {
                    2.0
                } else {
                    1.0
                };
                tracker
                    .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                    .unwrap();
            }
        };
        let (a, b) = (ShareTracker::default(), ShareTracker::default());
        feed(&a, 0x1111);
        feed(&b, 0x2222_0000);
        let pa = a.spill_fraction_of(1, &SETS, DEFAULT_SLACK);
        let pb = b.spill_fraction_of(1, &SETS, DEFAULT_SLACK);
        assert!(pa > 0.05 && pb > 0.05, "pa {pa} pb {pb}");
        assert!((pa - pb).abs() < 0.05, "pa {pa} pb {pb}");
        let probe = 20_000u64;
        let agree = (0..probe)
            .filter(|k| {
                let key = mixed(k + 1_000_000);
                a.peek(key, &SETS, DEFAULT_SLACK) == b.peek(key, &SETS, DEFAULT_SLACK)
            })
            .count();
        assert!(agree as f64 >= probe as f64 * 0.95, "agree {agree}/{probe}");
        for t in [&a, &b] {
            let tp2 = t.share(TP2);
            assert!(
                tp2 <= 2.0 / 3.0 + share_band(2.0 / 3.0, DEFAULT_SLACK) + 0.02,
                "{tp2}"
            );
        }
    }

    /// Spills stick per key across turns: with a stable workload, a conversation keeps its
    /// set on every turn (keys right at the spill boundary excepted).
    #[test]
    fn spills_stick_across_turns() {
        let tracker = ShareTracker::default();
        let charge_of = |key: u64| {
            if rendezvous_pick(key, &SETS) == Some(1) {
                2.0
            } else {
                1.0
            }
        };
        for i in 0..20_000u64 {
            let key = mixed(i);
            tracker
                .choose(SetAffinity::new(key, charge_of(key)), &SETS, DEFAULT_SLACK)
                .unwrap();
        }
        let p = tracker.spill_fraction_of(1, &SETS, DEFAULT_SLACK);
        assert!(p > 0.05, "p {p}");
        let conversations: Vec<u64> = (0..200u64).map(|c| mixed(c + 5_000_000)).collect();
        let mut first: HashMap<u64, usize> = HashMap::new();
        let mut flips = 0;
        let mut spilled = 0;
        let mut overflows = 0;
        for turn in 0..20u64 {
            for (c, &conv) in conversations.iter().enumerate() {
                // Background traffic between turns.
                for j in 0..10u64 {
                    let key = mixed(turn * 1_000_003 + c as u64 * 101 + j + 9_000_000);
                    tracker
                        .choose(SetAffinity::new(key, charge_of(key)), &SETS, DEFAULT_SLACK)
                        .unwrap();
                }
                let (idx, reason) = tracker
                    .choose(
                        SetAffinity::new(conv, charge_of(conv)),
                        &SETS,
                        DEFAULT_SLACK,
                    )
                    .unwrap();
                // Ordinary conversations are never heavy, so the overflow step never
                // touches them.
                if matches!(
                    reason,
                    SetChoiceReason::ShareCapOverflow | SetChoiceReason::ShareCapRevert
                ) {
                    overflows += 1;
                }
                if turn == 0 && reason == SetChoiceReason::ShareCapFallback {
                    spilled += 1;
                }
                let near_boundary = (spill_point(conv) - p).abs() < 0.05;
                match first.get(&conv) {
                    None => {
                        first.insert(conv, idx);
                    }
                    Some(&prev) if prev != idx && !near_boundary => flips += 1,
                    _ => {}
                }
            }
        }
        assert!(spilled > 0);
        assert_eq!(flips, 0);
        assert_eq!(overflows, 0);
    }

    // -- Affinity keys over the real request types --

    fn chat(body: serde_json::Value) -> NvCreateChatCompletionRequest {
        serde_json::from_value(body).expect("chat request")
    }

    fn chat_key(body: serde_json::Value) -> Option<u64> {
        let req = chat(body);
        chat_request_affinity_key(&req.inner.messages, chat_prompt_cache_key(&req), None)
    }

    fn responses(body: serde_json::Value) -> (NvCreateChatCompletionRequest, Option<String>) {
        let req: NvCreateResponse = serde_json::from_value(body).expect("responses request");
        let cache_key = req.inner.prompt_cache_key.clone();
        (req.try_into().expect("responses conversion"), cache_key)
    }

    fn responses_key(body: serde_json::Value) -> Option<u64> {
        let (req, cache_key) = responses(body);
        chat_request_affinity_key(&req.inner.messages, cache_key.as_deref(), None)
    }

    fn user_item(text: &str) -> serde_json::Value {
        json!({"type": "message", "role": "user",
               "content": [{"type": "input_text", "text": text}]})
    }

    const AGENTS_MD: &str = "# AGENTS.md instructions for /repo\n\nRun the tests.";
    const ENV_CONTEXT: &str = "<environment_context><cwd>/repo</cwd></environment_context>";

    fn codex_turn1(task: &str) -> serde_json::Value {
        json!({
            "model": "m",
            "instructions": "You are Codex.",
            "input": [user_item(AGENTS_MD), user_item(ENV_CONTEXT), user_item(task)],
        })
    }

    fn codex_turn2(task: &str) -> serde_json::Value {
        json!({
            "model": "m",
            "instructions": "You are Codex.",
            "input": [
                user_item(AGENTS_MD),
                user_item(ENV_CONTEXT),
                user_item(task),
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "function_call", "call_id": "c1", "name": "shell",
                 "arguments": "{\"cmd\":[\"ls\"]}"},
                {"type": "function_call_output", "call_id": "c1", "output": "README.md"},
                {"type": "message", "role": "assistant", "id": "m1",
                 "content": [{"type": "output_text", "text": "Done."}]},
                user_item("now run the tests"),
            ],
        })
    }

    /// A2: sessions from one harness share their leading user items; the key must still
    /// separate them by task.
    #[test]
    fn responses_sessions_with_shared_boilerplate_get_different_keys() {
        let a = responses_key(codex_turn1("fix the parser")).unwrap();
        let b = responses_key(codex_turn1("add a CLI flag")).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn responses_key_is_stable_across_turns() {
        let t1 = responses_key(codex_turn1("fix the parser")).unwrap();
        let t2 = responses_key(codex_turn2("fix the parser")).unwrap();
        assert_eq!(t1, t2);
    }

    /// Turn 1 sent as `input: "text"` and turn 2 as an item list must agree. This relies on
    /// the converter collapsing a single `input_text` part to plain text.
    #[test]
    fn responses_text_input_matches_item_list_follow_up() {
        let t1 = responses_key(json!({
            "model": "m", "instructions": "sys", "input": "fix the parser",
        }))
        .unwrap();
        let t2_items = responses_key(json!({
            "model": "m", "instructions": "sys",
            "input": [
                user_item("fix the parser"),
                {"type": "message", "role": "assistant", "id": "m1",
                 "content": [{"type": "output_text", "text": "ok"}]},
                user_item("more"),
            ],
        }))
        .unwrap();
        let t2_easy = responses_key(json!({
            "model": "m", "instructions": "sys",
            "input": [
                {"role": "user", "content": "fix the parser"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "more"},
            ],
        }))
        .unwrap();
        assert_eq!(t1, t2_items);
        assert_eq!(t1, t2_easy);
    }

    #[test]
    fn chat_key_is_stable_across_turns() {
        let t1 = chat_key(json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": AGENTS_MD},
                {"role": "user", "content": "task A"},
            ],
        }))
        .unwrap();
        let t2 = chat_key(json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": AGENTS_MD},
                {"role": "user", "content": "task A"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "shell", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "more"},
            ],
        }))
        .unwrap();
        let other = chat_key(json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": AGENTS_MD},
                {"role": "user", "content": "task B"},
            ],
        }))
        .unwrap();
        assert_eq!(t1, t2);
        assert_ne!(t1, other);
    }

    #[test]
    fn prompt_cache_key_overrides_message_hash() {
        let mut a = codex_turn1("fix the parser");
        a["prompt_cache_key"] = json!("conv-123");
        let mut b = codex_turn1("something else entirely");
        b["prompt_cache_key"] = json!("conv-123");
        let mut c = codex_turn1("fix the parser");
        c["prompt_cache_key"] = json!("conv-456");
        let (ka, kb, kc) = (
            responses_key(a).unwrap(),
            responses_key(b).unwrap(),
            responses_key(c).unwrap(),
        );
        assert_eq!(ka, kb);
        assert_ne!(ka, kc);
        assert_ne!(ka, responses_key(codex_turn1("fix the parser")).unwrap());

        // The Chat Completions extra-body field is honored the same way.
        let chat_a = chat_key(json!({"model": "m", "prompt_cache_key": "conv-123",
            "messages": [{"role": "user", "content": "x"}]}));
        let chat_b = chat_key(json!({"model": "m", "prompt_cache_key": "conv-123",
            "messages": [{"role": "user", "content": "y"}]}));
        assert_eq!(chat_a, chat_b);
        assert_eq!(chat_a, Some(ka));
    }

    #[test]
    fn session_affinity_header_beats_messages_but_not_prompt_cache_key() {
        let req = chat(json!({"model": "m", "messages": [{"role": "user", "content": "x"}]}));
        let msgs = &req.inner.messages;
        let by_messages = chat_request_affinity_key(msgs, None, None).unwrap();
        let by_session = chat_request_affinity_key(msgs, None, Some("s-1")).unwrap();
        assert_ne!(by_session, by_messages);
        assert_eq!(
            by_session,
            chat_request_affinity_key(&[], None, Some("s-1")).unwrap()
        );
        let by_cache = chat_request_affinity_key(msgs, Some("s-1"), Some("s-1")).unwrap();
        assert_eq!(
            by_cache,
            chat_request_affinity_key(msgs, Some("s-1"), None).unwrap()
        );
        // Same string under a different source is a different key.
        assert_ne!(by_cache, by_session);
        // Blank explicit keys fall through.
        assert_eq!(
            chat_request_affinity_key(msgs, Some(" "), Some("")),
            Some(by_messages)
        );
    }

    /// Requests whose opening has no user message get no key and take the random pick.
    #[test]
    fn requests_without_a_user_opening_have_no_key() {
        assert_eq!(
            chat_key(json!({"model": "m",
                "messages": [{"role": "system", "content": "sys"}]})),
            None
        );
        assert_eq!(
            chat_key(json!({"model": "m", "messages": [
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "hi"},
            ]})),
            None
        );
        assert_eq!(chat_request_affinity_key(&[], None, None), None);
        // An explicit key still applies.
        assert!(chat_request_affinity_key(&[], Some("conv"), None).is_some());
    }

    // -- R2-3: canonical text --

    #[test]
    fn chat_text_parts_hash_like_a_plain_string() {
        let plain = chat_key(json!({"model": "m", "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "fix the parser"},
        ]}));
        let parts = chat_key(json!({"model": "m", "messages": [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [
                {"type": "text", "text": "fix the "},
                {"type": "text", "text": "parser"},
            ]},
        ]}));
        assert!(plain.is_some());
        assert_eq!(plain, parts);
        // Role structure still matters: the same text split across two user messages
        // is a different opening.
        let split = chat_key(json!({"model": "m", "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "fix the "},
            {"role": "user", "content": "parser"},
        ]}));
        assert_ne!(plain, split);
    }

    #[test]
    fn chat_and_responses_openings_hash_alike() {
        let responses_single = responses_key(json!({
            "model": "m", "instructions": "sys", "input": [user_item("fix the parser")],
        }));
        let responses_multi = responses_key(json!({
            "model": "m", "instructions": "sys",
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "fix the "},
                {"type": "input_text", "text": "parser"},
            ]}],
        }));
        let chat_array = chat_key(json!({"model": "m", "messages": [
            {"role": "developer", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "fix the parser"}]},
        ]}));
        let chat_plain = chat_key(json!({"model": "m", "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "fix the parser"},
        ]}));
        assert!(chat_plain.is_some());
        assert_eq!(responses_single, chat_plain);
        assert_eq!(responses_multi, chat_plain);
        assert_eq!(chat_array, chat_plain);
    }

    #[test]
    fn non_text_parts_keep_their_identity() {
        let with_image = |url: &str| {
            chat_key(
                json!({"model": "m", "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": url}},
                ]}]}),
            )
        };
        let a = with_image("https://example.com/a.png");
        assert!(a.is_some());
        assert_ne!(a, with_image("https://example.com/b.png"));
        let text_only = chat_key(json!({"model": "m", "messages": [
            {"role": "user", "content": "what is this"},
        ]}));
        assert_ne!(a, text_only);
    }

    /// R3-4: the Responses conversion merges leading system/developer messages with
    /// "\n\n"; the key applies the same merge to both APIs.
    #[test]
    fn leading_instructions_merge_like_the_responses_conversion() {
        let chat_split = chat_key(json!({"model": "m", "messages": [
            {"role": "system", "content": "sys"},
            {"role": "developer", "content": [
                {"type": "text", "text": "de"},
                {"type": "text", "text": "v"},
            ]},
            {"role": "user", "content": "task 0"},
        ]}));
        let chat_merged = chat_key(json!({"model": "m", "messages": [
            {"role": "system", "content": "sys\n\ndev"},
            {"role": "user", "content": "task 0"},
        ]}));
        let responses = responses_key(json!({
            "model": "m", "instructions": "sys",
            "input": [
                {"type": "message", "role": "developer",
                 "content": [{"type": "input_text", "text": "dev"}]},
                user_item("task 0"),
            ],
        }));
        assert!(chat_split.is_some());
        assert_eq!(chat_split, responses);
        assert_eq!(chat_split, chat_merged);
        // A system message after the first user message is not merged.
        let late = chat_key(json!({"model": "m", "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task 0"},
            {"role": "developer", "content": "dev"},
        ]}));
        assert_ne!(late, chat_split);
    }

    // -- R2-2: size proxy --

    #[test]
    fn request_charge_counts_message_text() {
        let req = chat(json!({"model": "m", "messages": [
            {"role": "system", "content": "0123456789"},
            {"role": "user", "content": [
                {"type": "text", "text": "01234"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ]},
            {"role": "assistant", "content": "01234", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "sh", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "0123456789"},
        ]}));
        let expected = 10 + 5 + NON_TEXT_PART_CHARGE + 5 + 2 + 2 + 10;
        assert_eq!(request_charge(&req.inner.messages), expected as f64);
        assert_eq!(request_charge(&[]), 1.0);
        let big = chat(json!({"model": "m", "messages": [
            {"role": "user", "content": "x".repeat(10_000)},
        ]}));
        assert_eq!(request_charge(&big.inner.messages), 10_000.0);
        assert_eq!(SetAffinity::new(1, f64::NAN).charge, 1.0);
        assert_eq!(SetAffinity::new(1, 1e300).charge, MAX_REQUEST_CHARGE);
    }

    /// R3-5: developer content is charged by its text bytes (no JSON overhead or escaping),
    /// the same as plain text, part arrays, and the converted Responses request.
    #[test]
    fn developer_charge_counts_text_bytes() {
        let quotes = "\"".repeat(1000);
        let charge = |body: serde_json::Value| request_charge(&chat(body).inner.messages);
        let plain = charge(json!({"model": "m", "messages": [
            {"role": "developer", "content": quotes},
        ]}));
        let parts = charge(json!({"model": "m", "messages": [
            {"role": "developer", "content": [
                {"type": "text", "text": &quotes[..400]},
                {"type": "text", "text": &quotes[400..]},
            ]},
        ]}));
        let system = charge(json!({"model": "m", "messages": [
            {"role": "system", "content": [{"type": "text", "text": quotes}]},
        ]}));
        let (converted, _) = responses(json!({
            "model": "m",
            "input": [{"type": "message", "role": "developer", "content": [
                {"type": "input_text", "text": &quotes[..400]},
                {"type": "input_text", "text": &quotes[400..]},
            ]}],
        }));
        assert_eq!(plain, 1000.0);
        assert_eq!(parts, 1000.0);
        assert_eq!(system, 1000.0);
        assert_eq!(request_charge(&converted.inner.messages), 1000.0);
    }
}
