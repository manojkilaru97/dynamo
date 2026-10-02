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
//! Each frontend also keeps decayed load statistics, by size (charged by
//! [`request_charge`], a byte-count proxy for prompt size) and by request count:
//!
//! 1. **Heavy keys bypass affinity.** A key is heavy only when it alone could push its
//!    set out of band, which whole-key spilling cannot absorb. A Space-Saving table of 128
//!    counters (with per-counter error bounds; a counter is taken over only when its larger
//!    share is the smallest, so a key that is rare but carries much of the load keeps its
//!    counter between requests) tracks each key's decayed request count and unclamped
//!    charge. A key becomes heavy when EITHER its guaranteed request share (with at least 8
//!    guaranteed decayed requests in the window) OR its guaranteed share of unclamped charge
//!    (with at least 3), less four standard deviations of that share's sampling noise,
//!    reaches the narrowest share band among the candidate sets
//!    (`min_i slack·min(fair_i, 1 − fair_i)`, about 0.083 for 30/60 workers and slack 0.25).
//!    A lopsided key, whose share of one measure is at least four times its share of the
//!    other, is heavy from half the band: moving it barely changes the other measure, so
//!    a few of them can push a set out of band on one measure that spilling (it moves both
//!    measures in proportion) cannot fix without pushing the other out. A heavy key stays
//!    heavy until both branches fail at half the thresholds and floors. The request share
//!    catches many small requests; the unclamped share catches very large ones that repeat
//!    as rarely as once per few hundred requests (the charge clamp below applies only to
//!    the spill statistics). Heavy keys get the default weighted random pick
//!    (`heavy_key_random`): a key that hot is cached on every set anyway, and random
//!    routing balances it by construction. Ordinary conversations do not qualify, also on
//!    a frontend that sees only a few dozen concurrent conversations (with 12 or more
//!    similar conversations each carries at most about 1/12 of requests and load, which
//!    does not exceed the band; at exactly 12, a rare excursion of more than four standard
//!    deviations can still admit one), and neither does a single large request (it has no
//!    repetitions). A heavy key that stops qualifying is routed by affinity again and must
//!    meet the full entry threshold to re-enter.
//! 2. **Sticky spill** for the remaining (affinity-routed) load keeps every set within
//!    `fair ± slack·min(fair, 1 − fair)`. When the load or the requests whose rendezvous
//!    winner is set `P` exceed `P`'s fair share, the frontend derives a spill fraction `p`
//!    from the larger excess (zero until demand exceeds the middle of the upper band by
//!    more than two standard deviations of its sampling noise, at most `slack`) and moves
//!    exactly the keys whose [`spill_point`] is below `p` to their second rendezvous choice
//!    (`share_cap_fallback`). Whether a key spills, and where to, is a function of the key,
//!    the candidate set keys and `p`; `p` depends only on the aggregate load mix, so
//!    frontends with similar traffic agree and a conversation's turns stay on one set while
//!    `p` is stable. When a key becomes heavy, exactly what it contributed to the current
//!    window while affinity-routed leaves the spill statistics.
//! 3. **Routed-share feedback.** The demand-based `p` assumes load is spread evenly over
//!    keys; a persistent pool of a few conversations whose spill points all exceed `slack`
//!    defeats it. Each frontend therefore also tracks the load and requests each set
//!    actually received, and while a set stays above its band on either (beyond sampling
//!    noise) it raises that set's `p` beyond `slack`, up to 1 (an integral term, about the
//!    excess per window), moving further keys in spill-point order. The boost holds while
//!    the set is between its fair share and its band edge (each widened by the same
//!    noise allowance), shrinks once the set is below fair share on both measures beyond
//!    that allowance, and leaks away once the set's demand alone (what it would receive
//!    with no spill) is clearly within band. The deadband is wider than any non-heavy key,
//!    so whole-key moves settle instead of oscillating, also when one lands exactly on
//!    fair share. A balanced workload never engages it. The boost steps only on decisions
//!    that update the window, and is dropped whenever the window is cold.
//!
//! The window covers the last ~1000 affinity-routed decisions: only those decay it, so
//! heavy-key traffic in between neither ages its shares nor keeps it from warming up
//! (with 85% of requests from one heavy key, the guards still see the remaining 15%).
//! "Fair" is the decayed average of each set's weight share over the same window, so a
//! small worker-count drift shifts the targets as gradually as the observed load and only
//! the keys rendezvous hashing itself moves change sets. The window (demand, fair shares,
//! routed shares and boosts) restarts when the candidate sets change, when any set's
//! weight share has moved by more than 0.02 since the last restart (a capacity change:
//! old targets and boosts would keep spilling toward a set that just shrank; drifts that
//! arrive one worker at a time accumulate against that baseline), or when a selection
//! found fewer than two eligible sets, so a set that returns at a new size is not judged
//! against a stale fair share. Every restart re-baselines the capacity threshold.
//!
//! Charges in the spill statistics are clamped to 8× a decayed mean charge (seeded with a
//! 4 KiB prior), and the guard stays idle until its window holds about 200 decisions
//! (heavy-key detection starts after 100 decisions of any kind, so a hot key present from
//! the start leaves the statistics before the guard acts on them), so one huge request or
//! a freshly started frontend cannot swing them. Heavy-key detection uses the unclamped
//! charge.
//!
//! `DYN_WORKER_SET_WEIGHTS=suffix=weight,...` scales the per-worker weight of every set whose
//! namespace ends with `suffix` (for example `tp4=2,tp2=1` to weight sets by GPUs). Entries
//! must lie in `[1e-6, 1e6]`; anything else is ignored with a warning.
//! `DYN_WORKER_SET_AFFINITY_SLACK` (default 0.25) is a fraction clamped to `[0.05, 0.5]`;
//! a value above 1 (a percentage such as `25`) is rejected in favour of the default.
//!
//! Assumptions and limitations:
//!
//! - **Request size is bounded.** A request's charge is at most the model's maximum
//!   context, so its ratio to the mean charge is bounded by max context / mean context
//!   (for Nemotron 3 Super, 262144 tokens against a mean of about 31k: about 8.4x, close to
//!   the 8x clamp). Keys whose requests are hundreds of times the mean and that repeat
//!   less than about once per 300 requests (fewer than 3 decayed observations) are not
//!   classified heavy, and their bytes are hidden from the spill statistics by the clamp;
//!   such traffic is outside what this policy balances.
//! - **No overload fallback.** Affinity picks a set before that set's KV router looks at
//!   worker load. When busy thresholds are configured and every worker of the chosen set
//!   is overloaded, the request fails (`AllEligibleWorkersOverloaded`, HTTP 529) even if
//!   the other set has capacity, and a retry of the same conversation goes to the same
//!   set. The random policy would have sent a retry elsewhere with probability equal to
//!   the other sets' weight share. Prefer not to combine affinity with busy thresholds.

use std::collections::HashMap;
use std::io;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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
/// Largest accepted slack. Above it the bands are so wide that the guards barely act (at
/// slack 1 TP4 could take two thirds of the load); a value above 1 is almost certainly a
/// percentage (`25` meaning 0.25) and is rejected.
pub const MAX_SLACK: f64 = 0.5;
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
/// Decayed decisions required before a key may be classified heavy: half the guard's
/// warm-up, so a hot key present from the start is classified (and its load leaves the
/// window) before the guard first acts on that load.
const HEAVY_MIN_SAMPLES: f64 = SHARE_MIN_SAMPLES / 2.0;
/// A recorded charge is clamped to this multiple of the decayed mean charge.
const CHARGE_CLAMP_FACTOR: f64 = 8.0;
/// Standard deviations of sampling noise in the demand share tolerated before spilling.
const SPILL_NOISE_SIGMAS: f64 = 2.0;
/// Size of the per-frontend Space-Saving heavy-key table. A counter is taken over only
/// when its larger share (requests or unclamped charge) is the smallest in the table, and
/// those shares sum to at most 2, so any key whose request share or charge share exceeds
/// `2/HEAVY_KEYS` (1.6%) is guaranteed a counter. That is below the exit thresholds for
/// bands of at least 6.3% (production: 8.3%); for narrower bands (small slack, or a small
/// third set) detection of keys between the two is best-effort.
const HEAVY_KEYS: usize = 128;
/// Heavy-key thresholds (see [`ShareState::heavy_status`]). A key becomes heavy when its
/// guaranteed request share (with at least [`HEAVY_MIN_OBSERVATIONS`] guaranteed decayed
/// requests) or its guaranteed share of unclamped charge (with at least
/// [`HEAVY_MIN_CHARGE_OBSERVATIONS`]), less that share's sampling noise, reaches the
/// narrowest share band among the candidate sets (`min_i slack·min(fair_i, 1 − fair_i)`),
/// or reaches [`HEAVY_LOPSIDED_BAND_FRACTION`] of it while its other share is at most
/// `1/HEAVY_LOPSIDED_RATIO` of this one. It stays heavy until every branch fails at
/// [`HEAVY_EXIT_LOAD_RATIO`] times the thresholds and observation floors (and half the
/// lopsidedness ratio). The request share catches many small requests; the unclamped
/// charge share catches repeated very large ones (the charge clamp applies only to the
/// spill statistics), and its lower floor admits keys that repeat only once per few
/// hundred requests (the noise term already rejects a lone outlier, whose charge
/// collapses the effective sample size). A single large request cannot qualify.
const HEAVY_MIN_OBSERVATIONS: f64 = 8.0;
const HEAVY_MIN_CHARGE_OBSERVATIONS: f64 = 3.0;
const HEAVY_EXIT_LOAD_RATIO: f64 = 0.5;
/// Lopsided keys: a key that carries far more of one measure than of the other moves
/// that measure without moving the other, which whole-key spilling (it moves both in
/// proportion) cannot balance; a few such keys can push a set out of band on one measure
/// while the other is in band. They are heavy from half the band.
const HEAVY_LOPSIDED_BAND_FRACTION: f64 = 0.5;
const HEAVY_LOPSIDED_RATIO: f64 = 4.0;
/// Integral gain of the routed-share feedback: a set whose routed share stays above its
/// band (plus noise) raises its spill fraction by this times the excess per decision (so
/// by about the excess per window).
const SPILL_BOOST_GAIN: f64 = 1.0 / SHARE_WINDOW;
/// A shrinking boost below this is dropped (it decays roughly exponentially otherwise).
const SPILL_BOOST_FLOOR: f64 = 1e-3;
/// A change of any candidate's weight share by more than this since the window last
/// restarted (for example TP2 60 → 52 workers next to 30 TP4 workers) restarts it.
const MATERIAL_SHARE_CHANGE: f64 = 0.02;
/// Standard deviations of sampling noise subtracted from a key's request or charge share
/// before it is compared with the band, so keys sitting right at the band (for example 12
/// equal conversations against a 1/12 band) do not become heavy on a random excursion.
const HEAVY_NOISE_SIGMAS: f64 = 4.0;
/// Prior for the mean charge (pseudo-decisions and bytes), so early requests are clamped.
const PRIOR_DECISIONS: f64 = 10.0;
const PRIOR_MEAN_CHARGE: f64 = 4096.0;
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
                Ok(v) if v.is_finite() && v > 1.0 => tracing::warn!(
                    value = raw,
                    "{SLACK_ENV} is a fraction (0.25 means 25%); using {DEFAULT_SLACK}"
                ),
                Ok(v) if v.is_finite() && v > MAX_SLACK => {
                    tracing::warn!(
                        value = raw,
                        "{SLACK_ENV} above {MAX_SLACK}; using {MAX_SLACK}"
                    );
                    config.slack = MAX_SLACK;
                }
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
    /// A heavy key, routed by the weighted random pick instead of affinity.
    HeavyKeyRandom,
    /// No affinity key (or affinity disabled): weighted random pick.
    Random,
}

impl SetChoiceReason {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Affinity => "affinity",
            Self::ShareCapFallback => "share_cap_fallback",
            Self::HeavyKeyRandom => "heavy_key_random",
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
    weighted_pick(weights, rand::rng().random::<f64>())
}

/// [`weighted_random_pick`] driven by `uniform`, a sample from `[0, 1)`.
fn weighted_pick(weights: &[f64], uniform: f64) -> Option<usize> {
    let usable = |w: f64| w > 0.0 && w.is_finite();
    let total: f64 = weights.iter().copied().filter(|w| usable(*w)).sum();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let mut pick = uniform * total;
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

/// Effective number of samples `sum² / sq_sum` of decayed weights (0 without samples).
fn effective_samples(sum: f64, sq_sum: f64) -> f64 {
    if sq_sum > 0.0 {
        sum * sum / sq_sum
    } else {
        0.0
    }
}

/// Standard deviation of a share estimated from `samples` effective samples (at least 1).
fn sampling_sd(share: f64, samples: f64) -> f64 {
    (share * (1.0 - share) / samples.max(1.0)).sqrt()
}

/// [`HEAVY_NOISE_SIGMAS`] standard deviations of a decayed weighted share `share`, from
/// the effective number of samples of the weights behind it.
fn share_noise(share: f64, sum: f64, sq_sum: f64) -> f64 {
    HEAVY_NOISE_SIGMAS * sampling_sd(share, effective_samples(sum, sq_sum))
}

/// The narrowest share band among the candidates in `ranking`: the most load one key may
/// carry before it alone could push its set out of band.
fn narrowest_band(candidates: &[(&str, f64)], ranking: &[usize], slack: f64) -> f64 {
    let total: f64 = ranking.iter().map(|&i| candidates[i].1).sum();
    ranking
        .iter()
        .map(|&i| share_band(candidates[i].1 / total, slack))
        .fold(f64::INFINITY, f64::min)
}

/// Fraction of a set's keys to spill so that, if load were spread evenly over keys, its
/// share would come down to the middle of its upper band plus `noise` (the sampling
/// uncertainty of the demand share, so that random fluctuations of a balanced workload do
/// not spill keys back and forth). Zero until the demand share exceeds that point; never
/// above `slack`.
fn spill_fraction(demand_share: f64, fair: f64, band: f64, slack: f64, noise: f64) -> f64 {
    let target = fair + band / 2.0 + noise;
    if demand_share <= target {
        return 0.0;
    }
    (1.0 - target / demand_share).clamp(0.0, slack)
}

/// One Space-Saving counter of the heavy-key table. `raw_err` and `count_err` are the
/// counter's unclamped charge and request count when the key took the slot over, so
/// `raw - raw_err` and `count - count_err` (what the key sent since) are guaranteed lower
/// bounds of its decayed charge and request count. `raw` and `count` are upper bounds only
/// while the key has held the slot since it first appeared (eviction is by the larger
/// share, not by the smallest count).
#[derive(Debug, Clone)]
struct HeavyCounter {
    key: u64,
    raw: f64,
    raw_err: f64,
    count: f64,
    count_err: f64,
    /// Current heavy status (with hysteresis).
    heavy: bool,
    /// What this key contributed to the current fair-share window while affinity-routed,
    /// decayed with the window: the clamped charge and the requests it credited to each
    /// set as demand (its winner at the time; a winner change keeps the earlier set's
    /// credit), the fair charge and fair requests it credited to each candidate set (at
    /// the weights of the time), the charge and requests it placed on each set (winner or
    /// spill target), all by set-key hash, and its squared charges, decisions and squared
    /// decision weights. Cleared on window reset and once subtracted when the key becomes
    /// heavy; inherited by a key that takes the counter over, so the window always equals
    /// the sum of all counters' credits.
    window_demand: Vec<(u64, f64)>,
    window_expected: Vec<(u64, f64)>,
    window_demand_requests: Vec<(u64, f64)>,
    window_expected_requests: Vec<(u64, f64)>,
    window_routed: Vec<(u64, f64)>,
    window_routed_requests: Vec<(u64, f64)>,
    window_sq: f64,
    window_decisions: f64,
    window_decisions_sq: f64,
}

/// Add `amount` to `id`'s entry of a per-set credit list.
fn add_credit(credits: &mut Vec<(u64, f64)>, id: u64, amount: f64) {
    match credits.iter_mut().find(|(s, _)| *s == id) {
        Some((_, c)) => *c += amount,
        None => credits.push((id, amount)),
    }
}

impl HeavyCounter {
    fn new(key: u64, raw: f64, raw_err: f64, count: f64, count_err: f64) -> Self {
        Self {
            key,
            raw,
            raw_err,
            count,
            count_err,
            heavy: false,
            window_demand: Vec::new(),
            window_expected: Vec::new(),
            window_demand_requests: Vec::new(),
            window_expected_requests: Vec::new(),
            window_routed: Vec::new(),
            window_routed_requests: Vec::new(),
            window_sq: 0.0,
            window_decisions: 0.0,
            window_decisions_sq: 0.0,
        }
    }

    fn clear_window_credit(&mut self) {
        for list in self.credit_lists_mut() {
            list.clear();
        }
        self.window_sq = 0.0;
        self.window_decisions = 0.0;
        self.window_decisions_sq = 0.0;
    }

    fn credit_lists_mut(&mut self) -> [&mut Vec<(u64, f64)>; 6] {
        [
            &mut self.window_demand,
            &mut self.window_expected,
            &mut self.window_demand_requests,
            &mut self.window_expected_requests,
            &mut self.window_routed,
            &mut self.window_routed_requests,
        ]
    }
}

/// Identity of a set key in a counter's per-set credit lists.
fn set_id(name: &str) -> u64 {
    xxh3_64_with_seed(name.as_bytes(), 0)
}

/// Decayed per-frontend statistics behind affinity decisions.
#[derive(Debug)]
struct ShareState {
    /// Decayed number of keyed decisions (warm-up gate, heavy-key count shares).
    decisions: f64,
    /// Decayed sum of recorded (clamped) charges (mean charge for the clamp).
    charge_sum: f64,
    /// Decayed sum of unclamped charges and of their squares (heavy-key charge shares and
    /// their effective sample size).
    raw_sum: f64,
    raw_sq_sum: f64,
    /// Decayed sum of squared decision weights (effective sample size of request shares).
    decisions_sq: f64,
    /// Decayed weight of the mean-charge prior (starts at [`PRIOR_DECISIONS`]).
    prior_weight: f64,
    /// Space-Saving heavy-key table, at most [`HEAVY_KEYS`] counters.
    heavy: Vec<HeavyCounter>,
    /// Fair-share window over affinity-routed (non-heavy) decisions; reset when the
    /// candidate sets change. Decayed decision count, charge sum and squared-charge sum.
    window_decisions: f64,
    window_sum: f64,
    window_sq_sum: f64,
    /// Decayed sum of squared decision weights of the window (request-share noise).
    window_decisions_sq: f64,
    /// Load times each set's weight share at decision time: the decayed "fair" load.
    expected: HashMap<String, f64>,
    /// Load whose rendezvous winner was the set.
    demand: HashMap<String, f64>,
    /// The same two for request counts (each decision weighs 1).
    expected_requests: HashMap<String, f64>,
    demand_requests: HashMap<String, f64>,
    /// Load and requests the set actually received (winner or spill target), for the
    /// routed-share feedback.
    routed: HashMap<String, f64>,
    routed_requests: HashMap<String, f64>,
    /// Per-set integral term added to the spill fraction (routed-share feedback).
    boost: HashMap<String, f64>,
    /// Candidate set keys of the last recorded decision.
    candidates: Vec<String>,
    /// Each candidate's weight share when the window last restarted (same order as
    /// `candidates`).
    baseline_shares: Vec<f64>,
    /// Source of the heavy-key weighted random picks (seeded in tests, so they are
    /// deterministic).
    rng: StdRng,
}

impl Default for ShareState {
    fn default() -> Self {
        Self {
            decisions: 0.0,
            charge_sum: 0.0,
            raw_sum: 0.0,
            raw_sq_sum: 0.0,
            decisions_sq: 0.0,
            prior_weight: PRIOR_DECISIONS,
            heavy: Vec::new(),
            window_decisions: 0.0,
            window_sum: 0.0,
            window_sq_sum: 0.0,
            window_decisions_sq: 0.0,
            expected: HashMap::new(),
            demand: HashMap::new(),
            expected_requests: HashMap::new(),
            demand_requests: HashMap::new(),
            routed: HashMap::new(),
            routed_requests: HashMap::new(),
            boost: HashMap::new(),
            candidates: Vec::new(),
            baseline_shares: Vec::new(),
            rng: if cfg!(test) {
                StdRng::seed_from_u64(0x5eed)
            } else {
                StdRng::from_rng(&mut rand::rng())
            },
        }
    }
}

impl ShareState {
    fn stat(map: &HashMap<String, f64>, name: &str) -> f64 {
        map.get(name).copied().unwrap_or(0.0)
    }

    /// Forget the fair-share window (candidate sets changed).
    fn reset_window(&mut self) {
        self.window_decisions = 0.0;
        self.window_sum = 0.0;
        self.window_sq_sum = 0.0;
        self.window_decisions_sq = 0.0;
        self.expected.clear();
        self.demand.clear();
        self.expected_requests.clear();
        self.demand_requests.clear();
        self.routed.clear();
        self.routed_requests.clear();
        self.boost.clear();
        for c in &mut self.heavy {
            c.clear_window_credit();
        }
    }

    /// Each candidate's share of the total usable weight (0 for an unusable weight).
    fn weight_shares(candidates: &[(&str, f64)]) -> Vec<f64> {
        let usable = |w: f64| w.is_finite() && w > 0.0;
        let total: f64 = candidates
            .iter()
            .map(|(_, w)| *w)
            .filter(|w| usable(*w))
            .sum();
        candidates
            .iter()
            .map(|(_, w)| {
                if usable(*w) && total > 0.0 {
                    *w / total
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Restart the window at `candidates`, which become the baseline for detecting the
    /// next material weight change. Every restart goes through here (membership change,
    /// material weight change, a selection with fewer than two eligible sets), so the
    /// capacity-change threshold is always measured from the latest restart.
    fn restart_window(&mut self, candidates: &[(&str, f64)]) {
        self.reset_window();
        self.candidates = candidates.iter().map(|(n, _)| n.to_string()).collect();
        self.baseline_shares = Self::weight_shares(candidates);
    }

    /// Restart the window if the candidate set keys differ from the last decision's, or
    /// if any candidate's weight share moved by more than [`MATERIAL_SHARE_CHANGE`] since
    /// the window last restarted (a capacity change: the spill targets, boosts and demand
    /// of the old weights would otherwise keep spilling toward a set that just shrank).
    /// Drifts accumulate against that baseline, so a capacity change that arrives one
    /// worker at a time restarts the window too; smaller drifts keep the window, and
    /// fair shares follow them as gradually as the load.
    fn track_candidates(&mut self, candidates: &[(&str, f64)]) {
        let shares = Self::weight_shares(candidates);
        let same = self.candidates.len() == candidates.len()
            && candidates
                .iter()
                .all(|(name, _)| self.candidates.iter().any(|c| c == name));
        let material = !same
            || candidates.iter().zip(&shares).any(|((name, _), share)| {
                let i = self.candidates.iter().position(|c| c == name);
                let baseline = i.and_then(|i| self.baseline_shares.get(i)).copied();
                baseline.is_none_or(|b| (share - b).abs() > MATERIAL_SHARE_CHANGE)
            });
        if material {
            self.restart_window(candidates);
        }
    }

    /// Heavy status of `counter` against the narrowest candidate band `band`, applying
    /// hysteresis to its previous status. Never heavy before [`HEAVY_MIN_SAMPLES`].
    ///
    /// Two branches, one per measure: the request share (observation floor
    /// [`HEAVY_MIN_OBSERVATIONS`]) and the unclamped charge share (floor
    /// [`HEAVY_MIN_CHARGE_OBSERVATIONS`]). A branch qualifies when its guaranteed share,
    /// less [`HEAVY_NOISE_SIGMAS`] of sampling noise, reaches the band, or reaches
    /// [`HEAVY_LOPSIDED_BAND_FRACTION`] of it while the key's other share (upper bound) is
    /// at most `1/HEAVY_LOPSIDED_RATIO` of it. A heavy key stays heavy while some branch
    /// qualifies at [`HEAVY_EXIT_LOAD_RATIO`] times the thresholds and floors, without the
    /// noise term and with half the ratio.
    fn heavy_status(&self, counter: &HeavyCounter, band: f64) -> bool {
        if self.decisions < HEAVY_MIN_SAMPLES || self.raw_sum <= 0.0 || band <= 0.0 {
            return false;
        }
        let observations = counter.count - counter.count_err;
        let requests = observations / self.decisions;
        let charge = (counter.raw - counter.raw_err) / self.raw_sum;
        let requests_max = counter.count / self.decisions;
        let charge_max = counter.raw / self.raw_sum;
        let (scale, ratio, requests_noise, charge_noise) = if counter.heavy {
            (HEAVY_EXIT_LOAD_RATIO, HEAVY_LOPSIDED_RATIO / 2.0, 0.0, 0.0)
        } else {
            (
                1.0,
                HEAVY_LOPSIDED_RATIO,
                share_noise(requests, self.decisions, self.decisions_sq),
                share_noise(charge, self.raw_sum, self.raw_sq_sum),
            )
        };
        let branch = |share: f64, noise: f64, other_max: f64, floor: f64| {
            let share = share - noise;
            observations >= scale * floor
                && (share >= scale * band
                    || (share >= scale * HEAVY_LOPSIDED_BAND_FRACTION * band
                        && other_max * ratio <= share))
        };
        branch(requests, requests_noise, charge_max, HEAVY_MIN_OBSERVATIONS)
            || branch(
                charge,
                charge_noise,
                requests_max,
                HEAVY_MIN_CHARGE_OBSERVATIONS,
            )
    }

    /// How much a counter is worth keeping in the table: its larger share.
    fn importance(&self, counter: &HeavyCounter) -> f64 {
        let requests = counter.count / self.decisions.max(f64::MIN_POSITIVE);
        let charge = counter.raw / self.raw_sum.max(f64::MIN_POSITIVE);
        requests.max(charge)
    }

    /// Whether `key` is currently heavy against the narrowest candidate band `band`.
    fn is_heavy(&self, key: u64, band: f64) -> bool {
        self.heavy
            .iter()
            .find(|c| c.key == key)
            .is_some_and(|c| self.heavy_status(c, band))
    }

    /// `charge` clamped to [`CHARGE_CLAMP_FACTOR`] times the decayed mean charge, where
    /// the mean starts from [`PRIOR_DECISIONS`] pseudo-decisions of [`PRIOR_MEAN_CHARGE`]
    /// (decaying like real decisions), so the first requests of a fresh frontend are
    /// clamped too.
    fn clamp_charge(&self, charge: f64) -> f64 {
        let mean = (self.charge_sum + self.prior_weight * PRIOR_MEAN_CHARGE)
            / (self.decisions + self.prior_weight);
        charge.min(CHARGE_CLAMP_FACTOR * mean)
    }

    /// Spill fraction of the keys whose rendezvous winner is `candidates[preferred]`: the
    /// larger of the charge-demand and request-demand fractions ([`spill_fraction`], each
    /// with its own sampling noise, at most `slack`), plus the set's routed-share boost,
    /// at most 1. Zero before warm-up.
    fn spill_p(
        &self,
        candidates: &[(&str, f64)],
        ranking: &[usize],
        preferred: usize,
        slack: f64,
    ) -> f64 {
        if ranking.len() < 2 || self.window_decisions < SHARE_MIN_SAMPLES {
            return 0.0;
        }
        let name = candidates[preferred].0;
        let total = |map: &HashMap<String, f64>| -> f64 {
            ranking
                .iter()
                .map(|&i| Self::stat(map, candidates[i].0))
                .sum()
        };
        let mut p: f64 = 0.0;
        for (expected, demand, samples) in [
            (&self.expected, &self.demand, self.charge_samples()),
            (
                &self.expected_requests,
                &self.demand_requests,
                self.request_samples(),
            ),
        ] {
            let (expected_total, demand_total) = (total(expected), total(demand));
            if expected_total <= 0.0 || demand_total <= 0.0 {
                continue;
            }
            let fair = Self::stat(expected, name) / expected_total;
            let share = Self::stat(demand, name) / demand_total;
            let noise = SPILL_NOISE_SIGMAS * sampling_sd(share, samples);
            p = p.max(spill_fraction(
                share,
                fair,
                share_band(fair, slack),
                slack,
                noise,
            ));
        }
        (p + Self::stat(&self.boost, name)).min(1.0)
    }

    /// Affinity decision for a non-heavy `key`; pure in the state.
    fn decide(
        &self,
        key: u64,
        candidates: &[(&str, f64)],
        ranking: &[usize],
        slack: f64,
    ) -> (usize, SetChoiceReason) {
        let preferred = ranking[0];
        if ranking.len() >= 2
            && spill_point(key) < self.spill_p(candidates, ranking, preferred, slack)
        {
            (ranking[1], SetChoiceReason::ShareCapFallback)
        } else {
            (preferred, SetChoiceReason::Affinity)
        }
    }

    /// Routed-share feedback (integral term of the spill fraction). For each candidate
    /// set, compare the shares of the window's clamped charge and requests it actually
    /// received (winner or spill target) with its fair shares, with an allowance of two
    /// standard deviations of sampling noise on both sides:
    ///
    /// - while either share exceeds the upper band edge by more than the allowance, the
    ///   set's boost grows by [`SPILL_BOOST_GAIN`] times the excess;
    /// - while both shares are below fair share by more than the allowance, it shrinks by
    ///   the gain times the smaller shortfall;
    /// - while the set's demand (the load and requests whose rendezvous winner it is, that
    ///   is, what it would receive with no spill at all) is within its band on both
    ///   measures by more than the allowance, the boost is not needed and leaks away at
    ///   the gain times the band;
    /// - otherwise it holds.
    ///
    /// The deadband `[fair − allowance, fair + band + allowance]` is wider than any key
    /// below the heavy threshold, so a whole-key move that lands anywhere in it (also
    /// exactly on fair share) stays put instead of being undone by noise and redone.
    /// Sticky spill alone moves at most `slack` of a set's keys, which persistent pools
    /// whose keys have high spill points defeat; the boost moves more, still in spill-point
    /// order. Acting only outside the band leaves room for the two measures to disagree
    /// (a set over on charge spilling into one that is near its edge on requests).
    fn update_boost(&mut self, candidates: &[(&str, f64)], ranking: &[usize], slack: f64) {
        if ranking.len() < 2 || self.window_decisions < SHARE_MIN_SAMPLES {
            return;
        }
        let total = |map: &HashMap<String, f64>| -> f64 {
            ranking
                .iter()
                .map(|&i| Self::stat(map, candidates[i].0))
                .sum()
        };
        // (routed, demand, fair, effective samples) per measure.
        let measures = [
            (
                &self.routed,
                &self.demand,
                &self.expected,
                self.charge_samples(),
            ),
            (
                &self.routed_requests,
                &self.demand_requests,
                &self.expected_requests,
                self.request_samples(),
            ),
        ];
        let totals =
            measures.map(|(routed, demand, fair, _)| (total(routed), total(demand), total(fair)));
        for &i in ranking {
            let name = candidates[i].0;
            let mut over = f64::NEG_INFINITY;
            let mut below = f64::NEG_INFINITY;
            let mut needed = false;
            let mut band_of_set = 0.0;
            for ((routed, demand, fair, samples), (routed_total, demand_total, fair_total)) in
                measures.iter().zip(totals)
            {
                if routed_total <= 0.0 || demand_total <= 0.0 || fair_total <= 0.0 {
                    continue;
                }
                let share = Self::stat(routed, name) / routed_total;
                let demand_share = Self::stat(demand, name) / demand_total;
                let fair = Self::stat(fair, name) / fair_total;
                let band = share_band(fair, slack);
                let allowance = |x: f64| SPILL_NOISE_SIGMAS * sampling_sd(x, *samples);
                over = over.max(share - fair - band - allowance(share));
                below = below.max(share - fair + allowance(share));
                needed |= demand_share + allowance(demand_share) > fair + band;
                band_of_set = band;
            }
            let step = if over > 0.0 {
                over
            } else if below < 0.0 {
                below
            } else if !needed && over.is_finite() {
                -band_of_set
            } else {
                continue;
            };
            match self.boost.get_mut(name) {
                Some(b) => {
                    *b = (*b + SPILL_BOOST_GAIN * step).min(1.0);
                    if step < 0.0 && *b < SPILL_BOOST_FLOOR {
                        self.boost.remove(name);
                    }
                }
                None if step > 0.0 => {
                    self.boost
                        .insert(name.to_string(), (SPILL_BOOST_GAIN * step).min(1.0));
                }
                None => {}
            }
        }
    }

    /// Effective sample size of the window's charge shares, floored at its bound under
    /// the charge clamp.
    fn charge_samples(&self) -> f64 {
        effective_samples(self.window_sum, self.window_sq_sum)
            .max(self.window_decisions / CHARGE_CLAMP_FACTOR)
    }

    /// Effective sample size of the window's request shares.
    fn request_samples(&self) -> f64 {
        effective_samples(self.window_decisions, self.window_decisions_sq)
    }

    /// Decay the statistics every keyed decision feeds: the decision counts and charge
    /// sums behind the clamp and heavy-key shares, and the heavy-key counters.
    fn decay_global(&mut self) {
        let decay = 1.0 - 1.0 / SHARE_WINDOW;
        self.decisions *= decay;
        self.charge_sum *= decay;
        self.raw_sum *= decay;
        self.raw_sq_sum *= decay * decay;
        self.decisions_sq *= decay * decay;
        self.prior_weight *= decay;
        for c in &mut self.heavy {
            c.raw *= decay;
            c.raw_err *= decay;
            c.count *= decay;
            c.count_err *= decay;
        }
    }

    /// Decay the fair-share window and the counters' window credit (kept in step, so a
    /// heavy entry removes exactly what is left of the key's contribution). Only
    /// decisions that add to the window decay it, so its horizon is the last ~1000
    /// affinity-routed decisions however much heavy-key traffic lies between them: heavy
    /// traffic neither ages the window's shares nor keeps it from warming up.
    fn decay_window(&mut self) {
        let decay = 1.0 - 1.0 / SHARE_WINDOW;
        self.window_decisions *= decay;
        self.window_sum *= decay;
        self.window_sq_sum *= decay * decay;
        self.window_decisions_sq *= decay * decay;
        for map in [
            &mut self.expected,
            &mut self.demand,
            &mut self.expected_requests,
            &mut self.demand_requests,
            &mut self.routed,
            &mut self.routed_requests,
        ] {
            for v in map.values_mut() {
                *v *= decay;
            }
            map.retain(|_, v| *v > 1e-6);
        }
        for c in &mut self.heavy {
            c.window_sq *= decay * decay;
            c.window_decisions *= decay;
            c.window_decisions_sq *= decay * decay;
            for list in c.credit_lists_mut() {
                for (_, credit) in list.iter_mut() {
                    *credit *= decay;
                }
            }
        }
    }

    /// Record one keyed decision that placed the request on `candidates[chosen]`.
    /// Affinity-routed requests also feed the fair-share window (demand, fair share and
    /// placement), decay it and step the routed-share boost; heavy-key requests
    /// (weighted random) do none of that. Whenever the window is cold (below
    /// [`SHARE_MIN_SAMPLES`], for example after a heavy entry removed much of it) the boost
    /// is dropped, as on a restart, so it is never re-applied to a later load mix.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        key: u64,
        candidates: &[(&str, f64)],
        ranking: &[usize],
        chosen: usize,
        affinity_routed: bool,
        charge: f64,
        band: f64,
        slack: f64,
    ) {
        let raw = charge;
        let charge = self.clamp_charge(raw);
        self.decay_global();
        if affinity_routed {
            self.decay_window();
        }
        self.decisions += 1.0;
        self.decisions_sq += 1.0;
        self.charge_sum += charge;
        self.raw_sum += raw;
        self.raw_sq_sum += raw * raw;
        let total_weight: f64 = ranking.iter().map(|&i| candidates[i].1).sum();
        let bump = |map: &mut HashMap<String, f64>, name: &str, amount: f64| match map.get_mut(name)
        {
            Some(v) => *v += amount,
            None => {
                map.insert(name.to_string(), amount);
            }
        };
        let slot = self.record_heavy(key, raw);
        if affinity_routed {
            let (winner, target) = (candidates[ranking[0]].0, candidates[chosen].0);
            let counter = &mut self.heavy[slot];
            add_credit(&mut counter.window_demand, set_id(winner), charge);
            add_credit(&mut counter.window_demand_requests, set_id(winner), 1.0);
            add_credit(&mut counter.window_routed, set_id(target), charge);
            add_credit(&mut counter.window_routed_requests, set_id(target), 1.0);
            counter.window_sq += charge * charge;
            counter.window_decisions += 1.0;
            counter.window_decisions_sq += 1.0;
            self.window_decisions += 1.0;
            self.window_decisions_sq += 1.0;
            self.window_sum += charge;
            self.window_sq_sum += charge * charge;
            for &i in ranking {
                let (name, share) = (candidates[i].0, candidates[i].1 / total_weight);
                bump(&mut self.expected, name, charge * share);
                bump(&mut self.expected_requests, name, share);
                let counter = &mut self.heavy[slot];
                add_credit(&mut counter.window_expected, set_id(name), charge * share);
                add_credit(&mut counter.window_expected_requests, set_id(name), share);
            }
            bump(&mut self.demand, winner, charge);
            bump(&mut self.demand_requests, winner, 1.0);
            bump(&mut self.routed, target, charge);
            bump(&mut self.routed_requests, target, 1.0);
        }
        // An affinity-routed key is not heavy (the caller clears a stale flag first), so a
        // flagged counter never holds window credit. On entry its credit leaves.
        let status = self.heavy_status(&self.heavy[slot], band);
        if status && !self.heavy[slot].heavy {
            let credit = self.heavy[slot].clone();
            self.forget_window_load(&credit, candidates, ranking);
            self.heavy[slot].clear_window_credit();
        }
        self.heavy[slot].heavy = status;
        // A decision that added nothing to the window cannot move the shares the boost
        // integrates, so it does not step it.
        if affinity_routed {
            self.update_boost(candidates, ranking, slack);
        }
        if self.window_decisions < SHARE_MIN_SAMPLES {
            self.boost.clear();
        }
    }

    /// Space-Saving update for `key`; returns its counter's slot.
    fn record_heavy(&mut self, key: u64, raw: f64) -> usize {
        if let Some(slot) = self.heavy.iter().position(|c| c.key == key) {
            let c = &mut self.heavy[slot];
            c.raw += raw;
            c.count += 1.0;
            return slot;
        }
        let fresh = |raw_err: f64, count_err: f64| {
            HeavyCounter::new(key, raw_err + raw, raw_err, count_err + 1.0, count_err)
        };
        if self.heavy.len() < HEAVY_KEYS {
            self.heavy.push(fresh(0.0, 0.0));
            return self.heavy.len() - 1;
        }
        // Take over the least important counter by its larger share, so keys heavy by
        // requests and keys heavy by charge both keep their counters.
        let slot = self
            .heavy
            .iter()
            .enumerate()
            .min_by(|a, b| self.importance(a.1).total_cmp(&self.importance(b.1)))
            .map_or(0, |(i, _)| i);
        // The new key inherits the slot's window credit as it inherits its counts as
        // error, so the window always equals the sum of all counters' credits and a heavy
        // entry never leaves orphaned credit behind (it may remove up to the inherited
        // amount more than the key's own, within the same eviction bound). A flagged
        // victim holds no credit.
        let (taken_raw, taken_count) = (self.heavy[slot].raw, self.heavy[slot].count);
        let victim = std::mem::replace(&mut self.heavy[slot], fresh(taken_raw, taken_count));
        let taker = &mut self.heavy[slot];
        let HeavyCounter {
            window_demand,
            window_expected,
            window_demand_requests,
            window_expected_requests,
            window_routed,
            window_routed_requests,
            window_sq,
            window_decisions,
            window_decisions_sq,
            ..
        } = victim;
        taker.window_demand = window_demand;
        taker.window_expected = window_expected;
        taker.window_demand_requests = window_demand_requests;
        taker.window_expected_requests = window_expected_requests;
        taker.window_routed = window_routed;
        taker.window_routed_requests = window_routed_requests;
        taker.window_sq = window_sq;
        taker.window_decisions = window_decisions;
        taker.window_decisions_sq = window_decisions_sq;
        slot
    }

    /// A key that just became heavy stops counting toward the fair-share window: remove
    /// exactly what it contributed to the current window (the charge and requests it
    /// credited to each set as demand, the fair charge and fair requests it credited to
    /// each set, the charge and requests it placed on each set, its squared charges, its
    /// decisions and their squares), so its past turns neither keep the other keys
    /// spilling nor drive the routed-share feedback. Nothing is removed from a set that
    /// did not receive the credit.
    fn forget_window_load(
        &mut self,
        credit: &HeavyCounter,
        candidates: &[(&str, f64)],
        ranking: &[usize],
    ) {
        let mut load = 0.0;
        for &i in ranking {
            let (name, id) = (candidates[i].0, set_id(candidates[i].0));
            let find =
                |list: &[(u64, f64)]| list.iter().find(|(s, _)| *s == id).map_or(0.0, |(_, c)| *c);
            load += find(&credit.window_demand);
            for (map, list) in [
                (&mut self.demand, &credit.window_demand),
                (&mut self.expected, &credit.window_expected),
                (&mut self.demand_requests, &credit.window_demand_requests),
                (
                    &mut self.expected_requests,
                    &credit.window_expected_requests,
                ),
                (&mut self.routed, &credit.window_routed),
                (&mut self.routed_requests, &credit.window_routed_requests),
            ] {
                if let Some(v) = map.get_mut(name) {
                    *v = (*v - find(list)).max(0.0);
                }
            }
        }
        self.window_sum = (self.window_sum - load).max(0.0);
        self.window_sq_sum = (self.window_sq_sum - credit.window_sq).max(0.0);
        self.window_decisions = (self.window_decisions - credit.window_decisions).max(0.0);
        self.window_decisions_sq = (self.window_decisions_sq - credit.window_decisions_sq).max(0.0);
    }

    /// Clear a stale heavy flag of `key` before an affinity-routed request is recorded.
    fn unflag(&mut self, key: u64) {
        if let Some(counter) = self.heavy.iter_mut().find(|c| c.key == key) {
            counter.heavy = false;
        }
    }
}

/// Per-frontend affinity state for one model (see the module docs).
#[derive(Debug, Default)]
pub struct ShareTracker {
    state: Mutex<ShareState>,
    /// Set when a selection ran with fewer than two eligible sets; the next affinity
    /// decision starts a fresh fair-share window.
    stale: AtomicBool,
}

impl ShareTracker {
    /// Note that a selection ran with fewer than two eligible sets (lock-free).
    pub fn mark_single_eligible(&self) {
        self.stale.store(true, Ordering::Relaxed);
    }

    /// Choose among `candidates` (unique set key, weight) for `affinity` and record the
    /// decision. Heavy keys get the weighted random pick
    /// ([`SetChoiceReason::HeavyKeyRandom`]); other keys get their rendezvous winner or,
    /// when sticky spill applies, their second choice
    /// ([`SetChoiceReason::ShareCapFallback`]).
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
        if self.stale.swap(false, Ordering::Relaxed) {
            state.restart_window(candidates);
        }
        state.track_candidates(candidates);
        let band = narrowest_band(candidates, &ranking, slack);
        let choice = if state.is_heavy(affinity.key, band) {
            let weights: Vec<f64> = candidates.iter().map(|(_, w)| *w).collect();
            let uniform = state.rng.random::<f64>();
            (
                weighted_pick(&weights, uniform)?,
                SetChoiceReason::HeavyKeyRandom,
            )
        } else {
            // A key flagged heavy that no longer qualifies is routed by affinity: drop the
            // flag before recording, so its load stays in the window and re-entry needs
            // the full entry threshold.
            state.unflag(affinity.key);
            state.decide(affinity.key, candidates, &ranking, slack)
        };
        let affinity_routed = choice.1 != SetChoiceReason::HeavyKeyRandom;
        state.record(
            affinity.key,
            candidates,
            &ranking,
            choice.0,
            affinity_routed,
            affinity.charge,
            band,
            slack,
        );
        Some(choice)
    }

    /// The affinity decision `choose` would make now for a non-heavy key, without
    /// recording it (test hook).
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

    /// Current spill fraction (demand terms plus boost) for set `idx` of `candidates`
    /// (test hook).
    #[cfg(test)]
    pub(crate) fn spill_fraction_of(
        &self,
        idx: usize,
        candidates: &[(&str, f64)],
        slack: f64,
    ) -> f64 {
        let ranking: Vec<usize> = (0..candidates.len()).collect();
        self.state.lock().spill_p(candidates, &ranking, idx, slack)
    }

    /// Share of the window's affinity demand whose rendezvous winner is `name` (test hook).
    #[cfg(test)]
    pub(crate) fn demand_share(&self, name: &str) -> f64 {
        let state = self.state.lock();
        let total: f64 = state.demand.values().sum();
        ShareState::stat(&state.demand, name) / total.max(f64::MIN_POSITIVE)
    }

    /// Whether `key` currently counts as heavy among `candidates` (test hook).
    #[cfg(test)]
    pub(crate) fn is_heavy(&self, key: u64, candidates: &[(&str, f64)], slack: f64) -> bool {
        let ranking = rendezvous_ranking(key, candidates);
        let band = narrowest_band(candidates, &ranking, slack);
        self.state.lock().is_heavy(key, band)
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
const MESSAGES_TAG: &[u8] = b"messages/v4\0";

fn explicit_affinity_key(tag: &[u8], value: &str) -> u64 {
    let mut hasher = Xxh3::with_seed(AFFINITY_SEED);
    hasher.update(tag);
    hasher.update(value.as_bytes());
    hasher.digest()
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

    /// A non-text part, identified by the hash of its JSON serialization: one streaming
    /// pass into a separate hasher, then a fixed-size record (tag and 8-byte digest).
    fn part<T: serde::Serialize>(&mut self, part: &T) -> Option<()> {
        let mut inner = HashWriter(Xxh3::with_seed(AFFINITY_SEED));
        serde_json::to_writer(&mut inner, part).ok()?;
        self.tag(b'P');
        self.update(&inner.0.digest().to_le_bytes());
        Some(())
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
    /// TP2 60 → 58 workers: TP4's weight share moves by 0.008, below the material change
    /// that restarts the window.
    const SMALL_DRIFT: [(&str, f64); 2] = [(TP4, 30.0), (TP2, 58.0)];

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

    /// R10-4: a slack above 0.5 is clamped to 0.5; above 1 it is almost certainly a
    /// percentage ("25" for 25%) and is rejected in favour of the default.
    #[test]
    fn parse_caps_large_slack() {
        for (raw, slack) in [
            ("25", DEFAULT_SLACK),
            ("1.5", DEFAULT_SLACK),
            ("1", MAX_SLACK),
            ("0.7", MAX_SLACK),
            ("0.5", 0.5),
            ("0.3", 0.3),
        ] {
            let c = SetSelectionConfig::parse(Some("affinity"), Some(raw), None);
            assert_eq!(c.slack, slack, "slack {raw}");
        }
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

    /// R2-7 / R10-2: the share guard adds no spills of its own across a worker-count
    /// change. 60 → 30 TP2 workers is material, so the window restarts at the new weights
    /// (smaller drifts keep it, and fair shares are averaged over the same window as the
    /// observed load).
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

    fn key_preferring(name: &str, pred: impl Fn(f64) -> bool) -> u64 {
        (0..u64::MAX)
            .map(mixed)
            .find(|k| SETS[rendezvous_pick(*k, &SETS).unwrap()].0 == name && pred(spill_point(*k)))
            .unwrap()
    }

    /// Charge placed on each of the two sets.
    #[derive(Default)]
    struct Tally([f64; 2]);

    impl Tally {
        fn add(&mut self, idx: usize, charge: f64) {
            self.0[idx] += charge;
        }

        fn share(&self, idx: usize) -> f64 {
            self.0[idx] / (self.0[0] + self.0[1])
        }

        /// Both sets inside `fair ± band`, with `tolerance`.
        fn assert_in_band(&self, tolerance: f64, context: &str) {
            for (idx, fair) in [(0, 1.0 / 3.0), (1, 2.0 / 3.0)] {
                let band = share_band(fair, DEFAULT_SLACK);
                let share = self.share(idx);
                assert!(
                    (fair - band - tolerance..=fair + band + tolerance).contains(&share),
                    "{context}: {} share {share}",
                    SETS[idx].0
                );
            }
        }
    }

    #[test]
    fn share_guard_keeps_affinity_for_balanced_keys() {
        let tracker = ShareTracker::default();
        for key in 0..20_000u64 {
            let key = mixed(key);
            let (idx, reason) = tracker.choose(unit(key), &SETS, DEFAULT_SLACK).unwrap();
            assert_eq!(
                (Some(idx), reason),
                (rendezvous_pick(key, &SETS), SetChoiceReason::Affinity)
            );
        }
        // The routed-share feedback never engaged.
        assert!(tracker.state.lock().boost.is_empty());
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

        /// A lognormal charge with median 2000 bytes.
        fn charge(&mut self, sigma: f64) -> f64 {
            2000.0 * (sigma * self.normal()).exp()
        }
    }

    // -- R4-1: heavy keys bypass affinity --

    /// Feed `n` decisions in which 3 of every 10 requests come from `hot` (charge
    /// `hot_charge`) and the rest from distinct background keys with lognormal(`sigma`)
    /// charges. Returns the placement tally of the last `n / 2` decisions and every
    /// non-hot decision.
    fn run_with_hot_key(
        tracker: &ShareTracker,
        hot: u64,
        hot_charge: f64,
        sigma: f64,
        n: u64,
    ) -> (Tally, Vec<(u64, usize, SetChoiceReason)>) {
        let mut rng = Rng(hot ^ 0x77);
        let mut tally = Tally::default();
        let mut others = Vec::new();
        for i in 0..n {
            let (key, charge) = if i % 10 < 3 {
                (hot, hot_charge)
            } else {
                (mixed(i + 1_000_000), rng.charge(sigma))
            };
            let (idx, reason) = tracker
                .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                .unwrap();
            if i >= n / 2 {
                tally.add(idx, charge);
            }
            if key != hot {
                others.push((key, idx, reason));
            }
        }
        (tally, others)
    }

    /// A 30% hot key preferring either set (spill point low or high) is detected, routed
    /// by the weighted random pick, and both sets stay in band.
    #[test]
    fn hot_key_keeps_both_sets_within_band() {
        for (pref, low_spill_point) in [(TP4, false), (TP4, true), (TP2, false), (TP2, true)] {
            let tracker = ShareTracker::default();
            let hot = key_preferring(pref, |x| if low_spill_point { x < 0.05 } else { x > 0.5 });
            let (tally, _) = run_with_hot_key(&tracker, hot, 2000.0, 1.0, 20_000);
            assert!(tracker.is_heavy(hot, &SETS, DEFAULT_SLACK));
            tally.assert_in_band(0.03, &format!("hot key prefers {pref}"));
        }
    }

    /// With a hot key present, every other key keeps its rendezvous winner on every
    /// request: the hot key's load does not make them spill.
    #[test]
    fn non_hot_keys_never_change_set() {
        for pref in [TP4, TP2] {
            let tracker = ShareTracker::default();
            let hot = key_preferring(pref, |_| true);
            let (_, others) = run_with_hot_key(&tracker, hot, 2000.0, 1.0, 20_000);
            for (key, idx, reason) in others {
                assert_eq!(
                    (Some(idx), reason),
                    (rendezvous_pick(key, &SETS), SetChoiceReason::Affinity),
                    "hot key prefers {pref}"
                );
            }
        }
    }

    // -- R5-1: only a key's own window contribution leaves the window when it turns heavy --

    /// One phase of a scripted run: candidate weights, how often the hot key requests and
    /// with what charge (background keys are unit-charge), the number of decisions, and
    /// whether a single-eligible selection precedes it.
    struct Phase {
        sets: [(&'static str, f64); 2],
        hot_every: u64,
        hot_charge: f64,
        decisions: u64,
        reset_before: bool,
    }

    /// Spills of non-hot keys by the set their rendezvous winner was, the charge tally
    /// after warm-up, the hot key's heavy status per request, and how many heavy entries
    /// had their window bookkeeping checked.
    struct PhaseOutcome {
        spills_from: [usize; 2],
        decisions: usize,
        tally: Tally,
        hot_heavy: Vec<bool>,
        entries_checked: usize,
    }

    /// The window quantities a heavy entry changes, plus the hot key's counter.
    struct WindowSnapshot {
        counter: Option<HeavyCounter>,
        demand: [f64; 2],
        expected: [f64; 2],
        demand_requests: [f64; 2],
        expected_requests: [f64; 2],
        routed: [f64; 2],
        routed_requests: [f64; 2],
        sum: f64,
        sq_sum: f64,
        decisions: f64,
        decisions_sq: f64,
    }

    impl WindowSnapshot {
        fn take(tracker: &ShareTracker, hot: u64) -> Self {
            Self::take_for(tracker, hot, &SETS)
        }

        fn take_for(tracker: &ShareTracker, hot: u64, sets: &[(&str, f64); 2]) -> Self {
            let state = tracker.state.lock();
            let pair = |map: &HashMap<String, f64>| {
                [
                    ShareState::stat(map, sets[0].0),
                    ShareState::stat(map, sets[1].0),
                ]
            };
            Self {
                counter: state.heavy.iter().find(|c| c.key == hot).cloned(),
                demand: pair(&state.demand),
                expected: pair(&state.expected),
                demand_requests: pair(&state.demand_requests),
                expected_requests: pair(&state.expected_requests),
                routed: pair(&state.routed),
                routed_requests: pair(&state.routed_requests),
                sum: state.window_sum,
                sq_sum: state.window_sq_sum,
                decisions: state.window_decisions,
                decisions_sq: state.window_decisions_sq,
            }
        }

        /// From `self` (before the hot key's request, placed on `sets[chosen]`) to `post`:
        /// every window quantity decayed, plus this request if affinity-routed, minus
        /// exactly the key's window credit (also decayed, plus this request if it was
        /// credited).
        #[allow(clippy::too_many_arguments)]
        fn assert_credit_removed(
            &self,
            post: &Self,
            hot: u64,
            sets: &[(&str, f64); 2],
            chosen: usize,
            routed: bool,
            charge: f64,
        ) {
            // The window (and the counters' window credit) decays only on a decision that
            // adds to it.
            let d = if routed {
                1.0 - 1.0 / SHARE_WINDOW
            } else {
                1.0
            };
            let close = |actual: f64, expected: f64, what: &str| {
                assert!(
                    (actual - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                    "{what}: {actual} != {expected}"
                );
            };
            let winner = rendezvous_pick(hot, sets).unwrap();
            let total: f64 = sets.iter().map(|(_, w)| *w).sum();
            let fair = |s: usize| charge * sets[s].1 / total;
            let request = if routed { 1.0 } else { 0.0 };
            let counter = self
                .counter
                .clone()
                .unwrap_or_else(|| HeavyCounter::new(hot, 0.0, 0.0, 0.0, 0.0));
            let credit_of = |list: &[(u64, f64)], name: &str| {
                list.iter()
                    .find(|(id, _)| *id == set_id(name))
                    .map_or(0.0, |(_, c)| *c)
            };
            // A flagged counter holds no credit; a winner change keeps earlier credit.
            let mut credit_load = 0.0;
            for (s, (name, _)) in sets.iter().enumerate() {
                let won = if routed && s == winner { 1.0 } else { 0.0 };
                let placed = if routed && s == chosen { 1.0 } else { 0.0 };
                let weight_share = sets[s].1 / total;
                // (window quantity, its snapshot before, the counter's credit list, what
                // this request added to both)
                let quantities = [
                    (
                        "demand",
                        post.demand[s],
                        self.demand[s],
                        &counter.window_demand,
                        won * charge,
                    ),
                    (
                        "expected",
                        post.expected[s],
                        self.expected[s],
                        &counter.window_expected,
                        request * fair(s),
                    ),
                    (
                        "demand_requests",
                        post.demand_requests[s],
                        self.demand_requests[s],
                        &counter.window_demand_requests,
                        won,
                    ),
                    (
                        "expected_requests",
                        post.expected_requests[s],
                        self.expected_requests[s],
                        &counter.window_expected_requests,
                        request * weight_share,
                    ),
                    (
                        "routed",
                        post.routed[s],
                        self.routed[s],
                        &counter.window_routed,
                        placed * charge,
                    ),
                    (
                        "routed_requests",
                        post.routed_requests[s],
                        self.routed_requests[s],
                        &counter.window_routed_requests,
                        placed,
                    ),
                ];
                for (what, after, before, list, added) in quantities {
                    let credit = credit_of(list, name) * d + added;
                    if what == "demand" {
                        credit_load += credit;
                    }
                    close(after, before * d + added - credit, &format!("{what}[{s}]"));
                }
            }
            let credit_sq = counter.window_sq * d * d + request * charge * charge;
            let credit_decisions = counter.window_decisions * d + request;
            close(
                post.sum,
                self.sum * d + request * charge - credit_load,
                "window_sum",
            );
            close(
                post.sq_sum,
                self.sq_sum * d * d + request * charge * charge - credit_sq,
                "window_sq_sum",
            );
            close(
                post.decisions,
                self.decisions * d + request - credit_decisions,
                "window_decisions",
            );
            let credit_decisions_sq = counter.window_decisions_sq * d * d + request;
            close(
                post.decisions_sq,
                self.decisions_sq * d * d + request - credit_decisions_sq,
                "window_decisions_sq",
            );
        }
    }

    fn run_phases(hot: u64, phases: &[Phase]) -> PhaseOutcome {
        let tracker = ShareTracker::default();
        let mut outcome = PhaseOutcome {
            spills_from: [0; 2],
            decisions: 0,
            tally: Tally::default(),
            hot_heavy: Vec::new(),
            entries_checked: 0,
        };
        let mut i = 0u64;
        for phase in phases {
            if phase.reset_before {
                tracker.mark_single_eligible();
            }
            for step in 0..phase.decisions {
                i += 1;
                let is_hot = i.is_multiple_of(phase.hot_every);
                let (key, charge) = if is_hot {
                    (hot, phase.hot_charge)
                } else {
                    (mixed(i + 70_000_000), 1.0)
                };
                // Window bookkeeping before the decision, for the heavy-entry invariant.
                let pre = WindowSnapshot::take(&tracker, hot);
                let clamped = tracker.state.lock().clamp_charge(charge);
                let (idx, reason) = tracker
                    .choose(SetAffinity::new(key, charge), &phase.sets, DEFAULT_SLACK)
                    .unwrap();
                if i > 1000 {
                    outcome.tally.add(idx, charge);
                }
                if !is_hot {
                    outcome.decisions += 1;
                    if reason == SetChoiceReason::ShareCapFallback {
                        outcome.spills_from[rendezvous_pick(key, &phase.sets).unwrap()] += 1;
                    }
                    continue;
                }
                outcome
                    .hot_heavy
                    .push(reason == SetChoiceReason::HeavyKeyRandom);
                let routed = reason != SetChoiceReason::HeavyKeyRandom;
                let post = WindowSnapshot::take(&tracker, hot);
                let was_flagged = pre.counter.as_ref().is_some_and(|c| c.heavy);
                let now_flagged = post.counter.as_ref().is_some_and(|c| c.heavy);
                // All window credit leaves on a real entry. A flagged key routed by affinity
                // has its flag cleared first (a flagged counter never holds credit), so an
                // entry is "flagged after, and not flagged before or routed by affinity".
                let forgot = now_flagged && (!was_flagged || routed);
                // The first decision of a phase may restart the window (reset_before or a
                // material weight change), which the snapshot does not model.
                if !forgot || step == 0 {
                    continue;
                }
                pre.assert_credit_removed(&post, hot, &phase.sets, idx, routed, clamped);
                outcome.entries_checked += 1;
            }
        }
        outcome
    }

    /// R7-3: a periodic key cools to the load exit edge. Once it no longer qualifies it is
    /// routed by affinity with its flag cleared, so every such request stays in its
    /// winner's demand (nothing is forgotten at the edge) and re-entry needs the full
    /// entry threshold.
    #[test]
    fn exit_edge_keeps_affinity_load_in_the_window() {
        let key = key_preferring(TP4, |x| x > 0.5);
        let tracker = ShareTracker::default();
        let d = 1.0 - 1.0 / SHARE_WINDOW;
        let mut i = 0u64;
        let mut edge_routed = 0;
        // Heavy at one request in 4 (2x charge), then periodic at one in 24 with unit
        // charge: a 1/24 load share is exactly the exit threshold (half the 1/12 band), so
        // the share is just below it before each request and just above it after.
        for (every, charge, decisions) in [(4u64, 2.0, 6_000u64), (24, 1.0, 15_000)] {
            for _ in 0..decisions {
                i += 1;
                if !i.is_multiple_of(every) {
                    tracker
                        .choose(unit(mixed(i + 40_000_000)), &SETS, DEFAULT_SLACK)
                        .unwrap();
                    continue;
                }
                let pre = WindowSnapshot::take(&tracker, key);
                let clamped = tracker.state.lock().clamp_charge(charge);
                let (idx, reason) = tracker
                    .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                    .unwrap();
                let post = WindowSnapshot::take(&tracker, key);
                if every == 4 || reason == SetChoiceReason::HeavyKeyRandom {
                    continue;
                }
                edge_routed += 1;
                // Not heavy after its request: the request stays in the winner's demand.
                assert!(post.counter.as_ref().is_some_and(|c| !c.heavy));
                assert_eq!(idx, 0);
                let added = post.demand[0] - pre.demand[0] * d;
                assert!(
                    (added - clamped).abs() < 1e-6,
                    "demand change {added} != {clamped}"
                );
            }
        }
        assert!(edge_routed > 300, "edge requests {edge_routed}");
        let credit = tracker
            .state
            .lock()
            .heavy
            .iter()
            .find(|c| c.key == key)
            .cloned()
            .unwrap();
        assert!(
            credit.window_decisions > 30.0,
            "credit {}",
            credit.window_decisions
        );
    }

    /// Per-set credit oracle for one key, maintained independently of the tracker: demand
    /// and fair credit by charge and by request count.
    #[derive(Clone, Copy, Default)]
    struct CreditOracle {
        demand: [f64; 2],
        expected: [f64; 2],
        demand_requests: [f64; 2],
        expected_requests: [f64; 2],
    }

    /// Hot keys sending one request each per `period` requests at `charge` times the
    /// background, over `phases` (weights, decisions, period). Checks every heavy entry against
    /// the window snapshot and against an independent per-set credit oracle. Returns the
    /// background spills by phase and preferred set, the decision index of each entry, and
    /// how many hot requests were heavy in the first phase.
    struct RecoveryOutcome {
        spills: Vec<[usize; 2]>,
        spills_after_entries: [usize; 2],
        entries: Vec<u64>,
        heavy_in_first_phase: usize,
        /// Entries on an affinity-routed request that also moved the key's winner (it
        /// held demand credit on another set).
        coincident: usize,
    }

    /// Candidate weights, decisions, and the hot keys' request period of one phase.
    type RecoveryPhase = ([(&'static str, f64); 2], u64, u64);

    fn run_recovery(hot: &[u64], charge: f64, phases: &[RecoveryPhase]) -> RecoveryOutcome {
        let d = 1.0 - 1.0 / SHARE_WINDOW;
        let tracker = ShareTracker::default();
        let mut oracle = vec![CreditOracle::default(); hot.len()];
        let mut outcome = RecoveryOutcome {
            spills: Vec::new(),
            spills_after_entries: [0; 2],
            entries: Vec::new(),
            heavy_in_first_phase: 0,
            coincident: 0,
        };
        let mut i = 0u64;
        let mut baseline: Option<f64> = None;
        for (phase, (sets, decisions, period)) in phases.iter().enumerate() {
            let mut spills = [0usize; 2];
            let total: f64 = sets.iter().map(|(_, w)| *w).sum();
            // The tracker restarts its window (dropping all credit) on the first decision
            // of a phase whose weights moved materially since the last restart.
            let share = sets[0].1 / total;
            let material = baseline.is_none_or(|b| (share - b).abs() > MATERIAL_SHARE_CHANGE);
            if material {
                baseline = Some(share);
            }
            for step in 0..*decisions {
                i += 1;
                let restarted = material && step == 0;
                let slot = (i % *period) as usize;
                let (key, request_charge) = match hot.get(slot) {
                    Some(&k) => (k, charge),
                    None => (mixed(i + 90_000_000), 1.0),
                };
                let pre = WindowSnapshot::take_for(&tracker, key, sets);
                let clamped = tracker.state.lock().clamp_charge(request_charge);
                if let Some(o) = hot.get(slot).map(|_| &oracle[slot]) {
                    // The tracker's credit before the request matches the oracle.
                    let counter = pre.counter.clone();
                    for (s, (name, _)) in sets.iter().enumerate() {
                        let find = |list: &[(u64, f64)]| {
                            list.iter()
                                .find(|(id, _)| *id == set_id(name))
                                .map_or(0.0, |(_, c)| *c)
                        };
                        let credits = counter.as_ref().map_or([0.0; 4], |c| {
                            [
                                find(&c.window_demand),
                                find(&c.window_expected),
                                find(&c.window_demand_requests),
                                find(&c.window_expected_requests),
                            ]
                        });
                        let oracle = [
                            o.demand[s],
                            o.expected[s],
                            o.demand_requests[s],
                            o.expected_requests[s],
                        ];
                        for (what, (credit, expected)) in
                            ["demand", "fair", "demand request", "fair request"]
                                .iter()
                                .zip(credits.iter().zip(oracle))
                        {
                            assert!(
                                (credit - expected).abs() < 1e-6,
                                "{what} credit {credit} != {expected}"
                            );
                        }
                    }
                }
                let (chosen, reason) = tracker
                    .choose(SetAffinity::new(key, request_charge), sets, DEFAULT_SLACK)
                    .unwrap();
                let routed = reason != SetChoiceReason::HeavyKeyRandom;
                if restarted {
                    oracle.fill(CreditOracle::default());
                }
                // Window credit decays only on decisions that update the window.
                for o in oracle.iter_mut().filter(|_| routed) {
                    for list in [
                        &mut o.demand,
                        &mut o.expected,
                        &mut o.demand_requests,
                        &mut o.expected_requests,
                    ] {
                        for credit in list.iter_mut() {
                            *credit *= d;
                        }
                    }
                }
                if slot >= hot.len() {
                    if reason == SetChoiceReason::ShareCapFallback {
                        let preferred = rendezvous_pick(key, sets).unwrap();
                        spills[preferred] += 1;
                        if outcome.entries.len() >= hot.len() {
                            outcome.spills_after_entries[preferred] += 1;
                        }
                    }
                    continue;
                }
                if routed {
                    let winner = rendezvous_pick(key, sets).unwrap();
                    let o = &mut oracle[slot];
                    o.demand[winner] += clamped;
                    o.demand_requests[winner] += 1.0;
                    for (s, (_, weight)) in sets.iter().enumerate() {
                        o.expected[s] += clamped * weight / total;
                        o.expected_requests[s] += weight / total;
                    }
                } else if phase == 0 {
                    outcome.heavy_in_first_phase += 1;
                }
                let post = WindowSnapshot::take_for(&tracker, key, sets);
                let was = pre.counter.as_ref().is_some_and(|c| c.heavy);
                let now = post.counter.as_ref().is_some_and(|c| c.heavy);
                // An entry on the decision that restarted the window is counted, but its
                // pre-request snapshot is from the old window.
                if now && (!was || routed) {
                    let checked = !restarted;
                    let winner = rendezvous_pick(key, sets).unwrap();
                    if routed && oracle[slot].demand[1 - winner] > 0.0 {
                        outcome.coincident += 1;
                    }
                    if checked {
                        pre.assert_credit_removed(&post, key, sets, chosen, routed, clamped);
                    }
                    // Independent check: the window dropped by exactly the oracle's credit.
                    let d = if routed { d } else { 1.0 };
                    for s in (0..2).filter(|_| checked) {
                        let added = if routed && Some(s) == rendezvous_pick(key, sets) {
                            clamped
                        } else {
                            0.0
                        };
                        let demand_drop = pre.demand[s] * d + added - post.demand[s];
                        let fair = if routed {
                            clamped * sets[s].1 / total
                        } else {
                            0.0
                        };
                        let expected_drop = pre.expected[s] * d + fair - post.expected[s];
                        let won = if routed && Some(s) == rendezvous_pick(key, sets) {
                            1.0
                        } else {
                            0.0
                        };
                        let demand_requests_drop =
                            pre.demand_requests[s] * d + won - post.demand_requests[s];
                        let fair_requests = if routed { sets[s].1 / total } else { 0.0 };
                        let expected_requests_drop = pre.expected_requests[s] * d + fair_requests
                            - post.expected_requests[s];
                        let o = oracle[slot];
                        for (what, drop, credit) in [
                            ("demand", demand_drop, o.demand[s]),
                            ("fair", expected_drop, o.expected[s]),
                            ("demand request", demand_requests_drop, o.demand_requests[s]),
                            (
                                "fair request",
                                expected_requests_drop,
                                o.expected_requests[s],
                            ),
                        ] {
                            assert!(
                                (drop - credit).abs() < 1e-6,
                                "{what} drop {drop} != {credit}"
                            );
                        }
                    }
                    oracle[slot] = CreditOracle::default();
                    outcome.entries.push(i);
                }
            }
            outcome.spills.push(spills);
        }
        outcome
    }

    /// Keys whose rendezvous winner is the same under `a` and `b`, `per_set` of each set.
    fn stable_keys(a: &[(&str, f64); 2], b: &[(&str, f64); 2], per_set: usize) -> Vec<u64> {
        let stable = |set: usize| {
            move |k: &u64| {
                rendezvous_pick(*k, a) == Some(set) && rendezvous_pick(*k, b) == Some(set)
            }
        };
        let mut hot: Vec<u64> = (0..u64::MAX)
            .map(mixed)
            .filter(stable(0))
            .take(per_set)
            .collect();
        hot.extend((0..u64::MAX).map(mixed).filter(stable(1)).take(per_set));
        hot
    }

    /// R6-2 / R7-4: keys credited to the window at 30:60 become heavy after a small drift
    /// to 30:58 (not material, so the window is kept). Their fair-load credit is removed
    /// at the weights it was credited at, not the current ones (checked against an
    /// independent oracle), so the residual window stays balanced and nothing spills once
    /// they are heavy.
    #[test]
    fn heavy_entry_after_small_drift_removes_historical_fair_credit() {
        let hot = stable_keys(&SETS, &SMALL_DRIFT, 2);
        // Each hot key sends 2.5% of requests at 4x the background charge (about 7.7% of
        // load, below the band less noise), then 5% (12.5%: heavy).
        let outcome = run_recovery(&hot, 4.0, &[(SETS, 20_000, 40), (SMALL_DRIFT, 6_000, 20)]);
        assert_eq!(
            outcome.heavy_in_first_phase, 0,
            "hot keys must not be heavy at 30:60"
        );
        assert!(
            outcome.entries.len() >= hot.len(),
            "entries {:?}",
            outcome.entries
        );
        assert_eq!(
            outcome.spills_after_entries,
            [0, 0],
            "spurious spills after entry"
        );
    }

    /// R10-2: TP2 recovers from 30 to 60 workers, a material change, so the window
    /// restarts and the keys' credit from 30:30 is gone (the oracle restarts with it).
    /// They become heavy at 30:60 from their own load, their fresh-window credit is
    /// removed exactly, and nothing spills.
    #[test]
    fn heavy_entry_after_capacity_recovery_starts_from_a_fresh_window() {
        let even = [(TP4, 30.0), (TP2, 30.0)];
        let hot = stable_keys(&even, &SETS, 2);
        // Each hot key sends 5% of requests at 6x the background charge: below the 1/8
        // band at 30:30, above the 1/12 band at 30:60.
        let outcome = run_recovery(&hot, 6.0, &[(even, 20_000, 20), (SETS, 6_000, 20)]);
        assert_eq!(
            outcome.heavy_in_first_phase, 0,
            "hot keys must not be heavy at 30:30"
        );
        assert!(
            outcome.entries.len() >= hot.len(),
            "entries {:?}",
            outcome.entries
        );
        assert_eq!(outcome.spills, vec![[0, 0], [0, 0]], "spurious spills");
    }

    /// R7-2: heavy entry after a winner change. Four keys whose winner is TP2 at 30:60 and
    /// TP4 after a small drift to 30:58 (the window is kept) send one request each per 30
    /// at 3x charge (7.9% of load each, not heavy), then one per 12 (15%). They are
    /// affinity-routed to their new winner until they qualify, so at entry they hold
    /// demand credit on both sets. All of it, including the TP2 demand from before the
    /// change, leaves the window (checked against the independent oracle), and no
    /// background key spills afterwards.
    #[test]
    fn heavy_entry_after_winner_change_removes_credit_on_both_winners() {
        let hot: Vec<u64> = (0..u64::MAX)
            .map(mixed)
            .filter(|k| {
                rendezvous_pick(*k, &SETS) == Some(1)
                    && rendezvous_pick(*k, &SMALL_DRIFT) == Some(0)
            })
            .take(4)
            .collect();
        let outcome = run_recovery(&hot, 3.0, &[(SETS, 20_000, 30), (SMALL_DRIFT, 6_000, 12)]);
        assert_eq!(
            outcome.heavy_in_first_phase, 0,
            "hot keys must not be heavy at 30:60"
        );
        assert!(
            outcome.entries.len() >= hot.len(),
            "entries {:?}",
            outcome.entries
        );
        assert!(
            outcome.coincident >= hot.len(),
            "entries without credit on the old winner: {}",
            outcome.coincident
        );
        assert_eq!(
            outcome.spills_after_entries,
            [0, 0],
            "spurious spills after entry"
        );
    }

    /// R7-1: a TP4-preferring key that starts after warm-up and sends one request in 25 at
    /// 8x the background charge carries about a quarter of the load with 4% of requests.
    /// It must qualify as heavy by load (the request floor only asks for repeated
    /// observations), so TP4 stays in band even though its spill point keeps it from
    /// spilling. A request-share floor of half the band (4.2%) would never admit it.
    #[test]
    fn load_heavy_infrequent_key_qualifies_and_keeps_band() {
        let hot = key_preferring(TP4, |x| x > 0.25);
        let tracker = ShareTracker::default();
        let mut tally = Tally::default();
        for i in 0..35_000u64 {
            let (key, charge) = if i >= 5_000 && i.is_multiple_of(25) {
                (hot, 8.0)
            } else {
                (mixed(i + 60_000_000), 1.0)
            };
            let (idx, _) = tracker
                .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                .unwrap();
            if i >= 20_000 {
                tally.add(idx, charge);
            }
        }
        assert!(tracker.is_heavy(hot, &SETS, DEFAULT_SLACK));
        tally.assert_in_band(0.0, "8x key every 25 requests");
    }

    /// A `prompt_cache_key` that prefers TP4 and whose spill point is above the slack (so
    /// the sticky spill can never move it).
    fn unspillable_tp4_cache_key() -> u64 {
        (0..)
            .map(|n| chat_request_affinity_key(&[], Some(&format!("large-session-{n}")), None))
            .map(Option::unwrap)
            .find(|k| rendezvous_pick(*k, &SETS) == Some(0) && spill_point(*k) > DEFAULT_SLACK)
            .unwrap()
    }

    /// Feed `n` decisions of background (distinct keys, `background` charges) in which,
    /// from decision 5000 on, `hot` sends one request per `every` at `hot_charge`. Returns
    /// the charge tally from decision 20000 on, the decision of the hot key's first
    /// heavy-key request, and whether every hot request after that was heavy.
    fn run_periodic_hot_key(
        hot: u64,
        every: u64,
        hot_charge: f64,
        n: u64,
        mut background: impl FnMut() -> f64,
    ) -> (Tally, Option<u64>, bool) {
        let tracker = ShareTracker::default();
        let mut tally = Tally::default();
        let (mut detected, mut stayed) = (None, true);
        for i in 0..n {
            let is_hot = i >= 5_000 && i.is_multiple_of(every);
            let (key, charge) = if is_hot {
                (hot, hot_charge)
            } else {
                (mixed(i + 80_000_000), background())
            };
            let (idx, reason) = tracker
                .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                .unwrap();
            if is_hot {
                let heavy = reason == SetChoiceReason::HeavyKeyRandom;
                if heavy && detected.is_none() {
                    detected = Some(i);
                } else if detected.is_some() && !heavy {
                    stayed = false;
                }
            }
            if i >= 20_000 {
                tally.add(idx, charge);
            }
        }
        (tally, detected, stayed)
    }

    /// R8-1 / R9-1 (rare huge requests): a key sending one request per 100-300 requests
    /// at 100-300 times the 4 KiB background carries a third to a half of all prompt
    /// bytes. Its clamped charge is small and it repeats less often than the request
    /// branch's 8-observation floor (about 0.8% of requests) allows, but the charge
    /// branch's floor of 3 admits it, so TP4 stays in band by charge.
    #[test]
    fn rare_huge_requests_make_a_key_heavy() {
        let hot = unspillable_tp4_cache_key();
        for (every, hot_charge) in [
            (100u64, 400.0 * 1024.0),
            (150, 100.0 * 4096.0),
            (200, 400.0 * 1024.0),
            (300, 300.0 * 4096.0),
        ] {
            let (tally, detected, stayed) =
                run_periodic_hot_key(hot, every, hot_charge, 50_000, || 4096.0);
            let context = format!("{hot_charge} bytes once per {every} requests");
            let latency = detected.map(|d| d - 5_000);
            // Detected within 8 of its requests (observed: 5-7).
            assert!(
                latency.is_some_and(|l| l <= 8 * every),
                "{context}: detected after {latency:?}"
            );
            assert!(stayed, "{context}: left the heavy state");
            tally.assert_in_band(0.0, &context);
        }
    }

    /// R9-5: rare huge requests amid heavy-tailed (lognormal sigma 1.5 and 2) background.
    /// The charge-share noise grows with the background's spread, which delays detection;
    /// the delay stays bounded, the key stays heavy once detected, and its set stays in
    /// band from then on (by unclamped charge, with a tolerance for the background's own
    /// heavy tail).
    #[test]
    fn rare_huge_requests_are_detected_under_lognormal_background() {
        let hot = unspillable_tp4_cache_key();
        for sigma in [1.5, 2.0] {
            let mean = 2000.0 * (sigma * sigma / 2.0f64).exp();
            // Observed latencies: 1039 / 1402 decisions (1 in 33), 850 / 1000 (1 in 150).
            for (every, factor, bound) in [(33u64, 10.0, 2_500u64), (150, 100.0, 2_000)] {
                let mut rng = Rng(0x9e5 + every);
                let (tally, detected, stayed) =
                    run_periodic_hot_key(hot, every, factor * mean, 50_000, || rng.charge(sigma));
                let context = format!("sigma {sigma}: {factor}x mean once per {every}");
                let latency = detected.map(|d| d - 5_000);
                assert!(
                    latency.is_some_and(|l| l <= bound),
                    "{context}: detected after {latency:?}"
                );
                assert!(stayed, "{context}: left the heavy state");
                tally.assert_in_band(0.03, &context);
            }
        }
    }

    /// R9-4: a counter is taken over by the smallest larger share, not the smallest count.
    /// Under count-only eviction a key repeating once per 300 requests among distinct
    /// background keys is evicted between its requests (128 counters turn over every
    /// ~128 new keys), never accumulates observations, and is never heavy.
    #[test]
    fn rare_heavy_key_keeps_its_counter() {
        let hot = unspillable_tp4_cache_key();
        let tracker = ShareTracker::default();
        let (mut kept, mut checks) = (0, 0);
        for i in 0..20_000u64 {
            let is_hot = i >= 5_000 && i.is_multiple_of(300);
            let (key, charge) = if is_hot {
                (hot, 300.0 * 4096.0)
            } else {
                (mixed(i + 81_000_000), 4096.0)
            };
            tracker
                .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                .unwrap();
            if i > 6_000 && i % 300 == 150 {
                let state = tracker.state.lock();
                kept += usize::from(state.heavy.iter().any(|c| c.key == hot));
                checks += 1;
            }
        }
        assert_eq!(
            kept, checks,
            "the hot key lost its counter between requests"
        );
        assert!(tracker.is_heavy(hot, &SETS, DEFAULT_SLACK));
    }

    /// R8-1 (many small requests): a key sending 15% of requests at 0.05x the mean charge
    /// has a negligible charge share, but its request share makes it heavy, so per-set
    /// request counts stay in band.
    #[test]
    fn frequent_small_requests_make_a_key_heavy() {
        let hot = unspillable_tp4_cache_key();
        let tracker = ShareTracker::default();
        let mut requests = Tally::default();
        for i in 0..40_000u64 {
            let (key, charge) = if i >= 5_000 && i % 20 < 3 {
                (hot, 0.05 * 4096.0)
            } else {
                (mixed(i + 85_000_000), 4096.0)
            };
            let (idx, _) = tracker
                .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                .unwrap();
            if i >= 20_000 {
                requests.add(idx, 1.0);
            }
        }
        assert!(tracker.is_heavy(hot, &SETS, DEFAULT_SLACK));
        requests.assert_in_band(0.0, "15% of requests at 0.05x charge");
    }

    // -- R9-2 / R9-3: balance on requests and bytes, and routed-share feedback --

    /// Production set keys (`worker_set_key` strings) at 30/60 workers.
    const PROD_TP4: &str = r#"["dynamo_tp4","backend","generate","chat|completions","aggregated"]"#;
    const PROD_TP2: &str = r#"["dynamo_tp2","backend","generate","chat|completions","aggregated"]"#;
    const PROD_SETS: [(&str, f64); 2] = [(PROD_TP4, 30.0), (PROD_TP2, 60.0)];

    fn cache_key(name: &str) -> u64 {
        chat_request_affinity_key(&[], Some(name), None).unwrap()
    }

    /// Charge and request tallies of one run.
    #[derive(Default)]
    struct Tallies {
        charge: Tally,
        requests: Tally,
    }

    impl Tallies {
        fn add(&mut self, idx: usize, charge: f64) {
            self.charge.add(idx, charge);
            self.requests.add(idx, 1.0);
        }

        fn assert_in_band(&self, tolerance: f64, context: &str) {
            self.charge
                .assert_in_band(tolerance, &format!("{context} (charge)"));
            self.requests
                .assert_in_band(tolerance, &format!("{context} (requests)"));
        }
    }

    /// R9-2 (sol): four `prompt_cache_key`s that all prefer TP4 under the production set
    /// keys each send 7.5% of requests at 0.05x the background charge. None reaches the
    /// band alone, and together they would put 53% of requests on TP4. Their spill points
    /// (0.42-0.93) leave no spill fraction that fixes requests without pushing TP4's
    /// bytes below band, so they must be classified (lopsided: request share above half
    /// the band at a 14x smaller charge share) and spread by the weighted random pick.
    #[test]
    fn several_small_keys_keep_requests_and_bytes_in_band() {
        let hot: Vec<u64> = [0, 2, 5, 9]
            .iter()
            .map(|n| cache_key(&format!("session-{n}")))
            .collect();
        assert!(
            hot.iter()
                .all(|k| rendezvous_pick(*k, &PROD_SETS) == Some(0))
        );
        for sigma in [0.0, 1.0] {
            let tracker = ShareTracker::default();
            let mut rng = Rng(0x92 + sigma as u64);
            let mut tallies = Tallies::default();
            for i in 0..40_000u64 {
                let slot = (i % 40) as usize;
                let (key, charge) = if slot < 12 {
                    (hot[slot % 4], 0.05 * 4096.0)
                } else {
                    let charge = if sigma > 0.0 {
                        rng.charge(sigma)
                    } else {
                        4096.0
                    };
                    (mixed(i + 82_000_000), charge)
                };
                let (idx, _) = tracker
                    .choose(SetAffinity::new(key, charge), &PROD_SETS, DEFAULT_SLACK)
                    .unwrap();
                if i >= 20_000 {
                    tallies.add(idx, charge);
                }
            }
            for k in &hot {
                assert!(
                    tracker.is_heavy(*k, &PROD_SETS, DEFAULT_SLACK),
                    "sigma {sigma}"
                );
            }
            tallies.assert_in_band(0.0, &format!("4 small keys, sigma {sigma}"));
        }
    }

    /// R9-2: many TP4-preferring conversations that send small requests often (30% of
    /// requests from 200 keys at 0.3x charge) overload TP4 on requests (53%) while its
    /// bytes stay in band (39%). No key is heavy; the request-count demand drives the
    /// sticky spill, which a charge-only controller would leave idle.
    #[test]
    fn request_count_imbalance_drives_the_spill() {
        let chatty: Vec<u64> = (0..u64::MAX)
            .map(mixed)
            .filter(|k| rendezvous_pick(*k, &SETS) == Some(0))
            .take(200)
            .collect();
        let tracker = ShareTracker::default();
        let mut rng = Rng(0xc4a7);
        let mut tallies = Tallies::default();
        let mut spills = 0;
        for i in 0..40_000u64 {
            let (key, charge) = if i % 10 < 3 {
                (chatty[(rng.next_u64() % 200) as usize], 0.3 * 4096.0)
            } else {
                (mixed(i + 83_000_000), 4096.0)
            };
            let (idx, reason) = tracker
                .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                .unwrap();
            assert_ne!(reason, SetChoiceReason::HeavyKeyRandom);
            if i >= 20_000 {
                tallies.add(idx, charge);
                spills += usize::from(reason == SetChoiceReason::ShareCapFallback);
            }
        }
        assert!(spills > 0);
        tallies.assert_in_band(0.0, "chatty TP4 conversations");
    }

    /// A persistent pool: `keys` conversations of equal expected size (per-request
    /// lognormal `sigma` noise around 4 KiB, no turnover), each request from a uniformly
    /// random conversation through one of `frontends` trackers. Returns the tallies of
    /// the last third, the fraction of requests in the last third whose set differs from
    /// the conversation's previous request, and the number of heavy-key requests.
    fn persistent_pool(
        keys: &[u64],
        sets: &[(&str, f64); 2],
        frontends: usize,
        requests: u64,
        sigma: f64,
        seed: u64,
    ) -> (Tallies, f64, usize) {
        let mut rng = Rng(seed);
        let trackers: Vec<ShareTracker> = (0..frontends).map(|_| ShareTracker::default()).collect();
        let mut last = vec![usize::MAX; keys.len()];
        let mut tallies = Tallies::default();
        let (mut changes, mut heavy) = (0usize, 0usize);
        let tail = requests - requests / 3;
        for i in 0..requests {
            let c = (rng.next_u64() % keys.len() as u64) as usize;
            let charge = 4096.0 * (sigma * rng.normal()).exp();
            let frontend = (rng.next_u64() % frontends as u64) as usize;
            let (idx, reason) = trackers[frontend]
                .choose(SetAffinity::new(keys[c], charge), sets, DEFAULT_SLACK)
                .unwrap();
            heavy += usize::from(reason == SetChoiceReason::HeavyKeyRandom);
            if i >= tail {
                tallies.add(idx, charge);
                changes += usize::from(idx != last[c]);
            }
            last[c] = idx;
        }
        (tallies, changes as f64 / (requests / 3) as f64, heavy)
    }

    /// R9-3 (sol): 12 persistent equal conversations with `prompt_cache_key`s session-282
    /// to session-293 under the production set keys. Seven prefer TP4 (58% of load), and
    /// every one of their spill points exceeds the slack, so the sticky spill alone never
    /// moves one. The routed-share feedback raises TP4's spill fraction past them until
    /// TP4 is back in band, then holds: the pool settles (no set changes in the last
    /// third) and nobody is heavy.
    #[test]
    fn persistent_pool_converges_into_band() {
        let sol: Vec<u64> = (282..294)
            .map(|n| cache_key(&format!("session-{n}")))
            .collect();
        let tp4 = sol
            .iter()
            .filter(|k| rendezvous_pick(**k, &PROD_SETS) == Some(0))
            .count();
        assert_eq!(tp4, 7);
        assert!(
            sol.iter()
                .filter(|k| rendezvous_pick(**k, &PROD_SETS) == Some(0))
                .all(|k| spill_point(*k) > DEFAULT_SLACK)
        );
        for sigma in [0.0, 0.5] {
            let (tallies, changes, heavy) =
                persistent_pool(&sol, &PROD_SETS, 1, 40_000, sigma, 0x282);
            let context = format!("sol's 12 conversations, sigma {sigma}");
            assert_eq!(heavy, 0, "{context}");
            assert!(changes < 0.001, "{context}: {changes} set changes");
            tallies.assert_in_band(0.01, &context);
        }
    }

    /// R9-3: the boost lifts TP4's spill fraction past the slack while sol's persistent
    /// pool keeps TP4 above its band, and falls back to nothing once the traffic turns
    /// into many balanced keys (TP4 is then below its fair share), so it leaves no
    /// residual spill behind.
    #[test]
    fn boost_rises_past_slack_and_falls_back() {
        let sol: Vec<u64> = (282..294)
            .map(|n| cache_key(&format!("session-{n}")))
            .collect();
        let tracker = ShareTracker::default();
        let mut rng = Rng(0xb0057);
        for _ in 0..30_000u64 {
            let key = sol[(rng.next_u64() % 12) as usize];
            tracker
                .choose(SetAffinity::new(key, 4096.0), &PROD_SETS, DEFAULT_SLACK)
                .unwrap();
        }
        let p = tracker.spill_fraction_of(0, &PROD_SETS, DEFAULT_SLACK);
        assert!(p > DEFAULT_SLACK, "p {p}");
        for i in 0..30_000u64 {
            tracker
                .choose(
                    SetAffinity::new(mixed(i + 84_000_000), 4096.0),
                    &PROD_SETS,
                    DEFAULT_SLACK,
                )
                .unwrap();
        }
        assert!(tracker.state.lock().boost.is_empty());
        assert_eq!(tracker.spill_fraction_of(0, &PROD_SETS, DEFAULT_SLACK), 0.0);
    }

    /// R9-3: persistent pools of 12 and 24 conversations over several key sets (each
    /// drawn so that one set is out of band by rendezvous alone) converge into band.
    #[test]
    fn persistent_pools_converge_for_several_key_sets() {
        for n in [12usize, 24] {
            let mut tried = 0;
            for seed in 0..200u64 {
                let keys: Vec<u64> = (0..n as u64).map(|k| mixed(seed * 1_000 + k + 7)).collect();
                let tp4 = keys
                    .iter()
                    .filter(|k| rendezvous_pick(**k, &SETS) == Some(0))
                    .count() as f64
                    / n as f64;
                let band = share_band(1.0 / 3.0, DEFAULT_SLACK);
                if (tp4 - 1.0 / 3.0).abs() <= band {
                    continue;
                }
                let (tallies, changes, heavy) = persistent_pool(&keys, &SETS, 1, 40_000, 0.5, seed);
                let context = format!("{n} conversations, seed {seed}, tp4 by hashing {tp4:.3}");
                assert_eq!(heavy, 0, "{context}");
                assert!(changes < 0.005, "{context}: {changes} set changes");
                tallies.assert_in_band(0.01, &context);
                tried += 1;
                if tried == 4 {
                    break;
                }
            }
            assert_eq!(tried, 4, "{n} conversations: too few unbalanced key sets");
        }
    }

    /// R9-3: the same persistent pool behind 30 independent frontends converges too, and
    /// the frontends agree: a conversation keeps its set from one request to the next
    /// (whichever frontend serves it) on almost every request of the last third. Twelve
    /// equal conversations sit exactly at the 1/12 band, so over 30 frontends x 10k
    /// decisions a rare (about 4.7 sigma) excursion can classify one conversation heavy on
    /// one frontend, where it then stays (exit is at half the band); that is bounded here.
    #[test]
    fn persistent_pool_converges_across_thirty_frontends() {
        let sol: Vec<u64> = (282..294)
            .map(|n| cache_key(&format!("session-{n}")))
            .collect();
        let (tallies, changes, heavy) = persistent_pool(&sol, &PROD_SETS, 30, 300_000, 0.5, 0x30);
        assert!(heavy < 300_000 / 200, "{heavy} heavy-key requests");
        assert!(changes < 0.01, "{changes} set changes");
        tallies.assert_in_band(0.01, "30 frontends");
    }

    // -- R10: symmetric boost deadband, capacity changes, takeover ledger, steady state --

    /// Per conversation, the number of set changes between consecutive requests of the
    /// last half that were both not heavy-key picks, from a persistent pool of equal
    /// conversations (unit lognormal(0.25) noise around 4 KiB) spread over `frontends`.
    /// Also returns the tallies of the last half and the number of heavy-key requests.
    fn pool_flips(
        keys: &[u64],
        frontends: usize,
        requests: u64,
        seed: u64,
    ) -> (Vec<usize>, Tallies, usize) {
        let mut rng = Rng(seed);
        let trackers: Vec<ShareTracker> = (0..frontends).map(|_| ShareTracker::default()).collect();
        let mut last = vec![usize::MAX; keys.len()];
        let mut flips = vec![0usize; keys.len()];
        let mut tallies = Tallies::default();
        let mut heavy = 0;
        for i in 0..requests {
            let c = (rng.next_u64() % keys.len() as u64) as usize;
            let charge = 4096.0 * (0.25 * rng.normal()).exp();
            let frontend = (rng.next_u64() % frontends as u64) as usize;
            let (idx, reason) = trackers[frontend]
                .choose(SetAffinity::new(keys[c], charge), &SETS, DEFAULT_SLACK)
                .unwrap();
            if i >= requests / 2 {
                tallies.add(idx, charge);
            }
            if reason == SetChoiceReason::HeavyKeyRandom {
                heavy += 1;
                last[c] = usize::MAX;
                continue;
            }
            if i >= requests / 2 {
                flips[c] += usize::from(last[c] != usize::MAX && idx != last[c]);
            }
            last[c] = idx;
        }
        (flips, tallies, heavy)
    }

    /// A tracker whose window holds a TP4 boost of 0.4, fair shares 1:2, an effective
    /// sample size of 2000 on both measures (two standard deviations of a 1/3 share are
    /// about 0.021), TP4 demand share `demand` and TP4 routed share `routed` (both
    /// measures alike).
    fn boost_state(demand: f64, routed: f64) -> ShareState {
        let split = |tp4: f64, total: f64| {
            HashMap::from([
                (TP4.to_string(), total * tp4),
                (TP2.to_string(), total * (1.0 - tp4)),
            ])
        };
        ShareState {
            decisions: 1000.0,
            charge_sum: 1000.0,
            window_decisions: 1000.0,
            window_decisions_sq: 500.0,
            window_sum: 1000.0,
            window_sq_sum: 500.0,
            expected: split(1.0 / 3.0, 1000.0),
            expected_requests: split(1.0 / 3.0, 1000.0),
            demand: split(demand, 1000.0),
            demand_requests: split(demand, 1000.0),
            routed: split(routed, 1000.0),
            routed_requests: split(routed, 1000.0),
            boost: HashMap::from([(TP4.to_string(), 0.4)]),
            ..Default::default()
        }
    }

    /// R10-1 (grok): the boost's deadband is symmetric. A whole-key move that lands just
    /// below fair share (here 1/3 − 0.01, within the 0.021 noise allowance) holds the
    /// boost; the old rule shrank it there, so noise undid the move, the key came back,
    /// and the set went over its band again (a limit cycle, out of phase across
    /// frontends). The boost still shrinks below fair − allowance, grows above the band
    /// edge + allowance, and leaks once the set's demand alone is clearly within band.
    #[test]
    fn boost_deadband_is_symmetric() {
        let fair = 1.0 / 3.0;
        let step = |demand: f64, routed: f64| {
            let mut state = boost_state(demand, routed);
            state.update_boost(&SETS, &[0, 1], DEFAULT_SLACK);
            ShareState::stat(&state.boost, TP4) - 0.4
        };
        // TP4's demand alone (4/9) is over its band: the boost is needed.
        let needed = 4.0 / 9.0;
        assert_eq!(step(needed, fair - 0.01), 0.0, "hold just below fair");
        assert_eq!(step(needed, fair), 0.0, "hold at fair");
        assert_eq!(step(needed, fair + 0.1), 0.0, "hold inside the band");
        assert!(
            step(needed, fair - 0.04) < 0.0,
            "shrink below fair - allowance"
        );
        assert!(
            step(needed, 0.47) > 0.0,
            "grow above the band edge + allowance"
        );
        // Demand just under the band edge is not clearly within band: hold.
        assert_eq!(step(0.41, fair), 0.0, "hold while demand is near the edge");
        // Demand at fair share: the boost is not needed and leaks at gain x band.
        let leak = step(fair, fair);
        let band = share_band(fair, DEFAULT_SLACK);
        assert!(
            (leak + SPILL_BOOST_GAIN * band).abs() < 1e-12,
            "leak {leak}"
        );
    }

    /// R10-1: nine equal persistent conversations, four preferring TP4 (4/9, above the
    /// boost's grow threshold), over several key sets, on 1 and on 30 frontends. A 1/9
    /// share is above the 1/12 band, so after a small excursion past the noise allowance
    /// every one of them is classified heavy and spread at random (the case grok's limit
    /// cycle needs, a non-heavy key whose move lands on fair share, only exists in a
    /// sliver just below the heavy threshold; `boost_deadband_is_symmetric` pins the
    /// deadband itself). Before that, affinity-routed requests change set only a bounded
    /// number of times, and both measures stay in band.
    #[test]
    fn nine_equal_conversations_stay_in_band() {
        let mut tried = 0;
        for seed in 0..10_000u64 {
            let keys: Vec<u64> = (0..9u64).map(|k| mixed(seed * 100 + k + 3)).collect();
            let tp4: Vec<u64> = keys
                .iter()
                .copied()
                .filter(|k| rendezvous_pick(*k, &SETS) == Some(0))
                .collect();
            // Grok's case: no TP4 key low enough for the demand spill (at most ~0.11).
            if tp4.len() != 4 || tp4.iter().any(|k| spill_point(*k) < 0.15) {
                continue;
            }
            for (frontends, requests) in [(1usize, 60_000u64), (30, 600_000)] {
                let (flips, tallies, _) = pool_flips(&keys, frontends, requests, seed);
                let context = format!("seed {seed}, {frontends} frontends");
                assert!(
                    flips.iter().all(|f| *f <= 2 * frontends),
                    "{context}: flips {flips:?}"
                );
                tallies.assert_in_band(0.01, &context);
            }
            tried += 1;
            if tried == 4 {
                break;
            }
        }
        assert_eq!(tried, 4);
    }

    /// R10-2 (sol): 20 equal persistent conversations (session-0 to session-19) under the
    /// production set keys hold an active boost at 30/60 workers. TP2 then drops to 10
    /// workers (TP4's share 1/3 → 3/4): a material change, so the window, targets and
    /// boosts restart. Right after the cut TP2 is not overloaded (sol measured 46% of the
    /// next 1000 requests on TP2 with the stale state, against an upper band of 31.25%),
    /// and once the fresh window has warmed up it stays within its new band (fair 1/4,
    /// band ± 1/16) on both measures.
    #[test]
    fn capacity_cut_restarts_the_spill_state() {
        let keys: Vec<u64> = (0..20)
            .map(|n| cache_key(&format!("session-{n}")))
            .collect();
        let cut = [(PROD_TP4, 30.0), (PROD_TP2, 10.0)];
        let tracker = ShareTracker::default();
        let mut rng = Rng(0xc07);
        let mut run = |sets: &[(&str, f64); 2], n: u64| {
            let mut tallies = Tallies::default();
            for _ in 0..n {
                let key = keys[(rng.next_u64() % 20) as usize];
                let (idx, _) = tracker
                    .choose(SetAffinity::new(key, 4096.0), sets, DEFAULT_SLACK)
                    .unwrap();
                tallies.add(idx, 4096.0);
            }
            tallies
        };
        run(&PROD_SETS, 30_000);
        assert!(
            !tracker.state.lock().boost.is_empty(),
            "no active boost before the cut"
        );
        // While the fresh window warms up (the guard is idle for 200 decisions and
        // rendezvous alone puts a lumpy share on TP2) TP2 may sit below its band, but it
        // is never overloaded; afterwards it is within band on both measures.
        let upper = 0.25 + 0.0625;
        for (phase, n, lower) in [
            ("first 1000 after the cut", 1_000, 0.0),
            ("next 20000", 20_000, 0.25 - 0.0625),
        ] {
            let tallies = run(&cut, n);
            for (what, tally) in [("charge", &tallies.charge), ("requests", &tallies.requests)] {
                let tp2 = tally.share(1);
                assert!(
                    (lower..=upper).contains(&tp2),
                    "{phase}: TP2 {what} share {tp2}"
                );
            }
        }
    }

    // -- R11: heavy traffic and the window, restart baselines, accumulated drift --

    /// R11-1 (sol): one common `prompt_cache_key` sends ~85% of requests at 15k tokens
    /// (heavy: random routing), and 20 equally active sessions (session-0 to session-19)
    /// send 126k-token requests, behind 30 trackers with the production set keys. The
    /// sessions carry ~60% of the bytes and twelve of them prefer TP4. When every decision
    /// decayed the window but only the sessions' added to it, the window settled near 150
    /// decisions, below the guards' warm-up, so neither guard ever acted (49% of bytes on
    /// TP4 against a 41.7% band). The window now ages only with the decisions it holds, so
    /// the guards balance the sessions and both measures stay in band.
    #[test]
    fn heavy_common_key_does_not_disable_balancing() {
        const BYTES_PER_TOKEN: f64 = 4.0;
        let common = cache_key("common-system-prompt");
        let sessions: Vec<u64> = (0..20)
            .map(|n| cache_key(&format!("session-{n}")))
            .collect();
        let on_tp4 = sessions
            .iter()
            .filter(|k| rendezvous_pick(**k, &PROD_SETS) == Some(0))
            .count();
        assert_eq!(on_tp4, 12, "sessions preferring TP4 by rendezvous");
        let trackers: Vec<ShareTracker> = (0..30).map(|_| ShareTracker::default()).collect();
        let mut rng = Rng(0x5011);
        let mut tallies = Tallies::default();
        let requests = 30 * 20_000u64;
        for i in 0..requests {
            let (key, tokens) = if rng.uniform() < 0.85 {
                (common, 15_000.0)
            } else {
                (sessions[(rng.next_u64() % 20) as usize], 126_000.0)
            };
            let charge = tokens * BYTES_PER_TOKEN;
            let frontend = (rng.next_u64() % 30) as usize;
            let (idx, _) = trackers[frontend]
                .choose(SetAffinity::new(key, charge), &PROD_SETS, DEFAULT_SLACK)
                .unwrap();
            if i >= requests / 2 {
                tallies.add(idx, charge);
            }
        }
        for (f, tracker) in trackers.iter().enumerate() {
            assert!(
                tracker.is_heavy(common, &PROD_SETS, DEFAULT_SLACK),
                "frontend {f}"
            );
            let state = tracker.state.lock();
            assert!(
                state.window_decisions >= SHARE_MIN_SAMPLES,
                "frontend {f}: window {}",
                state.window_decisions
            );
        }
        tallies.assert_in_band(0.0, "heavy common key with 20 long sessions");
    }

    /// Whenever the window is cold, no boost is kept.
    fn assert_no_cold_boost(tracker: &ShareTracker, context: &str) {
        let state = tracker.state.lock();
        assert!(
            state.window_decisions >= SHARE_MIN_SAMPLES || state.boost.is_empty(),
            "{context}: boost {:?} with a cold window ({})",
            state.boost,
            state.window_decisions
        );
    }

    /// R11-1 (grok): sol's 12 persistent conversations drive TP4's boost up; then 2000
    /// requests come from a single key (heavy after its first ~100 requests), then the
    /// pool resumes. The heavy run used to keep stepping the boost on frozen shares (+0.3
    /// over the run) while decaying the window below warm-up, so the grown boost was
    /// re-applied when affinity resumed. Now the heavy decisions neither step the boost
    /// nor age the window: the pool resumes against the boost it left, and both measures
    /// stay in band.
    #[test]
    fn heavy_run_neither_grows_nor_strands_the_boost() {
        let sol: Vec<u64> = (282..294)
            .map(|n| cache_key(&format!("session-{n}")))
            .collect();
        let hot = cache_key("one-heavy-key");
        let tracker = ShareTracker::default();
        let mut rng = Rng(0x6e0c);
        let mut pool = |n: u64, tally: bool| {
            let mut tallies = Tallies::default();
            for _ in 0..n {
                let key = sol[(rng.next_u64() % 12) as usize];
                let (idx, _) = tracker
                    .choose(SetAffinity::new(key, 4096.0), &PROD_SETS, DEFAULT_SLACK)
                    .unwrap();
                assert_no_cold_boost(&tracker, "pool");
                if tally {
                    tallies.add(idx, 4096.0);
                }
            }
            tallies
        };
        // Stop while the boost is still climbing (TP4 still above its band).
        pool(500, false);
        let climbing = ShareState::stat(&tracker.state.lock().boost, PROD_TP4);
        pool(100, false);
        let before = ShareState::stat(&tracker.state.lock().boost, PROD_TP4);
        assert!(
            before > climbing && climbing > 0.0,
            "the boost is not climbing ({climbing} -> {before})"
        );
        let mut heavy = 0;
        for _ in 0..2_000 {
            let (_, reason) = tracker
                .choose(SetAffinity::new(hot, 4096.0), &PROD_SETS, DEFAULT_SLACK)
                .unwrap();
            heavy += usize::from(reason == SetChoiceReason::HeavyKeyRandom);
            assert_no_cold_boost(&tracker, "heavy run");
        }
        assert!(heavy >= 1_600, "heavy decisions {heavy}");
        let after = ShareState::stat(&tracker.state.lock().boost, PROD_TP4);
        // Only the hot key's affinity-routed requests before it became heavy could step
        // the boost, each by at most the gain (the excess is at most 1).
        let routed = 2_000 - heavy;
        assert!(
            after <= before + routed as f64 * SPILL_BOOST_GAIN,
            "the boost grew from {before} to {after} over {heavy} heavy decisions"
        );
        assert!(tracker.state.lock().window_decisions >= SHARE_MIN_SAMPLES);
        pool(20_000, false);
        pool(20_000, true).assert_in_band(0.01, "pool after the heavy run");
    }

    /// R11-1: a boost never survives a cold window. A heavy entry can remove most of a
    /// young window; the next decision, of either kind, drops the boost (it is not kept
    /// frozen for a later load mix), and a heavy decision on a warm window leaves it as
    /// it is.
    #[test]
    fn cold_window_drops_the_boost() {
        let record = |state: &mut ShareState, affinity_routed: bool| {
            state.record(
                mixed(1),
                &SETS,
                &[0, 1],
                0,
                affinity_routed,
                1.0,
                1.0 / 12.0,
                DEFAULT_SLACK,
            );
        };
        for affinity_routed in [false, true] {
            let mut state = boost_state(4.0 / 9.0, 1.0 / 3.0);
            state.window_decisions = 150.0;
            state.window_decisions_sq = 75.0;
            record(&mut state, affinity_routed);
            assert!(
                state.boost.is_empty(),
                "routed {affinity_routed}: {:?}",
                state.boost
            );
        }
        let mut state = boost_state(4.0 / 9.0, 1.0 / 3.0);
        record(&mut state, false);
        assert_eq!(ShareState::stat(&state.boost, TP4), 0.4);
        assert_eq!(
            state.window_decisions, 1000.0,
            "a heavy decision aged the window"
        );
    }

    /// R11-2 (opus): a "fewer than two eligible sets" restart re-baselines the capacity
    /// threshold. TP2 goes not-ready (only TP4 eligible), returns at 64 workers (TP4's
    /// share 0.319, within 0.02 of the old 0.333 baseline), then drops to 56 (0.349: 0.030
    /// from the restart point, but only 0.016 from the stale baseline). The drop restarts
    /// the window.
    #[test]
    fn stale_restart_rebaselines_the_capacity_threshold() {
        let tracker = ShareTracker::default();
        let mut key = 0u64;
        let mut run = |sets: &[(&str, f64); 2], n: u64| {
            for _ in 0..n {
                key += 1;
                tracker
                    .choose(unit(mixed(key + 88_000_000)), sets, DEFAULT_SLACK)
                    .unwrap();
            }
        };
        run(&SETS, 3_000);
        tracker.mark_single_eligible();
        run(&[(TP4, 30.0), (TP2, 64.0)], 1);
        assert!(tracker.state.lock().window_decisions <= 1.0);
        run(&[(TP4, 30.0), (TP2, 64.0)], 3_000);
        assert!(tracker.state.lock().window_decisions > 900.0);
        run(&[(TP4, 30.0), (TP2, 56.0)], 1);
        assert!(
            tracker.state.lock().window_decisions <= 1.0,
            "64 -> 56 after a stale restart did not restart the window"
        );
    }

    /// R11-3 (opus): capacity arrives one worker at a time. TP2 drops from 60 to 10
    /// workers, one worker per 50 decisions, under 20 persistent conversations that hold
    /// an active boost. No single step moves TP4's share by 0.02 (the largest, 11 → 10, is
    /// 0.018), but the drift accumulates against the last restart's baseline, so the
    /// window restarts several times on the way down; once TP2 is at 10 workers it stays at
    /// or below its new upper band (fair 1/4 + 1/16) on both measures. Comparing with the
    /// previous call's shares instead never restarts and overloads TP2 with stale state.
    #[test]
    fn capacity_drift_accumulates_to_a_restart() {
        let keys: Vec<u64> = (0..20)
            .map(|n| cache_key(&format!("session-{n}")))
            .collect();
        let tracker = ShareTracker::default();
        let mut rng = Rng(0xd41f);
        let restarts = std::cell::Cell::new(0usize);
        let mut run = |sets: &[(&str, f64); 2], n: u64| {
            let mut tallies = Tallies::default();
            for _ in 0..n {
                let key = keys[(rng.next_u64() % 20) as usize];
                let baseline = tracker.state.lock().baseline_shares.clone();
                let (idx, _) = tracker
                    .choose(SetAffinity::new(key, 4096.0), sets, DEFAULT_SLACK)
                    .unwrap();
                let restarted = tracker.state.lock().baseline_shares != baseline;
                restarts.set(restarts.get() + usize::from(restarted));
                tallies.add(idx, 4096.0);
            }
            tallies
        };
        run(&PROD_SETS, 30_000);
        assert!(
            !tracker.state.lock().boost.is_empty(),
            "no active boost before the drift"
        );
        let first = restarts.get();
        for workers in (10..60).rev() {
            run(&[(PROD_TP4, 30.0), (PROD_TP2, f64::from(workers))], 50);
        }
        let during = restarts.get() - first;
        assert!(
            during >= 1,
            "no restart while TP2 drifted from 60 to 10 workers"
        );
        let after = run(&[(PROD_TP4, 30.0), (PROD_TP2, 10.0)], 2_000);
        for (what, tally) in [("charge", &after.charge), ("requests", &after.requests)] {
            let tp2 = tally.share(1);
            assert!(
                tp2 <= 0.25 + 0.0625,
                "TP2 {what} share {tp2} after {during} restarts"
            );
        }
    }

    /// Window totals equal the sums of all counters' credits (the ledger is closed).
    fn assert_ledger_closed(tracker: &ShareTracker, sets: &[(&str, f64); 2], context: &str) {
        let state = tracker.state.lock();
        let close = |window: f64, credits: f64, what: &str| {
            assert!(
                (window - credits).abs() <= 1e-4 + 1e-9 * window.abs(),
                "{context}: {what}: window {window} != credits {credits}"
            );
        };
        let credit = |list: fn(&HeavyCounter) -> &Vec<(u64, f64)>, name: &str| -> f64 {
            state
                .heavy
                .iter()
                .flat_map(|c| list(c).iter())
                .filter(|(id, _)| *id == set_id(name))
                .map(|(_, v)| *v)
                .sum()
        };
        for (name, _) in sets {
            for (what, map, list) in [
                (
                    "demand",
                    &state.demand,
                    (|c| &c.window_demand) as fn(&HeavyCounter) -> &Vec<(u64, f64)>,
                ),
                ("expected", &state.expected, |c| &c.window_expected),
                ("demand_requests", &state.demand_requests, |c| {
                    &c.window_demand_requests
                }),
                ("expected_requests", &state.expected_requests, |c| {
                    &c.window_expected_requests
                }),
                ("routed", &state.routed, |c| &c.window_routed),
                ("routed_requests", &state.routed_requests, |c| {
                    &c.window_routed_requests
                }),
            ] {
                close(
                    ShareState::stat(map, name),
                    credit(list, name),
                    &format!("{what}[{name}]"),
                );
            }
        }
        let sum = |f: fn(&HeavyCounter) -> f64| -> f64 { state.heavy.iter().map(f).sum() };
        close(state.window_sq_sum, sum(|c| c.window_sq), "window_sq_sum");
        close(
            state.window_decisions,
            sum(|c| c.window_decisions),
            "window_decisions",
        );
        close(
            state.window_decisions_sq,
            sum(|c| c.window_decisions_sq),
            "window_decisions_sq",
        );
    }

    /// R10-3 (grok): a Space-Saving takeover used to drop the victim's window credit, so it
    /// stayed in the window with no counter able to remove it. The taker now inherits it
    /// (as it inherits the counts as error), so the window always equals the sum of all
    /// counters' credits, through thousands of takeovers and the heavy entry of a key
    /// that was evicted earlier: when it enters, every credit its counter holds leaves the
    /// window and none is orphaned.
    #[test]
    fn takeover_keeps_the_window_ledger_closed() {
        let hot = key_preferring(TP4, |x| x > 0.5);
        let tracker = ShareTracker::default();
        let (mut evicted, mut was_tracked, mut entered) = (false, false, false);
        for i in 0..20_000u64 {
            // The hot key sends a few early requests, goes quiet long enough to be evicted
            // by distinct background keys, then turns heavy (one request in 5).
            let is_hot = (i < 3_000 && i.is_multiple_of(50)) || (i >= 8_000 && i.is_multiple_of(5));
            let key = if is_hot { hot } else { mixed(i + 86_000_000) };
            let charge = if is_hot { 2.0 } else { 1.0 };
            let (_, reason) = tracker
                .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                .unwrap();
            let tracked = tracker.state.lock().heavy.iter().any(|c| c.key == hot);
            evicted |= was_tracked && !tracked;
            was_tracked = tracked;
            entered |= is_hot && reason == SetChoiceReason::HeavyKeyRandom;
            if i % 97 == 0 || (is_hot && (8_000..9_000).contains(&i)) {
                assert_ledger_closed(&tracker, &SETS, &format!("decision {i}"));
            }
        }
        assert!(evicted, "the hot key was never evicted");
        assert!(entered, "the hot key never became heavy");
        assert_ledger_closed(&tracker, &SETS, "end");
    }

    /// R10-6 (opus): long steady state, 30 frontends with 20k+ decisions each, a pool of
    /// 3000 concurrent conversations of 10 turns (context growing by turn) with turnover,
    /// lognormal sigma 1.5 and 2 per-conversation size. Follow-up turns keep their set,
    /// both measures stay in band, nobody is heavy, and every frontend ends with an empty
    /// boost map.
    #[test]
    fn long_steady_state_across_thirty_frontends() {
        for sigma in [1.5, 2.0] {
            let mut rng = Rng(0x10_0000 + (sigma * 10.0) as u64);
            let trackers: Vec<ShareTracker> = (0..30).map(|_| ShareTracker::default()).collect();
            let mut next_key = 0u64;
            let mut new_conversation = |rng: &mut Rng| {
                next_key += 1;
                // (key, size, turn, last set); start at a random turn to spread turnover.
                (
                    mixed(next_key + 87_000_000),
                    rng.charge(sigma),
                    rng.next_u64() % 10,
                    usize::MAX,
                )
            };
            let mut pool: Vec<(u64, f64, u64, usize)> =
                (0..3000).map(|_| new_conversation(&mut rng)).collect();
            let requests = 30 * 21_000u64;
            let (mut changes, mut follow_ups, mut heavy) = (0usize, 0usize, 0usize);
            let mut tallies = Tallies::default();
            for i in 0..requests {
                let slot = (rng.next_u64() % pool.len() as u64) as usize;
                let (key, size, turn, last) = pool[slot];
                let charge = size * (turn + 1) as f64;
                let frontend = (rng.next_u64() % 30) as usize;
                let (idx, reason) = trackers[frontend]
                    .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                    .unwrap();
                heavy += usize::from(reason == SetChoiceReason::HeavyKeyRandom);
                if i >= requests / 2 {
                    tallies.add(idx, charge);
                    if last != usize::MAX {
                        follow_ups += 1;
                        changes += usize::from(idx != last);
                    }
                }
                pool[slot] = if turn + 1 >= 10 {
                    new_conversation(&mut rng)
                } else {
                    (key, size, turn + 1, idx)
                };
            }
            let rate = changes as f64 / follow_ups as f64;
            let context = format!("sigma {sigma}");
            assert_eq!(heavy, 0, "{context}: heavy-key requests");
            assert!(
                rate < 0.005,
                "{context}: {:.3}% of follow-ups changed set",
                rate * 100.0
            );
            tallies
                .charge
                .assert_in_band(0.03, &format!("{context} (charge)"));
            tallies
                .requests
                .assert_in_band(0.0, &format!("{context} (requests)"));
            for (f, tracker) in trackers.iter().enumerate() {
                let boost = tracker.state.lock().boost.clone();
                assert!(boost.is_empty(), "{context}: frontend {f} boost {boost:?}");
            }
        }
    }

    /// The subtraction bug moved the hot key's whole history out of its winner's demand,
    /// which drives that set's demand toward 0 and makes the *other* set's keys spill.
    /// Spills of the hot key's own set before it is detected are a real response to its
    /// load and stay rare.
    fn assert_no_spurious_spill(outcome: &PhaseOutcome, hot_set: usize, context: &str) {
        assert!(
            outcome.entries_checked >= 1,
            "{context}: no heavy entry checked"
        );
        let other = 1 - hot_set;
        assert_eq!(
            outcome.spills_from[other], 0,
            "{context}: keys of the other set spilled"
        );
        assert!(
            (outcome.spills_from[hot_set] as f64) < 0.01 * outcome.decisions as f64,
            "{context}: {} spills from the hot key's set",
            outcome.spills_from[hot_set]
        );
    }

    fn phase(hot_every: u64, hot_charge: f64, decisions: u64) -> Phase {
        Phase {
            sets: SETS,
            hot_every,
            hot_charge,
            decisions,
            reset_before: false,
        }
    }

    /// Heavy → non-heavy → heavy. The key cools below the exit threshold while it keeps
    /// sending large requests, then re-enters: only the load credited to the window since
    /// it left the heavy state is removed, not its random-routed history.
    #[test]
    fn heavy_reentry_removes_only_its_window_credit() {
        let hot = key_preferring(TP4, |x| x > 0.5);
        let outcome = run_phases(
            hot,
            &[
                phase(3, 8.0, 8_000),
                // 1 in 400 requests at 8x: about 2% of load, below a quarter of the band
                // (the lopsided exit), so it exits and is affinity-routed (credited) until
                // it heats up again.
                phase(400, 8.0, 8_000),
                phase(10, 8.0, 6_000),
            ],
        );
        let transitions = outcome
            .hot_heavy
            .windows(2)
            .filter(|w| w[0] != w[1])
            .count();
        assert!(
            transitions >= 3,
            "expected heavy, non-heavy, heavy: {transitions}"
        );
        assert!(outcome.hot_heavy.last().copied().unwrap_or(false));
        assert!(outcome.entries_checked >= 2);
        assert_no_spurious_spill(&outcome, 0, "heavy re-entry");
        outcome.tally.assert_in_band(0.03, "heavy re-entry");
    }

    /// Heavy entry shortly after a window restart: the key's load from before the restart
    /// is not in the fresh window and must not be subtracted from it.
    #[test]
    fn heavy_entry_after_window_reset_keeps_the_window_intact() {
        let hot = key_preferring(TP4, |x| x > 0.5);
        let outcome = run_phases(
            hot,
            &[
                // ~7% of requests: a large counter, but below the band plus noise.
                phase(14, 1.0, 6_000),
                Phase {
                    reset_before: true,
                    ..phase(14, 1.0, 300)
                },
                phase(3, 1.0, 6_000),
            ],
        );
        assert!(outcome.hot_heavy.last().copied().unwrap_or(false));
        assert_no_spurious_spill(&outcome, 0, "heavy entry after reset");
        outcome
            .tally
            .assert_in_band(0.03, "heavy entry after reset");
    }

    /// The hot key's rendezvous winner changes with a small worker-count change (TP2 60 →
    /// 58 workers: not material, so the window is kept). Its credit to the old winner is
    /// kept across the change, and on heavy entry all of its credit (old and new winner)
    /// leaves the window.
    #[test]
    fn heavy_entry_after_winner_change_removes_all_credit() {
        let hot = (0..u64::MAX)
            .map(mixed)
            .find(|k| {
                rendezvous_pick(*k, &SETS) == Some(1)
                    && rendezvous_pick(*k, &SMALL_DRIFT) == Some(0)
            })
            .unwrap();
        let outcome = run_phases(
            hot,
            &[
                // Large but infrequent: credited to TP2 without becoming heavy.
                phase(60, 8.0, 6_000),
                Phase {
                    sets: SMALL_DRIFT,
                    ..phase(10, 8.0, 6_000)
                },
            ],
        );
        assert!(outcome.hot_heavy.last().copied().unwrap_or(false));
        // The tally mixes two weightings, so check the spill signature only.
        assert_no_spurious_spill(&outcome, 0, "heavy entry after winner change");
    }

    /// A mean-sized hot key that starts after 20k decisions of lognormal background is
    /// still admitted to the table (Space-Saving takes over the lightest counter and
    /// tracks its error), becomes heavy, and keeps the sets in band.
    #[test]
    fn late_hot_key_is_detected_under_lognormal_background() {
        for sigma in [1.0, 1.5] {
            let tracker = ShareTracker::default();
            let mut rng = Rng(0x1a7e);
            for i in 0..20_000u64 {
                tracker
                    .choose(
                        SetAffinity::new(mixed(i), rng.charge(sigma)),
                        &SETS,
                        DEFAULT_SLACK,
                    )
                    .unwrap();
            }
            let hot = key_preferring(TP4, |x| x > 0.5);
            let mean_charge = 2000.0 * (sigma * sigma / 2.0f64).exp();
            let (tally, _) = run_with_hot_key(&tracker, hot, mean_charge, sigma, 10_000);
            assert!(tracker.is_heavy(hot, &SETS, DEFAULT_SLACK), "sigma {sigma}");
            tally.assert_in_band(0.03, &format!("late hot key, sigma {sigma}"));
        }
    }

    /// Once heavy, a steadily hot key stays heavy on every turn (no flapping), even though
    /// its share sits close enough to the entry threshold that noise crosses it.
    #[test]
    fn heavy_status_does_not_flap() {
        let tracker = ShareTracker::default();
        let hot = key_preferring(TP4, |_| true);
        let mut rng = Rng(0xf1a9);
        let mut statuses = Vec::new();
        for i in 0..30_000u64 {
            // A 1-in-6 hot key: its load share (~0.17) is above the band plus noise, but
            // noise brings it near the entry threshold and it never nears the exit.
            let (key, charge) = if i % 6 == 0 {
                (hot, rng.charge(1.0))
            } else {
                (mixed(i + 3_000_000), rng.charge(1.0))
            };
            let (_, reason) = tracker
                .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                .unwrap();
            if key == hot {
                statuses.push(reason == SetChoiceReason::HeavyKeyRandom);
            }
        }
        let transitions = statuses.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(statuses.last().copied().unwrap_or(false));
        assert_eq!(transitions, 1, "heavy status changed {transitions} times");
    }

    /// The band for production weights (30/60 workers, slack 0.25) is 0.25 · 1/3 = 1/12.
    #[test]
    fn heavy_threshold_is_the_narrowest_band() {
        let ranking = [0, 1];
        let band = narrowest_band(&SETS, &ranking, DEFAULT_SLACK);
        assert!((band - 1.0 / 12.0).abs() < 1e-12, "band {band}");
        let three = [("a", 10.0), ("b", 45.0), ("c", 45.0)];
        let band = narrowest_band(&three, &[0, 1, 2], DEFAULT_SLACK);
        assert!((band - 0.025).abs() < 1e-12, "band {band}");
    }

    /// Steady pool of `concurrent` conversations of similar size (per-request lognormal
    /// `sigma` noise around a common median, context growing over 10 turns, then replaced
    /// by a new conversation), spread over `frontends`. Returns the fraction of follow-up
    /// turns that kept their set, the number of heavy classifications, and the placement
    /// tally of the second half.
    fn concurrent_pool(
        concurrent: usize,
        frontends: usize,
        requests: u64,
        sigma: f64,
        seed: u64,
    ) -> (f64, usize, Tally) {
        let mut rng = Rng(seed);
        let trackers: Vec<ShareTracker> = (0..frontends).map(|_| ShareTracker::default()).collect();
        let mut next_key = 0u64;
        let mut new_conversation = |rng: &mut Rng| {
            next_key += 1;
            // (key, turn, last set); start at a random turn so turnover is spread out.
            (mixed(next_key ^ seed), rng.next_u64() % 10, usize::MAX)
        };
        let mut pool: Vec<(u64, u64, usize)> = (0..concurrent)
            .map(|_| new_conversation(&mut rng))
            .collect();
        let (mut kept, mut follow_ups, mut heavy) = (0usize, 0usize, 0usize);
        let mut tally = Tally::default();
        for i in 0..requests {
            let slot = (rng.next_u64() % concurrent as u64) as usize;
            let (key, turn, last) = pool[slot];
            let charge = rng.charge(sigma) * (turn + 1) as f64;
            let frontend = (rng.next_u64() % frontends as u64) as usize;
            let (idx, reason) = trackers[frontend]
                .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                .unwrap();
            heavy += usize::from(reason == SetChoiceReason::HeavyKeyRandom);
            if last != usize::MAX {
                follow_ups += 1;
                kept += usize::from(idx == last);
            }
            if i >= requests / 2 {
                tally.add(idx, charge);
            }
            pool[slot] = if turn + 1 >= 10 {
                new_conversation(&mut rng)
            } else {
                (key, turn + 1, idx)
            };
        }
        (kept as f64 / follow_ups as f64, heavy, tally)
    }

    /// Low concurrency: a frontend that sees only 20-100 concurrent conversations keeps
    /// affinity on (no conversation is heavy), with one frontend or thirty.
    #[test]
    fn low_concurrency_keeps_affinity() {
        for concurrent in [20, 50, 100] {
            for (frontends, requests) in [(1usize, 20_000u64), (30, 60_000)] {
                let (kept, heavy, tally) = concurrent_pool(
                    concurrent,
                    frontends,
                    requests,
                    1.0,
                    0x10c0 + concurrent as u64,
                );
                let context = format!("{concurrent} conversations, {frontends} frontends");
                assert_eq!(heavy, 0, "{context}: heavy classifications");
                assert!(
                    kept > 0.98,
                    "{context}: {:.2}% kept their set",
                    kept * 100.0
                );
                tally.assert_in_band(0.03, &context);
            }
        }
    }

    /// Pins the boundary: `n` persistent equal conversations each carry 1/n of the load.
    /// With 12 or more (1/12 against a 1/12 band) none is heavy; with 8 (1/8) all are.
    #[test]
    fn equal_conversations_are_heavy_only_above_the_band() {
        let heavy_keys = |n: u64, sigma: f64| {
            let tracker = ShareTracker::default();
            let mut rng = Rng(0xb0 + n);
            let keys: Vec<u64> = (0..n).map(|k| mixed(k + 77)).collect();
            let mut ever_heavy = std::collections::HashSet::new();
            for _ in 0..30_000 {
                let key = keys[(rng.next_u64() % n) as usize];
                let charge = if sigma > 0.0 {
                    rng.charge(sigma)
                } else {
                    2000.0
                };
                let (_, reason) = tracker
                    .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                    .unwrap();
                if reason == SetChoiceReason::HeavyKeyRandom {
                    ever_heavy.insert(key);
                }
            }
            ever_heavy.len() as u64
        };
        for sigma in [0.0, 1.0] {
            for n in [12, 13, 16, 24] {
                assert_eq!(heavy_keys(n, sigma), 0, "{n} conversations, sigma {sigma}");
            }
        }
        assert_eq!(heavy_keys(8, 0.0), 8);
    }

    /// Fraction of follow-up turns that change set when `frontends` independent trackers
    /// serve `conversations` conversations of 10 turns, with per-conversation lognormal(σ)
    /// size, context growing by turn, and TP2-preferring conversations `skew` times larger.
    /// Panics if any conversation is classified heavy.
    fn follow_up_set_changes(
        sigma: f64,
        frontends: usize,
        conversations: u64,
        skew: f64,
        seed: u64,
    ) -> f64 {
        let mut rng = Rng(seed);
        let trackers: Vec<ShareTracker> = (0..frontends).map(|_| ShareTracker::default()).collect();
        let conversations: Vec<(u64, f64)> = (0..conversations)
            .map(|c| {
                let key = mixed(c + seed);
                let skew = if rendezvous_pick(key, &SETS) == Some(1) {
                    skew
                } else {
                    1.0
                };
                (key, rng.charge(sigma) * skew)
            })
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
                let (idx, reason) = trackers[frontend]
                    .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                    .unwrap();
                assert_ne!(
                    reason,
                    SetChoiceReason::HeavyKeyRandom,
                    "ordinary conversation classified heavy (turn {turn})"
                );
                if turn > 0 {
                    follow_ups += 1;
                    changes += usize::from(idx != last[c]);
                }
                last[c] = idx;
            }
        }
        changes as f64 / follow_ups as f64
    }

    /// R3-2 / R4-1: heavy-tailed request sizes neither make independent frontends bounce
    /// balanced conversations between sets nor classify long-context conversations heavy.
    #[test]
    fn lognormal_charges_keep_conversations_on_their_set() {
        for (sigma, frontends) in [(1.5, 10), (2.0, 10), (1.5, 30), (2.0, 30)] {
            let rate = follow_up_set_changes(sigma, frontends, 3000, 1.0, 0x5eed);
            assert!(
                rate < 0.005,
                "sigma {sigma}: {:.3}% of follow-ups changed set",
                rate * 100.0
            );
        }
    }

    /// Long-context conversations under a real 2x skew (sticky spill active) are never
    /// classified heavy either.
    #[test]
    fn long_context_conversations_are_never_heavy() {
        for (sigma, frontends) in [(1.5, 30), (2.0, 10)] {
            follow_up_set_changes(sigma, frontends, 3000, 2.0, 0xc0de);
        }
    }

    /// The noise allowance does not disable the guard: a real skew under heavy-tailed sizes
    /// (TP2-preferring conversations twice as large, demand ≈ 0.8) is still corrected.
    #[test]
    fn lognormal_skew_is_still_corrected() {
        for sigma in [1.5, 2.0] {
            let mut rng = Rng(0xabc);
            let tracker = ShareTracker::default();
            let mut tally = Tally::default();
            for i in 0..30_000u64 {
                let key = mixed(i);
                let skew = if rendezvous_pick(key, &SETS) == Some(1) {
                    2.0
                } else {
                    1.0
                };
                let charge = rng.charge(sigma) * skew;
                let (idx, _) = tracker
                    .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                    .unwrap();
                if i >= 15_000 {
                    tally.add(idx, charge);
                }
            }
            assert!(tracker.spill_fraction_of(1, &SETS, DEFAULT_SLACK) > 0.0);
            let cap = 2.0 / 3.0 + share_band(2.0 / 3.0, DEFAULT_SLACK) + 0.03;
            assert!(
                tally.share(1) <= cap,
                "sigma {sigma}: tp2 share {}",
                tally.share(1)
            );
        }
    }

    // -- R3-2 / R4-4: charge clamp and warm-up --

    /// One request near the maximum charge is clamped, is not a heavy key, and changes no
    /// other key's set.
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
        assert!(!tracker.is_heavy(huge, &SETS, DEFAULT_SLACK));
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
        for i in 0..5000u64 {
            tracker
                .choose(SetAffinity::new(mixed(i), 100.0), &SETS, DEFAULT_SLACK)
                .unwrap();
        }
        let state = tracker.state.lock();
        assert!((state.clamp_charge(1e9) - CHARGE_CLAMP_FACTOR * 100.0).abs() < 10.0);
        assert_eq!(state.clamp_charge(50.0), 50.0);
        // A fresh frontend clamps against the prior.
        let fresh = ShareState::default();
        assert_eq!(
            fresh.clamp_charge(1e9),
            CHARGE_CLAMP_FACTOR * PRIOR_MEAN_CHARGE
        );
    }

    /// R4-4: a very large first request must not disable the sticky spill on a fresh
    /// frontend: it is clamped against the prior, and the effective sample size is floored.
    #[test]
    fn large_first_request_does_not_disable_spill() {
        let skewed = |tracker: &ShareTracker, rng: &mut Rng, n: u64| {
            for i in 0..n {
                let key = mixed(i + 11);
                let skew = if rendezvous_pick(key, &SETS) == Some(1) {
                    2.0
                } else {
                    1.0
                };
                tracker
                    .choose(
                        SetAffinity::new(key, rng.charge(1.0) * skew),
                        &SETS,
                        DEFAULT_SLACK,
                    )
                    .unwrap();
            }
        };
        let reference = ShareTracker::default();
        skewed(&reference, &mut Rng(9), 1500);
        let p_reference = reference.spill_fraction_of(1, &SETS, DEFAULT_SLACK);
        assert!(p_reference > 0.0);
        for first in [1024.0 * 1024.0, MAX_REQUEST_CHARGE] {
            let tracker = ShareTracker::default();
            tracker
                .choose(SetAffinity::new(mixed(1), first), &SETS, DEFAULT_SLACK)
                .unwrap();
            skewed(&tracker, &mut Rng(9), 1500);
            let p = tracker.spill_fraction_of(1, &SETS, DEFAULT_SLACK);
            assert!(
                p > 0.0 && (p - p_reference).abs() < 0.05,
                "first charge {first}: p {p} vs {p_reference}"
            );
        }
    }

    /// The guard stays idle while a fresh frontend has few samples (even though all of
    /// its load prefers TP4), and no key is heavy before [`HEAVY_MIN_SAMPLES`].
    #[test]
    fn guard_waits_for_warm_up() {
        let tracker = ShareTracker::default();
        let tp4_keys = (0..u64::MAX)
            .map(mixed)
            .filter(|k| rendezvous_pick(*k, &SETS) == Some(0));
        for key in tp4_keys.take(SHARE_MIN_SAMPLES as usize) {
            let (idx, reason) = tracker.choose(unit(key), &SETS, DEFAULT_SLACK).unwrap();
            assert_eq!((idx, reason), (0, SetChoiceReason::Affinity));
        }
        let tracker = ShareTracker::default();
        let hot = key_preferring(TP4, |_| true);
        for _ in 0..(HEAVY_MIN_SAMPLES as usize) {
            let (idx, reason) = tracker.choose(unit(hot), &SETS, DEFAULT_SLACK).unwrap();
            assert_eq!((idx, reason), (0, SetChoiceReason::Affinity));
        }
    }

    // -- R4-3: the fair-share window restarts when the candidate sets change --

    /// TP2 goes 60 → 0 → 30 workers. While only TP4 is eligible the tracker sees no
    /// decisions (the selection only marks it); when TP2 returns at half its size, the
    /// window restarts and the new demand is judged against the new weights only, so no
    /// key spills.
    #[test]
    fn returning_set_at_a_new_size_is_not_judged_against_a_stale_window() {
        let full = [(TP4, 30.0), (TP2, 60.0)];
        let halved = [(TP4, 30.0), (TP2, 30.0)];
        let tracker = ShareTracker::default();
        for i in 0..5000u64 {
            tracker
                .choose(unit(mixed(i)), &full, DEFAULT_SLACK)
                .unwrap();
        }
        tracker.mark_single_eligible();
        tracker
            .choose(unit(mixed(49_999)), &halved, DEFAULT_SLACK)
            .unwrap();
        assert!(tracker.state.lock().window_decisions <= 1.0);
        for i in 0..2000u64 {
            let key = mixed(i + 50_000);
            let (idx, reason) = tracker.choose(unit(key), &halved, DEFAULT_SLACK).unwrap();
            assert_eq!(
                (Some(idx), reason),
                (rendezvous_pick(key, &halved), SetChoiceReason::Affinity)
            );
        }
        let state = tracker.state.lock();
        let fair_tp4 =
            ShareState::stat(&state.expected, TP4) / state.expected.values().sum::<f64>();
        assert!((fair_tp4 - 0.5).abs() < 1e-9, "fair {fair_tp4}");
    }

    /// A change in the candidate set keys also restarts the window.
    #[test]
    fn candidate_membership_change_restarts_the_window() {
        let tracker = ShareTracker::default();
        for i in 0..5000u64 {
            tracker
                .choose(unit(mixed(i)), &SETS, DEFAULT_SLACK)
                .unwrap();
        }
        assert!(tracker.state.lock().window_decisions > 900.0);
        let three = [(TP4, 30.0), (TP2, 60.0), ("ns-tp8", 10.0)];
        tracker.choose(unit(1), &three, DEFAULT_SLACK).unwrap();
        assert!(tracker.state.lock().window_decisions <= 1.0);
        // Same membership, a small weight drift (TP2's share 0.6 → 0.596): kept.
        tracker
            .choose(
                unit(2),
                &[(TP4, 30.0), (TP2, 59.0), ("ns-tp8", 10.0)],
                DEFAULT_SLACK,
            )
            .unwrap();
        assert!(tracker.state.lock().window_decisions > 1.0);
        // Drifts accumulate against the share at the last restart: a material change of
        // TP2's share (0.6 → 0.43) restarts the window too.
        tracker
            .choose(
                unit(3),
                &[(TP4, 30.0), (TP2, 30.0), ("ns-tp8", 10.0)],
                DEFAULT_SLACK,
            )
            .unwrap();
        assert!(tracker.state.lock().window_decisions <= 1.0);
    }

    // -- R2-1: key-deterministic sticky spill --

    /// A tracker whose decayed window is given directly: fair weight split 1:2, demand
    /// split as given (fractions of `total` load).
    fn tracker_with(demand_tp2: f64) -> ShareTracker {
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
                window_decisions: total,
                window_sum: total,
                // Effectively noiseless statistics, so `p` is exactly the demand formula.
                window_sq_sum: 1.0,
                expected: map(2.0 / 3.0),
                demand: map(demand_tp2),
                ..Default::default()
            }),
            stale: AtomicBool::new(false),
        }
    }

    #[test]
    fn spill_is_a_pure_function_of_key_at_fixed_fraction() {
        // Demand 0.8 on TP2 against fair 2/3: p = 1 − (2/3 + band/2) / 0.8 ≈ 0.115.
        let tracker = tracker_with(0.8);
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
        let a = tracker_with(0.80);
        let b = tracker_with(0.78);
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
            let mut tally = Tally::default();
            for i in 0..30_000u64 {
                let key = mixed(i ^ salt);
                // TP2-preferring conversations are twice as large: TP2 demand ≈ 0.8.
                let charge = if rendezvous_pick(key, &SETS) == Some(1) {
                    2.0
                } else {
                    1.0
                };
                let (idx, _) = tracker
                    .choose(SetAffinity::new(key, charge), &SETS, DEFAULT_SLACK)
                    .unwrap();
                if i >= 20_000 {
                    tally.add(idx, charge);
                }
            }
            tally
        };
        let (a, b) = (ShareTracker::default(), ShareTracker::default());
        let tallies = [feed(&a, 0x1111), feed(&b, 0x2222_0000)];
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
        for tally in &tallies {
            let tp2 = tally.share(1);
            assert!(
                tp2 <= 2.0 / 3.0 + share_band(2.0 / 3.0, DEFAULT_SLACK) + 0.02,
                "{tp2}"
            );
        }
    }

    /// Spills stick per key across turns: with a stable workload, a conversation keeps its
    /// set on every turn (keys right at the spill boundary excepted), and an ordinary
    /// conversation is never treated as heavy.
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
                assert_ne!(reason, SetChoiceReason::HeavyKeyRandom);
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
    }

    /// Sticky spill with three sets moves spilled keys to their second choice.
    #[test]
    fn three_set_spill_goes_to_the_second_choice() {
        let sets = [("a", 1.0), ("b", 1.0), ("c", 1.0)];
        let tracker = ShareTracker::default();
        let mut spills = 0;
        for i in 0..20_000u64 {
            let key = mixed(i);
            let ranking = rendezvous_ranking(key, &sets);
            let charge = if sets[ranking[0]].0 == "a" { 3.0 } else { 1.0 };
            let (idx, reason) = tracker
                .choose(SetAffinity::new(key, charge), &sets, DEFAULT_SLACK)
                .unwrap();
            match reason {
                SetChoiceReason::Affinity => assert_eq!(idx, ranking[0]),
                SetChoiceReason::ShareCapFallback => {
                    assert_eq!(idx, ranking[1]);
                    assert_eq!(sets[ranking[0]].0, "a");
                    spills += 1;
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert!(spills > 0);
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
