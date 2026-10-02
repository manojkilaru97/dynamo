// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire-format types for vLLM ZMQ KV event streams.
//!
//! These types mirror the Python `msgspec`-defined structures emitted by vLLM
//! engines over ZMQ PUB sockets. They are independent of the dynamo runtime
//! and can be used by any crate that needs to decode the raw ZMQ payloads.
//!
//! [`ZmqEventNormalizer`] also completes lower-tier (for example CPU) stores
//! that vLLM's lazy offload publishes without token IDs, from the identities
//! of the same blocks' device stores. This is an unconditional fix, independent
//! of `DYN_WORKER_SET_SELECTION`; it is on by default and
//! `DYN_KV_ROUTER_FILL_LOWER_TIER=0` turns it off.

use std::collections::VecDeque;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, LazyLock};

use rmp_serde as rmps;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::protocols::{
    BlockExtraInfo, DpRank, ExternalSequenceBlockHash, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, PlacementEvent, StorageTier, WorkerWithDpRank,
};

mod convert;
mod deserialize;
mod extra_keys;
mod filter;
#[cfg(test)]
mod tests;
mod types;

pub use convert::{
    StoredBlockOptions, convert_event, create_stored_block_from_parts, create_stored_blocks,
};
pub use extra_keys::{
    extra_keys_to_block_mm_infos, extra_keys_to_cache_namespace, parse_mm_hash_from_extra_key,
};
pub use filter::KvCacheSpecKind;
pub use types::{BlockHashValue, ExtraKeyItem, KvEventBatch, KvTokenIds, RawKvEvent};

use filter::KvCacheEventMetadata;

pub fn decode_event_batch(payload: &[u8]) -> Result<KvEventBatch, rmps::decode::Error> {
    rmps::from_slice(payload)
}

#[derive(Debug, Clone)]
pub struct ZmqEventNormalizer {
    kv_block_size: u32,
    /// Model's image placeholder token id, when MM-aware routing is active.
    /// Lets `convert_event` normalize vLLM BlockStored events to the canonical
    /// pad_value scheme. `None` for text-only models / non-MM deployments.
    image_token_id: Option<u32>,
    warning_count: Arc<AtomicU32>,
    group_metadata: FxHashMap<(DpRank, u32), KvCacheGroupMetadata>,
    cache_namespaces: FxHashMap<(WorkerWithDpRank, u64), CacheNamespaceState>,
    block_identities: BlockIdentityMemo,
    lower_tier_filled: u64,
    /// Complete token-less lower-tier stores from device identities
    /// (`DYN_KV_ROUTER_FILL_LOWER_TIER`, default on).
    fill_lower_tier: bool,
    /// Workers that have published a lower-tier store. Device identities are recorded
    /// only for these, so workers without CPU offload pay nothing; blocks a worker stored
    /// on the device before its first lower-tier store are not filled.
    lower_tier_workers: FxHashSet<WorkerWithDpRank>,
}

const FILL_LOWER_TIER_ENV: &str = "DYN_KV_ROUTER_FILL_LOWER_TIER";

/// Whether the lower-tier fill is enabled for an env value: on unless it is
/// `0`, `false`, `no` or `off` (case-insensitive).
fn fill_lower_tier_enabled(value: Option<&str>) -> bool {
    !value.map(str::trim).is_some_and(|v| {
        ["0", "false", "no", "off"]
            .iter()
            .any(|off| v.eq_ignore_ascii_case(off))
    })
}

static FILL_LOWER_TIER: LazyLock<bool> = LazyLock::new(|| {
    let enabled = fill_lower_tier_enabled(std::env::var(FILL_LOWER_TIER_ENV).ok().as_deref());
    if !enabled {
        tracing::info!("{FILL_LOWER_TIER_ENV} disables the lower-tier store fill");
    }
    enabled
});

/// Router identity of a device-tier block: its chain parent and token hash.
///
/// An engine block hash is a pure function of the parent hash and the block's
/// tokens (plus LoRA / salt / multimodal keys, which also feed the token hash),
/// so a remembered identity is never wrong, only possibly no longer needed.
#[derive(Debug, Clone)]
struct BlockIdentity {
    parent: Option<ExternalSequenceBlockHash>,
    tokens_hash: LocalBlockHash,
    mm_extra_info: Option<BlockExtraInfo>,
}

/// Bounded memo of the identities of device-resident blocks, used to complete
/// lower-tier stores that arrive without token IDs.
///
/// vLLM's lazy CPU offload copies blocks that are still cached on the GPU and
/// emits `BlockStored(medium=CPU)` with only the block hash. Without the
/// identity the router drops those stores, never indexes the CPU tier, and
/// later counts its removals as `block_not_found`.
///
/// The memo follows device residency exactly as the publisher's dedup filter
/// (`EventDedupFilter`) does: each device store of a hash takes a reference, a
/// device `BlockRemoved` releases one, and the identity is forgotten only when
/// the last reference goes (the point where the filter forwards the removal);
/// `AllBlocksCleared` forgets the worker. It therefore holds only blocks still
/// on the GPU, and long-lived hot prefixes are not pushed out by churn. This is
/// safe for lazy offload because the copy keeps the GPU block in use, so its
/// CPU store is published before the GPU removal. Past `capacity`, the least
/// recently stored identity is evicted (a re-store refreshes it). The
/// normalizer records identities only for workers that have published a
/// lower-tier store.
#[derive(Debug, Clone)]
struct BlockIdentityMemo {
    map: FxHashMap<(WorkerWithDpRank, u64), MemoEntry>,
    /// Store order; entries whose generation no longer matches `map` are stale.
    order: VecDeque<((WorkerWithDpRank, u64), u64)>,
    next_generation: u64,
    capacity: usize,
}

#[derive(Debug, Clone)]
struct MemoEntry {
    identity: BlockIdentity,
    /// Generation of the latest store (eviction order).
    generation: u64,
    /// Device stores not yet matched by a device removal.
    refs: usize,
}

const BLOCK_IDENTITY_MEMO_CAPACITY: usize = 65_536;

impl BlockIdentityMemo {
    fn new(capacity: usize) -> Self {
        Self {
            map: FxHashMap::default(),
            order: VecDeque::new(),
            next_generation: 0,
            capacity: capacity.max(1),
        }
    }

    fn record(&mut self, worker: WorkerWithDpRank, store: &KvCacheStoreData) {
        let mut parent = store.parent_hash;
        for block in &store.blocks {
            let key = (worker, block.block_hash.0);
            let identity = BlockIdentity {
                parent,
                tokens_hash: block.tokens_hash,
                mm_extra_info: block.mm_extra_info.clone(),
            };
            let generation = self.next_generation;
            self.next_generation += 1;
            let refs = self.map.get(&key).map_or(0, |e| e.refs) + 1;
            self.map.insert(
                key,
                MemoEntry {
                    identity,
                    generation,
                    refs,
                },
            );
            self.order.push_back((key, generation));
            parent = Some(block.block_hash);
        }
        while self.map.len() > self.capacity {
            let Some((key, generation)) = self.order.pop_front() else {
                break;
            };
            if self.is_current(&key, generation) {
                self.map.remove(&key);
            }
        }
        self.compact();
    }

    /// Release one device reference per hash; forget a hash at its last reference.
    fn remove(&mut self, worker: WorkerWithDpRank, hashes: &[ExternalSequenceBlockHash]) {
        for hash in hashes {
            let key = (worker, hash.0);
            if let Some(entry) = self.map.get_mut(&key) {
                entry.refs = entry.refs.saturating_sub(1);
                if entry.refs == 0 {
                    self.map.remove(&key);
                }
            }
        }
        self.compact();
    }

    /// Forget every block of a worker (`AllBlocksCleared`).
    fn clear_worker(&mut self, worker: WorkerWithDpRank) {
        self.map.retain(|(w, _), _| *w != worker);
        self.compact();
    }

    fn is_current(&self, key: &(WorkerWithDpRank, u64), generation: u64) -> bool {
        self.map
            .get(key)
            .is_some_and(|e| e.generation == generation)
    }

    /// Drop stale order entries once they outnumber live ones, keeping `order`
    /// within twice the capacity (amortized O(1) per operation).
    fn compact(&mut self) {
        if self.order.len() > 2 * self.capacity.max(self.map.len()) {
            let map = &self.map;
            self.order.retain(|(key, generation)| {
                map.get(key).is_some_and(|e| e.generation == *generation)
            });
        }
    }

    /// Fill an empty lower-tier store with the longest chain-consistent prefix
    /// of `hashes` whose identities are known. Returns the number filled.
    fn fill(
        &self,
        worker: WorkerWithDpRank,
        hashes: &[u64],
        store: &mut KvCacheStoreData,
    ) -> usize {
        let mut blocks = Vec::with_capacity(hashes.len());
        let mut first_parent = None;
        for (i, hash) in hashes.iter().enumerate() {
            let Some(MemoEntry { identity, .. }) = self.map.get(&(worker, *hash)) else {
                break;
            };
            if i == 0 {
                first_parent = identity.parent;
            } else if identity.parent != Some(ExternalSequenceBlockHash(hashes[i - 1])) {
                break;
            }
            blocks.push(KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(*hash),
                tokens_hash: identity.tokens_hash,
                mm_extra_info: identity.mm_extra_info.clone(),
            });
        }
        let filled = blocks.len();
        if filled > 0 {
            store.parent_hash = first_parent;
            store.blocks = blocks;
        }
        filled
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CacheNamespaceState {
    Namespaced(Arc<str>),
    Ambiguous,
}

#[derive(Debug, Clone, Copy)]
struct KvCacheGroupMetadata {
    kind: KvCacheSpecKind,
    sliding_window: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZmqEventFilterReason {
    IgnoredEvent,
    AmbiguousCacheNamespace,
    NonMainAttentionKind,
    UnknownKind,
    NonMainAttentionGroup,
    UnlearnedGroupIdx,
}

impl ZmqEventFilterReason {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::IgnoredEvent => "ignored_event",
            Self::AmbiguousCacheNamespace => "ambiguous_cache_namespace",
            Self::NonMainAttentionKind => "non_main_attention_kind",
            Self::UnknownKind => "unknown_kind",
            Self::NonMainAttentionGroup => "non_main_attention_group",
            Self::UnlearnedGroupIdx => "unlearned_group_idx",
        }
    }
}

impl ZmqEventNormalizer {
    pub fn new(kv_block_size: u32) -> Self {
        Self {
            kv_block_size,
            image_token_id: None,
            warning_count: Arc::new(AtomicU32::new(0)),
            group_metadata: FxHashMap::default(),
            cache_namespaces: FxHashMap::default(),
            block_identities: BlockIdentityMemo::new(BLOCK_IDENTITY_MEMO_CAPACITY),
            lower_tier_filled: 0,
            fill_lower_tier: *FILL_LOWER_TIER,
            lower_tier_workers: FxHashSet::default(),
        }
    }

    pub fn with_warning_count(kv_block_size: u32, warning_count: Arc<AtomicU32>) -> Self {
        Self {
            kv_block_size,
            image_token_id: None,
            warning_count,
            group_metadata: FxHashMap::default(),
            cache_namespaces: FxHashMap::default(),
            block_identities: BlockIdentityMemo::new(BLOCK_IDENTITY_MEMO_CAPACITY),
            lower_tier_filled: 0,
            fill_lower_tier: *FILL_LOWER_TIER,
            lower_tier_workers: FxHashSet::default(),
        }
    }

    /// Normalizer with a custom identity-memo capacity (tests).
    #[cfg(test)]
    fn with_identity_capacity(kv_block_size: u32, capacity: usize) -> Self {
        let mut normalizer = Self::new(kv_block_size);
        normalizer.block_identities = BlockIdentityMemo::new(capacity);
        normalizer
    }

    /// Enable or disable the lower-tier store fill (overrides
    /// `DYN_KV_ROUTER_FILL_LOWER_TIER`). Disabled, no identities are kept and
    /// token-less lower-tier stores stay empty, as before the fill existed.
    pub fn with_lower_tier_fill(mut self, enabled: bool) -> Self {
        self.fill_lower_tier = enabled;
        self
    }

    /// Set the model's image placeholder token id so vLLM BlockStored events
    /// get normalized to the canonical pad_value scheme. No-op for text-only
    /// models (leave unset).
    pub fn with_image_token_id(mut self, image_token_id: Option<u32>) -> Self {
        self.image_token_id = image_token_id;
        self
    }

    pub fn preprocess(&mut self, raw: RawKvEvent, worker: WorkerWithDpRank) -> Option<RawKvEvent> {
        self.preprocess_with_reason(raw, worker).ok()
    }

    pub fn preprocess_with_reason(
        &mut self,
        mut raw: RawKvEvent,
        worker: WorkerWithDpRank,
    ) -> Result<RawKvEvent, ZmqEventFilterReason> {
        if raw.is_ignored() {
            return Err(ZmqEventFilterReason::IgnoredEvent);
        }

        let metadata = raw.metadata();
        if matches!(raw, RawKvEvent::BlockStored { .. }) {
            self.learn_metadata(metadata, worker.dp_rank);
        }
        if let Some(reason) = self.filter_reason(metadata, worker.dp_rank) {
            return Err(reason);
        }
        self.propagate_cache_namespace(&mut raw, worker)?;
        Ok(raw)
    }

    pub fn normalize_preprocessed(
        &mut self,
        raw: RawKvEvent,
        event_id: u64,
        worker: WorkerWithDpRank,
    ) -> Option<PlacementEvent> {
        let lower_tier_hashes: Option<Vec<u64>> = match &raw {
            RawKvEvent::BlockStored {
                block_hashes,
                medium,
                ..
            } if self.fill_lower_tier
                && StorageTier::from_kv_medium_or_default(medium.as_deref())
                    != StorageTier::Device =>
            {
                Some(block_hashes.iter().map(|h| h.into_u64()).collect())
            }
            _ => None,
        };
        let mut event = convert_event(
            raw,
            event_id,
            self.kv_block_size,
            worker,
            &self.warning_count,
            self.image_token_id,
        )?;
        if !self.fill_lower_tier {
            return Some(event);
        }
        let is_device = event.placement.tier == StorageTier::Device;
        if lower_tier_hashes.is_some() {
            self.lower_tier_workers.insert(worker);
        } else if !self.lower_tier_workers.contains(&worker) {
            return Some(event);
        }
        match (&mut event.event.data, lower_tier_hashes) {
            (KvCacheEventData::Stored(store), Some(hashes)) => {
                if store.blocks.is_empty()
                    && !hashes.is_empty()
                    && self.block_identities.fill(worker, &hashes, store) > 0
                {
                    self.lower_tier_filled += 1;
                }
            }
            (KvCacheEventData::Stored(store), None) if is_device => {
                self.block_identities.record(worker, store);
            }
            (KvCacheEventData::Removed(removed), None) if is_device => {
                self.block_identities.remove(worker, &removed.block_hashes);
            }
            (KvCacheEventData::Cleared, None) if is_device => {
                self.block_identities.clear_worker(worker);
            }
            _ => {}
        }
        Some(event)
    }

    /// Number of token-less lower-tier stores completed from device identities
    /// since the last call.
    pub fn take_lower_tier_filled(&mut self) -> u64 {
        std::mem::take(&mut self.lower_tier_filled)
    }

    pub fn normalize(
        &mut self,
        raw: RawKvEvent,
        event_id: u64,
        worker: WorkerWithDpRank,
    ) -> Option<PlacementEvent> {
        let raw = self.preprocess(raw, worker)?;
        self.normalize_preprocessed(raw, event_id, worker)
    }

    fn learn_metadata(&mut self, metadata: KvCacheEventMetadata, dp_rank: DpRank) {
        let (Some(group_idx), Some(kind)) = (metadata.group_idx, metadata.kv_cache_spec_kind)
        else {
            return;
        };

        self.group_metadata.insert(
            (dp_rank, group_idx),
            KvCacheGroupMetadata {
                kind,
                sliding_window: metadata.kv_cache_spec_sliding_window,
            },
        );
    }

    fn propagate_cache_namespace(
        &mut self,
        raw: &mut RawKvEvent,
        worker: WorkerWithDpRank,
    ) -> Result<(), ZmqEventFilterReason> {
        match raw {
            RawKvEvent::BlockStored {
                block_hashes,
                parent_block_hash,
                cache_namespace,
                ..
            } => {
                if cache_namespace.as_deref() == Some("") {
                    *cache_namespace = None;
                }
                let namespace = if let Some(namespace) = cache_namespace.as_deref() {
                    parent_block_hash
                        .as_ref()
                        .and_then(|parent| {
                            self.cache_namespaces.get(&(worker, (*parent).into_u64()))
                        })
                        .and_then(|state| match state {
                            CacheNamespaceState::Namespaced(parent_namespace)
                                if parent_namespace.as_ref() == namespace =>
                            {
                                Some(Arc::clone(parent_namespace))
                            }
                            _ => None,
                        })
                        .or_else(|| Some(Arc::from(namespace)))
                } else if let Some(parent) = parent_block_hash.as_ref() {
                    match self.cache_namespaces.get(&(worker, (*parent).into_u64())) {
                        Some(CacheNamespaceState::Namespaced(namespace)) => {
                            *cache_namespace = Some(namespace.to_string());
                            Some(Arc::clone(namespace))
                        }
                        Some(CacheNamespaceState::Ambiguous) => {
                            return Err(ZmqEventFilterReason::AmbiguousCacheNamespace);
                        }
                        None => {
                            // Deliberately preserve the unsalted interpretation when a
                            // listener joins in the middle of a chain. The vLLM wire
                            // format cannot distinguish an unknown salted parent from a
                            // genuinely unsalted one, and retaining every unsalted block
                            // here would duplicate the index on the event hot path. The
                            // backend still enforces its own cache isolation; this narrow
                            // fail-open case can only pollute the router overlap score.
                            None
                        }
                    }
                } else {
                    None
                };

                if let Some(namespace) = namespace {
                    let state = CacheNamespaceState::Namespaced(namespace);
                    for block_hash in block_hashes.iter() {
                        self.cache_namespaces
                            .entry((worker, (*block_hash).into_u64()))
                            .and_modify(|existing| {
                                if *existing != state {
                                    *existing = CacheNamespaceState::Ambiguous;
                                }
                            })
                            .or_insert_with(|| state.clone());
                    }
                } else {
                    // Do not retain every unsalted block. If this hash already
                    // belongs to a namespace, however, fail closed on future
                    // propagation because the external hash is now ambiguous.
                    for block_hash in block_hashes.iter() {
                        if let Some(existing) = self
                            .cache_namespaces
                            .get_mut(&(worker, (*block_hash).into_u64()))
                        {
                            *existing = CacheNamespaceState::Ambiguous;
                        }
                    }
                }
            }
            RawKvEvent::BlockRemoved { block_hashes, .. } => {
                for block_hash in block_hashes.iter() {
                    let key = (worker, (*block_hash).into_u64());
                    if !matches!(
                        self.cache_namespaces.get(&key),
                        Some(CacheNamespaceState::Ambiguous)
                    ) {
                        self.cache_namespaces.remove(&key);
                    }
                }
            }
            RawKvEvent::AllBlocksCleared => {
                self.cache_namespaces
                    .retain(|(known_worker, _), _| *known_worker != worker);
            }
            RawKvEvent::Ignored => {}
        }
        Ok(())
    }

    fn filter_reason(
        &self,
        metadata: KvCacheEventMetadata,
        dp_rank: DpRank,
    ) -> Option<ZmqEventFilterReason> {
        if let Some(kind) = metadata.kv_cache_spec_kind {
            if kind.is_main_attention() {
                return None;
            }
            if kind == KvCacheSpecKind::Unknown {
                return Some(ZmqEventFilterReason::UnknownKind);
            }
            return Some(ZmqEventFilterReason::NonMainAttentionKind);
        }

        let group_idx = metadata.group_idx?;

        if let Some(metadata) = self.group_metadata.get(&(dp_rank, group_idx)) {
            let _sliding_window = metadata.sliding_window;
            if metadata.kind.is_main_attention() {
                return None;
            }
            if metadata.kind == KvCacheSpecKind::Unknown {
                return Some(ZmqEventFilterReason::UnknownKind);
            }
            return Some(ZmqEventFilterReason::NonMainAttentionGroup);
        }

        if group_idx == 0 {
            None
        } else {
            Some(ZmqEventFilterReason::UnlearnedGroupIdx)
        }
    }
}
