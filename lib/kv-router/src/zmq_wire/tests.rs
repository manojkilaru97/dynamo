// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use rmp_serde::{from_slice, to_vec, to_vec_named};
use serde::Serialize;

use crate::protocols::{
    BlockExtraInfo, BlockHashOptions, BlockMmObjectInfo, ExternalSequenceBlockHash,
    KvCacheEventData, StorageTier, WorkerWithDpRank, compute_block_hash_for_seq,
};

use super::filter::KvCacheSpecKind;
use super::*;

#[derive(Clone, Copy, Debug)]
enum TestEventKind {
    BlockStored,
    BlockRemoved,
}

#[test]
fn test_deserialize_bigram_block_stored_sequence() {
    let raw_event = (
        "BlockStored",
        vec![BlockHashValue::Unsigned(11), BlockHashValue::Unsigned(12)],
        Option::<BlockHashValue>::None,
        vec![(10u32, 11u32), (11, 12), (12, 13), (13, 14)],
        2usize,
        Option::<u64>::None,
        Option::<String>::None,
        Option::<String>::None,
    );
    let encoded = to_vec(&raw_event).unwrap();
    let event: RawKvEvent = from_slice(&encoded).unwrap();

    match event {
        RawKvEvent::BlockStored {
            token_ids,
            block_size,
            is_eagle,
            ..
        } => {
            assert_eq!(token_ids, vec![10, 11, 12, 13, 14]);
            assert_eq!(block_size, 2);
            assert_eq!(is_eagle, Some(true));
        }
        other => panic!("expected BlockStored, got {other:?}"),
    }
}

#[derive(Serialize)]
struct MapBlockStoredFixture {
    #[serde(rename = "type")]
    event_type: &'static str,
    block_hashes: Vec<BlockHashValue>,
    parent_block_hash: Option<BlockHashValue>,
    token_ids: Vec<u32>,
    block_size: usize,
    medium: Option<String>,
    lora_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_salt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extra_keys: Option<Vec<Option<Vec<String>>>>,
}

impl Default for MapBlockStoredFixture {
    fn default() -> Self {
        Self {
            event_type: "BlockStored",
            block_hashes: vec![BlockHashValue::Unsigned(11)],
            parent_block_hash: None,
            token_ids: vec![10, 11],
            block_size: 2,
            medium: None,
            lora_name: None,
            cache_salt: None,
            extra_keys: None,
        }
    }
}

#[test]
fn test_deserialize_map_block_stored_cache_salt() {
    let encoded = to_vec_named(&MapBlockStoredFixture {
        cache_salt: Some("tenant-a".to_string()),
        ..Default::default()
    })
    .unwrap();
    let event: RawKvEvent = from_slice(&encoded).unwrap();

    let RawKvEvent::BlockStored {
        cache_namespace, ..
    } = event
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(cache_namespace.as_deref(), Some("tenant-a"));
}

#[test]
fn test_deserialize_extra_keys_cache_namespace_fallback() {
    let mm_hash = "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210";
    let encoded = to_vec_named(&MapBlockStoredFixture {
        lora_name: Some("adapter-a".to_string()),
        extra_keys: Some(vec![Some(vec![
            "adapter-a".to_string(),
            mm_hash.to_string(),
            "dynamo-cache-salt:tenant-a".to_string(),
        ])]),
        ..Default::default()
    })
    .unwrap();
    let event: RawKvEvent = from_slice(&encoded).unwrap();

    let RawKvEvent::BlockStored {
        cache_namespace, ..
    } = event
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(cache_namespace.as_deref(), Some("tenant-a"));
}

#[test]
fn test_deserialize_hex_cache_namespace_is_not_multimodal() {
    let cache_namespace = "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210";
    let encoded = to_vec_named(&MapBlockStoredFixture {
        extra_keys: Some(vec![Some(vec![format!(
            "dynamo-cache-salt:{cache_namespace}"
        )])]),
        ..Default::default()
    })
    .unwrap();
    let event: RawKvEvent = from_slice(&encoded).unwrap();

    let RawKvEvent::BlockStored {
        cache_namespace: decoded_namespace,
        block_mm_infos,
        ..
    } = event
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(decoded_namespace.as_deref(), Some(cache_namespace));
    assert!(block_mm_infos.is_none());
}

fn block_stored_sequence(
    group_idx: Option<u32>,
    kv_cache_spec_kind: Option<&'static str>,
) -> Vec<u8> {
    match (group_idx, kv_cache_spec_kind) {
        (Some(group_idx), Some(kv_cache_spec_kind)) => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
            Option::<u8>::None,
            group_idx,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        (Some(group_idx), None) => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
            Option::<u8>::None,
            group_idx,
        ))
        .unwrap(),
        (None, Some(kv_cache_spec_kind)) => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
            Option::<u8>::None,
            Option::<u32>::None,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        (None, None) => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
        ))
        .unwrap(),
    }
}

fn block_removed_sequence(
    group_idx: Option<u32>,
    kv_cache_spec_kind: Option<&'static str>,
) -> Vec<u8> {
    match (group_idx, kv_cache_spec_kind) {
        (Some(group_idx), Some(kv_cache_spec_kind)) => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
            group_idx,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        (Some(group_idx), None) => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
            group_idx,
        ))
        .unwrap(),
        (None, Some(kv_cache_spec_kind)) => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
            Option::<u32>::None,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        (None, None) => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
        ))
        .unwrap(),
    }
}

fn sequence_with_group_idx(event_kind: TestEventKind, group_idx: Option<u32>) -> Vec<u8> {
    match event_kind {
        TestEventKind::BlockStored => block_stored_sequence(group_idx, None),
        TestEventKind::BlockRemoved => block_removed_sequence(group_idx, None),
    }
}

fn sequence_with_cache_spec_kind(
    event_kind: TestEventKind,
    group_idx: Option<u32>,
    kv_cache_spec_kind: &'static str,
) -> Vec<u8> {
    match event_kind {
        TestEventKind::BlockStored => block_stored_sequence(group_idx, Some(kv_cache_spec_kind)),
        TestEventKind::BlockRemoved => block_removed_sequence(group_idx, Some(kv_cache_spec_kind)),
    }
}

fn sequence_with_cache_spec_kind_without_group_idx_slot(
    event_kind: TestEventKind,
    kv_cache_spec_kind: &'static str,
) -> Vec<u8> {
    match event_kind {
        TestEventKind::BlockStored => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
            Option::<u8>::None,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        TestEventKind::BlockRemoved => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
            kv_cache_spec_kind,
        ))
        .unwrap(),
    }
}

fn assert_parsed_event_kind(event: RawKvEvent, expected_kind: TestEventKind) {
    match (event, expected_kind) {
        (RawKvEvent::BlockStored { .. }, TestEventKind::BlockStored)
        | (RawKvEvent::BlockRemoved { .. }, TestEventKind::BlockRemoved) => {}
        (event, expected_kind) => {
            panic!("expected {expected_kind:?}, got {event:?}");
        }
    }
}

fn assert_event_metadata(
    event: &RawKvEvent,
    expected_group_idx: Option<u32>,
    expected_kind: Option<KvCacheSpecKind>,
    expected_sliding_window: Option<u32>,
) {
    let metadata = event.metadata();
    assert_eq!(metadata.group_idx, expected_group_idx);
    assert_eq!(metadata.kv_cache_spec_kind, expected_kind);
    assert_eq!(
        metadata.kv_cache_spec_sliding_window,
        expected_sliding_window
    );
}

#[test]
fn test_deserialize_sequence_accepts_main_group_idx() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_group_idx(event_kind, Some(0))).unwrap();

        assert_event_metadata(&event, Some(0), None, None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_sequence_preserves_non_main_group_idx() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_group_idx(event_kind, Some(1))).unwrap();

        assert_event_metadata(&event, Some(1), None, None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_sequence_accepts_missing_group_idx() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_group_idx(event_kind, None)).unwrap();

        assert_event_metadata(&event, None, None, None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_sequence_accepts_main_attention_kind_with_nonzero_group_idx() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
            event_kind,
            Some(3),
            "full_attention",
        ))
        .unwrap();

        assert_event_metadata(&event, Some(3), Some(KvCacheSpecKind::FullAttention), None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_sequence_accepts_main_attention_kind_without_group_idx_slot() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_cache_spec_kind_without_group_idx_slot(
            event_kind,
            "full_attention",
        ))
        .unwrap();

        assert_event_metadata(&event, None, Some(KvCacheSpecKind::FullAttention), None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_block_stored_sequence_preserves_block_mm_infos_and_metadata() {
    let block_mm_infos = vec![Some(BlockExtraInfo {
        mm_objects: vec![BlockMmObjectInfo {
            mm_hash: 99,
            offsets: vec![(0, 1)],
        }],
    })];
    let raw_event = (
        "BlockStored",
        vec![BlockHashValue::Unsigned(11)],
        Option::<BlockHashValue>::None,
        vec![10u32, 11],
        2usize,
        Option::<u64>::None,
        Option::<String>::None,
        Option::<String>::None,
        Option::<u8>::None,
        block_mm_infos.clone(),
        3u32,
        "full_attention",
    );
    let encoded = to_vec(&raw_event).unwrap();
    let event: RawKvEvent = from_slice(&encoded).unwrap();

    match &event {
        RawKvEvent::BlockStored {
            block_mm_infos: Some(parsed),
            ..
        } => assert_eq!(parsed, &block_mm_infos),
        other => panic!("expected BlockStored with block_mm_infos, got {other:?}"),
    }
    assert_event_metadata(&event, Some(3), Some(KvCacheSpecKind::FullAttention), None);

    let remove: RawKvEvent =
        from_slice(&block_removed_sequence(Some(3), None)).expect("valid remove event");
    let mut normalizer = ZmqEventNormalizer::new(2);
    let worker = WorkerWithDpRank::new(7, 0);

    assert!(normalizer.preprocess(event, worker).is_some());
    assert!(normalizer.preprocess(remove, worker).is_some());
}

#[test]
fn test_deserialize_sequence_preserves_non_main_attention_kind_with_group_idx_zero() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent =
            from_slice(&sequence_with_cache_spec_kind(event_kind, Some(0), "mamba")).unwrap();

        assert_event_metadata(&event, Some(0), Some(KvCacheSpecKind::Mamba), None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_normalizer_ignores_non_main_group_idx_without_metadata() {
    let raw_event: RawKvEvent =
        from_slice(&block_removed_sequence(Some(1), None)).expect("valid raw event");
    let mut normalizer = ZmqEventNormalizer::new(2);

    assert_eq!(
        normalizer
            .preprocess_with_reason(raw_event, WorkerWithDpRank::new(3, 0))
            .unwrap_err(),
        ZmqEventFilterReason::UnlearnedGroupIdx
    );
}

#[test]
fn test_normalizer_ignores_map_serialized_non_main_attention_kind() {
    #[derive(serde::Serialize)]
    struct MapBlockStoredEvent {
        #[serde(rename = "type")]
        event_type: &'static str,
        block_hashes: Vec<u64>,
        parent_block_hash: Option<u64>,
        token_ids: Vec<u32>,
        block_size: usize,
        group_idx: Option<u32>,
        kv_cache_spec_kind: Option<&'static str>,
    }

    let event = MapBlockStoredEvent {
        event_type: "BlockStored",
        block_hashes: vec![11],
        parent_block_hash: None,
        token_ids: vec![10, 11],
        block_size: 2,
        group_idx: Some(1),
        kv_cache_spec_kind: Some("mamba"),
    };
    let encoded = rmp_serde::to_vec_named(&(0.0, vec![event], Some(0_i32)))
        .expect("serialize raw event batch");
    let mut batch = decode_event_batch(&encoded).expect("deserialize raw event batch");
    let decoded = batch.events.pop().expect("batch should contain event");
    let mut normalizer = ZmqEventNormalizer::new(2);

    assert_event_metadata(&decoded, Some(1), Some(KvCacheSpecKind::Mamba), None);
    assert_eq!(
        normalizer
            .preprocess_with_reason(decoded, WorkerWithDpRank::new(3, 0))
            .unwrap_err(),
        ZmqEventFilterReason::NonMainAttentionKind
    );
}

#[test]
fn test_normalizer_metadata_is_dp_rank_scoped() {
    let store: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
        TestEventKind::BlockStored,
        Some(3),
        "full_attention",
    ))
    .expect("valid store event");
    let same_rank_remove: RawKvEvent =
        from_slice(&block_removed_sequence(Some(3), None)).expect("valid same-rank remove event");
    let different_rank_remove: RawKvEvent = from_slice(&block_removed_sequence(Some(3), None))
        .expect("valid different-rank remove event");
    let mut normalizer = ZmqEventNormalizer::new(2);

    assert!(
        normalizer
            .preprocess(store, WorkerWithDpRank::new(7, 0))
            .is_some()
    );
    assert!(
        normalizer
            .preprocess(same_rank_remove, WorkerWithDpRank::new(7, 0))
            .is_some()
    );
    assert!(
        normalizer
            .preprocess(different_rank_remove, WorkerWithDpRank::new(7, 1))
            .is_none()
    );
}

#[test]
fn test_normalizer_does_not_learn_metadata_from_remove_events() {
    let metadata_remove: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
        TestEventKind::BlockRemoved,
        Some(3),
        "full_attention",
    ))
    .expect("valid metadata remove event");
    let bare_remove: RawKvEvent =
        from_slice(&block_removed_sequence(Some(3), None)).expect("valid bare remove event");
    let mut normalizer = ZmqEventNormalizer::new(2);
    let worker = WorkerWithDpRank::new(7, 0);

    assert!(normalizer.preprocess(metadata_remove, worker).is_some());
    assert!(normalizer.preprocess(bare_remove, worker).is_none());
}

#[test]
fn test_normalizer_propagates_cache_namespace_from_parent() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(2);
    let parent = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(1)],
        parent_block_hash: None,
        token_ids: vec![10, 11],
        block_size: 2,
        medium: None,
        lora_name: None,
        cache_namespace: Some("tenant-a".to_string()),
        block_mm_infos: None,
        is_eagle: Some(false),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    };
    let child = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(2)],
        parent_block_hash: Some(BlockHashValue::Unsigned(1)),
        token_ids: vec![12, 13],
        block_size: 2,
        medium: None,
        lora_name: None,
        cache_namespace: None,
        block_mm_infos: None,
        is_eagle: Some(false),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    };

    assert!(normalizer.preprocess(parent, worker).is_some());
    let child = normalizer.preprocess(child, worker).unwrap();

    let CacheNamespaceState::Namespaced(parent_namespace) =
        &normalizer.cache_namespaces[&(worker, 1)]
    else {
        panic!("expected namespaced parent");
    };
    let CacheNamespaceState::Namespaced(child_namespace) =
        &normalizer.cache_namespaces[&(worker, 2)]
    else {
        panic!("expected namespaced child");
    };
    assert!(Arc::ptr_eq(parent_namespace, child_namespace));

    let RawKvEvent::BlockStored {
        cache_namespace, ..
    } = child
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(cache_namespace.as_deref(), Some("tenant-a"));
}

#[test]
fn test_normalizer_shares_cache_namespace_across_blocks() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(2);
    let event = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(1), BlockHashValue::Unsigned(2)],
        parent_block_hash: None,
        token_ids: vec![10, 11, 12, 13],
        block_size: 2,
        medium: None,
        lora_name: None,
        cache_namespace: Some("tenant-a".to_string()),
        block_mm_infos: None,
        is_eagle: Some(false),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    };

    assert!(normalizer.preprocess(event, worker).is_some());

    let CacheNamespaceState::Namespaced(first) = &normalizer.cache_namespaces[&(worker, 1)] else {
        panic!("expected first block namespace");
    };
    let CacheNamespaceState::Namespaced(second) = &normalizer.cache_namespaces[&(worker, 2)] else {
        panic!("expected second block namespace");
    };
    assert!(Arc::ptr_eq(first, second));
}

#[test]
fn test_normalizer_rejects_ambiguous_parent_cache_namespace() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(2);
    let stored =
        |cache_namespace: Option<&str>, block_hashes, parent_block_hash| RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids: vec![10, 11],
            block_size: 2,
            medium: None,
            lora_name: None,
            cache_namespace: cache_namespace.map(str::to_owned),
            block_mm_infos: None,
            is_eagle: Some(false),
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
        };

    let parent_a = stored(Some("tenant-a"), vec![BlockHashValue::Unsigned(1)], None);
    let parent_b = stored(Some("tenant-b"), vec![BlockHashValue::Unsigned(1)], None);
    let child = stored(
        None,
        vec![BlockHashValue::Unsigned(2)],
        Some(BlockHashValue::Unsigned(1)),
    );

    assert!(normalizer.preprocess(parent_a, worker).is_some());
    assert!(normalizer.preprocess(parent_b, worker).is_some());
    assert_eq!(
        normalizer
            .preprocess_with_reason(child, worker)
            .expect_err("ambiguous parent must be rejected"),
        ZmqEventFilterReason::AmbiguousCacheNamespace
    );
}

#[test]
fn test_normalizer_treats_empty_namespace_as_absent() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(2);
    let parent = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(1)],
        parent_block_hash: None,
        token_ids: vec![10, 11],
        block_size: 2,
        medium: None,
        lora_name: None,
        cache_namespace: Some("tenant-a".to_string()),
        block_mm_infos: None,
        is_eagle: Some(false),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    };
    let child = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(2)],
        parent_block_hash: Some(BlockHashValue::Unsigned(1)),
        token_ids: vec![12, 13],
        block_size: 2,
        medium: None,
        lora_name: None,
        cache_namespace: Some(String::new()),
        block_mm_infos: None,
        is_eagle: Some(false),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    };

    assert!(normalizer.preprocess(parent, worker).is_some());
    let child = normalizer.preprocess(child, worker).unwrap();
    let RawKvEvent::BlockStored {
        cache_namespace, ..
    } = child
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(cache_namespace.as_deref(), Some("tenant-a"));
}

#[test]
fn test_normalizer_ignores_non_main_attention_kind_with_group_idx_zero() {
    let raw_event: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
        TestEventKind::BlockStored,
        Some(0),
        "mamba",
    ))
    .expect("valid raw event");
    let remove: RawKvEvent =
        from_slice(&block_removed_sequence(Some(0), None)).expect("valid remove event");
    let mut normalizer = ZmqEventNormalizer::new(2);
    let worker = WorkerWithDpRank::new(3, 0);

    assert!(normalizer.preprocess(raw_event, worker).is_none());
    assert!(normalizer.preprocess(remove, worker).is_none());
}

#[test]
fn test_convert_event_bigram_emits_eagle_windows() {
    let raw_event = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(21), BlockHashValue::Unsigned(22)],
        parent_block_hash: None,
        token_ids: vec![10, 11, 12, 13, 14],
        block_size: 2,
        medium: None,
        lora_name: None,
        cache_namespace: None,
        block_mm_infos: None,
        is_eagle: Some(true),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    };
    let warning_count = Arc::new(AtomicU32::new(0));
    let placement_event = convert_event(
        raw_event,
        7,
        2,
        WorkerWithDpRank::new(3, 0),
        &warning_count,
        None,
    );

    match placement_event.unwrap().event.data {
        KvCacheEventData::Stored(store_data) => {
            assert_eq!(store_data.blocks.len(), 2);
            assert_eq!(
                store_data.blocks[0].block_hash,
                ExternalSequenceBlockHash(21)
            );
            assert_eq!(
                store_data.blocks[1].block_hash,
                ExternalSequenceBlockHash(22)
            );

            let expected_first = compute_block_hash_for_seq(
                &[10, 11, 12],
                2,
                BlockHashOptions {
                    is_eagle: Some(true),
                    ..Default::default()
                },
            );
            let expected_second = compute_block_hash_for_seq(
                &[12, 13, 14],
                2,
                BlockHashOptions {
                    is_eagle: Some(true),
                    ..Default::default()
                },
            );

            assert_eq!(store_data.blocks[0].tokens_hash, expected_first[0]);
            assert_eq!(store_data.blocks[1].tokens_hash, expected_second[0]);
        }
        other => panic!("expected Stored event, got {other:?}"),
    }
}

struct CpuBlockStoredFixture<'a> {
    block_hashes: &'a [u64],
    token_ids: &'a [u32],
    block_size: usize,
    parent_block_hash: Option<u64>,
}

fn cpu_block_stored(fixture: CpuBlockStoredFixture<'_>) -> RawKvEvent {
    RawKvEvent::BlockStored {
        block_hashes: fixture
            .block_hashes
            .iter()
            .copied()
            .map(BlockHashValue::Unsigned)
            .collect(),
        parent_block_hash: fixture.parent_block_hash.map(BlockHashValue::Unsigned),
        token_ids: fixture.token_ids.to_vec(),
        block_size: fixture.block_size,
        medium: Some("CPU".to_string()),
        lora_name: None,
        cache_namespace: None,
        block_mm_infos: None,
        is_eagle: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    }
}

#[test]
fn cpu_event_with_placeholder_payload_is_dropped_safely() {
    let raw = cpu_block_stored(CpuBlockStoredFixture {
        block_hashes: &[201, 202, 203],
        token_ids: &[],
        block_size: 0,
        parent_block_hash: None,
    });
    let warning_count = Arc::new(AtomicU32::new(0));
    let placement = convert_event(
        raw,
        42,
        16,
        WorkerWithDpRank::new(7, 0),
        &warning_count,
        None,
    )
    .unwrap();

    assert_eq!(placement.placement.tier, StorageTier::HostPinned);
    match placement.event.data {
        KvCacheEventData::Stored(store_data) => {
            assert!(store_data.parent_hash.is_none());
            assert!(store_data.blocks.is_empty());
        }
        other => panic!("expected Stored event, got {other:?}"),
    }
    assert!(warning_count.load(Ordering::Relaxed) >= 1);
}

#[test]
fn cpu_event_with_full_payload_is_indexable() {
    let raw = cpu_block_stored(CpuBlockStoredFixture {
        block_hashes: &[201, 202],
        token_ids: &[10, 11, 12, 13, 14, 15, 16, 17],
        block_size: 4,
        parent_block_hash: Some(200),
    });
    let warning_count = Arc::new(AtomicU32::new(0));
    let placement = convert_event(
        raw,
        43,
        4,
        WorkerWithDpRank::new(7, 0),
        &warning_count,
        None,
    )
    .unwrap();

    assert_eq!(placement.placement.tier, StorageTier::HostPinned);
    match placement.event.data {
        KvCacheEventData::Stored(store_data) => {
            assert_eq!(store_data.parent_hash, Some(ExternalSequenceBlockHash(200)));
            assert_eq!(store_data.blocks.len(), 2);
            assert_eq!(
                store_data.blocks[0].block_hash,
                ExternalSequenceBlockHash(201)
            );
            assert_eq!(
                store_data.blocks[1].block_hash,
                ExternalSequenceBlockHash(202)
            );
        }
        other => panic!("expected Stored event, got {other:?}"),
    }
    assert_eq!(warning_count.load(Ordering::Relaxed), 0);
}

fn gpu_block_stored(
    block_hashes: &[u64],
    token_ids: &[u32],
    block_size: usize,
    parent: Option<u64>,
) -> RawKvEvent {
    let RawKvEvent::BlockStored {
        block_hashes,
        parent_block_hash,
        token_ids,
        block_size,
        lora_name,
        cache_namespace,
        block_mm_infos,
        is_eagle,
        group_idx,
        kv_cache_spec_kind,
        kv_cache_spec_sliding_window,
        ..
    } = cpu_block_stored(CpuBlockStoredFixture {
        block_hashes,
        token_ids,
        block_size,
        parent_block_hash: parent,
    })
    else {
        unreachable!()
    };
    RawKvEvent::BlockStored {
        block_hashes,
        parent_block_hash,
        token_ids,
        block_size,
        medium: None,
        lora_name,
        cache_namespace,
        block_mm_infos,
        is_eagle,
        group_idx,
        kv_cache_spec_kind,
        kv_cache_spec_sliding_window,
    }
}

fn stored_data(event: PlacementEvent) -> KvCacheStoreData {
    match event.event.data {
        KvCacheEventData::Stored(store) => store,
        other => panic!("expected Stored event, got {other:?}"),
    }
}

/// Lazy CPU offload emits stores with only the block hash; the normalizer
/// completes them from the device-tier store of the same block so the router
/// can index the CPU tier (and later match its removals).
#[test]
fn tokenless_cpu_store_is_completed_from_device_identity() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(4);
    let gpu = stored_data(
        normalizer
            .normalize(
                gpu_block_stored(&[201, 202], &[10, 11, 12, 13, 14, 15, 16, 17], 4, Some(200)),
                1,
                worker,
            )
            .unwrap(),
    );
    assert_eq!(gpu.blocks.len(), 2);

    let cpu_second = normalizer
        .normalize(
            cpu_block_stored(CpuBlockStoredFixture {
                block_hashes: &[202],
                token_ids: &[],
                block_size: 4,
                parent_block_hash: None,
            }),
            2,
            worker,
        )
        .unwrap();
    assert_eq!(cpu_second.placement.tier, StorageTier::HostPinned);
    let cpu_second = stored_data(cpu_second);
    assert_eq!(cpu_second.parent_hash, Some(ExternalSequenceBlockHash(201)));
    assert_eq!(cpu_second.blocks.len(), 1);
    assert_eq!(
        cpu_second.blocks[0].block_hash,
        ExternalSequenceBlockHash(202)
    );
    assert_eq!(cpu_second.blocks[0].tokens_hash, gpu.blocks[1].tokens_hash);

    let cpu_both = stored_data(
        normalizer
            .normalize(
                cpu_block_stored(CpuBlockStoredFixture {
                    block_hashes: &[201, 202],
                    token_ids: &[],
                    block_size: 4,
                    parent_block_hash: None,
                }),
                3,
                worker,
            )
            .unwrap(),
    );
    assert_eq!(cpu_both.parent_hash, Some(ExternalSequenceBlockHash(200)));
    assert_eq!(cpu_both.blocks.len(), 2);
    assert_eq!(cpu_both.blocks[0].tokens_hash, gpu.blocks[0].tokens_hash);
    assert_eq!(normalizer.take_lower_tier_filled(), 2);
    assert_eq!(normalizer.take_lower_tier_filled(), 0);
}

#[test]
fn tokenless_cpu_store_stays_empty_for_unknown_or_foreign_blocks() {
    let worker = WorkerWithDpRank::new(7, 0);
    let other = WorkerWithDpRank::new(8, 0);
    let mut normalizer = ZmqEventNormalizer::new(4);
    normalizer
        .normalize(
            gpu_block_stored(&[201], &[10, 11, 12, 13], 4, None),
            1,
            worker,
        )
        .unwrap();
    let tokenless = |hashes: &'static [u64]| {
        cpu_block_stored(CpuBlockStoredFixture {
            block_hashes: hashes,
            token_ids: &[],
            block_size: 4,
            parent_block_hash: None,
        })
    };
    let unknown = stored_data(normalizer.normalize(tokenless(&[999]), 2, worker).unwrap());
    assert!(unknown.blocks.is_empty());
    let foreign = stored_data(normalizer.normalize(tokenless(&[201]), 3, other).unwrap());
    assert!(foreign.blocks.is_empty());
    // A known head followed by an unknown block keeps only the known prefix.
    let partial = stored_data(
        normalizer
            .normalize(tokenless(&[201, 999]), 4, worker)
            .unwrap(),
    );
    assert_eq!(partial.blocks.len(), 1);
    assert_eq!(partial.parent_hash, None);
    assert_eq!(normalizer.take_lower_tier_filled(), 1);
}

/// D5: lazy CPU stores without token IDs are expected (the identity fill completes
/// them), so they must not consume the shared 3-warning budget that surfaces real drops.
#[test]
fn tokenless_cpu_store_does_not_consume_warning_budget() {
    let worker = WorkerWithDpRank::new(7, 0);
    let warning_count = Arc::new(AtomicU32::new(0));
    let mut normalizer = ZmqEventNormalizer::with_warning_count(4, warning_count.clone());
    let tokenless = |hashes: &'static [u64]| {
        cpu_block_stored(CpuBlockStoredFixture {
            block_hashes: hashes,
            token_ids: &[],
            block_size: 4,
            parent_block_hash: None,
        })
    };
    // Unknown identity (stays empty) and known identity (filled): neither warns.
    let unknown = stored_data(normalizer.normalize(tokenless(&[301]), 1, worker).unwrap());
    assert!(unknown.blocks.is_empty());
    normalizer
        .normalize(gpu_block_stored(&[401], &[1, 2, 3, 4], 4, None), 2, worker)
        .unwrap();
    let filled = stored_data(normalizer.normalize(tokenless(&[401]), 3, worker).unwrap());
    assert_eq!(filled.blocks.len(), 1);
    assert_eq!(warning_count.load(Ordering::Relaxed), 0);

    // Real truncation still warns: a device store, and a CPU store that carries some
    // (but too few) token IDs.
    let short_gpu = stored_data(
        normalizer
            .normalize(gpu_block_stored(&[501], &[1, 2], 4, None), 4, worker)
            .unwrap(),
    );
    assert!(short_gpu.blocks.is_empty());
    assert_eq!(warning_count.load(Ordering::Relaxed), 1);
    let short_cpu = normalizer
        .normalize(
            cpu_block_stored(CpuBlockStoredFixture {
                block_hashes: &[601],
                token_ids: &[1, 2],
                block_size: 4,
                parent_block_hash: None,
            }),
            5,
            worker,
        )
        .unwrap();
    assert!(stored_data(short_cpu).blocks.is_empty());
    assert_eq!(warning_count.load(Ordering::Relaxed), 2);
}

fn raw_removed(block_hashes: &[u64], medium: Option<&str>) -> RawKvEvent {
    RawKvEvent::BlockRemoved {
        block_hashes: block_hashes
            .iter()
            .copied()
            .map(BlockHashValue::Unsigned)
            .collect(),
        medium: medium.map(str::to_string),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    }
}

fn tokenless_cpu(block_hashes: &'static [u64]) -> RawKvEvent {
    cpu_block_stored(CpuBlockStoredFixture {
        block_hashes,
        token_ids: &[],
        block_size: 4,
        parent_block_hash: None,
    })
}

/// Number of blocks a tokenless CPU store of `hashes` is completed with.
fn cpu_fill(
    normalizer: &mut ZmqEventNormalizer,
    hashes: &'static [u64],
    worker: WorkerWithDpRank,
) -> usize {
    stored_data(
        normalizer
            .normalize(tokenless_cpu(hashes), 0, worker)
            .unwrap(),
    )
    .blocks
    .len()
}

fn gpu_store(normalizer: &mut ZmqEventNormalizer, hash: u64, worker: WorkerWithDpRank) {
    let tokens = [hash as u32, 1, 2, 3];
    normalizer
        .normalize(gpu_block_stored(&[hash], &tokens, 4, None), 0, worker)
        .unwrap();
}

fn gpu_remove(normalizer: &mut ZmqEventNormalizer, hash: u64, worker: WorkerWithDpRank) {
    normalizer
        .normalize(raw_removed(&[hash], None), 0, worker)
        .unwrap();
}

/// Lazy offload touches the GPU block during the async copy, so the CPU store is
/// published before the GPU removal; it must fill. A device removal then forgets the
/// identity.
#[test]
fn cpu_store_before_device_removal_fills_then_identity_is_forgotten() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(4);
    normalizer
        .normalize(
            gpu_block_stored(&[201, 202], &[1, 2, 3, 4, 5, 6, 7, 8], 4, None),
            1,
            worker,
        )
        .unwrap();
    assert_eq!(cpu_fill(&mut normalizer, &[201], worker), 1);
    gpu_remove(&mut normalizer, 201, worker);
    // 202 is still on the GPU; 201 is not.
    assert_eq!(cpu_fill(&mut normalizer, &[202], worker), 1);
    assert_eq!(cpu_fill(&mut normalizer, &[201], worker), 0);
    assert_eq!(normalizer.take_lower_tier_filled(), 2);
}

/// A device removal that precedes the CPU store leaves the store unfilled, which is the
/// behavior before the fill existed.
#[test]
fn device_removal_before_cpu_store_leaves_it_unfilled() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(4);
    gpu_store(&mut normalizer, 201, worker);
    gpu_remove(&mut normalizer, 201, worker);
    assert_eq!(cpu_fill(&mut normalizer, &[201], worker), 0);
}

/// Removing the CPU copy says nothing about the GPU copy, so the identity stays.
#[test]
fn cpu_removal_keeps_device_identity() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(4);
    gpu_store(&mut normalizer, 201, worker);
    assert_eq!(cpu_fill(&mut normalizer, &[201], worker), 1);
    normalizer
        .normalize(raw_removed(&[201], Some("CPU")), 0, worker)
        .unwrap();
    assert_eq!(cpu_fill(&mut normalizer, &[201], worker), 1);
}

#[test]
fn all_blocks_cleared_drops_only_that_workers_identities() {
    let (a, b) = (WorkerWithDpRank::new(7, 0), WorkerWithDpRank::new(8, 0));
    let mut normalizer = ZmqEventNormalizer::new(4);
    gpu_store(&mut normalizer, 201, a);
    gpu_store(&mut normalizer, 201, b);
    normalizer
        .normalize(RawKvEvent::AllBlocksCleared, 0, a)
        .unwrap();
    assert_eq!(cpu_fill(&mut normalizer, &[201], a), 0);
    assert_eq!(cpu_fill(&mut normalizer, &[201], b), 1);
}

/// Past capacity the least recently stored identity goes; a re-store refreshes it.
#[test]
fn re_store_refreshes_identity_position() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::with_identity_capacity(4, 3);
    for hash in [201, 202, 203] {
        gpu_store(&mut normalizer, hash, worker);
    }
    gpu_store(&mut normalizer, 201, worker);
    gpu_store(&mut normalizer, 204, worker);
    assert_eq!(cpu_fill(&mut normalizer, &[201], worker), 1);
    assert_eq!(cpu_fill(&mut normalizer, &[202], worker), 0);
    assert_eq!(cpu_fill(&mut normalizer, &[203], worker), 1);
    assert_eq!(cpu_fill(&mut normalizer, &[204], worker), 1);
}

/// A hot prefix stored early and offloaded much later still fills: device churn that is
/// evicted again does not push it out, and neither do further stores as long as the hot
/// block is re-stored before it becomes the oldest.
#[test]
fn hot_prefix_survives_churn_until_offloaded() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::with_identity_capacity(4, 8);
    gpu_store(&mut normalizer, 101, worker);
    for i in 0..10_000u64 {
        let hash = 1_000 + i;
        gpu_store(&mut normalizer, hash, worker);
        gpu_remove(&mut normalizer, hash, worker);
    }
    assert_eq!(cpu_fill(&mut normalizer, &[101], worker), 1);
    // Resident churn beyond capacity, with the hot block re-stored every few stores.
    for i in 0..1_000u64 {
        gpu_store(&mut normalizer, 50_000 + i, worker);
        if i % 4 == 0 {
            gpu_store(&mut normalizer, 101, worker);
        }
    }
    assert_eq!(cpu_fill(&mut normalizer, &[101], worker), 1);
    assert!(normalizer.block_identities.map.len() <= 8);
    assert!(normalizer.block_identities.order.len() <= 16);
}

/// R4-2: the publisher's dedup filter forwards a device removal only when the last of
/// duplicate stores is removed; until then the block is still resident, so the identity
/// must survive and a lazy CPU store must still fill.
#[test]
fn identity_is_reference_counted_like_dedup() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(4);
    gpu_store(&mut normalizer, 201, worker);
    gpu_store(&mut normalizer, 201, worker);
    gpu_remove(&mut normalizer, 201, worker);
    assert_eq!(cpu_fill(&mut normalizer, &[201], worker), 1);
    gpu_remove(&mut normalizer, 201, worker);
    assert_eq!(cpu_fill(&mut normalizer, &[201], worker), 0);
    // A removal of an unknown hash is ignored.
    gpu_remove(&mut normalizer, 999, worker);
}

#[test]
fn fill_lower_tier_env_values() {
    for on in [
        None,
        Some(""),
        Some("1"),
        Some("true"),
        Some("yes"),
        Some("on"),
    ] {
        assert!(fill_lower_tier_enabled(on), "{on:?}");
    }
    for off in ["0", "false", "FALSE", " no ", "off"] {
        assert!(!fill_lower_tier_enabled(Some(off)), "{off}");
    }
}

/// R4-5: with the fill disabled, token-less lower-tier stores stay empty (the behavior
/// before the fill existed) and no identities are kept.
#[test]
fn disabled_fill_leaves_lower_tier_stores_empty() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(4).with_lower_tier_fill(false);
    gpu_store(&mut normalizer, 201, worker);
    assert_eq!(cpu_fill(&mut normalizer, &[201], worker), 0);
    assert_eq!(normalizer.take_lower_tier_filled(), 0);
    assert!(normalizer.block_identities.map.is_empty());
    let mut enabled = ZmqEventNormalizer::new(4).with_lower_tier_fill(true);
    gpu_store(&mut enabled, 201, worker);
    assert_eq!(cpu_fill(&mut enabled, &[201], worker), 1);
}

/// R6-1: identities are recorded and reference counted from the worker's first device
/// event, not from its first lower-tier store. Unprimed sequence: device store H, a CPU
/// store of another hash, a duplicate device store H, one device removal H (the dedup
/// filter still holds a reference), then the CPU store of H must fill.
#[test]
fn unprimed_sequence_keeps_refcount_parity_with_dedup() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(4);
    gpu_store(&mut normalizer, 201, worker);
    assert_eq!(cpu_fill(&mut normalizer, &[999], worker), 0);
    gpu_store(&mut normalizer, 201, worker);
    gpu_remove(&mut normalizer, 201, worker);
    assert_eq!(cpu_fill(&mut normalizer, &[201], worker), 1);
}

/// R6-1: a hot prefix stored once on the device before the worker's first lower-tier
/// store (the first offload wave) fills when it is offloaded later.
#[test]
fn prefix_stored_before_first_lower_tier_store_fills() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(4);
    gpu_store(&mut normalizer, 101, worker);
    for hash in 1_000..1_200u64 {
        gpu_store(&mut normalizer, hash, worker);
    }
    // First offload wave: every resident block fills, including the early hot prefix.
    assert_eq!(cpu_fill(&mut normalizer, &[1_000], worker), 1);
    assert_eq!(cpu_fill(&mut normalizer, &[101], worker), 1);
}
