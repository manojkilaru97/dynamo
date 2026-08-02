// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::LazyLock;

use prometheus::{IntCounterVec, Opts, Registry};

pub const TRACE_CAPTURE_TYPE: &str = "trace";
pub const AUDIT_CAPTURE_TYPE: &str = "audit";

static RECORDS_WRITTEN: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_capture_records_written_total",
            "Capture records durably appended to gzip JSONL segments.",
        ),
        &["capture_type"],
    )
    .expect("capture records-written metric must be valid")
});

static UNCOMPRESSED_BYTES_WRITTEN: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_capture_uncompressed_bytes_written_total",
            "Serialized capture bytes durably appended before gzip compression.",
        ),
        &["capture_type"],
    )
    .expect("capture uncompressed-bytes metric must be valid")
});

static COMPRESSED_BYTES_WRITTEN: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_capture_compressed_bytes_written_total",
            "Compressed capture bytes durably appended to gzip JSONL segments.",
        ),
        &["capture_type"],
    )
    .expect("capture compressed-bytes metric must be valid")
});

static SEGMENTS_ROLLED: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_capture_segments_rolled_total",
            "Capture segments closed by the rolling gzip JSONL writer.",
        ),
        &["capture_type"],
    )
    .expect("capture segments-rolled metric must be valid")
});

static RECORDS_DROPPED: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_capture_records_dropped_total",
            "Capture records not durably written.",
        ),
        &["capture_type", "reason"],
    )
    .expect("capture records-dropped metric must be valid")
});

pub fn register(registry: &Registry) -> Result<(), prometheus::Error> {
    for capture_type in [TRACE_CAPTURE_TYPE, AUDIT_CAPTURE_TYPE] {
        RECORDS_WRITTEN.with_label_values(&[capture_type]);
        UNCOMPRESSED_BYTES_WRITTEN.with_label_values(&[capture_type]);
        COMPRESSED_BYTES_WRITTEN.with_label_values(&[capture_type]);
        SEGMENTS_ROLLED.with_label_values(&[capture_type]);
        for reason in [
            "bus_lag",
            "serialize",
            "sink_closed",
            "write",
            "writer_panic",
        ] {
            RECORDS_DROPPED.with_label_values(&[capture_type, reason]);
        }
    }
    registry.register(Box::new(RECORDS_WRITTEN.clone()))?;
    registry.register(Box::new(UNCOMPRESSED_BYTES_WRITTEN.clone()))?;
    registry.register(Box::new(COMPRESSED_BYTES_WRITTEN.clone()))?;
    registry.register(Box::new(SEGMENTS_ROLLED.clone()))?;
    registry.register(Box::new(RECORDS_DROPPED.clone()))?;
    Ok(())
}

pub fn record_written(
    capture_type: &'static str,
    records: u64,
    uncompressed_bytes: u64,
    compressed_bytes: u64,
) {
    RECORDS_WRITTEN
        .with_label_values(&[capture_type])
        .inc_by(records);
    UNCOMPRESSED_BYTES_WRITTEN
        .with_label_values(&[capture_type])
        .inc_by(uncompressed_bytes);
    COMPRESSED_BYTES_WRITTEN
        .with_label_values(&[capture_type])
        .inc_by(compressed_bytes);
}

pub fn record_segment_rolled(capture_type: &'static str) {
    SEGMENTS_ROLLED.with_label_values(&[capture_type]).inc();
}

pub fn record_dropped(capture_type: &'static str, reason: &'static str, records: u64) {
    RECORDS_DROPPED
        .with_label_values(&[capture_type, reason])
        .inc_by(records);
}

#[cfg(test)]
mod tests {
    use prometheus::{Encoder, TextEncoder};

    use super::*;

    #[test]
    fn registers_capture_metric_contract() {
        let registry = Registry::new();
        register(&registry).unwrap();
        record_written(TRACE_CAPTURE_TYPE, 2, 20, 10);
        record_segment_rolled(TRACE_CAPTURE_TYPE);
        record_dropped(AUDIT_CAPTURE_TYPE, "bus_lag", 3);

        let mut output = Vec::new();
        TextEncoder::new()
            .encode(&registry.gather(), &mut output)
            .unwrap();
        let text = String::from_utf8(output).unwrap();

        for name in [
            "dynamo_capture_records_written_total",
            "dynamo_capture_uncompressed_bytes_written_total",
            "dynamo_capture_compressed_bytes_written_total",
            "dynamo_capture_segments_rolled_total",
            "dynamo_capture_records_dropped_total",
        ] {
            assert!(text.contains(name), "missing {name}: {text}");
        }
    }
}
