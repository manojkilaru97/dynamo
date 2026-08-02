// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{config, handle::AuditRecord, suppression};
use crate::telemetry::bus::TelemetryBus;
use tokio::sync::broadcast;

static BUS: TelemetryBus<AuditRecord> = TelemetryBus::new();

pub fn init(capacity: usize) {
    BUS.init(capacity);
}

pub fn subscribe() -> broadcast::Receiver<AuditRecord> {
    BUS.subscribe()
}

pub fn publish(mut rec: AuditRecord) {
    suppression::suppress_record(&mut rec, &config::policy().suppressed_nca_ids);
    BUS.publish(rec);
}
