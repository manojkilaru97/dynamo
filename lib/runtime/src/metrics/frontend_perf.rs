// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend pipeline latency metrics exposed by the HTTP service registry.

use once_cell::sync::{Lazy, OnceCell};
use prometheus::{Histogram, HistogramOpts};

use super::prometheus_names::{frontend_perf, name_prefix};

fn frontend_metric_name(suffix: &str) -> String {
    format!("{}_{}", name_prefix::FRONTEND, suffix)
}

pub static TEMPLATE_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            frontend_metric_name(frontend_perf::TEMPLATE_SECONDS),
            "Time spent applying the chat template in the frontend preprocessor (seconds)",
        )
        .buckets(vec![
            0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
        ]),
    )
    .expect("frontend_template_seconds histogram")
});

static PROMETHEUS_REGISTERED: OnceCell<Result<(), String>> = OnceCell::new();

pub fn ensure_frontend_perf_metrics_registered_prometheus(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    PROMETHEUS_REGISTERED
        .get_or_init(|| {
            registry
                .register(Box::new(TEMPLATE_SECONDS.clone()))
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map(|_| ())
        .map_err(|e| prometheus::Error::Msg(e.clone()))
}
