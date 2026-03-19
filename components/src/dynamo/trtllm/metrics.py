# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Additional Prometheus metrics for dynamo-trtllm beyond what the engine provides.

The TRT-LLM engine MetricsCollector already provides 5 core metrics:
  request_success_total, e2e_request_latency_seconds,
  time_to_first_token_seconds, inter_token_latency_seconds,
  request_queue_time_seconds

The Rust frontend (metrics.rs) provides token counters:
  input_tokens_total, output_tokens_total, cached_tokens

This module adds metrics that have no engine/runtime/frontend equivalent:
  - Request types (image, structured output)
  - KV transfer metrics (success counter, latency, throughput, per-request bytes)
  - Abort tracking
"""

import logging
import time
from datetime import timedelta

from prometheus_client import Counter, Gauge, Histogram

from dynamo.prometheus_names import trtllm_additional as metric_names

logger = logging.getLogger(__name__)

# Histogram buckets for KV cache transfer metrics
KV_TRANSFER_LATENCY_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    float("inf"),
)
KV_TRANSFER_SPEED_BUCKETS = (
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    float("inf"),
)
KV_TRANSFER_BYTES_BUCKETS = (
    100_000,
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    500_000_000,
    1_000_000_000,
    5_000_000_000,
    float("inf"),
)


class AdditionalMetricsCollector:
    """
    Additional Prometheus metrics for dynamo-trtllm.

    Only creates metrics that have no engine/runtime/frontend equivalent.
    Metrics are registered in the default prometheus_client.REGISTRY.

    Args:
        labels: Dict with keys like model_name, disaggregation_mode, engine_type.
    """

    def __init__(self, labels: dict):
        self._labelnames = list(labels.keys())
        self._labelvalues = list(labels.values())

        # --- Abort tracking ---
        self.num_aborted_requests = Counter(
            metric_names.NUM_ABORTED_REQUESTS_TOTAL,
            "Total number of aborted/cancelled requests",
            labelnames=self._labelnames,
        )

        # --- Request type counters ---
        self.request_type_image = Counter(
            metric_names.REQUEST_TYPE_IMAGE_TOTAL,
            "Total number of requests containing image content",
            labelnames=self._labelnames,
        )
        self.request_type_structured_output = Counter(
            metric_names.REQUEST_TYPE_STRUCTURED_OUTPUT_TOTAL,
            "Total number of requests using guided/structured decoding",
            labelnames=self._labelnames,
        )

        # --- KV cache transfer metrics ---
        self.kv_transfer_success = Counter(
            metric_names.KV_TRANSFER_SUCCESS_TOTAL,
            "Total number of successful KV cache transfers",
            labelnames=self._labelnames,
        )
        self.kv_transfer_latency = Histogram(
            metric_names.KV_TRANSFER_LATENCY_SECONDS,
            "KV cache transfer latency per request in seconds",
            labelnames=self._labelnames,
            buckets=KV_TRANSFER_LATENCY_BUCKETS,
        )
        self.kv_transfer_bytes = Histogram(
            metric_names.KV_TRANSFER_BYTES,
            "KV cache transfer size per request in bytes",
            labelnames=self._labelnames,
            buckets=KV_TRANSFER_BYTES_BUCKETS,
        )
        self.kv_transfer_speed = Histogram(
            metric_names.KV_TRANSFER_SPEED_GB_S,
            "KV cache transfer speed per request in GB/s",
            labelnames=self._labelnames,
            buckets=KV_TRANSFER_SPEED_BUCKETS,
        )

        # --- Iteration-level gauges (updated from get_stats_async polling loop) ---
        # These mirror the core queue/throughput metrics in SGLang and vLLM.
        # The `disaggregation_mode` label ("prefill", "decode", "prefill_and_decode")
        # disambiguates workers in P/D disaggregated deployments so PromQL queries
        # can filter correctly (e.g. sum where disaggregation_mode=~"decode|prefill_and_decode").
        self.num_requests_running = Gauge(
            metric_names.NUM_REQUESTS_RUNNING,
            "Number of requests actively executing in the engine this iteration",
            labelnames=self._labelnames,
        )
        self.num_requests_waiting = Gauge(
            metric_names.NUM_REQUESTS_WAITING,
            "Number of requests waiting in the scheduler queue",
            labelnames=self._labelnames,
        )
        self.num_context_requests = Gauge(
            metric_names.NUM_CONTEXT_REQUESTS,
            "Number of requests in context (prefill) phase this iteration",
            labelnames=self._labelnames,
        )
        self.gen_throughput = Gauge(
            metric_names.GEN_THROUGHPUT,
            "Generation throughput in tokens/s computed over the stats polling window",
            labelnames=self._labelnames,
        )
        self.prompt_tokens_total = Counter(
            metric_names.PROMPT_TOKENS_TOTAL,
            "Cumulative prompt tokens processed by this worker",
            labelnames=self._labelnames,
        )
        self.generation_tokens_total = Counter(
            metric_names.GENERATION_TOKENS_TOTAL,
            "Cumulative generation tokens produced by this worker",
            labelnames=self._labelnames,
        )

        # Internal state for throughput and counter delta computation
        self._last_throughput_ts: float = time.monotonic()
        self._last_gen_tokens: int = 0

        logger.info("AdditionalMetricsCollector initialized")

    # --- Request helpers ---

    def record_request_abort(self):
        """Increment aborted requests counter."""
        self.num_aborted_requests.labels(*self._labelvalues).inc()

    # --- Request type tracking ---

    def record_request_type_image(self):
        """Increment the image request type counter."""
        self.request_type_image.labels(*self._labelvalues).inc()

    def record_request_type_structured_output(self):
        """Increment the structured output request type counter."""
        self.request_type_structured_output.labels(*self._labelvalues).inc()

    # --- KV transfer ---

    def record_kv_transfer_success(self):
        """Increment the KV transfer success counter."""
        self.kv_transfer_success.labels(*self._labelvalues).inc()

    def update_from_iteration_stats(self, stat: dict) -> None:
        """Update iteration-level gauges and counters from a get_stats_async() result.

        Called once per stats poll iteration alongside MetricsCollector.log_iteration_stats().
        The stat dict mirrors the /metrics JSON schema from TRT-LLM's executor.

        In P/D disaggregated deployments each worker only sees its own phase:
        - prefill worker: numContextRequests is active, numGeneratedTokens is ~0
        - decode worker: numActiveRequests drives gen_throughput, numContextRequests is ~0
        Use the `disaggregation_mode` label to filter correctly in PromQL.

        Args:
            stat: Iteration stats dict from engine.llm.get_stats_async().
        """
        try:
            lv = self._labelvalues

            # Queue state
            num_active = stat.get("numActiveRequests", 0)
            num_queued = stat.get("numQueuedRequests", 0)
            num_context = stat.get("numContextRequests", 0)
            self.num_requests_running.labels(*lv).set(num_active)
            self.num_requests_waiting.labels(*lv).set(num_queued)
            self.num_context_requests.labels(*lv).set(num_context)

            # Token counters — increment by delta since last poll
            total_prompt = stat.get("numPromptTokens", 0)
            total_gen = stat.get("numGeneratedTokens", 0)

            # Counters can only go forward; guard against resets or missing fields
            if total_prompt > 0:
                self.prompt_tokens_total.labels(*lv).inc(max(0, total_prompt))
            gen_delta = max(0, total_gen - self._last_gen_tokens)
            if gen_delta > 0:
                self.generation_tokens_total.labels(*lv).inc(gen_delta)

            # Generation throughput: tokens/s over the wall-clock interval
            now = time.monotonic()
            elapsed = now - self._last_throughput_ts
            if elapsed > 0 and gen_delta > 0:
                self.gen_throughput.labels(*lv).set(gen_delta / elapsed)
            elif elapsed > 0 and num_active == 0:
                # No active requests — throughput is 0
                self.gen_throughput.labels(*lv).set(0.0)

            self._last_gen_tokens = total_gen
            self._last_throughput_ts = now

        except Exception as e:
            logger.warning("update_from_iteration_stats failed: %s", e)

    def record_kv_transfer_perf(self, timing_metrics) -> bool:
        """Record KV transfer performance from RequestPerfMetrics.timing_metrics.

        Extracts kv_cache_transfer_start, kv_cache_transfer_end, and kv_cache_size
        from TRT-LLM's TimingMetrics and records latency, bytes, and speed.
        Only records when a transfer actually occurred (non-zero transfer times).

        Args:
            timing_metrics: TimingMetrics object from RequestPerfMetrics.

        Returns:
            True if a transfer was recorded, False if skipped.
        """
        transfer_start = timing_metrics.kv_cache_transfer_start
        transfer_end = timing_metrics.kv_cache_transfer_end

        # Only record when a transfer actually happened
        if transfer_end <= timedelta(0) or transfer_start <= timedelta(0):
            return False

        latency_s = (transfer_end - transfer_start).total_seconds()
        if latency_s <= 0:
            return False

        kv_bytes = timing_metrics.kv_cache_size

        self.kv_transfer_latency.labels(*self._labelvalues).observe(latency_s)
        self.kv_transfer_bytes.labels(*self._labelvalues).observe(kv_bytes)

        if kv_bytes > 0:
            speed_gb_s = kv_bytes / (latency_s * 1e9)
            self.kv_transfer_speed.labels(*self._labelvalues).observe(speed_gb_s)

        return True
