---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Payload Logging and KV Transfer Tracing
---

## Overview

Dynamo supports two complementary observability features for LLM request/response data:

1. **Payload Logging** — exports full request and response JSON to the OTEL Logs pipeline, modeled after vLLM's `VLLM_LOG_PAYLOADS`. Useful for auditing, debugging, and prompt analytics.
2. **KV Transfer Tracing** — instruments the disaggregated prefill→decode KV cache transfer path with OTEL spans. Useful for diagnosing hangs and latency regressions in disaggregated serving.

Both features use the same OTLP gRPC connection (`OTEL_EXPORT_ENABLED=1`) but flow through separate pipelines:

```
tracing::info!(target: "dynamo_payload", ...)
    → OpenTelemetryTracingBridge (otel_logs_filter_layer)
    → SdkLoggerProvider + LogExporter
    → OTLP gRPC → collector → Loki

tracing::span!() / #[instrument] / opentelemetry spans
    → tracing_opentelemetry::layer() (otel_filter_layer)
    → SdkTracerProvider + SpanExporter
    → OTLP gRPC → collector → Tempo
```

No separate port is needed. The OTLP protocol multiplexes signal types over the same connection. Separate endpoints can be configured via `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` and `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.

---

## Payload Logging

### How It Works

When `DYNAMO_LOG_PAYLOADS=1`, structured log records containing the full request and response JSON are emitted at the Rust HTTP frontend layer (`lib/llm/src/http/service/openai.rs`). These records use the `dynamo_payload` tracing target, which is:

- **Suppressed from console output** via `dynamo_payload=off` in `fmt_filter_layer` (`lib/runtime/src/logging.rs`)
- **Exported to OTEL** via `OpenTelemetryTracingBridge` when `OTEL_EXPORT_ENABLED=1`

Each log record carries structured fields:

| Field | Description |
|-------|-------------|
| `request_id` | Dynamo request ID (correlates with traces) |
| `model` | Model name |
| `endpoint` | `chat_completions` or `completions` |
| `streaming` | `true` or `false` |
| `payload_type` | `request` or `response` |
| `payload` | Full JSON payload |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DYNAMO_LOG_PAYLOADS` | Enable payload logging (`1` to enable) | `0` |
| `OTEL_EXPORT_ENABLED` | Enable OTLP export for logs and traces | `false` |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` | OTLP gRPC endpoint for logs | same as traces endpoint |

### Current Limitations

- **Streaming responses are not logged.** Only non-streaming (`stream: false`) responses are captured. The streaming path returns HTTP 200 immediately and drives the SSE stream through the HTTP layer, so there is no natural "after stream ends" hook in the current architecture.
- **Completion endpoint only** (non-streaming). Chat completions streaming is also not captured.

### Next Step: Streaming Response Assembly

To cover streaming responses, accumulate delta content within the existing `flat_map` closure in `chat_completions()` and `completions_single()`. The pattern follows how `streaming_tool_dispatch_events()` and `accumulate_reasoning_dispatch()` already inspect `&response` before `EventConverter::from(response)` consumes it.

**Implementation plan:**

1. **Add a helper** `extract_streaming_delta(response: &Annotated<...>) -> StreamingDelta` that borrows the response and extracts:
   - `delta.content: Option<String>`
   - `delta.reasoning_content: Option<String>`
   - `delta.tool_calls: Option<Vec<...>>`
   - `finish_reason: Option<String>`

2. **Add accumulator state** inside the `flat_map` closure (captured by `move`):
   ```rust
   let mut accumulated_content = String::new();
   let mut accumulated_reasoning = String::new();
   let mut accumulated_tool_calls: Vec<...> = vec![];
   ```

3. **Accumulate per chunk** before `EventConverter::from(response)`:
   ```rust
   let delta = extract_streaming_delta(&response);
   if let Some(c) = delta.content { accumulated_content.push_str(&c); }
   let is_final = delta.finish_reason.is_some();
   ```

4. **Emit payload log on the final chunk:**
   ```rust
   if is_final && log_payloads_enabled() {
       tracing::info!(
           target: PAYLOAD_LOG_TARGET,
           request_id = %request_id_log,
           model = %model_log,
           streaming = true,
           payload_type = "response",
           content = %accumulated_content,
       );
   }
   ```

5. Apply the same pattern to `completions_single()` streaming path.

This mirrors vLLM's approach (`previous_content_texts`, `previous_reasoning_texts`, `previous_tool_calls` accumulated in `chat_completion_stream_generator`, emitted after the loop).

---

## KV Transfer Tracing

### Background

In disaggregated serving, the decode worker calls into TRT-LLM's Python executor which blocks inside `kv_cache_transceiver.check_gen_transfer_status` waiting for the prefill worker to push KV cache over NIXL/UCX. If this transfer stalls (observed in production with Qwen3 Coder 480B), the decode worker hangs indefinitely with no timeout and no observability.

The existing `handle_payload` span in Grafana Tempo covers the full decode worker duration but does not break out the KV transfer wait separately. This makes it impossible to distinguish "slow decode generation" from "stuck KV transfer" from trace data alone.

### Architecture

The OTEL trace pipeline is already wired in `lib/runtime/src/logging.rs`:

```
tracing::instrument / span!()     ← Rust layer (already instrumented at HTTP level)
    → tracing_opentelemetry::layer()
    → SdkTracerProvider + SpanExporter
    → OTLP gRPC → Tempo

opentelemetry Python SDK             ← Python layer (to be added)
    → OTLPSpanExporter
    → OTLP gRPC → Tempo (same endpoint)
```

The Rust HTTP frontend already emits spans (`http-request`, `prefill_routing`, `handle_payload`) that propagate trace context via W3C `traceparent` headers through NATS into the Python worker. The Python worker receives this context via `trace_headers` in the request dict.

### What's Missing

The Python decode handler (`DecodeHandler.generate()`) does not create a child span for the KV transfer wait phase. Adding one would make the `handle_payload` span in Tempo show a sub-span for `kv_transfer_wait` with exact timing — immediately revealing whether a slow decode was KV-transfer-bound or generation-bound.

### Implementation Plan

In `components/src/dynamo/trtllm/request_handlers/handlers.py`, `DecodeHandler.generate()`:

```python
from opentelemetry import trace, context as otel_context
from opentelemetry.propagate import extract

tracer = trace.get_tracer("dynamo.trtllm.decode")

async def generate(self, request, context):
    # Extract parent trace context propagated from Rust frontend
    carrier = request.get("trace_headers", {})
    parent_ctx = extract(carrier)

    first_token = True
    with tracer.start_as_current_span(
        "kv_transfer_wait",
        context=parent_ctx,
        attributes={
            "request_id": str(context.id()),
            "worker_id": str(self.worker_id),
        }
    ) as span:
        async for res in self.generate_locally(request, context):
            if first_token:
                # First token = KV transfer complete
                span.set_attribute("kv_transfer_complete", True)
                span.end()
                first_token = False
            yield res
```

This produces a child span under `handle_payload` in Tempo that shows exactly how long the decode worker waited for KV cache before generating the first token.

### Metrics to Add Alongside Traces

In `components/src/dynamo/trtllm/metrics.py`:

| Metric | Type | Description |
|--------|------|-------------|
| `trtllm_kv_transfer_timeout_total` | Counter | Decodes that exceeded KV transfer timeout |
| `trtllm_decode_kv_wait_seconds` | Histogram | Time from decode start to first token (KV wait proxy) |
| `trtllm_decode_inflight_kv_wait` | Gauge | Current number of decodes blocked waiting for KV |

### KV Transfer Timeout (Recovery)

Add a watchdog alongside the trace instrumentation in `DecodeHandler.generate()`:

```python
KV_TRANSFER_TIMEOUT_S = float(os.environ.get("DYNAMO_KV_TRANSFER_TIMEOUT_S", "60"))

async def _watchdog():
    await asyncio.sleep(KV_TRANSFER_TIMEOUT_S)
    logging.error(
        "[DECODE] KV transfer timeout after %.0fs for request %s",
        KV_TRANSFER_TIMEOUT_S, context.id()
    )
    metrics_collector.record_kv_transfer_timeout()
    context.kill()

watchdog = asyncio.create_task(_watchdog())
try:
    async for res in self.generate_locally(request, context):
        watchdog.cancel()
        yield res
finally:
    watchdog.cancel()
```

---

## Relationship to Existing Observability

| Feature | Pipeline | Wired | Status |
|---------|----------|-------|--------|
| Payload logging (non-streaming) | OTEL Logs | Yes | Done (eb0b6a9) |
| Payload logging (streaming) | OTEL Logs | Yes | Planned (Part 1) |
| HTTP request spans (`http-request`) | OTEL Traces | Yes | Done |
| Worker spans (`handle_payload`) | OTEL Traces | Yes | Done |
| KV transfer span (`kv_transfer_wait`) | OTEL Traces | Infrastructure yes | Planned (Part 2) |
| KV transfer timeout counter | Prometheus | No | Planned (Part 2) |
| Decode KV wait histogram | Prometheus | No | Planned (Part 2) |

---

## Related Documentation

- [Distributed Tracing with Tempo](tracing.md)
- [OTLP Log Export](logging.md#otlp-log-export)
- [TRT-LLM KV Cache Transfer](../backends/trtllm/trtllm-kv-cache-transfer.md)
- [Disaggregated Serving Design](../design-docs/disagg-serving.md)
