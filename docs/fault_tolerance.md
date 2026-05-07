# Dynamo Fault Tolerance Notes

These notes document the production hardening added after the MiniMax reboot and
worker-backlog incidents. Keep this file updated when health, routing, or
admission-control behavior changes.

## Production Safety Goals

- Keep `/live` shallow. Liveness should mean the process and event loop are alive,
  not that generation canaries are passing.
- Keep `/health` conservative. Readiness may return `503` to drain traffic, but it
  must not cause Kubernetes liveness restarts.
- Bound worker-local backlog. Router queues may grow, but `Running + Waiting` at
  the backend worker should not creep without bound.
- Prefer quarantine/drain before restart. Restart only for process crash, startup
  deadlock, or true liveness failure.
- Preserve retry semantics. Backend overload must be surfaced as
  `service_overloaded` so the KV router can try another worker instead of failing
  the request immediately.

## Runtime Knobs

| Env var | Scope | Current intent |
| --- | --- | --- |
| `DYN_ROUTER_MAX_PENDING_PER_WORKER` | Frontend / KV router | Caps router-side per-worker pending dispatch. This is wired for SGLang and vLLM. |
| `DYN_REQUEST_MAX_TOTAL_REQUESTS` | Worker backend | Caps backend accepted requests per worker process. For SGLang this is a per-replica total, not multiplied by DP. |
| `DYN_REQUEST_MAX_TOTAL_REQUESTS_PER_DP` | Worker backend | Optional SGLang per-DP admission cap. Defaults to `DYN_REQUEST_MAX_TOTAL_REQUESTS / dp_size`. |
| `DYN_REQUEST_MAX_DECODE_WALL_CLOCK_SECS` | Worker backend | Cancels pathological long-running requests after the configured wall-clock limit. |
| `DYN_REQUEST_SLOT_LEASE_SECS` | Worker backend | Overrides the SGLang worker admission-slot fail-safe lease. Defaults to `DYN_REQUEST_MAX_DECODE_WALL_CLOCK_SECS`, then 600s. |
| `DYN_ROUTER_MAX_QUEUE_WAIT_MS` | Frontend / KV router | Maximum time a routed request can wait in the frontend/router queue before timing out. |
| `DYN_HEALTH_CHECK_*` | Runtime health | Controls canary generation cadence, timeout, stale window, and readiness behavior. |
| `DYN_REAL_TRAFFIC_HEALTH_*` | Runtime health | Lets recent real traffic keep readiness healthy even if canary checks are stale or noisy. |
| `DYN_ROUTER_KV_MISS_QUARANTINE_*` | KV router | Quarantines a worker when KV parent/remove miss rate spikes. |

## Backend Admission

`DYN_REQUEST_MAX_TOTAL_REQUESTS` is a worker-local admission limit. If the backend
is already at the limit, it should reject new work with Dynamo error type
`service_overloaded`. The router recognizes that type and can retry another
candidate worker. If all workers reject, the request stays bounded by router queue
timeout / client timeout instead of creating unbounded worker-local `Waiting`.

For SGLang, the default worker limit is derived from `--max-running-requests` as a
per-replica total when `DYN_REQUEST_MAX_TOTAL_REQUESTS` is unset. Set the env var
explicitly in prod when we need a stricter cap than SGLang's server default.

SGLang ActiveLoad now exposes backend scheduler truth through frontend gauges:

- `dynamo_frontend_worker_request_active_slots`
- `dynamo_frontend_worker_num_requests_waiting`
- `dynamo_frontend_worker_request_total_slots`

All three include `worker_id`, `dp_rank`, and `worker_type` labels. Use them to
reconcile frontend/router accounting against backend running + queued state.

## Canary Health

Canary health is readiness-only:

- `/live` should stay `200` while the runtime is alive.
- `/health` may return `503` when canary and real-traffic health both indicate the
  endpoint should drain.
- Healthy real traffic can keep `/health` ready even if a canary check times out.
- Canary lifecycle metrics should expose trigger, duration, result, and failures.

This avoids reboot hell: a slow or noisy generation canary should remove a worker
from routing, not force repeated model reloads.

## Fault-Tolerance Tests

The local regression tests live in `scripts/tests/fault_tolerance`.

### `test_inference_health_gate.py`

Validates `/live` vs `/health` semantics. Use `--mode healthy` when normal traffic
and readiness should pass. Use `--mode degraded` after intentionally making
generation unhealthy; `/live` must remain `200` while `/health` becomes `503`.

### `test_router_queue_pressure.py`

Generates concurrent requests and watches worker metrics:

- `vllm:num_requests_running` / `num_requests_running`
- `vllm:num_requests_waiting` / `num_requests_waiting`
- `dynamo_frontend_inflight_requests`

Use this to prove worker-local queue depth is bounded after enabling
`DYN_ROUTER_MAX_PENDING_PER_WORKER` and `DYN_REQUEST_MAX_TOTAL_REQUESTS`.

### `test_replica_loss_recovery.py`

Kills one `dynamo.vllm` worker during live traffic and verifies the frontend
continues serving. This test is currently vLLM-specific because it discovers and
kills `python3 -m dynamo.vllm` processes. Add an SGLang equivalent before claiming
replica-loss coverage for Dynamo + SGLang.

## Minimum Validation Before Prod Promotion

1. Run the health gate in healthy mode.
2. Run queue pressure with limits enabled and assert worker waiting stays bounded.
3. Verify overload responses are tagged `service_overloaded` and retried by the
   router.
4. For vLLM, run replica-loss recovery. For SGLang, run the equivalent once added.
5. Check Prometheus/OTel metrics for canary counts, `/live`, `/health`, request
   totals, queue depth, and overload/error labels.
