use dynamo_llm::kv_router::protocols::ActiveLoad;
use dynamo_llm::kv_router::queue::WorkerRequestLoads;
use std::time::Duration;

fn active_load(
    worker_id: u64,
    dp_rank: u32,
    active: u64,
    waiting: u64,
    cap: Option<u64>,
) -> ActiveLoad {
    ActiveLoad {
        worker_id,
        dp_rank,
        active_decode_blocks: None,
        active_prefill_tokens: None,
        request_active_slots: Some(active),
        num_requests_waiting: Some(waiting),
        request_total_slots: cap,
    }
}

#[test]
fn sglang_request_counts_sum_running_and_waiting_across_dp_ranks() {
    let loads = WorkerRequestLoads::default();

    assert!(loads.update_from_active_load(&active_load(7, 0, 3, 2, Some(8))));
    assert!(loads.update_from_active_load(&active_load(7, 1, 4, 1, Some(8))));

    assert_eq!(loads.total_requests_and_cap(7), Some((10, Some(16))));
}

#[test]
fn sglang_request_counts_decrease_when_scheduler_reports_lower_load() {
    let loads = WorkerRequestLoads::default();

    loads.update_from_active_load(&active_load(7, 0, 16, 0, Some(16)));
    assert_eq!(loads.total_requests_and_cap(7), Some((16, Some(16))));

    loads.update_from_active_load(&active_load(7, 0, 2, 1, Some(16)));
    assert_eq!(loads.total_requests_and_cap(7), Some((3, Some(16))));
}

#[test]
fn sglang_request_cap_sums_published_per_dp_caps() {
    let loads = WorkerRequestLoads::default();

    loads.update_from_active_load(&active_load(7, 0, 8, 0, Some(8)));
    loads.update_from_active_load(&active_load(7, 1, 8, 0, Some(8)));
    assert_eq!(loads.total_requests_and_cap(7), Some((16, Some(16))));

    loads.update_from_active_load(&active_load(7, 1, 7, 0, Some(8)));
    assert_eq!(loads.total_requests_and_cap(7), Some((15, Some(16))));
}

#[test]
fn sglang_request_cap_boundary_is_exact() {
    let loads = WorkerRequestLoads::default();

    loads.update_from_active_load(&active_load(7, 0, 15, 0, Some(16)));
    let (total, cap) = loads.total_requests_and_cap(7).unwrap();
    assert_eq!(total, 15);
    assert_eq!(cap, Some(16));
    assert!(total < cap.unwrap());

    loads.update_from_active_load(&active_load(7, 0, 16, 0, Some(16)));
    let (total, cap) = loads.total_requests_and_cap(7).unwrap();
    assert_eq!(total, 16);
    assert_eq!(cap, Some(16));
    assert!(total >= cap.unwrap());
}

#[test]
fn legacy_kv_metrics_do_not_poison_request_capacity() {
    let loads = WorkerRequestLoads::default();
    let legacy_load = ActiveLoad {
        worker_id: 7,
        dp_rank: 0,
        active_decode_blocks: Some(8),
        active_prefill_tokens: None,
        request_active_slots: None,
        num_requests_waiting: None,
        request_total_slots: Some(16),
    };

    assert!(!loads.update_from_active_load(&legacy_load));
    assert_eq!(loads.total_requests_and_cap(7), None);
}

#[test]
fn stale_request_loads_do_not_poison_capacity() {
    let loads = WorkerRequestLoads::new(Some(Duration::ZERO));

    assert!(loads.update_from_active_load(&active_load(7, 0, 8, 0, Some(8))));

    assert_eq!(loads.total_requests_and_cap(7), None);
}
