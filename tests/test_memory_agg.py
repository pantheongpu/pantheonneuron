"""Aggregate bandwidth: the arithmetic that makes it an aggregate.

The kernel each worker runs is already verified. What is new is combining
their figures, and the combination has one way to be quietly wrong -- summing
the time along with the bytes, which would report roughly one core's
bandwidth as the whole part's.
"""

import pantheon_neuron
from kernels import cores, memory_agg, registry


def _workload(name):
    return next(w for w in registry.WORKLOADS if w.name == name)


def _worker(core, gbps, byte_key, byte_count, span, ratio=1.0):
    return {"core": core, "analytic_gbps": gbps, byte_key: byte_count,
            "elapsed_s": span, "read_verified_ratio": ratio}


# -- the combination ---------------------------------------------------------

def test_bytes_sum_but_time_does_not():
    """Four cores moving 8 GB each in 10 s is 3.2 GB/s, not 0.8.

    Summing the spans as well would divide by four times too much and
    report one core's bandwidth under an aggregate label.
    """
    workers = [_worker(i, 0.8, "bytes_requested", 8_000_000_000, 10.0)
               for i in range(4)]
    summary = memory_agg.summarise(workers, [], 11.0, 4, "read")

    assert summary["bytes_moved"] == 32_000_000_000
    assert summary["worker_span_s"] == 10.0
    assert round(summary["analytic_gbps"], 2) == 3.2


def test_the_longest_span_is_used_not_the_shortest():
    """All that traffic moved within the longest worker's window."""
    workers = [
        _worker(0, 1.0, "bytes_requested", 10_000_000_000, 10.0),
        _worker(1, 1.0, "bytes_requested", 10_000_000_000, 20.0),
    ]
    summary = memory_agg.summarise(workers, [], 21.0, 2, "read")

    assert summary["worker_span_s"] == 20.0
    assert summary["analytic_gbps"] == 1.0


def test_write_direction_reads_the_write_counter():
    workers = [_worker(0, 2.0, "bytes_written", 4_000_000_000, 2.0)]
    summary = memory_agg.summarise(workers, [], 3.0, 1, "write")
    assert summary["bytes_moved"] == 4_000_000_000


# -- partial results are not aggregates --------------------------------------

def test_a_missing_core_is_flagged():
    """Three cores reporting out of four is not the part's bandwidth."""
    workers = [_worker(i, 1.0, "bytes_requested", 1_000_000_000, 1.0)
               for i in range(3)]
    summary = memory_agg.summarise(workers, [], 2.0, 4, "read")

    assert summary["cores_reporting"] == 3
    assert "not an aggregate over the whole part" in summary["warning"]


def test_worker_failures_are_reported():
    summary = memory_agg.summarise([], ["core 1 exited 1: boom"], 2.0, 2, "read")
    assert "core 1 exited 1" in summary["warning"]


def test_uneven_cores_are_flagged():
    """A core doing a fraction of its peers usually never got a device."""
    workers = [
        _worker(0, 200.0, "bytes_requested", 200_000_000_000, 1.0),
        _worker(1, 5.0, "bytes_requested", 5_000_000_000, 1.0),
    ]
    summary = memory_agg.summarise(workers, [], 2.0, 2, "read")
    assert "uneven" in summary["warning"]


def test_even_cores_pass_the_scaling_check():
    workers = [
        _worker(0, 200.0, "bytes_requested", 200_000_000_000, 1.0),
        _worker(1, 190.0, "bytes_requested", 190_000_000_000, 1.0),
    ]
    assert memory_agg.summarise(workers, [], 2.0, 2, "read")["warning"] is None


# -- the pinning that makes it meaningful ------------------------------------

def test_each_worker_is_pinned_to_one_core():
    """Threads would share one core allocation and measure it twice."""
    command = memory_agg.worker_command(
        "read", 3, {"bytes": 1 << 30, "dtype": "bf16"}, 30, "/tmp/out.json"
    )
    assert "--core" in command
    assert command[command.index("--core") + 1] == "3"
    assert "kernels.memory_agg" in command


def test_these_workloads_decline_the_reserved_core(monkeypatch):
    """cores: "all" means all, so the profiler does not get one."""
    from neuron_device import NeuronDevice

    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.delenv(cores.VISIBLE_CORES, raising=False)
    monkeypatch.delenv(cores.RESERVED_CORE, raising=False)

    devices = [NeuronDevice(0, "inf2", "v2", 2, 32 * 1024**3, False)]
    agg = [_workload("memory_read_agg")]

    assert pantheon_neuron.reserve_profiler_core(devices, agg) is None


def test_both_aggregates_are_dispatched():
    assert "memory_read_agg" in pantheon_neuron.IMPLEMENTED
    assert "memory_write_agg" in pantheon_neuron.IMPLEMENTED


def test_they_require_the_multicore_capability():
    """A single-core part cannot answer the question these ask."""
    for name in ("memory_read_agg", "memory_write_agg"):
        assert "multicore" in _workload(name).requires


# -- an aggregate has to be concurrent to be an aggregate --------------------
#
# The Score is summed bytes over the longest worker's span, and that
# arithmetic cannot tell two cores loading memory at once from two cores
# doing it one after the other. Only the first answers the question this
# workload asks -- whether the cores share a path to memory -- and nothing
# in the result showed which had happened.

def _timed_worker(finished_at, elapsed_s, bytes_moved=1 << 30):
    return {"finished_at": finished_at, "elapsed_s": elapsed_s,
            "bytes_requested": bytes_moved, "analytic_gbps": 100.0,
            "read_verified_ratio": 1.0, "core": 0}


def test_fully_overlapping_workers_report_the_whole_span():
    results = [_timed_worker(100.0, 20.0), _timed_worker(100.0, 20.0)]
    assert memory_agg.concurrent_window(results) == 20.0
    assert memory_agg.verify_workers_overlapped(results, 20.0) is None


def test_partly_overlapping_workers_report_the_intersection():
    """One worker compiled longer, so its loop started later."""
    results = [_timed_worker(100.0, 20.0),   # loop ran 80 -> 100
               _timed_worker(110.0, 20.0)]   # loop ran 90 -> 110
    assert memory_agg.concurrent_window(results) == 10.0


def test_workers_that_never_overlapped_are_refused():
    results = [_timed_worker(100.0, 20.0),   # 80 -> 100
               _timed_worker(140.0, 20.0)]   # 120 -> 140
    assert memory_agg.concurrent_window(results) == 0.0
    message = memory_agg.verify_workers_overlapped(results, 20.0)
    assert message is not None and "not an aggregate" in message


def test_a_thin_overlap_is_flagged():
    results = [_timed_worker(100.0, 20.0),   # 80 -> 100
               _timed_worker(115.0, 20.0)]   # 95 -> 115, overlap 5s of 20
    message = memory_agg.verify_workers_overlapped(results, 20.0)
    assert message is not None and "25%" in message


def test_a_single_worker_is_not_asked_to_overlap():
    assert memory_agg.verify_workers_overlapped([_timed_worker(100.0, 20.0)], 20.0) is None


def test_results_without_timestamps_report_unknown_not_zero():
    """Absence is not evidence of sequential execution.

    A result from before workers recorded timestamps says nothing about
    whether they overlapped, and treating that as zero overlap would flag
    every older aggregate as not an aggregate.
    """
    assert memory_agg.concurrent_window([{"elapsed_s": 20.0}]) is None
    assert memory_agg.verify_workers_overlapped(
        [{"elapsed_s": 20.0}, {"elapsed_s": 20.0}], 20.0) is None


def test_the_summary_carries_the_window():
    summary = memory_agg.summarise(
        [_timed_worker(100.0, 20.0), _timed_worker(100.0, 20.0)],
        failures=[], elapsed=25.0, core_count=2, direction="read")
    assert summary["concurrent_window_s"] == 20.0
    assert summary["analytic_gbps"] > 0


def test_the_worker_records_when_it_finished():
    import sourcecheck

    code = sourcecheck.function_code(memory_agg._worker_main)
    assert 'result [ "finished_at" ] = time . time ( )' in code
    # Wall clock, not monotonic: these are compared across processes.
    assert "time . monotonic" not in code
