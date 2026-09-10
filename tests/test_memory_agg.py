"""Aggregate bandwidth: the arithmetic that makes it an aggregate.

The kernel each worker runs is already verified. What is new is combining
their figures, and the combination has one way to be quietly wrong -- summing
the time along with the bytes, which would report roughly one core's
bandwidth as the whole part's.
"""

import pantheon_neuron
import sourcecheck
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


# -- an aggregate whose workers never ran together ---------------------------

def _timed(started, finished):
    """Not _worker: this file already has one with a different signature.

    Second time in one session that appending to a test file shadowed a
    helper already in it. Grep before you append.
    """
    return {"started_at": started, "finished_at": finished,
            "bytes_moved": 1 << 30, "elapsed_s": finished - started}


def test_workers_that_never_overlapped_invalidate_the_score():
    """Not a weak aggregate: not an aggregate.

    The message already said "this is not an aggregate -- the cores may
    have run one after another", and the Score was published anyway. That
    is the shape that let llm_prefill, llm_decode and speculative_decode
    publish Scores for runs that produced NaN.
    """
    sequential = [_timed(0.0, 5.0), _timed(6.0, 11.0)]
    assert memory_agg._no_overlap_at_all(sequential, 11.0) is True


def test_a_short_overlap_is_a_warning_not_an_invalidation():
    """19% still measures something; it is just not what the name says."""
    overlapping = [_timed(0.0, 10.0), _timed(8.1, 18.0)]
    assert memory_agg._no_overlap_at_all(overlapping, 18.0) is False
    assert memory_agg.verify_workers_overlapped(overlapping, 18.0) is not None


def test_a_full_overlap_is_neither():
    together = [_timed(0.0, 10.0), _timed(0.1, 10.1)]
    assert memory_agg._no_overlap_at_all(together, 10.1) is False
    assert memory_agg.verify_workers_overlapped(together, 10.1) is None


def test_unknown_timing_is_not_zero_overlap():
    """A worker that did not report when it ran says nothing either way,
    and inventing a zero would fail a run for missing telemetry."""
    silent = [{"bytes_moved": 1 << 30}, {"bytes_moved": 1 << 30}]
    assert memory_agg._no_overlap_at_all(silent, 10.0) is False


def test_one_worker_cannot_fail_to_overlap_with_itself():
    assert memory_agg._no_overlap_at_all([_timed(0.0, 5.0)], 5.0) is False
    assert memory_agg._no_overlap_at_all([], 5.0) is False


# -- a worker that did not move its planned bytes ----------------------------

def test_a_short_worker_is_reported():
    """The aggregate sums bytes *requested*, not bytes confirmed.

    So a worker covering half its plan contributes its whole plan to the
    numerator: the aggregate is overstated by exactly the shortfall, the
    per-core rates stay even so verify_cores_scaled says nothing, and the
    evidence sits in per_core["verified_ratio"] where nothing read it.
    """
    per_core = [{"core": 0, "verified_ratio": 1.0},
                {"core": 1, "verified_ratio": 0.5}]
    message = memory_agg.verify_cores_read_what_they_planned(per_core)
    assert message is not None
    assert "core 1 covered 0.5" in message
    assert "bytes requested rather than bytes confirmed" in message


def test_full_coverage_says_nothing():
    per_core = [{"core": 0, "verified_ratio": 1.0},
                {"core": 1, "verified_ratio": 1.0}]
    assert memory_agg.verify_cores_read_what_they_planned(per_core) is None


def test_a_worker_that_reported_no_ratio_is_not_accused():
    """Absent is not short, and inventing a verdict from a missing field
    would fail a run for incomplete telemetry."""
    assert memory_agg.verify_cores_read_what_they_planned(
        [{"core": 0, "verified_ratio": None}]) is None
    assert memory_agg.verify_cores_read_what_they_planned([{"core": 0}]) is None


def test_coverage_is_checked_before_overlap_and_evenness():
    """A worker that did not move its bytes makes the aggregate wrong in
    a way the other two findings would only distract from."""
    code = sourcecheck.flat_function_code(memory_agg.summarise)
    assert code.index("verify_cores_read_what_they_planned") < code.index(
        "verify_workers_overlapped")
    assert code.index("verify_cores_read_what_they_planned") < code.index(
        "verify_cores_scaled")


def test_verify_cores_scaled_cannot_see_a_uniform_shortfall():
    """Which is why the coverage check is separate rather than folded in.

    Two cores both covering half their plan are perfectly even, so the
    rate comparison is silent and the aggregate is half of what it
    claims.
    """
    even_but_short = [{"core": 0, "gbps": 250.0, "verified_ratio": 0.5},
                      {"core": 1, "gbps": 250.0, "verified_ratio": 0.5}]
    assert memory_agg.verify_cores_scaled(even_but_short) is None
    assert memory_agg.verify_cores_read_what_they_planned(even_but_short)
