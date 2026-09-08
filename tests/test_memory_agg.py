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
