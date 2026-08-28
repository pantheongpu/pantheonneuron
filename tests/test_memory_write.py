"""The memory_write kernel: byte accounting and its two guards.

The NKI kernel needs hardware. What is testable here is the accounting and
the checks that decide whether a number is trustworthy -- in particular the
write/read ratio guard, which is what distinguishes a write benchmark from
an accidental read-modify-write.
"""

import pytest

import pantheon_neuron
from kernels import memory_write, registry, tiling
from neuron_device import NeuronDevice


TRN1 = [NeuronDevice(i, "trn1", "v2", 2, 32 * 1024**3, True) for i in range(2)]


def _workload():
    return next(w for w in registry.WORKLOADS if w.name == "memory_write")


# -- shared geometry ---------------------------------------------------------

def test_shares_tile_geometry_with_memory_read():
    """One partition constant, verified once on hardware, used by both."""
    from kernels import memory_read

    assert memory_write.PARTITION == memory_read.PARTITION == tiling.PARTITION
    assert memory_write.tile_plan is memory_read.tile_plan


def test_pinned_problem_divides_into_whole_tiles():
    problem = _workload().problem
    plan = memory_write.tile_plan(problem["bytes"], problem["dtype"])
    assert plan["actual_bytes"] == problem["bytes"]
    assert plan["tiles"] == 16384


def test_partition_matches_hardware_reported_pmax():
    """nl.tile_size.pmax read 128 on both parts probed."""
    assert tiling.PARTITION == 128


# -- write/read dominance guard ----------------------------------------------

def test_guard_accepts_a_write_dominated_run():
    """One tile read, 16384 tiles written -- the intended shape."""
    assert memory_write.verify_write_dominates_read(8 << 30, 524288) is None


def test_guard_flags_reads_approaching_writes():
    """If the kernel re-reads per store, the Score is a mixed workload."""
    message = memory_write.verify_write_dominates_read(1_000_000, 900_000)
    assert message is not None
    assert "not a pure write" in message


def test_guard_flags_zero_writes():
    message = memory_write.verify_write_dominates_read(0, 1000)
    assert "zero HBM writes" in message


def test_guard_is_silent_when_counters_are_absent():
    """A missing counter is not evidence of a problem; trn1 reports 90
    counters against inf2's 108, so absence is expected."""
    assert memory_write.verify_write_dominates_read(None, 1000) is None
    assert memory_write.verify_write_dominates_read(1000, None) is None


def test_guard_tolerates_a_zero_read():
    assert memory_write.verify_write_dominates_read(1 << 30, 0) is None


# -- analytic divergence guard -----------------------------------------------

def test_divergence_guard_flags_eliminated_stores():
    message = memory_write.verify_against_analytic(0.0, 500.0)
    assert message is not None
    assert "eliminated" in message


def test_divergence_guard_passes_on_agreement():
    assert memory_write.verify_against_analytic(480.0, 500.0) is None


def test_divergence_guard_is_silent_without_a_profiler_figure():
    assert memory_write.verify_against_analytic(None, 500.0) is None


def test_divergence_guard_rejects_zero_analytic():
    assert "zero" in memory_write.verify_against_analytic(100.0, 0.0)


# -- orchestrator integration ------------------------------------------------

def test_score_method_names_the_write_counter():
    """The fallback label must cite hbm_write_bytes, not the read counter."""
    pantheon_neuron._LAST_RUN["memory_write"] = {"score_method": "analytic"}
    method = pantheon_neuron._score_method(_workload(), 500.0)
    assert "hbm_write_bytes" in method
    assert "hbm_read_bytes" not in method
    pantheon_neuron._LAST_RUN.pop("memory_write", None)


def test_profiler_score_reports_the_declared_source():
    pantheon_neuron._LAST_RUN["memory_write"] = {"score_method": registry.PROFILER}
    assert pantheon_neuron._score_method(_workload(), 500.0) == registry.PROFILER
    pantheon_neuron._LAST_RUN.pop("memory_write", None)


def test_registry_declares_the_write_counter():
    source = _workload().score_source
    assert source.source == registry.PROFILER
    assert "hbm_write_bytes" in source.counters
    assert "hbm_read_bytes" not in source.counters


def test_mock_mode_invents_no_score(monkeypatch):
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    row = pantheon_neuron.run_workload(_workload(), TRN1, 1, 0.01)
    assert row["Score"] is None
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_hardware_path_requires_the_toolchain(monkeypatch):
    from kernels import nki_backend

    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    with pytest.raises(nki_backend.BackendUnavailable, match="neuronx-cc"):
        memory_write.run(_workload().problem, duration=1)
