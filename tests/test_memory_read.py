"""The memory_read kernel: tile maths, byte accounting, and the honesty guards.

The NKI kernel itself cannot be tested here -- it needs a Neuron device.
What is testable without hardware is everything around it: whether the
byte accounting is right, whether a Score that did not come from its
declared source is labelled as such, and whether the guard that catches an
optimised-away kernel actually fires.
"""

import pytest

import pantheon_neuron
from kernels import memory_read, nki_backend, registry
from neuron_device import NeuronDevice


TRN1 = [NeuronDevice(i, "trn1", "v2", 2, 32 * 1024**3, True) for i in range(2)]


def _workload():
    return next(w for w in registry.WORKLOADS if w.name == "memory_read")


# -- tile plan ---------------------------------------------------------------

def test_pinned_problem_divides_into_whole_tiles():
    """8 GiB of bf16 must not leave a partial tile behind."""
    problem = _workload().problem
    plan = memory_read.tile_plan(problem["bytes"], problem["dtype"])
    assert plan["actual_bytes"] == problem["bytes"]
    assert plan["tiles"] == 16384


def test_actual_bytes_never_exceeds_requested():
    """The Score divides by bytes actually read; over-reporting inflates it."""
    for request in (1 << 20, 3_000_000, 8 << 30, (8 << 30) + 1):
        try:
            plan = memory_read.tile_plan(request, "bf16")
        except ValueError:
            continue
        assert plan["actual_bytes"] <= request


def test_actual_bytes_is_a_whole_number_of_tiles():
    plan = memory_read.tile_plan(5_000_000_000, "bf16")
    assert plan["actual_bytes"] == plan["tiles"] * plan["tile_bytes"]


def test_request_smaller_than_one_tile_is_rejected():
    with pytest.raises(ValueError, match="smaller than one"):
        memory_read.tile_plan(1024, "bf16")


def test_unsupported_dtype_is_rejected():
    with pytest.raises(ValueError, match="unsupported dtype"):
        memory_read.tile_plan(8 << 30, "fp8")


@pytest.mark.parametrize("dtype,width", [("bf16", 2), ("fp32", 4), ("int8", 1)])
def test_element_width_scales_the_tile(dtype, width):
    plan = memory_read.tile_plan(8 << 30, dtype)
    assert plan["tile_bytes"] == 128 * 2048 * width


def test_partition_dimension_is_the_hardware_constant():
    """128 is a NeuronCore-v2 partition limit, not a tunable."""
    assert memory_read.PARTITION == 128


# -- the optimised-away guard ------------------------------------------------

def test_guard_flags_a_kernel_whose_loads_were_eliminated():
    """Fast wall time, large analytic figure, no HBM traffic -- the exact
    signature of a compiler having deleted the DMA."""
    message = memory_read.verify_against_analytic(
        profiler_gbps=0.0, analytic_gbps=800.0
    )
    assert message is not None
    assert "eliminated" in message


def test_guard_flags_a_large_divergence():
    message = memory_read.verify_against_analytic(
        profiler_gbps=100.0, analytic_gbps=800.0
    )
    assert message is not None
    assert "differ" in message


def test_guard_passes_when_the_two_agree():
    assert memory_read.verify_against_analytic(780.0, 800.0) is None


def test_guard_rejects_a_zero_analytic():
    assert "zero" in memory_read.verify_against_analytic(100.0, 0.0)


# -- provisional score labelling ---------------------------------------------

def test_profiler_score_reports_the_declared_source(monkeypatch):
    """When the profiler produced the number, say so plainly."""
    monkeypatch.setitem(
        pantheon_neuron._LAST_RUN, "memory_read",
        {"score_method": registry.PROFILER},
    )
    assert pantheon_neuron._score_method(_workload(), 812.5) == registry.PROFILER


def test_analytic_fallback_is_labelled_as_provisional(monkeypatch):
    """When the profiler was unavailable and the kernel fell back, the row
    must not read as though the declared contract held."""
    monkeypatch.setitem(
        pantheon_neuron._LAST_RUN, "memory_read", {"score_method": "analytic"},
    )
    method = pantheon_neuron._score_method(_workload(), 812.5)
    assert "analytic" in method
    assert "neuron-profile" in method
    assert method != registry.PROFILER


def test_score_method_is_none_without_a_score():
    assert pantheon_neuron._score_method(_workload(), None) is None


def test_non_provisional_workload_reports_its_declared_source():
    workload = next(w for w in registry.WORKLOADS if w.name == "tensor_virus")
    assert pantheon_neuron._score_method(workload, 1.0) == registry.MONITOR


def test_row_shape_is_unchanged_by_the_new_field(monkeypatch):
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    inf2 = [NeuronDevice(i, "inf2", "v2", 2, 32 * 1024**3, False) for i in range(2)]
    ran = pantheon_neuron.run_workload(_workload(), inf2, 1, 0.01)
    skipped = pantheon_neuron.run_workload(
        next(w for w in registry.WORKLOADS if w.name == "transformer_train_step"),
        inf2, 1, 0.01,
    )
    assert sorted(ran) == sorted(skipped)
    assert "Score Method" in ran
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_mock_mode_still_refuses_to_invent_a_score(monkeypatch):
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    row = pantheon_neuron.run_workload(_workload(), TRN1, 1, 0.01)
    assert row["Score"] is None
    assert row["Score Method"] is None
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_hardware_path_requires_the_toolchain(monkeypatch):
    """Without neuronx-cc this raises BackendUnavailable specifically.

    Asserting on a bare Exception would pass on an ImportError from a typo
    in this module, which is exactly the bug a test like this should catch.
    """
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    with pytest.raises(nki_backend.BackendUnavailable, match="neuronx-cc"):
        memory_read.run(_workload().problem, duration=1)


# -- profiler counter reader -------------------------------------------------

def test_bandwidth_matches_the_hand_computed_probe_figure():
    """Counters captured on real hardware, reduced by the declared formula.
    56.58 GB/s was computed by hand from these two values during the probe;
    the code must agree."""
    from kernels import profiler as prof

    counters = {"hbm_read_bytes": 8470528, "total_time": 0.000149710075}
    assert round(prof.bandwidth_gbps(counters, "read"), 2) == 56.58


def test_bandwidth_rejects_missing_counters():
    from kernels import profiler as prof

    with pytest.raises(prof.ProfilerUnavailable, match="hbm_read_bytes"):
        prof.bandwidth_gbps({"total_time": 1.0}, "read")


def test_bandwidth_rejects_a_zero_total_time():
    from kernels import profiler as prof

    with pytest.raises(prof.ProfilerUnavailable, match="total_time"):
        prof.bandwidth_gbps({"hbm_read_bytes": 1, "total_time": 0}, "read")


def test_profiler_environment_sets_home_and_path():
    """Two traps that each cost a probe round trip: view exits on an unset
    HOME, and the Neuron tools shell out to each other via PATH."""
    from kernels import profiler as prof

    env = prof._environment()
    assert env["HOME"]
    assert prof.NEURON_BIN in env["PATH"].split(":")


def test_missing_neff_says_why(tmp_path):
    from kernels import profiler as prof

    with pytest.raises(prof.ProfilerUnavailable, match="compiler_workdir"):
        prof.find_neff(str(tmp_path))
