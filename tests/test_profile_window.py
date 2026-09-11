"""A profiled execution's bandwidth divides by the time it was active.

neuron-profile's total_time opens with an idle startup the workload's
back-to-back executions do not pay. Measured on trn1.2xlarge 2026-09-10,
memory_read's kernel graph (every row: that graph, identified by reading
the planned bytes with no writes):
"""

import pytest

from kernels import memory_read, memory_write, profiler
import pantheon_neuron

GIB = 1 << 30
# buffer GiB: (total_time s, total_active_time s, wall seconds per pass)
MEASURED = {
    1: (0.006020, 0.003934, 0.004025),
    2: (0.009950, 0.007867, 0.007968),
    4: (0.017825, 0.015746, 0.015855),
    8: (0.033540, 0.031463, 0.031614),
}


def _counters(gib):
    total, active, _ = MEASURED[gib]
    return {"hbm_read_bytes": gib * GIB, "total_time": total, "total_active_time": active}


def test_over_active_time_the_rate_does_not_depend_on_the_buffer():
    rates = [profiler.bandwidth_gbps(_counters(g), "read") for g in MEASURED]
    assert max(rates) - min(rates) < 1.0
    assert all(rate == pytest.approx(273.0, abs=1.0) for rate in rates)


def test_over_total_time_it_did():
    """What the declared formula published: 178 at 1 GiB, 256 at 8."""
    rates = {g: g * GIB / MEASURED[g][0] / 1e9 for g in MEASURED}
    assert rates[1] < 180 and rates[8] > 255
    assert rates[8] / rates[1] > 1.4


def test_the_startup_is_constant_and_the_loop_does_not_pay_it():
    startups = [MEASURED[g][0] - MEASURED[g][1] for g in MEASURED]
    assert max(startups) - min(startups) < 0.0001
    assert startups[0] == pytest.approx(0.00208, abs=0.0001)
    # The loop's fixed cost per pass: intercept of wall time against size.
    walls = [MEASURED[g][2] for g in (1, 2, 4, 8)]
    slope = (walls[3] - walls[0]) / 7
    assert walls[0] - slope < 0.0002


def test_the_active_rate_sits_just_above_the_wall_clock():
    """Wall time includes the loop's per-pass overhead, so the active-time
    rate should be a little higher, never lower and never much higher."""
    for gib, (_, active, wall) in MEASURED.items():
        ratio = (gib * GIB / active) / (gib * GIB / wall)
        assert 1.0 < ratio < 1.03, (gib, ratio)


def test_without_an_active_time_it_falls_back_and_says_so():
    counters = {"hbm_read_bytes": 8470528, "total_time": 0.000149710075}
    assert profiler.execution_window(counters) == (0.000149710075, "total_time")
    assert round(profiler.bandwidth_gbps(counters, "read"), 2) == 56.58


def test_an_active_time_longer_than_the_execution_is_not_believed():
    counters = {"hbm_read_bytes": GIB, "total_time": 0.004, "total_active_time": 0.005}
    assert profiler.execution_window(counters)[1] == "total_time"


def test_the_row_records_the_basis_and_the_startup():
    for key in ("profiler_time_basis", "profiler_startup_s"):
        assert key in pantheon_neuron._PROVENANCE_KEYS


# -- the tolerance the bias used to need ------------------------------------

def test_a_wrong_graph_the_old_tolerance_accepted_is_now_refused():
    """At 1 GiB a buffer-copy graph (reads 1 GiB, writes 1 GiB, no vector
    work) was also a candidate. Over its active time it reads 176.5 against
    the wall clock's 266.7 -- ratio 0.66, inside the old 50%."""
    for module in (memory_read, memory_write):
        assert module.verify_against_analytic(176.5, 266.7, tolerance=0.5) is None
        assert module.verify_against_analytic(176.5, 266.7) is not None


def test_the_kernels_own_graph_is_well_inside_it():
    for gib, (_, active, wall) in MEASURED.items():
        profile = gib * GIB / active / 1e9
        analytic = gib * GIB / wall / 1e9
        assert memory_read.verify_against_analytic(profile, analytic) is None, gib
