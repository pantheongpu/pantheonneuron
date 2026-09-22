"""The memory Score is one profiled execution; the run is the whole window.

neuron-profile replays the NEFF once, so an hour-long memory_read publishes
the same Score as a five-minute one. These check that a run which slowed over
its window says so, and that healthy hardware never trips it.
"""

import pytest

import sourcecheck
from kernels import memory_read, memory_write, profiler

# (profiler Score, analytic) for memory_read and memory_write on every host of
# the 2026-09-21 publication pass: three trn1.2xlarge, two inf2.xlarge.
HEALTHY = [
    (272.9152, 271.6866), (274.1841, 276.1467),
    (273.0441, 271.7434), (274.8217, 275.9865),
    (272.9804, 271.6374), (275.0706, 276.1197),
    (272.9404, 271.3718), (274.8987, 274.2130),
    (272.9933, 271.1493), (274.5629, 274.1244),
    # The same three trn1.2xlarge at --duration 3600: the case this exists
    # for. trn1-a's memory_read drifted to 0.9902 over the hour, the lowest
    # healthy ratio on record.
    (273.28, 270.59), (274.52, 273.89),
    (273.12, 271.46), (274.93, 275.52),
    (272.86, 271.49), (275.18, 275.53),
    # This branch's own confirmation run, trn1.2xlarge 2026-09-22, 60 s. The
    # memory_read pair is the lowest healthy ratio on record, 0.9894.
    (272.8049, 269.92), (275.0416, 274.10),
]


@pytest.mark.parametrize("profiled,sustained", HEALTHY)
def test_healthy_hardware_never_trips_it(profiled, sustained):
    assert profiler.verify_sustained_matches_burst(profiled, sustained) is None


def test_a_run_that_slowed_says_so():
    """A 10% throttle after warm-up: the Score would still say 273."""
    message = profiler.verify_sustained_matches_burst(273.0, 245.7)
    assert message is not None
    assert "10.0% below" in message and "burst" in message


def test_the_threshold_is_one_sided():
    """Sustained above burst is launch overhead going the other way, not a fault."""
    assert profiler.verify_sustained_matches_burst(273.0, 300.0) is None


def test_the_boundary():
    assert profiler.verify_sustained_matches_burst(100.0, 97.1) is None
    assert profiler.verify_sustained_matches_burst(100.0, 96.9) is not None


@pytest.mark.parametrize("profiled,sustained", [(None, 270.0), (270.0, None), (0, 270.0)])
def test_missing_figures_are_not_a_finding(profiled, sustained):
    assert profiler.verify_sustained_matches_burst(profiled, sustained) is None


@pytest.mark.parametrize("module", [memory_read, memory_write])
def test_both_kernels_run_the_check(module):
    code = sourcecheck.flat_function_code(module._run)
    assert "profiler . verify_sustained_matches_burst (" in code
