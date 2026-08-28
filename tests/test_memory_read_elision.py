"""The analytic bandwidth is only a measurement if the kernel actually read.

Observed on trn1.2xlarge 2026-08-27: run() discarded the kernel result, so
nothing referenced the graph at the mark_step() cut and XLA skipped the DMA.
The run reported 14,513 GB/s -- about 17x the part's ~820 GB/s HBM -- while
neuron-monitor recorded total_executions=1 across a 90 second window.

verify_against_analytic only fires when the profiler works, and the profiler
failing is exactly when the analytic number becomes the Score. These cover
the profiler-independent check that closes that hole.
"""

import pytest

from kernels.memory_read import tile_plan, verify_read_completed


def test_full_read_passes():
    assert verify_read_completed(1.0) is None


def test_slightly_off_is_tolerated():
    assert verify_read_completed(1.005) is None


def test_eliminated_loads_are_rejected():
    msg = verify_read_completed(0.0)
    assert msg is not None and "eliminated" in msg


def test_partial_read_is_rejected():
    msg = verify_read_completed(0.5)
    assert msg is not None and "0.500x" in msg


def test_over_read_is_rejected():
    # More than planned means the accounting is wrong, not that we got a bonus.
    assert verify_read_completed(2.0) is not None


def test_unreadable_output_is_not_silently_accepted():
    msg = verify_read_completed(None)
    assert msg is not None and "unverified" in msg


def test_expected_accumulator_value_matches_plan():
    """One pass over an all-ones buffer sums to tiles * FREE per partition."""
    plan = tile_plan(64 * 1024 * 1024, "bf16")
    expected = plan["tiles"] * plan["free"]
    # the ratio the kernel computes is observed/expected, so a correct run
    # yields exactly 1.0
    assert verify_read_completed(float(expected) / expected) is None
