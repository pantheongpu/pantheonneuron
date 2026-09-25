"""pcie_bandwidth records each leg's rate over time, not one figure per leg.

trn1-a's hour-long run read 17.9% under its own 300 s figure with both
directions down together, where the other two hosts held within 5%. That
rules out one direction and the d2h staging cliff; it does not say whether
the host slowed during the hour or was slow from the start, because each leg
reported a single 1800 s figure. WindowedRate is the bookkeeping that lets
the next long run say. These drive it with the timestamps a loop would.
"""

import pytest

from kernels import pcie_bandwidth
from kernels.pcie_bandwidth import WindowedRate

GIB = 1 << 30


def _leg(rate_gbps, seconds, start=100.0, slowdown_at=None, slow_rate=None):
    """Feed a WindowedRate one 1 GiB pass at a time at a given rate."""
    leg = WindowedRate(start, seconds)
    now = start
    while True:
        rate = (slow_rate if slowdown_at is not None and now - start >= slowdown_at
                else rate_gbps)
        now += GIB / (rate * 1e9)
        leg.record(now, GIB)
        if now >= start + seconds:
            return leg


def test_a_steady_leg_reads_the_same_in_every_window():
    summary = _leg(5.0, 150.0).summary()
    rates = summary["windows_gbps"]
    assert len(rates) == pcie_bandwidth.WINDOWS
    assert all(rate == pytest.approx(5.0, rel=0.1) for rate in rates)
    assert summary["last_over_first"] == pytest.approx(1.0, rel=0.1)


def test_a_leg_that_slows_part_way_says_so():
    """The distinction trn1-a's hour-long run could not make."""
    summary = _leg(6.2, 1800.0, slowdown_at=900.0, slow_rate=4.0).summary()
    assert summary["last_over_first"] == pytest.approx(4.0 / 6.2, rel=0.03)
    assert summary["windows_gbps"][0] == pytest.approx(6.2, rel=0.03)
    assert summary["windows_gbps"][-1] == pytest.approx(4.0, rel=0.03)


def test_a_leg_slow_from_the_start_is_flat():
    """The other half of the distinction: slow, but not slowing."""
    summary = _leg(5.1, 1800.0).summary()
    assert summary["last_over_first"] == pytest.approx(1.0, rel=0.05)


def test_every_byte_lands_in_some_window():
    leg = _leg(5.0, 150.0)
    assert sum(leg.bytes) == GIB * sum(leg.passes)


def test_a_window_with_no_pass_is_unknown_not_zero():
    """d2h at ~1 GB/s moves one 1 GiB pass a second or so; a window too
    short to finish one has no rate, which is not a rate of zero."""
    leg = WindowedRate(0.0, 10.0)
    leg.record(0.5, GIB)
    leg.record(9.5, GIB)
    summary = leg.summary()
    assert summary["windows_gbps"][0] is not None
    assert summary["windows_gbps"][4] is None
    assert summary["window_passes"][4] == 0


def test_a_last_pass_that_overruns_the_budget_is_charged_to_its_real_span():
    leg = WindowedRate(0.0, 10.0)
    for second in range(9):                 # one pass inside each of windows 0-8
        leg.record(second + 0.5, GIB)
    leg.record(12.0, GIB)   # started before the edge, finished after it
    last = leg.summary()["windows_gbps"][-1]
    # 1 GiB over the 3 s from the window's start to when the pass finished,
    # not over the nominal 1 s window.
    assert last == pytest.approx(GIB / 3.0 / 1e9, rel=1e-3)


def test_one_filled_window_has_no_trend():
    leg = WindowedRate(0.0, 10.0)
    leg.record(0.5, GIB)
    assert leg.summary()["last_over_first"] is None


def test_a_zero_budget_does_not_divide_by_zero():
    leg = WindowedRate(0.0, 0.0)
    leg.record(0.0, GIB)
    assert leg.summary()["window_passes"][0] == 1


def test_the_loop_feeds_it_and_the_leg_publishes_it():
    """The loop only runs on a device, so this reads it: every leg gets a
    WindowedRate, every pass is recorded, and the summary goes into the leg
    that per_direction publishes."""
    import sourcecheck
    code = sourcecheck.flat_function_code(pcie_bandwidth.run)
    assert "over_time = WindowedRate (" in code
    assert "over_time . record (" in code
    assert "** over_time . summary ( )" in code
