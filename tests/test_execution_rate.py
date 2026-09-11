"""delta(completed) / period, over the span the counter actually moved.

graph_replay reported 729.3 and 1174.8 graph-steps/s on two runs of the
same pinned problem, a 61% swing. The monitor starts before the workload
and stops after it, and a workload compiles before it executes -- during
which the counter is present and flat. Spanning the first sample that
*carried* the counter therefore put compile time in the denominator of a
rate, and the two runs differed in how long they compiled for.

The ratio implies 38% of the slower run's window was spent not executing,
which at DURATION=20 is about 7.6 seconds.
"""

import pytest

from neuron_monitor import execution_rate


def _flat_then_moving(idle_samples, moving_samples, per_sample=1000):
    """A counter that sits still while compiling, then advances."""
    times = [float(i) for i in range(idle_samples + moving_samples)]
    series = [(i, 0) for i in range(idle_samples)]
    series += [(idle_samples + i, (i + 1) * per_sample)
               for i in range(moving_samples)]
    return series, times


# -- the defect ---------------------------------------------------------------

def test_a_compile_prefix_is_not_counted_against_the_rate():
    """The whole point: 8 idle seconds must not dilute 12 working ones."""
    series, times = _flat_then_moving(idle_samples=9, moving_samples=12)
    rate = execution_rate(series, times)

    assert rate["executions_per_s"] == 1000.0
    assert rate["execution_span_s"] == 12.0
    assert rate["execution_idle_fraction"] == pytest.approx(0.4)

    # What the untrimmed span would have reported.
    untrimmed = (series[-1][1] - series[0][1]) / (
        times[series[-1][0]] - times[series[0][0]])
    assert untrimmed == 600.0
    assert untrimmed < rate["executions_per_s"] * 0.7


def test_the_graph_replay_swing_is_reproduced_by_an_idle_prefix():
    """Two runs, same true rate, different compile times.

    This is the shape of the 729 / 1175 pair: nothing about the workload
    differed, only how much of the window was spent compiling.
    """
    slow_series, slow_times = _flat_then_moving(9, 12)      # 38% idle
    fast_series, fast_times = _flat_then_moving(1, 20)      # barely any

    def untrimmed(series, times):
        return (series[-1][1] - series[0][1]) / (
            times[series[-1][0]] - times[series[0][0]])

    # Untrimmed, the same workload reports two very different rates.
    assert untrimmed(slow_series, slow_times) < 0.7 * untrimmed(
        fast_series, fast_times)

    # Trimmed, they agree.
    assert execution_rate(slow_series, slow_times)["executions_per_s"] == (
        execution_rate(fast_series, fast_times)["executions_per_s"])


def test_a_trailing_idle_run_is_trimmed_too():
    """The workload finishes; the monitor keeps sampling."""
    times = [float(i) for i in range(20)]
    series = [(i, i * 100) for i in range(11)]          # moving for 10s
    series += [(i, 1000) for i in range(11, 20)]        # then flat
    rate = execution_rate(series, times)

    assert rate["executions_per_s"] == 100.0
    assert rate["execution_span_s"] == 10.0


def test_both_ends_trim_together():
    times = [float(i) for i in range(30)]
    series = [(i, 0) for i in range(5)]
    series += [(5 + i, (i + 1) * 50) for i in range(10)]
    series += [(15 + i, 500) for i in range(15)]
    rate = execution_rate(series, times)

    assert rate["execution_span_s"] == 10.0
    assert rate["executions_per_s"] == 50.0


# -- refusing to report a rate that is not one --------------------------------

def test_a_counter_that_never_moved_reports_no_rate():
    """Not a slow rate -- no measurement. A workload that never executed
    must not publish a number that looks like throughput."""
    times = [float(i) for i in range(10)]
    series = [(i, 7) for i in range(10)]
    rate = execution_rate(series, times)

    assert "executions_per_s" not in rate
    assert rate["executions_delta"] == 0


def test_a_single_sample_reports_no_rate():
    """These two asserted `== {}` and now check the thing that matters.

    An empty dict was the old way of saying "nothing to measure", and it
    said nothing about *which* nothing -- see the three tests at the end
    of this file. What both cases have always meant is that no rate comes
    out, so that is what they assert.
    """
    assert "executions_per_s" not in execution_rate([(0, 100)], [0.0])


def test_no_samples_report_no_rate():
    assert "executions_per_s" not in execution_rate([], [])


def test_a_counter_that_went_backwards_reports_no_rate():
    """A runtime restart resets it; the delta is meaningless, not negative
    throughput."""
    times = [0.0, 1.0, 2.0]
    rate = execution_rate([(0, 900), (1, 950), (2, 10)], times)
    assert "executions_per_s" not in rate


def test_missing_timestamps_do_not_raise():
    """The times list is parallel to samples and could be short."""
    series, _ = _flat_then_moving(2, 3)
    assert "executions_per_s" not in execution_rate(series, [0.0])


def test_the_idle_fraction_says_how_much_was_compile():
    """The number that would have made the swing obvious at a glance."""
    series, times = _flat_then_moving(9, 12)
    assert execution_rate(series, times)["execution_idle_fraction"] > 0.3

    series, times = _flat_then_moving(1, 20)
    assert execution_rate(series, times)["execution_idle_fraction"] < 0.1


# -- "no rate" covered three different problems ------------------------------

def test_no_samples_says_so():
    absent = execution_rate([], [])
    assert absent["execution_samples"] == 0
    assert "no sample carried" in absent["execution_rate_absent"]
    assert "executions_per_s" not in absent


def test_one_sample_names_the_fix():
    """A single sample has no delta. Raise the duration or the rate."""
    absent = execution_rate([(0, 100)], [0.0])
    assert absent["execution_samples"] == 1
    message = absent["execution_rate_absent"]
    assert "no delta" in message
    assert "--duration" in message
    assert "executions_per_s" not in absent


def test_a_counter_that_never_moved_is_a_failing_workload():
    """Not a slow one, and the message must not read like one."""
    series = [(0, 100), (1, 100), (2, 100)]
    absent = execution_rate(series, [0.0, 1.0, 2.0])
    assert absent["executions_delta"] == 0
    assert absent["execution_samples"] == 3
    assert "did not advance" in absent["execution_rate_absent"]
    assert "executions_per_s" not in absent


def test_the_three_reasons_are_distinguishable():
    """Which was the whole point: one message for three causes is
    actionable in none of them.

    graph_replay produced a Score from this counter on 2026-09-08 and
    degraded to the analytic fallback on 2026-09-10, and the row said
    only "neuron-monitor reported no execution rate".
    """
    messages = {
        execution_rate([], [])["execution_rate_absent"],
        execution_rate([(0, 1)], [0.0])["execution_rate_absent"],
        execution_rate(
            [(0, 1), (1, 1)], [0.0, 1.0])["execution_rate_absent"],
    }
    assert len(messages) == 3, messages


def test_a_working_rate_carries_no_absence_message():
    """The control: a measurement must not explain why it is missing."""
    series = [(0, 0), (1, 10), (2, 20)]
    rate = execution_rate(series, [0.0, 1.0, 2.0])
    assert rate["executions_per_s"] > 0
    assert "execution_rate_absent" not in rate
