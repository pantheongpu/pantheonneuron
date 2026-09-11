"""Executions per second from neuron-monitor's per-period completion counts.

``execution_summary.completed`` is a tally for each sampling period, not a
running total. Measured on trn1.2xlarge 2026-09-10, graph_replay, 60,000
replays, the monitor sampling every ~5 s through compile and an idle tail:

    0 ... 0, 6657, 15409, 15273, 15199, 7465, 0, 0     sum 60,003

It falls to zero when the work stops, which a running total cannot. The
old reading -- last minus first -- was the difference between two
periods' tallies, and produced 729, 1012, 1506 or nothing depending on
where the samples fell.
"""

import pytest

from neuron_monitor import execution_rate

# The measured series, verbatim: (sample index, completed, period).
MEASURED_PERIODS = [0, 4.9947, 4.99946, 5.00083, 4.99976, 5.00031, 4.99906,
                    5.00079, 4.99991, 4.99956, 5.00011, 5.00062, 4.99909,
                    5.00031, 4.99987, 5.00072, 4.99947, 4.99998, 4.99965,
                    5.00087, 4.99935, 4.99986]
MEASURED_COUNTS = [0] * 15 + [6657, 15409, 15273, 15199, 7465, 0, 0]
MEASURED = [(i, c, p) for i, (c, p) in enumerate(zip(MEASURED_COUNTS, MEASURED_PERIODS))]
LOOP_RATE = 3057.1  # the kernel's own clock, same run
REPLAYS = 60000


def test_the_measured_run_gives_the_loops_rate():
    rate = execution_rate(MEASURED)
    assert rate["executions_per_s"] == pytest.approx(LOOP_RATE, rel=0.01)
    assert rate["execution_samples_used"] == 3


def test_the_total_is_every_replay_plus_setup():
    """The sum is what the device completed. The old max over samples
    reported one period's 15,409 as the run's total."""
    total = execution_rate(MEASURED)["executions_total"]
    assert total == 60003
    assert total - REPLAYS == 3
    assert max(MEASURED_COUNTS) < total / 3


def test_last_minus_first_would_have_said_nothing_ran():
    """What the old formula computed on this run: the last sample is idle,
    so last - first is 0 - 0."""
    assert MEASURED_COUNTS[-1] - MEASURED_COUNTS[0] == 0
    assert "executions_per_s" in execution_rate(MEASURED)


def test_the_partial_edge_periods_are_not_in_the_rate():
    """6657 and 7465 are the first and last busy periods, each partly idle;
    including them would read 2400/s against the loop's 3057."""
    active = [(c, p) for _, c, p in MEASURED if c > 0]
    with_edges = sum(c for c, _ in active) / sum(p for _, p in active)
    assert with_edges < LOOP_RATE * 0.8
    assert execution_rate(MEASURED)["executions_per_s"] > LOOP_RATE * 0.99


def test_the_span_is_the_rates_denominator_and_sits_inside_the_run():
    """Three whole periods, 15 s, inside the loop's 19.63 s -- which is what
    span_outran_the_kernel checks."""
    span = execution_rate(MEASURED)["execution_span_s"]
    assert span == pytest.approx(15.0, abs=0.01)
    assert span < 19.63


def test_compile_time_cannot_dilute_the_rate():
    """Idle periods before the work carry zero and are not active, however
    many there are -- the 2026-09-08 "compile in the span" hypothesis has
    no way in."""
    busy = [(0, 100, 1.0), (1, 500, 1.0), (2, 500, 1.0), (3, 500, 1.0), (4, 200, 1.0)]
    short = execution_rate([(i, 0, 1.0) for i in range(2)]
                           + [(i + 2, c, p) for i, c, p in busy])
    long = execution_rate([(i, 0, 1.0) for i in range(40)]
                          + [(i + 40, c, p) for i, c, p in busy])
    assert short["executions_per_s"] == long["executions_per_s"] == 500.0


# -- the three kinds of nothing ----------------------------------------------

def test_no_samples_says_so():
    absent = execution_rate([])
    assert absent["execution_samples"] == 0
    assert "no sample carried" in absent["execution_rate_absent"]
    assert "executions_per_s" not in absent


def test_a_counter_that_never_moved_is_a_failing_workload():
    absent = execution_rate([(i, 0, 5.0) for i in range(6)])
    assert absent["executions_total"] == 0
    assert "completed nothing" in absent["execution_rate_absent"]
    assert "executions_per_s" not in absent


def test_too_few_whole_periods_names_the_fix():
    """Two busy periods are both edges: neither was busy throughout."""
    absent = execution_rate([(0, 0, 5.0), (1, 9000, 5.0), (2, 4000, 5.0), (3, 0, 5.0)])
    assert absent["executions_total"] == 13000
    assert "run longer" in absent["execution_rate_absent"]
    assert "executions_per_s" not in absent


def test_the_three_reasons_are_distinguishable():
    reasons = {
        execution_rate([])["execution_rate_absent"],
        execution_rate([(0, 0, 5.0), (1, 0, 5.0)])["execution_rate_absent"],
        execution_rate([(0, 10, 5.0), (1, 10, 5.0)])["execution_rate_absent"],
    }
    assert len(reasons) == 3


def test_a_working_rate_carries_no_absence_message():
    assert "execution_rate_absent" not in execution_rate(MEASURED)


def test_a_period_the_monitor_did_not_report_is_not_divided_by():
    series = [(0, 10, 5.0), (1, 500, None), (2, 500, 5.0), (3, 10, 5.0)]
    rate = execution_rate(series)
    assert rate["execution_samples_used"] == 1
    assert rate["executions_per_s"] == 100.0


# -- the period neuron-monitor will actually honour -------------------------

from neuron_monitor import NeuronMonitor, period_string  # noqa: E402


def test_the_period_goes_out_in_whole_seconds():
    """inf2.xlarge 2026-09-11: "1.0s" and "0.5s" were ignored in favour of
    the 5 s default; "1s" delivered 1 s periods."""
    assert period_string(1.0) == "1s"
    assert period_string(5.0) == "5s"
    assert period_string(0.5) == "1s"
    assert period_string(0.2) == "1s"
    assert period_string(2.6) == "3s"
    for value in (0.2, 0.5, 1.0, 2.6, 5.0):
        text = period_string(value)
        assert "." not in text and text.endswith("s") and int(text[:-1]) >= 1


def test_the_config_uses_the_honoured_string():
    import inspect
    source = inspect.getsource(NeuronMonitor.start)
    assert '"period": period_string(self.period_seconds)' in source
    assert 'f"{self.period_seconds}s"' not in source


def test_the_delivered_period_is_reported():
    def sample(period):
        return {"neuron_runtime_data": [{"report": {"neuroncore_counters": {
            "period": period, "neuroncores_in_use": {
                "0": {"effective_flops": 7e13, "neuroncore_utilization": 99.0}}}}}]}
    monitor = NeuronMonitor(mock=True)
    monitor._samples = [sample(0.004), sample(5.0), sample(5.001), sample(4.999)]
    monitor._sample_times = [0.0, 5.0, 10.0, 15.0]
    assert monitor.aggregate()["sample_period_s"] == 5.0
