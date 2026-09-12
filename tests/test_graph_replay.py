"""graph_replay and the monitor's other Score formula.

Every other monitor-scored workload averages effective_flops. This one
takes sum(completed) / sum(period) over the execution counter, a rate
rather than an average, and asks a different question: not how fast the
engines are, but how fast the runtime dispatches.
"""

import pytest

import pantheon_neuron
import sourcecheck
from kernels import graph_replay, registry
from neuron_device import NeuronDevice
from neuron_monitor import NeuronMonitor


TRN1 = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]


def _workload():
    return next(w for w in registry.WORKLOADS if w.name == "graph_replay")


# -- which formula applies ---------------------------------------------------

def test_it_is_scored_from_the_execution_counter_not_flops():
    """The distinction the flops gate was written to preserve."""
    workload = _workload()
    assert workload.score_source.source == registry.MONITOR
    assert pantheon_neuron._wants_execution_rate(workload)
    assert not pantheon_neuron._wants_monitor_score(workload)
    assert workload.unit == "graph-steps/s"


def test_the_compute_workloads_do_not_use_the_rate_path():
    for name in ("tensor_virus", "int_virus", "pulse_virus"):
        workload = next(w for w in registry.WORKLOADS if w.name == name)
        assert not pantheon_neuron._wants_execution_rate(workload), name


def test_the_gate_follows_the_registry():
    for workload in registry.WORKLOADS:
        declares = (
            workload.score_source is not None
            and workload.score_source.source == registry.MONITOR
            and pantheon_neuron.EXECUTIONS_COUNTER in workload.score_source.counters
        )
        assert pantheon_neuron._wants_execution_rate(workload) is declares


# -- the rate itself ---------------------------------------------------------

def test_score_is_the_execution_rate_the_monitor_measured():
    metrics = {"samples": 10, "executions_per_s": 815.5}
    assert pantheon_neuron.monitor_score(_workload(), metrics) == 815.5


def test_no_score_without_a_rate():
    """A run too short for two counter samples has no rate to report."""
    assert pantheon_neuron.monitor_score(
        _workload(), {"samples": 1, "total_executions": 40}
    ) is None


def test_a_total_alone_is_not_a_rate():
    """total_executions says how far it got, not how fast."""
    assert pantheon_neuron.monitor_score(
        _workload(), {"samples": 9, "total_executions": 8000}
    ) is None


# -- the monitor's aggregation -----------------------------------------------

def _sample(completed, period=1.0):
    return {"neuron_runtime_data": [
        {"report": {"execution_stats": {
            "period": period,
            "execution_summary": {"completed": completed}}}}
    ]}


def test_monitor_sums_the_periods_and_rates_the_whole_ones():
    """Per-period tallies: the total is their sum, and the rate comes from
    the periods the device was busy throughout."""
    monitor = NeuronMonitor(mock=True)
    monitor._samples = [_sample(0), _sample(100), _sample(400),
                        _sample(400), _sample(50), _sample(0)]
    monitor._sample_times = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

    summary = monitor.aggregate()

    assert summary["total_executions"] == 950
    assert summary["executions_total"] == 950
    assert summary["execution_span_s"] == 2.0
    assert summary["executions_per_s"] == 400.0


def test_monitor_reports_no_rate_from_a_single_sample():
    monitor = NeuronMonitor(mock=True)
    monitor._samples = [_sample(100)]
    monitor._sample_times = [10.0]

    assert "executions_per_s" not in monitor.aggregate()


def test_start_clears_the_timestamps_with_the_samples():
    """Reusing a monitor must not date this run against the previous one.

    The timestamps are a parallel array to the samples, so leaving them
    behind would index this run's executions against the last run's clock
    and produce a rate over a span that never happened.
    """
    monitor = NeuronMonitor(mock=True)
    monitor._samples = [_sample(1)]
    monitor._sample_times = [99.0]

    monitor.start([0])
    monitor.stop()

    assert 99.0 not in monitor._sample_times
    assert len(monitor._sample_times) == len(monitor._samples)


# -- the kernel --------------------------------------------------------------

def test_pinned_problem_is_a_small_graph_replayed_often():
    """Small on purpose: a big graph would measure the engines instead.

    The replay count asserted a literal 10000 and failed when the
    registry repinned to 60000 -- testing that nobody had changed the
    pin, rather than that the pin means what it should. What it should
    mean is a measurement window long enough for the declared Score
    source to form a delta, so that is what is asserted.
    """
    plan = graph_replay.replay_plan(_workload().problem)
    assert plan["hidden"] == 2048
    # Measured 3044 graph-steps/s on trn1.2xlarge 2026-09-10. The count
    # must outlast the default --duration, so the caller's duration sets
    # the window: at 60,000 the count bound first, at ~20 s, and the
    # per-period rate had one whole ~5 s period to divide.
    default_duration = 30
    assert plan["replays"] / 3044.0 > default_duration, (
        f"{plan['replays']} replays is {plan['replays'] / 3044.0:.1f}s at "
        "the measured rate -- the count, not --duration, would bound the run")


def test_the_window_is_inversely_proportional_to_the_rate():
    """The uncomfortable part, asserted so it stays visible.

    A count-bounded workload measures itself over a window that shortens
    as the device gets faster, so a faster part is *more* likely to lose
    its declared Score. The pin has to hold up at rates well above the
    one it was chosen against.
    """
    replays = graph_replay.replay_plan(_workload().problem)["replays"]
    for rate, floor in ((3040.0, 15.0), (6080.0, 8.0)):
        assert replays / rate >= floor, (rate, replays / rate)


def test_invalid_graph_sizes_are_rejected():
    with pytest.raises(ValueError):
        graph_replay.replay_plan({"hidden": 100, "replays": 10})
    with pytest.raises(ValueError):
        graph_replay.replay_plan({"hidden": 2048, "replays": 0})


def test_guard_flags_replays_that_never_executed():
    """Submissions are cheap; mark_step queues and returns."""
    plan = {"hidden": 2048, "replays": 10}
    message = graph_replay.verify_replays_completed(500, plan, 1.0, None)
    assert message is not None
    assert "not shown to have executed" in message


def test_guard_flags_a_run_with_no_replays():
    plan = {"hidden": 2048, "replays": 10}
    assert "no replays" in graph_replay.verify_replays_completed(0, plan, 1.0, True)


def test_guard_accepts_only_a_chain_that_landed_exactly():
    plan = {"hidden": 2048, "replays": 10}
    assert graph_replay.verify_replays_completed(500, plan, 1.0, True) is None
    wrong = graph_replay.verify_replays_completed(500, plan, 1.0, False)
    assert wrong and "exactly once" in wrong


# -- a chain whose value says how many replays ran ---------------------------

def _matmul(a, b):
    return [[sum(a[r][j] * b[j][c] for j in range(len(b))) for c in range(len(b[0]))]
            for r in range(len(a))]


def _chain(hidden, executed):
    """The kernel's chain in plain Python: identity times the one-column
    shift, `executed` times."""
    identity = [[1.0 if r == c else 0.0 for c in range(hidden)] for r in range(hidden)]
    shift_one = [[1.0 if c == (r + 1) % hidden else 0.0 for c in range(hidden)]
                 for r in range(hidden)]
    state = identity
    for _ in range(executed):
        state = _matmul(state, shift_one)
    return state


def _lands(state, shift):
    hidden = len(state)
    return all(state[r][c] == (1.0 if c == (r + shift) % hidden else 0.0)
               for r in range(hidden) for c in range(hidden))


def test_the_chain_lands_where_the_replays_and_warmup_put_it():
    hidden, replays = 8, 13
    executed = replays + graph_replay.WARMUP_REPLAYS
    shift = graph_replay.chain_shift(replays, hidden)
    assert _lands(_chain(hidden, executed), shift)


def test_one_skipped_replay_is_visible():
    hidden, replays = 8, 13
    shift = graph_replay.chain_shift(replays, hidden)
    executed = replays + graph_replay.WARMUP_REPLAYS
    assert not _lands(_chain(hidden, executed - 1), shift)
    assert not _lands(_chain(hidden, executed + 1), shift)


def test_forgetting_the_warmup_would_fail_every_honest_run():
    """The warm-up is a replay of the same chain. A shift that ignored it
    would call every correct run wrong -- a check failing by accident is
    as useless as one passing by accident."""
    hidden, replays = 8, 13
    honest = _chain(hidden, replays + 1)
    assert not _lands(honest, replays % hidden)
    assert _lands(honest, graph_replay.chain_shift(replays, hidden))


def test_the_old_all_ones_chain_could_not_count():
    """Each all-ones replay multiplies every element by hidden. bf16 tops
    out near 2^128, and 2048^12 = 2^132: by the twelfth replay the value is
    inf and stays inf, so 12 executed replays and 60,000 read the same."""
    hidden, bf16_max_exponent = 2048, 128
    replays_to_overflow = next(n for n in range(1, 64)
                               if hidden ** n >= 2 ** bf16_max_exponent)
    assert replays_to_overflow == 12


def test_a_chain_that_did_not_land_invalidates_the_score():
    code = sourcecheck.flat_function_code(graph_replay.run)
    assert '"score_invalid" : chain_exact is not True' in code
    assert "torch . ones" not in code


def test_it_is_dispatched():
    assert "graph_replay" in pantheon_neuron.IMPLEMENTED


def test_mock_mode_invents_no_score(monkeypatch):
    """The mock stream carries total_executions, not execution_summary.

    That is what keeps mock mode honest here: the rate is computed from
    per-period completion tallies, which the mock never synthesises, so a mock run has
    no rate to report rather than a fabricated one.
    """
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    row = pantheon_neuron.run_workload(
        _workload(), TRN1, duration=1, monitor_period=0.05
    )
    assert row["Score"] is None
    assert row["Unit"] == "graph-steps/s"


# -- every replay is its own execution ---------------------------------------

def test_the_docstring_no_longer_claims_the_replays_are_batched():
    """It claimed about four replays per NEFF execution, from one period's
    tally (14,737) set against the run's 60,000. The per-period counts sum
    to 60,003 on trn1.2xlarge 2026-09-10."""
    doc = graph_replay.__doc__
    assert "It does batch them" not in doc
    assert "Every replay is its own execution" in doc
    assert "60,003" in doc


def test_the_measured_ratio_is_reported_as_a_disagreement():
    """A factor between the two figures has to reach the row.

    The factor of four graph_replay showed was a misread counter, not a
    property of the device -- but it reached the row, and the row saying
    so is what kept it from being quoted as a dispatch rate.
    """
    message = pantheon_neuron.override_disagreement(
        2931.7326, 589.5893, "the completion counter")
    assert message is not None
    assert "factor of 5" in message

    # And from the counts rather than the rates, which is the cleaner
    # comparison because the monitor's span outruns the kernel's window.
    assert pantheon_neuron.override_disagreement(
        60000, 14737, "the completion counter") is not None



# -- the last busy period is reported before the monitor stops ---------------

def _tally(completed):
    return {"neuron_runtime_data": [{"report": {"execution_stats": {
        "period": 5.0, "execution_summary": {"completed": completed}}}}]}


def test_the_wait_ends_on_the_first_idle_period(monkeypatch):
    monitor = NeuronMonitor(mock=False)
    monitor._samples = [_tally(15000)]
    arrivals = iter([_tally(7465), _tally(0)])

    def later(_):
        monitor._samples.append(next(arrivals))
    monkeypatch.setattr("neuron_monitor.time.sleep", later)
    assert monitor.await_idle_period(timeout=5.0) is True
    assert len(monitor._samples) == 3


def test_an_idle_period_from_before_the_call_does_not_count():
    """The compile's zeros are idle periods too; only one after the work
    says the last busy period has been reported."""
    monitor = NeuronMonitor(mock=False)
    monitor._samples = [_tally(0), _tally(0), _tally(15000)]
    assert monitor.await_idle_period(timeout=0.3, poll=0.1) is False


def test_the_orchestrator_waits_for_the_tail_only_for_a_rate():
    code = sourcecheck.flat_function_code(pantheon_neuron._measure_started)
    wait = code.index("monitor . await_idle_period ( )")
    assert "_wants_execution_rate ( workload )" in code[code.rindex("if", 0, wait):wait]
    assert wait < code.index("metrics = monitor . stop ( )")


def test_executions_that_match_the_replays_are_quiet():
    """60,003 for 60,000 on trn1.2xlarge 2026-09-10: warm-up and setup."""
    metrics = {"execution_tail_reported": True, "executions_total": 60003}
    assert pantheon_neuron.executions_unaccounted(metrics, {"replays": 60000}) is None


def test_a_shortfall_is_reported_once_the_tail_is_in():
    metrics = {"execution_tail_reported": True, "executions_total": 45709}
    why = pantheon_neuron.executions_unaccounted(metrics, {"replays": 60000})
    assert why and "45709" in why and "60000" in why


def test_a_shortfall_before_the_tail_is_the_monitors_not_the_devices():
    """45,709 was exactly this: the monitor stopped before the last busy
    period's tally. Without the idle period there is nothing to say."""
    metrics = {"execution_tail_reported": False, "executions_total": 45709}
    assert pantheon_neuron.executions_unaccounted(metrics, {"replays": 60000}) is None
