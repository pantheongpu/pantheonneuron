"""graph_replay and the monitor's other Score formula.

Every other monitor-scored workload averages effective_flops. This one
takes delta(completed) / period over the execution counter, which is a rate
rather than an average, and asks a different question: not how fast the
engines are, but how fast the runtime dispatches.
"""

import pytest

import pantheon_neuron
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

def _sample(completed):
    return {"neuron_runtime_data": [
        {"report": {"execution_stats": {
            "execution_summary": {"completed": completed}}}}
    ]}


def test_monitor_reports_a_delta_and_a_span():
    monitor = NeuronMonitor(mock=True)
    monitor._samples = [_sample(100), _sample(400), _sample(900)]
    monitor._sample_times = [10.0, 11.0, 12.0]

    summary = monitor.aggregate()

    assert summary["executions_delta"] == 800
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
    # Measured 3040 graph-steps/s on trn1.2xlarge 2026-09-10. The replay
    # count bounds the run before --duration does, so the pin *is* the
    # window: 10,000 replays gave 3.3 seconds, which is below what
    # neuron-monitor needs and is why the declared Score degraded to the
    # analytic fallback.
    assert plan["replays"] / 3040.0 >= 15.0, (
        f"{plan['replays']} replays is "
        f"{plan['replays'] / 3040.0:.1f}s at the measured rate -- too "
        "short for the monitor's completion counter to form a delta")


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
    assert "no replays" in graph_replay.verify_replays_completed(0, plan, 1.0, 1.0)


def test_guard_accepts_a_chain_that_read_back():
    plan = {"hidden": 2048, "replays": 10}
    assert graph_replay.verify_replays_completed(500, plan, 1.0, 4.2) is None


def test_it_is_dispatched():
    assert "graph_replay" in pantheon_neuron.IMPLEMENTED


def test_mock_mode_invents_no_score(monkeypatch):
    """The mock stream carries total_executions, not execution_summary.

    That is what keeps mock mode honest here: the rate is computed from
    delta(completed), which the mock never synthesises, so a mock run has
    no rate to report rather than a fabricated one.
    """
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    row = pantheon_neuron.run_workload(
        _workload(), TRN1, duration=1, monitor_period=0.05
    )
    assert row["Score"] is None
    assert row["Unit"] == "graph-steps/s"
