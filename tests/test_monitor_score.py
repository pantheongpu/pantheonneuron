"""The Score path for workloads that declare neuron-monitor as their source.

Five core workloads -- tensor_virus, int_virus, pulse_virus,
transformer_virus, omni_virus -- declare ``mean(effective_flops) / 1e12``.
That counter lives only in the neuron-monitor stream, so unlike the
bandwidth kernels the Score cannot come from the kernel: it has to be read
out of telemetry after the monitor stops. These tests cover that arithmetic
and, more importantly, the cases where the counter is missing -- where the
only correct answer is no Score at all.
"""

import pantheon_neuron
from kernels import registry
from neuron_device import NeuronDevice


TRN1 = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]


def _workload(name):
    return next(w for w in registry.WORKLOADS if w.name == name)


def _telemetry(*core_means):
    return {
        "samples": 10,
        "effective_flops": {
            str(index): {"mean": int(value), "peak": int(value)}
            for index, value in enumerate(core_means)
        },
    }


# -- which workloads use this path -------------------------------------------

def test_every_compute_workload_declares_the_monitor_source():
    """If one of these stops declaring it, its Score silently goes missing."""
    for name in ("tensor_virus", "int_virus", "pulse_virus",
                 "transformer_virus", "omni_virus"):
        assert pantheon_neuron._wants_monitor_score(_workload(name)), name


def test_bandwidth_workloads_do_not_use_the_monitor_path():
    """memory_read's Score comes from neuron-profile; this must not shadow it."""
    for name in ("memory_read", "memory_write"):
        assert not pantheon_neuron._wants_monitor_score(_workload(name)), name
        assert pantheon_neuron.monitor_score(
            _workload(name), _telemetry(8e11)
        ) is None


def test_graph_replay_is_monitor_sourced_but_not_scored_in_flops():
    """It reads neuron-monitor too, but its formula and unit are different.

    graph_replay declares delta(completed) / period in graph-steps/s. A gate
    on the source alone would hand it the FLOPS arithmetic and publish a
    TFLOPS magnitude under a graph-steps/s label. effective_flops is
    non-zero during a graph replay, so this would fire on real hardware
    rather than staying a theoretical mistake.
    """
    graph_replay = _workload("graph_replay")

    assert graph_replay.score_source.source == registry.MONITOR
    assert graph_replay.unit == "graph-steps/s"
    assert not pantheon_neuron._wants_monitor_score(graph_replay)
    assert pantheon_neuron.monitor_score(graph_replay, _telemetry(2e12)) is None


def test_the_gate_follows_the_registry_rather_than_a_hardcoded_list():
    """Any workload declaring the flops counter is scored through this path.

    A new compute workload that declares effective_flops must not need a
    second edit here to be scored, and one that stops declaring it must not
    keep being scored from it.
    """
    for workload in registry.WORKLOADS:
        declares = (
            workload.score_source is not None
            and workload.score_source.source == registry.MONITOR
            and pantheon_neuron.FLOPS_COUNTER in workload.score_source.counters
        )
        assert pantheon_neuron._wants_monitor_score(workload) is declares, (
            workload.name
        )


# -- the arithmetic ----------------------------------------------------------

def test_score_is_flops_over_one_trillion():
    """875,590,474,948 FLOP/s was the observed inf2 probe figure: 0.8756 TFLOPS."""
    score = pantheon_neuron.monitor_score(
        _workload("tensor_virus"), _telemetry(875_590_474_948)
    )
    assert round(score, 4) == 0.8756


def test_mean_is_taken_across_cores_not_sum():
    """A two-core part running the same work per core is not twice as fast."""
    score = pantheon_neuron.monitor_score(
        _workload("tensor_virus"), _telemetry(1e12, 1e12)
    )
    assert score == 1.0


def test_cores_are_averaged_when_they_differ():
    score = pantheon_neuron.monitor_score(
        _workload("tensor_virus"), _telemetry(1e12, 3e12)
    )
    assert score == 2.0


# -- absent counters produce no Score, never a fabricated one ----------------

def test_no_score_when_the_counter_is_absent():
    """The mock path and a kernel that never reached the engine land here."""
    assert pantheon_neuron.monitor_score(
        _workload("tensor_virus"), {"samples": 12}
    ) is None


def test_no_score_when_telemetry_never_started():
    assert pantheon_neuron.monitor_score(
        _workload("tensor_virus"), {"samples": 0}
    ) is None


def test_no_score_when_the_counter_is_present_but_empty():
    assert pantheon_neuron.monitor_score(
        _workload("tensor_virus"), {"samples": 5, "effective_flops": {}}
    ) is None


def test_non_numeric_counter_values_are_ignored():
    """Rather than crashing the run or coercing junk into a number."""
    metrics = {"samples": 5, "effective_flops": {"0": {"mean": None}}}
    assert pantheon_neuron.monitor_score(_workload("tensor_virus"), metrics) is None


# -- end to end through run_workload -----------------------------------------

def test_mock_run_records_no_score(monkeypatch):
    """Mock mode exercises the whole path and must still produce no number."""
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    row = pantheon_neuron.run_workload(
        _workload("tensor_virus"), TRN1, duration=1, monitor_period=0.1
    )
    assert row["Score"] is None
    assert row["Unit"] == "TFLOPS"


def test_a_failing_workload_gets_no_score_from_telemetry(monkeypatch):
    """Telemetry keeps sampling through a failure; that is not a result.

    Without the status gate a workload that raised on the first pass would
    still be scored from whatever the monitor happened to catch, and a FAIL
    row would carry a number that looks like a measurement.
    """
    def explode(*_args, **_kwargs):
        raise RuntimeError("kernel raised")

    monkeypatch.setattr(pantheon_neuron, "_execute", explode)
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "start",
        lambda self, indices: True,
    )
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "stop",
        lambda self: _telemetry(2e12),
    )

    row = pantheon_neuron.run_workload(
        _workload("tensor_virus"), TRN1, duration=1, monitor_period=0.1
    )
    assert row["Status"] == "FAIL"
    assert row["Score"] is None


def test_a_passing_run_is_scored_from_telemetry(monkeypatch):
    monkeypatch.setattr(pantheon_neuron, "_execute", lambda *a, **k: None)
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "start",
        lambda self, indices: True,
    )
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "stop",
        lambda self: _telemetry(2e12, 2e12),
    )

    row = pantheon_neuron.run_workload(
        _workload("tensor_virus"), TRN1, duration=1, monitor_period=0.1
    )
    assert row["Status"] == "PASS"
    assert row["Score"] == 2.0
    assert row["Score Method"] == registry.MONITOR


def test_a_passing_run_without_the_counter_says_so(monkeypatch):
    """A PASS with no Score must explain itself, not leave an empty cell."""
    monkeypatch.setattr(pantheon_neuron, "_execute", lambda *a, **k: None)
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "start",
        lambda self, indices: True,
    )
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "stop",
        lambda self: {"samples": 8},
    )

    row = pantheon_neuron.run_workload(
        _workload("tensor_virus"), TRN1, duration=1, monitor_period=0.1
    )
    assert row["Status"] == "PASS"
    assert row["Score"] is None
    assert "effective_flops" in row["Detail"]
