"""The branches that turn a monitor finding into row text, driven end to end.

executions_unaccounted, span_outran_the_kernel and thin_monitor_sample are
each unit-tested, and each ran on hardware on 2026-09-10 -- where every one
of them, correctly, stayed quiet. So the lines in _measure_once that append
their findings to the row had never executed anywhere: a typo there would
surface only on the day a real shortfall happened, which is catalogue #10
(correct, tested, and unreachable). This drives each one with a monitor
that reports exactly the problem.
"""

import typing

import pytest

import neuron_monitor
import pantheon_neuron
from kernels import registry
from neuron_device import NeuronDevice

TRN1 = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]


def _named(name):
    return next(w for w in registry.WORKLOADS if w.name == name)


class _StubMonitor:
    """Reports whatever metrics a test hands it; never touches hardware."""

    metrics: typing.ClassVar[dict] = {}
    tail = True

    def __init__(self, period_seconds=1.0, mock=False):
        pass

    def start(self, device_indices):
        return True

    def await_idle_period(self, timeout=15.0, poll=0.25):
        return type(self).tail

    def stop(self):
        return dict(type(self).metrics)

    def shutdown(self):
        """The teardown _measure_once guarantees in a finally."""


def _run(monkeypatch, workload, kernel_result, metrics, tail=True, score=3000.0):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "0")
    _StubMonitor.metrics, _StubMonitor.tail = metrics, tail
    monkeypatch.setattr(neuron_monitor, "NeuronMonitor", _StubMonitor)
    monkeypatch.setattr(pantheon_neuron, "_execute", lambda *a, **k: score)
    monkeypatch.setitem(pantheon_neuron._LAST_RUN, workload.name, dict(kernel_result))
    return pantheon_neuron._measure_once(workload, TRN1, 30, 1.0)


def _replay_metrics(**overrides):
    base = {"samples": 9, "executions_total": 90930, "execution_samples": 9,
            "execution_active_periods": 7, "execution_samples_used": 5,
            "execution_span_s": 25.0, "executions_per_s": 3038.8}
    base.update(overrides)
    return base


def test_a_quiet_replay_row_carries_no_detail(monkeypatch):
    """The control: the hardware run's own numbers produce no text."""
    row = _run(monkeypatch, _named("graph_replay"),
               {"replays": 90927, "elapsed_s": 30.0}, _replay_metrics())
    assert row["Status"] == "PASS" and row["Detail"] == ""
    assert row["Score"] == pytest.approx(3038.8)


def test_a_device_shortfall_reaches_the_row(monkeypatch):
    row = _run(monkeypatch, _named("graph_replay"),
               {"replays": 90927, "elapsed_s": 30.0},
               _replay_metrics(executions_total=45709))
    assert "45709 executions for 90927 replays" in row["Detail"]


def test_a_shortfall_before_the_tail_stays_quiet(monkeypatch):
    row = _run(monkeypatch, _named("graph_replay"),
               {"replays": 90927, "elapsed_s": 30.0},
               _replay_metrics(executions_total=45709), tail=False)
    assert "executions for" not in row["Detail"]
    assert row["Telemetry"]["execution_tail_reported"] is False


def test_a_span_longer_than_the_run_reaches_the_row(monkeypatch):
    row = _run(monkeypatch, _named("graph_replay"),
               {"replays": 90927, "elapsed_s": 20.0},
               _replay_metrics(execution_span_s=25.0))
    assert "execution span is 25.00s against the kernel's 20.00s" in row["Detail"]


def test_a_thin_rate_reaches_the_row(monkeypatch):
    row = _run(monkeypatch, _named("graph_replay"),
               {"replays": 90927, "elapsed_s": 30.0},
               _replay_metrics(execution_samples_used=1, execution_span_s=5.0))
    assert "1 whole sampling period(s)" in row["Detail"]


def test_a_thin_flops_mean_reaches_the_row(monkeypatch):
    metrics = {"samples": 3, "effective_flops": {"0": {
        "mean": int(72.4e12), "peak": int(72.6e12), "samples": 1,
        "samples_all": 3, "mean_all_samples": int(48e12)}}}
    row = _run(monkeypatch, _named("tensor_virus"),
               {"elapsed_s": 10.0, "analytic_tflops": 70.4}, metrics, score=70.4)
    assert row["Score"] == pytest.approx(72.4)
    assert "effective_flops averaged over 1 whole sampling period(s)" in row["Detail"]


def test_no_whole_flops_period_says_why(monkeypatch):
    """The absent path names the reason whole_period_flops gave."""
    metrics = {"samples": 2,
               "effective_flops": {"0": {"peak": int(53e12), "samples": 0, "samples_all": 2,
                                         "mean_all_samples": int(35e12),
                                         "absent": "the core was busy across 2 sampling "
                                                   "period(s) and none of them throughout"}},
               "effective_flops_absent": "the core was busy across 2 sampling period(s) "
                                         "and none of them throughout"}
    row = _run(monkeypatch, _named("tensor_virus"),
               {"elapsed_s": 5.0, "analytic_tflops": 70.4}, metrics, score=None)
    assert "none of them throughout" in row["Detail"]


def test_a_monitor_fallback_to_the_analytic_figure_says_why(monkeypatch):
    """trn1.2xlarge 2026-09-11, tensor_virus --duration 5: one busy period,
    none whole, analytic 72.45 published with an empty Detail."""
    reason = ("the core was busy across 1 sampling period(s) and none of them "
              "throughout, so no period measures its rate -- run longer")
    metrics = {"samples": 3,
               "effective_flops": {"0": {"peak": int(38.9e12), "samples": 0,
                                         "samples_all": 1, "mean_all_samples": int(38.9e12),
                                         "absent": reason}},
               "effective_flops_absent": reason}
    row = _run(monkeypatch, _named("tensor_virus"),
               {"elapsed_s": 5.0, "analytic_tflops": 72.45,
                "score_method": "analytic", "analytic_basis": "FLOPs issued / wall time"},
               metrics, score=72.45)
    assert row["Score"] == pytest.approx(72.45)
    assert "analytic figure is published instead" in row["Detail"]
    assert "run longer" in row["Detail"]
