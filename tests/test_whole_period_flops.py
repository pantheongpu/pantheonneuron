"""mean(effective_flops) over the periods a core was busy throughout.

effective_flops is a rate over each sampling period, so the first and last
busy periods -- which the work only partly filled -- read low. Measured on
trn1.2xlarge 2026-09-10, tensor_virus 8192^3 coalesced for 30 s, one
sample per ~5 s, the kernel's own analytic rate 71.93 TFLOPS:
"""

import pytest

import pantheon_neuron
from neuron_monitor import NeuronMonitor, whole_period_flops

TFLOPS = 1e12
MEASURED = [18.25, 72.34, 72.35, 71.02, 72.35, 72.57, 53.43]  # TFLOPS
UTILISATION = [25.08, 99.22, 99.24, 97.40, 99.23, 99.53, 73.29]
ANALYTIC = 71.93


def test_the_whole_periods_agree_with_the_kernels_own_clock():
    summary = whole_period_flops([v * TFLOPS for v in MEASURED])
    assert summary["mean"] / TFLOPS == pytest.approx(ANALYTIC, rel=0.005)
    assert summary["samples"] == 5


def test_the_all_sample_mean_was_fourteen_percent_low():
    summary = whole_period_flops([v * TFLOPS for v in MEASURED])
    assert summary["mean_all_samples"] / TFLOPS == pytest.approx(61.76, abs=0.01)
    assert summary["mean_all_samples"] < summary["mean"] * 0.87


def test_the_edges_are_the_partly_busy_periods():
    """Utilisation says so independently: 25% and 73% at the ends, 97-99.5%
    between. Dropping the first and last busy periods drops exactly those."""
    interior = UTILISATION[1:-1]
    assert min(interior) > 97 and max(UTILISATION[0], UTILISATION[-1]) < 75


def test_no_whole_period_means_no_mean():
    summary = whole_period_flops([18.25 * TFLOPS, 53.43 * TFLOPS])
    assert "mean" not in summary
    assert "run longer" in summary["absent"]
    assert summary["samples"] == 0


def _sample(tflops):
    return {"neuron_runtime_data": [{"report": {"neuroncore_counters": {
        "period": 5.0,
        "neuroncores_in_use": {"0": {"effective_flops": tflops * TFLOPS,
                                      "neuroncore_utilization": 99.0}}}}}]}


def test_the_monitor_scores_the_whole_periods(monkeypatch):
    monitor = NeuronMonitor(mock=True)
    monitor._samples = [_sample(0.0)] + [_sample(v) for v in MEASURED] + [_sample(0.0)]
    monitor._sample_times = [float(i) for i in range(len(monitor._samples))]
    metrics = monitor.aggregate()
    workload = next(w for w in pantheon_neuron.registry.WORKLOADS if w.name == "tensor_virus")
    score = pantheon_neuron.monitor_score(workload, metrics)
    assert score == pytest.approx(ANALYTIC, rel=0.005)


def test_a_run_with_no_whole_period_says_why(monkeypatch):
    monitor = NeuronMonitor(mock=True)
    monitor._samples = [_sample(18.25), _sample(53.43)]
    monitor._sample_times = [0.0, 1.0]
    metrics = monitor.aggregate()
    workload = next(w for w in pantheon_neuron.registry.WORKLOADS if w.name == "tensor_virus")
    assert pantheon_neuron.monitor_score(workload, metrics) is None
    assert "run longer" in metrics["effective_flops_absent"]


# -- utilisation, and the zeros around and inside a run ---------------------

from neuron_monitor import whole_period_utilisation, whole_periods  # noqa: E402

# The same probe run: 19 samples of compile, the busy span, a 2-sample tail.
UTILISATION_SERIES = [0.0] * 19 + UTILISATION + [0.0, 0.0]


def test_utilisation_is_the_whole_periods_not_the_compile():
    """The published mean was over every sample, so tensor_virus reported
    42.25% on the 2026-09-11 full pass for a core 97-99.5% busy in every
    whole period it ran."""
    summary = whole_period_utilisation(UTILISATION_SERIES)
    assert summary["mean"] == pytest.approx(98.92, abs=0.01)
    assert summary["samples"] == 5
    assert summary["mean_all_samples"] < 30


def test_the_compile_gap_after_a_setup_blip_is_not_the_run():
    """memory_read's core 0 on trn1.2xlarge 2026-09-11: a setup blip, a
    230 s compile, then the loop. First-to-last-nonzero read 11.55%."""
    series = [0, 0, 1.5] + [0.0] * 46 + [35.3, 99.6, 99.6, 99.6, 98.3, 100.0, 66.8, 0, 0]
    summary = whole_period_utilisation(series)
    assert summary["mean"] == pytest.approx(99.42, abs=0.01)
    assert summary["samples"] == 5


def test_a_pause_splits_the_run_and_the_longest_stretch_is_measured():
    """The stated trade-off, asserted so it stays visible."""
    span, whole = whole_periods([0, 0, 30, 90, 0, 90, 90, 90, 40, 0])
    assert span == [90, 90, 90, 40]
    assert whole == [90, 90]


def test_the_flops_mean_ignores_a_setup_blip_too():
    readings = [0, 1e12, 0, 0, 0, 20e12, 72e12, 72e12, 72e12, 50e12, 0]
    summary = whole_period_flops(readings)
    assert summary["mean"] == pytest.approx(72e12)
    assert summary["samples"] == 3 and summary["samples_all"] == 5


def test_a_core_that_never_ran_has_no_flops_summary():
    monitor = NeuronMonitor(mock=True)

    def two_cores(tflops):
        return {"neuron_runtime_data": [{"report": {"neuroncore_counters": {
            "period": 5.0, "neuroncores_in_use": {
                "0": {"effective_flops": tflops * TFLOPS, "neuroncore_utilization": 99.0},
                "1": {"effective_flops": 0, "neuroncore_utilization": 0}}}}}]}
    monitor._samples = [two_cores(v) for v in [0.0, *MEASURED, 0.0]]
    monitor._sample_times = [float(i) for i in range(len(monitor._samples))]
    metrics = monitor.aggregate()
    assert set(metrics["effective_flops"]) == {"0"}
    assert metrics["neuroncore_utilization"]["1"]["mean"] == 0.0


def test_two_runtimes_give_one_reading_per_core_per_sample():
    """memory_read_agg's workers, trn1.2xlarge 2026-09-11: each runtime
    reports every core, the one it does not drive at 0. Appending both
    left core 0 with no uninterrupted busy block."""
    def worker_sample(u0, u1):
        return {"neuron_runtime_data": [
            {"pid": 5349, "report": {"neuroncore_counters": {"neuroncores_in_use": {
                "0": {"neuroncore_utilization": u0}, "1": {"neuroncore_utilization": 0}}}}},
            {"pid": 5350, "report": {"neuroncore_counters": {"neuroncores_in_use": {
                "0": {"neuroncore_utilization": 0}, "1": {"neuroncore_utilization": u1}}}}},
        ]}
    measured = [(0, 0), (0, 0), (99.0, 8.8), (99.6, 99.4), (99.6, 100.0),
                (100.0, 99.4), (65.6, 99.4), (0, 100.0), (0, 91.8), (0, 0)]
    monitor = NeuronMonitor(mock=True)
    monitor._samples = [worker_sample(a, b) for a, b in measured]
    monitor._sample_times = [float(i) for i in range(len(measured))]
    util = monitor.aggregate()["neuroncore_utilization"]
    assert util["0"]["samples"] == 3 and util["0"]["mean"] == pytest.approx(99.73, abs=0.01)
    assert util["1"]["samples"] == 5 and util["1"]["mean"] == pytest.approx(99.64, abs=0.01)
