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
