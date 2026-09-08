"""The pulse_virus workload: duty cycling around a verified GEMM.

The kernel is tensor_virus's, already exercised on hardware. What is new
here is the pulse arithmetic and the claim that its Score means something
different from the constant-load one.
"""

import pytest

import pantheon_neuron
from kernels import pulse_virus, registry


def _workload():
    return next(w for w in registry.WORKLOADS if w.name == "pulse_virus")


def test_pinned_problem_splits_into_equal_halves():
    cycle = pulse_virus.duty_plan(_workload().problem)
    assert cycle["on_s"] == 1.0
    assert cycle["off_s"] == 1.0
    assert cycle["period"] == 2


def test_uneven_duty_cycles_split_proportionally():
    cycle = pulse_virus.duty_plan({"duty_cycle": 0.25, "period_s": 4})
    assert cycle["on_s"] == 1.0
    assert cycle["off_s"] == 3.0


def test_a_constant_load_is_rejected():
    """duty_cycle 1 is tensor_virus, and silently measuring that would lie."""
    with pytest.raises(ValueError) as excinfo:
        pulse_virus.duty_plan({"duty_cycle": 1.0, "period_s": 2})
    assert "tensor_virus" in str(excinfo.value)


def test_an_idle_load_is_rejected():
    with pytest.raises(ValueError):
        pulse_virus.duty_plan({"duty_cycle": 0.0, "period_s": 2})


def test_a_zero_period_is_rejected():
    with pytest.raises(ValueError):
        pulse_virus.duty_plan({"duty_cycle": 0.5, "period_s": 0})


def test_it_shares_the_gemm_with_tensor_virus():
    """Same pinned shape, so the two Scores are at least the same problem."""
    from kernels import tensor_virus

    assert _workload().problem["shape"] == next(
        w for w in registry.WORKLOADS if w.name == "tensor_virus"
    ).problem["shape"]
    assert pulse_virus.tensor_virus is tensor_virus


def test_it_is_scored_from_the_monitor_like_the_other_compute_workloads():
    assert pantheon_neuron._wants_monitor_score(_workload())
    assert _workload().unit == "TFLOPS"


def test_it_is_dispatched():
    assert "pulse_virus" in pantheon_neuron.IMPLEMENTED


def test_mock_mode_invents_no_score(monkeypatch):
    from neuron_device import NeuronDevice

    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    row = pantheon_neuron.run_workload(
        _workload(),
        [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)],
        duration=1, monitor_period=0.1,
    )
    assert row["Score"] is None
