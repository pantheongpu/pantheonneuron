"""The pulse_virus workload: duty cycling around a verified GEMM.

The kernel is tensor_virus's, already exercised on hardware. What is new
here is the pulse arithmetic and the claim that its Score means something
different from the constant-load one.
"""

import pytest

import pantheon_neuron
import sourcecheck
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


# -- a duty cycle that never idled -------------------------------------------

def test_a_run_that_never_idled_is_reported():
    """This workload's premise is that half the run is idle.

    A run whose idle half vanished is tensor_virus under another name at
    roughly twice the analytic figure, and every number in the row would
    look healthy. 13.85 TFLOPS against tensor_virus's 26.12 on
    trn1.2xlarge 2026-09-10 is the evidence it is pulsing today; nothing
    was reading for the ratio that would show it had stopped.
    """
    message = pulse_virus.verify_duty_cycle_was_observed(2.0, 2.0, 0.5)
    assert message is not None
    assert "idle half did not happen" in message


def test_a_run_that_barely_loaded_is_reported():
    message = pulse_virus.verify_duty_cycle_was_observed(0.1, 2.0, 0.5)
    assert message is not None
    assert "not filling its window" in message


def test_the_expected_duty_is_accepted():
    """Including the upward bias from the barrier: loaded_s carries the
    tail of the last submission, so the observed ratio runs slightly
    above the request."""
    for loaded in (1.0, 1.02, 1.1):
        assert pulse_virus.verify_duty_cycle_was_observed(
            loaded, 2.0, 0.5) is None


def test_a_zero_length_run_has_no_duty_to_observe():
    assert "no wall time" in pulse_virus.verify_duty_cycle_was_observed(
        0.0, 0.0, 0.5)


def test_the_tolerance_admits_the_barrier_bias_and_nothing_larger():
    """0.15 leaves room for a slower part -- where one pass is a larger
    share of the loaded half -- without admitting a run that never idled.
    """
    assert pulse_virus.DUTY_TOLERANCE < 0.5, (
        "a tolerance of half the duty would accept a run with no idle at all")
    assert pulse_virus.verify_duty_cycle_was_observed(
        (0.5 + pulse_virus.DUTY_TOLERANCE - 0.01) * 2.0, 2.0, 0.5) is None
    assert pulse_virus.verify_duty_cycle_was_observed(
        (0.5 + pulse_virus.DUTY_TOLERANCE + 0.01) * 2.0, 2.0, 0.5) is not None


def test_the_kernel_reports_and_checks_the_observed_duty():
    code = sourcecheck.flat_function_code(pulse_virus.run)
    assert '"observed_duty"' in code
    assert "verify_duty_cycle_was_observed" in code
