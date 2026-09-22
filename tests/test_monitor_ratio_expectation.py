"""A monitor Score's cross-check only catches things if we expect a value.

`declared_over_kernel` holds the monitor's Score against the kernel's own
figure. Every monitor-scored workload reads about 1.0 -- except omni_virus,
which has read 1.12 since it was written, excused in a comment as "a stated
reason". So a genuine 12% error on that row looked exactly like the quirk.

The reason turned out to be real work: the compiler lowers the chain's cumsum
onto the Tensor Engine and the monitor counts it, tile^3/2 beyond the kernel's
two matmuls at the pinned shape (measured 2026-09-22, tools/omni_flops.py).
It cannot be folded into the kernel's count with one formula, because the
lowering costs 2*tile^3 at tile 2048 and tile^3/2 at 8192. So the expectation
is declared and checked instead.
"""

import pytest

import pantheon_neuron
import sourcecheck
from kernels import registry

# Measured across five hosts, 2026-09-21: every monitor-scored workload's
# declared_over_kernel, and these must all pass.
MEASURED = {
    "tensor_virus": 0.9999,
    "int_virus": 1.0002,
    "pulse_virus": 0.9983,
    "transformer_virus": 0.9992,
    "graph_replay": 1.0001,
    "omni_virus": 1.1206,
}


def _workload(name):
    return next(w for w in registry.WORKLOADS if w.name == name)


@pytest.mark.parametrize("name,ratio", sorted(MEASURED.items()))
def test_every_measured_ratio_passes(name, ratio):
    """A check that fired on healthy hardware would be worse than none."""
    assert pantheon_neuron.monitor_ratio_unexpected(
        _workload(name), ratio, {"ran_pinned_shape": True}) is None


@pytest.mark.parametrize("ratio", [1.0, 1.05, 1.25, 1.5, 0.5])
def test_omni_virus_off_its_expectation_says_so(ratio):
    """The 12% that was invisible: off 1.125 is now a finding."""
    message = pantheon_neuron.monitor_ratio_unexpected(
        _workload("omni_virus"), ratio, {"ran_pinned_shape": True})
    assert message is not None
    assert "against 1.125 expected" in message
    assert "cumsum" in message


@pytest.mark.parametrize("ratio", [1.12, 1.1206, 1.125, 1.16])
def test_omni_virus_within_tolerance_is_silent(ratio):
    assert pantheon_neuron.monitor_ratio_unexpected(
        _workload("omni_virus"), ratio, {"ran_pinned_shape": True}) is None


def test_a_workload_at_the_default_is_checked_against_one():
    """Everything else counts what it issues; 1.12 there is a finding."""
    message = pantheon_neuron.monitor_ratio_unexpected(
        _workload("tensor_virus"), 1.12, {})
    assert message is not None and "against 1.000 expected" in message


def test_the_check_is_skipped_when_the_pinned_shape_did_not_run():
    """The gap IS the lowering, and a smaller shape lowers differently:
    omni_virus reads 1.500 at tile 2048 against 1.125 at 8192."""
    assert pantheon_neuron.monitor_ratio_unexpected(
        _workload("omni_virus"), 1.5, {"ran_pinned_shape": False}) is None
    # But a workload with no declared expectation is not excused by it.
    assert pantheon_neuron.monitor_ratio_unexpected(
        _workload("tensor_virus"), 1.5, {"ran_pinned_shape": False}) is not None


@pytest.mark.parametrize("ratio", [None, "1.1", float("nan")])
def test_a_missing_ratio_is_not_a_finding(ratio):
    result = pantheon_neuron.monitor_ratio_unexpected(
        _workload("omni_virus"), ratio, {})
    if ratio != ratio:  # NaN compares unequal to itself; it must not pass silently
        assert result is not None
    else:
        assert result is None


def test_every_expectation_names_a_monitor_scored_workload_with_a_reason():
    for name, (expected, because) in registry.MONITOR_OVER_KERNEL_EXPECTED.items():
        workload = _workload(name)
        assert workload.score_source.source == registry.MONITOR, name
        assert expected > 0
        assert len(because) > 40, f"{name}: the reason must say what the gap is"


def test_the_expectation_is_actually_consulted_where_the_ratio_is_recorded():
    """A check nothing calls is the failure this repo catalogues."""
    body = sourcecheck.flat_function_code(pantheon_neuron._measure_started)
    assert "monitor_ratio_unexpected (" in body
