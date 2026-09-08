"""The pcie_bandwidth workload: transfer planning and the asymmetry guard.

This is the workload that notices a link that trained down -- a card
running at half its lanes passes every compute test and shows up only
here -- so the guard that flags a one-sided link is the part worth testing
without hardware.
"""

import pytest

import pantheon_neuron
from kernels import pcie_bandwidth, registry


def _workload():
    return next(w for w in registry.WORKLOADS if w.name == "pcie_bandwidth")


def test_the_pinned_problem_moves_both_directions():
    plan = pcie_bandwidth.transfer_plan(_workload().problem)
    assert plan["directions"] == ["h2d", "d2h"]
    assert plan["bytes"] == 1 << 30


def test_a_single_direction_is_honoured():
    plan = pcie_bandwidth.transfer_plan({"bytes": 1 << 20, "direction": "d2h"})
    assert plan["directions"] == ["d2h"]


def test_invalid_transfers_are_rejected():
    with pytest.raises(ValueError):
        pcie_bandwidth.transfer_plan({"bytes": 0})
    with pytest.raises(ValueError):
        pcie_bandwidth.transfer_plan({"bytes": 1 << 20, "direction": "sideways"})


# -- the asymmetry guard -----------------------------------------------------

def test_guard_accepts_normal_asymmetry():
    """d2h is routinely slower than h2d; that is not a fault."""
    legs = {"h2d": {"gbps": 10.0}, "d2h": {"gbps": 7.0}}
    assert pcie_bandwidth.verify_directions_are_balanced(legs) is None


def test_guard_flags_a_one_sided_link():
    legs = {"h2d": {"gbps": 12.0}, "d2h": {"gbps": 1.0}}
    message = pcie_bandwidth.verify_directions_are_balanced(legs)
    assert message is not None
    assert "negotiated link width" in message
    assert "d2h" in message


def test_guard_flags_a_dead_link():
    legs = {"h2d": {"gbps": 0.0}, "d2h": {"gbps": 0.0}}
    assert "no bytes moved" in pcie_bandwidth.verify_directions_are_balanced(legs)


def test_guard_is_quiet_for_a_single_direction():
    """Nothing to compare against is not evidence of imbalance."""
    assert pcie_bandwidth.verify_directions_are_balanced(
        {"h2d": {"gbps": 9.0}}
    ) is None


# -- score source ------------------------------------------------------------

def test_the_workload_reports_its_own_score():
    """Device DMA counters are device-side and never see a host transfer."""
    assert _workload().score_source.source == registry.INTERNAL
    assert _workload().unit == "GB/s"


def test_it_needs_no_special_capability():
    """Every part has a host link, so this should never be skipped."""
    assert _workload().requires == frozenset()


def test_it_is_dispatched():
    assert "pcie_bandwidth" in pantheon_neuron.IMPLEMENTED
