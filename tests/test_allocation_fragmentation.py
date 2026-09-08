"""The allocation_fragmentation workload: sizes, bounds, and its Score source.

The device allocator needs hardware. The size sequence that puts it under
fragmentation pressure does not, and it is the part that decides whether
the workload stresses anything at all -- a run of identical sizes would
leave a tidy free list and measure nothing.
"""

import pytest

import pantheon_neuron
from kernels import allocation_fragmentation as fragmentation
from kernels import registry


def _workload():
    return next(w for w in registry.WORKLOADS
                if w.name == "allocation_fragmentation")


def test_sequence_length_matches_the_pinned_count():
    sizes = fragmentation.size_sequence(_workload().problem)
    assert len(sizes) == 10000


def test_sizes_stay_within_the_pinned_bounds():
    problem = _workload().problem
    sizes = fragmentation.size_sequence(problem)
    assert min(sizes) >= problem["size_min"]
    assert max(sizes) <= problem["size_max"]


def test_sizes_are_mixed_rather_than_uniform():
    """A uniform run leaves a tidy free list and stresses nothing."""
    sizes = fragmentation.size_sequence(_workload().problem)
    assert len(set(sizes)) > 5


def test_the_sequence_is_deterministic():
    """A fragmentation failure that cannot be reproduced proves nothing."""
    problem = _workload().problem
    assert fragmentation.size_sequence(problem) == fragmentation.size_sequence(problem)


def test_sizes_cycle_back_after_exceeding_the_maximum():
    sizes = fragmentation.size_sequence(
        {"allocations": 6, "size_min": 4096, "size_max": 16384}
    )
    assert sizes == [4096, 8192, 16384, 4096, 8192, 16384]


def test_invalid_bounds_are_rejected():
    with pytest.raises(ValueError):
        fragmentation.size_sequence(
            {"allocations": 10, "size_min": 0, "size_max": 4096}
        )
    with pytest.raises(ValueError):
        fragmentation.size_sequence(
            {"allocations": 10, "size_min": 8192, "size_max": 4096}
        )
    with pytest.raises(ValueError):
        fragmentation.size_sequence(
            {"allocations": 0, "size_min": 4096, "size_max": 8192}
        )


def test_the_live_budget_leaves_headroom_on_a_neuroncore():
    """A failure here should mean fragmentation, not simple exhaustion."""
    assert fragmentation.LIVE_BUDGET_BYTES < 16 * 1024**3


def test_the_workload_reports_its_own_score():
    """No hardware counter measures allocator behaviour."""
    assert _workload().score_source.source == registry.INTERNAL
    assert _workload().unit == "allocation-events/s"


def test_it_is_dispatched():
    assert "allocation_fragmentation" in pantheon_neuron.IMPLEMENTED
