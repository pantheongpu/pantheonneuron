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


def test_eviction_subtracts_the_block_it_freed():
    """The tally must follow what was actually released.

    Sizes span 4 KiB to 16 MiB here, so subtracting the size of the block
    just allocated -- rather than the one popped -- lets live_bytes drift
    from what is held. The eviction loop hides most of it by running until
    the tally drops under budget, but the tally is what the failure message
    reports when an allocation fails, and a diagnosis built on it would be
    wrong by whatever the drift is.
    """
    import sourcecheck
    from kernels import allocation_fragmentation as af

    code = sourcecheck.function_code(af.run)
    assert "retained . append ( ( block , size ) )" in code
    assert "_ , freed = retained . pop ( 0 )" in code
    assert "live_bytes -= freed" in code
    assert "live_bytes -= size" not in code


def test_the_drift_the_old_accounting_produced():
    """Replayed from the pinned sizes, so the size of the bug is recorded."""
    from kernels import registry
    from kernels.allocation_fragmentation import (
        size_sequence, KEEP_EVERY, LIVE_BUDGET_BYTES)

    problem = {w.name: w.problem for w in registry.WORKLOADS}[
        "allocation_fragmentation"]
    sizes = size_sequence(problem)

    def replay(correct):
        retained, tracked = [], 0
        for index, size in enumerate(sizes):
            if index % KEEP_EVERY == 0:
                retained.append(size)
                tracked += size
            while tracked > LIVE_BUDGET_BYTES and retained:
                popped = retained.pop(0)
                tracked -= popped if correct else size
        return sum(retained), tracked

    held_old, tracked_old = replay(correct=False)
    held_new, tracked_new = replay(correct=True)

    assert tracked_new == held_new, "the fixed tally matches what is held"
    assert abs(held_old - tracked_old) > 10 * 1024**2, "the old one drifts"


def test_every_distinct_size_is_warmed_before_the_clock():
    """Each size is its own graph shape, and there are thirteen of them.

    Without a warm-up the first pass compiles all thirteen inside the
    measurement. Measured on trn1.2xlarge 2026-09-10: three repeats read
    549, 2,646 and 2,750 allocation-events/s -- monotonically rising, which
    is warm-up rather than noise.
    """
    import sourcecheck
    from kernels import allocation_fragmentation as af

    code = sourcecheck.function_code(af.run)
    warmup = code[:code.index("started = time . perf_counter ( )")]
    assert "for size in sorted ( set ( sizes ) )" in warmup
    assert "xm . wait_device_ops ( )" in warmup


def test_the_warm_up_covers_every_shape_the_loop_will_use():
    """sorted(set(sizes)) is exactly the distinct shapes, no more."""
    from kernels import registry
    from kernels.allocation_fragmentation import size_sequence

    sizes = size_sequence({w.name: w.problem for w in registry.WORKLOADS}[
        "allocation_fragmentation"])
    assert len(set(sizes)) == 13
    assert set(sorted(set(sizes))) == set(sizes)


def test_the_row_says_which_limit_stopped_the_run():
    """--duration does not bound this workload, and the row must not imply it did.

    10,000 allocations take under four seconds on trn1, so --duration 30
    and --duration 60 measure the same four seconds. Measured 2026-09-10:
    3.69s for a requested 10s, 3.92s for a requested 30s.
    """
    from kernels import allocation_fragmentation as af

    short = af.verify_window_is_long_enough(3.8, 30, "allocations")
    assert short is not None
    assert "does not lengthen it" in short
    assert "--repeat" in short

    assert af.verify_window_is_long_enough(30.0, 30, "duration") is None


def test_a_duration_bound_run_is_not_flagged_for_being_short():
    """If the clock stopped it, the clock is what the caller asked for."""
    from kernels import allocation_fragmentation as af
    assert af.verify_window_is_long_enough(2.0, 2, "duration") is None


def test_the_short_window_explains_the_residual_scatter():
    """Recorded as arithmetic, because the drift and the scatter are
    different problems with different fixes.

    The warm-up removed the drift: repeats went from 549-2,750 ordered
    (cv 0.63) to 2,012-2,708 unordered (cv 0.15). What is left is a rate
    measured over 3.8 seconds, and --duration cannot lengthen it.
    """
    measured_window = 3.8
    from kernels.allocation_fragmentation import MIN_WINDOW_SECONDS

    assert measured_window < MIN_WINDOW_SECONDS
    # Before the warm-up the spread was four times worse and ordered.
    assert 0.63 / 0.15 > 4
