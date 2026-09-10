"""The pcie_bandwidth workload: transfer planning and the asymmetry guard.

This is the workload that notices a link that trained down -- a card
running at half its lanes passes every compute test and shows up only
here -- so the guard that flags a one-sided link is the part worth testing
without hardware.
"""

import pytest

import pantheon_neuron
import sourcecheck
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


# -- the asymmetry that was the harness, not the link ------------------------
#
# trn1.2xlarge 2026-09-08 reported d2h 1.0 GB/s against h2d 6.0 and the guard
# fired. The legs were not symmetric: h2d reused one host tensor while d2h
# called .cpu(), which allocates a fresh 1 GiB host destination every pass.
# Both legs now copy into a buffer allocated before the clock starts.

def test_the_code_only_filter_ignores_comments():
    """Otherwise the checks below pass on prose and prove nothing."""
    assert "forbidden" not in sourcecheck.code_only("x = 1  # forbidden\n")
    assert "forbidden" not in sourcecheck.code_only('def f():\n    """forbidden"""\n')
    assert "forbidden" in sourcecheck.code_only("forbidden = 1\n")


def test_the_filter_keeps_dict_keys_on_continuation_lines():
    """It used to eat them, which silently broke every check about one.

    A dict key after a comma sits behind a non-logical newline, and
    treating that as a statement boundary made it look like a docstring.
    An `in` assertion could then never pass, and a `not in` assertion
    passed for entirely the wrong reason.
    """
    kept = sourcecheck.code_only('x = {\n    "key": 1,\n    "other": 2,\n}\n')
    assert '"key"' in kept and '"other"' in kept


def test_the_filter_still_drops_real_docstrings():
    """Module, function and class, which is what it is for."""
    assert "mod" not in sourcecheck.code_only('"""mod"""\nx = 1\n')
    assert "doc" not in sourcecheck.code_only('def f():\n    """doc"""\n    return 1\n')
    assert "cls" not in sourcecheck.code_only('class C:\n    """cls"""\n    x = 1\n')


def test_both_legs_copy_into_a_preallocated_destination():
    """The fix, asserted against the source: neither leg may allocate per pass.

    Checked textually because running it needs a device. The two shapes
    that reintroduce the defect are `.cpu()` and rebinding a name to a
    freshly produced tensor inside the loop, so both are named here.
    """
    import inspect

    # Tokenise the whole function, then slice the result: slicing the raw
    # source first leaves a fragment tokenize cannot indent-parse.
    code = sourcecheck.code_only(inspect.getsource(pcie_bandwidth.run))
    cut = code.index("while time . perf_counter")
    setup, loop = code[:cut], code[cut:]

    assert ". cpu ( )" not in loop, "d2h must copy_ into a preallocated buffer"
    assert "resident = host . to ( device )" not in loop, (
        "h2d must copy_ into the resident buffer, not rebind a new one"
    )
    assert loop.count("copy_") == 2, "one copy_ per direction"

    # Both destinations exist before the timed region begins.
    assert "landing" in setup and "host_buffer" in setup
    assert "resident = host . to ( device )" in setup


def test_the_symmetry_check_fails_the_code_it_was_written_for():
    """The pre-fix loop, verbatim. A check that cannot catch it proves nothing."""
    before_the_fix = '''def run(problem, duration):
    host = torch.ones(4)
    resident = host.to(device)
    started = time.perf_counter()
    while time.perf_counter() < deadline:
        if direction == "h2d":
            resident = host.to(device)
        else:
            _ = resident.cpu()
'''
    code = sourcecheck.code_only(before_the_fix)
    loop = code[code.index("while time . perf_counter"):]

    assert ". cpu ( )" in loop
    assert "resident = host . to ( device )" in loop
    assert loop.count("copy_") != 2


def test_the_result_records_how_the_number_was_produced():
    """A row without these keys predates a fix and is not comparable.

    Three methodologies have now produced a d2h figure -- per-pass
    allocation, preallocated pageable, preallocated pinned -- and they are
    not comparable with each other. The row has to say which it was.
    """
    import inspect

    source = inspect.getsource(pcie_bandwidth.run)
    for key in ('"buffers": "preallocated"', '"host_source_pinned"',
                '"host_landing_pinned"', '"h2d_sources_alternate"'):
        assert key in source, key


def test_pinning_is_reported_as_achieved_not_as_requested():
    """pin_memory() is a CUDA-shaped API and may not apply on this stack.

    If it silently no-ops, the bounce-buffer explanation for the d2h
    asymmetry is still live -- so the flag has to come from whether the call
    succeeded, not from whether it was attempted.
    """
    import inspect

    source = inspect.getsource(pcie_bandwidth.run)
    assert "def host_buffer(" in source
    assert "except (RuntimeError, NotImplementedError, AssertionError)" in source
    assert "return plain, False" in source


def test_the_h2d_leg_does_not_send_identical_bytes_every_pass():
    """Otherwise an inflated h2d and a depressed d2h look the same.

    If the runtime can serve a repeated identical copy without moving
    bytes, the 6.0 GB/s is the wrong number rather than the 1.1 -- and the
    ratio alone cannot tell those apart.
    """
    import inspect

    code = sourcecheck.code_only(inspect.getsource(pcie_bandwidth.run))
    loop = code[code.index("while time . perf_counter"):]
    assert "host_alt" in loop, "h2d must alternate its source buffer"


def test_the_asymmetry_warning_points_at_the_harness_first():
    """The guard's first suspect must be the cause it has actually had."""
    message = pcie_bandwidth.verify_directions_are_balanced({
        "h2d": {"gbps": 6.0},
        "d2h": {"gbps": 1.0},
    })
    assert message is not None
    assert "d2h" in message
    assert "preallocated" in message
    # The link is still named, just not first.
    assert "link width" in message
    assert message.index("preallocated") < message.index("link width")


def test_a_balanced_link_raises_nothing():
    assert pcie_bandwidth.verify_directions_are_balanced({
        "h2d": {"gbps": 6.0},
        "d2h": {"gbps": 5.2},
    }) is None


def test_the_floor_is_where_it_is_documented():
    """25%: a 4x split is flagged, a 3x one is not."""
    flagged = pcie_bandwidth.verify_directions_are_balanced({
        "h2d": {"gbps": 8.0}, "d2h": {"gbps": 1.9},
    })
    quiet = pcie_bandwidth.verify_directions_are_balanced({
        "h2d": {"gbps": 8.0}, "d2h": {"gbps": 2.1},
    })
    assert flagged is not None and quiet is None


def test_the_in_place_assumption_is_flagged_not_asserted():
    """The h2d leg's justification rests on semantics XLA does not have.

    kv_cache_churn measured that an assignment produces a new tensor
    rather than writing in place. The same reasoning underpins this leg's
    "no per-pass allocation" claim, and the fix it belongs to moved the
    numbers by noise. Until a hardware run settles it, the source must say
    so rather than claim the allocation was avoided.
    """
    import inspect

    source = inspect.getsource(pcie_bandwidth.run)
    assert "SUSPECT" in source
    assert "xla_has_no_in_place_write" in source
    # The old, confident phrasing must not come back.
    assert "would allocate a new device\n                # buffer per pass" not in source


def test_pinning_is_reported_false_when_it_did_not_happen():
    """pin_memory() returned unpinned buffers on this stack, three runs
    running. The flag has to say that, because the bounce-buffer
    explanation for the d2h asymmetry stands or falls on it."""
    import inspect

    source = inspect.getsource(pcie_bandwidth.run)
    # Achieved, not requested: the except clause returns False rather than
    # letting the caller assume pinning worked.
    assert "return plain, False" in source
    assert '"host_source_pinned": pinned_source' in source


def test_the_docstring_records_what_the_sweep_settled():
    """The asymmetry is explained; a reader should not re-derive it.

    Three explanations failed one at a time before a size sweep showed the
    question was wrong: everything had assumed a constant rate, and the
    rate is not constant.
    """
    doc = pcie_bandwidth.__doc__
    assert "the link is healthy" in doc
    assert "bandwidth-bound, not overhead-bound" in doc
    assert "signature of a staging" in doc  # wrapped in the docstring
    # And that the pinned size sits in the degraded regime.
    assert "pinned 1 GiB measures the degraded regime" in doc


def test_the_pinned_size_is_known_to_be_past_the_cliff():
    """Recorded as arithmetic so the registry decision is informed.

    d2h holds ~2.9 GB/s to 16 MiB and collapses to 0.88 by 64 MiB, and the
    pinned problem is 1024 MiB -- past the cliff in both directions.
    """
    problem = {w.name: w.problem for w in registry.WORKLOADS}["pcie_bandwidth"]
    pinned_mib = problem["bytes"] // 1024**2
    assert pinned_mib == 1024

    # Measured on trn1.2xlarge 2026-09-10.
    d2h_small, d2h_large = 2.83, 1.09      # 16 MiB, 1024 MiB
    h2d_peak, h2d_pinned = 11.55, 7.37
    assert d2h_small > 2.5 * d2h_large, "the cliff is real"
    assert h2d_pinned < h2d_peak, "h2d is past its peak at the pin too"
