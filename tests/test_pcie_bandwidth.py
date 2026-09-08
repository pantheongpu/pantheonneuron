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


# -- the asymmetry that was the harness, not the link ------------------------
#
# trn1.2xlarge 2026-09-08 reported d2h 1.0 GB/s against h2d 6.0 and the guard
# fired. The legs were not symmetric: h2d reused one host tensor while d2h
# called .cpu(), which allocates a fresh 1 GiB host destination every pass.
# Both legs now copy into a buffer allocated before the clock starts.

def _code_only(source: str) -> str:
    """Strip comments and docstrings, leaving what actually executes.

    A textual check that reads comments is not checking the code: the
    comment explaining why `resident = host.to(device)` must not be in the
    loop contains that exact string, and matched.
    """
    import io
    import tokenize

    kept = []
    previous = None
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        # A STRING alone on a logical line is a docstring, not an operand.
        if (token.type == tokenize.STRING
                and previous in (None, tokenize.NEWLINE, tokenize.NL,
                                 tokenize.INDENT, tokenize.DEDENT)):
            continue
        kept.append(token.string)
        if token.type not in (tokenize.NL, tokenize.NEWLINE):
            previous = token.type
        else:
            previous = token.type
    return " ".join(kept)


def test_the_code_only_filter_ignores_comments():
    """Otherwise the checks below pass on prose and prove nothing."""
    assert "forbidden" not in _code_only("x = 1  # forbidden\n")
    assert "forbidden" not in _code_only('def f():\n    """forbidden"""\n')
    assert "forbidden" in _code_only("forbidden = 1\n")


def test_both_legs_copy_into_a_preallocated_destination():
    """The fix, asserted against the source: neither leg may allocate per pass.

    Checked textually because running it needs a device. The two shapes
    that reintroduce the defect are `.cpu()` and rebinding a name to a
    freshly produced tensor inside the loop, so both are named here.
    """
    import inspect

    # Tokenise the whole function, then slice the result: slicing the raw
    # source first leaves a fragment tokenize cannot indent-parse.
    code = _code_only(inspect.getsource(pcie_bandwidth.run))
    cut = code.index("while time . perf_counter")
    setup, loop = code[:cut], code[cut:]

    assert ". cpu ( )" not in loop, "d2h must copy_ into a preallocated buffer"
    assert "resident = host . to ( device )" not in loop, (
        "h2d must copy_ into the resident buffer, not rebind a new one"
    )
    assert loop.count("copy_") == 2, "one copy_ per direction"

    # Both destinations exist before the timed region begins.
    assert "landing = torch . empty" in setup
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
    code = _code_only(before_the_fix)
    loop = code[code.index("while time . perf_counter"):]

    assert ". cpu ( )" in loop
    assert "resident = host . to ( device )" in loop
    assert loop.count("copy_") != 2


def test_the_result_records_how_the_number_was_produced():
    """A row without this key predates the fix and is not comparable."""
    import inspect

    assert '"buffers": "preallocated"' in inspect.getsource(pcie_bandwidth.run)


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
