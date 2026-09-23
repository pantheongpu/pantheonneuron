"""The analytic bandwidth is only a measurement if the kernel actually read.

Observed on trn1.2xlarge 2026-08-27: run() discarded the kernel result, so
nothing referenced the graph at the mark_step() cut and XLA skipped the DMA.
The run reported 14,513 GB/s -- about 17x the part's ~820 GB/s HBM -- while
neuron-monitor recorded total_executions=1 across a 90 second window.

verify_against_analytic only fires when the profiler works, and the profiler
failing is exactly when the analytic number becomes the Score. These cover
the profiler-independent check that closes that hole.
"""

from kernels.memory_read import tile_plan, verify_read_completed


def test_full_read_passes():
    assert verify_read_completed(1.0) is None


def test_slightly_off_is_tolerated():
    assert verify_read_completed(1.005) is None


def test_eliminated_loads_are_rejected():
    msg = verify_read_completed(0.0)
    assert msg is not None and "eliminated" in msg


def test_partial_read_is_rejected():
    msg = verify_read_completed(0.5)
    assert msg is not None and "0.500x" in msg


def test_over_read_is_rejected():
    # More than planned means the accounting is wrong, not that we got a bonus.
    assert verify_read_completed(2.0) is not None


def test_unreadable_output_is_not_silently_accepted():
    msg = verify_read_completed(None)
    assert msg is not None and "unverified" in msg


def test_expected_accumulator_value_matches_plan():
    """One pass over an all-ones buffer sums to tiles * FREE per partition."""
    plan = tile_plan(64 * 1024 * 1024, "bf16")
    expected = plan["tiles"] * plan["free"]
    # the ratio the kernel computes is observed/expected, so a correct run
    # yields exactly 1.0
    assert verify_read_completed(float(expected) / expected) is None


# -- the verdict has to reach the row ---------------------------------------
#
# The checks above return a sentence. Returning it was all that happened: the
# kernel put it in `warning` and returned the analytic bandwidth as the
# Score, so a row whose kernel had just written "the analytic bandwidth is
# not a measurement" was published as PASS with that bandwidth in it.

import pytest  # noqa: E402

import pantheon_neuron  # noqa: E402
import sourcecheck  # noqa: E402
from kernels import memory_read, memory_write, registry  # noqa: E402
from neuron_device import NeuronDevice  # noqa: E402


@pytest.mark.parametrize("module", [memory_read, memory_write])
def test_a_failed_coverage_check_disowns_the_score(module):
    """The branch that sets the warning must set the flag the harness reads.

    `score_invalid` is the only thing that makes the harness drop a Score,
    and this branch did not set it -- the one branch whose own message says
    the number beside it is not a measurement.
    """
    # `run` is the compile-cache wrapper; the loop and the check are in
    # `_run`.
    code = sourcecheck.flat_function_code(module._run)
    setter = 'result [ "score_invalid" ] = True'
    assert setter in code, "the coverage branch does not invalidate the Score"
    # In the elided branch, not somewhere later: the branch returns early,
    # so a flag set after it would never run.
    assert code.index("elided") < code.index(setter)
    assert code.index(setter) < code.index("return result")


@pytest.mark.parametrize("name, message", [
    ("memory_read", "kernel read 0.500x the planned bytes -- loads were "
                    "coalesced or eliminated, so the analytic bandwidth is "
                    "not a measurement"),
    ("memory_write", "destination holds 0.500x the written value -- writes "
                     "were coalesced or eliminated"),
])
def test_the_row_fails_when_the_kernel_disowns_its_bandwidth(
        name, message, monkeypatch):
    """End to end: the kernel's early return, as the harness sees it."""
    workload = next(w for w in registry.WORKLOADS if w.name == name)
    monkeypatch.setattr(pantheon_neuron, "_execute", lambda *a, **k: 543.7)
    monkeypatch.setitem(pantheon_neuron._LAST_RUN, name, {
        "analytic_gbps": 543.7, "elapsed_s": 10.0,
        "warning": message, "score_invalid": True,
    })
    row = pantheon_neuron._measure_once(
        workload, [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)], 1, 0.5)
    assert row["Status"] == "FAIL"
    assert row["Score"] is None
    assert "0.500x" in row["Detail"]
