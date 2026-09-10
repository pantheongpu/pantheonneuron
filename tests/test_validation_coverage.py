"""Every runnable workload must be in the hardware validation pass.

`tools/validate_hardware.sh` is the only thing that ever runs this suite on a
device, and until 2026-09-08 its list held four names while the registry held
26. Eighteen workloads had therefore never executed on hardware at all --
which is how `omni_virus` shipped a call to a `_read_back` it did not have,
a NameError that only its first real run could have found.

A workload absent from that list is a workload nobody is checking. So the
list is compared against the registry here, and adding a workload without
adding it to the pass fails the build rather than quietly going unrun.
"""

import os
import re

import pytest

from kernels import registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tools", "validate_hardware.sh")


def _script() -> str:
    with open(SCRIPT, encoding="utf-8") as handle:
        return handle.read()


def orchestrated_names() -> set:
    """The workloads the validation pass actually runs."""
    text = _script()
    match = re.search(r'ORCHESTRATED=\$\{ORCHESTRATED:-"(.*?)"\}', text, re.S)
    assert match, "could not find the ORCHESTRATED list in validate_hardware.sh"
    return set(match.group(1).split())


# Workloads that cannot run on any instance this project can currently get,
# with the reason. Kept as data so a name here is a decision rather than an
# omission -- and so the day the quota lands, the test tells us to add them.
UNREACHABLE = {
    "all_reduce": "needs 2+ devices; trn1.32xlarge is 128 vCPU against a granted 64",
    "p2p_thrasher": "needs 2+ devices; same quota",
    "baseline_metrics": "idle telemetry, no load -- runs as part of any --test all",
}


def test_every_reachable_workload_is_validated():
    listed = orchestrated_names()
    expected = {w.name for w in registry.WORKLOADS} - set(UNREACHABLE)
    missing = expected - listed
    assert not missing, (
        f"not in validate_hardware.sh, so never run on hardware: {sorted(missing)}"
    )


def test_the_list_names_only_real_workloads():
    """A typo here silently drops a workload from the pass."""
    known = {w.name for w in registry.WORKLOADS}
    unknown = orchestrated_names() - known
    assert not unknown, f"not workloads in the registry: {sorted(unknown)}"


def test_unreachable_workloads_are_excluded_for_a_stated_reason():
    listed = orchestrated_names()
    for name, reason in UNREACHABLE.items():
        assert name not in listed, f"{name} cannot run: {reason}"


@pytest.mark.parametrize("name", sorted(UNREACHABLE))
def test_unreachable_names_are_still_real_workloads(name):
    """The exclusion list must not outlive the workloads it excuses."""
    assert name in {w.name for w in registry.WORKLOADS}


def test_the_collectives_are_excluded_only_while_the_quota_blocks_them():
    """Both need min_devices > 1. If that ever changes, so must this list."""
    by_name = {w.name for w in registry.WORKLOADS}
    for name in ("all_reduce", "p2p_thrasher"):
        workload = next(w for w in registry.WORKLOADS if w.name == name)
        assert workload.min_devices > 1, (
            f"{name} no longer needs multiple devices -- it should be validated"
        )
    assert "all_reduce" in by_name


def test_failures_do_not_end_the_run():
    """A workload that fails is a result; it must not cost the ones after it.

    The loop checks PIPESTATUS and reports, rather than running under `set
    -e` semantics that would abandon the remaining workloads.
    """
    text = _script()
    assert "PIPESTATUS" in text
    assert "TIMED OUT" in text
    # set -e would end the pass on the first non-zero exit.
    assert "set -uo pipefail" in text and "set -euo" not in text


def test_the_pass_exercises_the_neff_search_against_a_warm_cache():
    """The one thing the 2026-09-08 run could not test.

    It scored from candidate 1 of 1, because a fresh instance holds exactly
    one NEFF. Running a profiler-sourced workload again after everything
    else has compiled is what puts more than one graph in the ranking.
    """
    text = _script()
    assert "NEFF search against a warm compile cache" in text
    search = text[text.index("NEFF search against a warm compile cache"):]
    assert "--test memory_read" in search, (
        "the warm-cache pass must run a profiler-sourced workload"
    )


def test_the_pass_can_repeat_every_workload():
    """One run is a sample, not a measurement.

    memory_read's declared Score read 256.17, 178.7 and 119.19 GB/s on
    three runs of the same problem before its cause was found, and a
    single-pass harness could never have shown that. The default stays 1 --
    repeats multiply an already hour-long pass -- but a pass whose numbers
    will be quoted should set REPEAT.
    """
    text = _script()
    assert "REPEAT=${REPEAT:-1}" in text
    assert '--repeat "$REPEAT"' in text
    # Every orchestrated invocation carries it, not just the first.
    assert text.count('--repeat "$REPEAT"') >= 2


def test_the_summary_prints_the_spread():
    """A repeated run whose spread is invisible is a repeated run wasted."""
    text = _script()
    assert "repeats:" in text
    assert "cv" in text
