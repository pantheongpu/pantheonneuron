"""A kernel's STATUS line is a claim, and claims get checked here.

This suite's rule is that nothing is said to work without a hardware run
behind it. The inverse failed quietly for a fortnight: on 2026-09-10,
eleven kernel modules still carried ``STATUS: UNTESTED ON HARDWARE`` in
their docstrings, and every workload in all eleven had passed on a
trn1.2xlarge -- most of them on 2026-09-08, some twice since.

A stale UNTESTED label is not a harmless conservatism. It makes verified
work look unverified, and it makes the eleven indistinguishable from the
one module where the label is still true (``collectives``, whose two
workloads need a part this account's quota cannot reach). A status that
cannot be wrong is not a status.

So ``data/hardware_runs.json`` records which workloads ran where, with
the log that proves it, and these tests hold the docstrings against it.
"""

import json
import os

import pytest

from kernels import registry
from neuron_device import NeuronDevice

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(ROOT, "data", "hardware_runs.json")
KERNEL_DIR = os.path.join(ROOT, "kernels")

UNTESTED = "STATUS: UNTESTED ON HARDWARE"


def _record():
    with open(RECORD, encoding="utf-8") as handle:
        return json.load(handle)


def _verified_workloads():
    """Every workload that has passed on some part."""
    passed = set()
    for part in _record()["parts"].values():
        for run in part["runs"]:
            passed.update(run["passed"])
    return passed


def _module_source(name):
    with open(os.path.join(KERNEL_DIR, name), encoding="utf-8") as handle:
        return handle.read()


def _kernel_modules():
    return sorted(name for name in os.listdir(KERNEL_DIR)
                  if name.endswith(".py") and name != "__init__.py")


# Which module implements which workloads. Only modules that own a
# workload can have a hardware status at all -- tiling and nki_backend
# are support code and are deliberately absent.
OWNERS = {
    # baseline_metrics is implemented in the orchestrator itself, not in a
    # kernel module, so it has no docstring to go stale. Listed so the
    # completeness check below stays meaningful.
    None: ["baseline_metrics"],
    "allocation_fragmentation.py": ["allocation_fragmentation"],
    "collectives.py": ["all_reduce", "p2p_thrasher"],
    "encoders.py": ["rag_embedding", "vision_encoder"],
    "graph_replay.py": ["graph_replay"],
    "inference_mix.py": ["fused_attention", "moe_router", "quantized_gemm",
                         "serving_mix", "speculative_decode"],
    "llm_inference.py": ["llm_prefill", "llm_decode", "kv_cache_churn"],
    "memory_agg.py": ["memory_read_agg", "memory_write_agg"],
    "memory_read.py": ["memory_read"],
    "memory_write.py": ["memory_write"],
    "omni_virus.py": ["omni_virus"],
    "pcie_bandwidth.py": ["pcie_bandwidth"],
    "pulse_virus.py": ["pulse_virus"],
    "tensor_virus.py": ["tensor_virus", "int_virus"],
    "transformer_compute.py": ["transformer_virus", "transformer_train_step"],
}


def test_the_record_only_names_workloads_that_exist():
    """A record naming a deleted workload is a record nobody maintained."""
    known = {workload.name for workload in registry.WORKLOADS}
    record = _record()
    for part_name, part in record["parts"].items():
        for run in part["runs"]:
            for key in ("passed", "failed", "skipped"):
                unknown = set(run[key]) - known
                assert not unknown, f"{part_name} {run['date']} {key}: {unknown}"
    assert not set(record["never_run_anywhere"]) - known


def test_a_run_does_not_both_pass_and_fail_a_workload():
    for part_name, part in _record()["parts"].items():
        for run in part["runs"]:
            buckets = [set(run[k]) for k in ("passed", "failed", "skipped")]
            for i, first in enumerate(buckets):
                for second in buckets[i + 1:]:
                    assert not first & second, (
                        f"{part_name} {run['date']}: {first & second}")


def test_every_run_cites_its_evidence():
    """A date without a log is a memory, not a record."""
    for part in _record()["parts"].values():
        for run in part["runs"]:
            assert run["evidence"], run
            path = run["evidence"].split("#")[0]
            assert os.path.exists(os.path.join(ROOT, path)), path


@pytest.mark.parametrize("module", sorted(k for k in OWNERS if k))
def test_no_module_claims_untested_for_a_workload_that_has_run(module):
    """The defect this file exists for.

    Eleven modules said UNTESTED while their workloads had passed. The
    label is only honest for a module where *none* of its workloads has
    ever run -- which today is `collectives` alone.
    """
    source = _module_source(module)
    if UNTESTED not in source:
        return
    verified = _verified_workloads() & set(OWNERS[module])
    assert not verified, (
        f"{module} says '{UNTESTED}' but {sorted(verified)} have passed on "
        "hardware; see data/hardware_runs.json"
    )


def test_collectives_is_still_honestly_untested():
    """The control. If this ever fails, the check above proved nothing.

    A test that only ever asserts absence passes just as well when the
    record is empty, when the parser is broken, or when OWNERS is wrong.
    This is the case that must keep saying UNTESTED, so a green run means
    the machinery works rather than that it found nothing.
    """
    assert UNTESTED in _module_source("collectives.py")
    assert not _verified_workloads() & set(OWNERS["collectives.py"])


def test_every_workload_has_an_owning_module():
    """So a new workload cannot slip past the status check unnoticed."""
    owned = {name for names in OWNERS.values() for name in names}
    missing = {workload.name for workload in registry.WORKLOADS} - owned
    assert not missing, missing


def test_the_workloads_that_never_ran_are_the_ones_that_cannot():
    """And each says why, because "untested" and "unreachable" differ."""
    record = _record()
    never = set(record["never_run_anywhere"])
    # Derived from the capability gate rather than typed out, so a
    # workload that gains or loses the multi-device requirement cannot
    # leave this list describing the old set.
    one_device = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]
    gated = {workload.name for workload in registry.WORKLOADS
             if not workload.runnable_on(one_device)}
    assert never == gated, (never, gated)
    for name, reason in record["never_run_anywhere"].items():
        assert len(reason) > 40, name


def test_no_kernel_module_at_all_still_claims_untested():
    """The wider net, because the narrower one had a hole.

    ``test_no_module_claims_untested_for_a_workload_that_has_run`` keys
    on workload ownership, so support code owning no workload is invisible
    to it. ``transformer_ops`` said UNTESTED for a fortnight after every
    kernel importing it had passed, and that check ran green throughout --
    it was found by reading the list the check had filtered out.

    So this one asks the blunt question of every module in the package,
    and names the only module allowed to answer yes.
    """
    still_claiming = [name for name in _kernel_modules()
                      if UNTESTED in _module_source(name)]
    assert still_claiming == ["collectives.py"], still_claiming


def test_the_support_modules_are_the_ones_owning_no_workload():
    """Names the hole, so it cannot be reopened silently."""
    owned = {name for name in OWNERS if name}
    support = set(_kernel_modules()) - owned - {"registry.py"}
    # Every one of these is invisible to the ownership-keyed check above.
    assert "transformer_ops.py" in support
    assert "tiling.py" in support
