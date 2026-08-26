"""Workload resolution, capability gating, and name parity with pantheongpu."""

import pytest

from kernels import registry
from neuron_device import NeuronDevice


TRN1 = [NeuronDevice(i, "trn1", "v2", 2, 32 * 1024**3, True) for i in range(2)]
INF2 = [NeuronDevice(i, "inf2", "v2", 2, 32 * 1024**3, False) for i in range(2)]
SINGLE = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]


def _get(name):
    return next(w for w in registry.WORKLOADS if w.name == name)


def test_resolve_all():
    assert len(registry.resolve("all")) == len(registry.WORKLOADS)


def test_resolve_by_suite():
    names = {w.name for w in registry.resolve("inference")}
    assert "llm_decode" in names and "moe_router" in names


def test_resolve_by_name():
    assert [w.name for w in registry.resolve("all_reduce")] == ["all_reduce"]


def test_resolve_unknown_target_lists_options():
    with pytest.raises(KeyError, match="Known targets"):
        registry.resolve("does_not_exist")


def test_gpu_only_workload_explains_itself():
    """Asking for a GPU workload by name should say why it is absent."""
    with pytest.raises(KeyError, match="no FP64"):
        registry.resolve("fp64_virus")
    with pytest.raises(KeyError, match="ray-tracing"):
        registry.resolve("rt_virus")


# -- parity with pantheongpu ------------------------------------------------

def test_names_do_not_collide_with_gpu_only_list():
    """A name is either implemented here or declared N/A -- never both."""
    implemented = {w.name for w in registry.WORKLOADS}
    assert not (implemented & set(registry.NO_NEURON_EQUIVALENT))


def test_suites_match_pantheongpu_naming():
    """--test inference must mean the same thing on both platforms."""
    for suite in ("baseline", "core", "memory", "interconnect", "inference"):
        assert suite in registry.SUITES


def test_every_suite_has_at_least_one_workload():
    covered = {w.suite for w in registry.WORKLOADS}
    assert covered == set(registry.SUITES)


def test_shared_names_are_the_comparable_ones():
    """Spot-check that portable GPU concepts kept their GPU names."""
    for name in (
        "baseline_metrics",
        "all_reduce",
        "memory_read",
        "memory_write",
        "llm_decode",
        "llm_prefill",
        "kv_cache_churn",
        "transformer_train_step",
        "graph_replay",
    ):
        assert _get(name), name


# -- capability gating -------------------------------------------------------

def test_training_workload_gated_off_inferentia():
    workload = _get("transformer_train_step")
    assert workload.runnable_on(TRN1)
    assert not workload.runnable_on(INF2)
    assert "training" in workload.skip_reason(INF2)


def test_collectives_run_on_inferentia_too():
    """inf2 shards inference across devices over NeuronLink; do not gate
    collectives behind training."""
    for name in ("all_reduce", "p2p_thrasher"):
        workload = _get(name)
        assert workload.runnable_on(INF2), name
        assert workload.runnable_on(TRN1), name


def test_collectives_need_multiple_devices():
    workload = _get("all_reduce")
    assert not workload.runnable_on(SINGLE)
    assert "2 devices" in workload.skip_reason(SINGLE)


def test_inference_suite_runs_on_both_families():
    for workload in registry.resolve("inference"):
        assert workload.runnable_on(TRN1), workload.name
        assert workload.runnable_on(INF2), workload.name


def test_aggregate_memory_workloads_need_multicore():
    single_core = [NeuronDevice(0, "inf2", "v2", 1, 1, False)]
    assert not _get("memory_read_agg").runnable_on(single_core)
    assert _get("memory_read").runnable_on(single_core)
