"""Workload resolution and capability gating."""

import pytest

from kernels import registry
from neuron_device import NeuronDevice


TRN1 = [NeuronDevice(i, "trn1", "v2", 2, 32 * 1024**3, True) for i in range(2)]
INF2 = [NeuronDevice(i, "inf2", "v2", 2, 32 * 1024**3, False) for i in range(2)]
SINGLE = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]


def test_resolve_all():
    assert len(registry.resolve("all")) == len(registry.WORKLOADS)


def test_resolve_by_suite():
    names = {w.name for w in registry.resolve("core")}
    assert names == {"gemm_stress", "multicore_saturation"}


def test_resolve_by_name():
    assert [w.name for w in registry.resolve("hbm_bandwidth")] == ["hbm_bandwidth"]


def test_resolve_unknown_target_lists_options():
    with pytest.raises(KeyError, match="Known targets"):
        registry.resolve("does_not_exist")


def test_collectives_gated_off_inferentia():
    workload = next(w for w in registry.WORKLOADS if w.name == "collective_allreduce")
    assert workload.runnable_on(TRN1)
    assert not workload.runnable_on(INF2)
    assert "training" in workload.skip_reason(INF2)


def test_collectives_need_multiple_devices():
    workload = next(w for w in registry.WORKLOADS if w.name == "collective_allreduce")
    assert not workload.runnable_on(SINGLE)
    assert "2 devices" in workload.skip_reason(SINGLE)


def test_inference_workloads_run_on_both_families():
    for name in ("gemm_stress", "hbm_bandwidth", "baseline_metrics"):
        workload = next(w for w in registry.WORKLOADS if w.name == name)
        assert workload.runnable_on(TRN1), name
        assert workload.runnable_on(INF2), name


def test_every_suite_has_at_least_one_workload():
    covered = {w.suite for w in registry.WORKLOADS}
    assert covered == set(registry.SUITES)
