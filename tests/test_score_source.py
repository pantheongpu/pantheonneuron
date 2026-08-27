"""Every workload must declare where its Score comes from.

A Unit says what the number means; a ScoreSource says how it was obtained.
Without the second, a Score cannot be audited or reproduced, and a counter
rename in the Neuron SDK becomes a silently wrong number instead of a
broken reference.
"""

import json
import os

import pytest

from kernels import registry

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Counter names verified present on real hardware. Keys are the names a
# ScoreSource may reference; see data/probe-2026-08-26*/ for the captures.
VERIFIED_COUNTERS = {
    # neuron-monitor
    "neuroncore_counters.*.effective_flops",
    "execution_stats.execution_summary.completed",
    "execution_stats.period",
    # neuron-profile
    "hbm_read_bytes", "hbm_write_bytes", "total_time",
    "neuroncore_cycle_count", "throttle_active_nc0_time_ns",
    "tensor_engine_active_time_percent", "vector_engine_active_time_percent",
    "scalar_engine_active_time_percent", "gpsimd_engine_active_time_percent",
    # nccom-test
    "busbw",
}


def _scored():
    return [w for w in registry.WORKLOADS if w.unit is not None]


def test_every_scored_workload_declares_a_source():
    for workload in _scored():
        assert workload.score_source is not None, workload.name


def test_source_is_one_of_the_four_known_interfaces():
    allowed = {registry.PROFILER, registry.MONITOR, registry.NCCOM, registry.INTERNAL}
    for workload in _scored():
        assert workload.score_source.source in allowed, workload.name


def test_every_source_names_at_least_one_counter():
    for workload in _scored():
        assert workload.score_source.counters, workload.name


def test_every_source_has_a_formula():
    for workload in _scored():
        assert workload.score_source.formula.strip(), workload.name


def test_hardware_sourced_counters_were_actually_observed():
    """A ScoreSource may only cite a counter we have seen on real hardware.

    INTERNAL counters are produced by the kernel itself and have no
    hardware counterpart, so they are exempt.
    """
    for workload in _scored():
        source = workload.score_source
        if source.source == registry.INTERNAL:
            continue
        for counter in source.counters:
            assert counter in VERIFIED_COUNTERS, (
                f"{workload.name} cites {counter!r}, which is not in the set "
                f"verified on hardware"
            )


def test_internal_sources_measure_against_elapsed_time():
    """A rate needs a denominator; without one the Score is a raw count."""
    for workload in _scored():
        source = workload.score_source
        if source.source != registry.INTERNAL:
            continue
        assert any("elapsed" in c for c in source.counters), workload.name


def test_bandwidth_workloads_use_the_profiler():
    """HBM byte counters exist only in neuron-profile, not neuron-monitor."""
    for name in ("memory_read", "memory_write", "memory_read_agg", "memory_write_agg"):
        workload = next(w for w in registry.WORKLOADS if w.name == name)
        assert workload.score_source.source == registry.PROFILER, name
        assert any("hbm" in c for c in workload.score_source.counters), name


def test_flops_workloads_use_the_monitor():
    """effective_flops is a neuron-monitor field and appears nowhere else."""
    for name in ("tensor_virus", "int_virus", "transformer_virus", "omni_virus"):
        workload = next(w for w in registry.WORKLOADS if w.name == name)
        assert workload.score_source.source == registry.MONITOR, name


# -- baselines ---------------------------------------------------------------

def _baselines():
    with open(os.path.join(REPO_ROOT, "data", "baselines.json"), encoding="utf-8") as h:
        return json.load(h)


def test_baselines_file_is_valid_and_scoped():
    data = _baselines()
    assert data["schema_version"] == 1
    assert data["hardware"]["arch"] == "inf2"
    assert data["measurements"]


def test_baselines_carry_the_untuned_warning():
    """effective_flops here is 4 orders of magnitude below capability. Anyone
    reading this file must not mistake it for a benchmark result."""
    warning = _baselines()["_warning"].lower()
    assert "not tuned" in warning or "not a" in warning or "untuned" in warning
    assert "mfu" in warning


def test_every_measurement_records_its_source_and_probe():
    for name, entry in _baselines()["measurements"].items():
        assert "source" in entry, name
        assert "probe" in entry, name


def test_power_units_are_marked_undocumented():
    """Recording a number without its unit invites it being read as watts."""
    for key in ("power_utilization_idle", "power_utilization_load"):
        assert _baselines()["measurements"][key]["unit"] == "UNDOCUMENTED"


def test_collectives_baseline_is_not_mislabelled_as_neuronlink():
    scope = _baselines()["measurements"]["collectives_busbw"]["scope"].lower()
    assert "not neuronlink" in scope


def test_unmeasured_section_names_the_known_gaps():
    gaps = _baselines()["unmeasured"]
    for key in ("temperature", "neuronlink_device_to_device",
                "trainium_multi_device"):
        assert key in gaps


def test_trainium_is_no_longer_an_unmeasured_gap():
    """Measured on trn1.2xlarge 2026-08-27; it must not still be listed
    as unknown."""
    assert "trainium" not in _baselines()["unmeasured"]
    assert _baselines()["trainium"]["hardware"]["arch"] == "trn1"


def test_trainium_records_the_async_timing_bug():
    """The unsynchronised figure is kept deliberately. Losing it would make
    the 264 GB/s look arbitrary rather than hard-won."""
    measured = _baselines()["trainium"]["measurements"]
    assert "memory_read_gbps_unsynced_WRONG" in measured
    wrong = measured["memory_read_gbps_unsynced_WRONG"]["value"]
    right = measured["memory_read_gbps_synced"]["value"]
    assert wrong > right * 5, "the bug inflated bandwidth by roughly 6x"


def test_trainium_counter_set_differs_from_inferentia():
    """90 counters on trn1 against 108 on inf2 -- a kernel must not assume
    a counter exists because the other chip had it."""
    trn = _baselines()["trainium"]
    assert trn["measurements"]["profiler_counter_count"]["value"] == 90
    assert "throttle_active_nc0_time_ns" in trn["absent_on_trainium"]
