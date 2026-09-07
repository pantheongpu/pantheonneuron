"""Score/Unit parity with the pantheongpu report schema.

A cross-platform comparison joins Neuron results to GPU results on
(Test Name, Unit). A typo in a unit string does not raise anything -- it
just produces a row that never joins, and a silently missing comparison is
worse than a loud failure.

PANTHEONGPU_UNITS below is transcribed from the pantheongpu report database.
It was meant to fail if pantheongpu changed a unit string, and it did not:
v1.0.19 replaced the units of twelve AI workloads with a single ``ai-ops/s``
and this test kept passing, because it compared one stale transcription
against another. Both sides drifted together.

So the transcription now records what pantheongpu actually reports, and the
workloads whose units diverged are checked against
``registry.NOT_COMPARABLE_WITH_GPU`` instead of being forced to match. Those
Neuron workloads count real tokens and steps; the GPU ones report generic
synthetic throughput under a shared kernel. Making the strings match would
restore the join and compare unlike things.
"""

import json
import os

import pytest

import pantheon_neuron
from kernels import registry
from neuron_device import NeuronDevice


# Unit strings as they appear in pantheongpu reports, keyed by workload.
PANTHEONGPU_UNITS = {
    "tensor_virus": "TFLOPS",
    "int_virus": "TOPS",
    "pulse_virus": "TFLOPS",
    "transformer_virus": "TFLOPS",
    "omni_virus": "TFLOPS",
    "memory_read": "GB/s",
    "memory_write": "GB/s",
    "memory_read_agg": "GB/s",
    "memory_write_agg": "GB/s",
    "all_reduce": "GB/s",
    "p2p_thrasher": "GB/s",
    "pcie_bandwidth": "GB/s",
    "allocation_fragmentation": "allocation-events/s",
    # Since v1.0.19 every AI workload reports the same synthetic unit.
    "llm_decode": "ai-ops/s",
    "llm_prefill": "ai-ops/s",
    "kv_cache_churn": "ai-ops/s",
    "fused_attention": "ai-ops/s",
    "quantized_gemm": "ai-ops/s",
    "serving_mix": "ai-ops/s",
    "speculative_decode": "ai-ops/s",
    "moe_router": "ai-ops/s",
    "transformer_train_step": "ai-ops/s",
    "graph_replay": "ai-ops/s",
    "rag_embedding": "ai-ops/s",
    "vision_encoder": "ai-ops/s",
}

TRN1 = [NeuronDevice(i, "trn1", "v2", 2, 32 * 1024**3, True) for i in range(2)]


def _get(name):
    return next(w for w in registry.WORKLOADS if w.name == name)


@pytest.mark.parametrize("name,unit", sorted(PANTHEONGPU_UNITS.items()))
def test_unit_matches_pantheongpu(name, unit):
    if name in registry.NOT_COMPARABLE_WITH_GPU:
        pytest.skip(f"{name} is declared not comparable with the GPU workload")
    assert _get(name).unit == unit, (
        f"{name}: unit must match pantheongpu exactly for the comparison "
        f"join to work"
    )


def test_diverged_units_are_declared_not_comparable():
    """A unit that no longer matches must be declared, not left to drift.

    Left alone, a Neuron row simply never joins, and a comparison that is
    silently absent looks the same as one that found nothing to say.
    """
    diverged = {
        name for name, unit in PANTHEONGPU_UNITS.items()
        if _get(name).unit != unit
    }
    declared = set(registry.NOT_COMPARABLE_WITH_GPU)

    assert diverged == declared, (
        f"undeclared divergence: {sorted(diverged - declared)}; "
        f"declared but matching: {sorted(declared - diverged)}"
    )


def test_not_comparable_records_the_neuron_unit():
    """The table records what Neuron reports, so a reader sees both sides."""
    for name, unit in registry.NOT_COMPARABLE_WITH_GPU.items():
        assert _get(name).unit == unit
        assert PANTHEONGPU_UNITS[name] == registry.GPU_SYNTHETIC_AI_UNIT
        # These count real quantities; that is the whole reason they diverge.
        assert unit != registry.GPU_SYNTHETIC_AI_UNIT


def test_comparable_workloads_still_join():
    """The divergence must not have quietly swallowed everything."""
    comparable = {
        name for name in PANTHEONGPU_UNITS
        if name not in registry.NOT_COMPARABLE_WITH_GPU
    }
    assert len(comparable) >= 12, "cross-platform comparison has no rows left"
    for name in comparable:
        assert _get(name).unit == PANTHEONGPU_UNITS[name]


def test_every_scored_workload_has_a_unit():
    for workload in registry.WORKLOADS:
        if workload.name == "baseline_metrics":
            continue  # applies no load; nothing to score
        assert workload.unit, f"{workload.name} has no unit"


def test_every_scored_workload_pins_its_problem():
    """A Score without a pinned problem is not comparable to anything."""
    for workload in registry.WORKLOADS:
        if workload.unit is None:
            continue
        assert workload.problem, f"{workload.name} has a unit but no problem"


def test_compute_workloads_pin_a_dtype():
    """TFLOPS at bf16 and TFLOPS at fp32 are different numbers."""
    for workload in registry.WORKLOADS:
        if workload.unit in ("TFLOPS", "TOPS"):
            assert "dtype" in workload.problem, workload.name


def test_int_workloads_use_an_integer_dtype():
    for name in ("int_virus", "quantized_gemm"):
        assert _get(name).problem["dtype"].startswith("int")


# -- report row --------------------------------------------------------------

@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    yield
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_row_carries_score_unit_and_problem(mock_env):
    row = pantheon_neuron.run_workload(
        _get("memory_read"), TRN1, duration=1, monitor_period=0.01
    )
    assert row["Unit"] == "GB/s"
    assert row["Problem"]["dtype"] == "bf16"
    assert "Score" in row


def test_mock_mode_never_fabricates_a_score(mock_env):
    """A synthetic Score would flow into a report and be compared against
    real GPU numbers."""
    for workload in registry.WORKLOADS:
        if not workload.runnable_on(TRN1):
            continue
        row = pantheon_neuron.run_workload(
            workload, TRN1, duration=1, monitor_period=0.01
        )
        assert row["Score"] is None, f"{workload.name} invented a Score in mock mode"


def test_skipped_row_still_declares_its_unit(mock_env):
    """A skipped row must stay joinable, so a comparison can show a gap
    rather than dropping the row."""
    inf2 = [NeuronDevice(i, "inf2", "v2", 2, 32 * 1024**3, False) for i in range(2)]
    row = pantheon_neuron.run_workload(
        _get("transformer_train_step"), inf2, duration=1, monitor_period=0.01
    )
    assert row["Status"] == "SKIPPED"
    assert row["Unit"] == "train-steps/s"


def test_report_round_trips_score_fields(mock_env, tmp_path, monkeypatch):
    monkeypatch.setattr(pantheon_neuron, "DATABASE_DIR", str(tmp_path))
    snapshot = pantheon_neuron.get_system_snapshot(TRN1)
    row = pantheon_neuron.run_workload(
        _get("llm_decode"), TRN1, duration=1, monitor_period=0.01
    )
    path = pantheon_neuron.write_report(snapshot, [row], "runid")
    with open(path, encoding="utf-8") as handle:
        written = json.load(handle)["test_results"][0]
    assert written["Unit"] == "tokens/s"
    assert written["Problem"]["hidden"] == 4096
