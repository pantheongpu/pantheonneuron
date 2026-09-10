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
import sourcecheck
from kernels import registry, tiling
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
    """Asked of the dtype table, not of the name.

    `int_virus` pins uint8, because trn1's Tensor Engine rejects signed
    int8 outright. A prefix check on the string said uint8 was not an
    integer type, which is wrong and would have blocked the only dtype the
    part actually runs -- so the question goes to `tiling.is_integer`,
    which is the same predicate the kernel branches on for the accumulator
    and the unit.
    """
    for name in ("int_virus", "quantized_gemm"):
        dtype = _get(name).problem["dtype"]
        assert tiling.is_integer(dtype), f"{name} pins {dtype}"


def test_an_integer_dtype_is_reported_as_operations_not_flops():
    """TOPS and TFLOPS are different quantities; the dtype decides which."""
    for workload in registry.WORKLOADS:
        problem = workload.problem or {}
        if workload.unit in ("TFLOPS", "TOPS") and "dtype" in problem:
            integer = tiling.is_integer(problem["dtype"])
            assert (workload.unit == "TOPS") == integer, workload.name


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


# -- same unit, different quantity -------------------------------------------

def test_same_unit_different_quantity_is_disjoint_from_unit_divergence():
    """Two different failure modes, two registers, no overlap.

    NOT_COMPARABLE_WITH_GPU means the units diverge, so the join fails and
    the absence is visible. This register means the opposite and worse
    case: the unit matches, the join succeeds, and the quantities differ.
    """
    overlap = (set(registry.SAME_UNIT_DIFFERENT_QUANTITY)
               & set(registry.NOT_COMPARABLE_WITH_GPU))
    assert not overlap, overlap


def test_same_unit_different_quantity_really_does_share_the_unit():
    """If a unit ever diverges, the row belongs in the other register."""
    for name in registry.SAME_UNIT_DIFFERENT_QUANTITY:
        assert _get(name).unit == PANTHEONGPU_UNITS[name], (
            f"{name}'s unit now diverges -- it belongs in "
            "NOT_COMPARABLE_WITH_GPU instead"
        )


def test_every_flagged_name_is_a_real_workload_with_a_reason():
    for name, reason in registry.SAME_UNIT_DIFFERENT_QUANTITY.items():
        assert name in {w.name for w in registry.WORKLOADS}
        assert len(reason) > 40, f"{name}: the reason must say what differs"


def test_the_register_does_not_change_what_joins():
    """It records a finding; it must not silently drop rows.

    The comparison still has the same rows it had before, because deciding
    what to do about this is not a decision a commit should make.
    """
    comparable = {name for name in PANTHEONGPU_UNITS
                  if name not in registry.NOT_COMPARABLE_WITH_GPU}
    assert set(registry.SAME_UNIT_DIFFERENT_QUANTITY) <= comparable
    assert len(comparable) >= 12


# -- a pinned dtype is a claim that the engine will run it --------------------
#
# The first version of this asserted against a single table transcribed from
# a prose comment, and a probe falsified it the same afternoon: it had
# fp8_e4m3 as an accepted operand, and neuronx-cc refuses it outright. The
# tables now record what was measured, split by the path that measured it,
# because NKI and XLA do not accept the same set -- int8 runs through XLA
# and nc_matmul rejects it.

# Every workload with a pinned dtype except int_virus reaches the engine
# through XLA. int_virus is the NKI kernel, and it is why the split exists.
_NKI_WORKLOADS = frozenset({"tensor_virus", "int_virus", "pulse_virus",
                            "omni_virus", "transformer_virus",
                            "memory_read", "memory_write",
                            "memory_read_agg", "memory_write_agg"})


def _path(name):
    return "nki" if name in _NKI_WORKLOADS else "xla"


def test_no_pinned_problem_names_a_dtype_its_path_refuses():
    offenders = []
    for workload in registry.WORKLOADS:
        dtype = (workload.problem or {}).get("dtype")
        if dtype is None:
            continue
        path = _path(workload.name)
        if not tiling.engine_accepts(dtype, path):
            why = tiling.refusal(dtype, path) or "not an operand on this path"
            offenders.append(f"{workload.name} pins {dtype} on {path}: {why}")
    assert not offenders, "; ".join(offenders)


def test_the_two_paths_do_not_accept_the_same_set():
    """Collapsing them is what made the first version of this wrong."""
    assert tiling.NKI_OPERANDS != tiling.XLA_OPERANDS
    assert "int8" in tiling.XLA_OPERANDS
    assert "int8" not in tiling.NKI_OPERANDS
    assert tiling.engine_accepts("int8", "xla")
    assert not tiling.engine_accepts("int8", "nki")


def test_fp8_is_not_claimed_as_an_operand_on_either_path():
    """It was, on the strength of a comment. neuronx-cc says otherwise."""
    assert "fp8_e4m3" not in tiling.XLA_OPERANDS
    assert "fp8_e4m3" not in tiling.NKI_OPERANDS
    assert "NCC_ESPP047" in tiling.refusal("fp8_e4m3", "xla")


def test_every_refusal_is_keyed_by_the_path_that_refused():
    for key, why in tiling.OPERAND_REFUSALS.items():
        path, dtype = key
        assert path in ("nki", "xla"), key
        assert not tiling.engine_accepts(dtype, path), key
        # A refusal without a date is a claim, not a measurement.
        assert "2026-" in why, key


def test_eight_bit_is_not_a_throughput_win_on_this_part():
    """The assumption a workload named "quantized" invites, measured.

    int8 through XLA is the slowest path on the part -- 18.46 T-ops/s
    against bf16's 70.38 at the same 4096^3 shape in the same process --
    and uint8 only matches bf16. Anyone reading quantized_gemm's Score as
    an acceleration figure is reading it backwards, so the relationship
    is pinned here rather than left in a comment.
    """
    rates = tiling.OPERAND_RATES_4096
    assert rates["int8"] < rates["bf16"], rates
    assert rates["int8"] / rates["bf16"] < 0.3, rates
    assert 0.95 < rates["uint8"] / rates["bf16"] < 1.1, rates


# -- a declared counter its source cannot supply -----------------------------

def _reader_source():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = ""
    for path in ("neuron_monitor.py", "kernels/profiler.py",
                 "kernels/collectives.py"):
        with open(os.path.join(root, path), encoding="utf-8") as handle:
            text += sourcecheck.code_only(handle.read())
    return text


def test_every_declared_counter_is_read_by_something_or_declared_absent():
    """The counters tuple is the published answer to "where does this
    number come from", and five of them named a stream that does not
    carry them.

    Read through the comment filter, so a counter mentioned in a
    docstring about counters does not vouch for itself.
    """
    readers = _reader_source()
    assert readers, "no reader source read -- the sweep is broken"

    unsupplied = []
    for workload in registry.WORKLOADS:
        source = workload.score_source
        if not source or source.source == registry.INTERNAL:
            continue
        known_absent = registry.COUNTERS_THE_DECLARED_SOURCE_CANNOT_SUPPLY.get(
            workload.name, ())
        for counter in source.counters:
            leaf = counter.rsplit(".", 1)[-1]
            if leaf in known_absent:
                continue
            if f'"{leaf}"' not in readers and f"'{leaf}'" not in readers:
                unsupplied.append(f"{workload.name}: {counter}")

    assert not unsupplied, (
        f"declared but unread: {unsupplied} -- either the reader should "
        "parse it, or it belongs in "
        "COUNTERS_THE_DECLARED_SOURCE_CANNOT_SUPPLY with the evidence")


def test_the_absent_list_only_names_counters_that_are_declared():
    """A stale entry there would silently excuse a counter nobody asks for."""
    declared = set()
    for workload in registry.WORKLOADS:
        if workload.score_source:
            declared.update(c.rsplit(".", 1)[-1]
                            for c in workload.score_source.counters)
    for name, counters in (
            registry.COUNTERS_THE_DECLARED_SOURCE_CANNOT_SUPPLY.items()):
        workload = next((w for w in registry.WORKLOADS if w.name == name), None)
        assert workload is not None, name
        for counter in counters:
            assert counter in declared, (name, counter)


def test_the_absent_counters_really_are_absent():
    """The control. Without it the list could excuse anything, including
    counters the readers do parse -- which is how a gap becomes a habit.
    """
    readers = _reader_source()
    for name, counters in (
            registry.COUNTERS_THE_DECLARED_SOURCE_CANNOT_SUPPLY.items()):
        for counter in counters:
            assert f'"{counter}"' not in readers, (name, counter)
