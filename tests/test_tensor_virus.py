"""The tensor_virus kernel: tile maths, FLOP accounting, and its guards.

The NKI kernel needs hardware. What is testable here is the arithmetic that
decides whether a number means anything -- the GEMM tiling, the FLOP count
behind the analytic cross-check, and the correctness guard that separates a
real GEMM from one the compiler reshaped.
"""

import importlib.util
import os

import pytest

import pantheon_neuron
import sourcecheck
from kernels import registry, tensor_virus, tiling
from neuron_device import NeuronDevice


TRN1 = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]


def _workload():
    return next(w for w in registry.WORKLOADS if w.name == "tensor_virus")


# -- tile geometry -----------------------------------------------------------

def test_contraction_rides_the_hardware_partition_limit():
    """K sits on the partition axis, so pmax bounds it -- 128, read on both parts."""
    assert tensor_virus.CONTRACTION == tiling.PARTITION == 128
    assert tensor_virus.STATIONARY == 128


def test_pinned_problem_divides_into_whole_tiles():
    problem = _workload().problem
    plan = tensor_virus.gemm_plan(problem["shape"], problem["dtype"])

    assert plan["m_tiles"] == 8192 // 128
    assert plan["n_tiles"] == 8192 // 512
    assert plan["k_tiles"] == 8192 // 128


def test_partial_tiles_are_rejected_rather_than_rounded():
    """A rounded shape makes the FLOP count describe work that did not happen."""
    with pytest.raises(ValueError):
        tensor_virus.gemm_plan([8192, 8192, 100], "bf16")
    with pytest.raises(ValueError):
        tensor_virus.gemm_plan([200, 8192, 8192], "bf16")


def test_shape_must_be_three_dimensional():
    with pytest.raises(ValueError):
        tensor_virus.gemm_plan([8192, 8192], "bf16")


def test_unsupported_dtype_is_rejected():
    with pytest.raises(ValueError):
        tensor_virus.gemm_plan([8192, 8192, 8192], "fp8")


# -- FLOP accounting ---------------------------------------------------------

def test_a_gemm_is_two_flops_per_multiply_accumulate():
    """2*M*N*K: one multiply and one add per element of the contraction."""
    plan = tensor_virus.gemm_plan([128, 512, 128], "bf16")
    assert plan["flops_per_pass"] == 2 * 128 * 512 * 128


def test_pinned_problem_flop_count():
    problem = _workload().problem
    plan = tensor_virus.gemm_plan(problem["shape"], problem["dtype"])
    assert plan["flops_per_pass"] == 2 * 8192**3


# -- correctness guard -------------------------------------------------------

def test_guard_accepts_an_exact_product():
    """All-ones operands make every element exactly K."""
    assert tensor_virus.verify_product_is_correct(1.0) is None


def test_guard_flags_a_product_that_is_not_k():
    message = tensor_virus.verify_product_is_correct(0.5)
    assert message is not None
    assert "did not compute the pinned problem" in message


def test_guard_flags_an_unreadable_product():
    message = tensor_virus.verify_product_is_correct(None)
    assert message is not None
    assert "unverified" in message


# -- monitor cross-check -----------------------------------------------------
#
# Five tests lived here, all green, all exercising
# tensor_virus.verify_against_monitor -- which was never called from
# anywhere in the suite. A function written, tested, documented, and wired
# to nothing.
#
# That is the part worth pausing on: the tests passing said the function
# was correct, and it was. They said nothing whatever about whether it ran,
# and nothing else did either. Coverage of a function is not evidence that
# the function is reachable.
#
# Its job moved to pantheon_neuron.override_disagreement, which is called,
# covers every monitor-scored workload rather than this family alone, and
# carries both zero cases. The five cases moved with it, to
# tests/test_monitor_score.py, plus one asserting nothing calls the old
# name any more.


def test_nothing_calls_the_removed_cross_check():
    """So it cannot come back as a second, unreachable copy."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    callers = []
    for path in list(root.glob("*.py")) + list(root.glob("kernels/*.py")):
        code = sourcecheck.code_only(path.read_text(encoding="utf-8"))
        if "verify_against_monitor" in code:
            callers.append(path.name)
    assert not callers, callers


# -- orchestrator integration ------------------------------------------------

def test_registry_declares_the_flops_counter():
    source = _workload().score_source
    assert source.source == registry.MONITOR
    assert pantheon_neuron.FLOPS_COUNTER in source.counters
    assert _workload().unit == "TFLOPS"


def test_analytic_fallback_names_the_monitor_not_the_profiler():
    """The label has to describe this workload, not the bandwidth ones.

    tensor_virus falls back to FLOPs over wall time and its declared source
    is neuron-monitor. The old label was hardcoded to 'bytes moved' and
    'neuron-profile', which would have misdescribed both halves.
    """
    pantheon_neuron._LAST_RUN["tensor_virus"] = {
        "score_method": "analytic",
        "analytic_basis": "FLOPs issued / wall time",
    }
    method = pantheon_neuron._score_method(_workload(), 12.5)
    pantheon_neuron._LAST_RUN.pop("tensor_virus", None)

    assert "FLOPs issued / wall time" in method
    assert "neuron-monitor" in method
    assert "effective_flops" in method
    assert "bytes moved" not in method
    assert "neuron-profile" not in method


def test_monitor_score_overrides_the_analytic_figure(monkeypatch):
    """The declared source wins when the counter is there."""
    monkeypatch.setattr(pantheon_neuron, "_execute", lambda *a, **k: 3.0)
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "start",
        lambda self, indices: True,
    )
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "stop",
        lambda self: {
            "samples": 5,
            "effective_flops": {"0": {"mean": int(9e12), "peak": int(9e12)}},
        },
    )

    row = pantheon_neuron.run_workload(
        _workload(), TRN1, duration=1, monitor_period=0.1
    )
    assert row["Score"] == 9.0
    assert row["Score Method"] == registry.MONITOR


def test_mock_mode_invents_no_score(monkeypatch):
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    row = pantheon_neuron.run_workload(
        _workload(), TRN1, duration=1, monitor_period=0.1
    )
    assert row["Score"] is None


# -- int_virus rides the same kernel -----------------------------------------

class _FakeNL:
    """Stands in for neuronxcc.nki.language, which needs the toolchain."""
    int32 = "int32"
    float32 = "float32"


def _int_virus():
    return next(w for w in registry.WORKLOADS if w.name == "int_virus")


def test_int8_accumulates_into_int32():
    """Not fp32: an all-ones int8 GEMM reaches K, and rounding loses it."""
    assert tensor_virus.accumulator_dtype("int8", _FakeNL) == "int32"


def test_float_operands_accumulate_into_fp32():
    for dtype in ("bf16", "fp16", "fp32"):
        assert tensor_virus.accumulator_dtype(dtype, _FakeNL) == "float32"


def test_int_virus_shares_the_pinned_shape_and_tiling():
    """Same GEMM, different dtype -- so the tile plan must agree."""
    plan = tensor_virus.gemm_plan(*[
        _int_virus().problem["shape"], _int_virus().problem["dtype"]
    ])
    bf16 = tensor_virus.gemm_plan(*[
        _workload().problem["shape"], _workload().problem["dtype"]
    ])
    assert plan["m_tiles"] == bf16["m_tiles"]
    assert plan["k_tiles"] == bf16["k_tiles"]
    assert plan["element_bytes"] == 1


def test_int_virus_is_dispatched_by_the_orchestrator():
    assert "int_virus" in pantheon_neuron.IMPLEMENTED


def test_int_virus_reports_tops_not_tflops():
    """The arithmetic is identical; the unit is not, and the row must say so."""
    assert _int_virus().unit == "TOPS"
    assert _workload().unit == "TFLOPS"


# -- uint8, the dtype trn1 will actually run ---------------------------------
#
# int_virus pinned int8 and was unreachable: `nc_matmul does not support
# stationary.dtype=int8`, trn1.2xlarge 2026-09-08. The supported set is
# fp8_e4m3, fp8_e5m2, bf16, fp16, tf32, fp32 and uint8, so the registry pins
# uint8 and everything that branched on the string "int8" now asks the dtype
# table instead.

def test_uint8_is_an_integer_dtype():
    assert tiling.is_integer("uint8")
    assert tiling.is_integer("int8")
    assert not tiling.is_integer("bf16")
    assert not tiling.is_integer("fp32")


def test_uint8_is_one_byte_wide():
    assert tiling.DTYPE_BYTES["uint8"] == 1


def test_uint8_has_a_torch_dtype():
    assert tiling.TORCH_DTYPES["uint8"] == "uint8"


def test_uint8_accumulates_into_int32_like_int8():
    """The accumulator follows integer-ness, not the specific width.

    An all-ones GEMM of size K reaches K, which overflows any 8-bit type
    long before the last tile, and accumulating into a float would round
    partial sums and break the exactness verify_product_is_correct needs.
    """
    assert tensor_virus.accumulator_dtype("uint8", _FakeNL) == "int32"
    assert tensor_virus.accumulator_dtype("bf16", _FakeNL) == "float32"


def test_the_pinned_int_virus_problem_plans():
    """It could not, while it pinned a dtype the tile table did not carry."""
    workload = {w.name: w for w in registry.WORKLOADS}["int_virus"]
    plan = tensor_virus.gemm_plan(workload.problem["shape"],
                                  workload.problem["dtype"])
    assert plan["element_bytes"] == 1
    assert plan["m"] == plan["n"] == plan["k"] == 8192


def test_an_integer_run_is_labelled_tops_not_tflops():
    """The unit follows the dtype, so uint8 keeps the TOPS the registry declares."""
    workload = {w.name: w for w in registry.WORKLOADS}["int_virus"]
    assert workload.unit == "TOPS"
    assert tiling.is_integer(workload.problem["dtype"])


def test_the_accumulation_loop_is_not_unrolled():
    """The pinned 8192^3 problem compiles only because this loop is rolled.

    The `depth` loop that accumulates into `acc` carries a dependency; NKI
    reserves `affine_range` for loops without one, and it fully unrolls.
    The unroll is cubic in the shape -- 1,024 matmul calls at 2048^3
    against 65,536 at 8192^3 -- which is why the pinned problem never
    compiled and why no compute workload had ever produced the
    neuron-monitor Score its registry entry declares.

    Both kernels must obey it, so this counts accumulation loops rather
    than trusting that a new one inherited the property.
    """
    source = sourcecheck.function_code(tensor_virus._build_kernel)

    # Every loop that accumulates into `acc` is rolled -- one per kernel.
    # Counted against the number of tilings rather than a literal 2, which
    # it was: adding the coalesced kernel made it 3, and the literal failed
    # the new kernel for correctly having the property the test demands.
    assert source.count("for depth in nl . sequential_range (") == len(
        tensor_virus.STRATEGIES), (
        "each kernel needs exactly one rolled accumulation loop"
    )
    # ...and none of them is the unrolling kind.
    assert "for depth in nl . affine_range ( k // CONTRACTION )" not in source


def test_the_blocked_kernel_preloads_and_reuses_the_moving_operand():
    """The whole point of it: rhs read once per column, not per (row, col).

    The load must sit outside the row loop, or nothing has changed.
    """
    source = sourcecheck.function_code(tensor_virus._build_kernel)
    blocked = source[source.index("def tensor_virus_blocked"):]

    assert "rhs_block = nl . ndarray" in blocked
    assert "buffer = nl . sbuf" in blocked
    # The preload precedes the row loop, and the matmul reads the block.
    assert blocked.index("rhs_block [ depth ] = nl . load") < blocked.index(
        "for row in nl . affine_range")
    assert "nl . matmul ( lhs_tile , rhs_block [ depth ]" in blocked
    # lhs is still streamed: holding it too would need the whole operand.
    assert "lhs_tile = nl . load" in blocked


@pytest.mark.parametrize("strategy", ["streaming", "blocked"])
def test_both_tilings_are_selectable(strategy, monkeypatch):
    monkeypatch.setenv("PANTHEON_NEURON_GEMM_TILING", strategy)
    import importlib
    reloaded = importlib.reload(tensor_virus)
    assert reloaded.TILING == strategy
    monkeypatch.delenv("PANTHEON_NEURON_GEMM_TILING", raising=False)
    importlib.reload(tensor_virus)


def test_an_unknown_tiling_is_refused():
    with pytest.raises(ValueError, match="unknown tiling"):
        tensor_virus._build_kernel("bf16", "sideways")


def test_the_blocked_sbuf_block_fits_on_chip():
    """8 MiB against roughly 24 MB of SBUF at the pinned shape."""
    plan = tensor_virus.gemm_plan([8192, 8192, 8192], "bf16")
    block_bytes = (plan["k_tiles"] * tensor_virus.CONTRACTION
                   * tensor_virus.MOVING * 2)
    assert block_bytes == 8 * 1024**2
    assert block_bytes < 20 * 1024**2, "must leave room for lhs and psum"


def test_blocking_raises_arithmetic_intensity_where_streaming_cannot():
    """Streaming's intensity is flat in the shape; blocking's grows."""
    def intensity(n, blocked):
        plan = tensor_virus.gemm_plan([n, n, n], "bf16")
        tiles = plan["m_tiles"] * plan["n_tiles"] * plan["k_tiles"]
        lhs = tiles * tensor_virus.CONTRACTION * tensor_virus.STATIONARY * 2
        rhs_reads = (plan["n_tiles"] * plan["k_tiles"] if blocked else tiles)
        rhs = rhs_reads * tensor_virus.CONTRACTION * tensor_virus.MOVING * 2
        return (2 * n ** 3) / (lhs + rhs)

    streaming = [intensity(n, False) for n in (2048, 4096, 8192)]
    blocked = [intensity(n, True) for n in (2048, 4096, 8192)]

    assert max(streaming) / min(streaming) < 1.01, "flat, which is the defect"
    assert blocked[-1] > 4 * streaming[-1]
    assert blocked[-1] > blocked[0], "and it improves with size"


def test_the_unroll_is_cubic_in_the_shape():
    """Why the pinned shape was unreachable, as arithmetic rather than prose."""
    def bodies(n):
        plan = tensor_virus.gemm_plan([n, n, n], "bf16")
        return plan["m_tiles"] * plan["n_tiles"] * plan["k_tiles"]

    assert bodies(2048) == 1024
    assert bodies(8192) == 65536
    # 4x the shape is 64x the unrolled body count.
    assert bodies(8192) == 64 * bodies(2048)


def test_operand_traffic_scales_like_the_flops():
    """Why the pinned shape is bandwidth-bound, as arithmetic.

    Operand tiles are re-read for every (row, col) pair, so traffic grows
    as n^3 exactly like the FLOPs. Arithmetic intensity is therefore
    constant in the shape rather than growing with it, which is what puts
    the 8192^3 figure on the HBM bandwidth ceiling instead of the engine's.
    """
    def traffic_and_flops(n):
        plan = tensor_virus.gemm_plan([n, n, n], "bf16")
        tiles = plan["m_tiles"] * plan["n_tiles"] * plan["k_tiles"]
        per_tile = (tiling.PARTITION * tiling.PARTITION * 2
                    + tiling.PARTITION * tensor_virus.MOVING * 2)
        return tiles * per_tile, 2 * n ** 3

    intensity = []
    for n in (2048, 4096, 8192):
        traffic, flops = traffic_and_flops(n)
        intensity.append(flops / traffic)

    # Constant to within rounding: the ratio does not improve with size,
    # which is the defect. A blocked GEMM's would grow with the tile.
    assert max(intensity) / min(intensity) < 1.01, intensity


def test_cutting_operand_traffic_did_not_buy_the_speedup_it_implied():
    """The refutation, kept as arithmetic so it cannot quietly lapse.

    Implied traffic at 8192^3 does match memory_read's measured bandwidth
    to within 1%, which is why it read as a bandwidth wall. But a tiling
    that cuts that traffic 4.7x moved throughput 1.06x, so the agreement
    was a coincidence and operand bandwidth is not the binding constraint.

    Both halves are asserted: the match that misled, and the measurement
    that settled it.
    """
    plan = tensor_virus.gemm_plan([8192, 8192, 8192], "bf16")
    tiles = plan["m_tiles"] * plan["n_tiles"] * plan["k_tiles"]
    per_tile = (tiling.PARTITION * tiling.PARTITION * 2
                + tiling.PARTITION * tensor_virus.MOVING * 2)

    implied_gbps = (tiles * per_tile) / (41.870 / 1000) / 1e9
    assert abs(implied_gbps - 256.2) / 256.2 < 0.01, "the coincidence"

    # Measured on trn1.2xlarge 2026-09-08 by tools/compare_tiling.py.
    streaming, blocked = 26.26, 27.93
    traffic_cut = 10.74 / 2.28
    speedup = blocked / streaming
    assert traffic_cut > 4.0
    assert speedup < 1.10, "if bandwidth bound, this would track the cut"


def test_coalesced_is_the_default_tiling(monkeypatch):
    """Default since 2026-09-10, on a trn1.2xlarge run at the pinned shape:
    70.42 TFLOPS analytic, 71.80 by neuron-monitor, every row-tile exact.
    Blocked never was: its justification did not hold."""
    import importlib
    monkeypatch.delenv("PANTHEON_NEURON_GEMM_TILING", raising=False)
    reloaded = importlib.reload(tensor_virus)
    assert reloaded.TILING == "coalesced"


# -- the headline figure is a floor, and has to say so -----------------------

def test_the_headline_figure_is_declared_a_property_of_the_kernel():
    """26.19 TFLOPS against torch.matmul's 66.32 at a matched shape, then
    70.42 against 66.25 once coalesced.

    trn1.2xlarge 2026-09-10, 8192^3 bf16, one process, both products
    verified exact. A plain matmul lowered by neuronx-cc was 2.53x the
    streaming kernel, so the headline compute number was a property of
    the kernel rather than of the part. Coalescing overturned the ratio,
    not the lesson: 71.80 is 75.6% of one core, not its peak.
    """
    doc = tensor_virus.__doc__
    assert "PROPERTY OF THIS KERNEL, NOT OF THE PART" in doc
    assert "66.32" in doc and "26.19" in doc
    assert "70.42" in doc and "71.80" in doc and "28.08" in doc
    assert "compare_matmul_paths" in doc


def test_the_comparison_tool_exists_and_verifies_before_it_divides():
    """A ratio between a correct kernel and a rounded one means nothing."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "tools", "compare_matmul_paths.py")
    assert os.path.exists(path), "the claim above has to stay re-runnable"
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()

    # Through the comment-and-docstring filter, not the raw text. The
    # first version of this read the file directly, and an ordering
    # assertion over raw source is satisfied by the first *mention* of a
    # string -- a line of prose above the code would have made it pass
    # while saying nothing about the code. Three checks in this repo have
    # already passed by accident; this one was written knowing that and
    # still had to be fixed.
    # main()'s own code, not the whole file. Matching "ratio =" across the
    # module found nki_kernel's local `ratio = result.get(...)` first --
    # an ordering assertion is only as good as the two things it orders,
    # and one of mine was the wrong statement in a different function.
    spec = importlib.util.spec_from_file_location("compare_matmul_paths",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code = sourcecheck.function_code(module.main)

    assert "INCORRECT PRODUCT" in code
    # The refusal has to come before the division, not beside it: a ratio
    # between a correct kernel and a rounded one is not a slow kernel.
    assert code.index("INCORRECT PRODUCT") < code.index(
        'ratio = xla [ "tflops" ] / nki [ "tflops" ]')
    assert "return 1" in code[:code.index("ratio =")]
    del raw


def _finding_doc():
    """The headline document, with line wrapping normalised away.

    Two checks here matched "cause unknown" and broke when an edit moved
    the phrase across a line break -- a check coupled to the formatting
    rather than to the content, which is the shape in
    docs/checks_that_pass_by_accident.md that matches a phrasing rather
    than a claim. Collapsing whitespace is what makes them about the
    prose.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "docs",
                           "the_headline_number_is_the_kernel.md"),
              encoding="utf-8") as handle:
        return " ".join(handle.read().split())


def test_the_bandwidth_explanation_is_recorded_as_falsified():
    """Two confident diagnoses were wrong; the third is "unknown".

    102 FLOP/byte times memory_read's 256.2 GB/s is 26.1 TFLOPS, which
    matches the observed figure almost exactly and is a coincidence --
    blocked tiling cuts operand traffic 4.7x and buys 1.06x. A doc that
    dropped the falsification would leave the next reader to believe the
    arithmetic all over again.
    """
    doc = _finding_doc()
    assert "coincidence" in doc
    assert "4.7" in doc and "1.06" in doc
    assert "cause unknown" in doc.lower()


def test_the_tiling_tool_does_not_state_the_falsified_ceiling_as_fact():
    """It was the tool that falsified it, and said otherwise for days.

    "operand traffic 256.4 GB/s against memory_read's 256.2" is a real
    measurement and a coincidence: blocked tiling cuts traffic 4.7x and
    buys 1.06x. A tool whose own output refutes its docstring is worse
    than one that says nothing.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "tools", "compare_tiling.py"),
              encoding="utf-8") as handle:
        doc = handle.read()
    assert "That reading is wrong" in doc
    assert "coincidence" in doc
    # And points at the larger open question rather than leaving the
    # tiling comparison looking like the whole of it.
    assert "compare_matmul_paths" in doc


def test_three_falsified_explanations_are_all_recorded():
    """A repo that keeps only its confirmed guesses keeps a biased sample.

    Bandwidth, tiling volume and k-loop serialisation were each a
    plausible mechanism for the 2.53x gap, and each was measured and
    found wrong. The document keeps all three, because the next person to
    have one of those ideas should find the measurement rather than
    repeat it.
    """
    doc = _finding_doc()
    assert "coincidence" in doc                 # bandwidth
    assert "4.7" in doc and "1.06" in doc       # tiling volume
    assert "0.99" in doc                        # k-loop serialisation
    assert "cause unknown" in doc.lower()


def test_the_psum_constraint_is_recorded():
    """Two PSUM tensors cannot be added directly, which is what stopped
    the split-accumulator variant compiling. A hardware fact worth
    keeping for anyone writing NKI here.
    """
    doc = _finding_doc()
    assert "NCC_IBVF027" in doc
    assert "copied to SBUF" in doc


def test_the_tiling_traffic_cut_is_stated_as_modelled_and_measured():
    """The 4.7x was the model at 8192^3, and it was quoted as measured.

    neuron-profile's full trace counted the transfers at 4096^3 on
    trn1.2xlarge 2026-09-10: streaming moved 631.3 MB and blocked 248.0,
    a 2.55x cut, where the model predicts 1,342 and 302, a 4.44x cut. A
    doc presenting the model's figure as a measurement is the defect; the
    fix is to say which is which wherever the number appears.
    """
    doc = _finding_doc()
    assert "2.55" in doc
    assert "631.3" in doc and "248.0" in doc
    assert "never a measurement" in doc


def test_the_transfer_count_finding_is_recorded_with_its_limits():
    """Blocked matches XLA on bytes and makes 5.1x the transfers, and the
    doc says that is a correlation rather than proof."""
    doc = _finding_doc()
    assert "34,031" in doc and "174,942" in doc
    # \u00d7, the multiplication sign the doc uses; ruff (RUF001)
    # flags a literal one in source as ambiguous with the letter x.
    assert "5.1\u00d7 as many transfers" in doc
    assert "not proof" in doc


# -- the coalesced tiling: larger lhs transfers ------------------------------

def test_coalesced_is_an_accepted_strategy():
    assert "coalesced" in tensor_virus.STRATEGIES
    assert tensor_virus.TILING in tensor_virus.STRATEGIES


def test_the_default_is_the_fastest_tiling_measured_correct():
    """A new tiling is an option until it is measured faster and correct.

    Blocked was justified by a modelled traffic cut that measured smaller,
    and bought 6%. Coalesced became the default on a measurement: the
    recorded 8192^3 rates, all three checked row-tile by row-tile in one
    session on trn1.2xlarge 2026-09-10.
    """
    measured = {"streaming": 26.45, "blocked": 28.16, "coalesced": 70.42}
    assert set(measured) == set(tensor_virus.STRATEGIES)
    assert max(measured, key=measured.get) == "coalesced"
    assert tensor_virus.TILING == "coalesced" or (
        "PANTHEON_NEURON_GEMM_TILING" in os.environ)


def test_the_lhs_load_width_grows_by_the_coalesce_factor():
    """Transfer size is set by the tile's free-dimension width.

    A stationary lhs tile reads STATIONARY bf16 per partition row -- 256
    contiguous bytes. Loading COALESCE_ROWS tiles side by side makes each
    row COALESCE_ROWS times that.
    """
    element = tiling.DTYPE_BYTES["bf16"]
    single = tensor_virus.STATIONARY * element
    coalesced = tensor_virus.STATIONARY * tensor_virus.COALESCE_ROWS * element
    assert single == 256
    assert coalesced == 256 * tensor_virus.COALESCE_ROWS


def test_the_accumulator_fits_in_half_of_psum():
    """COALESCE_ROWS fp32 tiles of MOVING columns, against NeuronCore-v2's
    16 KiB of PSUM per partition. Half leaves the compiler room to
    double-buffer; a full PSUM would not."""
    per_partition = tensor_virus.COALESCE_ROWS * tensor_virus.MOVING * 4
    assert per_partition <= 16 * 1024 // 2, per_partition


@pytest.mark.parametrize("m", [4096, 8192, 512])
def test_shapes_that_divide_are_accepted(m):
    plan = tensor_virus.gemm_plan([m, 4096, 4096], "bf16")
    tensor_virus.validate_tiling(plan, "coalesced")


def test_a_shape_that_would_leave_rows_uncomputed_is_refused():
    """640 is a multiple of STATIONARY but not of STATIONARY * 4.

    `m // width` would floor and the last 128 rows would never be
    computed, while the FLOP count still claimed the whole matrix. The
    product check samples corners and could miss it.
    """
    plan = tensor_virus.gemm_plan([640, 4096, 4096], "bf16")
    with pytest.raises(ValueError, match="rows uncomputed"):
        tensor_virus.validate_tiling(plan, "coalesced")
    # The other tilings step one stationary tile at a time and are fine.
    tensor_virus.validate_tiling(plan, "streaming")
    tensor_virus.validate_tiling(plan, "blocked")


def test_an_unknown_strategy_is_refused_before_anything_compiles():
    plan = tensor_virus.gemm_plan([4096, 4096, 4096], "bf16")
    with pytest.raises(ValueError, match="unknown tiling"):
        tensor_virus.validate_tiling(plan, "tiled_harder")


def test_run_validates_before_building_the_kernel():
    """The check has to come before the compile, or a bad shape costs a
    two-minute compile to discover."""
    code = sourcecheck.flat_function_code(tensor_virus.run)
    assert code.index("validate_tiling") < code.index("_build_kernel")


# -- a check that can see rows in the wrong place ----------------------------

def _output(tiles, k, tile=4, cols=3, store=None):
    """A synthetic kernel output: `store(t)` names which tile's value tile
    t actually received. Identity is a correct kernel."""
    store = store or (lambda t: t)
    rows = []
    for t in range(tiles):
        value = float(k * tensor_virus.row_tile_scale(store(t)))
        rows.extend([[value] * cols for _ in range(tile)])
    return rows


def test_a_correct_output_has_nothing_in_the_wrong_place():
    out = _output(8, k=4096)
    assert tensor_virus.rows_in_wrong_place(out, 4096, tile=4) == []


def test_the_planted_first_accumulator_defect_is_caught():
    """The mutation run on hardware, reproduced in arithmetic.

    Every row-tile in a coalesced block storing acc[0]: tiles 1..3 of
    each block hold tile 0's value. On trn1.2xlarge this left 24 of 32
    tiles wrong at 4096^3.
    """
    rows = tensor_virus.COALESCE_ROWS
    out = _output(32, k=4096, store=lambda t: t - (t % rows))
    wrong = tensor_virus.rows_in_wrong_place(out, 4096, tile=4)
    assert len(wrong) == 32 - 32 // rows == 24


def test_the_all_ones_check_cannot_see_the_same_defect():
    """The reason this check exists. With all-ones inputs every element
    is K regardless of which accumulator wrote it, so the corner ratio the
    old check reads is exactly 1.0 for the broken kernel too."""
    k = 4096
    all_ones_output = [[float(k)] * 3 for _ in range(32 * 4)]
    corner = all_ones_output[0][0] / k
    far = all_ones_output[-1][-1] / k
    assert tensor_virus.verify_product_is_correct((corner + far) / 2) is None


def test_a_permutation_is_caught_too():
    out = _output(8, k=4096, store=lambda t: (t + 1) % 8)
    assert tensor_virus.rows_in_wrong_place(out, 4096, tile=4)


def test_adjacent_tiles_never_share_a_scale():
    """Or a swap between neighbours would go unseen."""
    for t in range(64):
        assert tensor_virus.row_tile_scale(t) != tensor_virus.row_tile_scale(t + 1)


def test_the_largest_expected_value_is_exact_in_fp32():
    """K * 7 must survive the fp32 accumulator and output unrounded."""
    for k in (4096, 8192):
        biggest = k * tensor_virus.ROW_CHECK_PERIOD
        assert biggest < 2 ** 24, biggest
        assert float(biggest) == biggest


class _FakeTensor:
    """Just the torch surface _tile_ranges uses: len, reshape, float, amin,
    amax, tolist. CI has no torch, and an importorskip test would skip
    there every time -- a check that never runs. The real torch path runs
    on hardware in every tensor_virus.run()."""

    def __init__(self, rows):
        self.rows = [list(map(float, r)) for r in rows]

    def __len__(self):
        return len(self.rows)

    def reshape(self, slabs, width):
        assert width == -1
        flat = [v for r in self.rows for v in r]
        size = len(flat) // slabs
        return _FakeTensor([flat[i * size:(i + 1) * size] for i in range(slabs)])

    def float(self):
        return self

    def amin(self, dim):
        assert dim == 1
        return _FakeList([min(r) for r in self.rows])

    def amax(self, dim):
        assert dim == 1
        return _FakeList([max(r) for r in self.rows])


class _FakeList(list):
    def tolist(self):
        return list(self)


def test_a_tensor_output_is_checked_by_reduction_and_agrees_with_lists():
    out = _output(8, k=4096, store=lambda t: t - (t % 4))
    as_lists = tensor_virus.rows_in_wrong_place(out, 4096, tile=4)
    as_tensor = tensor_virus.rows_in_wrong_place(_FakeTensor(out), 4096, tile=4)
    assert as_lists == as_tensor and len(as_tensor) == 6


def test_the_run_operands_are_the_ones_the_check_reads():
    """lhs_t[k, m] = row_scales(M)[m] and rhs = 1 give out[m, n] = K * scale[m];
    that product passes, and the same product with two row-tiles swapped
    does not."""
    tile, k, m, n = 4, 64, 32, 3
    scales = tensor_virus.row_scales(m, tile=tile)
    product = [[float(k * scales[row])] * n for row in range(m)]
    assert tensor_virus.rows_in_wrong_place(product, k, tile=tile) == []
    swapped = product[tile:2 * tile] + product[:tile] + product[2 * tile:]
    assert [w[0] for w in tensor_virus.rows_in_wrong_place(swapped, k, tile=tile)] == [0, 1]


def test_row_scales_follow_the_stationary_tile():
    scales = tensor_virus.row_scales(3 * tensor_virus.STATIONARY)
    stat = tensor_virus.STATIONARY
    assert set(scales[:stat]) == {1} and set(scales[stat:2 * stat]) == {2}
    assert set(scales[2 * stat:]) == {3}


def test_run_checks_the_whole_product_after_the_clock_stops():
    code = sourcecheck.flat_function_code(tensor_virus.run)
    assert "row_check_operands" in code and "torch.ones" not in code
    assert code.index("elapsed = time . perf_counter ( )") < code.index("read_product")
    assert "product_warning" in code


def test_a_wrong_product_invalidates_the_score_not_just_warns():
    """A warning alone left a verified-wrong product reporting PASS."""
    code = sourcecheck.flat_function_code(tensor_virus.run)
    assert 'result [ "score_invalid" ] = result [ "warning" ] is not None' in code


def test_pulse_virus_uses_the_same_row_check():
    """pulse_virus builds the default tiling. If that is coalesced, an
    all-ones check there is blind to the one thing coalescing changed."""
    from kernels import pulse_virus
    code = sourcecheck.flat_function_code(pulse_virus.run)
    for needed in ("row_check_operands", "read_product", "product_warning",
                   "validate_tiling", "score_invalid"):
        assert needed in code, needed
    assert "torch.ones" not in code


class _FakeSink:
    def __init__(self, rows):
        self.rows = rows

    def to(self, where):
        assert where == "cpu"
        return self.rows


def test_read_product_reads_corners_against_their_own_scale():
    """The far corner's expected value is K * its tile's scale, not K."""
    plan = {"m": 3 * tensor_virus.STATIONARY, "n": 2, "k": 64}
    scales = tensor_virus.row_scales(plan["m"])
    rows = [[float(plan["k"] * scales[r])] * plan["n"] for r in range(plan["m"])]
    ratio, misplaced = tensor_virus.read_product(_FakeSink(rows), plan)
    assert ratio == 1.0 and misplaced == []


def test_read_product_is_none_when_nothing_can_be_read():
    class Broken:
        def to(self, where):
            raise RuntimeError("materialisation failed")
    plan = {"m": 128, "n": 2, "k": 64}
    assert tensor_virus.read_product(None, plan) == (None, None)
    assert tensor_virus.read_product(Broken(), plan) == (None, None)
    # And that is reported, not silently passed.
    assert "unverified" in tensor_virus.product_warning(None, None)


def test_product_warning_reports_both_when_both_fire():
    both = tensor_virus.product_warning(0.625, [(1, 128.0, (64.0, 64.0))])
    assert "0.625x" in both and "1 row-tile" in both
    assert tensor_virus.product_warning(1.0, []) is None


def test_verify_rows_is_quiet_when_nothing_is_wrong_or_nothing_was_read():
    assert tensor_virus.verify_rows([]) is None
    assert tensor_virus.verify_rows(None) is None


def test_verify_rows_names_the_first_wrong_tile():
    warning = tensor_virus.verify_rows([(3, 16384.0, (4096.0, 4096.0))])
    assert "1 row-tile" in warning and "tile 3" in warning and "16384" in warning
