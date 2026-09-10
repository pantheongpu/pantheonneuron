"""The tensor_virus kernel: tile maths, FLOP accounting, and its guards.

The NKI kernel needs hardware. What is testable here is the arithmetic that
decides whether a number means anything -- the GEMM tiling, the FLOP count
behind the analytic cross-check, and the correctness guard that separates a
real GEMM from one the compiler reshaped.
"""

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

def test_cross_check_accepts_agreement():
    assert tensor_virus.verify_against_monitor(90.0, 100.0) is None


def test_cross_check_flags_an_idle_engine():
    """The signal that the matmuls were folded away."""
    message = tensor_virus.verify_against_monitor(0.0, 100.0)
    assert message is not None
    assert "eliminated" in message


def test_cross_check_flags_order_of_magnitude_disagreement():
    message = tensor_virus.verify_against_monitor(5.0, 100.0)
    assert message is not None
    assert "differ by more than" in message


def test_cross_check_is_quiet_when_the_monitor_said_nothing():
    """Absent telemetry is handled by monitor_score, not reported as divergence."""
    assert tensor_virus.verify_against_monitor(None, 100.0) is None


def test_cross_check_flags_zero_analytic_throughput():
    message = tensor_virus.verify_against_monitor(10.0, 0.0)
    assert message is not None
    assert "no arithmetic was issued" in message


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

    # Every loop that accumulates into `acc` is rolled...
    assert source.count("for depth in nl . sequential_range (") == 2, (
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


def test_streaming_is_the_default_tiling():
    """The proven path stays default: blocked's justification did not hold."""
    import importlib
    reloaded = importlib.reload(tensor_virus)
    assert reloaded.TILING == "streaming"


# -- the headline figure is a floor, and has to say so -----------------------

def test_the_headline_figure_is_declared_a_floor_not_a_capability():
    """26.19 TFLOPS against torch.matmul's 66.32 at a matched shape.

    trn1.2xlarge 2026-09-10, 8192^3 bf16, one process, both products
    verified exact. A plain matmul lowered by neuronx-cc is 2.53x this
    kernel, so the suite's headline compute number is a property of the
    kernel rather than of the part -- and a cross-platform comparison
    that quotes it against another accelerator's peak is comparing
    against a ceiling this repo built.
    """
    doc = tensor_virus.__doc__
    assert "PROPERTY OF THIS KERNEL, NOT OF THE PART" in doc
    assert "66.32" in doc and "26.19" in doc
    assert "compare_matmul_paths" in doc


def test_the_comparison_tool_exists_and_verifies_before_it_divides():
    """A ratio between a correct kernel and a rounded one means nothing."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "tools", "compare_matmul_paths.py")
    assert os.path.exists(path), "the claim above has to stay re-runnable"
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    assert "INCORRECT PRODUCT" in source
    # The check has to come before the division, not beside it.
    assert source.index("INCORRECT PRODUCT") < source.index(
        'ratio = xla["tflops"] / nki["tflops"]')


def test_the_bandwidth_explanation_is_recorded_as_falsified():
    """Two confident diagnoses were wrong; the third is "unknown".

    102 FLOP/byte times memory_read's 256.2 GB/s is 26.1 TFLOPS, which
    matches the observed figure almost exactly and is a coincidence --
    blocked tiling cuts operand traffic 4.7x and buys 1.06x. A doc that
    dropped the falsification would leave the next reader to believe the
    arithmetic all over again.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "docs",
                           "the_headline_number_is_the_kernel.md"),
              encoding="utf-8") as handle:
        doc = handle.read()
    assert "coincidence" in doc
    assert "4.7" in doc and "1.06" in doc
    assert "cause unknown" in doc.lower()
