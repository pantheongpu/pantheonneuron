"""The tensor_virus kernel: tile maths, FLOP accounting, and its guards.

The NKI kernel needs hardware. What is testable here is the arithmetic that
decides whether a number means anything -- the GEMM tiling, the FLOP count
behind the analytic cross-check, and the correctness guard that separates a
real GEMM from one the compiler reshaped.
"""

import pytest

import pantheon_neuron
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

    The `depth` loop accumulates into `acc`, which is a loop-carried
    dependency; NKI reserves `affine_range` for loops without one, and it
    fully unrolls. The unroll is cubic in the shape -- 1,024 matmul calls at
    2048^3 against 65,536 at 8192^3 -- which is why the pinned problem never
    compiled and why no compute workload had ever produced the
    neuron-monitor Score its registry entry declares.

    Checked textually because reproducing it needs a compiler. The two outer
    loops stay affine: they are genuinely independent, and rolling them
    would cost throughput for nothing.
    """
    import inspect

    source = inspect.getsource(tensor_virus._build_kernel)
    assert "for depth in nl.sequential_range(" in source, (
        "the accumulation loop must stay rolled or the pinned shape stops "
        "compiling"
    )
    assert "for depth in nl.affine_range(" not in source
    # The independent loops are unchanged.
    assert source.count("nl.affine_range(") == 2


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


def test_the_pinned_shape_sits_on_the_measured_bandwidth_ceiling():
    """8192^3 implied traffic against memory_read's measured bandwidth.

    Measured on trn1.2xlarge 2026-09-08: 41.871 ms/pass at 8192^3, and
    256.2 GB/s of single-core HBM read bandwidth from memory_read on the
    same part. If the two agree, the compute workload is reporting the
    memory system.
    """
    plan = tensor_virus.gemm_plan([8192, 8192, 8192], "bf16")
    tiles = plan["m_tiles"] * plan["n_tiles"] * plan["k_tiles"]
    per_tile = (tiling.PARTITION * tiling.PARTITION * 2
                + tiling.PARTITION * tensor_virus.MOVING * 2)

    implied_gbps = (tiles * per_tile) / (41.871 / 1000) / 1e9
    measured_hbm_gbps = 256.2

    assert abs(implied_gbps - measured_hbm_gbps) / measured_hbm_gbps < 0.01
