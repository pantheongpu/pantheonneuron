"""Dense matmul kernel for the ``tensor_virus`` workload.

Score: **TFLOPS**, from ``mean(effective_flops) / 1e12`` as declared in the
registry. That counter comes from neuron-monitor rather than from this
kernel, so the Score is read by the orchestrator after the monitor stops --
see ``pantheon_neuron.monitor_score``. What this module returns is the
analytic cross-check: FLOPs issued over wall time, which is what the kernel
asked the hardware to do rather than what the hardware reports doing.

The two figures answer different questions and both matter. The analytic
number counts the arithmetic in the pinned problem; ``effective_flops``
counts what the Tensor Engine actually retired. A kernel whose matmuls were
folded away still posts a large analytic number, while the monitor reports
an idle engine. A large divergence means the kernel is not doing what it
looks like it is doing.

Unlike the bandwidth kernels, the anti-elimination hazard here is the
operand shape rather than the store: a matmul whose result is never read is
dead code, and one over constant operands can in principle be folded at
compile time. The output is the kernel's return value, held live by the
caller across ``mark_step()``, for the reason memory_read documents.

STATUS: verified on inf2.xlarge 2026-09-07, at reduced shapes. ``nl.matmul``
and the PSUM accumulator ran correctly on their first execution:
``verify_product_is_correct`` returned exactly 1.0 at 1024^3 (2.68 TFLOPS,
12,481 passes) and 2048^3 (21.05 TFLOPS, 12,257 passes), so both sampled
corners held exactly K and the GEMM computed the declared problem.

The pinned 8192^3 shape has **not** run. It unrolls to 65,536 matmul calls,
four times the 16,384-iteration graph that already took roughly seven
minutes to compile on this part, so its compile cost is the open question
rather than its correctness. The near-8x throughput jump between the two
verified shapes says the smaller one is launch-overhead bound, so neither
figure should be read as this part's compute capability.
"""

import os
import time
import typing

from . import nki_backend, tiling


# Tensor Engine tile geometry. The contraction dimension rides the partition
# axis, so it is bounded by the same pmax=128 that tiling.PARTITION records
# from hardware. The moving operand's free axis may be wider than the
# stationary one, which is why N and M differ.
CONTRACTION = tiling.PARTITION      # K per matmul call
STATIONARY = tiling.PARTITION       # M per matmul call
MOVING = 512                        # N per matmul call

# Which tiling the kernel uses. Both compute the same product -- both
# product-verify at exactly 1.0 on hardware -- and they differ only in how
# often an operand tile is re-read from HBM.
#
#   "streaming"  both operands loaded inside the contraction loop. Every
#                (row, col) pair re-reads every tile it touches.
#   "blocked"    one column's rhs tiles held in SBUF across the row loop,
#                so each is read once per column. 4.7x less operand traffic
#                at 8192^3, for 8 MiB of SBUF.
#
# **Streaming is the default, and the reason is a measurement that refuted
# the argument for blocking.** The shape sweep found the kernel at 26.26
# TFLOPS at 8192^3 with implied operand traffic of 256.4 GB/s against
# memory_read's 256.2 GB/s on the same part, and that 0.1% agreement looked
# like a bandwidth wall. It was a coincidence. Cutting operand traffic 4.7x
# moved throughput by 1.06x:
#
#     shape   streaming   blocked   traffic cut   speedup
#     2048^3      23.30     23.38          4.2x     1.00x
#     4096^3      36.55     38.85          4.5x     1.06x
#     8192^3      26.26     27.93          4.7x     1.06x
#
# So operand bandwidth was not the binding constraint, and what is remains
# unknown. Both tilings sit at 25-41% of the ~95 TFLOPS one NeuronCore-v2
# should reach in bf16, and 4096^3 still beats 8192^3 under both. The next
# place to look is per-tile issue overhead and the 128x128x512 tile shape,
# not the memory system.
#
# blocked stays available and correct. It is not the default because a 6%
# gain does not pay for the extra SBUF block and a deprecated NKI layout
# ("Block dimension is deprecated. The leading dimension of SBUF tensor
# must be partition dimension"), and because the reason it was written
# turned out not to be true.
TILING = os.environ.get("PANTHEON_NEURON_GEMM_TILING", "streaming")


def gemm_plan(shape: typing.Sequence[int], dtype: str) -> typing.Dict[str, int]:
    """Split a GEMM into whole Tensor Engine tiles.

    Returns the tile counts and the FLOP count for one pass. A GEMM is
    2*M*N*K FLOPs -- one multiply and one add per element of the contraction
    -- and that figure is the numerator of the analytic cross-check, so it
    must describe the arithmetic actually issued rather than the arithmetic
    requested. Shapes that do not divide into whole tiles are rejected
    instead of being silently rounded, because a partial tile would make the
    FLOP count and the work disagree.
    """
    if len(shape) != 3:
        raise ValueError(f"expected [M, N, K], got {list(shape)}")
    m, n, k = (int(value) for value in shape)
    if dtype not in tiling.DTYPE_BYTES:
        raise ValueError(f"unsupported dtype {dtype!r}")

    for label, value, tile in (("M", m, STATIONARY), ("N", n, MOVING),
                               ("K", k, CONTRACTION)):
        if value <= 0 or value % tile:
            raise ValueError(
                f"{label}={value} must be a positive multiple of {tile}"
            )

    return {
        "m": m, "n": n, "k": k,
        "m_tiles": m // STATIONARY,
        "n_tiles": n // MOVING,
        "k_tiles": k // CONTRACTION,
        "flops_per_pass": 2 * m * n * k,
        "element_bytes": tiling.DTYPE_BYTES[dtype],
    }


def accumulator_dtype(dtype: str, nl):
    """The type the Tensor Engine accumulates a product of ``dtype`` into.

    Signed int8 operands do not reach here on trn1 at all: the engine
    rejects them before accumulation, with `nc_matmul does not support
    stationary.dtype=int8`. Measured 2026-09-08. The supported operand set
    is fp8_e4m3, fp8_e5m2, bf16, fp16, tf32, fp32 and **uint8**, so
    int_virus now pins uint8 -- see kernels/registry.py for why that was a
    registry decision and not a kernel fix.

    Integer operands accumulate into int32, floating-point ones into fp32.
    Accumulating an integer product into a float would round partial sums
    and break the exactness the all-ones check relies on -- and a GEMM of
    size K reaches K in the accumulator, which overflows an 8-bit type long
    before the last tile.
    """
    return nl.int32 if tiling.is_integer(dtype) else nl.float32


def _build_kernel(dtype: str = "bf16", tiling_strategy: typing.Optional[str] = None):
    """Import NKI and construct the kernel.

    Lazy so this module can be imported, and its tile maths tested, on a
    machine with no Neuron toolchain.

    ``dtype`` selects the accumulator only; the operand types travel with
    the tensors. int_virus runs this same kernel over int8 inputs, which is
    the whole reason the accumulator is a parameter rather than a literal.
    """
    # Checked before the toolchain import, so a bad strategy fails on any
    # machine rather than only on one with the Neuron SDK installed.
    strategy = tiling_strategy or TILING
    if strategy not in ("streaming", "blocked"):
        raise ValueError(
            f"unknown tiling {strategy!r}; expected 'streaming' or 'blocked'"
        )

    import neuronxcc.nki as nki  # type: ignore
    import neuronxcc.nki.language as nl  # type: ignore

    accumulate_into = accumulator_dtype(dtype, nl)

    # MEASURED, and not explained: both tilings reach 25-41% of the ~95
    # TFLOPS one NeuronCore-v2 should manage in bf16, and 4096^3 beats the
    # pinned 8192^3 under both. Operand bandwidth was the obvious suspect
    # and is ruled out -- see TILING above for the numbers that ruled it
    # out. What binds this kernel is still open.
    @nki.jit
    def tensor_virus_kernel(lhs_t, rhs):
        """Compute ``lhs_t.T @ rhs`` tile by tile on the Tensor Engine.

        ``lhs_t`` arrives transposed -- [K, M] rather than [M, K] -- because
        the contraction dimension has to ride the partition axis for the
        engine to consume it. Transposing on the host keeps that reshape out
        of the timed region, where it would be measured as if it were
        arithmetic.
        """
        k, m = lhs_t.shape
        _, n = rhs.shape

        out = nl.ndarray((m, n), dtype=accumulate_into, buffer=nl.shared_hbm)

        for row in nl.affine_range(m // STATIONARY):
            for col in nl.affine_range(n // MOVING):
                # Accumulate wider than the operands. Rounding each partial
                # product back to the operand type would lose the sum and
                # stop the all-ones check from landing on an exact integer;
                # for int8 it would also overflow, since the product of an
                # all-ones GEMM reaches K.
                acc = nl.zeros(
                    (nl.par_dim(STATIONARY), MOVING),
                    dtype=accumulate_into, buffer=nl.psum,
                )
                # sequential_range, not affine_range: this loop accumulates
                # into `acc`, which is a loop-carried dependency, and NKI
                # reserves affine_range for loops that do not have one.
                #
                # It is also what makes the pinned problem compilable.
                # affine_range is fully unrolled, and the unroll is cubic in
                # the shape: 2048^3 is 1,024 matmul calls and compiles in
                # seconds, while 8192^3 is 65,536 and has never compiled at
                # all -- which is why every compute workload in this suite
                # has only ever run at a reduced shape, and why none of them
                # has ever produced the neuron-monitor Score the registry
                # declares. Rolling the innermost loop divides the unrolled
                # body count by k_tiles: 64x at the pinned shape.
                for depth in nl.sequential_range(k // CONTRACTION):
                    lhs_tile = nl.load(
                        lhs_t[depth * CONTRACTION:(depth + 1) * CONTRACTION,
                              row * STATIONARY:(row + 1) * STATIONARY]
                    )
                    rhs_tile = nl.load(
                        rhs[depth * CONTRACTION:(depth + 1) * CONTRACTION,
                            col * MOVING:(col + 1) * MOVING]
                    )
                    acc += nl.matmul(lhs_tile, rhs_tile, transpose_x=True)

                nl.store(
                    out[row * STATIONARY:(row + 1) * STATIONARY,
                        col * MOVING:(col + 1) * MOVING],
                    value=acc,
                )
        return out

    @nki.jit
    def tensor_virus_blocked(lhs_t, rhs):
        """Same product, one column's rhs tiles held in SBUF.

        Each rhs tile is read once per column instead of once per (row,
        col): 10.74 GB/pass becomes 2.28 at 8192^3, arithmetic intensity
        102 becomes 482 FLOP/byte, for one SBUF block of
        k_tiles x CONTRACTION x MOVING -- 8 MiB at the pinned shape.

        **That buys 6%, not the 4.7x the traffic figures suggest**, and
        measuring it is how the bandwidth explanation for this kernel's
        throughput was refuted. See TILING above. The kernel is kept
        because it is correct and marginally faster, not because the
        argument for it held.

        lhs is deliberately left streaming. Holding it too would need the
        whole 128 MiB left operand resident, and the rhs tile is four times
        the size of the lhs tile.
        """
        k, m = lhs_t.shape
        _, n = rhs.shape
        k_tiles = k // CONTRACTION

        out = nl.ndarray((m, n), dtype=accumulate_into, buffer=nl.shared_hbm)

        for col in nl.affine_range(n // MOVING):
            # This column's slice of the moving operand, read once and
            # reused by every row below.
            rhs_block = nl.ndarray(
                (k_tiles, nl.par_dim(CONTRACTION), MOVING),
                dtype=rhs.dtype, buffer=nl.sbuf,
            )
            for depth in nl.affine_range(k_tiles):
                rhs_block[depth] = nl.load(
                    rhs[depth * CONTRACTION:(depth + 1) * CONTRACTION,
                        col * MOVING:(col + 1) * MOVING]
                )

            for row in nl.affine_range(m // STATIONARY):
                acc = nl.zeros(
                    (nl.par_dim(STATIONARY), MOVING),
                    dtype=accumulate_into, buffer=nl.psum,
                )
                # Rolled, like the streaming kernel: this accumulates into
                # `acc`, and unrolling it is what made the pinned shape
                # uncompilable.
                for depth in nl.sequential_range(k_tiles):
                    lhs_tile = nl.load(
                        lhs_t[depth * CONTRACTION:(depth + 1) * CONTRACTION,
                              row * STATIONARY:(row + 1) * STATIONARY]
                    )
                    acc += nl.matmul(lhs_tile, rhs_block[depth],
                                     transpose_x=True)

                nl.store(
                    out[row * STATIONARY:(row + 1) * STATIONARY,
                        col * MOVING:(col + 1) * MOVING],
                    value=acc,
                )
        return out

    chosen = (tensor_virus_blocked if strategy == "blocked"
              else tensor_virus_kernel)
    return nki, nl, chosen


def run(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Execute the GEMM loop and return timing plus FLOP accounting.

    Returns the raw material for a Score, not the Score. The registry
    declares tensor_virus is scored from neuron-monitor's effective_flops,
    which does not exist until the monitor stops, so the caller reads it.
    """
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    dtype = str(problem["dtype"])
    plan = gemm_plan(problem["shape"], dtype)
    strategy = str(problem.get("tiling") or TILING)
    _, _, kernel = _build_kernel(dtype, strategy)

    device = xm.xla_device()
    torch_dtype = tiling.torch_dtype(dtype)

    # All-ones operands make the product exactly K in every element, which is
    # the correctness check. They are also the reason the kernel must not be
    # constant-folded: the operands live in device memory and are read as
    # data, not compiled in as literals.
    lhs_t = torch.ones((plan["k"], plan["m"]), dtype=torch_dtype, device=device)
    rhs = torch.ones((plan["k"], plan["n"]), dtype=torch_dtype, device=device)
    xm.mark_step()

    # Compile outside the timed region, and compile the graph the loop will
    # actually run. Holding the result changes the graph -- the output
    # becomes live at the mark_step() cut -- so a warm-up that discards it
    # compiles a different graph and leaves the real one to be built inside
    # the measurement. memory_read measured that on inf2.xlarge 2026-09-07:
    # 0.0208 GB/s over a 45 s run, because a seven-minute compile landed in
    # the middle of it.
    warm = kernel(lhs_t, rhs)
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    # The barrier belongs inside the timed region and the result must stay
    # live -- both lessons are memory_read's, measured on trn1.2xlarge
    # 2026-08-27, where omitting them reported 1636 GB/s against a true 264
    # and 14,513 GB/s against the same, respectively.
    sink = None
    passes = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        sink = kernel(lhs_t, rhs)
        xm.mark_step()
        passes += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    product_verified = None
    if sink is not None:
        try:
            corner = float(sink[0][0])
            far = float(sink[plan["m"] - 1][plan["n"] - 1])
        except Exception:  # materialisation failed; leave unverified
            corner = far = None
        if corner is not None and far is not None and plan["k"]:
            product_verified = (corner + far) / 2.0 / plan["k"]

    flops_issued = plan["flops_per_pass"] * passes
    analytic_tflops = flops_issued / elapsed / 1e12

    result = {
        "passes": passes,
        "elapsed_s": elapsed,
        "flops_issued": flops_issued,
        "analytic_tflops": analytic_tflops,
        # Integer operands make these integer ops, not floating-point ones.
        # The arithmetic is identical and the name is not, so the row says
        # which rather than letting a TOPS figure read as TFLOPS.
        "analytic_unit": "TOPS" if tiling.is_integer(dtype) else "TFLOPS",
        "score_method": "analytic",
        "analytic_basis": (
            "integer ops issued / wall time" if tiling.is_integer(dtype)
            else "FLOPs issued / wall time"
        ),
        "warning": None,
        # Which tiling produced this figure. The two differ by ~4.7x in
        # operand traffic at the pinned shape, so a number without this
        # label is not comparable with one that has it.
        "tiling": strategy,
        "plan": plan,
        # 1.0 means both sampled elements equal K exactly.
        "product_verified_ratio": product_verified,
    }

    wrong = verify_product_is_correct(product_verified)
    if wrong:
        result["warning"] = wrong
    return result


def verify_product_is_correct(
    product_verified_ratio: typing.Optional[float], tolerance: float = 0.01
) -> typing.Optional[str]:
    """Check the GEMM computed the product it claims to have computed.

    With all-ones operands every element of the product is exactly K, so a
    ratio of 1.0 means the sampled corners hold the right value. The far
    corner matters as much as the near one: it is produced by the last tile
    of both loops, so a kernel that computed only its first tiles -- or whose
    accumulation was reshaped -- fails here while still posting a full FLOP
    count and a plausible wall time.

    This check needs no monitor, which is the point. effective_flops is what
    the Score is read from, and a kernel that never reached the Tensor Engine
    reports no flops at all rather than wrong ones; the case this catches is
    the opposite one, where the engine was busy computing something other
    than the declared problem.
    """
    if product_verified_ratio is None:
        return "product could not be read back -- GEMM correctness unverified"
    if abs(product_verified_ratio - 1.0) > tolerance:
        return (
            f"product is {product_verified_ratio:.3f}x the expected value "
            "-- the GEMM did not compute the pinned problem, so its FLOP "
            "count describes work that did not happen"
        )
    return None


def verify_against_monitor(
    monitor_tflops: typing.Optional[float],
    analytic_tflops: float,
    tolerance: float = 0.5,
) -> typing.Optional[str]:
    """Compare the declared Score against the analytic cross-check.

    ``effective_flops`` is what the hardware retired; the analytic figure is
    what the kernel issued. They should agree to within a wide margin -- the
    monitor samples on a period and a short run catches ramp-up, so this is
    deliberately loose. What it is looking for is the order-of-magnitude
    disagreement that means the matmuls were folded away or the engine sat
    idle while the wall clock ran.
    """
    if analytic_tflops <= 0:
        return "analytic throughput is zero -- no arithmetic was issued"
    if monitor_tflops is None:
        return None
    if monitor_tflops <= 0:
        return (
            "neuron-monitor reported no Tensor Engine activity while the "
            f"kernel claimed {analytic_tflops:.2f} TFLOPS -- the matmuls "
            "were probably eliminated"
        )
    ratio = monitor_tflops / analytic_tflops
    if ratio < (1 - tolerance) or ratio > (1 + tolerance):
        return (
            f"monitor {monitor_tflops:.2f} TFLOPS and analytic "
            f"{analytic_tflops:.2f} TFLOPS differ by more than "
            f"{tolerance:.0%} (ratio {ratio:.2f})"
        )
    return None
