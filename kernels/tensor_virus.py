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

STATUS: UNTESTED ON HARDWARE. ``nl.load``, ``nl.store``, ``nl.ndarray`` with
``shared_hbm``, ``nl.zeros``, ``nl.affine_range`` and ``nl.par_dim`` were all
exercised on trn1.2xlarge on 2026-08-27 by ``memory_read``. ``nl.matmul`` and
the PSUM accumulation buffer were **not** -- they are new here. Treat the
first hardware run as bring-up, not measurement, and read
``verify_product_is_correct`` first: with all-ones operands every output
element must equal K exactly, which is what distinguishes a real GEMM from
one the compiler reshaped.
"""

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


def _build_kernel():
    """Import NKI and construct the kernel.

    Lazy so this module can be imported, and its tile maths tested, on a
    machine with no Neuron toolchain.
    """
    import neuronxcc.nki as nki  # type: ignore
    import neuronxcc.nki.language as nl  # type: ignore

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

        out = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.shared_hbm)

        for row in nl.affine_range(m // STATIONARY):
            for col in nl.affine_range(n // MOVING):
                # PSUM accumulates in fp32 regardless of operand dtype: the
                # engine's accumulator is fp32, and rounding each partial
                # product back to bf16 would both lose the sum and stop the
                # all-ones check below from landing on an exact integer.
                acc = nl.zeros(
                    (nl.par_dim(STATIONARY), MOVING),
                    dtype=nl.float32, buffer=nl.psum,
                )
                for depth in nl.affine_range(k // CONTRACTION):
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

    return nki, nl, tensor_virus_kernel


def run(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Execute the GEMM loop and return timing plus FLOP accounting.

    Returns the raw material for a Score, not the Score. The registry
    declares tensor_virus is scored from neuron-monitor's effective_flops,
    which does not exist until the monitor stops, so the caller reads it.
    """
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    plan = gemm_plan(problem["shape"], str(problem["dtype"]))
    _, _, kernel = _build_kernel()

    device = xm.xla_device()
    torch_dtype = tiling.torch_dtype(str(problem["dtype"]))

    # All-ones operands make the product exactly K in every element, which is
    # the correctness check. They are also the reason the kernel must not be
    # constant-folded: the operands live in device memory and are read as
    # data, not compiled in as literals.
    lhs_t = torch.ones((plan["k"], plan["m"]), dtype=torch_dtype, device=device)
    rhs = torch.ones((plan["k"], plan["n"]), dtype=torch_dtype, device=device)
    xm.mark_step()

    # Compile outside the timed region; a NEFF build is tens of seconds and
    # would otherwise be counted as execution.
    kernel(lhs_t, rhs)
    xm.wait_device_ops()

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
        "score_method": "analytic",
        "analytic_basis": "FLOPs issued / wall time",
        "warning": None,
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
