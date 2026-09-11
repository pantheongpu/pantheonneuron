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

STATUS: VERIFIED ON HARDWARE at the pinned 8192^3 shape, trn1.2xlarge.
Coalesced tiling (the default) 2026-09-10: 70.42 TFLOPS analytic, 71.80
by neuron-monitor (median of three), every row-tile of the product exact.
Streaming tiling 2026-09-08 and 2026-09-10: 26.1 TFLOPS, product exact.
Earlier verification at reduced shapes on inf2.xlarge 2026-09-07:
``verify_product_is_correct`` returned exactly 1.0 at 1024^3 (2.68 TFLOPS)
and 2048^3 (21.05 TFLOPS).

**THIS NUMBER IS A PROPERTY OF THIS KERNEL, NOT OF THE PART.**

That was learned the hard way. trn1.2xlarge 2026-09-10, same 8192^3
shape, same bf16, same process, both products verified exact, under the
then-default streaming tiling:

    NKI (this kernel)      26.19 TFLOPS    598 passes
    XLA (torch.matmul)     66.32 TFLOPS   1510 passes

A plain ``torch.matmul`` was **2.53x** this kernel, so the headline was a
floor this repo built, not a ceiling Trainium imposed. The coalesced
tiling closed it the same day -- 70.42 against torch.matmul's 66.25 in
one session -- and the lesson stands the other way round too: 71.80 is
75.6% of one NeuronCore's 95 TFLOPS, the best kernel measured here, and
still not the part's peak. ``tools/compare_matmul_paths.py`` re-runs the
comparison; keeping it runnable is what stops either claim going stale.

The cause was not operand bandwidth -- 102 FLOP/byte x memory_read's
256.2 GB/s lands on 26.1 by coincidence, and blocked tiling's 2.55x
traffic cut bought 1.06x. It was the lhs load: four one-tile loads per
four matmuls instead of one four-tile load. A variant with coalesced's
four accumulators and narrow loads read 28.08, blocked's figure. See
docs/the_headline_number_is_the_kernel.md.
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

# Stationary tiles loaded by one lhs DMA in the "coalesced" tiling.
#
# Why this exists: neuron-profile's full trace counted the transfers on
# trn1.2xlarge 2026-09-10. At almost equal bytes the blocked kernel made
# 5.1x as many DMA transfers as torch.matmul (174,942 against 34,031) and
# ran 1.35x slower, and neither NKI tiling made a single transfer over
# 64 KB where XLA made 128.
#
# The transfer size is set by the tile's free-dimension width, not its
# size. `lhs_t[k-slice, m-slice]` reads 128 partition rows of STATIONARY
# bf16 each -- 256 contiguous bytes per row, strided in HBM -- so every
# lhs load is 128 transfers of 256 bytes. Loading COALESCE_ROWS stationary
# tiles at once makes each row COALESCE_ROWS * 256 bytes and cuts the
# number of lhs loads by the same factor.
#
# 4 because the accumulator grows with it: COALESCE_ROWS fp32 tiles of
# MOVING columns is 8 KiB per partition at 4, half of NeuronCore-v2's
# 16 KiB of PSUM per partition, which leaves room for the compiler to
# double-buffer. 8 would fill it.
COALESCE_ROWS = 4

# Which tiling the kernel uses. Both compute the same product -- both
# product-verify at exactly 1.0 on hardware -- and they differ only in how
# often an operand tile is re-read from HBM.
#
#   "streaming"  both operands loaded inside the contraction loop. Every
#                (row, col) pair re-reads every tile it touches.
#   "blocked"    one column's rhs tiles held in SBUF across the row loop,
#                so each is read once per column. 4.7x less operand traffic
#                by the model, 2.55x by neuron-profile at 4096^3,
#                at 8192^3, for 8 MiB of SBUF.
#
# **Streaming was the default until 2026-09-10, and the reason was a
# measurement that refuted the argument for blocking.** The shape sweep found the kernel at 26.26
# TFLOPS at 8192^3 with implied operand traffic of 256.4 GB/s against
# memory_read's 256.2 GB/s on the same part, and that 0.1% agreement looked
# like a bandwidth wall. It was a coincidence. Cutting operand traffic 4.7x
# (modelled; 2.55x measured at 4096^3)
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
# blocked stays available and correct. It was never the default: a 6%
# gain did not pay for the extra SBUF block and a deprecated NKI layout
# ("Block dimension is deprecated. The leading dimension of SBUF tensor
# must be partition dimension"), and the reason it was written turned out
# not to be true.
#
# **Coalesced is the default, since 2026-09-10, on a hardware run.** It
# keeps blocked's rhs block and loads the lhs four stationary tiles wide
# in one nl.load. trn1.2xlarge, 8192^3 bf16, one session, every row-tile
# of the product checked:
#
#     streaming   26.45    blocked   28.16    coalesced   70.42
#     torch.matmul through neuronx-cc: 66.25
#
# and 71.80 TFLOPS by neuron-monitor's effective_flops (median of three,
# the declared Score source). The accumulators are not the cause: four
# accumulators fed by four narrow loads read 28.08. The wide load is all
# of it. See docs/the_headline_number_is_the_kernel.md.
#
# Changing the default moves every published tensor_virus, int_virus and
# pulse_virus Score by ~2.7x, which is why each row carries "tiling".
# Set PANTHEON_NEURON_GEMM_TILING=streaming to reproduce an older figure.
TILING = os.environ.get("PANTHEON_NEURON_GEMM_TILING", "coalesced")

# Every tiling the kernel builder accepts. A new one becomes the default
# only with a hardware run showing it faster *and* correct at the pinned
# shape -- through the row-distinguishing check, not the all-ones one.
STRATEGIES = ("streaming", "blocked", "coalesced")


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


# Distinct values per row-tile, so a kernel that stores the right number
# into the wrong rows cannot pass. Cycles through 1..ROW_CHECK_PERIOD.
ROW_CHECK_PERIOD = 7


def row_tile_scale(tile_index: int) -> int:
    """The multiple of K that row-tile ``tile_index`` must hold.

    With ``lhs_t[k, m] = row_tile_scale(m // STATIONARY)`` and rhs all
    ones, ``out[m, n] = K * row_tile_scale(m // STATIONARY)`` exactly.
    Adjacent tiles -- and the COALESCE_ROWS tiles inside one coalesced
    block -- all expect different values.
    """
    return tile_index % ROW_CHECK_PERIOD + 1


def rows_in_wrong_place(output, k: int, tile: int = STATIONARY):
    """Row-tiles whose values are not the ones only their own rows produce.

    **Why the all-ones product check is not enough.** With lhs and rhs
    all ones, every output element is K whichever accumulator produced
    it, so a kernel that stored its first accumulator into every row --
    or permuted them -- passes ``verify_product_is_correct`` with ratio
    exactly 1.0. The coalesced tiling's entire change is splitting rows
    across COALESCE_ROWS accumulators, which is exactly what that check
    cannot see.

    Measured on trn1.2xlarge 2026-09-10. A deliberately planted
    ``value=acc[0]`` in the coalesced kernel left 24 of 32 row-tiles
    wrong under this check at 4096^3 -- tiles 1..3 of each block holding
    tile 0's value -- and would have passed the all-ones check with every
    element exact. The real kernel left none wrong.

    ``output`` is a CPU tensor or nested lists of rows; returns
    ``[(tile_index, expected, (lowest, highest)), ...]`` for each wrong
    tile. Empty means every element of every tile is exact. A tile is
    exact when its lowest and highest elements both equal the expected
    value, which a tensor answers in two reductions -- a Python pass over
    the 67M elements of an 8192^2 output would take minutes.
    """
    wrong = []
    for index, (lowest, highest) in enumerate(_tile_ranges(output, tile)):
        expected = float(k * row_tile_scale(index))
        if lowest != expected or highest != expected:
            wrong.append((index, expected, (lowest, highest)))
    return wrong


def _tile_ranges(output, tile: int):
    """(lowest, highest) element of each ``tile``-row slab of ``output``."""
    if hasattr(output, "amin"):
        slabs = output.reshape(len(output) // tile, -1).float()
        return list(zip(slabs.amin(dim=1).tolist(), slabs.amax(dim=1).tolist()))
    ranges = []
    for index in range(len(output) // tile):
        values = [float(v) for row in output[index * tile:(index + 1) * tile]
                  for v in row]
        ranges.append((min(values), max(values)))
    return ranges


def row_scales(rows: int, tile: int = STATIONARY) -> typing.List[int]:
    """The scale each output row carries: ``row_tile_scale(row // tile)``."""
    return [row_tile_scale(i // tile) for i in range(rows)]


def row_check_operands(plan: typing.Mapping[str, int], torch_dtype):
    """lhs_t and rhs on the host, built so ``rows_in_wrong_place`` can read
    the product: ``lhs_t[k, m] = row_tile_scale(m // STATIONARY)`` and rhs
    all ones. Every value is at most ROW_CHECK_PERIOD, exact in each
    operand dtype the Tensor Engine takes (bf16, fp16, fp32, uint8, fp8),
    and K * ROW_CHECK_PERIOD is exact in the fp32 and int32 accumulators.
    """
    import torch  # type: ignore

    scale = torch.tensor(row_scales(plan["m"]), dtype=torch.float32)
    lhs_t = scale.unsqueeze(0).expand(plan["k"], plan["m"]).contiguous()
    rhs = torch.ones((plan["k"], plan["n"]), dtype=torch.float32)
    return lhs_t.to(torch_dtype), rhs.to(torch_dtype)


def validate_tiling(plan: typing.Mapping[str, int], strategy: str) -> None:
    """Reject a shape a tiling would silently compute wrongly.

    The coalesced kernel steps the row loop COALESCE_ROWS stationary tiles
    at a time, so M must divide by STATIONARY * COALESCE_ROWS. If it did
    not, `m // width` would floor and the trailing rows would simply never
    be computed -- a partial product, and one whose FLOP count would still
    claim the whole matrix. The product check samples corners and could
    well miss it; refusing the shape cannot.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown tiling {strategy!r}; expected one of {STRATEGIES}")
    if strategy == "coalesced":
        width = STATIONARY * COALESCE_ROWS
        if plan["m"] % width:
            raise ValueError(
                f"M={plan['m']} must be a multiple of {width} for the "
                f"coalesced tiling ({COALESCE_ROWS} stationary tiles per "
                "load); a remainder would leave rows uncomputed"
            )


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
    and break the exactness the product check relies on -- and a GEMM of
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
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown tiling {strategy!r}; expected one of {STRATEGIES}"
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
                # stop the product check from landing on an exact integer;
                # for 8-bit types it would also overflow, since the product
                # reaches K times the row-tile's scale.
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

        **That buys 6%, not the 4.7x the modelled traffic figures suggest** --
        and the measured cut is smaller still, 2.55x at 4096^3, and
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

    @nki.jit
    def tensor_virus_coalesced(lhs_t, rhs):
        """The blocked kernel, with lhs loaded COALESCE_ROWS tiles at a time.

        rhs is held in SBUF per column exactly as in the blocked kernel.
        What changes is the lhs load: one DMA covers COALESCE_ROWS
        stationary tiles side by side, so each partition row reads
        COALESCE_ROWS * STATIONARY contiguous elements instead of
        STATIONARY, and the row loop needs a quarter of the loads.

        Written against a measurement rather than a model, which is the
        difference from the tiling before it: blocked was justified by a
        modelled 4.7x traffic cut that turned out to be 2.55x when
        measured, and bought 6%. This one is justified by a transfer count
        read from the hardware, and will be judged by the same count.

        The accumulator is one 3D PSUM tensor indexed by row-in-block,
        the same shape rhs_block uses in SBUF, so the rows accumulate
        independently -- and two PSUM tensors are never added together,
        which the compiler refuses (NCC_IBVF027, measured 2026-09-10).
        """
        k, m = lhs_t.shape
        _, n = rhs.shape
        k_tiles = k // CONTRACTION
        width = STATIONARY * COALESCE_ROWS

        out = nl.ndarray((m, n), dtype=accumulate_into, buffer=nl.shared_hbm)

        for col in nl.affine_range(n // MOVING):
            rhs_block = nl.ndarray(
                (k_tiles, nl.par_dim(CONTRACTION), MOVING),
                dtype=rhs.dtype, buffer=nl.sbuf,
            )
            for depth in nl.affine_range(k_tiles):
                rhs_block[depth] = nl.load(
                    rhs[depth * CONTRACTION:(depth + 1) * CONTRACTION,
                        col * MOVING:(col + 1) * MOVING]
                )

            for block in nl.affine_range(m // width):
                acc = nl.zeros(
                    (COALESCE_ROWS, nl.par_dim(STATIONARY), MOVING),
                    dtype=accumulate_into, buffer=nl.psum,
                )
                for depth in nl.sequential_range(k_tiles):
                    # One load, COALESCE_ROWS stationary tiles wide.
                    lhs_wide = nl.load(
                        lhs_t[depth * CONTRACTION:(depth + 1) * CONTRACTION,
                              block * width:(block + 1) * width]
                    )
                    for i in range(COALESCE_ROWS):
                        acc[i] += nl.matmul(
                            lhs_wide[:, i * STATIONARY:(i + 1) * STATIONARY],
                            rhs_block[depth], transpose_x=True)

                for i in range(COALESCE_ROWS):
                    row = block * COALESCE_ROWS + i
                    nl.store(
                        out[row * STATIONARY:(row + 1) * STATIONARY,
                            col * MOVING:(col + 1) * MOVING],
                        value=acc[i],
                    )
        return out

    chosen = {"streaming": tensor_virus_kernel,
              "blocked": tensor_virus_blocked,
              "coalesced": tensor_virus_coalesced}[strategy]
    return nki, nl, chosen


def run(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Execute the GEMM loop and return timing plus FLOP accounting.

    Returns the raw material for a Score, not the Score. The registry
    declares tensor_virus is scored from neuron-monitor's effective_flops,
    which does not exist until the monitor stops, so the caller reads it.
    """
    nki_backend.require_toolchain()

    import torch_xla.core.xla_model as xm  # type: ignore

    dtype = str(problem["dtype"])
    plan = gemm_plan(problem["shape"], dtype)
    strategy = str(problem.get("tiling") or TILING)
    validate_tiling(plan, strategy)
    _, _, kernel = _build_kernel(dtype, strategy)

    device = xm.xla_device()
    torch_dtype = tiling.torch_dtype(dtype)

    # Row-distinct operands make every row-tile of the product a known,
    # different multiple of K -- see rows_in_wrong_place for why all-ones
    # operands could not tell a correct kernel from one that stored the
    # wrong accumulator. The operands live in device memory and are read
    # as data, not compiled in as literals, so they cannot be folded.
    host_lhs_t, host_rhs = row_check_operands(plan, torch_dtype)
    lhs_t = host_lhs_t.to(device)
    rhs = host_rhs.to(device)
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

    # Read back after the clock stops.
    product_verified, misplaced = read_product(sink, plan)

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
        # Which tiling produced this figure. The two differ by ~4.7x (modelled;
        # 2.55x measured) in
        # operand traffic at the pinned shape, so a number without this
        # label is not comparable with one that has it.
        "tiling": strategy,
        "plan": plan,
        # 1.0 means both corners equal their expected multiple of K.
        "product_verified_ratio": product_verified,
        # Every row-tile of the product, each checked against the value
        # only its own rows produce. None when the product was not read.
        "row_tiles_checked": plan["m"] // STATIONARY if misplaced is not None else None,
        "row_tiles_wrong": len(misplaced) if misplaced is not None else None,
    }

    result["warning"] = product_warning(product_verified, misplaced)
    # A product that is wrong, or was never read, means the FLOPs beside it
    # describe work that did not happen; the orchestrator fails such a row
    # rather than publishing its Score. Before 2026-09-10 this path only
    # warned, so a verified-wrong product still reported PASS.
    result["score_invalid"] = result["warning"] is not None
    return result


def read_product(sink, plan: typing.Mapping[str, int]):
    """``(product_verified_ratio, misplaced_row_tiles)`` for a product of
    ``row_check_operands``, both None when it could not be read.

    The whole product, not two corners: a sampled check is what let a
    row-mixing kernel pass before. Called after the clock stops -- reading
    back 8192^2 fp32 is 256 MiB and must not land in the timed region.
    """
    if sink is None or not plan["k"]:
        return None, None
    try:
        host = sink.to("cpu")
    except Exception:  # broad: materialisation failed; leave unverified
        return None, None
    last_tile = (plan["m"] - 1) // STATIONARY
    corner = float(host[0][0]) / (plan["k"] * row_tile_scale(0))
    far = float(host[-1][-1]) / (plan["k"] * row_tile_scale(last_tile))
    return (corner + far) / 2.0, rows_in_wrong_place(host, plan["k"])


def product_warning(product_verified, misplaced) -> typing.Optional[str]:
    """Both warnings when both fire: the corner ratio says the product is
    wrong, the row-tile count says how much of it."""
    warnings = [w for w in (verify_product_is_correct(product_verified),
                            verify_rows(misplaced)) if w]
    return "; ".join(warnings) or None


def verify_rows(misplaced) -> typing.Optional[str]:
    """A warning when any row-tile holds a value its own rows cannot produce.

    ``None`` misplaced means the product was never read, which
    verify_product_is_correct already reports, so this stays quiet then.
    """
    if not misplaced:
        return None
    index, expected, (lowest, highest) = misplaced[0]
    return (
        f"{len(misplaced)} row-tile(s) hold values their own rows cannot "
        f"produce (tile {index}: expected {expected:g}, found "
        f"{lowest:g}..{highest:g}) -- the kernel wrote the wrong accumulator "
        "or the wrong rows, and its FLOP count describes a product it did "
        "not deliver"
    )


def verify_product_is_correct(
    product_verified_ratio: typing.Optional[float], tolerance: float = 0.01
) -> typing.Optional[str]:
    """Check the GEMM computed the product it claims to have computed.

    Each corner of the product is a known multiple of K (see
    row_check_operands), so a ratio of 1.0 means both corners hold the
    right value. The far
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


# ``verify_against_monitor`` lived here and was never called from
# anywhere. It compared the declared Score against the analytic
# cross-check and would have caught a monitor reading of zero against a
# kernel claiming throughput -- a real check: written, tested, documented,
# and wired to nothing. The purest form of the defect catalogued in
# docs/checks_that_pass_by_accident.md, since the check did exist.
#
# Its job is now ``pantheon_neuron.override_disagreement``, which is
# called, covers every monitor-scored workload rather than this family
# alone, and carries the two zero cases this one had.
