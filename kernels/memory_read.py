"""HBM streaming-read kernel for the ``memory_read`` workload.

Score: **GB/s**, from ``hbm_read_bytes / total_time`` as declared in the
registry. That counter comes from ``neuron-profile``, so the measured
number reflects HBM traffic the hardware actually performed, which is not
necessarily the traffic we asked for -- the compiler may coalesce, and
values already resident in SBUF are not re-read. An analytic figure
(bytes we requested / wall time) is recorded alongside as a cross-check;
a large divergence between the two means the kernel is not doing what it
looks like it is doing.

STATUS: verified on trn1.2xlarge 2026-08-27 and inf2.xlarge 2026-09-07.
The read-coverage check has returned exactly 1.0 on both parts, so the
kernel reads every byte the plan describes. Measured 264 GB/s on trn1 at
1 GiB and 236.9 GB/s on inf2 at the pinned 8 GiB.

Its declared Score source **now produces a number**: 256.0888 GB/s via
neuron-profile on trn1.2xlarge 2026-09-10 -- and 272.94 (cv 0.0006) once
the profile's 2.08 ms idle startup stopped counting as transfer time.
See profiler.bandwidth_gbps. For a long time it did not --
the profiler needs a NeuronCore to replay the NEFF and the workload held
them all -- and the core reservation in kernels/cores.py is what closed it.

Two things had to be true and the second took longer. Reserving the core
let the capture run at all; identifying *which* NEFF was this kernel's
needed compile-cache isolation, because timestamp narrowing alone selected
the wrong graph and verify_profile_covers_plan correctly refused it
(*profiled graph moved 4 bytes against a plan of 8589934592*).

The reservation used to be off for any selection containing a
`cores: "all"` workload, which included --test all and --test memory.
Since 2026-09-11 it is made after those aggregates run, so a full pass
reaches the profiler too (pantheon_neuron.reservation_point).

**What sets the rate, measured rather than assumed** (trn1.2xlarge
2026-09-10, 2 GiB bf16, one core, each variant in its own process, every
product exact):

    variant                          GB/s   vector   DMA
    this kernel: 2048 wide, cast+sum 269.5   0.55   0.79
    8192 wide (1/4 the loads)        269.8
     512 wide (4x the loads)         208.4
    sum as fp32, no explicit cast    269.4   0.55   0.79   (same code emitted)
    cast + t*t + sum                 136.8   0.88   0.50
    torch.sum through neuronx-cc     174.2                 (122.4 at 8192 wide)

- **The loads set it, not the reduction.** The vector engine is busy 55%
  of the execution and wider loads change nothing. At the pinned 8 GiB,
  through the harness, DMA is active 0.94 and vector 0.65 (256.10 GB/s by
  neuron-profile), so the DMA side is close to saturated. The two-core
  aggregate reads exactly 2x (541 GB/s), which also points at a per-core
  limit on the DMA side, not at HBM.
- **The loads sit at a per-core DMA ceiling, not at HBM.** Reads reach
  ~273 and 8192-wide writes ~275 GB/s on one core; a read-and-write copy
  on one core moves ~207 combined (kv_cache_churn); two cores sum to
  exactly 2x (542 of the chip's 880). And more DMA concurrency does not
  lift it -- trn1.2xlarge 2026-09-11, 2 GiB: two independent loads per
  iteration 268.8, one 16384-wide load 271.1, against 269.5 for this
  kernel. Whatever the ceiling is, it is per core and it is not issue rate.
- **The kernel is not a floor.** The compiler's own reduction over the
  same bytes is 1.55x slower. That is the opposite of tensor_virus, where
  torch.matmul was 2.53x the hand-written kernel.
- **The consumer has about 1.5x headroom at the pinned size (0.65), and
  crossing it is silent.**
  Doubling its work halved the "bandwidth" with coverage still 1.000000
  and the product still exact. verify_consumer_not_binding reads the
  vector engine's share from the profile this kernel already captures,
  and warns past CONSUMER_BOUND. Loads narrower than ~2 KiB per
  partition row cost throughput too (208 at 1 KiB), which matches what
  tensor_virus's coalesced tiling found.
"""

import os
import time
import typing

from . import cores, nki_backend, profiler, registry, tiling


# Tile geometry lives in kernels/tiling.py, shared with memory_write.
# Re-exported here so existing callers and tests keep working.
PARTITION = tiling.PARTITION
FREE_ELEMENTS = tiling.FREE_ELEMENTS
DTYPE_BYTES = tiling.DTYPE_BYTES
tile_plan = tiling.tile_plan


def _build_kernel():
    """Import NKI and construct the kernel.

    Imported lazily so the module can be inspected, and its tile maths
    tested, on a machine with no Neuron toolchain installed.
    """
    import neuronxcc.nki as nki  # type: ignore
    import neuronxcc.nki.language as nl  # type: ignore

    @nki.jit
    def memory_read_kernel(source):
        """Stream `source` from HBM through SBUF, reducing as we go.

        The reduction is not the point -- it exists so the loads have a
        consumer. Without it the compiler is free to prove the loaded
        values are unused and delete the DMA, which would produce a
        kernel that runs fast and reads nothing.
        """
        partitions, free_size = source.shape

        total = nl.zeros((nl.par_dim(PARTITION), 1), dtype=nl.float32)
        accumulator = nl.ndarray(
            (nl.par_dim(PARTITION), 1), dtype=nl.float32, buffer=nl.shared_hbm
        )

        rows = partitions // PARTITION
        for row in nl.affine_range(rows):
            tile = nl.load(
                source[row * PARTITION:(row + 1) * PARTITION, 0:free_size]
            )
            total += nl.sum(tile.astype(nl.float32), axis=1, keepdims=True)

        nl.store(accumulator, value=total)
        return accumulator

    return nki, nl, memory_read_kernel


def run(problem: typing.Mapping[str, typing.Any], duration: int,
        before_loop: typing.Optional[typing.Callable[[], None]] = None) -> dict:
    """Execute the streaming read and return timing plus byte accounting.

    Returns the raw material for a Score; it does not compute the Score
    itself. The registry declares the Score comes from the profiler's
    ``hbm_read_bytes``, and the caller is responsible for reading it.
    """
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    plan = tile_plan(int(problem["bytes"]), str(problem["dtype"]))
    _, _nl, kernel = _build_kernel()

    # torch_neuronx deletes its compiler workdir unless told otherwise, and
    # neuron-profile capture needs the NEFF that lives there.
    workdir = os.environ.get("PANTHEON_NEURON_WORKDIR", "/tmp/pantheon_ccwork")
    os.makedirs(workdir, exist_ok=True)
    # Compile into the same directory the profiler will search, so it holds
    # this run's graphs and nothing else. See kernels/cores.py.
    os.environ.setdefault(cores.COMPILE_CACHE, workdir)

    device = xm.xla_device()
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[problem["dtype"]]

    # One tall buffer; the kernel walks it PARTITION rows at a time.
    rows = plan["tiles"] * PARTITION
    source = torch.ones(
        (rows, plan["free"]), dtype=torch_dtype, device=device
    )
    xm.mark_step()

    # Taken before the kernel compiles, so the profiler can tell this
    # kernel's graph apart from every other NEFF in the compiler's cache.
    compile_started = time.time()

    # Warm up so compilation is not inside the timed region -- and warm up
    # the graph the loop actually runs. Holding the result changes the
    # graph, because the output becomes live at the mark_step() cut, so a
    # warm-up that discards it compiles a different one and leaves the real
    # graph to be built inside the measurement.
    #
    # Measured on inf2.xlarge 2026-09-07 with the pinned 8 GiB problem: the
    # discarding warm-up compiled at 22:22:53, the timed loop's first call
    # triggered a second compile that finished at 22:29:46, and the run
    # reported 0.0208 GB/s over 478 s with total_executions=2 and 0.02%
    # NeuronCore utilisation. Nearly seven minutes of compilation was
    # measured as though it were bandwidth.
    warm = kernel(source)
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    # xm.mark_step() queues work and returns; it does not wait for the
    # device. Timing without a barrier measures queue submission, not
    # execution, and the error grows with the buffer -- measured on a
    # trn1.2xlarge on 2026-08-27:
    #
    #     buffer     no barrier      with barrier
    #      128 MiB    208 GB/s        200 GB/s
    #      512 MiB    838 GB/s        256 GB/s
    #     1024 MiB   1636 GB/s        264 GB/s
    #
    # Elapsed stayed pinned at 0.013s in the unsynchronised case regardless
    # of size, so the "bandwidth" was just bytes divided by a constant. The
    # barrier must be inside the timed region.
    # The kernel result must stay LIVE across mark_step(). Discarding it --
    # as `kernel(source)` on its own does -- leaves nothing referencing the
    # graph at the cut point, so XLA proves it dead and skips the DMA.
    # Measured on trn1.2xlarge 2026-08-27: a 90s run reported 14,513 GB/s
    # (16x this part's HBM -- 820 GiB/s per chip, verified 2026-09-10
    # against the AWS Neuron architecture docs; see registry.PART_PEAKS)
    # while neuron-monitor recorded
    # total_executions=1 and NeuronCore utilisation of 0.05%. `passes`
    # counted thousands of submissions; the device ran the graph once.
    sink = None
    passes = 0
    # memory_agg's start barrier: every worker compiled and warmed up, then
    # released together, so the aggregate's loops coincide.
    if before_loop is not None:
        before_loop()
    # Wall-clock brackets of the timed loop itself, comparable across
    # processes. The aggregate placed each loop by subtracting elapsed_s
    # from when run() *returned* -- but run() goes on after the loop (read
    # back, a profiler attempt), for a time that differs per worker.
    loop_started_at = time.time()
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        sink = kernel(source)
        xm.mark_step()
        passes += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started
    loop_finished_at = time.time()

    # One pass reduces an all-ones buffer along the free axis, so every
    # partition of the accumulator must equal tiles * FREE_ELEMENTS exactly.
    # This is a direct check that the kernel read the whole plan rather than
    # a coalesced fraction of it, and it needs no profiler -- which matters,
    # because the profiler is exactly what is unavailable when the analytic
    # figure is load-bearing.
    read_verified = None
    if sink is not None:
        expected_per_pass = float(plan["tiles"] * plan["free"])
        try:
            observed = float(sink[0][0])
        except Exception:  # materialisation failed; leave unverified
            observed = None
        if observed is not None and expected_per_pass > 0:
            read_verified = observed / expected_per_pass

    bytes_requested = plan["actual_bytes"] * passes
    analytic = bytes_requested / elapsed / 1e9

    result = {
        "passes": passes,
        "elapsed_s": elapsed,
        "loop_started_at": loop_started_at,
        "loop_finished_at": loop_finished_at,
        "bytes_requested": bytes_requested,
        "analytic_gbps": analytic,
        "profiler_gbps": None,
        "score_method": "analytic",
        "analytic_basis": "bytes moved / wall time",
        "warning": None,
        "plan": plan,
        # 1.0 means the last pass read exactly the planned bytes.
        "read_verified_ratio": read_verified,
    }

    elided = verify_read_completed(read_verified)
    if elided:
        result["warning"] = elided
        return result

    # The declared Score source. A failure here degrades to the analytic
    # figure rather than aborting the run -- but the row records which was
    # used, so a provisional number is never mistaken for the real one.
    try:
        result.update(_profile(workdir, compile_started, plan["actual_bytes"]))
    except profiler.ProfilerUnavailable as error:
        result["warning"] = f"profiler unavailable, Score is analytic: {error}"
        return result

    divergence = verify_against_analytic(result["profiler_gbps"], analytic)
    if divergence:
        result["warning"] = divergence
        # The profiler figure is the one that loses. Coverage says the
        # captured graph moved about the planned bytes, but two workloads
        # in this suite pin 8 GiB, so byte-coverage cannot tell their
        # graphs apart -- on trn1.2xlarge 2026-09-08 a warm cache had the
        # selector publish 23.58 GB/s against an analytic 271.7.
        #
        # read_verified_ratio breaks the tie. It is measured from the
        # kernel's own accumulator, not from any profile, and at 1.0 it
        # proves this kernel touched every planned byte. The analytic
        # figure is then trustworthy and the profile is somebody else's
        # graph, so the Score degrades rather than publishing a number
        # from a capture we cannot attribute.
        if _touched_the_whole_plan(result.get("read_verified_ratio")):
            result["profiler_gbps"] = None
            # And its engine counters, which describe the same foreign
            # graph. On trn1.2xlarge 2026-09-10 two trees sharing a
            # compile workdir left the selector holding a planted
            # heavy-consumer kernel's NEFF: the Score correctly fell back
            # to analytic, and the row still reported vector 0.97 --
            # the other kernel's reading, beside a Score that disowned it.
            result["consumer_engine_active"] = None
            result["dma_active"] = None
            result["score_method"] = "analytic"
            result["analytic_basis"] = (
                "bytes moved / wall time; the profiled graph could not be "
                "attributed to this kernel"
            )
        return result

    bound = verify_consumer_not_binding(result.get("consumer_engine_active"))
    if bound:
        result["warning"] = bound
    return result


# The reduction that keeps the loads alive runs on the vector engine, and
# past this fraction of the execution it is the reduction's speed the
# bandwidth measures. Measured on trn1.2xlarge 2026-09-10, 2 GiB bf16 on
# one core, each variant its own process:
#
#     consumer              vector   DMA    GB/s
#     cast + sum (this)      0.55   0.79   269.5
#     cast + t*t + sum       0.88   0.50   136.8
#
# At the pinned 8 GiB through the harness: this kernel vector 0.65 / DMA
# 0.94, the heavier consumer vector 0.97 at 133.2 GB/s.
#
# Both products exact, both coverage 1.000000. The heavier consumer halved
# the "bandwidth" without moving one byte fewer -- a correct-looking wrong
# measurement that nothing in the row could see. The threshold sits
# between the readings -- above 0.65, below 0.88.
CONSUMER_BOUND = 0.8


def verify_consumer_not_binding(vector_active) -> typing.Optional[str]:
    """Flag a read whose rate is set by its consumer rather than by HBM.

    ``vector_active`` is neuron-profile's ``vector_engine_active_time_percent``
    for the captured graph -- a 0-1 fraction, like every ``_percent``
    counter (docs/neuron_counters.md). None when there was no profile,
    which says nothing either way.
    """
    if not isinstance(vector_active, (int, float)):
        return None
    if vector_active < CONSUMER_BOUND:
        return None
    return (
        f"the vector engine was active {vector_active:.0%} of the execution: "
        "the reduction that consumes the loads is setting the rate, so this "
        "is the reduction's throughput rather than HBM's"
    )


def _touched_the_whole_plan(ratio, tolerance: float = 0.01) -> bool:
    """Did the kernel provably move the bytes its plan describes?

    Read from the kernel's own output, so it is independent of whatever
    the profiler captured -- which is exactly what makes it able to
    adjudicate between them.
    """
    return isinstance(ratio, (int, float)) and abs(ratio - 1.0) <= tolerance


def _profile(workdir: str, since: float, planned_bytes: int) -> dict:
    """Identify this kernel's graph among the compiler's NEFFs, and read it.

    ``since`` and ``planned_bytes`` both exist to make sure the counters
    came from this kernel: the first ranks the candidates, the second
    decides which one is actually ours.

    The plan check used to run once, against a single guess, and reject it
    -- which is how every scored run in this suite's history ended up on
    the analytic fallback. It now selects: each candidate is captured until
    one accounts for the planned traffic. See ``profiler.select_by_plan``.
    """
    candidates = profiler.find_neffs(workdir, since=since)
    session = os.path.join(workdir, "memory_read.ntff")
    found = profiler.select_by_plan(candidates, session, "read", planned_bytes)
    counters = found["counters"]
    return {
        "profiler_gbps": profiler.bandwidth_gbps(counters, "read"),
        # Matches registry.PROFILER exactly, so "declared source" and
        # "method used" are string-comparable when they agree.
        "score_method": registry.PROFILER,
        "hbm_read_bytes": counters.get("hbm_read_bytes"),
        "profiler_total_time_s": counters.get("total_time"),
        # The Score's denominator and what it excludes: the profiled
        # execution's idle startup, 2.08 ms on trn1.2xlarge 2026-09-10,
        # which the workload's back-to-back executions do not pay. See
        # profiler.bandwidth_gbps.
        "profiler_time_basis": profiler.execution_window(counters)[1],
        "profiler_startup_s": _startup(counters),
        # Which side of the kernel set the rate: the loads (DMA) or the
        # reduction that consumes them (vector). See CONSUMER_BOUND.
        "consumer_engine_active": counters.get("vector_engine_active_time_percent"),
        "dma_active": counters.get("dma_active_time_percent"),
        # What the search had to do to find it. A run that needed the
        # fourth candidate is telling us mtime ranking is worth revisiting.
        "profiler_neff": os.path.basename(found["neff"]),
        "profiler_plan_coverage": found["plan_coverage"],
        "profiler_candidates_tried": found["candidates_tried"],
        "profiler_candidates_available": found["candidates_available"],
    }


def verify_read_completed(
    read_verified_ratio: typing.Optional[float], tolerance: float = 0.01
) -> typing.Optional[str]:
    """Check the kernel actually read the buffer the plan describes.

    ``verify_against_analytic`` catches an eliminated kernel only when the
    profiler is available -- but the profiler failing is precisely when the
    analytic figure becomes the Score, so that net has a hole exactly where
    it is needed. This check closes it using the accumulator alone.

    A ratio of 1.0 means the last pass summed every planned element. Less
    means loads were coalesced or skipped; more means the accounting is
    wrong. Either way the analytic bandwidth cannot be trusted.
    """
    if read_verified_ratio is None:
        return "kernel output could not be read back -- read coverage unverified"
    if abs(read_verified_ratio - 1.0) > tolerance:
        return (
            f"kernel read {read_verified_ratio:.3f}x the planned bytes "
            "-- loads were coalesced or eliminated, so the analytic "
            "bandwidth is not a measurement"
        )
    return None


# How far the profile figure may sit from the wall-clock one. It was 50%,
# and had to be while the profile divided by total_time: its bias ran from
# 6% at 8 GiB to 51% at 1 GiB (trn1.2xlarge 2026-09-10). Over
# total_active_time the kernel's graph lands 0.5-2.3% above the wall clock
# at every size, while a buffer-copy graph captured by mistake at 1 GiB
# read 0.66 of it -- inside the old tolerance, so it would have been
# published. When the two diverge the kernel's own coverage check decides
# and the Score falls back to analytic, so tightening this can only cost a
# declared Score, never publish a wrong one.
ANALYTIC_TOLERANCE = 0.15


def _startup(counters) -> typing.Optional[float]:
    """Seconds of the profiled execution in which nothing was active."""
    total, active = counters.get("total_time"), counters.get("total_active_time")
    if isinstance(total, (int, float)) and isinstance(active, (int, float)):
        return round(total - active, 9)
    return None


def verify_against_analytic(
    profiler_gbps: float, analytic_gbps: float,
    tolerance: float = ANALYTIC_TOLERANCE,
) -> typing.Optional[str]:
    """Compare the profiler Score against the analytic cross-check.

    A kernel that compiles but whose loads were optimised away still
    produces a fast wall time and a large analytic figure, while the
    profiler reports almost no HBM traffic. That divergence is the signal;
    returns a message when the two disagree beyond ``tolerance``, else None.
    """
    if analytic_gbps <= 0:
        return "analytic bandwidth is zero -- no bytes were requested"
    if profiler_gbps <= 0:
        return (
            "profiler reported no HBM traffic while the kernel claimed "
            f"{analytic_gbps:.1f} GB/s -- the loads were probably eliminated"
        )
    ratio = profiler_gbps / analytic_gbps
    if ratio < (1 - tolerance) or ratio > (1 + tolerance):
        return (
            f"profiler {profiler_gbps:.1f} GB/s and analytic "
            f"{analytic_gbps:.1f} GB/s differ by more than "
            f"{tolerance:.0%} (ratio {ratio:.2f})"
        )
    return None
