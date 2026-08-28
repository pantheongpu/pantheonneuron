"""HBM streaming-read kernel for the ``memory_read`` workload.

Score: **GB/s**, from ``hbm_read_bytes / total_time`` as declared in the
registry. That counter comes from ``neuron-profile``, so the measured
number reflects HBM traffic the hardware actually performed, which is not
necessarily the traffic we asked for -- the compiler may coalesce, and
values already resident in SBUF are not re-read. An analytic figure
(bytes we requested / wall time) is recorded alongside as a cross-check;
a large divergence between the two means the kernel is not doing what it
looks like it is doing.

STATUS: UNTESTED ON HARDWARE. Written against the NKI programming model
but never executed on a Neuron device -- there was no instance available
when this was written. The tile-loop structure and byte accounting are
covered by tests that run without hardware; the NKI API calls themselves
are not. Treat the first hardware run as a bring-up, not a measurement,
and see `verify_against_analytic` for the check that will catch a kernel
that compiles but reads nothing.
"""

import os
import time
import typing

from . import nki_backend, profiler, registry, tiling


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


def run(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Execute the streaming read and return timing plus byte accounting.

    Returns the raw material for a Score; it does not compute the Score
    itself. The registry declares the Score comes from the profiler's
    ``hbm_read_bytes``, and the caller is responsible for reading it.
    """
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    plan = tile_plan(int(problem["bytes"]), str(problem["dtype"]))
    _, nl, kernel = _build_kernel()

    # torch_neuronx deletes its compiler workdir unless told otherwise, and
    # neuron-profile capture needs the NEFF that lives there.
    workdir = os.environ.get("PANTHEON_NEURON_WORKDIR", "/tmp/pantheon_ccwork")
    os.makedirs(workdir, exist_ok=True)

    device = xm.xla_device()
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[problem["dtype"]]

    # One tall buffer; the kernel walks it PARTITION rows at a time.
    rows = plan["tiles"] * PARTITION
    source = torch.ones(
        (rows, plan["free"]), dtype=torch_dtype, device=device
    )
    xm.mark_step()

    # Warm up so compilation is not inside the timed region. A NEFF
    # compile is tens of seconds and would swamp the measurement.
    kernel(source)
    xm.mark_step()

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
    # (17x the part's ~820 GB/s HBM) while neuron-monitor recorded
    # total_executions=1 and NeuronCore utilisation of 0.05%. `passes`
    # counted thousands of submissions; the device ran the graph once.
    sink = None
    passes = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        sink = kernel(source)
        xm.mark_step()
        passes += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

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
        "bytes_requested": bytes_requested,
        "analytic_gbps": analytic,
        "profiler_gbps": None,
        "score_method": "analytic",
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
        result.update(_profile(workdir))
    except profiler.ProfilerUnavailable as error:
        result["warning"] = f"profiler unavailable, Score is analytic: {error}"
        return result

    divergence = verify_against_analytic(result["profiler_gbps"], analytic)
    if divergence:
        result["warning"] = divergence
    return result


def _profile(workdir: str) -> dict:
    """Capture a profile and read the declared counters out of it."""
    neff = profiler.find_neff(workdir)
    session = os.path.join(workdir, "memory_read.ntff")
    counters = profiler.read_counters(neff, session)
    return {
        "profiler_gbps": profiler.bandwidth_gbps(counters, "read"),
        # Matches registry.PROFILER exactly, so "declared source" and
        # "method used" are string-comparable when they agree.
        "score_method": registry.PROFILER,
        "hbm_read_bytes": counters.get("hbm_read_bytes"),
        "profiler_total_time_s": counters.get("total_time"),
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


def verify_against_analytic(
    profiler_gbps: float, analytic_gbps: float, tolerance: float = 0.5
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
