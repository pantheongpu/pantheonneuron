"""HBM streaming-write kernel for the ``memory_write`` workload.

Score: **GB/s**, from ``hbm_write_bytes / total_time`` as declared in the
registry.

The write path needs a different anti-elimination trick than the read path.
``memory_read`` reduces its loads so they have a consumer; here the hazard
is the reverse -- stores into a buffer nothing ever reads are dead, and the
compiler may drop them. The destination is therefore the kernel's returned
output, which cannot be eliminated.

The kernel loads exactly **one** tile and stores it across every row, so
read traffic is one tile while write traffic is the full buffer. That
asymmetry is deliberate: it keeps ``hbm_write_bytes`` clean, and it gives a
cheap sanity check -- if ``hbm_read_bytes`` comes back anywhere near
``hbm_write_bytes``, the kernel is not doing what it looks like.

STATUS: verified on inf2.xlarge 2026-09-07, but **not at the pinned size**.
The kernel ran and its destination check passed exactly (ratio 1.0) at 4 GiB
(255.1 GB/s) and 6 GiB (162.5 GB/s). The registry pins 8 GiB, and that does
not fit: a NeuronCore on this part has 16 GB, the destination is the whole
plan, and the runtime still holds the previous destination when the next is
allocated -- 8.59 GB requested against 8.099 GB resident, failing by about
the size of the model code. Releasing the reference, forcing collection and
syncing did not reclaim it in time.

So the pinned problem is unreachable on a 2-core, 32 GB part, which is both
parts this suite currently targets. Whether to lower the pin or to split the
destination across cores is a registry decision, not a kernel one, and is
left open rather than silently changed here.
"""

import gc
import os
import time
import typing

from . import nki_backend, profiler, registry, tiling


PARTITION = tiling.PARTITION
FREE_ELEMENTS = tiling.FREE_ELEMENTS
tile_plan = tiling.tile_plan


def _build_kernel(total_rows: int):
    """Import NKI and construct the kernel.

    Lazy so this module can be imported, and its byte accounting tested, on
    a machine with no Neuron toolchain.

    ``total_rows`` sizes the destination and is a Python int captured at
    trace time, not a tensor dimension. The source carries only the single
    tile the kernel broadcasts, so the destination's shape cannot be
    inferred from it -- see ``run`` for why the source must stay that small.
    """
    import neuronxcc.nki as nki  # type: ignore
    import neuronxcc.nki.language as nl  # type: ignore

    @nki.jit
    def memory_write_kernel(source):
        """Broadcast one tile across a large HBM buffer.

        Returned rather than written to a scratch buffer: a store whose
        destination is never read is dead code, and returning the
        destination is what keeps it alive.
        """
        _, free_size = source.shape

        destination = nl.ndarray(
            (total_rows, free_size), dtype=source.dtype, buffer=nl.shared_hbm
        )

        # One tile in, many tiles out.
        tile = nl.load(source[0:PARTITION, 0:free_size])

        for row in nl.affine_range(total_rows // PARTITION):
            nl.store(
                destination[row * PARTITION:(row + 1) * PARTITION, 0:free_size],
                value=tile,
            )
        return destination

    return nki, nl, memory_write_kernel


def run(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Execute the streaming write and return timing plus byte accounting."""
    nki_backend.require_toolchain()

    import torch_xla.core.xla_model as xm  # type: ignore

    plan = tile_plan(int(problem["bytes"]), str(problem["dtype"]))
    rows = plan["tiles"] * PARTITION
    _, _, kernel = _build_kernel(rows)

    workdir = os.environ.get("PANTHEON_NEURON_WORKDIR", "/tmp/pantheon_ccwork")
    os.makedirs(workdir, exist_ok=True)

    device = xm.xla_device()
    import torch  # type: ignore

    # The source is ONE tile, not the whole plan. The kernel reads a single
    # tile and broadcasts it, so a full-size source is memory the run never
    # touches -- and on inf2.xlarge 2026-09-07 it was fatal: the destination
    # is the plan's full 8 GiB, a full-size source is another 8 GiB, and a
    # NeuronCore has 16 GB. The allocation failed with NRT_RESOURCE
    # ("Not enough Neuron memory on core 0 for size=8589934592") before a
    # single byte was written. One tile is 256 KiB.
    source = torch.ones(
        (PARTITION, plan["free"]), dtype=tiling.torch_dtype(problem["dtype"]),
        device=device,
    )
    xm.mark_step()

    # Compile outside the timed region, and compile the graph the loop will
    # actually run. Holding the result changes the graph -- the output
    # becomes live at the mark_step() cut -- so a warm-up that discards it
    # compiles a *different* graph and leaves the real one to be built
    # inside the timed region. memory_read measured that on inf2.xlarge
    # 2026-09-07: a 45 s run reported 478 s and 0.0208 GB/s because a
    # seven-minute compile landed in the middle of the measurement.
    compile_started = time.time()
    warm = kernel(source)
    xm.mark_step()
    xm.wait_device_ops()
    # Release the warm-up's destination before the loop allocates its own.
    # Dropping the reference alone does not do it -- the buffer lives until
    # the tensor is finalised -- and even forcing collection was not enough
    # on inf2.xlarge 2026-09-07, where the next request for 8.59 GB met
    # 8.099 GB still resident on a 16 GB core. See the module docstring for
    # what that means for the pinned problem.
    warm = None
    gc.collect()
    xm.mark_step()
    xm.wait_device_ops()

    # Two constraints meet in this loop and pull in opposite directions.
    #
    # The barrier must be inside the timed region: xm.mark_step() queues work
    # and returns, and timing without waiting measures submission. On
    # trn1.2xlarge 2026-08-27 that reported 1636 GB/s against a true 264.
    #
    # The output must be live at the mark_step() cut, or XLA proves the graph
    # dead and skips the stores -- 14,513 GB/s on the same part, 17x its HBM,
    # while neuron-monitor recorded a single execution.
    #
    # But here the output IS the buffer, so holding one across the next call
    # means two full destinations resident at once, which the part cannot
    # afford. Releasing immediately after mark_step() satisfies both: the
    # graph had a live consumer when it was dispatched, and the runtime owns
    # the buffer from that point, so dropping our reference frees it for the
    # next pass rather than pruning the computation.
    passes = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        written = kernel(source)
        xm.mark_step()
        written = None
        passes += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    # One extra pass, kept, purely to read the destination back. It is
    # deliberately outside the timed region: it exists to prove the stores
    # landed, and its cost is not part of the bandwidth it is verifying.
    sink = kernel(source)
    xm.mark_step()
    xm.wait_device_ops()

    # The source is all ones and the kernel broadcasts one tile across every
    # row, so every element of the destination must be 1.0. Checking the LAST
    # row is the point: it is written by the final iteration of the store
    # loop, so a loop that exited early or was partly elided fails here. This
    # needs no profiler, which matters because the profiler being unavailable
    # is exactly when the analytic figure becomes the Score.
    write_verified = None
    if sink is not None:
        try:
            last_row = float(sink[plan["tiles"] * PARTITION - 1][0])
            first_row = float(sink[0][0])
        except Exception:  # materialisation failed; leave unverified
            last_row = first_row = None
        if last_row is not None and first_row is not None:
            write_verified = (first_row + last_row) / 2.0

    bytes_written = plan["actual_bytes"] * passes
    analytic = bytes_written / elapsed / 1e9

    result = {
        "passes": passes,
        "elapsed_s": elapsed,
        "bytes_written": bytes_written,
        "analytic_gbps": analytic,
        "profiler_gbps": None,
        "score_method": "analytic",
        "analytic_basis": "bytes moved / wall time",
        "warning": None,
        "plan": plan,
        # 1.0 means the first and last rows both hold the broadcast value.
        "write_verified_ratio": write_verified,
    }

    elided = verify_write_completed(write_verified)
    if elided:
        result["warning"] = elided
        return result

    try:
        result.update(_profile(workdir, compile_started, plan["actual_bytes"]))
    except profiler.ProfilerUnavailable as error:
        result["warning"] = f"profiler unavailable, Score is analytic: {error}"
        return result

    warnings = [
        verify_against_analytic(result["profiler_gbps"], analytic),
        verify_write_dominates_read(
            result.get("hbm_write_bytes"), result.get("hbm_read_bytes")
        ),
    ]
    found = [w for w in warnings if w]
    if found:
        result["warning"] = "; ".join(found)
    return result


def _profile(workdir: str, since: float, planned_bytes: int) -> dict:
    """Identify this kernel's graph among the compiler's NEFFs, and read it.

    ``since`` ranks the candidates and ``planned_bytes`` decides which one
    is actually ours. The plan check used to run once against a single
    guess and reject it; it now selects. See ``profiler.select_by_plan``.
    """
    candidates = profiler.find_neffs(workdir, since=since)
    session = os.path.join(workdir, "memory_write.ntff")
    found = profiler.select_by_plan(candidates, session, "write", planned_bytes)
    counters = found["counters"]
    return {
        "profiler_gbps": profiler.bandwidth_gbps(counters, "write"),
        "score_method": registry.PROFILER,
        "hbm_write_bytes": counters.get("hbm_write_bytes"),
        "hbm_read_bytes": counters.get("hbm_read_bytes"),
        "profiler_total_time_s": counters.get("total_time"),
        "profiler_neff": os.path.basename(found["neff"]),
        "profiler_plan_coverage": found["plan_coverage"],
        "profiler_candidates_tried": found["candidates_tried"],
        "profiler_candidates_available": found["candidates_available"],
    }


def verify_against_analytic(
    profiler_gbps: typing.Optional[float],
    analytic_gbps: float,
    tolerance: float = 0.5,
) -> typing.Optional[str]:
    """Flag a profiler figure that disagrees with the analytic one."""
    if analytic_gbps <= 0:
        return "analytic bandwidth is zero -- no bytes were written"
    if profiler_gbps is None:
        return None
    if profiler_gbps <= 0:
        return (
            "profiler reported no HBM writes while the kernel claimed "
            f"{analytic_gbps:.1f} GB/s -- the stores were probably eliminated"
        )
    ratio = profiler_gbps / analytic_gbps
    if ratio < (1 - tolerance) or ratio > (1 + tolerance):
        return (
            f"profiler {profiler_gbps:.1f} GB/s and analytic "
            f"{analytic_gbps:.1f} GB/s differ by more than {tolerance:.0%} "
            f"(ratio {ratio:.2f})"
        )
    return None


def verify_write_completed(
    write_verified_ratio: typing.Optional[float], tolerance: float = 0.01
) -> typing.Optional[str]:
    """Check the stores actually landed in the destination.

    ``verify_against_analytic`` and ``verify_write_dominates_read`` both need
    the profiler, and the profiler failing is precisely when the analytic
    figure becomes the Score -- so those nets have a hole exactly where it
    matters. This check closes it from the destination alone.

    A ratio of 1.0 means the first and last rows both hold the broadcast
    value. Anything else means the store loop did not cover the buffer the
    plan describes, so the byte count behind the analytic bandwidth is
    fiction.
    """
    if write_verified_ratio is None:
        return (
            "destination could not be read back -- write coverage unverified"
        )
    if abs(write_verified_ratio - 1.0) > tolerance:
        return (
            f"destination holds {write_verified_ratio:.3f}x the written value "
            "-- the stores were coalesced or eliminated, so the analytic "
            "bandwidth is not a measurement"
        )
    return None


def verify_write_dominates_read(
    write_bytes: typing.Optional[int],
    read_bytes: typing.Optional[int],
    min_ratio: float = 8.0,
) -> typing.Optional[str]:
    """The kernel loads one tile and stores many; writes must dominate.

    If read traffic approaches write traffic, the kernel is moving data it
    was not supposed to -- a re-read per store, say -- and the Score is
    measuring a mixed workload rather than a write.
    """
    if write_bytes is None or read_bytes is None:
        return None
    if write_bytes <= 0:
        return "profiler reported zero HBM writes"
    if read_bytes <= 0:
        return None
    ratio = write_bytes / read_bytes
    if ratio < min_ratio:
        return (
            f"write/read byte ratio {ratio:.1f} is below {min_ratio:.0f} -- "
            f"the kernel is reading {read_bytes:,} bytes against "
            f"{write_bytes:,} written, so this is not a pure write"
        )
    return None
