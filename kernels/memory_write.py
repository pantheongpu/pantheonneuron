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

STATUS: kernel UNTESTED, primitives verified. Every NKI call used here
(``nl.load``, ``nl.store``, ``nl.ndarray`` with ``shared_hbm``,
``nl.affine_range``, ``nl.par_dim``) was exercised on trn1.2xlarge on
2026-08-27 by ``memory_read``. This particular arrangement of them has not
run.
"""

import os
import time
import typing

from . import nki_backend, profiler, registry, tiling


PARTITION = tiling.PARTITION
FREE_ELEMENTS = tiling.FREE_ELEMENTS
tile_plan = tiling.tile_plan


def _build_kernel():
    """Import NKI and construct the kernel.

    Lazy so this module can be imported, and its byte accounting tested, on
    a machine with no Neuron toolchain.
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
        rows, free_size = source.shape

        destination = nl.ndarray(
            (rows, free_size), dtype=source.dtype, buffer=nl.shared_hbm
        )

        # One tile in, many tiles out.
        tile = nl.load(source[0:PARTITION, 0:free_size])

        for row in nl.affine_range(rows // PARTITION):
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
    _, _, kernel = _build_kernel()

    workdir = os.environ.get("PANTHEON_NEURON_WORKDIR", "/tmp/pantheon_ccwork")
    os.makedirs(workdir, exist_ok=True)

    device = xm.xla_device()
    import torch  # type: ignore

    rows = plan["tiles"] * PARTITION
    source = torch.ones(
        (rows, plan["free"]), dtype=tiling.torch_dtype(problem["dtype"]),
        device=device,
    )
    xm.mark_step()

    # Compile outside the timed region; a NEFF build is tens of seconds.
    kernel(source)
    xm.wait_device_ops()

    # xm.mark_step() queues work and returns without waiting for the device.
    # Measured on trn1.2xlarge 2026-08-27, omitting the barrier reported
    # 1636 GB/s against a true 264 GB/s, because elapsed time stayed
    # constant regardless of buffer size. The barrier belongs inside the
    # timed region.
    passes = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        kernel(source)
        xm.mark_step()
        passes += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    bytes_written = plan["actual_bytes"] * passes
    analytic = bytes_written / elapsed / 1e9

    result = {
        "passes": passes,
        "elapsed_s": elapsed,
        "bytes_written": bytes_written,
        "analytic_gbps": analytic,
        "profiler_gbps": None,
        "score_method": "analytic",
        "warning": None,
        "plan": plan,
    }

    try:
        result.update(_profile(workdir))
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


def _profile(workdir: str) -> dict:
    neff = profiler.find_neff(workdir)
    session = os.path.join(workdir, "memory_write.ntff")
    counters = profiler.read_counters(neff, session)
    return {
        "profiler_gbps": profiler.bandwidth_gbps(counters, "write"),
        "score_method": registry.PROFILER,
        "hbm_write_bytes": counters.get("hbm_write_bytes"),
        "hbm_read_bytes": counters.get("hbm_read_bytes"),
        "profiler_total_time_s": counters.get("total_time"),
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
