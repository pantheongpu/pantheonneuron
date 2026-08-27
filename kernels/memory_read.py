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

from . import nki_backend, profiler, registry


# NeuronCore-v2 tile geometry. The partition dimension is a hardware
# constant; the free dimension is chosen so one tile is a comfortable
# fraction of SBUF (24 MiB/core on v2) leaving room for double buffering.
PARTITION = 128
FREE_ELEMENTS = 2048
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "int8": 1}


def tile_plan(total_bytes: int, dtype: str) -> typing.Dict[str, int]:
    """Split a requested byte count into whole tiles.

    Returned ``actual_bytes`` is what the kernel will really read, which
    is ``requested`` rounded down to a whole number of tiles. The Score
    must be computed from the actual figure, never the requested one.
    """
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    element = DTYPE_BYTES[dtype]
    tile_elements = PARTITION * FREE_ELEMENTS
    tile_bytes = tile_elements * element

    tiles = total_bytes // tile_bytes
    if tiles < 1:
        raise ValueError(
            f"requested {total_bytes} bytes is smaller than one "
            f"{tile_bytes}-byte tile"
        )
    return {
        "tiles": tiles,
        "tile_bytes": tile_bytes,
        "actual_bytes": tiles * tile_bytes,
        "partition": PARTITION,
        "free": FREE_ELEMENTS,
        "element_bytes": element,
    }


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

    passes = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        kernel(source)
        xm.mark_step()
        passes += 1
    elapsed = time.perf_counter() - started

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
    }

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
