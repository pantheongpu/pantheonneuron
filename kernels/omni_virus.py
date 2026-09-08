"""All engines at once: tensor, vector, scalar and GpSimd.

Score: **TFLOPS**, from ``mean(effective_flops) / 1e12``, with per-engine
``active_time_percent`` recorded alongside as the registry declares.

Every other compute workload leans on one engine. `tensor_virus` saturates
the Tensor Engine and leaves the rest idle; `fused_attention` adds softmax
but is still matmul-dominated. This one interleaves work for all four in a
single dependent chain, which is the condition that finds two failures the
single-engine tests cannot:

- a power ceiling reached only when every engine draws at once, where each
  alone stays inside budget
- contention for SBUF and the internal buses, which is invisible when only
  one consumer is active

The chain is dependent on purpose. Independent streams would let the
compiler schedule the engines in sequence and quietly turn a concurrency
test into four consecutive single-engine tests.

Engine attribution is the compiler's, not ours: which operation lands on
which engine is its decision, and the per-engine counters in the report are
what actually happened rather than what this file intended. The comments
name the intended engine so a divergence is visible rather than assumed.

STATUS: UNTESTED ON HARDWARE. The per-engine counters exist on inf2 (108
counters) but the trn1 set is smaller (90) and omits at least one throttle
counter, so a missing engine reading is expected on Trainium rather than a
fault.
"""

import time
import typing

from . import nki_backend, tiling, transformer_ops


ENGINE_COUNTERS = (
    "tensor_engine_active_time_percent",
    "vector_engine_active_time_percent",
    "scalar_engine_active_time_percent",
    "gpsimd_engine_active_time_percent",
)


def run(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Drive all four engines in one dependent chain."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    m, n, k = (int(value) for value in problem["shape"])
    dtype = tiling.torch_dtype(str(problem["dtype"]))

    # The pinned shape is 8192^3, which as a dense matmul is the same graph
    # tensor_virus has never compiled. Here the matmul is one link in a
    # chain rather than the whole workload, so the tile is cut down and the
    # chain length carries the load instead. The Score is the monitor's
    # reading either way, so this changes what is issued, not how it is
    # measured.
    tile = min(m, 2048)

    device = xm.xla_device()
    lhs = torch.ones((tile, tile), dtype=dtype, device=device)
    rhs = torch.ones((tile, tile), dtype=dtype, device=device)
    xm.mark_step()

    def chain():
        # Tensor engine: the matmul.
        state = torch.matmul(lhs, rhs)
        # Vector engine: elementwise arithmetic over the result.
        state = state * 1.0001 + 0.5
        # Scalar engine: transcendentals, which is where GELU and the
        # softmax exponential live.
        state = torch.tanh(state.float()).to(dtype)
        # GpSimd: reductions and data movement that do not map to the
        # systolic array.
        reduced = torch.cumsum(state.float(), dim=-1).to(dtype)
        # Fed back in, so the next link cannot be hoisted above this one.
        return torch.matmul(reduced, rhs)

    warm = chain()
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    sink = None
    passes = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        sink = chain()
        xm.mark_step()
        passes += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    # Two matmuls per link. The elementwise, transcendental and reduction
    # stages are not counted: they are a small fraction of the arithmetic
    # and counting them as FLOPs would inflate a figure whose whole purpose
    # is to be compared against the monitor's.
    flops = passes * 2 * (2 * tile * tile * tile)

    return {
        "passes": passes,
        "elapsed_s": elapsed,
        "tile": tile,
        "flops_issued": flops,
        "analytic_tflops": flops / elapsed / 1e12 if elapsed else 0.0,
        "analytic_unit": "TFLOPS",
        "score_method": "analytic",
        "analytic_basis": "matmul FLOPs issued / wall time, chain stages excluded",
        "warning": transformer_ops.verify_output_is_a_number(
            transformer_ops.read_back(sink), "chain output"),
    }


def engine_activity(metrics: typing.Mapping[str, typing.Any]) -> typing.Dict[str, float]:
    """Pull whatever per-engine activity the monitor reported.

    Returns only the counters actually present. inf2 exposes 108 counters
    and trn1 exposes 90, so a missing engine here is a property of the part
    rather than a failed run -- and inventing a zero for it would read as an
    idle engine instead of an unreported one.
    """
    found = {}
    for counter in ENGINE_COUNTERS:
        value = metrics.get(counter)
        if isinstance(value, (int, float)):
            found[counter] = float(value)
    return found
