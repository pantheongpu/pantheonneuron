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

import os
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

    # The pinned shape, measured. This was capped at 2048 because 8192^3
    # "has never compiled" -- true when it was written, and untrue since
    # the accumulation loop was rolled (see tensor_virus.TILING). Measured
    # on trn1.2xlarge 2026-09-08:
    #
    #     tile   TFLOPS   setup
    #     2048    22.53       -
    #     4096    34.83     21s
    #     8192    48.13    120s
    #
    # So the cap was costing more than half the throughput as well as
    # making the row advertise a shape it did not run. `problem` is the
    # comparison contract, and the kernel now honours it.
    #
    # Two minutes of compile is the price, which is why the cap survives as
    # an override: a bring-up on a new part wants a shape that compiles in
    # seconds before it wants the pinned one.
    tile = min(m, int(os.environ.get("PANTHEON_NEURON_OMNI_TILE", m)))

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
        "pinned_shape": [m, n, k],
        # Says whether the row's Problem describes what ran.
        "ran_pinned_shape": tile == m == n == k,
        "flops_issued": flops,
        "analytic_tflops": flops / elapsed / 1e12 if elapsed else 0.0,
        "analytic_unit": "TFLOPS",
        "score_method": "analytic",
        "analytic_basis": "matmul FLOPs issued / wall time, chain stages excluded",
        **_shape_warning(tile, m, n, k, transformer_ops.output_check(
            transformer_ops.read_back(sink), "chain output")),
    }


def _shape_warning(tile: int, m: int, n: int, k: int,
                   checked: typing.Dict[str, typing.Any]
                   ) -> typing.Dict[str, typing.Any]:
    """Add the cut-down shape to whatever the output check already said.

    A row whose Problem reads 8192^3 while the kernel ran 2048^3 is
    advertising work it did not do, and a cross-platform comparison joining
    on that Problem would put a GPU's 8192^3 against a Neuron 2048^3.
    """
    if tile == m == n == k:
        return checked
    note = (
        f"ran {tile}^3, not the pinned {m}x{n}x{k} -- the row's Problem "
        "describes a larger shape than this Score measures"
    )
    existing = checked.get("warning")
    return {**checked,
            "warning": f"{existing}; {note}" if existing else note}


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
