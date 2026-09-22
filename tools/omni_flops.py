#!/usr/bin/env python3
"""What does omni_virus's chain actually issue, and why is its cross-check 12% out?

`omni_virus` drives four engines in one dependent chain: a matmul, elementwise
arithmetic, `tanh`, a `cumsum`, and a second matmul feeding back in. Its Score
is the monitor's `effective_flops`; its cross-check is the kernel's own
analytic count, and that count is **the two matmuls only**:

    flops = passes * 2 * (2 * tile ** 3)

On every host of the 2026-09-21 pass the monitor read 12% above it
(`declared_over_kernel` 1.1206 on trn1, 1.1111 on inf2) -- steady to four
digits, so it is structural, not noise. The Score is not in question; the
check that exists to catch a wrong Score is, because a cross-check that is
reliably 12% out cannot tell a 12% error from correct behaviour.

12% of `4 * tile**3` is about `0.5 * tile**3`, which no elementwise stage can
explain: the three non-matmul stages touch `tile**2` elements, six orders of
magnitude less. A `cumsum` lowered onto the Tensor Engine as a triangular
matmul would be exactly that order. This measures instead of assuming: it
captures the graph with `neuron-profile` and prints what the hardware counted
next to each candidate formula.

    python tools/omni_flops.py              # pinned 8192, plus 2048 for shape scaling
    TILES=2048,4096 python tools/omni_flops.py

Needs a reserved core for the capture, which the harness normally provides;
this sets it itself.
"""

import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kernels import cores, nki_backend, profiler, tiling  # noqa: E402


def candidates(tile):
    """Analytic FLOP counts for one pass of the chain, by hypothesis.

    Each is what the chain issues if that hypothesis about the compiler's
    lowering is right. The hardware's own count decides between them.
    """
    t = float(tile)
    two_matmuls = 2 * (2 * t ** 3)                   # what the kernel counts today
    elementwise = 2 * t ** 2                         # mul + add
    activation = t ** 2                              # tanh, one op per element
    cumsum_sequential = t ** 2                       # (tile-1) adds per row
    cumsum_triangular_macs = t ** 3 / 2              # lowered to a triangular matmul
    return {
        "two_matmuls (current)": two_matmuls,
        "+ elementwise + tanh + sequential cumsum": (
            two_matmuls + elementwise + activation + cumsum_sequential),
        "+ cumsum as triangular MACs": (
            two_matmuls + elementwise + activation + cumsum_triangular_macs),
        "+ cumsum as triangular FLOPs (2x MACs)": (
            two_matmuls + elementwise + activation + 2 * cumsum_triangular_macs),
        "three full matmuls": 3 * (2 * t ** 3),
    }


def build_chain(tile, dtype):
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    lhs = torch.full((tile, tile), 1.0 / tile, dtype=dtype, device=device)
    rhs = torch.ones((tile, tile), dtype=dtype, device=device)
    xm.mark_step()

    def chain():
        state = torch.matmul(lhs, rhs)
        state = state * 1.0001 + 0.5
        state = torch.tanh(state.float()).to(dtype)
        reduced = torch.cumsum(state.float(), dim=-1).to(dtype)
        return torch.matmul(reduced, rhs)

    return chain


def capture(workdir, since):
    """Read the counters of the newest graph compiled into `workdir`."""
    session = os.path.join(workdir, "omni_flops.ntff")
    found = []
    for neff in sorted(glob.glob(os.path.join(workdir, "**", "*.neff"), recursive=True),
                       key=os.path.getmtime, reverse=True):
        if os.path.getmtime(neff) < since:
            continue
        try:
            found.append((neff, profiler.read_counters(neff, session)))
        except Exception as error:
            print(f"    capture failed for {os.path.basename(os.path.dirname(neff))}: "
                  f"{str(error)[:120]}")
    return found


def main() -> int:
    tiles = [int(t) for t in os.environ.get("TILES", "2048,8192").split(",")]
    if nki_backend.mock_mode():
        print("this tool needs a real Neuron device; the mock backend has no "
              "graphs to capture")
        return 2
    try:
        nki_backend.require_toolchain()
    except Exception as error:
        print(f"this tool needs a Neuron device and toolchain: {error}")
        return 2

    import torch_xla.core.xla_model as xm

    os.environ.setdefault(cores.RESERVED_CORE, "1")
    os.environ.setdefault(cores.VISIBLE_CORES, "0")

    for tile in tiles:
        print(f"\n######## tile {tile}", flush=True)
        workdir = cores.kernel_workdir(f"omni_flops_{tile}")
        since = time.time()
        with cores.compile_cache(workdir):
            chain = build_chain(tile, tiling.torch_dtype("bf16"))
            sink = chain()
            xm.mark_step()
            xm.wait_device_ops()
            sink = None  # noqa: F841
        print("  candidate analytic counts, one pass:", flush=True)
        for name, value in candidates(tile).items():
            print(f"    {value:>18.0f}  {name}", flush=True)
        for neff, counters in capture(workdir, since):
            module = os.path.basename(os.path.dirname(neff))
            keys = ("model_flops", "hardware_flops", "transpose_flops",
                    "tensor_engine_instruction_count", "tensor_engine_active_time_percent",
                    "vector_engine_active_time_percent", "scalar_engine_active_time_percent",
                    "gpsimd_engine_active_time_percent", "neuroncore_cycle_count",
                    "mfu_estimated_percent")
            print(f"  {module}:", flush=True)
            for key in keys:
                if counters.get(key) is not None:
                    print(f"    {key} = {counters[key]}", flush=True)
            model = counters.get("model_flops")
            if model:
                print("    ratios of hardware's model_flops to each candidate:", flush=True)
                for name, value in candidates(tile).items():
                    print(f"      {model / value:6.4f}  {name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
