#!/usr/bin/env python3
"""Why does one NeuronCore read HBM at 273 GB/s and not 435?

`memory_read` reaches 273 GB/s on one NeuronCore, and two cores reach exactly
twice that -- 543.7 GB/s against the 880.5 GB/s AWS publishes for the chip,
measured on five hosts 2026-09-21. So HBM is not the limit; the limit is per
core. AWS documents what sits there: **16 DMA engines per NeuronCore-v2, each
with "a theoretical bandwidth of 27.2 GB/s"**, so 435 GB/s per core and about
870 for the chip, which is the published figure. 273 is 63% of 435.

Three explanations were already ruled out on hardware (see
kernels/memory_read.py): transfer width (16 KiB per partition reads 269.8, no
better than 4 KiB's 269.5, and 1 KiB costs 208), a second concurrent load per
iteration (268.8), and the consumer, whose vector engine sits at 65%.

This measures the levers that remain, each against the same 2 GiB of HBM, each
checking it actually moved the bytes. Every variant is independent: a compile
failure in one prints and the rest still run.

| variant | what it tests |
|---|---|
| `baseline` | the shipped kernel, to anchor against 273 GB/s |
| `prefetch_N` | N tiles in flight in an SBUF block before any reduction -- "invoke as many parallel DMA transfers as possible" |
| `dma_copy` | `nisa.dma_copy` instead of `nl.load`: the compiler's queue assignment is what one transfer per queue ID may be serializing |
| `hbm_to_hbm` | `nisa.dma_copy` HBM to HBM, no SBUF at all, so the partition layout cannot be the constraint. Counts both directions. |

Not tested: `dge_mode=hwdge` needs NeuronCore-v3 or newer, and this part is v2.

Bandwidth here is bytes over wall clock, not the profiler figure the harness
publishes, and this tool's baseline is not comparable with the published 273
GB/s: 2 GiB rather than 8, wall clock rather than the profiler, and one
output allocated per pass. Measured 2026-09-22 it reads 262.59. Compare
variants against that baseline, not against 273. A lever worth having has to
beat it by more than 1%, since profiler and wall clock agree to within that
on every host of the 2026-09-21 pass.

RESULT, trn1.2xlarge 2026-09-22: none of them does. prefetch 2/4/8 read
265.25, 264.88, 264.82 and dma_copy 265.25 -- 1% over baseline and flat in
depth. hbm_to_hbm moves 307.92 GB/s across both directions, so SBUF is not
the constraint. data/validation-2026-09-22/trn1-dma-levers.log has the
reasoning; the ceiling is per-core DMA throughput at about 60% of the
engines' rated aggregate.

    python tools/dma_levers.py                  # all variants, 2 GiB, 20 s each
    GIB=4 SECONDS=30 python tools/dma_levers.py
"""

import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kernels import nki_backend, tiling  # noqa: E402

PARTITION = 128
FREE = 2048               # 4 KiB per partition row in bf16: AWS's rule of thumb
DTYPE = "bf16"
PREFETCH_DEPTHS = (2, 4, 8)


def _tile_rows(total_bytes):
    plan = tiling.tile_plan(total_bytes, DTYPE, free=FREE)
    return plan["tiles"], plan["actual_bytes"]


def build_baseline():
    """The shipped kernel: one nl.load per iteration, reduced as it goes."""
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

    @nki.jit
    def kernel(source):
        partitions, free_size = source.shape
        total = nl.zeros((nl.par_dim(PARTITION), 1), dtype=nl.float32)
        out = nl.ndarray((nl.par_dim(PARTITION), 1), dtype=nl.float32,
                         buffer=nl.shared_hbm)
        for row in nl.affine_range(partitions // PARTITION):
            tile = nl.load(source[row * PARTITION:(row + 1) * PARTITION, 0:free_size])
            total += nl.sum(tile.astype(nl.float32), axis=1, keepdims=True)
        nl.store(out, value=total)
        return out

    return kernel


def build_prefetch(depth):
    """`depth` tiles loaded into one SBUF block before any is consumed.

    The point is how many DMA transfers are in flight. The shipped kernel
    reduces each tile before loading the next, so the reduction sits between
    consecutive transfers; here the loads are issued together.
    """
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

    @nki.jit
    def kernel(source):
        partitions, free_size = source.shape
        total = nl.zeros((nl.par_dim(PARTITION), 1), dtype=nl.float32)
        out = nl.ndarray((nl.par_dim(PARTITION), 1), dtype=nl.float32,
                         buffer=nl.shared_hbm)
        groups = partitions // (PARTITION * depth)
        for group in nl.affine_range(groups):
            block = nl.ndarray((depth, nl.par_dim(PARTITION), free_size),
                               dtype=source.dtype, buffer=nl.sbuf)
            for slot in nl.affine_range(depth):
                start = (group * depth + slot) * PARTITION
                block[slot] = nl.load(source[start:start + PARTITION, 0:free_size])
            for slot in nl.affine_range(depth):
                total += nl.sum(block[slot].astype(nl.float32), axis=1, keepdims=True)
        nl.store(out, value=total)
        return out

    return kernel


def build_dma_copy():
    """nisa.dma_copy into SBUF instead of nl.load."""
    import neuronxcc.nki as nki
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl

    @nki.jit
    def kernel(source):
        partitions, free_size = source.shape
        total = nl.zeros((nl.par_dim(PARTITION), 1), dtype=nl.float32)
        out = nl.ndarray((nl.par_dim(PARTITION), 1), dtype=nl.float32,
                         buffer=nl.shared_hbm)
        for row in nl.affine_range(partitions // PARTITION):
            tile = nl.ndarray((nl.par_dim(PARTITION), free_size),
                              dtype=source.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile,
                          src=source[row * PARTITION:(row + 1) * PARTITION, 0:free_size])
            total += nl.sum(tile.astype(nl.float32), axis=1, keepdims=True)
        nl.store(out, value=total)
        return out

    return kernel


def build_hbm_to_hbm():
    """HBM to HBM, never touching SBUF. Moves twice the bytes: read and write."""
    import neuronxcc.nki as nki
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl

    @nki.jit
    def kernel(source):
        partitions, free_size = source.shape
        out = nl.ndarray((partitions, free_size), dtype=source.dtype,
                         buffer=nl.shared_hbm)
        for row in nl.affine_range(partitions // PARTITION):
            lo, hi = row * PARTITION, (row + 1) * PARTITION
            nisa.dma_copy(dst=out[lo:hi, 0:free_size], src=source[lo:hi, 0:free_size])
        return out

    return kernel


def measure(name, kernel, source, moved_bytes, seconds, verify):
    """Run `kernel` for `seconds`, then say how fast and whether it was right."""
    import torch  # noqa: F401
    import torch_xla.core.xla_model as xm

    warm = kernel(source)
    xm.mark_step()
    xm.wait_device_ops()
    checked = verify(warm)
    del warm

    passes = 0
    started = time.perf_counter()
    deadline = started + seconds
    while time.perf_counter() < deadline:
        sink = kernel(source)
        xm.mark_step()
        passes += 1
        # Not a dead store: the output must be live at the mark_step cut or
        # XLA proves the graph dead and skips the DMA, and it must be released
        # before the next pass allocates its own. memory_write.py documents
        # both halves; this is the same discipline.
        sink = None  # noqa: F841
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started
    gbps = passes * moved_bytes / elapsed / 1e9
    return {"variant": name, "gbps": round(gbps, 2), "passes": passes,
            "elapsed_s": round(elapsed, 2), "correct": checked}


def main() -> int:
    gib = int(os.environ.get("GIB", "2"))
    seconds = int(os.environ.get("SECONDS", "20"))
    try:
        nki_backend.require_toolchain()
    except Exception as error:
        print(f"this tool needs a Neuron device and toolchain: {error}")
        return 2

    import torch
    import torch_xla.core.xla_model as xm

    rows, actual = _tile_rows(gib * 1024 ** 3)
    device = xm.xla_device()
    source = torch.ones((rows * PARTITION, FREE),
                        dtype=tiling.torch_dtype(DTYPE), device=device)
    xm.mark_step()
    xm.wait_device_ops()

    expected_sum = float(rows * FREE)     # every element is 1.0

    def sum_is_right(result):
        try:
            return abs(float(result.to("cpu").float().reshape(-1)[0]) - expected_sum) < 1.0
        except Exception:
            return None

    def copy_is_right(result):
        try:
            return float(result.to("cpu").float().reshape(-1)[0]) == 1.0
        except Exception:
            return None

    variants = [("baseline", build_baseline, actual, sum_is_right)]
    for depth in PREFETCH_DEPTHS:
        if rows % depth == 0:
            variants.append((f"prefetch_{depth}",
                             (lambda d=depth: build_prefetch(d)), actual, sum_is_right))
    variants += [
        ("dma_copy", build_dma_copy, actual, sum_is_right),
        ("hbm_to_hbm", build_hbm_to_hbm, actual * 2, copy_is_right),
    ]

    print(f"source {actual / 1024**3:.0f} GiB as {rows * PARTITION} x {FREE} {DTYPE}"
          f" ({FREE * 2 / 1024:.0f} KiB per partition row), {seconds}s per variant",
          flush=True)
    print(f"{'variant':14} {'GB/s':>8} {'passes':>7} {'elapsed':>8}  correct  note", flush=True)

    results = []
    for name, build, moved, verify in variants:
        try:
            started = time.perf_counter()
            kernel = build()
            row = measure(name, kernel, source, moved, seconds, verify)
            row["build_and_run_s"] = round(time.perf_counter() - started, 1)
            results.append(row)
            note = "moves 2x bytes (read+write)" if name == "hbm_to_hbm" else ""
            print(f"{name:14} {row['gbps']:>8.2f} {row['passes']:>7} "
                  f"{row['elapsed_s']:>8.2f}  {row['correct']!s:>7}  {note}", flush=True)
        except Exception as error:
            print(f"{name:14} {'-':>8} {'-':>7} {'-':>8}  {'-':>7}  "
                  f"failed: {type(error).__name__}: {str(error)[:160]}", flush=True)
            traceback.print_exc(limit=3)
            results.append({"variant": name, "error": f"{type(error).__name__}: {error}"[:400]})

    best = max((r for r in results if r.get("gbps") and r["variant"] != "hbm_to_hbm"),
               key=lambda r: r["gbps"], default=None)
    base = next((r for r in results if r["variant"] == "baseline" and r.get("gbps")), None)
    if best and base:
        print(f"\nbest SBUF-bound variant: {best['variant']} at {best['gbps']} GB/s, "
              f"{best['gbps'] / base['gbps']:.3f}x the baseline "
              f"({base['gbps']} GB/s). Per-core DMA theoretical is 435 GB/s "
              f"(16 x 27.2); baseline is {100 * base['gbps'] / 435:.1f}% of it, "
              f"best is {100 * best['gbps'] / 435:.1f}%.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
