#!/usr/bin/env python3
"""Is the headline TFLOPS figure the part, or the kernel?

`tensor_virus` is this suite's compute number: a NKI kernel reporting
26.1 TFLOPS bf16 at 8192^3 on trn1.2xlarge. On 2026-09-10 a plain
`torch.matmul` compiled by neuronx-cc reached 66.3 TFLOPS at the same
shape, in the same process, with both products verified exact.

**2.53x.** The headline figure is a floor on what the device can do, and
a cross-platform reader comparing it against another accelerator's peak
is comparing against a ceiling this repo built.

**Closed the same day by the coalesced tiling, now the default.** One
session, 8192^3, trn1.2xlarge 2026-09-10: tensor_virus 70.42 TFLOPS
against torch.matmul's 66.25, every row-tile of the NKI product checked.
The tool stays, because the question does not go away: it is what says
whether the next change to the kernel kept it there.

That is a fact about the suite, not a bug report about Trainium, and it
belongs where anyone can re-run it:

    python tools/compare_matmul_paths.py
    SHAPE=4096 DURATION=10 python tools/compare_matmul_paths.py

The comparison is deliberately narrow. Both sides run the same shape, the
same dtype and the same duration in one process, because the earlier
version of this question -- 26.1 at 8192^3 against 70.4 at 4096^3 -- was
comparing two shapes as well as two implementations and could not have
settled anything.

Both sides verify their product. A fast kernel that computes the wrong
thing is not a fast kernel, and the ratio between a correct kernel and a
rounded one means nothing at all.

**Until 2026-09-13 the XLA half verified its warm-up, not its measurement.**
It read one element of the pass that compiled the graph, and read the timed
loop's product back without comparing it. The NKI half has always checked its
timed product at every row-tile. The XLA figures recorded before then are
throughput a warm-up pass was shown to compute correctly; the XLA half now
checks every element of its timed product, the same standard as the NKI half.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHAPE = int(os.environ.get("SHAPE", "8192"))
DURATION = int(os.environ.get("DURATION", "25"))

# Below this the two paths are the same speed and there is nothing to
# report. Above it, the headline figure understates the part.
MEANINGFUL_GAP = 1.2


def xla_matmul():
    """A matmul as a model would express it, lowered by neuronx-cc."""
    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    device = xm.xla_device()
    n = SHAPE
    lhs = torch.ones((n, n), dtype=torch.bfloat16, device=device)
    rhs = torch.ones((n, n), dtype=torch.bfloat16, device=device)
    xm.mark_step()
    # Compile only. This used to be where the product was verified, which
    # made "exact" a statement about the warm-up pass: the timed loop's own
    # product was read back and thrown away, so a loop the compiler elided or
    # a graph that went wrong after the first pass still reported exact. The
    # NKI side verifies its timed product, so the two halves were not held to
    # the same test.
    warm = torch.matmul(lhs, rhs)
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    sink = None
    passes = 0
    started = time.perf_counter()
    while time.perf_counter() < started + DURATION:
        sink = torch.matmul(lhs, rhs)
        xm.mark_step()
        passes += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    # The timed loop's last product, whole, after the clock has stopped.
    # All-ones operands make every element exactly K. bf16 carries 8 mantissa
    # bits, so a sum reaching 8192 is exact only if the accumulator is wider;
    # a single element (what this checked) cannot see a product that is right
    # at [0][0] and wrong elsewhere, where the NKI side checks every row-tile.
    host = sink.to("cpu").float()
    wrong = product_elements_wrong(host, n)

    return {
        "tflops": passes * 2 * n ** 3 / elapsed / 1e12,
        "passes": passes,
        "product": float(host[0][0]),
        "elements_wrong": wrong,
        "exact": wrong == 0,
    }


def product_elements_wrong(host, k: int) -> int:
    """How many elements of an all-ones product are not exactly ``k``.

    Split out so the arithmetic can be checked without a device: it takes any
    tensor on the host.
    """
    return int(((host - k).abs() >= 0.5).sum())


def nki_kernel():
    """The suite's own kernel, at its pinned tiling."""
    from kernels import tensor_virus

    result = tensor_virus.run(
        {"op": "matmul", "shape": [SHAPE] * 3, "dtype": "bf16"}, DURATION)
    ratio = result.get("product_verified_ratio")
    return {
        "tflops": result["analytic_tflops"],
        "passes": result["passes"],
        "product": ratio,
        "tiling": result.get("tiling"),
        # Both corners and every row-tile. The corners alone cannot see a
        # kernel that wrote the right value into the wrong rows -- see
        # tensor_virus.rows_in_wrong_place.
        "exact": (ratio is not None and abs(ratio - 1.0) < 0.01
                  and result.get("row_tiles_wrong") == 0),
    }


PATHS = (("NKI tensor_virus", nki_kernel),
         ("XLA torch.matmul", xla_matmul))


def main() -> int:
    print(f"{SHAPE}^3 bf16, {DURATION}s each, one process\n")
    print(f"{'path':>18} {'TFLOPS':>9} {'passes':>8} {'product':>14} "
          f"{'exact':>6}")

    results = {}
    for name, run in PATHS:
        try:
            result = run()
        except Exception as error:  # broad: a failure IS the result
            print(f"{name:>18}  FAILED  {type(error).__name__}: "
                  f"{str(error)[:110]}")
            sys.stdout.flush()
            continue
        results[name] = result
        print(f"{name:>18} {result['tflops']:>9.2f} {result['passes']:>8} "
              f"{result['product']!s:>14} {result['exact']!s:>6}")
        sys.stdout.flush()

    print("\nverdict")
    if len(results) < len(PATHS):
        print("  incomplete -- one path did not run, so there is no ratio")
        return 1

    nki = results["NKI tensor_virus"]
    xla = results["XLA torch.matmul"]

    if not (nki["exact"] and xla["exact"]):
        wrong = [n for n, r in results.items() if not r["exact"]]
        print(f"  INCORRECT PRODUCT: {wrong}")
        print("  The ratio means nothing until that is resolved.")
        return 1

    ratio = xla["tflops"] / nki["tflops"]
    print(f"  XLA is {ratio:.2f}x the NKI kernel at a matched shape.")
    if ratio > MEANINGFUL_GAP:
        print("  The suite's headline TFLOPS figure is a property of the "
              "hand-written kernel, not of the part. It is a floor, and "
              "must not be quoted as this device's bf16 throughput.")
    elif ratio < 1 / MEANINGFUL_GAP:
        print("  The NKI kernel is ahead, and the headline figure is the "
              "better estimate of the two.")
    else:
        print("  No meaningful gap: the headline figure and the compiler "
              "agree on what this part does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
