#!/usr/bin/env python3
"""Measure both GEMM tilings against each other, on hardware.

The streaming kernel re-reads every operand tile for every (row, col) pair,
so operand traffic scales as n^3 -- the same order as the FLOPs -- and
arithmetic intensity stays flat at about 102 FLOP/byte at any shape. At the
pinned 8192^3 that appears to put it on the HBM bandwidth ceiling rather
than the Tensor Engine's: measured operand traffic 256.4 GB/s against
memory_read's 256.2 GB/s on the same part.

**That reading is wrong, and this tool is what falsified it.** The blocked
kernel cuts operand traffic 4.7x -- modelled; neuron-profile measured
2.55x at 4096^3 on 2026-09-10 -- and buys 1.06x. A kernel genuinely against
a bandwidth wall does not behave that way, so the two figures agreeing to
0.1% is a coincidence -- a very persuasive one, which is why it survived as
an explanation until something measured against it.

What the ceiling actually is remains unknown, and it matters more than the
tiling question: at a matched 8192^3 a plain torch.matmul reaches 66.3
TFLOPS against this kernel's 26.2, so the suite's headline compute figure
is 2.53x below what the part does through the compiler. See
tools/compare_matmul_paths.py and
docs/the_headline_number_is_the_kernel.md.

The coalesced tiling closed that gap on 2026-09-10 -- 70.42 TFLOPS at
8192^3 against torch.matmul's 66.25 -- by loading the lhs four tiles wide.
Operand traffic did not change much (269.5 MB against blocked's 248.0 at
4096^3); the number of lhs load instructions per matmul did.

The blocked kernel holds one column's rhs tiles in SBUF across the row loop,
so each is read once per column instead of once per (row, col).

Two things have to be true before the faster one is worth having, and this
checks both:

1. **It computes the same product.** Both kernels are run over all-ones
   operands, where every output element must equal K exactly. A kernel that
   is fast and wrong is worse than one that is slow and right.
2. **It is actually faster.** Same shape, same duration, both figures
   printed side by side with the traffic each implies.

    bash tools/validate_hardware.sh          # runs the default tiling
    python tools/compare_tiling.py           # runs both and compares
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels import tensor_virus

SHAPES = [int(s) for s in os.environ.get("SHAPES", "2048,4096,8192").split(",")]
DURATION = int(os.environ.get("DURATION", "15"))


def operand_traffic(plan, strategy: str) -> int:
    """Bytes of operand read per pass, as the tiling implies."""
    tiles = plan["m_tiles"] * plan["n_tiles"] * plan["k_tiles"]
    lhs = tiles * tensor_virus.CONTRACTION * tensor_virus.STATIONARY * 2
    # coalesced holds the same rhs block blocked does; what it changes is
    # how the lhs is loaded, not how much of it.
    rhs_reads = (plan["n_tiles"] * plan["k_tiles"]
                 if strategy in ("blocked", "coalesced") else tiles)
    rhs = rhs_reads * tensor_virus.CONTRACTION * tensor_virus.MOVING * 2
    return lhs + rhs


def main() -> int:
    print(f"{'shape':>7} {'tiling':>10} {'TFLOPS':>9} {'ms/pass':>9} "
          f"{'GB/pass':>8} {'implied GB/s':>13} {'verified':>9}")
    results = {}
    for shape in SHAPES:
        for strategy in tensor_virus.STRATEGIES:
            problem = {"op": "matmul", "shape": [shape] * 3,
                       "dtype": "bf16", "tiling": strategy}
            try:
                r = tensor_virus.run(problem, DURATION)
            except Exception as error:  # broad: a failure IS the result
                print(f"{shape:>7} {strategy:>10}   FAIL "
                      f"{type(error).__name__}: {str(error)[:110]}")
                sys.stdout.flush()
                continue

            ms = r["elapsed_s"] / r["passes"] * 1000 if r["passes"] else 0.0
            traffic = operand_traffic(r["plan"], strategy)
            implied = traffic / (ms / 1000) / 1e9 if ms else 0.0
            verified = r.get("product_verified_ratio")
            # score_invalid covers the corners, every row-tile, and a
            # product that could not be read at all.
            results[(shape, strategy)] = (r["analytic_tflops"],
                                          r.get("score_invalid", True))
            print(f"{shape:>7} {strategy:>10} {r['analytic_tflops']:>9.2f} "
                  f"{ms:>9.3f} {traffic / 1e9:>8.2f} {implied:>13.1f} "
                  f"{verified!s:>9}")
            sys.stdout.flush()

    print("\nverdict")
    wrong = [k for k, (_, invalid) in results.items() if invalid]
    if wrong:
        print(f"  INCORRECT PRODUCT: {wrong}")
        print("  A faster kernel that computes the wrong thing is not a "
              "faster kernel.")
        return 1

    for shape in SHAPES:
        base = results.get((shape, "streaming"))
        if not base:
            continue
        for strategy in tensor_virus.STRATEGIES[1:]:
            other = results.get((shape, strategy))
            if not other:
                continue
            change = other[0] / base[0] if base[0] else 0.0
            verdict = "faster" if change > 1.05 else (
                "slower" if change < 0.95 else "no change")
            print(f"  {shape}^3: {strategy} is {change:.2f}x streaming "
                  f"({verdict}), both product-verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
