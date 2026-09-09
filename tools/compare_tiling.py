#!/usr/bin/env python3
"""Measure both GEMM tilings against each other, on hardware.

The streaming kernel re-reads every operand tile for every (row, col) pair,
so operand traffic scales as n^3 -- the same order as the FLOPs -- and
arithmetic intensity stays flat at about 102 FLOP/byte at any shape. At the
pinned 8192^3 that puts it on the HBM bandwidth ceiling rather than the
Tensor Engine's: measured operand traffic 256.4 GB/s against memory_read's
256.2 GB/s on the same part.

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

from kernels import tensor_virus  # noqa: E402

SHAPES = [int(s) for s in os.environ.get("SHAPES", "2048,4096,8192").split(",")]
DURATION = int(os.environ.get("DURATION", "15"))


def operand_traffic(plan, strategy: str) -> int:
    """Bytes of operand read per pass, as the tiling implies."""
    tiles = plan["m_tiles"] * plan["n_tiles"] * plan["k_tiles"]
    lhs = tiles * tensor_virus.CONTRACTION * tensor_virus.STATIONARY * 2
    rhs_reads = (plan["n_tiles"] * plan["k_tiles"] if strategy == "blocked"
                 else tiles)
    rhs = rhs_reads * tensor_virus.CONTRACTION * tensor_virus.MOVING * 2
    return lhs + rhs


def main() -> int:
    print(f"{'shape':>7} {'tiling':>10} {'TFLOPS':>9} {'ms/pass':>9} "
          f"{'GB/pass':>8} {'implied GB/s':>13} {'verified':>9}")
    results = {}
    for shape in SHAPES:
        for strategy in ("streaming", "blocked"):
            problem = {"op": "matmul", "shape": [shape] * 3,
                       "dtype": "bf16", "tiling": strategy}
            try:
                r = tensor_virus.run(problem, DURATION)
            except Exception as error:  # noqa: BLE001 - a failure is a result
                print(f"{shape:>7} {strategy:>10}   FAIL "
                      f"{type(error).__name__}: {str(error)[:110]}")
                sys.stdout.flush()
                continue

            ms = r["elapsed_s"] / r["passes"] * 1000 if r["passes"] else 0.0
            traffic = operand_traffic(r["plan"], strategy)
            implied = traffic / (ms / 1000) / 1e9 if ms else 0.0
            verified = r.get("product_verified_ratio")
            results[(shape, strategy)] = (r["analytic_tflops"], verified)
            print(f"{shape:>7} {strategy:>10} {r['analytic_tflops']:>9.2f} "
                  f"{ms:>9.3f} {traffic / 1e9:>8.2f} {implied:>13.1f} "
                  f"{str(verified):>9}")
            sys.stdout.flush()

    print("\nverdict")
    wrong = [k for k, (_, v) in results.items()
             if v is not None and abs(v - 1.0) > 0.01]
    if wrong:
        print(f"  INCORRECT PRODUCT: {wrong}")
        print("  A faster kernel that computes the wrong thing is not a "
              "faster kernel.")
        return 1

    for shape in SHAPES:
        pair = [results.get((shape, s)) for s in ("streaming", "blocked")]
        if not all(pair):
            continue
        streaming, blocked = pair[0][0], pair[1][0]
        change = blocked / streaming if streaming else 0.0
        verdict = "faster" if change > 1.05 else (
            "slower" if change < 0.95 else "no change")
        print(f"  {shape}^3: blocked is {change:.2f}x streaming ({verdict}), "
              f"both product-verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
