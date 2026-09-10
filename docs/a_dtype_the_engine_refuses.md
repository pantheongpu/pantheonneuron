# "The engine supports X" is not a well-formed claim

**trn1.2xlarge, 2026-09-08 and 2026-09-10.**

A dtype reaches the Tensor Engine by one of two paths — a NKI kernel
calling `nc_matmul`, or a torch graph compiled by `neuronx-cc` — and the
two do not accept the same set. Saying a part "supports int8" without
saying which path asked is not a claim that can be checked.

## What each path accepts

| dtype | NKI (`nc_matmul`) | XLA (`neuronx-cc`) |
|---|---|---|
| bf16, fp16, fp32 | yes | yes |
| uint8 | yes | yes |
| **int8** | **refused** | yes |
| **fp8_e4m3** | untested | **refused** |

```
nc_matmul does not support stationary.dtype=int8          2026-09-08
[NCC_ESPP047] Data type F8E4M3FN is not supported         2026-09-10
```

int8 sits on one side of that split and fp8 on the other, which is why a
single table could not hold either finding correctly.

## Eight-bit is not a throughput win here

All five paths at 4096³, in one process so nothing differs but the
dtype, 20s each. Every product verified exact against all-ones
arithmetic.

| path | T-ops/s | vs bf16 |
|---|---|---|
| int8 → int32 | 18.46 | 0.26× |
| int8 direct | 18.45 | 0.26× |
| uint8 → int32 | 72.78 | 1.03× |
| bf16 | 70.38 | 1.00× |
| fp8_e4m3 | refused | — |

Two things follow, and both are backwards from what a workload named
"quantized" invites a reader to assume:

1. **int8 is the slowest arithmetic measured on this part**, not the
   fastest. Eight-bit here buys footprint and accuracy tradeoffs, never
   throughput.
2. **uint8 merely matches bf16.** There is no 8-bit speedup in either
   direction, because both operands are promoted to int32 and int32 is
   the width the arithmetic actually runs at.

`int8 direct` and `int8 -> int32` agreeing to three digits is the
evidence for that promotion. `quantized_gemm` wrote the conversion
explicitly, with a comment saying int8 accumulators overflow at K far
below 4096 — true, and irrelevant, because XLA had already inserted the
widening. The call is a no-op. It is kept, because it states the
accumulator width this kernel's correctness depends on at the point that
depends on it; the claim that it *prevents* the overflow is gone.

## Why the correctness check could not catch any of this

The kernel reads its output back and checks it: every element of a
4096-deep GEMM over ones must be exactly `4096 * 0.02 = 81.92`. It is.
**The product is correct in every row of that table**, including the two
that are 3.8× slower than they should be.

Correctness says the matmul happened. It says nothing about which path
ran it, and no output check ever could. What catches it is a second
quantity — five dtypes measured side by side, where one being 0.26× its
neighbour is a fact no single number could have carried.

This is the fifth defect in this repo found by one number disagreeing
with another, after the KV-cache scatter, the PCIe asymmetry, the
allocation-fragmentation drift and the attention/router rates.

## The first version of this document was wrong

It claimed fp8_e4m3 was an accepted operand. That came from a prose
comment transcribed into a table rather than from a run, and the probe
falsified it within the hour — which is the whole argument for the
tables existing, turned on the person who wrote them.

`kernels/tiling.py` now carries `NKI_OPERANDS`, `XLA_OPERANDS`,
`OPERAND_RATES_4096` and `OPERAND_REFUSALS` keyed by `(path, dtype)`.
Every refusal carries the date it was measured, and
`tests/test_score_schema.py` fails a refusal that does not — a claim
without a run behind it should not be able to sit in the table quietly.

## What it costs

There is no FP8 row and there will not be one until `neuronx-cc` accepts
the type. A cross-platform reader comparing against a part with a real
FP8 unit is comparing against a gap, not a slow result, and the dtype
travels with the Score in `problem` so that gap is visible.
