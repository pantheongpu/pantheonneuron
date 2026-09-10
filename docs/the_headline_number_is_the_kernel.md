# The headline number is the kernel, not the part

**trn1.2xlarge, 2026-09-10.** 8192³ bf16, 25s each, one process, both
products verified exact against all-ones arithmetic.

| path | TFLOPS | passes | product |
|---|--:|--:|---|
| NKI (`tensor_virus`) | 26.19 | 598 | ratio 1.0 |
| XLA (`torch.matmul`) | 66.32 | 1510 | 8192.0 |

**A plain `torch.matmul` compiled by `neuronx-cc` is 2.53× this suite's
own compute kernel**, at the same shape, in the same dtype, in the same
process.

## Why this matters more than a tuning opportunity

`tensor_virus` is the headline compute figure of the whole suite. It is
what a cross-platform comparison joins on, and the premise of that
comparison is that both sides ran the same problem on comparable terms.

That premise does not survive a 2.53× gap. Quoting 26.1 TFLOPS against
another accelerator's peak compares a compiler-generated matmul on one
side against a hand-written kernel on the other — and reports the
difference as a property of the silicon.

So the figure is a **floor**, not a capability. It is a real, correct,
sustained measurement of what this kernel does on this part; it is not
what this part does.

## How it was found

The dtype probe of 2026-09-10 measured a bf16 `torch.matmul` at 70.38
TFLOPS at 4096³, incidentally, while answering a different question. That
was 2.7× the headline figure — but at a different shape and through a
different implementation, so it was two variables and settled nothing.

Running both at a matched 8192³ removed one of them, and the gap
survived at 2.53×.

This is the second time in this repo that a number measured for one
purpose has contradicted a number published for another, and the reason
both were catchable is that they were put side by side. A rate on its own
cannot be wrong.

## What the cause is not

**Operand bandwidth.** That was the first diagnosis and it is wrong. The
streaming tiling re-reads every operand tile per (row, col), so operand
traffic scales as n³ — the same order as the FLOPs — and arithmetic
intensity stays flat near 102 FLOP/byte at any shape. 102 × `memory_read`'s
measured 256.2 GB/s is 26.1 TFLOPS, which lands on the observed figure
almost exactly.

It is a coincidence. The blocked tiling cuts operand traffic 4.7× and
buys **1.06×**. A kernel genuinely against a bandwidth wall does not
behave that way.

**Nor is it the k-loop's serialisation.** That was the third hypothesis,
and it had a mechanism: the innermost loop uses `nl.sequential_range`,
which declares a loop-carried dependency, so a scheduler that took that
literally could not overlap the next iteration's `nl.load` with this
iteration's `nl.matmul`. Testing it means running the same kernel with
`nl.affine_range` at a shape small enough that the full unroll compiles.

trn1.2xlarge 2026-09-10, 2048³ bf16, one process, both products exact:

| k-loop | TFLOPS | passes |
|---|--:|--:|
| `sequential_range` | 23.67 | 16,532 |
| `affine_range` (fully unrolled) | 23.51 | 16,419 |

**0.99×.** Unrolling the loop entirely changes nothing, so the dependency
is not costing anything and the scheduler was never the constraint.

A third variant — two accumulators over interleaved k tiles, to halve the
dependency chain while staying rolled — did not compile:

```
[NCC_IBVF027] Instruction can only read one of its non-scalar inputs
from PSUM, but inputs 0, 1 are read from PSUM
```

Two PSUM tensors cannot be added directly; one has to be copied to SBUF
first. Worth recording for anyone writing NKI here, and it means the
split-accumulator idea costs a copy it was not budgeted for.

So the ceiling is somewhere none of the three experiments has looked, and
this document does not claim to know where. Saying "2.53× slower, cause
unknown" is worth more than a fourth confident diagnosis — the first
three were each believed because a plausible mechanism agreed with them,
and each was wrong.

The shape is not the story either: this kernel reaches 23.67 TFLOPS at
2048³ and 26.19 at 8192³, so it is within 10% of its own ceiling across a
64× range of problem sizes while `torch.matmul` is 2.5× above it at both.

## What is settled

The kernel is **correct and slow**, which is the right way round. Its
product verifies exactly at every shape tested, on both parts. Nothing
here is a reason to distrust the arithmetic; it is a reason not to quote
the throughput as the device's.

## Keeping this from going stale

`tools/compare_matmul_paths.py` re-runs the comparison:

```bash
python tools/compare_matmul_paths.py
```

Both sides verify their product before the ratio is computed, because a
ratio between a correct kernel and a rounded one means nothing. If the
NKI kernel is ever tuned, this is the tool that says by how much — and
until then it is the tool that stops the 26.1 figure being quoted as
Trainium's bf16 throughput.
