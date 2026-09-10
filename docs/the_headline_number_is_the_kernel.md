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

It is a coincidence. The blocked tiling cuts operand traffic — 4.7× by
the model, **2.55× as measured** (see below) — and
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

## What the engines are doing

The three experiments above each changed the kernel and watched the rate.
This one reads the hardware. `neuron-profile` captured one execution of
each path's kernel graph at 4096³ bf16, trn1.2xlarge 2026-09-10, both
products verified exact:

| counter (0–1 fraction, shown as %) | NKI | XLA |
|---|--:|--:|
| **DMA active** | **59.1%** | **20.5%** |
| tensor engine active | 37.9% | 45.1% |
| GpSimd active | 8.9% | 0.1% |
| tensor engine instructions | 15,373 | 18,338 |
| time per execution | 5.74 ms | 3.96 ms |

**The profile agrees with the wall clock** — 1.45× slower per execution
by the profiler, 1.43× by rate in the same run — so these counters
describe the gap the rate measured rather than something beside it.

What they show: **the hand-written kernel spends most of its execution
moving data.** Its DMA is active 59% of the time against the compiler's
20%, while its tensor engine is active less (38% against 45%). The
compiler's matmul inverts the balance: little movement, more arithmetic.
The GpSimd engine — address generation and data movement that does not
map to the systolic array — is busy nearly ninety times as much.

That fits the streaming tiling, which re-reads every operand tile for
every (row, col) pair.

**It does not fit the blocked-tiling result, and that tension is left
standing rather than explained away.** Blocked tiling cut the modelled
operand traffic 4.7× *by the model* and bought 1.06×. If data movement
dominates, cutting
it should have helped more. Two readings would reconcile them — DMA time
set by transfer *count* and per-transfer overhead rather than by bytes,
or blocked tiling's remaining lhs stream being no more efficient — and
neither has been measured. `dma_transfer_total_bytes` is unreported for
the NKI graph, which is the counter that would distinguish them.

Two things worth keeping from how this was found:

- **The probe's own verdict was wrong.** It read the `_percent` counters
  as percentages, printed *"tensor engine active: NKI 0.379%"*, and
  concluded *"comparable; the gap is elsewhere."* They are 0–1 fractions
  — `mfu_max_achievable_estimated_percent` reads exactly 1 — and on the
  right scale they say the opposite. See `docs/neuron_counters.md`.
- **The gap depends on shape.** 1.43× at 4096³, 2.53× at 8192³. The
  hand-written kernel falls further behind as the problem grows, which is
  what a data-movement cost that scales faster than the compiler's would
  do.

## Counting the transfers

The profile above left `dma_transfer_count` empty for the NKI graph. The
detail is in the full trace: `neuron-profile view --output-format json`
writes a file called `ntff.json` **into the working directory** — not to
stdout, which a first probe read as an empty trace — and its top-level
`dma` list has one entry per transfer with its `transfer_size`,
`duration` and queue. One execution each, 4096³ bf16, trn1.2xlarge
2026-09-10, all three products exact:

| graph | transfers | MB moved | mean size | Σ DMA time | TFLOPS |
|---|--:|--:|--:|--:|--:|
| streaming | 247,017 | 631.3 | 2.56 KB | 33.3 ms | 36.2 |
| blocked | 174,942 | 248.0 | 1.42 KB | 14.5 ms | 38.3 |
| XLA | **34,031** | 236.2 | 6.94 KB | 10.1 ms | **51.8** |

Largest transfers: XLA made 128 over 64 KB. **Neither NKI tiling made
any.**

**The modelled traffic was wrong, and it had been quoted as measured.**
`tools/compare_tiling.operand_traffic` predicts 1,342 MB for streaming
and 302 MB for blocked at this shape, a 4.44× cut. The trace says 631
and 248 — a **2.55×** cut. The "4.7× less traffic for 1.06×" repeated
across this repo was the model's figure at 8192³, never a measurement.
The argument it supported survives (2.55× less traffic for 6% more speed
is still nowhere near proportional); the number did not.

**Whether transfer count or bytes limits the kernel was the wrong
question, because the answer is both, in turn:**

- **Streaming → blocked: bytes.** Bytes fell to 0.39×, DMA time to 0.43×
  — time follows bytes closely, and the transfer count (0.71×) much
  less so. Streaming's first problem is moving too much data.
- **Blocked → XLA: transfers.** At almost equal bytes (248 MB against
  236), blocked makes **5.1× as many transfers**, spends 1.43× the DMA
  time, and is 1.35× slower. Once the bytes are right, what is left
  lines up with transfer count and size: the compiler moves the same
  data in fewer, larger pieces.

That last point is a correlation with a plausible mechanism, not proof.
XLA's graph differs from the blocked kernel in more than transfer size —
scheduling, overlap and instruction mix too — and nothing here isolates
one. What the data does support is a direction: **the next version of this
kernel should coalesce its loads into larger DMA transfers**, and the
trace can tell whether it did.

One thing still unexplained: blocked cut summed DMA time 2.3× and ran
only 1.06× faster. Transfers on different queues overlap, so summed
duration overstates wall time — DMA *active* time fell only from 59% to
46% — but even so, the kernel did not get as much faster as its DMA got
lighter. DMA is not the whole critical path, and what is has not been
found.

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
