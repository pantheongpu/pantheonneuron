# Where the cross-platform comparison holds, and where it does not

The registry's opening claim is that workload names match pantheongpu
"wherever the underlying concept is the same, so a Neuron result and a GPU
result for a given name are comparing like with like."

`registry.NOT_COMPARABLE_WITH_GPU` records the twelve names where that broke
down in a specific, mechanical way: pantheongpu v1.0.19 collapsed its AI
workloads to a single `ai-ops/s`, so the **units** diverge and the join
simply fails. `tests/test_score_schema.py` enforces exactly that reading —
every entry must be a name whose GPU unit is `ai-ops/s`.

**This document is about a different category, which has no register: names
where the unit matches, the join succeeds, and the two numbers are not
measuring the same thing.** A failed join is visible. A successful join
between unlike quantities is not, which makes this the worse of the two.

Found 2026-09-08 while checking whether the pinned 8192³ shape was a
comparison contract worth preserving. It is not, and the reason is below.

## What each side actually computes

Read from `pantheongpu/kernels/*/`. Neuron figures are from the registry.

| Workload | pantheongpu | Neuron | Same quantity? |
|---|---|---|---|
| `tensor_virus` | `__hfma2` FMA chains on the FP16 vector lanes. No matrix. `ops = blocks × threads × loops × 12` | Dense systolic GEMM on the Tensor Engine, 8192³ bf16 | **No** |
| `int_virus` | Integer FMA chains. `ops = blocks × threads × loops × 32 × 4` | Dense uint8 GEMM on the Tensor Engine | **No** |
| `pulse_virus` | Duty-cycled scalar `fmaf` chains — FP32 | Duty-cycled dense **bf16** GEMM | **No** |
| `omni_virus` | `fp16_ops + fp32_ops + sfu_ops` per launch, summed analytically | Four engines in one dependent chain | **No** |
| `transformer_virus` | MFMA/WMMA matrix instructions — real matrix units | Transformer block on the Tensor Engine | **Closest**, see below |
| `memory_read` / `memory_write` | `bytes_transferred / seconds` | `hbm_read_bytes / total_time` | **Yes** |

Three differences stack on the first four rows:

**Different functional unit.** pantheongpu's `tensor_virus` drives the FP16
vector lanes, not the matrix units — its own README describes "half-precision
pipelines". Neuron's drives the systolic Tensor Engine. Comparing them puts
the GPU's *secondary* math path against Neuron's *primary* one, and the bias
has a direction: it flatters Neuron. For a suite whose value proposition is
independent validation, a comparison that favours the vendor it runs on is
the wrong way to be wrong.

**Different provenance.** pantheongpu counts operations it *issued*, derived
from occupancy and loop counts. Neuron reads `effective_flops`, a hardware
counter of what the engine *retired*. This suite's own README makes exactly
this distinction load-bearing: "the analytic figure counts arithmetic issued,
the counter counts what the Tensor Engine retired, and a matmul folded away
at compile time shows up as the gap between them." Measured on trn1 at 8192³
the gap is real — 26.24 TFLOPS analytic against 22.60 counted.

**Different problem.** `problem` exists so "a Score is only comparable across
platforms if both ran the same problem". pantheongpu's compute viruses pin no
shape at all; they size themselves from occupancy. So the pinned 8192³ is
matched against nothing, and lowering or raising it costs nothing in
comparability — which is what made the unroll fix a free choice.

## `transformer_virus` is the arguable one

It is the only compute workload where pantheongpu uses genuine matrix
instructions (`__builtin_amdgcn_mfma_f32_16x16x16f16`, rocwmma fragments), so
the functional-unit objection does not apply. Two caveats remain: the matrix
path sits behind `PANTHEON_ENABLE_EXPERIMENTAL_WMMA` and a header check, with
a non-matrix fallback that produces a number under the same name; and the
issued-versus-retired difference still stands.

## What has not been decided

Nothing here changes what joins. `NOT_COMPARABLE_WITH_GPU` means "the units
diverge" and these units do not, so forcing them into it would break both its
tests and its meaning.

The options, none of them taken:

1. **A second register** — `SAME_UNIT_DIFFERENT_QUANTITY` or similar —
   letting the comparison tooling render "not comparable" for a row that
   would otherwise join silently. Additive, and honest about what is known.
2. **Change the GPU kernels** so `tensor_virus` drives the matrix units, at
   which point three of these rows become genuinely comparable. Work in the
   other repo, and it changes what pantheongpu's published numbers mean.
3. **Accept the comparison** on the argument that a power virus is defined by
   "maximum sustainable throughput on the primary math units" and each
   platform should use its best path. Defensible, but then the GPU side is
   not using its best path, so the argument requires option 2 first.

This is a decision about what the two suites claim about each other, so it is
recorded here rather than made in a commit.
