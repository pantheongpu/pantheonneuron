# Where the cross-platform comparison holds, and where it does not

The registry's opening claim is that workload names match pantheongpu
"wherever the underlying concept is the same, so a Neuron result and a GPU
result for a given name are comparing like with like."

`registry.NOT_COMPARABLE_WITH_GPU` records the nine names where that broke
down in a specific, mechanical way: the **units** diverge and the join simply
fails. Eight are in pantheongpu's shared AI harness, which since v1.1.0
reports `ai-ops/s`; the ninth is `allocation_fragmentation`, `alloc-events/s`
there against `allocation-events/s` here. `tests/test_score_schema.py`
enforces exactly that reading — every entry's GPU unit must differ from its
Neuron one.

**This document is about a different category, now recorded in
`registry.SAME_UNIT_DIFFERENT_QUANTITY`: names where the unit matches, the
join succeeds, and the two numbers are not measuring the same thing.** A failed join is visible. A successful join
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
| `transformer_virus` | On NVIDIA: `wmma::mma_sync` on constant register-resident fragments, no memory traffic -- a Tensor Core burner | A whole transformer block on the Tensor Engine | **No**, see below |
| `memory_read` / `memory_write` | `bytes_transferred / seconds`, **whole GPU** | `hbm_read_bytes / total_time`, **one NeuronCore of two** | **No** -- same quantity, different scope; see below |

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

## `transformer_virus` was the arguable one, and is not comparable either

It is the only compute workload where pantheongpu uses genuine matrix
instructions, so the functional-unit objection does not apply, and this
section used to leave it there with two caveats: the matrix path sits behind
`PANTHEON_ENABLE_EXPERIMENTAL_WMMA` with a non-matrix fallback under the same
name, and the issued-versus-retired difference stands.

**Both caveats are about AMD.** The flag gates the MFMA and rocwmma paths. On
NVIDIA sm_70 and later -- which is what a Trainium-against-NVIDIA comparison
is about -- the kernel takes an unconditional `nvcuda::wmma` path. Read that
path and the question changes from *which unit* to *what work*:

```cpp
wmma::fill_fragment(a, __float2half(1.01f * sign));
wmma::fill_fragment(b, __float2half(0.99f * sign));
for (int i = 0; i < iters; i++) {
    #pragma unroll 16
    for (int j = 0; j < 16; j++) wmma::mma_sync(c, a, b, c);
    ...
}
```

Fragments filled with constants, multiplied in a loop that never touches
global memory, counted analytically. That is a Tensor Core issue-rate burner,
and the numbers say so: an A100 reads 297.7 TFLOPS, 95% of its 312 dense-FP16
peak. There is no attention, no softmax, no FFN -- nothing transformer about
it but the name.

This suite's `transformer_virus` runs an actual block: hidden 4096, 32 heads,
seq 2048, attention and FFN, through the compiler, read from
`effective_flops`. It reaches 53.3 TFLOPS on one Trainium1 core -- 56% of that
core's share of peak -- because a real block spends time outside the matmuls.

Same name, same unit, matrix units on both sides; one is peak MMA issue rate
and the other is transformer-block throughput. Now in
`registry.SAME_UNIT_DIFFERENT_QUANTITY`.

## Four AI workloads were in the wrong register

Found 2026-09-13 by tallying the units in pantheongpu's published reports
instead of reading the transcription in `tests/test_score_schema.py`.
`NOT_COMPARABLE_WITH_GPU` used to list twelve names, on the belief that
pantheongpu moved every AI workload to `ai-ops/s`. Four never moved:

| Workload | pantheongpu unit, every version | pantheongpu counts |
|---|---|---|
| `llm_decode` | `tokens/s` | blocks × threads × loops of a KV-cache gather kernel |
| `llm_prefill` | `prompt-tokens/s` | synthetic prompt-token iterations from loop geometry |
| `kv_cache_churn` | `cache-updates/s` | blocks × threads × loops of a hashed read/write kernel |
| `graph_replay` | `graph-steps/s` | blocks × threads × loops per launch of a captured graph |

Those are the unit strings this suite uses, so the four rows **joined**, and
compared loop geometry against tokens decoded and executions completed,
while the register said the join could not happen. They are
`SAME_UNIT_DIFFERENT_QUANTITY` entries now. `allocation_fragmentation` failed
the other way: pantheongpu's unit changed in v1.1.0, nothing declared it, and
its row stopped joining without a word.

Every test passed throughout, because the registry and the transcription it
was checked against came from the same belief. The tally is committed as
`data/validation-2026-09-13/pantheongpu-published-units.json` (7,326 reports,
per version), and the test now checks the transcription against it.

## The memory rows name different scopes, and "_agg" means two different things

Found 2026-09-22, and it matters more than anything above, because these are
the rows a publication would stand on. This table said `memory_read` and
`memory_write` compared cleanly. They measure the same *quantity* -- bytes
moved through HBM per second, both from measurement -- over different
*amounts of hardware*:

| Name | pantheongpu | pantheonneuron |
|---|---|---|
| `memory_read` | whole GPU, standard pattern | **one NeuronCore** of the device's two |
| `memory_write` | whole GPU, standard pattern | **one NeuronCore** of the device's two |
| `memory_read_agg` | whole GPU, **rail-to-rail** data pattern (`0x00000000`/`0xFFFFFFFF`, `--init_pattern rail_to_rail`) | **all NeuronCores**, bandwidth summed |
| `memory_write_agg` | whole GPU, **crosstalk** data pattern (`--init_pattern crosstalk`) | **all NeuronCores**, bandwidth summed |

In pantheongpu `_agg` is the same binary run with an aggressive data pattern
(`pantheon.py`: `"memory_read_agg": {"bin": "memory_read", "args":
["--init_pattern", "rail_to_rail"]}`). Here it is an aggregate across cores.
Same suffix, unrelated meanings.

So the join on (Test Name, Unit) pairs:

- **`memory_read`**: one Trainium1 core, 273 GB/s, against a whole A100,
  1,496 GB/s. The Neuron figure is about half its device's by construction.
  Now in `registry.SAME_UNIT_DIFFERENT_QUANTITY`, with `memory_write`.
- **`memory_read_agg`**: the whole Trainium1 device, 543.7 GB/s, against a
  whole A100 reading a stress pattern. The scope matches. Whether the pattern
  matters is answerable from pantheongpu's own data, since it runs both:

| GPU | `memory_read` | `memory_read_agg` | `memory_write` | `memory_write_agg` |
|---|---|---|---|---|
| A100-SXM4-40GB | 1496.2 | 1495.5 | 1475.4 | 1473.9 |
| H100 80GB HBM3 | 3044.0 | 3047.5 | 3172.3 | 3177.6 |
| L40S | 728.8 | 728.8 | 432.9 | 432.8 |
| RTX PRO 6000 | 1533.2 | 1533.2 | 1446.3 | 1446.3 |

(Medians over pantheongpu's published reports, tallied 2026-09-22.) The data
pattern moves bandwidth by under 0.3% on every part. So `memory_*_agg` joins
device against device and the result stands -- **by accident**, because two
unrelated meanings of `_agg` happen to converge on the same scope.

**What a device-level comparison should use**: Neuron `memory_*_agg` against
GPU `memory_*` (or `memory_*_agg`; the table shows it makes no difference).
That pairing crosses names, so no join on (Test Name, Unit) will ever produce
it, and the comparison tooling has to be told.

For the record, the per-device figures, measured 2026-09-21 across three
trn1.2xlarge, spread across hosts under 0.05%:

| Trainium1, whole device | GB/s | Share of 880.5 GB/s published |
|---|---|---|
| `memory_read_agg` | 543.7 | 61.8% |
| `memory_write_agg` | 537.2 | 61.0% |

## What has not been decided

Nothing here changes what joins. `NOT_COMPARABLE_WITH_GPU` means "the units
diverge" and these units do not, so forcing them into it would break both its
tests and its meaning.

The options:

1. **A second register** — taken, as `registry.SAME_UNIT_DIFFERENT_QUANTITY` —
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
