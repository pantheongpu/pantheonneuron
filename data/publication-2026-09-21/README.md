# Publication pass, 2026-09-21

The first multi-host dataset this suite has produced: every Score here comes
from more than one rented instance, so the spread between hosts is measured
rather than assumed.

## What ran

| Part | Hosts | Commit | Groups |
|---|---|---|---|
| trn1.2xlarge (Trainium1) | trn1-a, trn1-b, trn1-c | `3ae6426` | `300x3`, `3600`, `sweep` (trn1-a only), `3600-fixed` (trn1-c only) |
| inf2.xlarge (Inferentia2) | inf2-a, inf2-b | `3ae6426` | `300x3` |

All in us-east-1, launched 2026-09-21 20:29 UTC, on-demand. Host labels are
arbitrary; no instance identifier appears anywhere in this directory, and
`tests/test_report_privacy.py` checks that it stays that way.

- **`300x3`** -- every suite, one harness process per suite,
  `--duration 300 --repeat 3`. The variance run: three repeats on each host.
- **`3600`** -- the rows that can be compared with GPUs, one hour each, to
  match the duration of the GPU study: the memory suite, `pcie_bandwidth`,
  `transformer_virus`. One repeat.
- **`sweep`** -- `tools/sweep_memory_size.py`: `memory_read` and
  `memory_write` at 1, 2, 4, 8 and 12 GiB, 60 s x 2 repeats, through the full
  harness.
- **`3600-fixed`** -- the two aggregates rerun at 3600 s after the fix in #25;
  see *What failed*.

These reports predate the `invocation` block (#23), so they do not record the
flags they ran with. The group directory does, sorted by matching each
report's timestamp to the phase window its host logged; none was ambiguous.

## Summary

`summary-trn1.txt` and `summary-inf2.txt` are `tools/summarise_hosts.py` over
each part's hosts; the `.json` files are the same data. Two spreads per row,
kept apart: the worst within-host cv over repeats, and the cv of the hosts'
medians.

**The instance barely matters.** On trn1, 21 of 23 scored workloads agree
across the three hosts within 0.2%. On inf2, 19 of 22 agree across its two
hosts within 0.1% and all 22 within 0.5% (`allocation_fragmentation` 0.46%,
`pcie_bandwidth` 0.25%, `graph_replay` 0.24%). The trn1 exceptions:

| Workload | Between-host cv (trn1) | Why |
|---|---|---|
| `allocation_fragmentation` | 6.9% | A ~17 s count-bound window over the host's allocator path. One host ran two slow repeats (2019, 2126) against 2390-2470 elsewhere. Quote with its spread, never from one host. |
| `pcie_bandwidth` | 2.5% at 300 s, 8.6% at 3600 s | Host-side staging copies. Not comparable with GPUs anyway (#27). |

## What compares with NVIDIA, and what does not

A comparison joins Neuron and GPU rows on (Test Name, Unit). This pass was
the occasion for checking every row that joins against what the GPU side
actually computes. Most do not compare, each for a documented reason:

| Rows | Status | Why | Where |
|---|---|---|---|
| AI workloads | not comparable | GPU side: synthetic loop counts, often under `ai-ops/s` | #20, `docs/cross_platform_comparability.md` |
| `tensor_virus`, `int_virus`, `pulse_virus`, `omni_virus` | not comparable | Vector-lane FMA chains counted analytically, against a systolic GEMM read from a counter | `docs/cross_platform_comparability.md` |
| `transformer_virus` | not comparable | On NVIDIA: `wmma::mma_sync` on constant fragments with no memory traffic, against a whole transformer block | #29 |
| `pcie_bandwidth` | not comparable | Concurrent pinned copies summed, against sequential unpinned copies averaged | #27 |
| `memory_read`, `memory_write` | not comparable | GPU: a whole device. Here: one NeuronCore of two | #28 |
| **`memory_read_agg`, `memory_write_agg`** | **comparable** | Whole device on both sides | #28 |

**The fair device-level comparison is Neuron `memory_*_agg` against GPU
`memory_*`.** In pantheongpu `_agg` means a data-pattern stress variant of the
same whole-GPU kernel, which its own reports show moves bandwidth under 0.3%,
so either GPU row serves. The pairing crosses names, so no join on
(Test Name, Unit) will produce it.

| Part | Device read (GB/s) | Device write (GB/s) | Published peak (GB/s) | Read % | Write % |
|---|---|---|---|---|---|
| **Trainium1** | **543.7** | **537.2** | 880.5 | **61.8%** | **61.0%** |
| **Inferentia2** | **542.8** | **548.6** | 880.5 | **61.6%** | **62.3%** |
| A10 | 501.3 | 479.5 | 600 | 83.6% | 79.9% |
| A10G | 501.9 | 474.2 | *none published* | -- | -- |
| L4 | 253.0 | 237.6 | 300 | 84.3% | 79.2% |
| L40S | 728.8 | 432.9 | 864 | 84.4% | 50.1% |
| A100-SXM4-40GB | 1496.2 | 1475.4 | 1,555 | 96.2% | 94.9% |
| H100 PCIe | 1968.9 | 1934.2 | 2,000 | 98.4% | 96.7% |
| H100 80GB HBM3 | 3044.0 | 3172.3 | 3,350 | 90.9% | 94.7% |

**That table is generated, not typed.** `tools/compare_with_gpu.py --render`
prints it from `summary-{trn1,inf2}.json` (this pass) and
`gpu-reference.json` (pantheongpu's medians and each part's published peak,
tallied by `tools/tally_gpu_reference.py` from its report database), and
`tests/test_compare_with_gpu.py` fails if this file drifts from it. The
first version of this table was typed from medians computed in a shell,
which is the transcription failure #20 was about.

Both suites count decimal gigabytes (bytes / 1e9), as NVIDIA and AWS quote
them, so the percentages carry no GiB/GB error. The same tool prints the 21
names that join on (Test Name, Unit) and must not be compared, each with its
reason -- the honest "not comparable" the registers were written to make
possible, and which nothing rendered until now.

**Published peaks, each checked on 2026-09-22 against the vendor's own
document:**

| Part | Figure | Source |
|---|---|---|
| Trainium1, Inferentia2 | 820 GiB/s = 880.5 GB/s | AWS Neuron architecture docs; `registry.PART_PEAKS`. **AWS's pages disagree; see below.** |
| A10 | 600 GB/s | [NVIDIA A10 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/datasheet-new/nvidia-a10-datasheet.pdf) |
| L40S | 864GB/s | [NVIDIA L40S product page](https://www.nvidia.com/en-us/data-center/l40s/) |
| A100 40GB SXM | 1,555GB/s | [NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf), SXM 40GB column |
| H100 PCIe | 2,000 GB/s | [NVIDIA H100 PCIe product brief PB-11133](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf), Table 2 |
| H100 SXM | 3.35TB/s | [NVIDIA H100 product page](https://www.nvidia.com/en-us/data-center/h100/) |

**AWS's own pages disagree about the Neuron figure, by 7.4%.** The
architecture pages say "32 GiB of device memory (for storing model state),
with 820 **GiB**/sec of bandwidth". The NKI architecture guide says "2 HBM
stacks with a total device memory capacity of 32GiB and bandwidth of 820
**GB/s**". Both read 2026-09-22. 820 GiB/s is 880.5 GB/s, so the two
readings put Trainium1's measured 543.7 GB/s at either **61.8%** or
**66.3%** of peak.

This dataset uses the GiB reading, 880.5 GB/s, for three reasons: the
architecture page is the source `registry.PART_PEAKS` cites, it states GiB
in both of its copies, and it is the less flattering of the two -- where
the pages cannot both be right, the reading that does not flatter the
part is the one to publish. A third figure agrees with it: 16 DMA engines
per NeuronCore-v2 at 27.2 GB/s is 870 GB/s across the chip's two cores,
near 880.5 and above 820.

Anything published from this data should carry the range rather than one
number, because the discrepancy is AWS's and not ours to resolve.

Three things the NVIDIA checking turned up:

- **The A10G has no published memory bandwidth.** It is an AWS-specific
  part, not the A10: NVIDIA's documents cover the A10, and AWS's give only
  "24 GB of memory" per GPU. Its reports also show a 300 W power limit
  against the A10's 150 W, so the A10's 600 GB/s cannot be assumed to apply.
  The figure widely repeated for it is an inference, and is not used here.
- **H100 PCIe's brief disagrees with itself.** It prints 2,000 GB/s peak
  beside a 1,593 MHz clock and a 5,120-bit bus, which multiply to
  2,039 GB/s. The printed figure is used; against 2,039 the percentages are
  96.6% read and 94.9% write.
- **L40S writes reach half its published bandwidth.** 432.9 GB/s against
  864, while its reads reach 84.4%. That is a GPU-side finding, not a
  Trainium one, and it is the only row here where read and write diverge by
  more than a few points.

An earlier summary of this comparison put the NVIDIA parts at "81-98% of
datasheet" from figures recalled rather than checked. Checked, reads run
83.5-98.4%; writes run 50.1-96.7%, and the range was only ever right for
reads.

**Why Trainium1 sits at 62% where the NVIDIA parts reach 84-98%.** The
limit is not HBM: two NeuronCores read exactly twice what one does, so
nothing is contended at 543.7 GB/s. Every byte reaches HBM through the 16
DMA engines AWS documents per NeuronCore-v2 at 27.2 GB/s each -- 435 GB/s
per core, about 870 for the chip -- and one core reaches 60-63% of that
whatever the kernel does. Five hypotheses have been tested on hardware and
refuted: transfer width, a second concurrent load, the consumer engine
(all in `kernels/memory_read.py`), and on 2026-09-22 prefetch depth 2/4/8
and explicit `nisa.dma_copy`, with an HBM-to-HBM copy showing the on-chip
path is not the constraint either
(`data/validation-2026-09-22/trn1-dma-levers.log`). What remains is
per-engine efficiency or a core-to-HBM fabric limit, neither reachable from
a kernel. A GPU keeps thousands of requests in flight from its
multiprocessors; on Trainium the DMA engines are the only path, and AWS's
own guidance -- 128 partitions, 4 KiB or more per partition row -- was
already being followed.

**The pinned size does not bias it.** The sweep ran `memory_read` from 1 to
12 GiB: 273.07, 272.82, 272.84, 272.95, 273.04 GB/s. Flat. The 8 GiB pin
compares fairly against GPU runs that size themselves from free memory.

**Nor does the duration.** At 3600 s against 300 s on the same hosts:
`memory_read` 273.1 against 273.0, `memory_read_agg` 543.53 against 543.74
and `memory_write_agg` 536.79 against 537.25 (`3600-fixed`, both over the
full hour: 3600.07 and 3600.04 s). No host slowed over the hour; wall-clock over the profiled
burst stayed between 0.9902 and 1.0022 (#30).

## Trainium1 against Inferentia2

AWS publishes the same per-chip figures for both. Memory agrees closely --
single-core `memory_read` and `memory_write` within 0.1%, device
`memory_read_agg` within 0.2%, device `memory_write_agg` within 2.1% with
inf2 the higher -- and compute does not. inf2 sustains 0.723 of trn1's dense-GEMM rate,
because it applies a utilization-limit throttle far more deeply -- about a
0.63 limit where trn1 applies 0.875. `docs/inf2_sustains_less_than_trn1.md`
has the eight explanations ruled out and the profiler evidence.

## What failed

- **`memory_read_agg` and `memory_write_agg` at 3600 s, all three trn1 hosts.**
  "core 0 timed out; core 1 timed out": the aggregate's worker budget was a
  fixed 1800 s, so any run long enough to matter was killed mid-measurement.
  A harness bug, fixed in #25, and the reason `3600-fixed` exists. The failed
  rows are kept in `3600` as they were written.
- **`memory_write` at 8 and 12 GiB in the sweep.** "Not enough Neuron memory
  on core 0": a write's destination is still resident when the next pass
  allocates, so two full buffers must fit on a 16 GiB core -- the constraint
  the kernel documents, and why it pins 4 GiB. The sweep asked for sizes it
  cannot run; its defaults are fixed in #32.
- **`all_reduce`, `p2p_thrasher`**: skipped everywhere. They need a second
  Neuron device, and no part within the account's quota has one.

## Cost

About $69 of on-demand instance time -- 42.5 trn1.2xlarge-hours at
$1.34375 and 15.2 inf2.xlarge-hours at $0.7582, from launch to terminate --
against an approved cap of $100.
Every instance was terminated after its results were collected, the last at
11:34 UTC on 2026-09-22.
