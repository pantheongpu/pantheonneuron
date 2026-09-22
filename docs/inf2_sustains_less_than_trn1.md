# Inferentia2 sustains 72% of Trainium1's matmul rate, and the registry said they were the same

`registry.PART_PEAKS` carries the same compute ceiling for both parts, 190
BF16 TFLOPS per chip, with a comment that reads:

> **The two parts are the same silicon per chip.** They differ in how many
> chips an instance carries, not in what a chip does.

The first half is AWS's published figure for each part and is not in
question. The second half is a claim about behaviour, and the 2026-09-21
passes measured it directly: **three trn1.2xlarge and two inf2.xlarge, same
commit, same compiler and runtime build, 300 s per workload, three repeats.**

## What the two parts do

| Workload | trn1 median | inf2 median | inf2 / trn1 |
|---|---|---|---|
| `tensor_virus` (dense bf16 GEMM) | 78.60 TFLOPS | 56.85 TFLOPS | **0.723** |
| `int_virus` (dense uint8 GEMM) | 78.84 TOPS | 56.98 TOPS | **0.723** |
| `pulse_virus` (same GEMM, 50% duty) | 39.94 TFLOPS | 29.04 TFLOPS | **0.727** |
| `transformer_virus` | 53.25 TFLOPS | 44.42 TFLOPS | 0.834 |
| `omni_virus` | 54.01 TFLOPS | 42.42 TFLOPS | 0.785 |
| `memory_read` | 272.98 GB/s | 272.94 GB/s | **1.000** |
| `memory_write` | 274.82 GB/s | 274.90 GB/s | 1.000 |
| `memory_read_agg` | 543.74 GB/s | 542.99 GB/s | 0.998 |
| `llm_decode` | 17.54 tokens/s | 17.52 tokens/s | 0.999 |
| `speculative_decode` | 73.80 verified-tokens/s | 73.66 | 0.998 |

Memory is identical to three decimals. The three kernels that do nothing but
feed the Tensor Engine land on the *same* ratio, 0.723, and the latency-bound
workloads are unaffected. Whatever this is, it is specific to sustained
Tensor Engine work, and it is not a general "inf2 is slower" effect.

## What it is not

Each of these was checked rather than assumed, because the obvious
explanations are all measurement errors and would have meant a harness bug:

| Ruled out | Evidence |
|---|---|
| Counter error | `declared_over_kernel` is 1.0000 on inf2: the monitor's `effective_flops` and the kernel's own `flops_issued / perf_counter` agree. |
| Wrong results | `product_verified_ratio` 1.0 on both parts. |
| An idle core | `neuroncore_utilization` mean 99.27% on inf2, 99.26% on trn1. |
| Different toolchain | Both hosts: `neuronxcc 2.23.6484.0+3b612583`, `torch_neuronx 2.8.0.2.12.22436+0f1dac25`. |
| Part-specific code | No kernel branches on `arch`; the compile cache shows the same module hashes on both parts. |
| A slow host | inf2.xlarge has 4 vCPUs against trn1.2xlarge's 8, but one execution of the pinned 8192³ GEMM is ~19 ms of device work. |
| Thermal drift | The inf2 rate is flat across 300 s (mean 56.79, final-sample mean 56.85) and matches a 30 s run from 2026-09-11 (56.46). |
| A lower base clock | Cycle-derived and wall-clock bandwidth agree within 0.7% on both parts, so both cores were near 1.4 GHz while the DMA-bound kernels ran. |

## What it is

`neuron-profile` carries throttle counters that `neuron-monitor` does not
(see [neuron_counters.md](neuron_counters.md)). Capturing the compute graphs
on both inf2 hosts after their passes finished, with nothing else on the
device:

| Graph (module hash) | Tensor Engine active | `throttle_activity_1_active_time_nc0_percent` | `throttle_activity_1_avg_util_limit_nc0_percent` | `throttle_avg_util_limit_nc0_percent` |
|---|---|---|---|---|
| `MODULE_11435754646458375002` | 92.3% | 0.842 | 0.633 | 0.641 |
| `MODULE_707385231116392439` | 90.0% | 0.900 | 0.626 | 0.563 |
| `MODULE_13945462658086779221` | 90.0% | 0.900 | 0.626 | 0.564 |
| `MODULE_13478107821426688942` | 76.0% | 0.686 | 0.633 | 0.633 |

The five graphs whose Tensor Engine sat near idle carry no throttle activity
at all. **The device is holding the core to about a 63% utilization limit for
84-90% of the time a matmul graph runs.** Taken at face value that predicts
`0.9 x 0.63 + 0.1 x 1 = 0.67` of unthrottled throughput, against the 0.723
measured -- the right size and the right shape, from a counter that names
itself a utilization limit.

Both inf2 hosts agree, so it is not one machine.

## What is still open

**The trn1 comparison is not in yet.** These captures say inf2 throttles;
they do not yet say trn1 throttles less, and that is the claim the table at
the top implies. The trn1 hosts were still running their 3600 s passes when
this was written, and the same probe
(`throttle_probe.sh` in the run directory) will run on one of them before it
is terminated. Until then the cause is *consistent with* a utilization cap on
inf2 and not established.

**Cycles per execution turned out not to be a usable comparison.** Two
captures of the same module on the two inf2 hosts returned 83,411,352 and
273,732,448 cycles, so a single capture's cycle count is not a stable
per-execution figure and cannot be differenced against trn1's.

**The registry comment needs correcting either way.** Whatever the mechanism,
"not in what a chip does" is false as measured: the same code on the same
compiler reaches 72% of the rate on one part. `PART_PEAKS` itself stays as
it is -- 190 TFLOPS is what AWS publishes for both, and `Percent Of Peak` is
honest against the published figure. What it now shows is that inf2 delivers
60% of its advertised compute on dense matmul where trn1 delivers 83%.
