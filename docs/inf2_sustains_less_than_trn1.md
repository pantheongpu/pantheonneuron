# Inferentia2 sustains 72% of Trainium1's matmul rate because it throttles harder

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

## trn1 throttles too, and less -- by the right amount

The same probe on all three trn1.2xlarge hosts after their passes, same
graphs by module hash. The two parts carry the same throttle mechanism and
apply it to very different depths. `throttle_avg_util_limit_nc0_percent`:

| Graph (module hash) | FLOPs / execution | inf2-a | inf2-b | trn1-a | trn1-b | inf2 / trn1 |
|---|---|---|---|---|---|---|
| `MODULE_707385231116392439` | 2^40 = 2 x 8192^3 | 0.563 | 0.563 | 0.766 | 0.765 | **0.736** |
| `MODULE_13945462658086779221` | 8192^3 | 0.564 | 0.564 | 0.762 | 0.761 | 0.741 |
| `MODULE_13478107821426688942` | 9.77e11 | 0.633 | 0.633 | 0.826 | 0.825 | 0.767 |

**These three reproduce to three decimals on every host of each part**, which
is what lets them carry a quantitative claim. The first is exactly one pinned
bf16 GEMM -- 2^40 FLOPs is 2 x 8192^3 -- and its ratio, 0.736, sits within
2% of the measured `tensor_virus` and `pulse_virus` ratios (0.723, 0.727),
the two workloads that run that GEMM.

The difference is the depth of `throttle_activity_1`: on inf2 it holds the
core at a **0.63** utilization limit for 84-90% of a matmul graph's run; on
trn1 the same activity applies a **0.875** limit for 15-38% of it.

**The cause is a utilization-limit throttle that inf2 applies far more
deeply than trn1.** Not the clock, not the code, not the counters -- each of
those was ruled out above -- and not a difference in the silicon's peak,
which AWS publishes as equal and which the memory rows show equal in
practice.

### One graph does not reproduce, and an earlier version of this section led with it

`MODULE_11435754646458375002`, the densest graph (2.78e12 hardware FLOPs per
execution), was the first row of this table as merged in #31, with a ratio of
0.719. A single capture of it is not reproducible:

| Host | avg util limit | cycles |
|---|---|---|
| inf2-a | 0.641 | 83,411,352 |
| inf2-b | *no throttle counters at all* | 273,732,448 |
| trn1-a | 0.971 | -- |
| trn1-b | 0.891 | -- |
| trn1-c | 0.918 | -- |

The throughput of every workload was identical to 0.02% across the three trn1
hosts, so this spread is in the one-execution capture, not in the hardware.
The 0.719 was one host's reading of a graph whose readings range across
0.66-0.72 depending on which trn1 host is the denominator, and on one inf2
host it carried no throttle data at all. It is kept here as evidence and not
used as a figure.

## What follows from it

**trn1 is throttled too.** Its 78.60 TFLOPS runs under an average 0.89
utilization limit on the densest graph, so its 83% of published peak is
itself capped. Neither part's `Percent Of Peak` is a statement about
unthrottled silicon; both are statements about what the part sustains as
AWS runs it, which is the thing a buyer gets.

**Single captures of the densest graph are not a usable comparison** --
not its cycle count, not its throttle counters -- for the reasons in the
table above. The three reproducible graphs are.

**The registry comment was corrected in #24 and stays correct.**
`PART_PEAKS` keeps AWS's published 190 TFLOPS for both parts, and what
`Percent Of Peak` now says -- inf2 delivers 60% of its advertised compute on
dense matmul, trn1 83% -- is explained rather than merely observed.

Evidence: `data/validation-2026-09-21/{inf2-a,inf2-b,trn1-a,trn1-b,trn1-c}-throttle-probe.log`.
