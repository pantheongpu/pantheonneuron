# Pantheon Neuron

A stress and validation suite for AWS Neuron accelerators — **AWS Trainium**
(`trn1`, `trn1n`, `trn2`) and **AWS Inferentia2** (`inf2`).

This is a third-party tool for exercising Neuron devices. It is not affiliated
with, endorsed by, or officially supported by Amazon Web Services.

## Validation status

Both chips have now run this code on real hardware.

| Architecture | Device model | Status |
|---|---|---|
| `inf2` | NeuronCore-v2, 2 cores/device | **Verified** — inf2.xlarge, 2026-08-26 |
| `trn1` | NeuronCore-v2, 2 cores/device, training | **Verified** — trn1.2xlarge, 2026-08-27 |
| `trn1n` | NeuronCore-v2, 2 cores/device, training | Assumed — same silicon, more network |
| `trn2` | NeuronCore-v3, 8 cores/device, training | Assumed — least confident |

Verified on Trainium: architecture detection, the `training` capability path
(a real backward pass and optimiser step at 23.25 train-steps/s), the
`memory_read` NKI kernel, and the profiler counter reader.

**Device generation does not track product naming.** Trainium1 reports
`NDv2`; Inferentia2 reports `NDv3`. The newer NeuronDevice generation
belongs to the older-numbered product, so never infer the core version from
the device version — both chips run NeuronCore-**v2**.

**The counter sets differ between chips.** `neuron-profile` returns 108
counters on inf2 and 90 on trn1, and `throttle_active_nc0_time_ns` is
present on Inferentia but `None` on Trainium. A kernel must not assume a
counter exists because the other chip had it.

Still unverified:

- Device-to-device NeuronLink. The Trn quota was granted at 64 vCPUs;
  `trn1.32xlarge` needs 128, so multi-device remains out of reach.
- `trn1n` and `trn2`.

## Kernel status

Every workload in the registry now has an implementation: **26 of 26**. What
differs is how much of each has met hardware.

| Workload | Kernel | Score source |
|---|---|---|
| `baseline_metrics` | ✅ telemetry only, no load | — |
| `memory_read` | ✅ **verified**, and **scored from `neuron-profile`** on both parts | `neuron-profile` |
| `memory_write` | ✅ **verified** on both parts at the 4 GiB pin; **scored from `neuron-profile`** | `neuron-profile` |
| `tensor_virus` | ✅ **verified at the pinned 8192³**; scored from `neuron-monitor` | `neuron-monitor` |
| `int_virus` | ✅ **verified at the pinned 8192³** at uint8; scored from `neuron-monitor` | `neuron-monitor` |
| `pulse_virus` | ✅ **verified at the pinned 8192³**; scored from `neuron-monitor` | `neuron-monitor` |
| `omni_virus` | ⚠️ all four engines in one dependent chain, untested; a NameError that would have failed its first hardware run is now fixed | `neuron-monitor` |
| `transformer_virus` | ⚠️ realistic instruction mix, untested | `neuron-monitor` |
| `graph_replay` | ⚠️ dispatch rate, untested | `neuron-monitor` execution counter |
| `memory_read_agg` / `memory_write_agg` | ⚠️ one process per core, untested | workload |
| `pcie_bandwidth` | ⚠️ **verified**; d2h asymmetry reproduces on trn1, absent on inf2, unexplained | workload |
| `allocation_fragmentation` | ✅ **verified on trn1.2xlarge** | workload |
| `llm_prefill` / `llm_decode` / `kv_cache_churn` | ⚠️ untested | workload |
| `fused_attention` / `quantized_gemm` / `moe_router` | ⚠️ untested | workload |
| `speculative_decode` / `serving_mix` | ⚠️ untested | workload |
| `rag_embedding` / `vision_encoder` | ⚠️ untested | workload |
| `transformer_train_step` | ⚠️ untested; skips on Inferentia | workload |
| `all_reduce` / `p2p_thrasher` | ⚠️ **cannot be run yet** | `nccom-test` |

`all_reduce` and `p2p_thrasher` need two or more devices. `trn1.32xlarge` is
the smallest instance with device-to-device NeuronLink and needs 128 vCPUs
against a granted 64, so they are written against the documented
`nccom-test` output rather than against observed output, and their tests are
the only thing behind them until that quota lands.

### What the 2026-09-08 findings were resolved into

The four open items that run recorded are now closed in code. None of the
fixes has met hardware; what changed is that each has a decided answer and a
test, so the next window spends its time confirming rather than discovering.

**The profiler now searches for its graph instead of guessing.** The capture
worked and captured the wrong NEFF: *profiled graph moved 4 bytes against a
plan of 8589934592*. `find_neff` ranked candidates by mtime and returned the
top one, and `verify_profile_covers_plan` then rejected it — so the plan
check was a rejector standing next to a guess. It is now the selector.
`profiler.find_neffs` returns the ranking and `profiler.select_by_plan`
captures candidates in turn until one accounts for the planned traffic. Both
times this failed on hardware, the kernel's own graph was in the list and was
not first, so a wrong first guess now costs another capture rather than the
whole Score. The search is capped at 6 captures (each is a real NEFF replay);
`PANTHEON_NEURON_NEFF_CANDIDATES` raises it, and exhausting it reports what
every candidate moved rather than only that one was wrong.

**`int_virus` is repinned to uint8.** trn1's Tensor Engine rejects signed
int8 and accepts uint8, so the workload was unreachable as declared. uint8
rather than fp8 of the reachable options: the unit is TOPS, which means
integer operations, and fp8 would keep the label while changing the quantity
underneath it. An all-ones GEMM still reaches exactly K, so the correctness
check is unchanged. What this costs is that a signed-int8 path is not
measured here — which is why the dtype travels with the Score in `problem`.

**`memory_write` is repinned to 4 GiB.** A write's destination is the whole
plan and the runtime still holds the previous one while the next is
allocated, so the pin costs twice its size in residency against 16 GB a core.
4 GiB rather than 6: both fit, but the drop from 255.1 to 162.5 GB/s at 6 GiB
is the part running out of room, and a number measured under allocation
pressure is not the write bandwidth this workload claims. `memory_read` keeps
8 GiB — a read allocates only a source and was verified there.

**`pcie_bandwidth`: the fix was right and the diagnosis was wrong.** The legs
genuinely were not symmetric — h2d reused one host tensor while d2h called
`.cpu()`, which allocates a fresh 1 GiB host destination every pass — so both
now `copy_` into a destination allocated before the clock starts, and the row
records `buffers: preallocated`.

But that was not the cause. **The rerun measured d2h at 1.1 GB/s against h2d
6.0, essentially unchanged from the 1.0 that started this.** The allocation
was a real defect in the harness and removing it moved nothing. See the
2026-09-08 rerun below for what the two parts then said, which is the useful
part.

### The pinned problem compiles, and neuron-monitor finally scored

`tensor_virus` and its four relatives declare `mean(effective_flops) / 1e12`
from neuron-monitor. **That source had never once produced a Score**, and the
reason was not the monitor: the pinned 8192³ problem had never compiled, so
the compute workloads had only ever been run by hand at a reduced shape,
which bypasses `monitor_score` entirely.

**The cause was one word.** The kernel's innermost loop accumulates into
`acc` — a loop-carried dependency — but was declared `nl.affine_range`, which
NKI reserves for loops *without* one and which the compiler fully unrolls.
The unroll is cubic in the shape:

| shape | matmul calls | before | after |
|---|--:|---|---|
| 2048³ | 1,024 | compiled | 22.46 TFLOPS, product verified 1.0 |
| 4096³ | 8,192 | untested | **36.56 TFLOPS**, verified |
| 8192³ | 65,536 | **never compiled** | **26.24 TFLOPS in 77 s**, verified |

`nl.sequential_range` is both the semantically correct choice for an
accumulator and the one that rolls the loop, dividing the unrolled body count
by `k_tiles` — 64× at the pinned shape.

With the pinned problem reachable, the orchestrated run produced this:

| Workload | Score | Source | Problem |
|---|--:|---|---|
| `tensor_virus` | 22.60 TFLOPS | **`neuron-monitor`** | 8192³ bf16 |
| `int_virus` | 27.28 TOPS | **`neuron-monitor`** | 8192³ uint8 |
| `pulse_virus` | 13.38 TFLOPS | **`neuron-monitor`** | 8192³ bf16, 50% duty |

Three things worth reading off that table. **uint8 is faster than bf16**
(27.28 against 22.60), which is what an 8-bit integer datapath should do and
is the first evidence the repin measures a real path rather than just a
compilable one. **`pulse_virus` lands at 59% of sustained** for a 50% duty
cycle, consistent with the wall-clock/loaded-only split it reports itself.
And the counter reads **below** the kernel's analytic figure — 22.60 against
26.24 at the same shape — which is the right direction: the analytic number
counts arithmetic issued and the counter counts what the engine retired.

**4096³ is faster than the pinned 8192³**, by a wide margin, and the reason
is not what it looked like.

| shape | streaming | blocked | operand traffic cut | speedup |
|---|--:|--:|--:|--:|
| 2048³ | 23.30 | 23.38 | 4.2× | 1.00× |
| 4096³ | **36.55** | **38.85** | 4.5× | 1.06× |
| 8192³ | 26.26 | 27.93 | 4.7× | 1.06× |

The kernel re-reads operand tiles for every (row, col) pair, so operand
traffic scales as n³ exactly like the FLOPs and arithmetic intensity stays
flat at ~102 FLOP/byte. At 8192³ that implied 256.4 GB/s of traffic against
`memory_read`'s measured 256.2 GB/s on the same part — a 0.1% agreement that
looked exactly like a bandwidth wall.

**It was a coincidence.** A blocked tiling that cuts operand traffic 4.7×
moved throughput by 1.06×. Had bandwidth been the constraint, the speedup
would have tracked the traffic. So operand bandwidth is ruled out, and what
actually binds this kernel is still unknown: both tilings sit at 25–41% of
the ~95 TFLOPS one NeuronCore-v2 should reach in bf16, and 4096³ beats 8192³
under both. The next place to look is per-tile issue overhead and the
128×512 tile shape, not the memory system.

Both tilings ship and both product-verify at exactly 1.0. `streaming` is the
default: a 6% gain does not pay for an extra SBUF block and a deprecated NKI
layout, and the argument that motivated `blocked` turned out to be wrong.
Select it with `PANTHEON_NEURON_GEMM_TILING=blocked` or
`problem["tiling"]`, and compare them with `tools/compare_tiling.py`.

### What the 2026-09-08 rerun found, on both parts

trn1.2xlarge (us-east-1f) and inf2.xlarge (us-east-1d), same commit, run in
parallel. Raw logs in [`data/validation-2026-09-08/`](data/validation-2026-09-08/).

**The declared profiler Score fired, for the first time in this suite's
history: 2 of 4, on both parts.**

| Workload | trn1.2xlarge | inf2.xlarge | Source |
|---|--:|--:|---|
| `memory_read` | 256.17 GB/s | 256.19 GB/s | **`neuron-profile`** |
| `memory_write` | 226.50 GB/s | 226.77 GB/s | **`neuron-profile`** |
| `allocation_fragmentation` | 544.42 | 492.93 | workload |
| `pcie_bandwidth` | 3.56 GB/s | 2.12 GB/s | workload |

`memory_read`'s 256.17 GB/s cross-checks against the 264 GB/s wall-clock
figure measured on this part in August, which is what
`verify_against_analytic` exists to confirm. **`memory_write` ran at the new
4 GiB pin with no `NRT_RESOURCE` failure on either part**, which is the
repin doing its job.

**The two parts agree to four significant figures, and that is expected
rather than suspicious.** Both are NeuronCore-v2 against the same 32 GB HBM
config, and the profiler figure is one NEFF replay with no cache and no
contention — a deterministic per-execution measurement, which is exactly why
it makes a better regression signal than wall-clock timing.

**`int_virus` ran for the first time**: 23.05 TOPS on trn1 and 20.71 on inf2
at 2048³ uint8, `product verified 1.0` on both. It sits just under
`tensor_virus` on the same shape (23.31 / 20.41 TFLOPS), which is what an
engine treating uint8 and bf16 at the same rate looks like. The workload was
unreachable before the repin.

**The NEFF search returned the right graph and never had to search.** Both
parts reported `plan_coverage 1.0` from `candidates_tried 1` of
`candidates_available 1`: a fresh instance with `PANTHEON_NEURON_WORKDIR`
set has exactly one NEFF, so the ranking had nothing to rank. What is
confirmed is that the coverage check identifies the right graph and does not
reject a correct one. **The multi-candidate search — the actual fix — is
still unexercised on hardware**, because the failure it addresses needs an
accumulated compile cache to reproduce.

**The PCIe asymmetry is a trn1 property, not a harness artifact.** The same
commit, with both legs preallocated, produced d2h 1.1 GB/s against h2d 6.0
on trn1 and **no asymmetry warning at all on inf2**. A defect in the harness
would have shown on both. So the split is reproducible on trn1 across two
different methodologies and is not explained by the allocation, which was
the hypothesis.

Before it is read as a link property, one thing is worth ruling out: the h2d
leg copies the same `host` into `resident` every pass, and if XLA elides a
copy whose result never changes, the inflated figure would be the 6.0 rather
than the depressed one being the 1.1. That is a testable question and it has
not been tested.

**Profiler and wall-clock diverge by more than the summary shows.** A
direct 1 GiB `memory_read` on inf2 gave 178.7 GB/s from the profiler against
260.1 GB/s analytic — a ratio of 0.69, inside the 50% tolerance
`verify_against_analytic` allows, so no warning fired. The profiler times a
single cold NEFF replay while the analytic figure averages thousands of
steady-state passes, so some gap is expected; whether 31% is the right
amount of gap is not something this run answers.

### What the 2026-09-08 trn1.2xlarge validation found

The first Trainium run of the harness, and the first time the reserved
profiler core was exercised.

**int8 does not exist on this Tensor Engine.** `int_virus` failed with
`nc_matmul does not support stationary.dtype=int8`; the supported set is
fp8_e4m3, fp8_e5m2, bf16, fp16, tf32, fp32 and **uint8**. The registry pinned
int8, so the workload was unreachable on trn1 as declared. *Resolved: uint8,
see above.*

**The reserved core worked and the profiler still produced nothing.** The run
logged `cores 0 to the workload, 1 reserved for neuron-profile`, so the
capture ran for the first time. It then captured the wrong graph, and
`verify_profile_covers_plan` refused it: *profiled graph moved 4 bytes
against a plan of 8589934592*. Scores from a declared hardware source: 0 of
4. Narrowing NEFF selection by compile timestamp is not enough to identify
the kernel's own graph. *Resolved: the plan check is now the selector rather
than only the rejector, see above.*

**pulse_virus behaves as designed.** Its loaded-only figure (23.74 TFLOPS)
lands on `tensor_virus`'s sustained figure (23.25 TFLOPS) while its
wall-clock figure is 11.88, which is what a 50% duty cycle should look like
if pulsing costs no throughput while loaded.

**memory_write fails at the pinned 8 GiB on trn1 exactly as on inf2**, same
`NRT_RESOURCE` exhaustion. Both target parts have 32 GB across two cores, so
the pin is unreachable on either -- now confirmed rather than inferred.

**pcie_bandwidth's asymmetry guard fired on its first run**: d2h 1.0 GB/s
against h2d 6.0 GB/s. *Reproduced on trn1 after the harness was fixed, and
absent on inf2. See the rerun below.*

### What the transformer family's first tests found

The five modules behind the ten AI workloads — `transformer_ops`,
`llm_inference`, `inference_mix`, `encoders`, `transformer_compute` — were
the only kernel modules no test imported, roughly 1,100 lines. Adding
`tests/test_transformer_family.py` found one defect immediately:

**`omni_virus` called a `_read_back` it did not have.** The helper was copied
privately into four modules and `omni_virus` was not one of them, so its
`run()` ended in `NameError: name '_read_back' is not defined` — *after* the
full-duration stress loop, turning a completed run into a FAIL row with
nothing to show for the device time. It could only ever have surfaced on
hardware, and `omni_virus` has never run there.

The helper now lives once, in `transformer_ops.read_back`. Two tests keep
that class of defect closed: one asserts there is exactly one definition, and
one walks every kernel module's AST for names that are neither defined,
imported, nor builtin — with a self-test that the checker actually fails the
`omni_virus` code, since a checker that cannot catch its own motivating bug
proves nothing.

The rest of the file covers what can be checked without a device: the FLOP
arithmetic (including that prefill and decode still differ by orders of
magnitude, which is the registry's stated reason for keeping them separate
workloads), the vision encoder's patch geometry, the serving interleave, and
that every workload's dispatch key exists in the kernel that must produce it.

`verify_output_is_finite` was renamed to `verify_output_is_a_number`. Its
name and summary line promised an inf check the body deliberately did not do
— inf is expected here, since all-ones weights with no normalisation saturate
bf16 — and a check whose name overstates it is the same defect this suite
spends its Score labelling on.

### Why the AI workloads are separate workloads

They share transformer primitives and compose them into deliberately
different computations. pantheongpu collapsed twelve AI workloads into one
`ai-ops/s` because ten shared a kernel body and six compiled to
byte-identical SASS -- one measurement wearing twelve names. Here,
`llm_prefill` runs a 2048-token prompt through every layer and is quadratic
in sequence; `llm_decode` runs a single token against a cache and is linear
in context, roughly three orders of magnitude less arithmetic per step; and
`kv_cache_churn` never runs the model at all. Anything that would make two
of them compile to the same graph belongs in one workload, not two.

### What the 2026-09-07 inf2.xlarge bring-up changed

Three bugs, each of which produced a plausible-looking number rather than an
error, and none of which any test could have caught:

**The warm-up compiled a different graph than the loop ran.** Holding the
kernel result makes the output live at the `mark_step()` cut, so a warm-up
that discards it compiles one graph and leaves the real one to be built
*inside* the timed region. `memory_read` reported **0.0208 GB/s over 478 s**
with two executions and 0.02% NeuronCore utilisation, because a seven-minute
compile was measured as bandwidth. Warming up with the same liveness gives
**236.9 GB/s over 8,826 passes**.

**`memory_write` allocated a full-size source for a kernel that reads one
tile.** 8 GiB of source for 256 KiB of use, against an 8 GiB destination on a
16 GB core.

**The pinned 8 GiB write does not fit, even so.** The destination is the
whole plan and the runtime still holds the previous one when the next is
allocated. Measured ceiling on this part: 4 GiB runs at 255.1 GB/s and 6 GiB
at 162.5 GB/s, both with the destination check at exactly 1.0; 8 GiB fails
with 8.59 GB requested against 8.099 GB resident. Both parts this suite
targets have 32 GB across 2 cores, so the pin was unreachable on either.
*Resolved: lowered to 4 GiB, see above.*

`tensor_virus` is the first kernel whose Score does not come from the kernel.
`effective_flops` lives only in the neuron-monitor stream and does not exist
until the monitor stops, so `pantheon_neuron.monitor_score` reads it after the
run and the kernel supplies FLOPs-over-wall-time as the cross-check. The two
answer different questions: the analytic figure counts arithmetic issued, the
counter counts what the Tensor Engine retired, and a matmul folded away at
compile time shows up as the gap between them. Its all-ones operands make
every output element exactly K, which is what `verify_product_is_correct`
checks — the far corner especially, since it is produced by the last tile of
both loops.

`memory_read` is the first real kernel. Two caveats travel with it:

**It runs, and it is numerically correct.** Bring-up on trn1.2xlarge
2026-08-27: the kernel compiles in 1.5 s and returns exactly
`tiles x free_elements` for an all-ones input — it reads every byte and
reduces correctly. Sustained read bandwidth measured **264 GB/s** on a
1 GiB bf16 buffer, single NeuronCore.

**Bring-up found a real bug, which is why the barrier is there.**
`xm.mark_step()` queues work and returns without waiting for the device.
Timing without a barrier measures queue submission, and the error grows
with buffer size:

| Buffer | no barrier | with barrier |
|---|---|---|
| 128 MiB | 208 GB/s | 200 GB/s |
| 512 MiB | 838 GB/s | 256 GB/s |
| 1024 MiB | **1636 GB/s** | **264 GB/s** |

Elapsed time stayed pinned at 0.013 s regardless of size, so the
unsynchronised "bandwidth" was bytes divided by a constant. At small
buffers it looks nearly right, which is what makes it dangerous. The
barrier is inside the timed region, and the wrong figure is kept in
`data/baselines.json` as a regression marker.

**Its Score has never actually come from the declared source.** The kernel
captures a profile after the timed loop and computes
`hbm_read_bytes / total_time / 1e9`, exactly the formula the registry
declares — but that capture replays the NEFF, which needs a NeuronCore, and
the workload process held every one of them. Every scored run in this
suite's history has therefore degraded to the analytic figure. The row's
`Score Method` records which was used, so no provisional number was ever
presented as the real one, but the declared path had never once run.

The run now reserves a core for the profiler (`kernels/cores.py`), which
should close this. That reservation is written and tested but has **not**
been exercised on hardware.

The distinction matters: the analytic figure counts bytes we *asked* for
and cannot detect loads the compiler eliminated. A kernel whose DMA was
optimised away still posts a fast wall time and a large analytic number,
while the profiler reports almost no HBM traffic.
`memory_read.verify_against_analytic` compares the two and puts the
divergence in the row's `Detail`.

The profiler reader (`kernels/profiler.py`) encodes five environment traps,
each found the hard way on hardware: `view` exits on an unset `$HOME`; the
Neuron bin directory must be on `PATH` because the tools shell out to each
other; `capture` writes readable NTFF v6 while `inspect` writes v115 that the
same AMI's tooling cannot read; the tools interleave log lines with JSON on
stdout; and `capture` needs a NeuronCore of its own, because it replays the
NEFF rather than reading counters from the running process.

### memory_write

Loads exactly **one** tile and stores it across every row, so read traffic
is one tile while write traffic is the whole buffer. That asymmetry is the
point: it keeps `hbm_write_bytes` clean and gives a cheap correctness
check. `verify_write_dominates_read` fails the run if read bytes approach
write bytes, which would mean the kernel is doing a read-modify-write and
the Score is measuring a mixed workload rather than a write.

The anti-elimination trick differs from the read path. `memory_read`
reduces its loads so they have a consumer; here the hazard is inverted —
stores into a buffer nothing reads are dead code. The destination is the
kernel's returned output, which is what keeps the stores alive.

It ran for the first time on inf2.xlarge 2026-09-07 and its destination
check passed exactly, at 4 GiB (255.1 GB/s) and 6 GiB (162.5 GB/s). The
pinned 8 GiB does not fit; see the bring-up notes above.

## Requirements

The orchestrator itself needs only Python 3.9+ and `psutil`. For real runs you
need a Neuron instance with the driver and toolchain installed:

```bash
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
python -m pip install neuronx-cc torch-neuronx neuronx-distributed
```

The Neuron packages are deliberately **not** in `requirements.txt` — they come
from the AWS Neuron pip index rather than PyPI.

## Usage

```bash
python pantheon_neuron.py --list
```

```bash
python pantheon_neuron.py --test all --duration 60 --device all
```

```bash
python pantheon_neuron.py --test interconnect --duration 120 --device 0,1
```

Key flags: `--test` (workload name, suite, or `all`), `--duration` (seconds per
workload), `--device` (indices or `all`), `--monitor-period` (telemetry
sampling interval), `--mock`, `--no-report`.

### The first clean run: 23 of 23, and eight fixes confirmed

`validate_hardware.sh` over every single-device workload, trn1.2xlarge,
after a day of fixes. **23 PASS, 0 FAIL**, 8 of 23 Scores from a declared
hardware source.

| workload | before | after | predicted |
|---|--:|--:|---|
| `kv_cache_churn` | 0.85 → 8,716 → 133 | **97,167/s** | memory-bound at last |
| `moe_router` | FAIL (SIGABRT) | **254,743/s** | — |
| `llm_prefill` | FAIL (NaN) | **3,808/s** | — |
| `graph_replay` | 729 / 1,175 swing | **1,189** | trimming removes dilution |
| `speculative_decode` | 2,180 | **74.0** | ÷32 — measured ÷29.5 |
| `rag_embedding` | 1,551,194 | **853.7** | ÷2040 — measured ÷1817 |
| `vision_encoder` | 1,676,047 | **58,131** | ÷18 — measured ÷28.8 |
| `serving_mix` | 138 | **0.0312** | a real request is many steps |

The last four are the size of the "ran a fraction of the model" defect,
and in each case the prediction made from arithmetic beforehand matched
the measurement to within a factor of two. That is the one class of
off-hardware reasoning that held up all day.

`serving_mix` at 0.0312 requests/s is honest and not yet useful: a real
request is 32 scheduler steps, so a 20-second run completes well under one.
The workload needs a longer duration or a shorter pinned decode length
before its Score means anything.

### Coverage is necessary and not sufficient

The same run regressed `memory_read` to 23.58 GB/s against an analytic
271.7 — a ratio of 0.09, caught by the divergence guard.

The captured graph cleared the coverage floor, because **two workloads
here pin 8 GiB and byte-coverage cannot tell their graphs apart.** The
exact-match early exit could not help: another graph also covers the plan.
Cold-cache reproducibility was fixed; warm-cache *identification* was not.

`read_verified_ratio` adjudicates. It is measured from the kernel's own
accumulator rather than from any capture, so when it reads 1.0 the kernel
provably touched every planned byte, the analytic figure is the
trustworthy one, and the profile belongs to somebody else's graph. Both
bandwidth kernels now degrade to analytic and say so, rather than
publishing a number from a capture they cannot attribute.

### A KV cache cannot be updated in place on this stack

The most useful thing `kv_cache_churn` has produced is not a Score. XLA is
functional: `cache[:, a:b, :] = entry` lowers to a dynamic-update-slice,
which produces a **new tensor**. There is no in-place write. Appending 512
tokens to a 2 GiB cache does not move 256 MiB — it reads 2 GiB and writes
2 GiB.

It took three measurements on trn1.2xlarge to see that, and each one first
looked like a different problem:

| attempt | measured | what it looked like |
|---|---|---|
| 1 layer, 1 token | 115 µs for 16 KiB — 1,794× what HBM needs | dispatch overhead |
| 32 layers, 64 tokens | 455 ms for 32 MiB — 54× more than rewriting every layer's whole slice | a slow scatter |
| 8 static ring slots | **~7 minutes to compile each slot's graph** | — |

A graph that compiles for seven minutes to write a slice is a graph handling
the whole 2 GiB tensor. Once that is true the other two follow, and the first
two diagnoses were both wrong.

The consequence for anyone serving on Neuron is larger than this workload:
**the cost of appending to a KV cache is proportional to the size of the
cache, not to the number of tokens appended.** The pinned problem is now
sized so a whole-cache copy is tractable — 8 layers, 2048 context, 2048
hidden, a 128 MiB cache and a 256 MiB step — and `bytes_per_step` reports the
copy rather than the slice, because counting the slice would report a
sixteenth of what the hardware moves.

**Unverified at this size.** The 2 GiB version never finished compiling.

### Repeats

Most Scores in this README are from a single run, and the one quantity that
was ever measured twice turned out not to be reproducible: `memory_read`'s
declared profiler Score read 256.17, 178.7 and 119.19 GB/s on three runs of
the same pinned problem.

The cause was the NEFF selector taking the first candidate over the coverage
floor rather than the best match, so a partially-matching graph could win.
With the selector fixed, three consecutive runs read **256.2692, 256.2847
and 256.1250 GB/s** — a coefficient of variation of 0.0003 against roughly
0.4 before — each identifying its graph at coverage exactly 1.0, on the
second of three candidates. The search skipping a wrong first candidate is
the fix doing its job.

What is not fixed by finding that is that nothing would have caught it,
because nothing ever ran a workload twice.

```bash
python pantheon_neuron.py --test memory_read --duration 60 --repeat 5
```

The row then carries `Repeats` beside the Score: the range, the median and
the coefficient of variation. **`Score` becomes the median**, which is what
to quote; the spread is what says whether to quote it at all. Above a
coefficient of variation of 0.10 the row says so in its `Detail`, because a
median of unlike numbers is not a measurement.

A failure in any repeat fails the row. A workload that works four times in
five is not a workload that works.

### What `--test all` cannot measure

The profiler needs a NeuronCore of its own to replay a NEFF, and the Neuron
runtime reads `NEURON_RT_VISIBLE_CORES` once at initialisation — so the split
between workload and profiler is fixed for a whole run and cannot be
renegotiated per workload.

`memory_read_agg` and `memory_write_agg` declare `cores: "all"`. Holding a
core back from them would report the aggregate of all-but-one core under a
name that says otherwise, so their presence in a selection turns the
reservation off for the entire run. **`--test all` and `--test memory` both
select them**, which means `memory_read` and `memory_write` report the
analytic fallback rather than the `neuron-profile` Score they declare.

To reach the declared source, run them in a selection with no `cores: "all"`
workload in it:

```bash
python pantheon_neuron.py --test memory_read --duration 60
```

The run says which it did — the console names the workloads that are paying,
and each row's `Score Method` records the method actually used.

## Running without hardware

The full orchestrator, telemetry and reporting path runs on any machine via a
CPU mock backend. This is what CI exercises:

```bash
PANTHEON_NEURON_MOCK=1 python pantheon_neuron.py --duration 2
```

Mock mode never reports a real workload as having run on hardware. On a real
device, a workload with no NKI implementation raises rather than passing.

## Reports

Runs write JSON to `database/`, which is gitignored — reports are not
committed. The invariant does not rest on that. A report is the artifact that
gets pasted into an issue, attached to a mail, or quoted in a write-up, and it
is produced on a rented instance whose identifiers belong to somebody's
account. One paste away from public is the same requirement as public, so
reports must never contain host identifiers — no hostname, no IP, no EC2
instance ID, no availability zone.

`neuron-monitor` volunteers several of these in every sample, so telemetry is
scrubbed at ingest rather than at write time. `tests/test_report_privacy.py`
enforces the invariant and runs as its own required CI job. If it fails, find
what started emitting the identifier — do not relax the test.

## Custom kernels

There is no hand-written device C++ path on Neuron the way there is on CUDA.
Deliberate stress patterns go through **NKI** (the Neuron Kernel Interface), a
tile-based Python DSL that `neuronx-cc` lowers to NeuronCore instructions.
Graph-level workloads built with `torch-neuronx` are subject to compiler
optimisation and will have an idle stress loop folded away, so anything that
must genuinely keep the hardware busy belongs in NKI.

## Development

```bash
make test
```

```bash
make mock
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
