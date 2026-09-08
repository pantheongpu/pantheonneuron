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
| `memory_read` | ✅ **verified** on trn1.2xlarge and inf2.xlarge | `neuron-profile`, analytic fallback |
| `memory_write` | ✅ **verified** on inf2.xlarge; pin lowered 8 GiB → 4 GiB | `neuron-profile`, analytic fallback |
| `tensor_virus` | ✅ **verified** on inf2.xlarge at 1024³/2048³, not at 8192³ | `neuron-monitor` |
| `int_virus` | ⚠️ repinned to **uint8**, which trn1 does support; untested | `neuron-monitor` |
| `pulse_virus` | ✅ **verified on trn1.2xlarge** at 2048³ | `neuron-monitor` |
| `omni_virus` | ⚠️ all four engines in one dependent chain, untested; a NameError that would have failed its first hardware run is now fixed | `neuron-monitor` |
| `transformer_virus` | ⚠️ realistic instruction mix, untested | `neuron-monitor` |
| `graph_replay` | ⚠️ dispatch rate, untested | `neuron-monitor` execution counter |
| `memory_read_agg` / `memory_write_agg` | ⚠️ one process per core, untested | workload |
| `pcie_bandwidth` | ⚠️ the guard fired on a harness artifact, now fixed; needs a rerun | workload |
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

**`pcie_bandwidth`'s asymmetry was the harness.** d2h 1.0 GB/s against h2d
6.0 was not a link property: the legs were not symmetric. h2d reused one host
tensor while d2h called `.cpu()`, which allocates a fresh 1 GiB host
destination on every pass. A 6x split between reusing a buffer and
allocating, faulting in and freeing 1 GiB per pass is what an allocator
costs. Both legs now `copy_` into a destination allocated before the clock
starts, and the result records `buffers: preallocated` so a row from before
the fix is identifiable. **The 1.0 GB/s figure should not be cited.** This
does not show the link is healthy — it shows the old number could not have
told us either way, and the workload needs a rerun.

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
against h2d 6.0 GB/s. *Resolved: an artifact, though not the one suspected --
`.cpu()` allocates a fresh host destination per pass while h2d reused one
buffer. See above; the figure should not be cited.*

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
