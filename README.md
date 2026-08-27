# Pantheon Neuron

A stress and validation suite for AWS Neuron accelerators — **AWS Trainium**
(`trn1`, `trn1n`, `trn2`) and **AWS Inferentia2** (`inf2`).

This is a third-party tool for exercising Neuron devices. It is not affiliated
with, endorsed by, or officially supported by Amazon Web Services.

## Validation status

**Only Inferentia2 has been validated on real hardware.** Two probes, both
`inf2.xlarge`. No Trainium part has ever run this code.

| Architecture | Device model | Status |
|---|---|---|
| `inf2` | NeuronCore-v2, 2 cores/device, no training | **Verified** on hardware |
| `trn1` | NeuronCore-v2, 2 cores/device, training | Assumed |
| `trn1n` | NeuronCore-v2, 2 cores/device, training | Assumed |
| `trn2` | NeuronCore-v3, 8 cores/device, training | Assumed — least confident |

What that leaves untested:

- The entire `training` capability path, and `transformer_train_step` with it.
- Architecture detection on Trainium. `_arch_from_sysfs` reads
  `info/architecture/instance_type`, verified to return `"Inf2"` on
  Inferentia. What a Trainium part returns has not been observed, and
  `_normalise_arch` handles the expected spellings without confirmation.
- Device-to-device NeuronLink — needs ≥ 2 devices, which needs a quota
  increase.
- Whether `neuron-profile` reports the same 108 counters on Trainium.

Trainium testing is blocked, not skipped. The account's Trn quota is 4 vCPU
and the smallest Trainium instance (`trn1.2xlarge`) needs 8. A launch
attempt on 2026-08-26 returned:

```
VcpuLimitExceeded: You have requested more vCPU capacity than your current
vCPU limit of 4 allows for the instance bucket that the specified instance
type belongs to.
```

Quota increases to 128 vCPU (Trn) and 96 vCPU (Inf) were requested the same
day and are `CASE_OPENED` with AWS support. Note that `aws ec2 run-instances
--dry-run` reports success here — dry run validates permissions and
parameters, not vCPU quota, which is only enforced on a real launch.

Treat every Trainium-specific claim in this repository as unverified until
this section says otherwise.

## Why one repository for both chips

Trainium and Inferentia are two product families but **one software stack**.
Both are driven by the AWS Neuron SDK: the same `neuronx-cc` compiler, the same
`torch-neuronx` runtime, the same `neuron-monitor` telemetry. Splitting them
into separate repositories would duplicate nearly everything and force every
fix to be made twice.

The real fault line is the NeuronCore generation, not the brand name — so this
suite models it as a capability layer. Each device reports what it can do, and
each workload declares what it needs. Workloads a device cannot support are
reported as `SKIPPED` with a reason, never silently omitted and never falsely
passed.

## Name parity with pantheongpu

Workload and suite names deliberately match the `pantheongpu` suite wherever
the underlying concept is the same, so that a Neuron result and a GPU result
for a given name are comparing like with like. `--test inference` means the
same thing on both platforms.

That constraint cuts both ways. Where a GPU workload targets a structure Neuron
does not have, there is deliberately **no Neuron workload of that name** —
reusing the name for something else would make comparison worse, not better.
Asking for one by name explains why it is absent:

```
$ python pantheon_neuron.py --test sfu_stress
[PANTHEON-NEURON] 'sfu_stress' is a pantheongpu workload with no Neuron
equivalent: SFU is an NVIDIA SM structure; Neuron's Scalar/GpSimd engines
are not equivalent.
```

The full list lives in `NO_NEURON_EQUIVALENT` in
[`kernels/registry.py`](kernels/registry.py) — 17 GPU workloads covering FP64,
ray tracing, media encode, warp scheduling, device atomics, and GPU-specific
memory-hierarchy structures (TLB, HBM banks, TSVs, L2 partitioning).

Suites shared with `pantheongpu`: `baseline`, `core`, `memory`,
`interconnect`, `inference`, `training`, `runtime`, `ai_auxiliary`.

### Inf1 is out of scope

Inf1 uses NeuronCore-v1 with the legacy `neuron-cc` compiler and `torch-neuron`
on PyTorch 1.x — a genuinely different toolchain. Supporting it would double
the backend surface for inference-only hardware that new deployments are not
buying. Discovery rejects Inf1 explicitly rather than misreporting it.

## Comparing results with pantheongpu

Every scored workload carries a `Score` and a `Unit`, using the unit strings
from the `pantheongpu` report schema verbatim. A comparison joins the two
platforms on `(Test Name, Unit)`:

| Unit | Workloads |
|---|---|
| `TFLOPS` | `tensor_virus`, `pulse_virus`, `transformer_virus`, `omni_virus` |
| `TOPS` | `int_virus` |
| `GB/s` | the four `memory_*`, `all_reduce`, `p2p_thrasher`, `pcie_bandwidth` |
| `tokens/s`, `prompt-tokens/s`, `requests/s`, … | the eight inference workloads |
| `train-steps/s` | `transformer_train_step` |

`tests/test_score_schema.py` asserts every unit string against a transcript
of the pantheongpu vocabulary. A typo there would not raise anything — it
would just produce a row that never joins.

Each workload also pins a `problem`: shape, dtype, and any workload-specific
parameters. **A Score is only comparable if both platforms ran the same
problem.** TFLOPS at bf16 and TFLOPS at fp32 are different numbers, so the
dtype travels with the score into the report.

Skipped workloads still emit their `Unit` and `Problem` with a null `Score`,
so a comparison renders an explicit gap instead of dropping the row.

### Where each Score comes from

Every scored workload declares a `score_source`: which interface produces
the number, which counters it reads, and the formula. See
[`docs/workload_counter_map.md`](docs/workload_counter_map.md) for the full
table, generated from the registry.

| Source | Workloads | Why this source |
|---|---|---|
| `neuron-profile` | the four `memory_*` | HBM byte counters exist only in the profiler |
| `neuron-monitor` | the four TFLOPS/TOPS viruses, `graph_replay` | `effective_flops` exists only in the monitor |
| `nccom-test` | `all_reduce`, `p2p_thrasher` | reports `busbw` directly |
| `workload` | the 8 inference workloads, training, and 3 others | no hardware counter measures tokens, steps or requests |

A test asserts that any hardware-sourced counter named here was actually
observed on real hardware, so a counter rename in the Neuron SDK breaks a
reference instead of silently producing a wrong number.

`data/baselines.json` records what each counter read during the probes.
**Those are observations, not benchmark results** — the probe load was an
untuned matmul at 0.0049% MFU, roughly four orders of magnitude below the
hardware's capability. They exist to prove each counter is readable and to
catch plumbing regressions.

### What cannot be compared

`Efficiency (MB/J)` and everything derived from watts. Neuron exposes no
power figure in documented units — see
[`docs/neuron_counters.md`](docs/neuron_counters.md). Perf-per-watt is not
computable across these platforms, and neither is temperature, fan or
voltage.

## Kernel status

| Workload | Kernel | Score source |
|---|---|---|
| `baseline_metrics` | ✅ telemetry only, no load | — |
| `memory_read` | ⚠️ written, **untested on hardware** | analytic (profiler wiring pending) |
| the other 24 | ❌ none | — |

`memory_read` is the first real kernel. Two caveats travel with it:

**It has never run on a Neuron device.** It was written against the NKI
programming model with no hardware available. The tile geometry and byte
accounting are covered by tests that run anywhere; the NKI calls are not.
Treat the first hardware run as bring-up, not measurement.

**Its Score is provisional.** The registry declares the Score comes from
`neuron-profile`'s `hbm_read_bytes`; that wiring does not exist yet, so the
kernel reports the analytic figure — bytes requested over wall time. Every
report row carries a `Score Method` field recording this, so a result is
never read as though the declared contract held.

The two differ in a way that matters: the analytic figure counts bytes we
*asked* for and cannot detect loads the compiler eliminated. A kernel whose
DMA was optimised away still posts a fast wall time and a large analytic
number. `memory_read.verify_against_analytic` is the guard for that — it
compares the profiler figure against the analytic one and fires when they
diverge. It is tested, but it cannot run until the profiler is wired in.

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

## Running without hardware

The full orchestrator, telemetry and reporting path runs on any machine via a
CPU mock backend. This is what CI exercises:

```bash
PANTHEON_NEURON_MOCK=1 python pantheon_neuron.py --duration 2
```

Mock mode never reports a real workload as having run on hardware. On a real
device, a workload with no NKI implementation raises rather than passing.

## Reports

Runs write JSON to `database/`, which is gitignored. **This repository is
public**, so reports must never contain host identifiers — no hostname, no IP,
no EC2 instance ID, no availability zone.

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
