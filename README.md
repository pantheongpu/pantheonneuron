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

| Workload | Kernel | Score source |
|---|---|---|
| `baseline_metrics` | ✅ telemetry only, no load | — |
| `memory_read` | ✅ **verified on trn1.2xlarge** | `neuron-profile`, analytic fallback |
| the other 24 | ❌ none | — |

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

**Its Score now comes from the declared source.** After the timed loop the
kernel captures a profile, reads `hbm_read_bytes` and `total_time`, and
computes `hbm_read_bytes / total_time / 1e9` — exactly the formula the
registry declares. If the profiler is unavailable it degrades to the
analytic figure (bytes requested over wall time) rather than failing the
run, and the row's `Score Method` field records which was used. A
provisional number is never presented as the real one.

The distinction matters: the analytic figure counts bytes we *asked* for
and cannot detect loads the compiler eliminated. A kernel whose DMA was
optimised away still posts a fast wall time and a large analytic number,
while the profiler reports almost no HBM traffic.
`memory_read.verify_against_analytic` compares the two and puts the
divergence in the row's `Detail`.

The profiler reader (`kernels/profiler.py`) encodes four environment traps,
each found the hard way during the probes: `view` exits on an unset `$HOME`;
the Neuron bin directory must be on `PATH` because the tools shell out to
each other; `capture` writes readable NTFF v6 while `inspect` writes v115
that the same AMI's tooling cannot read; and the tools interleave log lines
with JSON on stdout.

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
