# Pantheon Neuron

A stress and validation suite for AWS Neuron accelerators — **AWS Trainium**
(`trn1`, `trn1n`, `trn2`) and **AWS Inferentia2** (`inf2`).

This is a third-party tool for exercising Neuron devices. It is not affiliated
with, endorsed by, or officially supported by Amazon Web Services.

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
