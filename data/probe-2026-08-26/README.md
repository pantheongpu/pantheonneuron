# Counter probe — 2026-08-26

Raw captures from a live probe run. These are the primary evidence behind
[`docs/neuron_counters.md`](../../docs/neuron_counters.md) and the fixture in
`tests/test_real_hardware_schema.py`.

## Run conditions

| | |
|---|---|
| Instance | inf2.xlarge |
| Device | 1 × Inferentia2, 2 × NeuronCore-v2, 32 GB |
| Device generation | NDv3 (`neuron_device_version` v3) |
| PCIe | Gen4 ×8 negotiated, Gen5 ×8 capable |
| Neuron runtime | 2.30.51 |
| Image | Deep Learning AMI Neuron (Ubuntu 22.04) 20260227 |
| Load | `tools/probe_load.py` — 88,154 executions over 120 s |
| Duration / cost | ~32 min, ~$0.40 |
| Instance | terminated; IAM role, instance profile and security group deleted |

## Files

| File | Contents |
|---|---|
| `01-neuron-ls.txt` | Topology, human and JSON. Shows the missing architecture field. |
| `02-neuron-toolchain.txt` | Installed Neuron binaries and venvs; what remains unexplored. |
| `03-sysfs-tree.txt` | Full driver sysfs tree, thermal/power grep, hwmon, thermal zones. |
| `04-device-info.txt` | Device identity values and PCIe link state. |
| `05-power-timeseries.txt` | `power/utilization` at idle, during compile, and under load. |
| `06-workload.txt` | Load definition, result, and environment gotchas. |
| `07-neuron-monitor-schema.txt` | Complete key-path schema across all metric groups. |

## Provenance and redaction

These files were transcribed from the probe session's command output rather
than copied off the instance filesystem — the instance was terminated before
an archive was pulled, so it can no longer be reached. Key names, structure
and all measurement values are verbatim.

Identifier **values** are redacted, because this repository is public:
instance ID, AMI ID, subnet ID, region, availability zone, device serial
number, model UUID, and filesystem paths that contain a username. Every
redaction is marked `<REDACTED-...>` in place, so the shape of the data is
preserved and nothing is silently dropped.

## Not captured

A relaunch (~$0.20) would be needed for any of these:

- `neuron-profile` output — the hardware profiler, the most likely source of
  per-engine counters we have not seen.
- `neuron-bench` results — built-in execution and inference benchmarks.
- `nccom-test` — collectives; requires ≥ 2 devices, so also a quota increase.
- Sustained thermal/power behaviour over more than one 60-second refresh.
- Any multi-device topology: `connected_to` was null and `connected_devices`
  empty on this single-device instance, so NeuronLink is entirely unmeasured.
