# Tool sweep — 2026-08-26 (second probe)

Every tool in `/opt/aws/neuron/bin` exercised on an inf2.xlarge. This probe
exists because the first one never ran `neuron-profile` — it was visible in
the very first command's output and was not followed up.

**It corrected four findings from the first probe.** The profiler exposes
108 counters that no other Neuron interface reports.

| | |
|---|---|
| Instance | inf2.xlarge |
| Tools | 2.28.23.0 (kaena-tools/2.28) |
| Runtime / driver | 2.30.51 / 2.26.0 |
| Compiler | 2.23.6484.0 |
| Duration / cost | ~24 min, ~$0.30 |
| Instance | terminated; IAM role, profile and security group deleted |

## Corrections to probe 1

| Finding in probe 1 | Actual |
|---|---|
| Throttle counters absent | **6 throttle counters exist**, per NeuronCore, in the profiler |
| Clock absent | **Derivable** — `neuroncore_cycle_count / total_time` = 1.400 GHz |
| NeuronLink entirely unmeasured | Core-to-core collectives measurable on one device — 50.66 GB/s. Device-to-device still unmeasured. |
| No GPU-style memory structures to target | SBUF, PSUM and spill/reload byte counters all exist |

Probe 1's conclusions were drawn from `neuron-monitor` and sysfs. Both are
*continuous telemetry* interfaces. The profiler is a *per-execution capture*
interface, and it is where the hardware detail lives. Neither is a superset
of the other.

## Files

| File | Contents |
|---|---|
| `01-tool-inventory.txt` | Every tool, its status, and three environment traps |
| `02-profiler-counters.txt` | All 108 profiler counters, grouped |
| `03-profile-session-summary.txt` | Session decode, per-engine table, DMA queue structure |
| `04-collectives-nccom-test.txt` | Collectives bandwidth sweep, 1 MiB → 8 MiB |
| `05-neuron-dbg.txt` | Debugger interface and raw register access |
| `06-neuron-bench.txt` | Benchmark harness, its BPF failure, latency data |

## Still absent after exhausting every tool

**Temperature, fan speed, voltage.** No interface reports them —
`neuron-monitor`, sysfs, hwmon, thermal zones, or the profiler.

`neuron-dbg --device-address` allows arbitrary MMIO reads and is the only
remaining path, but the address map is unpublished. That is reverse
engineering, not instrumentation, and is not a basis for a test suite.

## Not captured

- Device-to-device NeuronLink (needs ≥ 2 devices → quota increase).
- `neuron-profile view --output-format perfetto|parquet|db`.
- `neuron-dbg` register reads against a live workload.
- `neuron-dump`.
