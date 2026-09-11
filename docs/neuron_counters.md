# Neuron performance counters

Measured on an **inf2.xlarge** (1 NeuronDevice, 2 NeuronCores-v2, Neuron
runtime 2.30.51, DLAMI Neuron Ubuntu 22.04 20260227) under a sustained
`torch-neuronx` matmul load. Reproduce with
[`tools/probe_counters.sh`](../tools/probe_counters.sh).

## Mapping to pantheongpu

### Counters that carry over

| pantheongpu | Neuron source | Notes |
|---|---|---|
| `avg/max_gpu_util` | `neuroncore_counters.*.neuroncore_utilization` | Per NeuronCore. Measured 89.4% under load. |
| `peak_mem_used` | `memory_used.neuron_runtime_used_bytes.device` | Plus an 11-way breakdown GPU has no analogue for. |
| `mem_total` | `neuron_hardware_info.neuron_device_memory_size` | 34359738368 (32 GB). |
| `pcie_gen` / `pcie_width` | `/sys/bus/pci/devices/<bdf>/current_link_speed`, `current_link_width` | Not via neuron-monitor. Observed Gen4 x8 current, Gen5 x8 max. |

### Counters Neuron has and pantheongpu does not

| Counter | Source | Why it matters |
|---|---|---|
| `effective_flops` | `neuroncore_counters.*.effective_flops` | **Achieved FLOP/s per core.** 875,590,474,948 (~875 GFLOP/s) under load. Undocumented — not in the CloudWatch metric set. |
| Latency percentiles | `execution_stats.latency_stats.device_latency` and `.total_latency` | p0/p1/p25/p50/p75/p99/p100, both device-only and end-to-end. |
| ECC, four counters | `neuron_hw_counters.neuron_devices[]` | `mem_ecc_corrected`, `mem_ecc_uncorrected`, `sram_ecc_corrected`, `sram_ecc_uncorrected` — device memory *and* on-chip SRAM. |
| Typed error summary | `execution_stats.error_summary` | `hardware`, `model`, `numerical`, `runtime`, `transient`. |
| Execution outcomes | `execution_stats.execution_summary` | `completed`, `completed_with_err`, `completed_with_num_err`, `failed_to_queue`, `incorrect_input`, `timed_out`. |
| Memory by category | `memory_used.*.usage_breakdown` | tensors, constants, model_code, shared/nonshared scratchpad, dma_rings, collectives, notifications, driver, runtime, uncategorized — each with present/peak/total. |

### Counters that do not exist

| pantheongpu | Status |
|---|---|
| `avg/max_temp`, `thermal_rise`, `avg/max_mem_temp` | **No temperature sensor anywhere.** Checked neuron-monitor, the driver sysfs tree, `/sys/class/hwmon` (only `nvme`), `/sys/class/thermal` (cooling devices only, no zones), and every tool in `/opt/aws/neuron/bin`. |
| `max_fan` | Not exposed by any interface. |
| `max_volts_core`, `max_volts_soc` | Not exposed by any interface. |

### Recovered by the profiler

A second probe found these through `neuron-profile`, which is a
per-execution capture interface rather than continuous telemetry. They are
**not** available through `neuron-monitor` or sysfs, which is why the first
pass recorded them as absent.

| pantheongpu | Neuron source | Notes |
|---|---|---|
| `throttle_reason`, `throttle_time` | `throttle_active_nc0_time_ns` and 5 related | Per NeuronCore. Measured 774 ns active. |
| `avg/min/max_clk` | `neuroncore_cycle_count` ÷ `total_time` | Derived, not reported: 209594 / 0.000149710075 s = **1.400 GHz**. |

See [`data/probe-2026-08-26-tools/`](../data/probe-2026-08-26-tools/) for the
full 108-counter set, including MFU/HFU/MBU, per-engine instruction counts
for all five engines, the SBUF/PSUM/spill memory hierarchy, and 17 DMA
counters.

### The `_percent` counters are fractions

**Every `neuron-profile` counter whose name ends in `_percent` reports a
0–1 fraction, not a percentage.** Measured on trn1.2xlarge 2026-09-10:

| counter | torch.matmul, 4096³ bf16 |
|---|--:|
| `tensor_engine_active_time_percent` | 0.451 |
| `dma_active_time_percent` | 0.205 |
| `mfu_max_achievable_estimated_percent` | **1** |

A "maximum achievable" of exactly 1 is 100%, and a tensor engine active
0.45% of the time could not have delivered the 51.8 TFLOPS the same run
measured. The name says percent; the value is a fraction.

This was found by getting it wrong. The probe that captured these
compared them against a percentage threshold, printed *"tensor engine
active: NKI 0.379%"*, and concluded the engines were comparable and the
gap lay elsewhere. Read on the right scale, the same numbers say the
opposite — see `docs/the_headline_number_is_the_kernel.md`.

**Per-transfer DMA detail is in the full trace, not the summary.**
`neuron-profile view --output-format json` writes **`ntff.json` into the
current working directory** and prints nothing useful to stdout, so a
caller reading stdout gets an empty trace. Its top-level `dma` list has
one record per transfer (`transfer_size`, `duration`, `dma_queue`,
`timestamp`, ...), which is what `summary-json` leaves as `None` for NKI
graphs. The file is large: 173 MB for one 4096³ NKI matmul execution.

Nothing in production reads these counters yet
(`omni_virus.engine_activity` is deliberately unwired — see
`COUNTERS_THE_DECLARED_SOURCE_CANNOT_SUPPLY`), so the trap has only
caught a probe so far. `tests/test_neuron_counters_units.py` exists so it
cannot catch the first real consumer.

## Power: partially recoverable

`/sys/devices/virtual/neuron_device/neuron0/stats/power/utilization` is
undocumented but real:

```
POWER_STATUS_VALID,<epoch>,<min>,<max>,<avg>
```

| State | Reading |
|---|---|
| Idle | `0.00, 2.06, 0.56` |
| Under load | `0.86, 25.00, 18.86` |

Clearly responsive. Two caveats:

1. **It refreshes once per 60 seconds.** The embedded timestamp steps in
   exact 60-second increments and the values are frozen between steps.
   Usable for sustained soaks; useless for transient detection, which is
   what `pulse_virus` is for.
2. **Units are not documented.** The ordering is consistent with
   min < avg < max, and the values behave like percentages, but nothing
   confirms whether the scale is percent of a power budget or something
   else. Do not report it as watts.

`energy_wh` cannot be derived from this without a unit definition.

### `execution_summary.completed` is a tally per period

**neuron-monitor's `execution_stats.execution_summary.completed` counts the
executions in its sampling period. It is not a running total.** The
AWS guide says only "executions completed successfully". Measured on
trn1.2xlarge 2026-09-10: graph_replay, 60,000 replays, the monitor
sampling every ~5 s through the compile and an idle tail:

```
completed:  0 ... 0, 6657, 15409, 15273, 15199, 7465, 0, 0     sum 60,003
```

It falls back to zero when the work stops, and the tallies sum to the
replays run plus three setup graphs. Whole periods divided by their
`period` give 3082, 3055 and 3040 per second, against the kernel's own
3057.

This suite read it as cumulative: its maximum as the run's total, and
last minus first as a rate's numerator. The maximum is one period's
tally, which is where "about four replays per NEFF execution" came from.
Last minus first is the difference between two periods' tallies, which
put graph_replay's declared Score anywhere from 729 to 1506, or nowhere.
The same stats block's `error_summary` counts were already being summed,
which is right for per-period tallies and inconsistent with treating
`completed` as a total. That inconsistency sat in one function the whole
time.

**A sample carries one `neuron_runtime_data` block per runtime on the
device, and the tally is per runtime.** The device's count for a period is
their sum. Measured on trn1.2xlarge 2026-09-11, `memory_read`'s 342
samples were 310 with one runtime and 32 with two -- the second being the
`neuron-profile` capture running beside the harness. A reader taking one
block per sample sees a fraction of the device; a rate built from one
entry per block divides one period's completions by two periods' worth of
time (`neuron_monitor.execution_rate`).

**The ~5 s periods above are what the harness was receiving, not what
neuron-monitor floors at.** The config asked for `"1.0s"`, which the tool
ignores in favour of its 5 s default; `"1s"` delivers 1 s. Since
2026-09-11 the period goes out in whole seconds and every row records the
period its samples carried (`sample_period_s`). A pulsed workload is
sampled over whole pulse cycles instead, since a sample covering half a
cycle reads the duty of wherever it fell.

### `effective_flops` is a rate per period, and the edge periods read low

The AWS guide defines `effective_flops` as operations per second during
the captured period, which is right. But the first and last periods of a
run are only partly busy, so they report a fraction of the rate.
Measured on trn1.2xlarge 2026-09-10, `tensor_virus` 8192³ for 30 s:

```
effective_flops (TFLOPS):  18.25, 72.34, 72.35, 71.02, 72.35, 72.57, 53.43
utilisation (%):           25,    99,    99,    97,    99,    99.5,  73
```

The mean of all seven is 61.76. The mean of the five whole periods is
72.13, against the kernel's own analytic 71.93. The declared Score was
the first, and how low it read depended on where the monitor's 5-second
grid fell against the run. That is the spread the repeats kept showing.
The Score is now the mean over whole periods, and the all-sample mean
is reported beside it (`mean_all_samples`).

### `total_time` opens with an idle startup; divide by `total_active_time`

A profiled execution's `total_time` starts about 2 ms before its first
byte moves. Measured on trn1.2xlarge 2026-09-10, `memory_read`'s kernel
graph at four sizes:

| buffer | `total_time` | `total_active_time` | wall per pass (loop) |
|---|--:|--:|--:|
| 1 GiB | 6.020 ms | 3.934 ms | 4.025 ms |
| 2 GiB | 9.950 | 7.867 | 7.968 |
| 4 GiB | 17.825 | 15.746 | 15.855 |
| 8 GiB | 33.540 | 31.463 | 31.614 |

The difference is 2.08 ms at every size. The trace's DMA throughput reads
0 across it, then a steady 273–275 GB/s. The kernel run back to back pays
about 0.08 ms per pass, so the startup belongs to the one profiled
execution. Over `total_time` the bandwidth depended on the buffer (178 to
256 GB/s). Over `total_active_time`, the union of time any engine or DMA
queue was busy, it's 273 at every size. `memory_write` shows the same
2.08 ms.

## sysfs counters that stay zero

`stats/other_info/` exposes `flop_count`, `inference_count`,
`nc_time_in_use` and `model_load_count`. **All remained 0 through 88,154
executions.** They are not populated by the torch-neuronx inference path on
runtime 2.30.51. Use `neuron-monitor`'s `effective_flops` and
`execution_summary.completed` instead.

`reset_req_count` and `reset_fail_count` do populate (1 and 0 at boot) and
are worth collecting as RAS signals.

## neuron-ls has no architecture field

`neuron-ls --json-output` returns a top-level list of devices with
`neuron_device`, `bdf`, `cpu_affinity`, `numa_node`, `connected_to`,
`nc_count`, `memory_size`, `neuroncore_ids`, `neuron_processes` — and **no
device type or architecture**. Read it from the driver instead:

```
/sys/devices/virtual/neuron_device/neuron0/info/architecture/instance_type  -> "Inf2"
/sys/devices/virtual/neuron_device/neuron0/info/architecture/device_name    -> "Inferentia2"
/sys/devices/virtual/neuron_device/neuron0/info/architecture/arch_type      -> "NDv3"
```

Note `arch_type` is the *device* generation (NDv3), not the NeuronCore
generation (v2). `neuron-monitor`'s `neuron_hardware_info` reports both
separately and is the clearer source.

## Host identifiers observed

`neuron-monitor` emits all of these in `instance_info` on **every sample**:
`ami_id`, `instance_availability_zone`, `instance_availability_zone_id`,
`instance_id`, `instance_name`, `instance_region`, `instance_type`,
`subnet_id`.

Additionally:

- `neuron-ls` prints `instance-id` in its human-readable header, and
  `neuron_processes[].command` carries the full command line (and therefore
  filesystem paths and usernames).
- sysfs exposes `info/serial_number` — a per-device identifier.
- `memory_used.loaded_models[].name` is a filesystem path to the compiled
  NEFF.

All are scrubbed at ingest; see `tests/test_report_privacy.py` and
`tests/test_real_hardware_schema.py`.
