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
| `avg/max_temp`, `thermal_rise`, `avg/max_mem_temp` | **No temperature sensor anywhere.** Checked neuron-monitor, the driver sysfs tree, `/sys/class/hwmon` (only `nvme`), and `/sys/class/thermal` (cooling devices only, no zones). |
| `avg/min/max_clk` | Not exposed. |
| `max_fan` | Not exposed. |
| `throttle_reason`, `throttle_time` | Not exposed. |
| `max_volts_core`, `max_volts_soc` | Not exposed. |

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
