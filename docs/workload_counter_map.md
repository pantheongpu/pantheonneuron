# Workload reference

Generated from `kernels/registry.py` by `tools/gen_workload_table.py`.
Do not hand-edit.

`Score` column: where the number comes from.
`profile` = neuron-profile, `monitor` = neuron-monitor,
`nccom` = nccom-test, `kernel` = counted by the workload itself.

Instance columns show whether the capability gate admits the workload —
**not** whether its kernel has met hardware. Every workload in the registry has an implementation (26 of 26); a name absent from `pantheon_neuron.IMPLEMENTED` raises `NotImplementedError` on hardware rather than reporting a silent PASS. See the README for which kernels have actually run on a device.

| Workload | Suite | Unit | Score | Measured | inf2.xl | inf2.24xl | trn1.2xl | trn1.32xl |
|---|---|---|---|--:|:--:|:--:|:--:|:--:|
| `baseline_metrics` | baseline | — | — | — | ✅ | ✅ | ✅ | ✅ |
| `tensor_virus` | core | TFLOPS | monitor | **0.8756** | ✅ | ✅ | ✅ | ✅ |
| `int_virus` | core | TOPS | monitor | — | ✅ | ✅ | ✅ | ✅ |
| `pulse_virus` | core | TFLOPS | monitor | — | ✅ | ✅ | ✅ | ✅ |
| `transformer_virus` | core | TFLOPS | monitor | — | ✅ | ✅ | ✅ | ✅ |
| `omni_virus` | core | TFLOPS | monitor | — | ✅ | ✅ | ✅ | ✅ |
| `memory_read` | memory | GB/s | profile | **56.58** | ✅ | ✅ | ✅ | ✅ |
| `memory_write` | memory | GB/s | profile | **0.1094** | ✅ | ✅ | ✅ | ✅ |
| `memory_read_agg` | memory | GB/s | profile | — | ✅ | ✅ | ✅ | ✅ |
| `memory_write_agg` | memory | GB/s | profile | — | ✅ | ✅ | ✅ | ✅ |
| `all_reduce` | interconnect | GB/s | nccom | **50.66** | — | ✅ | — | ✅ |
| `p2p_thrasher` | interconnect | GB/s | nccom | — | — | ✅ | — | ✅ |
| `pcie_bandwidth` | interconnect | GB/s | kernel | — | ✅ | ✅ | ✅ | ✅ |
| `llm_decode` | inference | tokens/s | kernel | — | ✅ | ✅ | ✅ | ✅ |
| `llm_prefill` | inference | prompt-tokens/s | kernel | — | ✅ | ✅ | ✅ | ✅ |
| `kv_cache_churn` | inference | cache-updates/s | kernel | — | ✅ | ✅ | ✅ | ✅ |
| `fused_attention` | inference | attention-tiles/s | kernel | — | ✅ | ✅ | ✅ | ✅ |
| `quantized_gemm` | inference | quantized-ops/s | kernel | — | ✅ | ✅ | ✅ | ✅ |
| `serving_mix` | inference | requests/s | kernel | — | ✅ | ✅ | ✅ | ✅ |
| `speculative_decode` | inference | verified-tokens/s | kernel | — | ✅ | ✅ | ✅ | ✅ |
| `moe_router` | inference | routed-tokens/s | kernel | — | ✅ | ✅ | ✅ | ✅ |
| `transformer_train_step` | training | train-steps/s | kernel | — | — | — | ✅ | ✅ |
| `allocation_fragmentation` | runtime | allocation-events/s | kernel | — | ✅ | ✅ | ✅ | ✅ |
| `graph_replay` | runtime | graph-steps/s | monitor | **815.5** | ✅ | ✅ | ✅ | ✅ |
| `rag_embedding` | ai_auxiliary | embedding-vectors/s | kernel | — | ✅ | ✅ | ✅ | ✅ |
| `vision_encoder` | ai_auxiliary | image-tiles/s | kernel | — | ✅ | ✅ | ✅ | ✅ |

**Measured** applies each workload's declared formula to the counters actually read during the probe. Only five workloads have one, because only their counters were captured. These are **not Scores** — no kernel ran, and the load was an untuned matmul at 0.0049% MFU rather than the pinned problem each workload declares. A real Score will differ by orders of magnitude.

A `—` in an instance column means the capability gate skips it: `all_reduce` and `p2p_thrasher` need 2+ devices for NeuronLink, and `transformer_train_step` needs a Trainium part.

## How a monitor-sourced Score is read

The five compute workloads declare `mean(effective_flops) / 1e12`. That counter exists only in the neuron-monitor stream — it is absent from the CloudWatch metric set, and sysfs leaves `flop_count` at zero — so unlike the bandwidth kernels, their Score cannot come from the kernel. `pantheon_neuron.monitor_score` reads it from the telemetry the run just collected, after the monitor stops.

`mean` is across NeuronCores. A part reports one series per core, and a workload that saturates the device runs on all of them; summing would make a two-core part look twice as fast as the same silicon reported per core.

Two cases deliberately produce no Score rather than a number:

- **The counter is absent** — a mock run, telemetry disabled, or a kernel that never reached the Tensor Engine. The row records a PASS with no Score and says why.
- **The workload failed** — telemetry keeps sampling through a failure, so without a status gate a FAIL row would carry whatever the monitor caught and read as a measurement.

`graph_replay` is also neuron-monitor-sourced but is **not** on this path: its formula is `delta(completed) / period` in graph-steps/s. The gate matches on the declared counter, not on the source, so the FLOPS arithmetic cannot reach it.

## The declared profiler Score has never been produced by a run

`memory_read` and `memory_write` declare `neuron-profile` as their Score source, and every run so far has degraded to the analytic fallback instead. Two causes, found in that order:

- **inf2.xlarge 2026-09-07** — `neuron-profile capture` replays the NEFF, which needs NeuronCores, and the workload process held them all (`Logical Neuron Core(s) not available - Requested:2 Available:0`). That is why the profiler figures in `data/baselines.json` exist at all: they came from standalone probe sessions, never from a scored run. `kernels/cores.py` now reserves a core to close it.
- **trn1.2xlarge 2026-09-08** — the reservation worked and the capture ran for the first time, against the wrong graph. `verify_profile_covers_plan` refused it: *profiled graph moved 4 bytes against a plan of 8589934592*. Narrowing NEFF selection by compile timestamp is not enough to identify the kernel's own graph. Scores from a declared hardware source that run: 0 of 4.

So the fallback is not a rare degradation, it is the only path these Scores have ever taken — but it is now a loud one. The failure is a refusal rather than a plausible bandwidth computed from four bytes, and the row's `Score Method` names the method actually used.

## Reserving the core costs a selection

The Neuron runtime reads `NEURON_RT_VISIBLE_CORES` once at initialisation, so the workload/profiler split is fixed for a whole run and cannot be renegotiated per workload. `memory_read_agg` and `memory_write_agg` declare `cores: "all"`, and holding a core back from them would report the aggregate of all-but-one core under a name that says otherwise — so their presence turns the reservation off for the entire run.

**`--test all` and `--test memory` both select them**, which means neither invocation can reach the profiler for `memory_read` or `memory_write`. The declared source is available only to a selection with no `cores: "all"` workload in it, such as `--test memory_read`. `pantheon_neuron.reservation_cost` derives which workloads are paying and the run names them on the console.

## Where the comparison does not hold

12 workloads exist on both platforms under the same name and must **not** be compared: `fused_attention`, `graph_replay`, `kv_cache_churn`, `llm_decode`, `llm_prefill`, `moe_router`, `quantized_gemm`, `rag_embedding`, `serving_mix`, `speculative_decode`, `transformer_train_step`, `vision_encoder`.

pantheongpu v1.0.19 replaced their units with a single `ai-ops/s`. Ten of its AI workloads shared one kernel body and six compiled to byte-identical SASS, so what it reports is generic synthetic throughput rather than the quantity each name suggests. The Neuron implementations count the real thing — tokens generated, cache updates applied, training steps completed.

## Where the units match and the quantities do not

4 workloads join cleanly on (Test Name, Unit) and should not be read as a comparison. This is the worse of the two failure modes: a failed join is visible, a successful join between unlike quantities is not.

| Workload | Why the two numbers differ |
|---|---|
| `int_virus` | pantheongpu runs integer FMA chains counted from occupancy; this is a dense uint8 GEMM on the Tensor Engine. |
| `omni_virus` | pantheongpu sums analytic per-engine op counts; this drives four engines in one dependent chain and reads effective_flops. |
| `pulse_virus` | pantheongpu duty-cycles scalar fp32 fmaf chains; this duty-cycles a dense bf16 GEMM. |
| `tensor_virus` | pantheongpu runs __hfma2 chains on the FP16 vector lanes with no matrix at all, counted analytically from occupancy; this is a dense systolic GEMM read from a hardware counter. |

Nothing here changes what joins. See `docs/cross_platform_comparability.md` for the evidence and the three options, none of them taken.

Copying `ai-ops/s` here would restore the join and compare unlike quantities, so these keep their own units and are listed in `registry.NOT_COMPARABLE_WITH_GPU`. `tests/test_score_schema.py` fails if a unit diverges without being declared there.

## Pinned problems

A Score is comparable across platforms only if both ran the same problem, so shape and dtype travel with the score into the report.

| Workload | Problem |
|---|---|
| `tensor_virus` | op=matmul, shape=[8192, 8192, 8192], dtype=bf16 |
| `int_virus` | op=matmul, shape=[8192, 8192, 8192], dtype=uint8 |
| `pulse_virus` | op=matmul, shape=[8192, 8192, 8192], dtype=bf16, duty_cycle=0.5, period_s=2 |
| `transformer_virus` | hidden=4096, heads=32, seq=2048, dtype=bf16 |
| `omni_virus` | op=mixed, shape=[8192, 8192, 8192], dtype=bf16, engines=all |
| `memory_read` | bytes=8589934592, dtype=bf16, cores=1 |
| `memory_write` | bytes=4294967296, dtype=bf16, cores=1 |
| `memory_read_agg` | bytes=8589934592, dtype=bf16, cores=all |
| `memory_write_agg` | bytes=4294967296, dtype=bf16, cores=all |
| `all_reduce` | op=all_reduce, bytes_min=1048576, bytes_max=8388608, dtype=fp32 |
| `p2p_thrasher` | op=sendrecv, bytes=67108864, dtype=fp32 |
| `pcie_bandwidth` | bytes=1073741824, direction=bidirectional |
| `llm_decode` | hidden=4096, layers=32, batch=1, context=2048, dtype=bf16 |
| `llm_prefill` | hidden=4096, layers=32, batch=1, prompt=2048, dtype=bf16 |
| `kv_cache_churn` | hidden=4096, heads=32, context=4096, layers=32, ring_slots=8, dtype=bf16 |
| `fused_attention` | heads=32, seq=2048, head_dim=128, dtype=bf16 |
| `quantized_gemm` | op=matmul, shape=[4096, 4096, 4096], dtype=int8 |
| `serving_mix` | prefill_ratio=0.2, batch=8, prompt=1024, decode=256 |
| `speculative_decode` | draft_len=4, hidden=4096, dtype=bf16 |
| `moe_router` | experts=8, top_k=2, hidden=4096, tokens=4096 |
| `transformer_train_step` | hidden=4096, layers=4, batch=1, seq=2048, dtype=bf16 |
| `allocation_fragmentation` | allocations=10000, size_min=4096, size_max=16777216 |
| `graph_replay` | hidden=2048, replays=10000, dtype=bf16 |
| `rag_embedding` | dim=1024, batch=256, dtype=bf16 |
| `vision_encoder` | resolution=224, patch=14, batch=64, dtype=bf16 |

## Counters referenced

| Workload | Counters |
|---|---|
| `tensor_virus` | `neuroncore_counters.*.effective_flops` |
| `int_virus` | `neuroncore_counters.*.effective_flops` |
| `pulse_virus` | `neuroncore_counters.*.effective_flops`<br>`throttle_active_nc0_time_ns` |
| `transformer_virus` | `neuroncore_counters.*.effective_flops` |
| `omni_virus` | `neuroncore_counters.*.effective_flops`<br>`tensor_engine_active_time_percent`<br>`vector_engine_active_time_percent`<br>`scalar_engine_active_time_percent`<br>`gpsimd_engine_active_time_percent` |
| `memory_read` | `hbm_read_bytes`<br>`total_time` |
| `memory_write` | `hbm_write_bytes`<br>`total_time` |
| `memory_read_agg` | `hbm_read_bytes`<br>`total_time` |
| `memory_write_agg` | `hbm_write_bytes`<br>`total_time` |
| `all_reduce` | `busbw` |
| `p2p_thrasher` | `busbw` |
| `pcie_bandwidth` | `bytes_transferred`<br>`elapsed_s` |
| `llm_decode` | `tokens_generated`<br>`elapsed_s` |
| `llm_prefill` | `prompt_tokens`<br>`elapsed_s` |
| `kv_cache_churn` | `cache_updates`<br>`elapsed_s` |
| `fused_attention` | `attention_tiles`<br>`elapsed_s` |
| `quantized_gemm` | `quantized_ops`<br>`elapsed_s` |
| `serving_mix` | `requests_completed`<br>`elapsed_s` |
| `speculative_decode` | `verified_tokens`<br>`elapsed_s` |
| `moe_router` | `routed_tokens`<br>`elapsed_s` |
| `transformer_train_step` | `steps_completed`<br>`elapsed_s` |
| `allocation_fragmentation` | `allocation_events`<br>`elapsed_s` |
| `graph_replay` | `execution_stats.execution_summary.completed`<br>`execution_stats.period` |
| `rag_embedding` | `vectors_embedded`<br>`elapsed_s` |
| `vision_encoder` | `image_tiles`<br>`elapsed_s` |

## No Neuron equivalent

17 pantheongpu workloads have no counterpart here. Asking for one by name explains why rather than reporting an unknown test.

| pantheongpu workload | Reason |
|---|---|
| `atomic_virus` | NKI does not expose GPU-style device-wide atomics. |
| `cache_lat` | Neuron's on-chip memory hierarchy is not a GPU-style cache. |
| `fp64_virus` | Neuron devices have no FP64 units. |
| `incinerator` | GPU-specific thermal virus with no documented Neuron analogue. |
| `media_enc_virus` | No media encode block. |
| `memory_bank_thrash` | HBM bank mapping is not exposed on Neuron. |
| `memory_cache_fracture` | No GPU-style L2 partitioning to fracture. |
| `memory_pc_pingpong` | No partition-camping analogue. |
| `memory_retention_bake` | Requires refresh-interval control Neuron does not expose. |
| `memory_thermal_asym` | Depends on GPU-specific per-stack thermal telemetry. |
| `memory_tsv_thrasher` | TSV-level access patterns are not addressable. |
| `mma_virus` | MMA is an NVIDIA tensor-core instruction; Neuron's Tensor Engine has a different ISA. |
| `rt_virus` | No ray-tracing hardware. |
| `scheduler` | No warp scheduler; Neuron dispatch is compiler-scheduled. |
| `sfu_stress` | SFU is an NVIDIA SM structure; Neuron's Scalar/GpSimd engines are not equivalent. |
| `tlb_avalanche` | No exposed TLB behaviour to target. |
| `voltage` | No equivalent voltage-rail control exposed by the Neuron driver. |

## Measured values

`data/baselines.json` records what each counter actually read during the probes. **Those are observations, not benchmark results** — the probe load was an untuned matmul at 0.0049% MFU. They prove each counter is readable and catch plumbing regressions; they are not Inferentia2's throughput.
