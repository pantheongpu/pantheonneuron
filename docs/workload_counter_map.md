# Workload → counter map

Which measured counter produces each workload's `Score`. Generated from
`kernels/registry.py` — the registry is the source of truth, this file is
the readable view of it.

Four sources, because no single Neuron interface covers everything:

| Source | What it is | Why it's needed |
|---|---|---|
| `neuron-profile` | Per-execution capture | Only source for HBM bytes, per-engine time, cycles, throttle |
| `neuron-monitor` | Continuous stream | Only source for `effective_flops` and execution counts |
| `nccom-test` | Collectives benchmark | Reports `busbw` directly |
| `workload` | Counted by the kernel | No hardware counter measures tokens, steps or requests |

## `neuron-profile`

| Workload | Unit | Formula | Counters |
|---|---|---|---|
| `memory_read` | GB/s | `hbm_read_bytes / total_time / 1e9` | `hbm_read_bytes`<br>`total_time` |
| `memory_write` | GB/s | `hbm_write_bytes / total_time / 1e9` | `hbm_write_bytes`<br>`total_time` |
| `memory_read_agg` | GB/s | `sum(hbm_read_bytes over cores) / total_time / 1e9` | `hbm_read_bytes`<br>`total_time` |
| `memory_write_agg` | GB/s | `sum(hbm_write_bytes over cores) / total_time / 1e9` | `hbm_write_bytes`<br>`total_time` |

## `neuron-monitor`

| Workload | Unit | Formula | Counters |
|---|---|---|---|
| `tensor_virus` | TFLOPS | `mean(effective_flops) / 1e12` | `neuroncore_counters.*.effective_flops` |
| `int_virus` | TOPS | `mean(effective_flops) / 1e12   # int8 ops, reported as TOPS` | `neuroncore_counters.*.effective_flops` |
| `pulse_virus` | TFLOPS | `mean(effective_flops) / 1e12; throttle_active_nc0_time_ns recorded alongside` | `neuroncore_counters.*.effective_flops`<br>`throttle_active_nc0_time_ns` |
| `transformer_virus` | TFLOPS | `mean(effective_flops) / 1e12` | `neuroncore_counters.*.effective_flops` |
| `omni_virus` | TFLOPS | `mean(effective_flops) / 1e12; per-engine active_time_percent recorded alongside` | `neuroncore_counters.*.effective_flops`<br>`tensor_engine_active_time_percent`<br>`vector_engine_active_time_percent`<br>`scalar_engine_active_time_percent`<br>`gpsimd_engine_active_time_percent` |
| `graph_replay` | graph-steps/s | `delta(completed) / period` | `execution_stats.execution_summary.completed`<br>`execution_stats.period` |

## `nccom-test`

| Workload | Unit | Formula | Counters |
|---|---|---|---|
| `all_reduce` | GB/s | `nccom-test all_reduce busbw, averaged over the size sweep` | `busbw` |
| `p2p_thrasher` | GB/s | `nccom-test sendrecv busbw` | `busbw` |

## `workload`

| Workload | Unit | Formula | Counters |
|---|---|---|---|
| `pcie_bandwidth` | GB/s | `bytes_transferred / elapsed_s / 1e9   # device DMA counters are device-side only` | `bytes_transferred`<br>`elapsed_s` |
| `llm_decode` | tokens/s | `tokens_generated / elapsed_s` | `tokens_generated`<br>`elapsed_s` |
| `llm_prefill` | prompt-tokens/s | `prompt_tokens / elapsed_s` | `prompt_tokens`<br>`elapsed_s` |
| `kv_cache_churn` | cache-updates/s | `cache_updates / elapsed_s` | `cache_updates`<br>`elapsed_s` |
| `fused_attention` | attention-tiles/s | `attention_tiles / elapsed_s` | `attention_tiles`<br>`elapsed_s` |
| `quantized_gemm` | quantized-ops/s | `quantized_ops / elapsed_s` | `quantized_ops`<br>`elapsed_s` |
| `serving_mix` | requests/s | `requests_completed / elapsed_s` | `requests_completed`<br>`elapsed_s` |
| `speculative_decode` | verified-tokens/s | `verified_tokens / elapsed_s` | `verified_tokens`<br>`elapsed_s` |
| `moe_router` | routed-tokens/s | `routed_tokens / elapsed_s` | `routed_tokens`<br>`elapsed_s` |
| `transformer_train_step` | train-steps/s | `steps_completed / elapsed_s` | `steps_completed`<br>`elapsed_s` |
| `allocation_fragmentation` | allocation-events/s | `allocation_events / elapsed_s` | `allocation_events`<br>`elapsed_s` |
| `rag_embedding` | embedding-vectors/s | `vectors_embedded / elapsed_s` | `vectors_embedded`<br>`elapsed_s` |
| `vision_encoder` | image-tiles/s | `image_tiles / elapsed_s` | `image_tiles`<br>`elapsed_s` |

## Not scored

`baseline_metrics` applies no load and reports telemetry only. It has no
unit and no Score.

## Measured values

`data/baselines.json` holds what each counter actually read on hardware.
**Those are probe observations, not benchmark results** — the load was an
untuned matmul at 0.0049% MFU. They exist to prove each counter is
readable and to catch plumbing regressions.
