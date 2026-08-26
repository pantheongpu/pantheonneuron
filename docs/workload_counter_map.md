# Workload reference

Generated from `kernels/registry.py` by `tools/gen_workload_table.py`.
Do not hand-edit.

`Score` column: where the number comes from.
`profile` = neuron-profile, `monitor` = neuron-monitor,
`nccom` = nccom-test, `kernel` = counted by the workload itself.

Instance columns show whether the capability gate admits the workload —
**not** whether a kernel exists. Only `baseline_metrics` is implemented;
everything else raises `NotImplementedError` on hardware.

| Workload | Suite | Unit | Score | Formula | inf2.xl | inf2.24xl | trn1.2xl | trn1.32xl |
|---|---|---|---|---|:--:|:--:|:--:|:--:|
| `baseline_metrics` | baseline | — | — | `—` | ✅ | ✅ | ✅ | ✅ |
| `tensor_virus` | core | TFLOPS | monitor | `mean(effective_flops) / 1e12` | ✅ | ✅ | ✅ | ✅ |
| `int_virus` | core | TOPS | monitor | `mean(effective_flops) / 1e12` | ✅ | ✅ | ✅ | ✅ |
| `pulse_virus` | core | TFLOPS | monitor | `mean(effective_flops) / 1e12; throttle_acti…` | ✅ | ✅ | ✅ | ✅ |
| `transformer_virus` | core | TFLOPS | monitor | `mean(effective_flops) / 1e12` | ✅ | ✅ | ✅ | ✅ |
| `omni_virus` | core | TFLOPS | monitor | `mean(effective_flops) / 1e12; per-engine ac…` | ✅ | ✅ | ✅ | ✅ |
| `memory_read` | memory | GB/s | profile | `hbm_read_bytes / total_time / 1e9` | ✅ | ✅ | ✅ | ✅ |
| `memory_write` | memory | GB/s | profile | `hbm_write_bytes / total_time / 1e9` | ✅ | ✅ | ✅ | ✅ |
| `memory_read_agg` | memory | GB/s | profile | `sum(hbm_read_bytes over cores) / total_time…` | ✅ | ✅ | ✅ | ✅ |
| `memory_write_agg` | memory | GB/s | profile | `sum(hbm_write_bytes over cores) / total_tim…` | ✅ | ✅ | ✅ | ✅ |
| `all_reduce` | interconnect | GB/s | nccom | `nccom-test all_reduce busbw, averaged over …` | — | ✅ | — | ✅ |
| `p2p_thrasher` | interconnect | GB/s | nccom | `nccom-test sendrecv busbw` | — | ✅ | — | ✅ |
| `pcie_bandwidth` | interconnect | GB/s | kernel | `bytes_transferred / elapsed_s / 1e9` | ✅ | ✅ | ✅ | ✅ |
| `llm_decode` | inference | tokens/s | kernel | `tokens_generated / elapsed_s` | ✅ | ✅ | ✅ | ✅ |
| `llm_prefill` | inference | prompt-tokens/s | kernel | `prompt_tokens / elapsed_s` | ✅ | ✅ | ✅ | ✅ |
| `kv_cache_churn` | inference | cache-updates/s | kernel | `cache_updates / elapsed_s` | ✅ | ✅ | ✅ | ✅ |
| `fused_attention` | inference | attention-tiles/s | kernel | `attention_tiles / elapsed_s` | ✅ | ✅ | ✅ | ✅ |
| `quantized_gemm` | inference | quantized-ops/s | kernel | `quantized_ops / elapsed_s` | ✅ | ✅ | ✅ | ✅ |
| `serving_mix` | inference | requests/s | kernel | `requests_completed / elapsed_s` | ✅ | ✅ | ✅ | ✅ |
| `speculative_decode` | inference | verified-tokens/s | kernel | `verified_tokens / elapsed_s` | ✅ | ✅ | ✅ | ✅ |
| `moe_router` | inference | routed-tokens/s | kernel | `routed_tokens / elapsed_s` | ✅ | ✅ | ✅ | ✅ |
| `transformer_train_step` | training | train-steps/s | kernel | `steps_completed / elapsed_s` | — | — | ✅ | ✅ |
| `allocation_fragmentation` | runtime | allocation-events/s | kernel | `allocation_events / elapsed_s` | ✅ | ✅ | ✅ | ✅ |
| `graph_replay` | runtime | graph-steps/s | monitor | `delta(completed) / period` | ✅ | ✅ | ✅ | ✅ |
| `rag_embedding` | ai_auxiliary | embedding-vectors/s | kernel | `vectors_embedded / elapsed_s` | ✅ | ✅ | ✅ | ✅ |
| `vision_encoder` | ai_auxiliary | image-tiles/s | kernel | `image_tiles / elapsed_s` | ✅ | ✅ | ✅ | ✅ |

A `—` in an instance column means the capability gate skips it: `all_reduce` and `p2p_thrasher` need 2+ devices for NeuronLink, and `transformer_train_step` needs a Trainium part.

## Pinned problems

A Score is comparable across platforms only if both ran the same problem, so shape and dtype travel with the score into the report.

| Workload | Problem |
|---|---|
| `tensor_virus` | op=matmul, shape=[8192, 8192, 8192], dtype=bf16 |
| `int_virus` | op=matmul, shape=[8192, 8192, 8192], dtype=int8 |
| `pulse_virus` | op=matmul, shape=[8192, 8192, 8192], dtype=bf16, duty_cycle=0.5, period_s=2 |
| `transformer_virus` | hidden=4096, heads=32, seq=2048, dtype=bf16 |
| `omni_virus` | op=mixed, shape=[8192, 8192, 8192], dtype=bf16, engines=all |
| `memory_read` | bytes=8589934592, dtype=bf16, cores=1 |
| `memory_write` | bytes=8589934592, dtype=bf16, cores=1 |
| `memory_read_agg` | bytes=8589934592, dtype=bf16, cores=all |
| `memory_write_agg` | bytes=8589934592, dtype=bf16, cores=all |
| `all_reduce` | op=all_reduce, bytes_min=1048576, bytes_max=8388608, dtype=fp32 |
| `p2p_thrasher` | op=sendrecv, bytes=67108864, dtype=fp32 |
| `pcie_bandwidth` | bytes=1073741824, direction=bidirectional |
| `llm_decode` | hidden=4096, layers=32, batch=1, context=2048, dtype=bf16 |
| `llm_prefill` | hidden=4096, layers=32, batch=1, prompt=2048, dtype=bf16 |
| `kv_cache_churn` | hidden=4096, heads=32, context=4096, dtype=bf16 |
| `fused_attention` | heads=32, seq=2048, head_dim=128, dtype=bf16 |
| `quantized_gemm` | op=matmul, shape=[4096, 4096, 4096], dtype=int8 |
| `serving_mix` | prefill_ratio=0.2, batch=8, prompt=1024, decode=256 |
| `speculative_decode` | draft_len=4, hidden=4096, dtype=bf16 |
| `moe_router` | experts=8, top_k=2, hidden=4096, tokens=4096 |
| `transformer_train_step` | hidden=4096, layers=8, batch=4, seq=2048, dtype=bf16 |
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
