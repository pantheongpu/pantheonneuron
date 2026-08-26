"""Workload registry with capability gating.

Workload and suite names deliberately match the pantheongpu suite wherever
the underlying concept is the same, so a Neuron result and a GPU result for
a given name are comparing like with like.

That constraint cuts both ways: where a GPU workload targets a structure
Neuron does not have -- SFUs, ray-tracing cores, warp schedulers, HBM bank
conflicts, FP64 -- there is deliberately no Neuron workload of that name.
Reusing the name for something else would make cross-platform comparison
worse, not better.  See NO_NEURON_EQUIVALENT for the full list and reasons.
"""

import dataclasses
import typing


# Where a Score comes from. Four sources, because Neuron has no single
# interface that covers everything:
#
#   PROFILER  neuron-profile summary-json -- per-execution capture. The only
#             source for HBM bytes, per-engine time, cycles and throttle.
#   MONITOR   neuron-monitor -- continuous stream. effective_flops and
#             execution_summary live here and nowhere else.
#   NCCOM     nccom-test -- the collectives benchmark; reports busbw directly.
#   INTERNAL  counted by the workload itself (tokens, steps, requests). No
#             hardware counter measures these; the kernel must report them.
PROFILER = "neuron-profile"
MONITOR = "neuron-monitor"
NCCOM = "nccom-test"
INTERNAL = "workload"


@dataclasses.dataclass(frozen=True)
class ScoreSource:
    """How a workload's Score is produced from measured counters.

    ``formula`` is the arithmetic, written against ``counters``. It is
    documentation and a review target, not evaluated code -- the kernel
    implements it. Recording it here means a Score can be audited without
    reading the kernel, and a counter rename shows up as a broken reference
    rather than a silently wrong number.
    """

    source: str
    counters: typing.Tuple[str, ...]
    formula: str


@dataclasses.dataclass(frozen=True)
class Workload:
    """One workload.

    ``unit`` and ``problem`` exist so a Neuron result can be compared with a
    pantheongpu result for the same name. The unit strings are copied
    verbatim from the pantheongpu report schema -- a comparison joins on
    (Test Name, Unit), so a mismatch here silently breaks the join.

    ``problem`` pins shape and dtype. A Score is only comparable across
    platforms if both ran the same problem; without it the two numbers share
    a column while measuring different things.
    """

    name: str
    suite: str
    summary: str
    requires: typing.FrozenSet[str] = frozenset()
    min_devices: int = 1
    unit: typing.Optional[str] = None
    problem: typing.Optional[typing.Mapping[str, typing.Any]] = None
    score_source: typing.Optional[ScoreSource] = None

    def runnable_on(self, devices) -> bool:
        if len(devices) < self.min_devices:
            return False
        return all(self.requires <= device.capabilities() for device in devices)

    def skip_reason(self, devices) -> typing.Optional[str]:
        if len(devices) < self.min_devices:
            return f"needs {self.min_devices} devices, {len(devices)} selected"
        for device in devices:
            missing = self.requires - device.capabilities()
            if missing:
                return (
                    f"device {device.index} ({device.arch}) lacks: "
                    + ", ".join(sorted(missing))
                )
        return None


_COMPUTE = frozenset({"compute"})
_HBM = frozenset({"hbm"})
_COLLECTIVE = frozenset({"collectives"})
_TRAINING = frozenset({"training"})


WORKLOADS: typing.Tuple[Workload, ...] = (
    # -- baseline ---------------------------------------------------------
    Workload("baseline_metrics", "baseline",
             "Idle telemetry baseline; no load applied."),

    # -- core: power/thermal viruses on the tensor engine ------------------
    Workload("tensor_virus", "core",
             "Saturate the Tensor Engine with dense matmul.", _COMPUTE,
             unit="TFLOPS",
             problem={"op": "matmul", "shape": [8192, 8192, 8192], "dtype": "bf16"},
             score_source=ScoreSource(MONITOR,
                 counters=(
                     'neuroncore_counters.*.effective_flops',
                 ),
                 formula='mean(effective_flops) / 1e12')),
    Workload("int_virus", "core",
             "Sustained INT8 throughput on the Tensor Engine.", _COMPUTE,
             unit="TOPS",
             problem={"op": "matmul", "shape": [8192, 8192, 8192], "dtype": "int8"},
             score_source=ScoreSource(MONITOR,
                 counters=(
                     'neuroncore_counters.*.effective_flops',
                 ),
                 formula='mean(effective_flops) / 1e12   # int8 ops, reported as TOPS')),
    Workload("pulse_virus", "core",
             "Duty-cycled load to provoke power/clock transients.", _COMPUTE,
             unit="TFLOPS",
             problem={"op": "matmul", "shape": [8192, 8192, 8192], "dtype": "bf16",
                      "duty_cycle": 0.5, "period_s": 2},
             score_source=ScoreSource(MONITOR,
                 counters=(
                     'neuroncore_counters.*.effective_flops',
                     'throttle_active_nc0_time_ns',
                 ),
                 formula='mean(effective_flops) / 1e12; throttle_active_nc0_time_ns recorded alongside')),
    Workload("transformer_virus", "core",
             "Full transformer block under sustained load.", _COMPUTE,
             unit="TFLOPS",
             problem={"hidden": 4096, "heads": 32, "seq": 2048, "dtype": "bf16"},
             score_source=ScoreSource(MONITOR,
                 counters=(
                     'neuroncore_counters.*.effective_flops',
                 ),
                 formula='mean(effective_flops) / 1e12')),
    Workload("omni_virus", "core",
             "All engines concurrently: tensor, vector, scalar, GpSimd.", _COMPUTE,
             unit="TFLOPS",
             problem={"op": "mixed", "shape": [8192, 8192, 8192], "dtype": "bf16",
                      "engines": "all"},
             score_source=ScoreSource(MONITOR,
                 counters=(
                     'neuroncore_counters.*.effective_flops',
                     'tensor_engine_active_time_percent',
                     'vector_engine_active_time_percent',
                     'scalar_engine_active_time_percent',
                     'gpsimd_engine_active_time_percent',
                 ),
                 formula='mean(effective_flops) / 1e12; per-engine active_time_percent recorded alongside')),

    # -- memory: HBM bandwidth --------------------------------------------
    Workload("memory_read", "memory",
             "Streaming HBM reads on one NeuronCore.", _HBM,
             unit="GB/s",
             problem={"bytes": 8 << 30, "dtype": "bf16", "cores": 1},
             score_source=ScoreSource(PROFILER,
                 counters=(
                     'hbm_read_bytes',
                     'total_time',
                 ),
                 formula='hbm_read_bytes / total_time / 1e9')),
    Workload("memory_write", "memory",
             "Streaming HBM writes on one NeuronCore.", _HBM,
             unit="GB/s",
             problem={"bytes": 8 << 30, "dtype": "bf16", "cores": 1},
             score_source=ScoreSource(PROFILER,
                 counters=(
                     'hbm_write_bytes',
                     'total_time',
                 ),
                 formula='hbm_write_bytes / total_time / 1e9')),
    Workload("memory_read_agg", "memory",
             "Aggregate HBM read bandwidth, all NeuronCores.",
             _HBM | frozenset({"multicore"}),
             unit="GB/s",
             problem={"bytes": 8 << 30, "dtype": "bf16", "cores": "all"},
             score_source=ScoreSource(PROFILER,
                 counters=(
                     'hbm_read_bytes',
                     'total_time',
                 ),
                 formula='sum(hbm_read_bytes over cores) / total_time / 1e9')),
    Workload("memory_write_agg", "memory",
             "Aggregate HBM write bandwidth, all NeuronCores.",
             _HBM | frozenset({"multicore"}),
             unit="GB/s",
             problem={"bytes": 8 << 30, "dtype": "bf16", "cores": "all"},
             score_source=ScoreSource(PROFILER,
                 counters=(
                     'hbm_write_bytes',
                     'total_time',
                 ),
                 formula='sum(hbm_write_bytes over cores) / total_time / 1e9')),

    # -- interconnect ------------------------------------------------------
    Workload("all_reduce", "interconnect",
             "All-reduce collective over NeuronLink.", _COLLECTIVE, min_devices=2,
             unit="GB/s",
             problem={"op": "all_reduce", "bytes_min": 1 << 20, "bytes_max": 8 << 20,
                      "dtype": "fp32"},
             score_source=ScoreSource(NCCOM,
                 counters=(
                     'busbw',
                 ),
                 formula='nccom-test all_reduce busbw, averaged over the size sweep')),
    Workload("p2p_thrasher", "interconnect",
             "Sustained device-to-device traffic over NeuronLink.",
             _COLLECTIVE, min_devices=2,
             unit="GB/s",
             problem={"op": "sendrecv", "bytes": 1 << 26, "dtype": "fp32"},
             score_source=ScoreSource(NCCOM,
                 counters=(
                     'busbw',
                 ),
                 formula='nccom-test sendrecv busbw')),
    Workload("pcie_bandwidth", "interconnect",
             "Host-to-device and device-to-host transfer over PCIe.",
             unit="GB/s",
             problem={"bytes": 1 << 30, "direction": "bidirectional"},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'bytes_transferred',
                     'elapsed_s',
                 ),
                 formula='bytes_transferred / elapsed_s / 1e9   # device DMA counters are device-side only')),

    # -- inference ---------------------------------------------------------
    Workload("llm_decode", "inference",
             "Autoregressive decode; latency-bound token generation.", _COMPUTE,
             unit="tokens/s",
             problem={"hidden": 4096, "layers": 32, "batch": 1, "context": 2048,
                      "dtype": "bf16"},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'tokens_generated',
                     'elapsed_s',
                 ),
                 formula='tokens_generated / elapsed_s')),
    Workload("llm_prefill", "inference",
             "Prompt prefill; compute-bound batched attention.", _COMPUTE,
             unit="prompt-tokens/s",
             problem={"hidden": 4096, "layers": 32, "batch": 1, "prompt": 2048,
                      "dtype": "bf16"},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'prompt_tokens',
                     'elapsed_s',
                 ),
                 formula='prompt_tokens / elapsed_s')),
    Workload("kv_cache_churn", "inference",
             "KV cache allocation and eviction under pressure.", _COMPUTE | _HBM,
             unit="cache-updates/s",
             problem={"hidden": 4096, "heads": 32, "context": 4096, "dtype": "bf16"},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'cache_updates',
                     'elapsed_s',
                 ),
                 formula='cache_updates / elapsed_s')),
    Workload("fused_attention", "inference",
             "Fused attention kernel throughput.", _COMPUTE,
             unit="attention-tiles/s",
             problem={"heads": 32, "seq": 2048, "head_dim": 128, "dtype": "bf16"},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'attention_tiles',
                     'elapsed_s',
                 ),
                 formula='attention_tiles / elapsed_s')),
    Workload("quantized_gemm", "inference",
             "INT8/FP8 quantized GEMM paths.", _COMPUTE,
             unit="quantized-ops/s",
             problem={"op": "matmul", "shape": [4096, 4096, 4096], "dtype": "int8"},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'quantized_ops',
                     'elapsed_s',
                 ),
                 formula='quantized_ops / elapsed_s')),
    Workload("serving_mix", "inference",
             "Mixed prefill/decode traffic at serving ratios.", _COMPUTE,
             unit="requests/s",
             problem={"prefill_ratio": 0.2, "batch": 8, "prompt": 1024, "decode": 256},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'requests_completed',
                     'elapsed_s',
                 ),
                 formula='requests_completed / elapsed_s')),
    Workload("speculative_decode", "inference",
             "Draft-and-verify speculative decoding.", _COMPUTE,
             unit="verified-tokens/s",
             problem={"draft_len": 4, "hidden": 4096, "dtype": "bf16"},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'verified_tokens',
                     'elapsed_s',
                 ),
                 formula='verified_tokens / elapsed_s')),
    Workload("moe_router", "inference",
             "Mixture-of-experts routing and expert dispatch.", _COMPUTE,
             unit="routed-tokens/s",
             problem={"experts": 8, "top_k": 2, "hidden": 4096, "tokens": 4096},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'routed_tokens',
                     'elapsed_s',
                 ),
                 formula='routed_tokens / elapsed_s')),

    # -- training (Trainium only) -----------------------------------------
    Workload("transformer_train_step", "training",
             "Forward, backward and optimiser step.", _TRAINING,
             unit="train-steps/s",
             problem={"hidden": 4096, "layers": 8, "batch": 4, "seq": 2048,
                      "dtype": "bf16"},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'steps_completed',
                     'elapsed_s',
                 ),
                 formula='steps_completed / elapsed_s')),

    # -- runtime -----------------------------------------------------------
    Workload("allocation_fragmentation", "runtime",
             "Device memory allocator under fragmentation pressure.", _HBM,
             unit="allocation-events/s",
             problem={"allocations": 10000, "size_min": 1 << 12, "size_max": 1 << 24},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'allocation_events',
                     'elapsed_s',
                 ),
                 formula='allocation_events / elapsed_s')),
    Workload("graph_replay", "runtime",
             "Repeated replay of a compiled NEFF graph.", _COMPUTE,
             unit="graph-steps/s",
             problem={"hidden": 2048, "replays": 10000, "dtype": "bf16"},
             score_source=ScoreSource(MONITOR,
                 counters=(
                     'execution_stats.execution_summary.completed',
                     'execution_stats.period',
                 ),
                 formula='delta(completed) / period')),

    # -- ai_auxiliary ------------------------------------------------------
    Workload("rag_embedding", "ai_auxiliary",
             "Embedding generation at retrieval batch sizes.", _COMPUTE,
             unit="embedding-vectors/s",
             problem={"dim": 1024, "batch": 256, "dtype": "bf16"},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'vectors_embedded',
                     'elapsed_s',
                 ),
                 formula='vectors_embedded / elapsed_s')),
    Workload("vision_encoder", "ai_auxiliary",
             "Vision encoder forward pass.", _COMPUTE,
             unit="image-tiles/s",
             problem={"resolution": 224, "patch": 14, "batch": 64, "dtype": "bf16"},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'image_tiles',
                     'elapsed_s',
                 ),
                 formula='image_tiles / elapsed_s')),
)


# GPU workloads with no Neuron counterpart, and why.  Kept as data so the
# comparison tooling can render an honest "N/A" instead of a missing row.
NO_NEURON_EQUIVALENT = {
    "fp64_virus": "Neuron devices have no FP64 units.",
    "mma_virus": "MMA is an NVIDIA tensor-core instruction; Neuron's Tensor Engine has a different ISA.",
    "sfu_stress": "SFU is an NVIDIA SM structure; Neuron's Scalar/GpSimd engines are not equivalent.",
    "rt_virus": "No ray-tracing hardware.",
    "media_enc_virus": "No media encode block.",
    "atomic_virus": "NKI does not expose GPU-style device-wide atomics.",
    "scheduler": "No warp scheduler; Neuron dispatch is compiler-scheduled.",
    "voltage": "No equivalent voltage-rail control exposed by the Neuron driver.",
    "incinerator": "GPU-specific thermal virus with no documented Neuron analogue.",
    "cache_lat": "Neuron's on-chip memory hierarchy is not a GPU-style cache.",
    "tlb_avalanche": "No exposed TLB behaviour to target.",
    "memory_bank_thrash": "HBM bank mapping is not exposed on Neuron.",
    "memory_cache_fracture": "No GPU-style L2 partitioning to fracture.",
    "memory_pc_pingpong": "No partition-camping analogue.",
    "memory_tsv_thrasher": "TSV-level access patterns are not addressable.",
    "memory_retention_bake": "Requires refresh-interval control Neuron does not expose.",
    "memory_thermal_asym": "Depends on GPU-specific per-stack thermal telemetry.",
}


SUITES = (
    "baseline",
    "core",
    "memory",
    "interconnect",
    "inference",
    "training",
    "runtime",
    "ai_auxiliary",
)

_BY_NAME = {workload.name: workload for workload in WORKLOADS}


def resolve(target: str) -> typing.List[Workload]:
    """Resolve a --test value: a workload name, a suite name, or 'all'."""
    key = target.strip().lower()
    if key == "all":
        return list(WORKLOADS)
    if key in _BY_NAME:
        return [_BY_NAME[key]]
    matched = [w for w in WORKLOADS if w.suite == key]
    if matched:
        return matched
    if key in NO_NEURON_EQUIVALENT:
        raise KeyError(
            f"'{target}' is a pantheongpu workload with no Neuron equivalent: "
            f"{NO_NEURON_EQUIVALENT[key]}"
        )
    known = ", ".join(sorted(_BY_NAME) + list(SUITES))
    raise KeyError(f"Unknown test '{target}'. Known targets: {known}")
