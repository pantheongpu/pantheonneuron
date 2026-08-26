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


@dataclasses.dataclass(frozen=True)
class Workload:
    name: str
    suite: str
    summary: str
    requires: typing.FrozenSet[str] = frozenset()
    min_devices: int = 1

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
    Workload("baseline_metrics", "baseline", "Idle telemetry baseline; no load applied."),

    # -- core: power/thermal viruses on the tensor engine ------------------
    Workload("tensor_virus", "core", "Saturate the Tensor Engine with dense matmul.", _COMPUTE),
    Workload("int_virus", "core", "Sustained INT8 throughput on the Tensor Engine.", _COMPUTE),
    Workload("pulse_virus", "core", "Duty-cycled load to provoke power/clock transients.", _COMPUTE),
    Workload("transformer_virus", "core", "Full transformer block under sustained load.", _COMPUTE),
    Workload("omni_virus", "core", "All engines concurrently: tensor, vector, scalar, GpSimd.", _COMPUTE),

    # -- memory: HBM bandwidth --------------------------------------------
    Workload("memory_read", "memory", "Streaming HBM reads on one NeuronCore.", _HBM),
    Workload("memory_write", "memory", "Streaming HBM writes on one NeuronCore.", _HBM),
    Workload("memory_read_agg", "memory", "Aggregate HBM read bandwidth, all NeuronCores.", _HBM | frozenset({"multicore"})),
    Workload("memory_write_agg", "memory", "Aggregate HBM write bandwidth, all NeuronCores.", _HBM | frozenset({"multicore"})),

    # -- interconnect ------------------------------------------------------
    Workload("all_reduce", "interconnect", "All-reduce collective over NeuronLink.", _COLLECTIVE, min_devices=2),
    Workload("p2p_thrasher", "interconnect", "Sustained device-to-device traffic over NeuronLink.", _COLLECTIVE, min_devices=2),
    Workload("pcie_bandwidth", "interconnect", "Host-to-device and device-to-host transfer over PCIe."),

    # -- inference ---------------------------------------------------------
    Workload("llm_decode", "inference", "Autoregressive decode; latency-bound token generation.", _COMPUTE),
    Workload("llm_prefill", "inference", "Prompt prefill; compute-bound batched attention.", _COMPUTE),
    Workload("kv_cache_churn", "inference", "KV cache allocation and eviction under pressure.", _COMPUTE | _HBM),
    Workload("fused_attention", "inference", "Fused attention kernel throughput.", _COMPUTE),
    Workload("quantized_gemm", "inference", "INT8/FP8 quantized GEMM paths.", _COMPUTE),
    Workload("serving_mix", "inference", "Mixed prefill/decode traffic at serving ratios.", _COMPUTE),
    Workload("speculative_decode", "inference", "Draft-and-verify speculative decoding.", _COMPUTE),
    Workload("moe_router", "inference", "Mixture-of-experts routing and expert dispatch.", _COMPUTE),

    # -- training (Trainium only) -----------------------------------------
    Workload("transformer_train_step", "training", "Forward, backward and optimiser step.", _TRAINING),

    # -- runtime -----------------------------------------------------------
    Workload("allocation_fragmentation", "runtime", "Device memory allocator under fragmentation pressure.", _HBM),
    Workload("graph_replay", "runtime", "Repeated replay of a compiled NEFF graph.", _COMPUTE),

    # -- ai_auxiliary ------------------------------------------------------
    Workload("rag_embedding", "ai_auxiliary", "Embedding generation at retrieval batch sizes.", _COMPUTE),
    Workload("vision_encoder", "ai_auxiliary", "Vision encoder forward pass.", _COMPUTE),
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
