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
    pantheongpu result for the same name, where such a comparison is
    meaningful at all. A comparison joins on (Test Name, Unit), so a mismatch
    silently breaks the join rather than raising.

    Matching the GPU unit is not always the right thing. pantheongpu v1.0.19
    replaced the units of its AI workloads with a single ``ai-ops/s``, because
    ten of them shared one kernel body and six compiled to byte-identical
    SASS: the numbers were generic synthetic throughput wearing twelve
    different metric names. The Neuron workloads of those names count real
    tokens, cache updates and training steps. Copying ``ai-ops/s`` here to
    restore the join would make these numbers less honest, not more, so those
    workloads keep their own units and are listed in
    ``NOT_COMPARABLE_WITH_GPU`` instead.

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
                 formula='mean(effective_flops over whole busy periods) / 1e12')),
    # dtype is uint8, not int8, and the difference is measured rather than
    # stylistic. trn1's Tensor Engine rejects signed int8 outright --
    # `nc_matmul does not support stationary.dtype=int8`, 2026-09-08 -- and
    # the supported operand set is fp8_e4m3, fp8_e5m2, bf16, fp16, tf32,
    # fp32 and uint8. Pinning int8 made this workload unreachable on the
    # only Trainium part we can run.
    #
    # uint8 rather than fp8, of the reachable options: the unit is TOPS,
    # which means integer operations, and fp8 would keep the label while
    # changing the quantity underneath it to floating-point. uint8 is the
    # same 8-bit integer datapath the name claims, and an all-ones GEMM
    # reaches exactly K either way, so the correctness check is unchanged.
    #
    # What this costs: a signed-int8 path exists on other accelerators and
    # is not measured here. A cross-platform reader must not read this row
    # as a signed-int8 figure, which is why the dtype travels with the
    # Score in ``problem`` rather than living only in this comment.
    Workload("int_virus", "core",
             "Sustained UINT8 throughput on the Tensor Engine.", _COMPUTE,
             unit="TOPS",
             problem={"op": "matmul", "shape": [8192, 8192, 8192],
                      "dtype": "uint8"},
             score_source=ScoreSource(MONITOR,
                 counters=(
                     'neuroncore_counters.*.effective_flops',
                 ),
                 formula='mean(effective_flops over whole busy periods) / 1e12   # uint8 ops, reported as TOPS')),
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
                 formula='mean(effective_flops over whole busy periods) / 1e12; throttle_active_nc0_time_ns recorded alongside')),
    Workload("transformer_virus", "core",
             "Full transformer block under sustained load.", _COMPUTE,
             unit="TFLOPS",
             problem={"hidden": 4096, "heads": 32, "seq": 2048, "dtype": "bf16"},
             score_source=ScoreSource(MONITOR,
                 counters=(
                     'neuroncore_counters.*.effective_flops',
                 ),
                 formula='mean(effective_flops over whole busy periods) / 1e12')),
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
                 formula='mean(effective_flops over whole busy periods) / 1e12; per-engine active_time_percent recorded alongside')),

    # -- memory: HBM bandwidth --------------------------------------------
    Workload("memory_read", "memory",
             "Streaming HBM reads on one NeuronCore.", _HBM,
             unit="GB/s",
             problem={"bytes": 8 << 30, "dtype": "bf16", "cores": 1},
             score_source=ScoreSource(PROFILER,
                 counters=(
                     'hbm_read_bytes',
                     'total_time',
                     'total_active_time',
                 ),
                 formula='hbm_read_bytes / total_active_time / 1e9')),
    # 4 GiB, not the 8 GiB memory_read uses, and the asymmetry is measured.
    # A write's destination is the whole plan and the runtime still holds
    # the previous one while the next is allocated, so the pin costs twice
    # its size in residency. Both parts this suite targets have 32 GB
    # across two cores. Measured on inf2.xlarge 2026-09-07: 4 GiB runs at
    # 255.1 GB/s with the destination check at exactly 1.0, 6 GiB at 162.5
    # GB/s, and 8 GiB fails outright with 8.59 GB requested against 8.099
    # GB resident. trn1.2xlarge 2026-09-08 failed at 8 GiB identically.
    #
    # 4 GiB rather than 6: both fit, but the 36% drop at 6 GiB is the part
    # running out of room, not the memory system going slower. A number
    # measured under allocation pressure is not the write bandwidth this
    # workload claims to report.
    #
    # memory_read keeps 8 GiB because a read allocates only a source and
    # was verified there (236.9 GB/s on inf2). The two are joined against
    # their own name on another platform, not against each other, so they
    # do not need the same size -- but ``problem`` carries the size into
    # the report precisely so nobody compares them as though they did.
    Workload("memory_write", "memory",
             "Streaming HBM writes on one NeuronCore.", _HBM,
             unit="GB/s",
             problem={"bytes": 4 << 30, "dtype": "bf16", "cores": 1},
             score_source=ScoreSource(PROFILER,
                 counters=(
                     'hbm_write_bytes',
                     'total_time',
                     'total_active_time',
                 ),
                 formula='hbm_write_bytes / total_active_time / 1e9')),
    Workload("memory_read_agg", "memory",
             "Aggregate HBM read bandwidth, all NeuronCores.",
             _HBM | frozenset({"multicore"}),
             unit="GB/s",
             problem={"bytes": 8 << 30, "dtype": "bf16", "cores": "all"},
             score_source=ScoreSource(PROFILER,
                 counters=(
                     'hbm_read_bytes',
                     'total_time',
                     'total_active_time',
                 ),
                 formula='sum(hbm_read_bytes over cores) / total_active_time / 1e9')),
    # 4 GiB per core, for the same residency reason as memory_write: each
    # worker allocates its own destination on its own core, so the pin is
    # per-core and the arithmetic is identical.
    Workload("memory_write_agg", "memory",
             "Aggregate HBM write bandwidth, all NeuronCores.",
             _HBM | frozenset({"multicore"}),
             unit="GB/s",
             problem={"bytes": 4 << 30, "dtype": "bf16", "cores": "all"},
             score_source=ScoreSource(PROFILER,
                 counters=(
                     'hbm_write_bytes',
                     'total_time',
                     'total_active_time',
                 ),
                 formula='sum(hbm_write_bytes over cores) / total_active_time / 1e9')),

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
    # Sized so that a whole-cache copy per append is tractable, because on
    # this stack that is what an append costs. XLA is functional, so
    # cache[:, a:b, :] = entry produces a new tensor rather than writing in
    # place: appending to a 2 GiB cache reads 2 GiB and writes 2 GiB, and
    # its graph takes about seven minutes to compile. See
    # llm_inference.cache_plan for the three measurements that established
    # it, each of which first looked like a different problem.
    #
    # 8 layers x 2048 context x 2048 hidden is a 128 MiB cache, so a step
    # moves 256 MiB -- about 1 ms of bandwidth -- and its graphs compile in
    # something like a minute rather than an hour. Smaller than a
    # production cache on purpose: the alternative is a workload that
    # cannot finish, and a number that does not exist is worse than a
    # number from a small cache that says so.
    Workload("kv_cache_churn", "inference",
             "KV cache allocation and eviction under pressure.", _COMPUTE | _HBM,
             unit="cache-updates/s",
             problem={"hidden": 2048, "heads": 16, "context": 2048,
                      "layers": 8, "ring_slots": 8, "dtype": "bf16"},
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
    # "INT8/FP8" was aspirational on both halves. neuronx-cc refuses
    # fp8_e4m3 outright (`NCC_ESPP047`, trn1.2xlarge 2026-09-10), so there
    # is no FP8 path to measure; and int8 runs at 0.26x bf16 on the same
    # shape, so the INT8 path is the slowest arithmetic on the part rather
    # than an acceleration. The description now says what the row is.
    Workload("quantized_gemm", "inference",
             "INT8 GEMM with dequantisation. Slower than bf16 on this "
             "part -- a footprint figure, not a throughput win.", _COMPUTE,
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
             # layers is pinned because a verification pass runs the target
             # model, and the kernel used to run one of its blocks -- making
             # the expensive half of speculative decoding a thirty-second of
             # its real cost, which is the half the technique exists to
             # amortise. Same defect serving_mix had.
             problem={"draft_len": 4, "hidden": 4096, "layers": 32,
                      "dtype": "bf16"},
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
    # batch 1 and 4 layers, not batch 4 and 8. The original pin needed
    # 38.18 GB of peak HBM against the 16 GB a NeuronCore has -- 6.12 GB of
    # I/O tensors and 30.94 GB of intermediates -- and the compiler refused
    # it outright on trn1.2xlarge 2026-09-08 with NCC_EOOM001. Backward is
    # what makes training different here: it keeps every forward activation
    # alive until its gradient is consumed, so the intermediates scale with
    # batch x layers in a way no forward-only workload pays.
    #
    # This is the fourth pinned problem that turned out to be unreachable
    # on the hardware it targets, after int_virus's int8, memory_write's
    # 8 GiB and the 8192^3 unroll. A pin is a claim about what the part can
    # do, and it needs measuring like any other.
    #
    # batch 1 / layers 4 lands near 6.5 GB, which leaves room for the
    # optimiser state rather than only just fitting. hidden and seq are
    # unchanged, so a step still exercises the shapes a real model uses.
    Workload("transformer_train_step", "training",
             "Forward, backward and optimiser step.", _TRAINING,
             unit="train-steps/s",
             problem={"hidden": 4096, "layers": 4, "batch": 1, "seq": 2048,
                      "dtype": "bf16"},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'steps_completed',
                     'elapsed_s',
                 ),
                 formula='steps_completed / elapsed_s')),

    # -- runtime -----------------------------------------------------------
    #
    # 40,000 allocations, not the 10,000 this was pinned at until
    # 2026-09-10. The count bounds the run, not the clock, so the pin *is*
    # the measurement window and 10,000 of them finished in 3.98s. A sweep
    # on trn1.2xlarge, three repeats each:
    #
    #      10,000    3.98s   2512.9 events/s   cv 0.083
    #      40,000   16.66s   2400.7 events/s   cv 0.038
    #     120,000   54.37s   2295.3 events/s   cv 0.025
    #
    # 40,000 halves the scatter for a window that is a real measurement
    # rather than a moment, without a single workload taking a minute of a
    # 23-workload pass. 120,000 is better still and is the pin to reach for
    # when the number itself matters more than the pass duration.
    #
    # **The Score is not comparable across pins.** The rate falls as the
    # count rises, because a longer run works a more fragmented allocator
    # -- which is the thing being measured, not drift. Any figure quoted
    # against a different allocation count is a different quantity, and the
    # count travels with the Score in ``problem`` so that is visible.
    Workload("allocation_fragmentation", "runtime",
             "Device memory allocator under fragmentation pressure.", _HBM,
             unit="allocation-events/s",
             problem={"allocations": 40000, "size_min": 1 << 12, "size_max": 1 << 24},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'allocation_events',
                     'elapsed_s',
                 ),
                 formula='allocation_events / elapsed_s')),
    Workload("graph_replay", "runtime",
             "Repeated replay of a compiled NEFF graph.", _COMPUTE,
             unit="graph-steps/s",
             # 60,000, not the 10,000 this was pinned at until 2026-09-10.
             # The replay count bounds the run before --duration does, so
             # the pin is the measurement window -- and 10,000 replays at
             # the 3040 graph-steps/s this part reaches is 3.3 seconds,
             # measured, whatever duration is asked for.
             #
             # 3.3 seconds is below what neuron-monitor needs to form a
             # delta from its sampled completion counter, which is why the
             # declared Score fired on 2026-09-08 (13.7s window at the
             # 729 steps/s that run reached) and degraded to the analytic
             # fallback on 2026-09-10. The Score source was never
             # intermittent; the window was.
             #
             # **The window is inversely proportional to the rate**, which
             # is the uncomfortable part: a faster device measures itself
             # over a shorter window and is more likely to lose its
             # declared Score. 60,000 gives about 20 seconds at the
             # observed rate and would still give 13 at half of it.
             #
             # 200,000 since 2026-09-10, so that --duration bounds the
             # window rather than the count. The completion counter is a
             # tally per ~5 s sampling period, and the rate is taken over
             # whole periods only; 60,000 replays is ~20 s, four periods,
             # and on trn1.2xlarge it left one whole period to divide.
             # 200,000 is ~65 s at the measured rate, past any default
             # duration, so the window is whatever the caller asked for.
             problem={"hidden": 2048, "replays": 200000, "dtype": "bf16"},
             score_source=ScoreSource(MONITOR,
                 counters=(
                     'execution_stats.execution_summary.completed',
                     'execution_stats.period',
                 ),
                 # completed is a tally per sampling period, not a running
                 # total (trn1.2xlarge 2026-09-10: it falls to zero when the
                 # work stops, and the tallies sum to the replays run). This
                 # read delta(completed) / period until then.
                 formula='sum(completed) / sum(period) over whole busy periods')),

    # -- ai_auxiliary ------------------------------------------------------
    Workload("rag_embedding", "ai_auxiliary",
             "Embedding generation at retrieval batch sizes.", _COMPUTE,
             unit="embedding-vectors/s",
             # A retrieval embedder is a transformer stack. This pinned a
             # projection: two matmuls and an L2 normalise, reporting
             # 1,551,194 vectors/s on trn1.2xlarge 2026-09-08, roughly
             # 2,000x what a 12-layer encoder over 128-token documents
             # reaches on this part. batch drops to 64 so a real encoder
             # pass stays tractable.
             problem={"dim": 1024, "batch": 64, "seq": 128, "layers": 12,
                      "dtype": "bf16"},
             score_source=ScoreSource(INTERNAL,
                 counters=(
                     'vectors_embedded',
                     'elapsed_s',
                 ),
                 formula='vectors_embedded / elapsed_s')),
    Workload("vision_encoder", "ai_auxiliary",
             "Vision encoder forward pass.", _COMPUTE,
             unit="image-tiles/s",
             # A ViT-B is twelve blocks; this ran one, and reported about
             # eighteen times the throughput the model it names can reach.
             problem={"resolution": 224, "patch": 14, "batch": 64,
                      "layers": 12, "dtype": "bf16"},
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


# Workloads that exist on both platforms under the same name, but whose
# Scores must not be joined. pantheongpu v1.0.19 collapsed these to a single
# ``ai-ops/s``: ten of its AI workloads shared one kernel body and six
# compiled to byte-identical SASS, so what it reports is generic synthetic
# throughput, not the quantity the name suggests. The Neuron implementations
# count the real thing -- tokens generated, cache updates applied, training
# steps completed -- so the two numbers share a name while measuring
# different quantities.
#
# Kept as data for the same reason as NO_NEURON_EQUIVALENT: the comparison
# tooling can render an honest "not comparable" rather than a row that
# quietly never joins, or worse, one that joins and misleads.
NOT_COMPARABLE_WITH_GPU = {
    "fused_attention": "attention-tiles/s",
    "graph_replay": "graph-steps/s",
    "kv_cache_churn": "cache-updates/s",
    "llm_decode": "tokens/s",
    "llm_prefill": "prompt-tokens/s",
    "moe_router": "routed-tokens/s",
    "quantized_gemm": "quantized-ops/s",
    "rag_embedding": "embedding-vectors/s",
    "serving_mix": "requests/s",
    "speculative_decode": "verified-tokens/s",
    "transformer_train_step": "train-steps/s",
    "vision_encoder": "image-tiles/s",
}

# Workloads whose Score is a function of the pinned problem in a way that
# makes two different pins two different quantities, rather than the same
# quantity measured at two sizes.
#
# Every Score here depends on its problem -- that is what pinning is for --
# but most vary the way a rate does: a bigger matmul at the same TFLOPS is
# the same number. These do not. allocation_fragmentation's rate *falls* as
# the allocation count rises, because the count bounds the run and a longer
# run works a more fragmented allocator. 2512.9 events/s at 10,000
# allocations and 2295.3 at 120,000 are not the same measurement disagreeing;
# they are two measurements of different things.
#
# Declared rather than left in a comment because two textual checks in this
# repo have passed on their own comments, and a third ended in `or True`.
SCORE_DEPENDS_ON_PIN = {
    "allocation_fragmentation": (
        "the pinned allocations count bounds the run, and the event rate "
        "falls as it rises -- a figure at a different count is a different "
        "quantity, not a disagreeing one"
    ),
}

# Counters a workload declares that its declared Score source cannot
# supply. Every one of these is real and readable -- through
# ``neuron-profile``, which is a per-execution capture interface rather
# than continuous telemetry, and which no workload here declares alongside
# a monitor source.
#
# Found 2026-09-10 by checking each declared counter against the code of
# the reader named beside it. Nothing had ever asked, so five counters sat
# in the registry attributed to a stream that does not carry them.
#
# This is not cosmetic. The counters tuple is the published answer to
# "where does this number come from", and for omni_virus it is the answer
# to a sharper question: the workload exists to load all four engines at
# once, and the per-engine counters that would show whether it did are on
# the other reader. **The workload whose premise is per-engine behaviour
# is scored by the reader that cannot see any engine.**
#
# Reaching them needs a profiler capture of the workload's own NEFF, which
# needs a reserved core, which is off for any selection containing a
# cores: "all" workload. Real work, not an oversight, and not done -- so
# it is written down here rather than left as a false attribution.
#
# docs/neuron_counters.md records the probe that established which reader
# has what; data/probe-2026-08-26-tools/ carries the raw 108-counter set.
COUNTERS_THE_DECLARED_SOURCE_CANNOT_SUPPLY = {
    "omni_virus": (
        "tensor_engine_active_time_percent",
        "vector_engine_active_time_percent",
        "scalar_engine_active_time_percent",
        "gpsimd_engine_active_time_percent",
    ),
    # docs/neuron_counters.md lists this one under "Recovered by the
    # profiler", explicitly noting it is not available through
    # neuron-monitor or sysfs -- and the registry declared it as a
    # neuron-monitor counter anyway, contradicting the repo's own
    # documentation with nothing to notice.
    "pulse_virus": ("throttle_active_nc0_time_ns",),
}

# Published peak figures, per accelerator chip. **VERIFIED 2026-09-10**
# against the AWS Neuron architecture documentation.
#
#   Trainium1:   https://awsdocs-neuron.readthedocs-hosted.com/en/latest/
#                general/arch/neuron-hardware/trainium.html
#   Inferentia2: .../neuron-hardware/inferentia2.html
#
# Both quote, per chip, in identical words: two NeuronCore-v2, "32GiB of
# high-bandwidth device memory (HBM)" with "820 GiB/sec of bandwidth",
# and "190 FP16/BF16/cFP8/TF32 TFLOPS" (Trainium1 adds 47.5 FP32).
#
# **The two parts are the same silicon per chip.** They differ in how
# many chips an instance carries, not in what a chip does -- which the
# repo already knew from the other direction: both report NeuronCore-v2.
#
# UNITS. The doc says 820 **GiB**/sec and every Score here is decimal
# GB/s (bytes / 1e9), so the ceiling is 820 * 2^30 / 1e9 = 880.5 GB/s.
# Storing the GiB figure and converting at the point of use would invite
# the 7% error every time; the conversion is done once, here, and the
# original is kept beside it so the citation can be checked without
# undoing arithmetic.
#
# WHAT DID NOT RECONCILE, recorded because it is the reason to trust the
# architecture page over the marketing one. The instance pages say "9.8
# TB/s of total memory bandwidth" for *both* trn1.32xlarge (16 chips) and
# inf2.48xlarge (12). At 820 GiB/s per chip those are 14.09 and 10.57
# TB/s -- inf2 is close, trn1 is not, and 9.8 looks like inf2's number
# printed on both pages. Compute reconciles cleanly on both: 16 x 190 =
# 3.04 PFLOPS against "up to 3", and 12 x 190 = 2.28 against "up to 2.3".
#
# A PREVIOUS VERSION OF THIS TABLE WAS WRONG, and in the direction that
# flatters. It carried 613 GB/s for trn1, derived by dividing the
# instance page's 9.8 TB/s by 16, on a suspicion that the "~820" long
# quoted in kernels/memory_read.py was really Inferentia2's figure. The
# suspicion was backwards: 820 was right, for both parts, and dividing by
# the smaller ceiling reported the bandwidth kernels at 83-88% of peak
# when they reach about 60%.
PART_PEAKS = {
    "trn1": {
        "device_name": "Trainium1",
        "neuroncores": 2,
        "hbm_gibps": 820.0,
        "hbm_gbps": 880.5,
        "bf16_tflops": 190.0,
        "fp32_tflops": 47.5,
        "source": (
            "AWS Neuron architecture docs, neuron-hardware/trainium.html, "
            "read 2026-09-10: two NeuronCore-v2, 32GiB HBM at 820 GiB/sec, "
            "190 FP16/BF16/cFP8/TF32 TFLOPS, 47.5 FP32 TFLOPS"
        ),
        "verified": True,
    },
    "inf2": {
        "device_name": "Inferentia2",
        "neuroncores": 2,
        "hbm_gibps": 820.0,
        "hbm_gbps": 880.5,
        "bf16_tflops": 190.0,
        "fp32_tflops": None,
        "source": (
            "AWS Neuron architecture docs, neuron-hardware/inferentia2.html, "
            "read 2026-09-10: two NeuronCore-v2, 32GiB HBM at 820 GiB/sec, "
            "190 FP16/BF16/cFP8/TF32 TFLOPS"
        ),
        "verified": True,
    },
}

# Which peak a workload's unit should be measured against. A unit alone
# cannot say: GB/s is a memory figure and TFLOPS an arithmetic one, but
# TOPS is arithmetic too and graph-steps/s is neither.
PEAK_FOR_UNIT = {
    "GB/s": "hbm_gbps",
    "TFLOPS": "bf16_tflops",
}

# GB/s workloads whose bytes do not cross the HBM path, so the HBM ceiling
# is not theirs. The unit matched and the column divided anyway: the
# 2026-09-10 --test all on trn1.2xlarge printed pcie_bandwidth at "0.44%
# of 880.5 GB/s" -- a host-to-device PCIe rate as a share of device
# memory bandwidth, which is not a fraction of anything. No PCIe or
# NeuronLink ceiling has been verified for these parts, so they get none
# rather than a borrowed one.
NO_PUBLISHED_PEAK = {
    "pcie_bandwidth": "host-to-device PCIe transfers; the HBM ceiling is not "
                      "on this path and no PCIe ceiling has been verified",
    "all_reduce": "a collective's bus bandwidth between cores; no NeuronLink "
                  "or on-chip interconnect ceiling has been verified",
    "p2p_thrasher": "device-to-device transfers; no NeuronLink ceiling has "
                    "been verified",
}

# What pantheongpu reports for those names since v1.0.19.
GPU_SYNTHETIC_AI_UNIT = "ai-ops/s"


# NOT_COMPARABLE_WITH_GPU covers one failure mode: the units diverge, so the
# join fails and the absence is visible. There is a second, worse one that
# has no register here -- names where the unit matches, the join succeeds,
# and the two numbers measure different quantities.
#
# It applies to the compute viruses. pantheongpu's tensor_virus is __hfma2
# chains on the FP16 vector lanes with no matrix at all, counted analytically
# from occupancy; Neuron's is a dense systolic GEMM read from a hardware
# counter. Same name, same TFLOPS, different functional unit and different
# provenance -- and the bias has a direction, because it puts the GPU's
# secondary math path against Neuron's primary one.
#
# Encoded below as its own register rather than forced into
# NOT_COMPARABLE_WITH_GPU, whose tests define it as "the units diverge".
# These units do not diverge -- that is the whole problem. The evidence is
# in docs/cross_platform_comparability.md.
SAME_UNIT_DIFFERENT_QUANTITY = {
    "tensor_virus": (
        "pantheongpu runs __hfma2 chains on the FP16 vector lanes with no "
        "matrix at all, counted analytically from occupancy; this is a dense "
        "systolic GEMM read from a hardware counter."
    ),
    "int_virus": (
        "pantheongpu runs integer FMA chains counted from occupancy; this is "
        "a dense uint8 GEMM on the Tensor Engine."
    ),
    "pulse_virus": (
        "pantheongpu duty-cycles scalar fp32 fmaf chains; this duty-cycles a "
        "dense bf16 GEMM."
    ),
    "omni_virus": (
        "pantheongpu sums analytic per-engine op counts; this drives four "
        "engines in one dependent chain and reads effective_flops."
    ),
}

# transformer_virus is deliberately absent. pantheongpu does use real matrix
# instructions there (MFMA/WMMA), so the functional-unit objection does not
# apply -- though the path sits behind an experimental flag with a
# non-matrix fallback under the same name, and the issued-versus-retired
# difference still stands. Listing it would overstate what is known; the
# doc records the caveat.

# Nothing consumes this to change a join. It exists so the comparison
# tooling can render a warning where a row would otherwise join silently,
# and so the finding cannot be lost. Fixing it properly means deciding what
# the two suites claim about each other, which is not a decision a commit
# should make on its own.


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
