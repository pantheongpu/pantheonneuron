"""Prefill, decode and the KV cache: three workloads, three shapes of work.

Score: **workload-counted**, as the registry declares. No hardware counter
counts tokens, so the kernel is the only possible source.

These live in one module because they share a model, and are three separate
workloads because they stress three different things:

``llm_prefill``  runs a 2048-token prompt through every layer at once. Every
                 matmul is large and the attention term is quadratic in
                 sequence length, so it is compute-bound and it is what
                 saturates the Tensor Engine.
``llm_decode``   runs a single token per step against a cached context. The
                 same projections shrink to a batch of one, attention becomes
                 linear in context, and the step is dominated by reading the
                 cache back -- latency-bound, and roughly three orders of
                 magnitude less arithmetic per step than prefill.
``kv_cache_churn`` never runs the model at all. It writes and evicts cache
                 entries against a full context, so it measures the memory
                 traffic a long-running server pays and nothing else.

Reporting one number for these three would be the mistake pantheongpu made.

STATUS: UNTESTED ON HARDWARE.
"""

import time
import typing

from . import nki_backend, tiling, transformer_ops


def run_prefill(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Push whole prompts through the stack and count prompt tokens."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    hidden = int(problem["hidden"])
    layers = int(problem["layers"])
    batch = int(problem["batch"])
    prompt = int(problem["prompt"])
    dtype = tiling.torch_dtype(str(problem["dtype"]))

    device = xm.xla_device()
    # One block's weights, reused for every layer. The arithmetic and the
    # memory traffic are what is being measured, and both are identical
    # whether the layers hold distinct values or not -- while distinct
    # weights for 32 layers of hidden 4096 would be 12 GB of parameters
    # that the part cannot spare alongside the activations.
    params = transformer_ops.weights(hidden, dtype, device, heads=32)
    hidden_states = torch.ones((batch, prompt, hidden), dtype=dtype, device=device)
    xm.mark_step()

    warm = hidden_states
    for _ in range(layers):
        warm = transformer_ops.block(warm, params)
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    sink = None
    prompts = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        state = hidden_states
        for _ in range(layers):
            state = transformer_ops.block(state, params)
        sink = state
        xm.mark_step()
        prompts += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    observed = transformer_ops.read_back(sink)
    tokens = prompts * prompt * batch
    flops = prompts * layers * transformer_ops.block_flops(hidden, prompt, batch)

    return {
        "prompts": prompts,
        "prompt_tokens": tokens,
        "elapsed_s": elapsed,
        "prompt_tokens_per_s": tokens / elapsed if elapsed else 0.0,
        "flops_issued": flops,
        "implied_tflops": flops / elapsed / 1e12 if elapsed else 0.0,
        "score_method": "workload",
        "analytic_basis": "prompt tokens / wall time",
        **transformer_ops.output_check(
            observed, "prefill output"),
    }


def run_decode(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Generate tokens one at a time against a cache, and count them."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    hidden = int(problem["hidden"])
    layers = int(problem["layers"])
    batch = int(problem["batch"])
    context = int(problem["context"])
    dtype = tiling.torch_dtype(str(problem["dtype"]))

    device = xm.xla_device()
    params = transformer_ops.weights(hidden, dtype, device, heads=32)

    # The cache is the point of this workload: it is resident, it is read
    # every step, and it is what makes decode memory-bound. One K and one V
    # per layer, sized to the full context.
    cache_k = torch.ones((layers, batch, context, hidden), dtype=dtype, device=device)
    cache_v = torch.ones((layers, batch, context, hidden), dtype=dtype, device=device)
    token = torch.ones((batch, 1, hidden), dtype=dtype, device=device)
    xm.mark_step()
    xm.wait_device_ops()

    def step(state):
        for layer in range(layers):
            # Normalised like transformer_ops.block, and for the same
            # reason. Decode survived unnormalised only because its
            # attention is against a constant cache rather than against the
            # growing state, so its scores stayed linear in magnitude where
            # prefill's were quadratic. That is luck, not a property worth
            # relying on.
            normed = transformer_ops.rms_norm(state)
            q = torch.matmul(normed, params["q"])
            # Attention against the cached context rather than against the
            # single token: linear in context, which is the shape decode
            # actually has and prefill does not.
            scores = torch.matmul(q, cache_k[layer].transpose(-1, -2))
            probs = torch.softmax(scores.float(), dim=-1).to(state.dtype)
            attended = torch.matmul(probs, cache_v[layer])
            state = state + torch.matmul(attended, params["o"])
            expanded = torch.matmul(transformer_ops.rms_norm(state),
                                    params["w1"])
            state = state + torch.matmul(
                torch.nn.functional.gelu(expanded), params["w2"]
            )
        return state

    warm = step(token)
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    sink = None
    tokens = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        sink = step(token)
        xm.mark_step()
        tokens += batch
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    observed = transformer_ops.read_back(sink)
    steps = tokens // batch if batch else 0
    flops = steps * layers * transformer_ops.decode_step_flops(hidden, context, batch)

    return {
        "tokens_generated": tokens,
        "steps": steps,
        "elapsed_s": elapsed,
        "tokens_per_s": tokens / elapsed if elapsed else 0.0,
        "flops_issued": flops,
        "implied_tflops": flops / elapsed / 1e12 if elapsed else 0.0,
        "score_method": "workload",
        "analytic_basis": "tokens generated / wall time",
        **transformer_ops.output_check(
            observed, "decode output"),
    }


def cache_plan(problem: typing.Mapping[str, typing.Any]) -> typing.Dict[str, int]:
    """Cache geometry, and the traffic one append actually costs.

    **On this stack a KV cache cannot be updated in place, and that is the
    finding this workload exists to report.** XLA is functional:
    ``cache[:, a:b, :] = entry`` lowers to a dynamic-update-slice, which
    produces a *new* tensor. So appending 512 tokens to a 2 GiB cache does
    not move 256 MiB -- it reads 2 GiB and writes 2 GiB.

    Three measurements on trn1.2xlarge 2026-09-08 say so, each of which
    looked like a different problem:

      one layer, one token   115 us for 16 KiB, 1,794x what HBM needs
      32 layers, 64 tokens   455 ms for 32 MiB, 54x more than rewriting
                             every layer's whole slice would cost
      8 static ring slots    ~7 minutes to *compile* each slot's graph

    The first two read as dispatch overhead and then as a slow scatter.
    Neither was it. A graph that compiles for seven minutes to write a
    slice is a graph handling the whole 2 GiB tensor, and once that is
    true, everything else follows.

    So ``bytes_per_step`` reports the copy, not the slice. A workload that
    counted the slice would be reporting a twentieth of the traffic the
    hardware moves, which is the mistake this kernel has now made twice.

    Pure, so the sizing can be checked without a device.
    """
    hidden = int(problem["hidden"])
    context = int(problem["context"])
    layers = int(problem.get("layers", 8))
    slots = int(problem.get("ring_slots", 8))
    width = tiling.DTYPE_BYTES[str(problem["dtype"])]

    for label, value in (("hidden", hidden), ("context", context),
                         ("layers", layers), ("ring_slots", slots)):
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
    if context % slots:
        raise ValueError(
            f"context {context} must divide into {slots} whole ring slots"
        )

    tokens = context // slots
    # Two caches, K and V, each [layers, context, hidden].
    resident = 2 * layers * context * hidden * width
    # What the slice appears to write, and what the copy really costs.
    slice_bytes = tokens * layers * 2 * hidden * width
    per_step = 2 * resident        # read the cache, write a new one

    return {
        "hidden": hidden, "context": context, "layers": layers,
        "ring_slots": slots, "tokens_per_step": tokens,
        "element_bytes": width,
        "resident_bytes": resident,
        "slice_bytes": slice_bytes,
        "bytes_per_step": per_step,
    }


# Single-core HBM read bandwidth measured by memory_read on trn1.2xlarge,
# 2026-09-08. Used only to judge whether this workload is timing memory or
# timing the runtime -- not to score anything.
MEASURED_HBM_GBPS = 256.2


def verify_memory_bound(bytes_per_s: float,
                        hbm_gbps: float = MEASURED_HBM_GBPS,
                        floor: float = 0.05) -> typing.Optional[str]:
    """Say so when the Score is dispatch latency wearing a memory name.

    This workload claims to measure the memory traffic a long-running
    server pays. Before 2026-09-08 it did not: it wrote 16 KiB per step --
    K and V for a single layer -- and took 115 microseconds to do it, which
    is 1,794 times longer than HBM needs to move 16 KiB. The Score was the
    runtime's dispatch rate, and 0.056% of the part's bandwidth.

    Rate alone cannot show that. cache-updates/s looks the same whether
    each update moved a cache or a register, so the row has to carry the
    bandwidth it actually achieved and say when that is implausible.
    """
    if bytes_per_s <= 0:
        return "no bytes written -- the cache was never touched"
    achieved = bytes_per_s / 1e9
    if achieved < hbm_gbps * floor:
        return (
            f"wrote {achieved:.2f} GB/s, {achieved / hbm_gbps:.1%} of the "
            f"{hbm_gbps:.0f} GB/s this part measures -- this run timed "
            "dispatch, not memory, so the rate is not a cache measurement"
        )
    return None


def run_cache_churn(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Append K and V across every layer; count the entries, not the arithmetic."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    plan = cache_plan(problem)
    hidden, context = plan["hidden"], plan["context"]
    layers, tokens = plan["layers"], plan["tokens_per_step"]
    slots = plan["ring_slots"]
    dtype = tiling.torch_dtype(str(problem["dtype"]))

    device = xm.xla_device()
    # Per layer, as a real cache is. Writing one layer's worth per step was
    # what first made this workload measure the runtime instead of memory.
    cache_k = torch.ones((layers, context, hidden), dtype=dtype, device=device)
    cache_v = torch.ones((layers, context, hidden), dtype=dtype, device=device)
    entry = torch.ones((layers, tokens, hidden), dtype=dtype, device=device)
    xm.mark_step()
    xm.wait_device_ops()

    # A ring buffer, which is how a server evicts: the write wraps and
    # overwrites the oldest entries. Always writing slot zero would let the
    # compiler keep one block resident and never touch HBM.
    #
    # Each slot is a *static* slice, so the write is a contiguous copy and
    # the slot index never reaches the graph as a value. That is the whole
    # point. The previous version used index_copy_ with a runtime index --
    # a scatter -- and on trn1.2xlarge 2026-09-08 that cost 455 ms to move
    # 32 MiB: 3,471x what HBM needs for those bytes, and 54x more than
    # rewriting every layer's entire slice would have cost. The bytes were
    # never the problem; the scatter was.
    #
    # The price is one compiled graph per slot, which is why the ring is a
    # handful of large slots rather than a position per token.
    def writer(slot):
        start, stop = slot * tokens, (slot + 1) * tokens

        def write():
            cache_k[:, start:stop, :] = entry
            cache_v[:, start:stop, :] = entry

        return write

    writers = [writer(slot) for slot in range(slots)]

    for write in writers:      # compile every slot before the clock starts
        write()
        xm.mark_step()
    xm.wait_device_ops()

    updates = 0
    step = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        writers[step % slots]()
        xm.mark_step()
        step += 1
        updates += tokens
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    observed = transformer_ops.read_back(cache_k)
    steps = step
    bytes_written = steps * plan["bytes_per_step"]
    bytes_per_s = bytes_written / elapsed if elapsed else 0.0

    result = {
        "cache_updates": updates,
        "steps": steps,
        "elapsed_s": elapsed,
        "cache_updates_per_s": updates / elapsed if elapsed else 0.0,
        "bytes_written": bytes_written,
        # The number that says whether the rate above means anything.
        "cache_gbps": bytes_per_s / 1e9,
        "score_method": "workload",
        "analytic_basis": "cache entries written / wall time",
        "plan": plan,
        **transformer_ops.output_check(observed, "cache"),
    }

    # An unreadable or NaN cache still fails the row; a dispatch-bound run
    # is a warning, because the number is real, it just is not the number
    # the workload's name promises.
    if not result.get("warning"):
        result["warning"] = verify_memory_bound(bytes_per_s)
    return result
