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
            q = torch.matmul(state, params["q"])
            # Attention against the cached context rather than against the
            # single token: linear in context, which is the shape decode
            # actually has and prefill does not.
            scores = torch.matmul(q, cache_k[layer].transpose(-1, -2))
            probs = torch.softmax(scores.float(), dim=-1).to(state.dtype)
            attended = torch.matmul(probs, cache_v[layer])
            state = state + torch.matmul(attended, params["o"])
            expanded = torch.matmul(state, params["w1"])
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


def run_cache_churn(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Write and evict cache entries; count the updates, not the arithmetic."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    hidden = int(problem["hidden"])
    context = int(problem["context"])
    dtype = tiling.torch_dtype(str(problem["dtype"]))

    device = xm.xla_device()
    cache_k = torch.ones((context, hidden), dtype=dtype, device=device)
    cache_v = torch.ones((context, hidden), dtype=dtype, device=device)
    entry = torch.ones((1, hidden), dtype=dtype, device=device)
    xm.mark_step()
    xm.wait_device_ops()

    # A ring buffer, which is how a server actually evicts: the write index
    # wraps and overwrites the oldest entry. Writing always to position zero
    # would let the compiler keep one row in SBUF and never touch HBM,
    # measuring a register file instead of a cache.
    #
    # The index has to reach the device as a *value*, not as a Python int.
    # `cache_k[position] = ...` bakes the position into the graph, so every
    # distinct position is a different graph and every iteration pays a
    # compile. Measured on trn1.2xlarge 2026-09-08: 0.85 cache-updates/s,
    # which is a compiler's throughput, not a cache's. Copying a host
    # scalar into a device tensor of fixed shape keeps one graph and makes
    # the position an input to it.
    host_index = torch.zeros(1, dtype=torch.int64)
    index = torch.zeros(1, dtype=torch.int64, device=device)

    def churn(position):
        host_index[0] = position
        index.copy_(host_index)
        cache_k.index_copy_(0, index, entry)
        cache_v.index_copy_(0, index, entry)

    churn(0)
    xm.mark_step()
    xm.wait_device_ops()

    updates = 0
    position = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        churn(position)
        xm.mark_step()
        position = (position + 1) % context
        updates += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    observed = transformer_ops.read_back(cache_k)
    bytes_written = updates * 2 * hidden * tiling.DTYPE_BYTES[str(problem["dtype"])]

    return {
        "cache_updates": updates,
        "elapsed_s": elapsed,
        "cache_updates_per_s": updates / elapsed if elapsed else 0.0,
        "bytes_written": bytes_written,
        "score_method": "workload",
        "analytic_basis": "cache updates / wall time",
        **transformer_ops.output_check(observed, "cache"),
    }
