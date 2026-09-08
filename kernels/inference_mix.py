"""The rest of the inference family: five workloads, five distinct shapes.

Score: **workload-counted** in every case. Nothing in hardware counts
requests, routed tokens or verified tokens.

What separates them, since sharing a module is not sharing a measurement:

``fused_attention``     attention alone -- no projections, no MLP. Isolates
                        the softmax and the two attention matmuls, so a
                        part whose softmax is slow shows up here and nowhere
                        else.
``quantized_gemm``      int8 operands with a dequantisation scale. A
                        different datapath from every bf16 workload, and the
                        scale is applied so the measurement includes the
                        conversion a real quantised model pays.
``moe_router``          top-k selection and expert dispatch. Dominated by
                        gather/scatter, not arithmetic: the routing decides
                        where tokens go, and moving them is the cost.
``speculative_decode``  a small draft model proposing several tokens, then
                        one batched verification pass over all of them.
                        Measures the acceptance pipeline's shape, not raw
                        decode.
``serving_mix``         prefill and decode interleaved at a serving ratio.
                        The only workload whose graph changes shape run to
                        run, which is what a real server does and what makes
                        it a distinct test.

STATUS: UNTESTED ON HARDWARE.
"""

import time
import typing

from . import nki_backend, tiling, transformer_ops


def run_fused_attention(problem: typing.Mapping[str, typing.Any],
                        duration: int) -> dict:
    """Attention only. Count tiles, where a tile is one head's sequence."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    heads = int(problem["heads"])
    seq = int(problem["seq"])
    head_dim = int(problem["head_dim"])
    dtype = tiling.torch_dtype(str(problem["dtype"]))
    hidden = heads * head_dim

    device = xm.xla_device()
    q = torch.ones((1, seq, hidden), dtype=dtype, device=device)
    k = torch.ones((1, seq, hidden), dtype=dtype, device=device)
    v = torch.ones((1, seq, hidden), dtype=dtype, device=device)
    xm.mark_step()

    warm = transformer_ops.attention(q, k, v, heads)
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    sink = None
    passes = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        sink = transformer_ops.attention(q, k, v, heads)
        xm.mark_step()
        passes += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    tiles = passes * heads
    flops = passes * 2 * (2 * seq * seq * hidden)

    return {
        "attention_tiles": tiles,
        "passes": passes,
        "elapsed_s": elapsed,
        "attention_tiles_per_s": tiles / elapsed if elapsed else 0.0,
        "flops_issued": flops,
        "score_method": "workload",
        "analytic_basis": "attention tiles / wall time",
        **transformer_ops.output_check(
            transformer_ops.read_back(sink), "attention output"),
    }


def run_quantized_gemm(problem: typing.Mapping[str, typing.Any],
                       duration: int) -> dict:
    """int8 GEMM with a dequantisation scale, counting quantised ops."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    m, n, k = (int(value) for value in problem["shape"])
    device = xm.xla_device()

    lhs = torch.ones((m, k), dtype=torch.int8, device=device)
    rhs = torch.ones((k, n), dtype=torch.int8, device=device)
    # Per-tensor scale, the simplest real quantisation scheme. Applying it
    # matters: a bare int8 matmul without dequantisation is not a path any
    # deployed model takes, and it would skip the conversion this workload
    # exists to measure.
    scale = torch.ones((1,), dtype=torch.float32, device=device) * 0.02
    xm.mark_step()

    def gemm():
        # int32 accumulation, then scaled back. int8 accumulators overflow
        # at K far below 4096.
        product = torch.matmul(lhs.to(torch.int32), rhs.to(torch.int32))
        return product.to(torch.float32) * scale

    warm = gemm()
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    sink = None
    passes = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        sink = gemm()
        xm.mark_step()
        passes += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    ops = passes * 2 * m * n * k

    return {
        "quantized_ops": ops,
        "passes": passes,
        "elapsed_s": elapsed,
        "quantized_ops_per_s": ops / elapsed if elapsed else 0.0,
        "score_method": "workload",
        "analytic_basis": "quantised ops / wall time",
        **transformer_ops.output_check(
            transformer_ops.read_back(sink), "quantised output"),
    }


def expert_capacity(tokens: int, top_k: int, experts: int) -> int:
    """Token slots each expert is given.

    Real MoE serving calls this a capacity factor. It exists because the
    number of tokens choosing a given expert is data-dependent, and a
    data-dependent shape cannot be compiled once.
    """
    return max(1, (tokens * top_k) // experts)


def routing_balance(tokens: int, experts: int,
                    top_k: int = 2) -> typing.Dict[int, int]:
    """How many tokens each expert receives under the pinned input pattern.

    The inputs are built so token ``t`` prefers expert ``t % experts`` and
    then ``(t + 1) % experts``. That is deterministic, needs no seed, and
    lands every expert on exactly ``tokens * top_k / experts`` -- which is
    the capacity, so nothing is dropped and no expert idles.

    **Why it is not uniform.** The workload previously fed all-ones
    activations through an all-ones gate. Every token's logits were then
    identical across experts, so top-k picked experts 0 and 1 for *every*
    token: six of eight experts received nothing, and three quarters of the
    routed slots were dropped at the capacity limit. The Score still
    counted every token as routed. A dispatch benchmark that dispatches
    everything to the same two experts is not measuring dispatch.

    Pure arithmetic so the balance can be asserted without a device.
    """
    counts = {expert: 0 for expert in range(experts)}
    for token in range(tokens):
        counts[token % experts] += 1
        if top_k > 1:
            counts[(token + 1) % experts] += 1
    return counts


def run_moe_router(problem: typing.Mapping[str, typing.Any],
                   duration: int) -> dict:
    """Route tokens to experts and dispatch them. Count routed tokens."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    experts = int(problem["experts"])
    top_k = int(problem["top_k"])
    hidden = int(problem["hidden"])
    tokens = int(problem["tokens"])

    device = xm.xla_device()
    dtype = torch.bfloat16

    # Built on the host and moved once: these are setup, not the workload,
    # and scattered index writes on the device would compile a graph each.
    #
    # The pattern gives token t a peak at expert t % experts and a second
    # peak at (t + 1) % experts, so top-2 routing lands every expert on
    # exactly `capacity` tokens. See routing_balance for why uniform inputs
    # -- what this used to have -- made six of eight experts idle.
    positions = torch.arange(tokens)
    host_states = torch.full((tokens, hidden), 0.1, dtype=torch.float32)
    host_states[positions, positions % experts] = 1.0
    host_states[positions, (positions + 1) % experts] = 0.5
    hidden_states = host_states.to(dtype).to(device)

    # Expert e reads dimension e, so logits[t, e] is hidden_states[t, e]
    # and the routing follows the pattern above rather than the gate.
    host_gate = torch.zeros((hidden, experts), dtype=torch.float32)
    host_gate[torch.arange(experts), torch.arange(experts)] = 1.0
    gate = host_gate.to(dtype).to(device)

    expert_weights = torch.ones((experts, hidden, hidden), dtype=dtype, device=device)
    xm.mark_step()

    capacity = expert_capacity(tokens, top_k, experts)

    def route():
        # The routing decision itself: a gate projection and a top-k over
        # experts. Cheap in FLOPs, and the reason this workload is not a
        # GEMM test.
        logits = torch.matmul(hidden_states, gate).float()
        weights, indices = torch.topk(logits, top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1).to(dtype)

        # Dispatch: gather the tokens an expert was given, run that
        # expert once, scatter the results back.
        #
        # What this replaced, and why. The previous version indexed the
        # expert weights by token -- `expert_weights[chosen]` with one
        # entry per token -- which materialises [tokens, hidden, hidden].
        # At the pinned problem that is 137 GB of weights for a 0.3 GB
        # weight table, and the compiler refused it outright on
        # trn1.2xlarge 2026-09-08: "Instructions generated by compiler
        # 4194304 exceeds the typical limit of 150000". No model dispatches
        # that way; the gather is over activations, not over parameters.
        output = torch.zeros_like(hidden_states)
        for expert in range(experts):
            # Which tokens picked this expert. topk over the mask gives a
            # fixed-size selection, so every expert compiles to one graph
            # regardless of how the routing actually fell.
            picked = (indices == expert).any(dim=-1).to(dtype)
            order = torch.topk(picked, capacity, dim=0).indices

            # The gather that dominates this workload: indirect reads of
            # token vectors, which defeat the prefetching a dense matmul
            # enjoys.
            gathered = hidden_states.index_select(0, order)
            transformed = torch.matmul(gathered, expert_weights[expert])

            # A token that did not choose this expert has mask 0, so it
            # contributes nothing even though capacity made room for it.
            scale = picked.index_select(0, order).unsqueeze(-1)
            output = output.index_add(0, order, transformed * scale)
        return output

    warm = route()
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    sink = None
    routed = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        sink = route()
        xm.mark_step()
        routed += tokens
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    return {
        "routed_tokens": routed,
        "elapsed_s": elapsed,
        "routed_tokens_per_s": routed / elapsed if elapsed else 0.0,
        "experts": experts,
        "top_k": top_k,
        # Slots per expert. The dispatch is one matmul of this many token
        # vectors per expert, so it is what the arithmetic scales with.
        "capacity": capacity,
        "score_method": "workload",
        "analytic_basis": "routed tokens / wall time",
        **transformer_ops.output_check(
            transformer_ops.read_back(sink), "router output"),
    }


def run_speculative_decode(problem: typing.Mapping[str, typing.Any],
                           duration: int) -> dict:
    """Draft several tokens cheaply, verify them in one pass, count verified."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    draft_len = int(problem["draft_len"])
    hidden = int(problem["hidden"])
    dtype = tiling.torch_dtype(str(problem["dtype"]))

    device = xm.xla_device()
    # The draft model is deliberately a quarter the width. Speculative
    # decoding only pays off when drafting is cheap, and a draft the same
    # size as the target would make this workload a slower copy of decode.
    draft_hidden = hidden // 4
    draft_w = torch.ones((draft_hidden, draft_hidden), dtype=dtype, device=device)
    project_up = torch.ones((draft_hidden, hidden), dtype=dtype, device=device)
    target = transformer_ops.weights(hidden, dtype, device, heads=32)

    draft_state = torch.ones((1, 1, draft_hidden), dtype=dtype, device=device)
    xm.mark_step()

    def cycle():
        # Draft: draft_len sequential cheap steps.
        proposals = []
        state = draft_state
        for _ in range(draft_len):
            state = torch.matmul(state, draft_w)
            proposals.append(state)
        drafted = torch.cat(proposals, dim=1)

        # Verify: one batched pass over all drafted positions at full width.
        # Batching the verification is the entire economic argument for
        # speculative decoding, so verifying one at a time would measure a
        # different algorithm.
        wide = torch.matmul(drafted, project_up)
        return transformer_ops.block(wide, target)

    warm = cycle()
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    sink = None
    verified = 0
    cycles = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        sink = cycle()
        xm.mark_step()
        cycles += 1
        verified += draft_len
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    return {
        "verified_tokens": verified,
        "cycles": cycles,
        "draft_len": draft_len,
        "elapsed_s": elapsed,
        "verified_tokens_per_s": verified / elapsed if elapsed else 0.0,
        "score_method": "workload",
        # Every drafted token is verified here; acceptance rate is a
        # property of a real model's agreement with its draft, which a
        # synthetic workload cannot honestly simulate. The number is
        # verification throughput, not end-to-end speculative speedup.
        "analytic_basis": "verified tokens / wall time",
        **transformer_ops.output_check(
            transformer_ops.read_back(sink), "verification output"),
    }


def interleave_period(ratio: float) -> int:
    """One prefill every N requests, for a requested prefill ratio.

    A deterministic interleave rather than a sampled one: sampling would
    make two runs of the same workload measure different mixes, and the mix
    *is* the workload. Returns 0 for a ratio of zero, meaning no prefills at
    all.

    **The period quantises the ratio, and the row reports both.** A period
    is a whole number of requests, so only ratios of the form 1/N are
    reproduced exactly. The pinned 0.2 is one of them (period 5). A ratio of
    0.6 rounds to period 2 and actually serves 0.5, which is why the result
    carries ``observed_prefill_ratio`` beside the requested one rather than
    echoing back the number that was asked for.
    """
    if not 0 <= ratio <= 1:
        raise ValueError(f"prefill_ratio must be in [0, 1], got {ratio}")
    if not ratio:
        return 0
    return max(int(round(1 / ratio)), 1)


def run_serving_mix(problem: typing.Mapping[str, typing.Any],
                    duration: int) -> dict:
    """Interleave prefill and decode at a serving ratio. Count requests."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    ratio = float(problem["prefill_ratio"])
    batch = int(problem["batch"])
    prompt = int(problem["prompt"])
    decode = int(problem["decode"])

    # Resolved before anything is allocated, so a bad ratio is rejected
    # rather than discovered after the weights are on the device. It also
    # validates: interleave_period raises on a ratio outside [0, 1].
    period = interleave_period(ratio)

    hidden = 4096
    dtype = torch.bfloat16
    device = xm.xla_device()
    params = transformer_ops.weights(hidden, dtype, device, heads=32)
    prompt_batch = torch.ones((1, prompt, hidden), dtype=dtype, device=device)
    decode_batch = torch.ones((batch, 1, hidden), dtype=dtype, device=device)
    xm.mark_step()

    warm = transformer_ops.block(decode_batch, params)
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    sink = None
    requests = 0
    prefills = 0
    decodes = 0
    started = time.perf_counter()
    deadline = started + duration

    while time.perf_counter() < deadline:
        is_prefill = period and (requests % period == 0)
        if is_prefill:
            sink = transformer_ops.block(prompt_batch, params)
            prefills += 1
        else:
            sink = transformer_ops.block(decode_batch, params)
            decodes += 1
        xm.mark_step()
        requests += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    return {
        "requests_completed": requests,
        "prefills": prefills,
        "decodes": decodes,
        "observed_prefill_ratio": prefills / requests if requests else 0.0,
        "elapsed_s": elapsed,
        "requests_per_s": requests / elapsed if elapsed else 0.0,
        "decode_tokens": decodes * batch,
        "requested_decode_length": decode,
        "score_method": "workload",
        "analytic_basis": "requests completed / wall time",
        **transformer_ops.output_check(
            transformer_ops.read_back(sink), "serving output"),
    }
