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

    rate = ops / elapsed if elapsed else 0.0

    return {
        "quantized_ops": ops,
        "passes": passes,
        "elapsed_s": elapsed,
        "quantized_ops_per_s": rate,
        # The same number a reader can hold in their head. The registry
        # declares quantized-ops/s and that stays the Score, but 1.8e13 is
        # not a figure anyone checks against a datasheet -- the 2026-09-08
        # run printed 18442342834453.8 and it says nothing at a glance.
        # Recorded beside it, never instead of it.
        "quantized_tops": rate / 1e12,
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
            # argsort, not topk. torch-xla's topk here returns a pair that
            # neither `.indices` nor tuple unpacking gets a usable index
            # tensor out of: the first raised "'list' object has no
            # attribute 'indices'", and the second reached index_select
            # with the pair still intact -- "Expected an array shape. Got
            # (bf16[1024], u32[1024])", a SIGABRT inside the runtime rather
            # than a Python error, on trn1.2xlarge 2026-09-08. Both fired
            # only after the graph had compiled.
            #
            # argsort returns one tensor, so there is no pair to mishandle.
            # Descending puts the tokens that chose this expert first, and
            # the slice takes its capacity.
            order = torch.argsort(picked, descending=True)[:capacity]

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
    # A verification pass runs the target *model*, not one of its blocks.
    # Running a single block was the same defect serving_mix had: it makes
    # the expensive half of speculative decoding 1/layers of its real cost,
    # which is the half the whole technique is trying to amortise.
    layers = int(problem.get("layers", 32))
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

        # Verify: one batched pass over all drafted positions at full
        # width, through every layer of the target. Batching the
        # verification is the entire economic argument for speculative
        # decoding, so verifying one at a time would measure a different
        # algorithm -- and verifying through one block would measure a
        # thirty-second of the cost the technique exists to amortise.
        state = torch.matmul(drafted, project_up)
        for _ in range(layers):
            state = transformer_ops.block(state, target)
        return state

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
        "target_layers": layers,
        # The verification cost, which is what speculative decoding is
        # amortising. A cycle count alone cannot show whether the target
        # model was actually run.
        "verify_blocks_per_cycle": layers,
        "verify_flops_per_cycle": layers * transformer_ops.block_flops(
            hidden, draft_len),
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

def serving_plan(problem: typing.Mapping[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    """What one scheduler step costs, and when a request is actually done.

    A continuous-batching server schedules *steps*, not requests: each
    iteration is either a prefill (a whole prompt through every layer) or a
    decode step (one token for each sequence in flight, through every
    layer). A request is finished when its tokens have been produced.

    This exists because the workload counted neither. It ran **one
    transformer block** per "request", while a real prefill runs `layers`
    blocks and a real 256-token decode request runs `decode * layers` --
    overstating by 32x and 8,192x at the pinned problem. It also reported
    ``requested_decode_length: 256`` beside a loop that never generated a
    second token.

    Pure, so the accounting can be checked without a device.
    """
    ratio = float(problem["prefill_ratio"])
    batch = int(problem["batch"])
    prompt = int(problem["prompt"])
    decode = int(problem["decode"])
    layers = int(problem.get("layers", 32))
    hidden = int(problem.get("hidden", 4096))

    for label, value in (("batch", batch), ("prompt", prompt),
                         ("decode", decode), ("layers", layers)):
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")

    return {
        "period": interleave_period(ratio),
        "batch": batch, "prompt": prompt, "decode": decode,
        "layers": layers, "hidden": hidden,
        "blocks_per_step": layers,
        # A decode request is done when it has produced `decode` tokens,
        # and every decode step produces one per sequence in flight.
        "decode_steps_per_request": -(-decode // batch),
        "prefill_flops": layers * transformer_ops.block_flops(hidden, prompt),
        "decode_flops": layers * transformer_ops.block_flops(hidden, 1, batch),
    }


def verify_mix_was_observed(steps: int, period: int) -> typing.Optional[str]:
    """Say so when the run never completed one interleave cycle.

    The mix is defined by ``period`` -- one prefill every N steps -- so
    fewer than N steps have not sampled it. The observed ratio is then an
    artefact of where the run stopped, and requests/s is whatever the first
    step or two happened to be.

    Measured on trn1.2xlarge 2026-09-09: a 20-second run completed **one**
    scheduler step, reported 0.031 requests/s from that single prefill, and
    carried no caveat at all -- because the other guard only fires once a
    decode step has run, and none had.
    """
    if period and steps < period:
        return (
            f"{steps} scheduler step(s) against an interleave period of "
            f"{period}: the run never completed one cycle of the mix, so "
            "this rate is whichever step it managed rather than the mix"
        )
    return None


def verify_requests_completed(prefills: int, decode_steps: int,
                              decode_requests: int) -> typing.Optional[str]:
    """Say so when the mix never finished a decode request.

    The Score is prefills plus completed decode requests. A decode request
    needs `decode / batch` steps, and at the pinned 256 tokens that is 32 --
    so a run short enough to produce decode *tokens* but no decode
    *request* reports a requests/s that is prefills only, while the row
    still claims to measure a prefill/decode mix.

    Measured on trn1.2xlarge 2026-09-08: 0.0312 requests/s from a 20-second
    run, where the scheduler sustained about one step a second. The number
    is arithmetically correct and describes half the workload.
    """
    if decode_steps > 0 and decode_requests == 0:
        return (
            f"{decode_steps} decode steps completed no decode request, so "
            "this requests/s is prefills only -- the run is too short for "
            "the pinned decode length, and the mix it reports is not the "
            "mix it measured"
        )
    return None


def run_serving_mix(problem: typing.Mapping[str, typing.Any],
                    duration: int) -> dict:
    """Interleave prefill and decode steps at a serving ratio."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    plan = serving_plan(problem)
    batch, prompt = plan["batch"], plan["prompt"]
    layers, hidden = plan["layers"], plan["hidden"]
    period = plan["period"]

    dtype = torch.bfloat16
    device = xm.xla_device()
    params = transformer_ops.weights(hidden, dtype, device, heads=32)
    prompt_batch = torch.ones((1, prompt, hidden), dtype=dtype, device=device)
    decode_batch = torch.ones((batch, 1, hidden), dtype=dtype, device=device)
    xm.mark_step()

    def step(state):
        """One scheduler step: the whole stack, not one block.

        Running a single block and calling it a request was the defect
        here. A prefill runs every layer, and so does a decode step.
        """
        for _ in range(layers):
            state = transformer_ops.block(state, params)
        return state

    # Both shapes, not just decode. A prefill and a decode step are
    # different graphs, so warming only one leaves the other to compile
    # inside the timed region -- the defect memory_read documents, where a
    # mismatched warm-up measured seven minutes of compilation as
    # bandwidth. Measured here on trn1.2xlarge 2026-09-09: a 20-second run
    # completed exactly one scheduler step, and that step took about 32
    # seconds because it was the prefill compiling.
    for shape in (decode_batch, prompt_batch):
        warm = step(shape)
        xm.mark_step()
        xm.wait_device_ops()
        del warm

    sink = None
    steps = prefills = decode_steps = 0
    started = time.perf_counter()
    deadline = started + duration

    # A deterministic interleave rather than a sampled one: sampling would
    # make two runs of the same workload measure different mixes, and the
    # mix is the workload.
    while time.perf_counter() < deadline:
        if period and steps % period == 0:
            sink = step(prompt_batch)
            prefills += 1
        else:
            sink = step(decode_batch)
            decode_steps += 1
        xm.mark_step()
        steps += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    # A decode request finishes when its tokens exist. Every decode step
    # produces one token per sequence in flight, so completed requests are
    # tokens over the requested length -- not, as before, one per step
    # regardless of how many tokens the request asked for.
    decode_tokens = decode_steps * batch
    decode_requests = decode_tokens // plan["decode"]
    requests = prefills + decode_requests
    flops = prefills * plan["prefill_flops"] + decode_steps * plan["decode_flops"]

    return {
        "requests_completed": requests,
        "prefills": prefills,
        "decode_requests": decode_requests,
        "decode_steps": decode_steps,
        "scheduler_steps": steps,
        "observed_prefill_ratio": prefills / steps if steps else 0.0,
        "elapsed_s": elapsed,
        "requests_per_s": requests / elapsed if elapsed else 0.0,
        # The rate the scheduler actually sustained. A request is many
        # steps, so these differ by a large factor and both are wanted.
        "scheduler_steps_per_s": steps / elapsed if elapsed else 0.0,
        "decode_tokens": decode_tokens,
        "decode_length": plan["decode"],
        "blocks_executed": steps * layers,
        # The cross-check: arithmetic issued, which a request count alone
        # cannot contradict.
        "implied_tflops": flops / elapsed / 1e12 if elapsed else 0.0,
        "plan": plan,
        "score_method": "workload",
        "analytic_basis": "requests completed / wall time",
        **_mix_warning(
            steps, period, prefills, decode_steps, decode_requests,
            transformer_ops.output_check(
                transformer_ops.read_back(sink), "serving output")),
    }


def _mix_warning(steps: int, period: int, prefills: int, decode_steps: int,
                 decode_requests: int,
                 checked: typing.Dict[str, typing.Any]
                 ) -> typing.Dict[str, typing.Any]:
    """Add whichever mix caveats apply, without displacing an output failure.

    Two of them, and the narrower one alone was not enough: a run short
    enough to complete no decode step slipped past it entirely.
    """
    notes = [
        verify_mix_was_observed(steps, period),
        verify_requests_completed(prefills, decode_steps, decode_requests),
    ]
    found = [note for note in notes if note]
    if not found:
        return checked
    existing = checked.get("warning")
    return {**checked,
            "warning": "; ".join(([existing] if existing else []) + found)}
