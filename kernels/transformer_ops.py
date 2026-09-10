"""Transformer building blocks shared by the inference and training workloads.

These are torch operations on the XLA device, not NKI kernels. The Neuron
compiler lowers them, which is the point: these workloads measure the path a
real model takes, and a hand-written NKI kernel would measure something no
model actually runs.

**Sharing primitives is not sharing a workload.** pantheongpu collapsed the
units of twelve AI workloads into one `ai-ops/s` because ten of them shared
a kernel body and six compiled to byte-identical SASS -- they were one
measurement wearing twelve names. The workloads here compose these pieces
into genuinely different computations: prefill runs a 2048-token sequence
through every layer while decode runs a single token against a cache, and
those differ by three orders of magnitude in arithmetic per step. Anything
that would make two of them compile to the same graph belongs in one
workload, not two.

STATUS: VERIFIED ON HARDWARE indirectly, trn1.2xlarge 2026-09-10. This
module owns no workload of its own -- its status is the status of the
kernels that import it, and llm_inference, inference_mix, encoders and
transformer_compute have all passed on that part.

Saying so is not pedantry. This module said UNTESTED for a fortnight
after everything built on it had run, and because it owns no workload,
the check that caught the other ten could not see it: the check keyed on
workload ownership, and support code owns nothing. It was found by
reading the list the check had just filtered.
"""

import math
import typing


def weights(hidden: int, dtype, device, heads: int = 1):
    """Allocate one transformer block's parameters.

    Constant rather than random. Nothing here checks numerical accuracy
    against a reference -- these workloads measure throughput -- and a
    deterministic tensor makes a run reproducible without carrying a seed
    whose meaning depends on the torch version.

    **Each tensor is scaled by 1/fan_in, where fan_in is the dimension the
    matmul contracts.** Ones would grow activations by a factor of `hidden`
    per layer, leaving bf16's range after 11 layers of a 32-layer model.
    Scaling by a single 1/hidden is not enough either: `w2` contracts over
    4*hidden, so it still multiplies by four, and trn1.2xlarge measured
    llm_prefill as NaN with exactly that scaling on 2026-09-08.

    fan_in scaling makes every matmul norm-preserving on its own, which is
    necessary and -- on its own -- still not sufficient. See ``block`` for
    the normalisation that finishes the job.

    The arithmetic is untouched: same shapes, same graph, same FLOP count.
    Only the values differ.
    """
    import torch  # type: ignore

    def tensor(rows, columns):
        # rows is the contracted dimension for `x @ W`.
        return torch.full((rows, columns), 1.0 / rows,
                          dtype=dtype, device=device)

    return {
        "q": tensor(hidden, hidden),
        "k": tensor(hidden, hidden),
        "v": tensor(hidden, hidden),
        "o": tensor(hidden, hidden),
        # 4x expansion, the standard MLP ratio: the FLOP counts below assume
        # it, so changing it here changes what every derived figure means.
        "w1": tensor(hidden, 4 * hidden),
        "w2": tensor(4 * hidden, hidden),
        "heads": heads,
        "hidden": hidden,
    }


def attention(query, key, value, heads: int):
    """Scaled dot-product attention over [batch, seq, hidden].

    Written out rather than calling a fused implementation, because whether
    the Neuron compiler fuses it is one of the things these workloads are
    measuring. A library call that silently dispatches to a hand-tuned path
    would measure that path instead.
    """
    import torch  # type: ignore

    batch, seq, hidden = query.shape
    head_dim = hidden // heads

    def split(tensor):
        return tensor.view(batch, seq, heads, head_dim).transpose(1, 2)

    q, k, v = split(query), split(key), split(value)
    scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    context = torch.matmul(probs, v)
    return context.transpose(1, 2).reshape(batch, seq, hidden)


def rms_norm(hidden_states, eps: float = 1e-6):
    """Root-mean-square normalisation, as modern decoder stacks use.

    Not decoration. Without it the residual stream grows multiplicatively
    with depth, and attention scores are *quadratic* in that magnitude --
    a dot product over head_dim of values that are themselves growing. On
    trn1.2xlarge 2026-09-08 the scores left fp32's range around layer 19 of
    32, softmax turned the inf into a NaN, and llm_prefill reported a
    throughput number for a model that had computed nothing.

    Scaling the weights delays that; it does not prevent it. Normalising
    the branch input does, because the input to every matmul is then unit
    scale no matter how deep the stack, and the residual stream grows
    additively rather than multiplicatively -- about 2 per block instead of
    a factor of 4.

    It is also what a real transformer does, which is the whole premise of
    these workloads, and it puts real work on the vector and scalar engines
    that a pure matmul chain never exercises.
    """
    import torch  # type: ignore

    squared = hidden_states.float().pow(2).mean(-1, keepdim=True)
    return (hidden_states.float() * torch.rsqrt(squared + eps)).to(
        hidden_states.dtype)


def block(hidden_states, params):
    """One transformer block: attention, then the MLP, with residuals.

    Pre-norm, like Llama and GPT-NeoX: the normalisation is on the branch
    input, and the residual path stays clean. That ordering is what keeps a
    deep stack numerically alive; see ``rms_norm``.
    """
    import torch  # type: ignore

    residual = hidden_states
    normed = rms_norm(hidden_states)
    q = torch.matmul(normed, params["q"])
    k = torch.matmul(normed, params["k"])
    v = torch.matmul(normed, params["v"])
    attended = attention(q, k, v, params["heads"])
    hidden_states = residual + torch.matmul(attended, params["o"])

    residual = hidden_states
    normed = rms_norm(hidden_states)
    expanded = torch.matmul(normed, params["w1"])
    # GELU rather than ReLU: it exercises the scalar engine's transcendental
    # path, which a max() against zero does not.
    activated = torch.nn.functional.gelu(expanded)
    return residual + torch.matmul(activated, params["w2"])


def block_flops(hidden: int, seq: int, batch: int = 1) -> int:
    """FLOPs in one block for one forward pass.

    Four hidden x hidden projections, the two attention matmuls whose cost
    is quadratic in sequence length, and the 4x MLP. Counted so a workload
    can report what it issued; a GEMM is two FLOPs per multiply-accumulate.
    """
    projections = 4 * (2 * batch * seq * hidden * hidden)
    attention_matmuls = 2 * (2 * batch * seq * seq * hidden)
    mlp = 2 * (2 * batch * seq * hidden * 4 * hidden)
    return projections + attention_matmuls + mlp


def decode_step_flops(hidden: int, context: int, batch: int = 1) -> int:
    """FLOPs for one autoregressive step against a cached context.

    A single token's projections plus attention against ``context`` keys,
    which is linear in context rather than quadratic in sequence -- the
    arithmetic that makes decode latency-bound where prefill is
    compute-bound, and the reason they are separate workloads.
    """
    projections = 4 * (2 * batch * hidden * hidden)
    attention_matmuls = 2 * (2 * batch * context * hidden)
    mlp = 2 * (2 * batch * hidden * 4 * hidden)
    return projections + attention_matmuls + mlp


def read_back(tensor) -> typing.Optional[float]:
    """Materialise one element, proving the graph executed.

    Lives here because every workload that calls ``verify_output_is_a_number``
    needs it first, and the two belong together: this produces the value,
    that judges it. It was previously copied privately into four kernel
    modules, and ``omni_virus`` called it without having a copy -- a
    NameError that fired only after a full-duration run on real hardware,
    turning a completed stress run into a FAIL row. Nothing caught it
    because none of those modules had a test.

    Returns None when the read fails, which the caller reports as
    "unverified" rather than as a pass.
    """
    if tensor is None:
        return None
    try:
        # detach() before the scalar conversion. transformer_train_step
        # samples a *parameter* to see whether the optimiser moved it, and
        # a parameter carries requires_grad=True -- which made torch warn
        # "Converting a tensor with requires_grad=True to a scalar may
        # lead to unexpected behavior" on trn1.2xlarge 2026-09-10.
        #
        # The read was correct and the warning was right to fire: taking a
        # scalar off the autograd graph is exactly what this does, and
        # saying so explicitly is the difference between meaning it and
        # getting away with it. It also matters now that pytest.ini treats
        # warnings as errors -- this one only appears on hardware, so CI
        # would never have caught it.
        flat = tensor.reshape(-1)[0]
        return float(flat.detach() if hasattr(flat, "detach") else flat)
    except Exception:  # materialisation failed; leave unverified
        return None


# GELU of one, which is what every MLP branch in this suite computes.
#
# The exact (erf) form, which is torch's default. The tanh approximation
# gives 0.841192 -- 0.02% away, far inside the tolerance below, so which
# one a build uses does not change the verdict.
_GELU_OF_ONE = 0.8413447460685429


def ulp(magnitude: float, mantissa_bits: int = 8) -> float:
    """Spacing between representable values near ``magnitude``.

    Defaults to bfloat16's 8 explicit mantissa bits, which is what every
    compute workload here runs in.
    """
    if magnitude == 0:
        return 0.0
    if not math.isfinite(magnitude):
        # An infinity has no neighbours, so nothing added to it is
        # observable. Returning inf makes ``increment_is_observable`` say
        # so instead of raising, which is what it did: an inf output
        # reached this through the NaN gate -- inf is not NaN -- and
        # math.floor(inf) raised OverflowError from inside a verdict
        # function whose whole job is to return a message.
        return float("inf")
    return 2.0 ** (math.floor(math.log2(abs(magnitude))) - (mantissa_bits - 1))


def increment_is_observable(magnitude: float, increment: float,
                            mantissa_bits: int = 8) -> bool:
    """Whether adding ``increment`` to ``magnitude`` changes the value.

    This names the property that three separate defects in this suite
    violated, each found by deriving an expected output rather than by
    reading code:

    - **vision_encoder** projected patches with an unscaled ``torch.ones``,
      so the embedding was patch_dim = 588. bf16's ulp there is 4.0 and a
      block adds 1.84. Twelve blocks moved the output by exactly 0.0.
    - **speculative_decode** drafted through an unscaled square weight, so
      the state grew as ``1024^draft_len`` -- 1.1e15 after four steps,
      where the ulp is 8.8e12. All 32 target blocks were nine orders of
      magnitude below the resolution of the number carrying them.
    - **llm_prefill** is the same defect from the other end: unscaled
      weights grew the residual multiplicatively until it left bf16's
      range entirely and the output was NaN. That one was loud.

    The first two were silent, and silence is the dangerous case. The
    arithmetic ran in all three -- the FLOPs were issued and the
    throughput was real -- but in the first two the output did not depend
    on it, so no check on the output could distinguish a working kernel
    from a broken one.

    Scaling weights by ``1/fan_in`` keeps every stack at unit scale, which
    is what makes the derived outputs in this module checkable at all.
    """
    return abs(increment) >= ulp(magnitude, mantissa_bits) / 2


def stacked_block_output(layers: int, start: float = 1.0) -> float:
    """What a stack of ``layers`` blocks over all-ones input must produce.

    Derived, not measured. The kernels feed ``torch.ones`` through
    ``block`` with ``weights`` scaled by ``1/fan_in``, and every step of
    that is analytically determined:

        normed    = rms_norm(x)              -> ones, whatever x is
        q,k,v     = normed @ W(1/hidden)     -> ones
        attended  = attention(1, 1, 1)       -> ones (uniform softmax)
        h         = x + attended @ W         -> x + 1
        normed2   = rms_norm(h)              -> ones
        expanded  = normed2 @ W1(1/hidden)   -> ones
        activated = gelu(ones)               -> gelu(1)
        out       = h + activated @ W2       -> h + gelu(1)

    So each block adds ``1 + gelu(1) = 1.8413`` regardless of depth, and
    the answer is ``start + layers * 1.8413``. For llm_prefill's 32
    layers that is 59.92.

    **Two independent routes agree on this.** ``rms_norm``'s docstring
    says, from the reasoning that motivated it, that the residual stream
    "grows additively rather than multiplicatively -- about 2 per block
    instead of a factor of 4". 1.8413 is that 2.

    Pre-norm is what makes it depth-independent: the input to every matmul
    is unit scale however deep the stack, so neither branch's contribution
    depends on x. A stack that had lost its normalisation would grow
    multiplicatively and miss this by orders of magnitude.
    """
    return start + layers * (1.0 + _GELU_OF_ONE)


def verify_stack_computed_its_depth(
    observed: typing.Optional[float],
    layers: int,
    tolerance: float = 0.1,
) -> typing.Optional[str]:
    """Check a block stack produced the value its depth implies.

    Until 2026-09-10 these kernels checked only that the output was a
    readable number and not a NaN. That admits a stack that ran the wrong
    number of layers, dropped a residual, lost its normalisation, or
    scaled its weights wrongly -- each produces a finite number, and one
    of them (unscaled weights) had already put llm_prefill into NaN, which
    is the same defect at a magnitude loud enough to notice.

    **This is the check that verifies the layer count.** Nothing else in
    the suite does: the FLOP figure multiplies by ``layers`` whether or
    not that many ran, so a stack executing half its depth reports the
    full arithmetic at twice the throughput and looks like good news.

    The tolerance is 10%, and that number is measured rather than
    guessed. Simulating the residual walk in bf16 -- the rounding lands on
    the two additions per block, not on the matmuls, which accumulate in
    fp32 -- gives:

           1 layer    2.8438 against 2.8413    +0.08%
           4 layers   8.3750 against 8.3654    +0.12%
          32 layers  58.7500 against 59.9230    -1.96%

    So 10% is about five times the drift the arithmetic actually produces.
    A first estimate from worst-case ulp accumulation said 7%, which would
    have left almost no headroom; round-to-nearest does far better than
    worst case because the errors do not share a sign.

    The headroom matters in one direction specifically. This check sets
    ``score_invalid``, so a false positive turns a working run into a
    FAIL -- and a check that fails good runs gets its threshold widened
    until it means nothing, or deleted. The failures worth catching here
    are factors, not percentages.
    """
    if observed is None:
        return "stack output could not be read back to verify"
    if observed != observed:
        return "stack output is NaN"

    # Before comparing, ask whether a block could have moved this value at
    # all. At a magnitude where 1.8413 is below half an ulp the answer is
    # no, and "wrong value" is then the wrong complaint: the output is
    # bit-identical to running zero blocks, so it is not evidence about
    # the stack in either direction. Three kernels shipped in that state.
    step = 1.0 + _GELU_OF_ONE
    if not math.isfinite(observed):
        # Saturation. verify_output_is_a_number lets infinity pass on
        # purpose, because unnormalised stacks were expected to reach it;
        # a stack that is meant to land on a derived value has not.
        return (
            f"stack output saturated to {observed} -- a {layers}-layer "
            f"stack over all-ones input must produce "
            f"{stacked_block_output(layers):.4g}"
        )
    if layers and not increment_is_observable(observed, step):
        return (
            f"stack output {observed:.6g} sits where one bf16 step is "
            f"{ulp(observed):.4g}, so a block's {step:.4f} cannot change "
            "it -- this output is bit-identical to running zero blocks and "
            "says nothing about whether the stack ran"
        )

    expected = stacked_block_output(layers)
    if abs(observed - expected) > tolerance * expected:
        ran = (observed - 1.0) / (1.0 + _GELU_OF_ONE)
        return (
            f"a {layers}-layer stack over all-ones input must produce "
            f"{expected:.4g} and produced {observed:.6g} -- consistent with "
            f"about {ran:.1f} layers, or with a residual or normalisation "
            "that is not doing what the arithmetic assumes"
        )
    return None


def stack_check(observed: typing.Optional[float],
                layers: int) -> typing.Dict[str, typing.Any]:
    """The paired form, for the same reason ``output_check`` is paired."""
    message = verify_stack_computed_its_depth(observed, layers)
    return {"warning": message, "score_invalid": message is not None}


def verify_equals(observed: typing.Optional[float],
                  expected: float,
                  what: str,
                  tolerance: float = 0.01) -> typing.Optional[str]:
    """Check a value against one the pinned problem determines exactly.

    For the cases where the answer is a single number rather than a
    stack's depth: an all-ones GEMM over K terms reaches K, and scaling
    it by a dequantisation factor gives K * scale. No search, no
    accumulation, nothing to drift -- so the tolerance is tight.
    """
    if observed is None:
        return f"{what} could not be read back to verify"
    if observed != observed:
        return f"{what} is NaN"
    if not (abs(observed - expected) <= tolerance * max(1.0, abs(expected))):
        return (
            f"{what} is {observed:.6g} where the pinned problem determines "
            f"{expected:.6g} -- the arithmetic ran and produced the wrong "
            "number"
        )
    return None


def equals_check(observed: typing.Optional[float],
                 expected: float,
                 what: str) -> typing.Dict[str, typing.Any]:
    """The paired form, for the same reason ``output_check`` is paired."""
    message = verify_equals(observed, expected, what)
    return {"warning": message, "score_invalid": message is not None}


def verify_scatter_landed(observed: typing.Optional[float],
                          what: str = "output") -> typing.Optional[str]:
    """Check a scattered output is not still the zeros it started as.

    ``moe_router`` builds ``torch.zeros_like(hidden_states)`` and scatters
    each expert's result into it. A dispatch that routed nothing, or
    scattered to the wrong positions, or was elided, leaves that tensor
    untouched -- and ``verify_output_is_a_number`` passes it, because 0.0
    is a number.

    A zero here is not a small result. The expert matmul is over
    ``hidden`` terms of an input that is never zero, so every element the
    dispatch touches is far from zero and an exact zero means the
    dispatch did not touch this one.
    """
    if observed is None:
        return f"{what} could not be read back to verify"
    if observed != observed:
        return f"{what} is NaN"
    if observed == 0.0:
        return (
            f"{what} is exactly zero, which is what the destination was "
            "initialised to -- the dispatch scattered nothing here, and a "
            "token count cannot show that"
        )
    return None


def scatter_check(observed: typing.Optional[float],
                  what: str = "output") -> typing.Dict[str, typing.Any]:
    """The paired form, for the same reason ``output_check`` is paired."""
    message = verify_scatter_landed(observed, what)
    return {"warning": message, "score_invalid": message is not None}


def verify_optimiser_moved_the_model(
    before: typing.Optional[float],
    after: typing.Optional[float],
) -> typing.Optional[str]:
    """Check a training step actually changed a parameter.

    A workload called ``transformer_train_step`` runs a forward pass, a
    backward pass and an optimiser step, and reports train-steps/s. Every
    one of those executes. **None of them is evidence that the model was
    updated**, and until 2026-09-10 nothing here asked.

    The arithmetic says it is not. Parameters are bf16 at ``1/fan_in``;
    ``q`` sits at 2.44e-4 where bf16's ulp is 1.91e-6. The loss is a mean
    over batch*seq*hidden elements, so a weight's gradient is about
    2.05e-4 and an SGD step at lr=1e-4 moves it by 2.05e-8 -- **2% of a
    half-ulp.** SGD applies each update independently rather than into an
    accumulator, so every step rounds back to the same value and the
    weights are bit-identical after a thousand steps as after none.

    That does not make the Score wrong. train-steps/s measures the cost of
    forward, backward and step, and that cost is real -- the gradients are
    computed and the FLOPs are issued. It makes the Score's *name*
    misleading to anyone who reads "train" as "learns", and it means no
    check on the output could tell a working optimiser from one wired to
    nothing.

    So this warns rather than invalidating. A run that measured the right
    cost is a result; a run whose optimiser is decorative is a result the
    reader has to be told about.
    """
    if before is None or after is None:
        return "could not read a parameter back to see whether it moved"
    if before != after:
        return None
    return (
        f"the sampled parameter is unchanged at {before:.6g} after the "
        "whole run -- an SGD step at this learning rate moves a bf16 "
        "weight by a small fraction of one ulp, so the optimiser rounds "
        "to no change and the model never updates. The Score still "
        "measures what a training step costs; it does not measure "
        "training"
    )


def verify_attention_is_uniform(
    observed: typing.Optional[float],
    expected: float = 1.0,
    tolerance: float = 0.02,
) -> typing.Optional[str]:
    """Check attention over constant inputs produced the value it must.

    ``fused_attention`` runs q, k and v all ones. Every score is then
    ``head_dim / sqrt(head_dim)``, the same for every pair, so softmax
    returns exactly uniform weights of ``1/seq`` and the context is the
    weighted mean of v -- which is v's own constant value.

    **The answer is known in advance, so this is a correctness check and
    not a plausibility one.** Until 2026-09-10 the only check on this
    kernel was that the output was a readable number, which a saturated
    softmax, a transposed head reshape, an off-by-one in the sequence
    axis or a mask applied by mistake would all survive: each produces a
    finite number that is not 1.0.

    The tolerance is loose on purpose. ``1/2048`` is exactly representable
    in bf16 and the Tensor Engine accumulates in fp32, so the sum should
    land on 1.0 exactly -- but a check that is right for a reason this
    specific will fire on the first legitimate change to the pinned shape
    or dtype, and a check that cries wolf gets deleted.
    """
    if observed is None:
        return "attention output could not be read back to verify"
    if observed != observed:
        return "attention output is NaN"
    if abs(observed - expected) > tolerance * max(1.0, abs(expected)):
        return (
            f"attention over constant inputs returned {observed:.6g} where "
            f"uniform weights over identical values must give {expected:.6g} "
            "-- the arithmetic ran and computed the wrong thing"
        )
    return None


def attention_check(observed: typing.Optional[float],
                    expected: float = 1.0) -> typing.Dict[str, typing.Any]:
    """``output_check``'s stricter sibling, for the one kernel that can.

    Same contract -- a message and a ``score_invalid`` flag travelling
    together -- because separating them is how three NaN runs published
    Scores.
    """
    message = verify_attention_is_uniform(observed, expected)
    return {"warning": message, "score_invalid": message is not None}


def output_check(observed: typing.Optional[float],
                 what: str = "output") -> typing.Dict[str, typing.Any]:
    """Judge an output, and say whether a Score computed beside it survives.

    Both halves have to travel together. The 2026-09-08 full-coverage run
    caught three workloads whose output was NaN -- llm_prefill, llm_decode
    and speculative_decode -- and reported all three as PASS with a
    published Score, because the message went into the row's Detail and
    nothing read it. A number nobody can verify, wearing a PASS, is the
    exact failure this suite is built to prevent, and it took writing the
    check to notice the check was decorative.

    So a kernel returns ``**output_check(...)`` rather than
    ``"warning": verify_output_is_a_number(...)``, and the orchestrator
    fails the row on ``score_invalid``.
    """
    message = verify_output_is_a_number(observed, what)
    return {"warning": message, "score_invalid": message is not None}


def verify_output_is_a_number(observed: typing.Optional[float],
                              what: str = "output") -> typing.Optional[str]:
    """Check the model produced a number: readable, and not a NaN.

    **Infinity passes on purpose.** These blocks run on all-ones weights
    with no normalisation, so values grow with depth and bf16 saturates to
    inf long before the last layer. That is expected and it is fine --
    throughput is what is being measured, not numerical accuracy, and an
    inf still proves the graph ran.

    A NaN does not. It is what a graph that produced nothing readable looks
    like, which is indistinguishable from a graph that never ran, and that
    is the failure worth catching. NaN is literally Not a Number, so the
    name says exactly what is checked.

    Named this way after an earlier ``verify_output_is_finite``, whose name
    and summary line promised an inf check the body deliberately did not do.
    A check whose name overstates it is the same defect this suite spends
    its Score labelling on: the reader trusts the label, not the body.
    """
    if observed is None:
        return f"{what} could not be read back, so execution is unverified"
    if observed != observed:  # NaN
        return f"{what} is NaN, so this run computed nothing meaningful"
    return None
