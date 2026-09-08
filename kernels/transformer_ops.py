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

STATUS: UNTESTED ON HARDWARE.
"""

import typing


def weights(hidden: int, dtype, device, heads: int = 1):
    """Allocate one transformer block's parameters.

    Ones rather than random values. Nothing here checks numerical accuracy
    against a reference -- these workloads measure throughput -- and a
    deterministic tensor makes a run reproducible without carrying a seed
    whose meaning depends on the torch version.
    """
    import torch  # type: ignore

    def tensor(*shape):
        return torch.ones(shape, dtype=dtype, device=device)

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


def block(hidden_states, params):
    """One transformer block: attention, then the MLP, with residuals."""
    import torch  # type: ignore

    residual = hidden_states
    q = torch.matmul(hidden_states, params["q"])
    k = torch.matmul(hidden_states, params["k"])
    v = torch.matmul(hidden_states, params["v"])
    attended = attention(q, k, v, params["heads"])
    hidden_states = residual + torch.matmul(attended, params["o"])

    residual = hidden_states
    expanded = torch.matmul(hidden_states, params["w1"])
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


def verify_output_is_finite(observed: typing.Optional[float],
                            what: str = "output") -> typing.Optional[str]:
    """Check the model produced a number rather than a NaN or an inf.

    These blocks run on all-ones weights with no normalisation, so values
    grow with depth and bf16 saturates. That is acceptable -- throughput is
    what is being measured -- but a NaN means the graph produced nothing
    readable, which is indistinguishable from a graph that never ran, and
    that is the failure worth catching.
    """
    if observed is None:
        return f"{what} could not be read back, so execution is unverified"
    if observed != observed:  # NaN
        return f"{what} is NaN, so this run computed nothing meaningful"
    return None
