"""Sustained transformer load, and the training step.

Two workloads that run the same block for different reasons:

``transformer_virus``     forward passes back to back, scored from
                          neuron-monitor's effective_flops. It is a power
                          virus with a realistic instruction mix rather than
                          a synthetic GEMM -- the question is what a real
                          model's blend of matmul, softmax and GELU does to
                          clocks and thermals, which a pure matmul cannot
                          answer.
``transformer_train_step`` forward, backward and an optimiser step, scored by
                          the workload in steps/s. Backward roughly doubles
                          the arithmetic and adds gradient traffic the
                          forward path never generates, and on Neuron it is
                          the only workload that exercises the training
                          capability at all -- which is why it is gated on
                          it and skips on Inferentia.

STATUS: VERIFIED ON HARDWARE, trn1.2xlarge 2026-09-10:
``transformer_virus`` 53.1 TFLOPS via neuron-monitor,
``transformer_train_step`` 3.64 train-steps/s.

``transformer_train_step`` failed on 2026-09-08 -- the pinned problem did
not fit device memory -- and the registry repinned it to batch 1 /
layers 4, near 6.5 GB, which leaves room for the optimiser state rather
than only just fitting. That was the fourth pinned problem in this repo
found unreachable on the part it targets. The 23.25 train-steps/s
measured on 2026-08-27 was a different, smaller problem and is not
comparable to the 3.64 above.
"""

import time
import typing

from . import nki_backend, tiling, transformer_ops


def run_virus(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Run blocks back to back; the Score comes from the monitor."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    hidden = int(problem["hidden"])
    heads = int(problem["heads"])
    seq = int(problem["seq"])
    dtype = tiling.torch_dtype(str(problem["dtype"]))

    device = xm.xla_device()
    params = transformer_ops.weights(hidden, dtype, device, heads=heads)
    hidden_states = torch.ones((1, seq, hidden), dtype=dtype, device=device)
    xm.mark_step()

    warm = transformer_ops.block(hidden_states, params)
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    sink = None
    passes = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        sink = transformer_ops.block(hidden_states, params)
        xm.mark_step()
        passes += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    observed = transformer_ops.read_back(sink)
    flops = passes * transformer_ops.block_flops(hidden, seq)

    return {
        "passes": passes,
        "elapsed_s": elapsed,
        "flops_issued": flops,
        "analytic_tflops": flops / elapsed / 1e12 if elapsed else 0.0,
        "analytic_unit": "TFLOPS",
        "score_method": "analytic",
        "analytic_basis": "block FLOPs issued / wall time",
        # One block over all-ones input, so the answer is 1 + gelu(1) =
        # 2.8413 exactly. Checking it verifies the residuals and the
        # normalisation actually ran, which a finiteness check cannot:
        # every way of getting this wrong produces a finite number.
        **transformer_ops.stack_check(observed, 1),
    }


def run_train_step(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Forward, backward, optimiser step. Count completed steps."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    hidden = int(problem["hidden"])
    layers = int(problem["layers"])
    batch = int(problem["batch"])
    seq = int(problem["seq"])
    dtype = tiling.torch_dtype(str(problem["dtype"]))

    device = xm.xla_device()

    # Real parameters this time: backward needs gradients, which needs
    # leaves that require grad, so these cannot be the shared constant
    # tensors the forward-only workloads use.
    # Scaled like transformer_ops.weights, and for the same reason: ones
    # grow activations by `hidden` per layer and leave bf16's range within
    # a few blocks. Backward makes it worse -- a NaN loss produces NaN
    # gradients, so the optimiser step then corrupts every parameter and
    # every later step measures a model of NaNs.
    def parameter(rows, columns):
        # 1/fan_in, where fan_in is the contracted dimension -- see
        # transformer_ops.weights. A single 1/hidden leaves w2 four times
        # too large, which is what made llm_prefill NaN at depth.
        tensor = torch.full((rows, columns), 1.0 / rows,
                            dtype=dtype, device=device)
        tensor.requires_grad_(True)
        return tensor

    params = [
        {
            "q": parameter(hidden, hidden), "k": parameter(hidden, hidden),
            "v": parameter(hidden, hidden), "o": parameter(hidden, hidden),
            "w1": parameter(hidden, 4 * hidden),
            "w2": parameter(4 * hidden, hidden),
            "heads": 32, "hidden": hidden,
        }
        for _ in range(layers)
    ]
    flat = [tensor for layer in params for key, tensor in layer.items()
            if hasattr(tensor, "requires_grad")]
    optimiser = torch.optim.SGD(flat, lr=1e-4)

    inputs = torch.ones((batch, seq, hidden), dtype=dtype, device=device)
    xm.mark_step()

    # One parameter element, sampled before any step runs. Nothing else in
    # this workload can tell a working optimiser from a decorative one:
    # the forward, the backward and the step all execute either way, and
    # the loss is a function of parameters that may never have moved.
    sampled = params[0]["w2"]
    before = transformer_ops.read_back(sampled)

    def one_step():
        optimiser.zero_grad()
        state = inputs
        for layer in params:
            state = transformer_ops.block(state, layer)
        # A scalar objective, which is all a backward pass needs. The value
        # is meaningless; the gradient traffic it generates is the workload.
        loss = state.float().mean()
        loss.backward()
        # xm.optimizer_step is the XLA-aware step: it inserts the barrier
        # the Neuron runtime needs and is what the 2026-08-27 trn1 bring-up
        # exercised.
        xm.optimizer_step(optimiser, barrier=True)
        return loss

    warm = one_step()
    xm.wait_device_ops()
    del warm

    sink = None
    steps = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        sink = one_step()
        steps += 1
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    observed = transformer_ops.read_back(sink)
    after = transformer_ops.read_back(sampled)
    # Backward costs roughly twice the forward: one pass for input
    # gradients and one for weight gradients.
    forward = steps * layers * transformer_ops.block_flops(hidden, seq, batch)

    return {
        "steps_completed": steps,
        "elapsed_s": elapsed,
        "train_steps_per_s": steps / elapsed if elapsed else 0.0,
        "flops_issued": forward * 3,
        "implied_tflops": (forward * 3) / elapsed / 1e12 if elapsed else 0.0,
        # Whether the model moved, which "train-steps/s" does not say.
        "parameter_before": before,
        "parameter_after": after,
        "parameter_moved": (before is not None and after is not None
                            and before != after),
        "score_method": "workload",
        "analytic_basis": "optimiser steps / wall time",
        # Two things, and only one of them invalidates a Score.
        #
        # A NaN loss means nothing ran usefully, so output_check's verdict
        # stands. A model that never moved is a different statement: the
        # forward, the backward and the step all executed and the Score
        # correctly measures what they cost. It is the *name* that
        # misleads, so it warns.
        "expected_loss": transformer_ops.stacked_block_output(layers),
        **_train_step_check(observed, before, after, layers),
    }


def _train_step_check(observed, before, after, layers):
    """Pair the loss verdict with the did-it-train verdict.

    The loss is derivable, and the two verdicts constrain each other.
    Inputs are ones and parameters are 1/fan_in, so a forward pass through
    ``layers`` blocks produces ``stacked_block_output(layers)`` and the
    loss is its mean over identical elements -- 8.3654 at the pinned four
    layers.

    That is the value on the *first* step. On later steps it holds only if
    the parameters have not moved, which is exactly what the other verdict
    reports. So a run where the loss stays at 8.3654 and the parameter is
    unchanged is internally consistent, and a run where one moves without
    the other is not: **either both or neither**, and a row showing one
    alone means one of the two checks is measuring the wrong thing.

    Order matters. A NaN or wrong loss invalidates the Score and is
    reported first; an unmoved model does not invalidate anything and is
    appended, so a run with both says the more serious thing first.
    """
    expected = transformer_ops.stacked_block_output(layers)
    result = transformer_ops.equals_check(observed, expected, "loss")
    moved = transformer_ops.verify_optimiser_moved_the_model(before, after)
    if moved:
        result["warning"] = "; ".join(
            part for part in (result.get("warning"), moved) if part)
    return result
