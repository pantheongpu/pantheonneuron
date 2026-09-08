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

STATUS: UNTESTED ON HARDWARE as workloads, though the training path itself
was verified on trn1.2xlarge 2026-08-27 at 23.25 train-steps/s with a real
backward pass and optimiser step.
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
        "warning": transformer_ops.verify_output_is_a_number(observed, "block output"),
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
    def parameter(*shape):
        tensor = torch.ones(shape, dtype=dtype, device=device)
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
    # Backward costs roughly twice the forward: one pass for input
    # gradients and one for weight gradients.
    forward = steps * layers * transformer_ops.block_flops(hidden, seq, batch)

    return {
        "steps_completed": steps,
        "elapsed_s": elapsed,
        "train_steps_per_s": steps / elapsed if elapsed else 0.0,
        "flops_issued": forward * 3,
        "implied_tflops": (forward * 3) / elapsed / 1e12 if elapsed else 0.0,
        "score_method": "workload",
        "analytic_basis": "optimiser steps / wall time",
        "warning": transformer_ops.verify_output_is_a_number(observed, "loss"),
    }
