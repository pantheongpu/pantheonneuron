"""Duty-cycled matmul for the ``pulse_virus`` workload.

Score: **TFLOPS**, from ``mean(effective_flops) / 1e12`` as declared in the
registry, read by the orchestrator from neuron-monitor after the run.

The load is the same GEMM ``tensor_virus`` runs, switched on and off on a
fixed period. That is the entire point of the workload: a part that is
stable under constant load can still fail on the *transitions*, when the
clock and voltage regulators have to track a step change. Holding the
engine at 100% never exercises that, which is why this exists as a separate
workload rather than a flag on tensor_virus.

**Its Score is deliberately not comparable with tensor_virus.**
``effective_flops`` is averaged over the whole run by the monitor, and half
of this run is idle by construction, so a healthy part reports roughly the
duty cycle times the sustained figure. Reading it as a throughput
regression would be a mistake; what it is for is the *stability* of the
number and the throttle counter recorded beside it.

STATUS: VERIFIED ON HARDWARE, trn1.2xlarge 2026-09-08 and 2026-09-10,
at the pinned 8192^3 shape that had never compiled when this note was
first written: 14.4 TFLOPS via neuron-monitor.

That figure is roughly half tensor_virus's sustained number and is meant
to be. The analytic basis spans the idle halves of the duty cycle, so it
lines up with the monitor's average rather than with a sustained rate --
which is the entire point of the workload.
"""

import time
import typing

from . import nki_backend, tensor_virus, tiling


def duty_plan(problem: typing.Mapping[str, typing.Any]) -> typing.Dict[str, float]:
    """Split the pinned period into its loaded and idle halves.

    A duty cycle of 0 or 1 is rejected rather than clamped: both are
    legitimate loads and neither is this workload, so a run that asked for
    one is asking for the wrong thing and should say so instead of
    silently measuring something else.
    """
    duty = float(problem.get("duty_cycle", 0.5))
    period = float(problem.get("period_s", 2))

    if not 0 < duty < 1:
        raise ValueError(
            f"duty_cycle must be strictly between 0 and 1, got {duty} -- "
            "a constant load is tensor_virus, and an idle one measures nothing"
        )
    if period <= 0:
        raise ValueError(f"period_s must be positive, got {period}")

    return {"duty": duty, "period": period,
            "on_s": period * duty, "off_s": period * (1 - duty)}


def run(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Pulse the Tensor Engine and return timing plus FLOP accounting.

    Returns the analytic cross-check, not the Score. The declared source is
    neuron-monitor, which the orchestrator reads after the monitor stops.
    """
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    dtype = str(problem["dtype"])
    plan = tensor_virus.gemm_plan(problem["shape"], dtype)
    cycle = duty_plan(problem)
    _, _, kernel = tensor_virus._build_kernel(dtype)

    device = xm.xla_device()
    torch_dtype = tiling.torch_dtype(dtype)

    lhs_t = torch.ones((plan["k"], plan["m"]), dtype=torch_dtype, device=device)
    rhs = torch.ones((plan["k"], plan["n"]), dtype=torch_dtype, device=device)
    xm.mark_step()

    # Warm up the graph the loop runs, holding the result so the liveness
    # matches -- tensor_virus documents why a discarding warm-up compiles a
    # different graph and leaves the real compile inside the measurement.
    warm = kernel(lhs_t, rhs)
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    sink = None
    passes = 0
    pulses = 0
    loaded_s = 0.0
    started = time.perf_counter()
    deadline = started + duration

    while time.perf_counter() < deadline:
        # Loaded half. The barrier stays inside it: without one the pulse
        # would end when submission ended rather than when the device did,
        # and the "idle" half would absorb the tail of the load -- which is
        # exactly the transition this workload exists to provoke.
        pulse_start = time.perf_counter()
        pulse_end = min(pulse_start + cycle["on_s"], deadline)
        while time.perf_counter() < pulse_end:
            sink = kernel(lhs_t, rhs)
            xm.mark_step()
            passes += 1
        xm.wait_device_ops()
        loaded_s += time.perf_counter() - pulse_start
        pulses += 1

        # Idle half, trimmed so a pulse never runs past the deadline.
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        time.sleep(min(cycle["off_s"], remaining))

    elapsed = time.perf_counter() - started

    product_verified = None
    if sink is not None:
        try:
            corner = float(sink[0][0])
            far = float(sink[plan["m"] - 1][plan["n"] - 1])
        except Exception:  # materialisation failed; leave unverified
            corner = far = None
        if corner is not None and far is not None and plan["k"]:
            product_verified = (corner + far) / 2.0 / plan["k"]

    flops_issued = plan["flops_per_pass"] * passes

    return {
        "passes": passes,
        "pulses": pulses,
        "elapsed_s": elapsed,
        "loaded_s": loaded_s,
        "flops_issued": flops_issued,
        # Over wall time, including the idle halves, so it is directly
        # comparable with the monitor's average and not with tensor_virus.
        "analytic_tflops": flops_issued / elapsed / 1e12,
        # Over the loaded halves only: what the engine did while it was
        # actually working, which is the figure to compare against
        # tensor_virus when asking whether pulsing cost throughput.
        "loaded_tflops": (flops_issued / loaded_s / 1e12) if loaded_s else 0.0,
        "analytic_unit": "TFLOPS",
        "score_method": "analytic",
        "analytic_basis": "FLOPs issued / wall time, including idle halves",
        "warning": tensor_virus.verify_product_is_correct(product_verified),
        "plan": plan,
        "duty": cycle,
        "product_verified_ratio": product_verified,
    }
