"""Repeated replay of a compiled NEFF graph.

Score: **graph-steps/s**, from ``delta(completed) / period`` as declared in
the registry -- the monitor's execution counter, read by the orchestrator
after sampling stops. Not ``effective_flops``: this workload is not asking
how fast the engines are, it is asking how fast the runtime can dispatch.

That distinction is the whole point. A device can be perfectly healthy at
arithmetic and still serve inference badly, because every request pays
dispatch overhead before any engine does work. Small graph, many replays,
so the measurement is dominated by the submit-execute-complete cycle rather
than by the arithmetic inside it.

The graph is deliberately trivial and deliberately *not* eliminable: a
single small matmul whose result feeds the next iteration's input, so the
runtime cannot batch the replays into one execution or prove them dead. The
chain is what makes each replay a separate completion for the counter to
see.

STATUS: VERIFIED ON HARDWARE as a workload, trn1.2xlarge 2026-09-08
and 2026-09-10. What is **not** settled is its declared Score source: the
monitor's execution counter produced a Score on 2026-09-08 (729.3
graph-steps/s) and not on 2026-09-10, where the row degraded to the
analytic fallback (3051.2 graph-steps/s via replays submitted / wall
time). Those two numbers are four times apart and are not measurements of
the same thing -- one counts what the device finished, the other what the
loop asked for -- so neither should be quoted without its Score Method.
The 2026-09-10 pass also flagged it irreproducible at cv 0.63.
"""

import time
import typing

from . import nki_backend, tiling


def replay_plan(problem: typing.Mapping[str, typing.Any]) -> typing.Dict[str, int]:
    """Graph size and replay count for the pinned problem.

    ``hidden`` sizes a square matmul. It is small on purpose: a graph big
    enough to saturate the engines would measure the engines, and this
    workload exists to measure everything around them.
    """
    hidden = int(problem["hidden"])
    replays = int(problem["replays"])

    if hidden <= 0 or hidden % 128:
        raise ValueError(
            f"hidden must be a positive multiple of 128, got {hidden}"
        )
    if replays <= 0:
        raise ValueError(f"replays must be positive, got {replays}")

    return {"hidden": hidden, "replays": replays,
            "flops_per_replay": 2 * hidden * hidden * hidden}


def run(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Replay a small graph and return the dispatch accounting.

    Returns what the kernel can see -- replays submitted and completed per
    second by its own clock. The Score comes from the monitor's execution
    counter instead, and the two disagreeing is itself informative: this
    counts what we asked for, the counter counts what the device finished.
    """
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    plan = replay_plan(problem)
    device = xm.xla_device()
    torch_dtype = tiling.torch_dtype(str(problem.get("dtype", "bf16")))

    hidden = plan["hidden"]
    weight = torch.ones((hidden, hidden), dtype=torch_dtype, device=device)
    state = torch.ones((hidden, hidden), dtype=torch_dtype, device=device)
    xm.mark_step()
    xm.wait_device_ops()

    # Warm up so the compile is not inside the timed region, and warm up the
    # same shape the loop runs -- memory_read documents what a mismatched
    # warm-up costs.
    state = torch.matmul(state, weight)
    xm.mark_step()
    xm.wait_device_ops()

    replays = 0
    started = time.perf_counter()
    deadline = started + duration

    while replays < plan["replays"] and time.perf_counter() < deadline:
        # Each replay consumes the previous result, so the runtime cannot
        # coalesce them: the chain forces one completion per iteration,
        # which is exactly what delta(completed) counts.
        state = torch.matmul(state, weight)
        xm.mark_step()
        replays += 1

    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    # Reading the result back proves the chain actually executed rather than
    # being queued and discarded. The value itself is unbounded -- an
    # all-ones chain grows without limit -- so only its finiteness is
    # meaningful, and bf16 saturates to inf quickly by design of the test.
    replayed = None
    try:
        replayed = float(state[0][0])
    except Exception:  # materialisation failed; leave unverified
        replayed = None

    return {
        "replays": replays,
        "requested_replays": plan["replays"],
        "elapsed_s": elapsed,
        "graph_steps_per_s": replays / elapsed if elapsed else 0.0,
        "score_method": "analytic",
        "analytic_basis": "replays submitted / wall time",
        "warning": verify_replays_completed(replays, plan, elapsed, replayed),
        "plan": plan,
    }


def verify_replays_completed(
    replays: int,
    plan: typing.Mapping[str, int],
    elapsed: float,
    replayed: typing.Optional[float],
) -> typing.Optional[str]:
    """Check the replays were dispatched rather than merely counted.

    The loop counts submissions, and submissions are cheap: ``mark_step``
    queues and returns. If the device never ran the chain, the count still
    climbs and the rate still looks plausible -- the same failure that made
    memory_read report 14,513 GB/s. Reading the chained result back is what
    distinguishes them, since an unexecuted chain has no value to read.
    """
    if replays == 0:
        return "no replays were submitted"
    if replayed is None:
        return (
            "the replay chain could not be read back, so these steps were "
            "submitted but not shown to have executed"
        )
    if elapsed <= 0:
        return "no time elapsed"
    return None
