"""Repeated replay of a compiled NEFF graph.

Score: **graph-steps/s**, from neuron-monitor's execution counter --
``sum(completed) / sum(period)`` over the sampling periods the device was
busy throughout, read by the orchestrator after sampling stops. Not
``effective_flops``: this workload is not asking how fast the engines are,
it is asking how fast the runtime can dispatch.

That distinction is the whole point. A device can be perfectly healthy at
arithmetic and still serve inference badly, because every request pays
dispatch overhead before any engine does work. Small graph, many replays,
so the measurement is dominated by the submit-execute-complete cycle rather
than by the arithmetic inside it.

The graph is deliberately trivial and deliberately *not* eliminable: a
single small matmul whose result feeds the next iteration's input, so the
runtime cannot prove the replays dead. The chain is a cyclic permutation,
so its final value also says how many replays ran -- see ``run``.

**Every replay is its own execution, measured.** trn1.2xlarge 2026-09-10,
the monitor sampling every ~5 s through the compile, the 60,000 replays
and an idle tail, ``completed`` per sample:

    0 ... 0, 6657, 15409, 15273, 15199, 7465, 0, 0      sum 60,003

60,000 replays plus three setup graphs. The three whole periods read
3082, 3055 and 3040 per second against the loop's own 3057.

**This docstring said the opposite for most of a day, and the error was
in the reading, not the device.** It recorded "60,000 replays submitted,
14,737 executions completed, ratio 4.07" and concluded the runtime batched
about four replays per NEFF execution. ``completed`` is a tally per
sampling period -- it falls back to zero when the work stops -- and
neuron_monitor took its *maximum* as the run's total and *last minus
first* as the rate's numerator. 14,737 was one period's count. The
declared Score that came from last-minus-first -- 729.3, 1174.8, 1012 or
1506 graph-steps/s on different runs, or nothing when the last sample
was idle -- was the difference between two periods' tallies.

Both figures were from one run and they disagreed by four. Holding them
side by side is what showed something was wrong; it took the idle tail,
where a running total cannot fall and this counter did, to show which.

**The window.** This workload is bounded by ``replays`` as well as by
``duration``. At the old pin of 60,000 the count bound first, at about
20 seconds, and the harness run that verified the per-period reading had
one whole ~5 s period to divide. The pin is now 200,000, past the default
--duration, so the caller's duration sets the window. The orchestrator
also waits for one idle period before stopping the monitor, since the
last busy period's tally only arrives when it closes. Until then the
row's execution total read 45,709 for 60,000 replays.

STATUS: VERIFIED ON HARDWARE, trn1.2xlarge 2026-09-08 and 2026-09-10.
The declared Score source is settled: the per-period reading agrees with
the kernel's clock to 0.1% on the measured run.
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
    # A cyclic permutation, so the chain's value records how many replays
    # ran. weight shifts columns right by one; starting from the identity,
    # after n replays row r holds its single 1 at column (r + n) % hidden.
    #
    # Both operands were all ones until 2026-09-10. An all-ones chain grows
    # by `hidden` per replay and bf16 overflows to inf within a dozen, so
    # the read-back check -- "the chain has a value" -- was satisfied the
    # same way by 12 executed replays and by 60,000. Every value here is 0
    # or 1, exact in bf16, and each replay is still a full hidden^3 matmul:
    # the Tensor Engine's cost does not depend on the data.
    identity = torch.eye(hidden, dtype=torch.float32)
    weight = torch.roll(identity, shifts=1, dims=1).to(torch_dtype).to(device)
    state = identity.to(torch_dtype).to(device)
    xm.mark_step()
    xm.wait_device_ops()

    # Warm up so the compile is not inside the timed region, and warm up the
    # same shape the loop runs -- memory_read documents what a mismatched
    # warm-up costs. It is one replay of the chain, and the check counts it.
    state = torch.matmul(state, weight)
    xm.mark_step()
    xm.wait_device_ops()

    replays = 0
    started = time.perf_counter()
    deadline = started + duration

    while replays < plan["replays"] and time.perf_counter() < deadline:
        # Each replay consumes the previous result, so none can be proved
        # dead. That does not stop the runtime batching them -- see the
        # docstring: the chain prevents elimination, not coalescing.
        state = torch.matmul(state, weight)
        xm.mark_step()
        replays += 1

    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    # Every row must hold its 1 exactly where replays + WARMUP_REPLAYS
    # shifts put it. A skipped or repeated replay moves it; a chain that
    # never ran leaves nothing to read.
    shift = chain_shift(replays, hidden)
    try:
        host = state.to("cpu").float()
        expected = torch.roll(identity, shifts=shift, dims=1)
        chain_exact = bool(torch.equal(host, expected))
    except Exception:  # broad: materialisation failed; leave unverified
        chain_exact = None

    return {
        "replays": replays,
        "requested_replays": plan["replays"],
        "elapsed_s": elapsed,
        "graph_steps_per_s": replays / elapsed if elapsed else 0.0,
        "score_method": "analytic",
        "analytic_basis": "replays submitted / wall time",
        "warning": verify_replays_completed(replays, plan, elapsed, chain_exact),
        # A chain that ran the wrong number of replays, or none, means the
        # rate counted replays that did not happen.
        "score_invalid": chain_exact is not True,
        "chain_shift": shift,
        "plan": plan,
    }


# The warm-up is one replay of the same chain, and it shifts the state too.
WARMUP_REPLAYS = 1


def chain_shift(replays: int, hidden: int) -> int:
    """Columns the identity has moved after the timed replays and warm-up."""
    return (replays + WARMUP_REPLAYS) % hidden


def verify_replays_completed(
    replays: int,
    plan: typing.Mapping[str, int],
    elapsed: float,
    chain_exact: typing.Optional[bool],
) -> typing.Optional[str]:
    """Check the replays were executed rather than merely counted.

    The loop counts submissions, and submissions are cheap: ``mark_step``
    queues and returns. If the device never ran the chain, the count still
    climbs and the rate still looks plausible -- the same failure that made
    memory_read report 14,513 GB/s.

    ``chain_exact`` is whether the permutation chain landed where
    ``replays`` shifts put it (None when it could not be read). This used
    to be "could the result be read at all", which an all-ones chain that
    had overflowed to inf satisfied whatever the device ran.
    """
    if replays == 0:
        return "no replays were submitted"
    if chain_exact is None:
        return (
            "the replay chain could not be read back, so these steps were "
            "submitted but not shown to have executed"
        )
    if chain_exact is False:
        return (
            "the replay chain did not land where the submitted replays put "
            "it, so the device did not execute each replay exactly once"
        )
    if elapsed <= 0:
        return "no time elapsed"
    return None
