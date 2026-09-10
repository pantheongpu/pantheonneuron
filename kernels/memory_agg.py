"""Aggregate HBM bandwidth across every NeuronCore.

Score: **GB/s**, from ``sum(hbm_read_bytes over cores) / total_time`` as the
registry declares.

The single-core kernels answer "how fast is one core's path to memory". This
answers a different question: whether the cores *share* that path. A part
whose per-core bandwidth is healthy can still aggregate to barely more than
one core's worth, because HBM, the on-package interconnect, or the memory
controller is the shared bottleneck. That gap is invisible to
``memory_read`` and is exactly what an inference server hits when it finally
uses every core at once.

**Why processes and not threads.** The Neuron runtime binds a process to the
cores visible to it, and `NEURON_RT_VISIBLE_CORES` is read once at
initialisation. Two threads in one process therefore share one core
allocation and would measure that core twice -- producing a number twice as
large as the truth with no error anywhere. One process per core is the only
arrangement where the aggregate means what it says.

This is also why these workloads decline the profiler's reserved core: they
declare ``cores: "all"``, and the orchestrator skips the reservation rather
than quietly aggregating over all-but-one core. The Score therefore comes
from the analytic figure, which for this workload is honest -- each worker
counts the bytes it moved, and the bytes are real whether or not a profiler
watched them.

STATUS: VERIFIED ON HARDWARE, trn1.2xlarge 2026-09-08 and 2026-09-10:
543.2 GB/s read and 505.8 GB/s write aggregated across cores.

The concurrency guard is doing real work at short durations. On the
2026-09-10 ten-second pass the workers overlapped for only 1.9s of the
10s span -- 19% -- and the row says so rather than reporting a summed
bandwidth that was never a measurement of cores contending.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import typing

from . import cores as core_planning
from . import nki_backend


# Long enough to cover a cold NEFF compile in every worker at once. The 8 GiB
# graph took about seven minutes on inf2, and workers compile concurrently
# while contending for the same host CPUs.
_WORKER_TIMEOUT = 1800


def worker_command(direction: str, core: int, problem: typing.Mapping,
                   duration: int, result_path: str) -> typing.List[str]:
    """The command one worker runs: this module, as a script, pinned to a core.

    Built as a function so a test can assert the pinning without launching
    anything -- the pin is the part that makes the aggregate meaningful.
    """
    return [
        sys.executable, "-m", "kernels.memory_agg",
        "--direction", direction,
        "--core", str(core),
        "--duration", str(duration),
        "--problem", json.dumps(dict(problem)),
        "--result", result_path,
    ]


def run(problem: typing.Mapping[str, typing.Any], duration: int,
        direction: str, core_count: int) -> dict:
    """Launch one worker per core and sum what they moved."""
    nki_backend.require_toolchain()

    if core_count < 1:
        raise ValueError(f"need at least one core, got {core_count}")

    workers = []
    results = []
    with tempfile.TemporaryDirectory(prefix="pantheon-agg-") as workdir:
        started = time.perf_counter()
        for core in range(core_count):
            result_path = os.path.join(workdir, f"core{core}.json")
            environment = dict(os.environ)
            # Each worker sees exactly one core, so the runtime cannot hand
            # two of them the same one.
            environment[core_planning.VISIBLE_CORES] = str(core)
            # Separate compiler workdirs: concurrent workers writing NEFFs
            # into one directory would race, and find_neff would then pick
            # another worker's graph.
            environment["PANTHEON_NEURON_WORKDIR"] = os.path.join(
                workdir, f"cc{core}"
            )
            workers.append((
                core,
                result_path,
                subprocess.Popen(
                    worker_command(direction, core, problem, duration,
                                   result_path),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                ),
            ))

        failures = []
        for core, result_path, process in workers:
            try:
                _, stderr = process.communicate(timeout=_WORKER_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()
                # Reap after killing. kill() only sends the signal; without
                # a second communicate() the child stays a zombie and its
                # stdout/stderr pipes stay open, and this loop still has
                # every other worker to wait on -- on an 8-core part that
                # is seven more processes holding descriptors for a run
                # that has already lost its result.
                try:
                    process.communicate(timeout=30)
                except subprocess.TimeoutExpired:  # pragma: no cover - refuses SIGKILL
                    pass
                failures.append(f"core {core} timed out")
                continue
            if process.returncode != 0:
                tail = (stderr or "").strip().splitlines()
                failures.append(
                    f"core {core} exited {process.returncode}: "
                    f"{tail[-1] if tail else 'no stderr'}"
                )
                continue
            try:
                with open(result_path, encoding="utf-8") as handle:
                    results.append(json.load(handle))
            except (OSError, json.JSONDecodeError) as error:
                failures.append(f"core {core} wrote no result: {error}")

        elapsed = time.perf_counter() - started

    return summarise(results, failures, elapsed, core_count, direction)


def concurrent_window(results: typing.Sequence[dict]) -> typing.Optional[float]:
    """Seconds during which every worker was inside its timed loop.

    Each worker brackets its whole subprocess -- compile included -- so the
    loop is the last ``elapsed_s`` of that window. The overlap is the
    intersection of those loops: latest start to earliest finish.

    Returns None when a worker did not report its timestamps, which is
    unknown rather than zero -- a result from before workers recorded them
    is not evidence that they ran sequentially, and treating it as such
    would flag every older aggregate.
    """
    windows = []
    for result in results:
        finished = result.get("finished_at")
        loop = result.get("elapsed_s")
        if not isinstance(finished, (int, float)) or not isinstance(
                loop, (int, float)):
            return None
        windows.append((finished - loop, finished))
    if not windows:
        return None
    latest_start = max(start for start, _ in windows)
    earliest_finish = min(finish for _, finish in windows)
    return max(0.0, earliest_finish - latest_start)


def _no_overlap_at_all(results: typing.Sequence[dict],
                       span: float) -> bool:
    """Whether the workers shared no window at all.

    Deliberately narrower than ``verify_workers_overlapped``: unknown is
    not zero, and a short overlap is not none. Only a measured zero
    invalidates the Score.
    """
    if len(results) < 2 or span <= 0:
        return False
    overlap = concurrent_window(results)
    return overlap is not None and overlap <= 0


def verify_workers_overlapped(results: typing.Sequence[dict], span: float,
                              floor: float = 0.5) -> typing.Optional[str]:
    """Flag an "aggregate" whose workers were not running together.

    The Score is summed bytes over the longest worker's span, and that
    arithmetic cannot distinguish two cores loading memory simultaneously
    from two cores doing it one after the other. Only the simultaneous case
    answers anything -- the whole question this workload asks is whether the
    cores *share* a path to memory, and cores that never overlap do not
    contend for it.

    The workers compile before they run and contend for host CPUs while
    doing it, so their loops can start seconds apart without anything being
    wrong. That is exactly why it needs measuring rather than assuming.
    """
    if len(results) < 2 or span <= 0:
        return None
    overlap = concurrent_window(results)
    if overlap is None:
        # The workers did not report when they ran. Unknown, not zero.
        return None
    if overlap <= 0:
        return (
            "workers reported no overlapping window, so this is not an "
            "aggregate -- the cores may have run one after another"
        )
    if overlap < span * floor:
        return (
            f"workers overlapped for only {overlap:.1f}s of a {span:.1f}s "
            f"span ({overlap / span:.0%}), so the summed bandwidth is not a "
            "measurement of the cores contending"
        )
    return None


def summarise(results: typing.Sequence[dict], failures: typing.Sequence[str],
              elapsed: float, core_count: int, direction: str) -> dict:
    """Combine the workers' figures into the aggregate the registry declares.

    The bytes sum across cores; the time does not. Each worker ran for the
    same wall-clock window concurrently, so dividing summed bytes by summed
    time would divide by ``core_count`` times too much and report roughly
    one core's bandwidth as the aggregate. The span used is the longest
    worker's, which is the window during which all that traffic moved.
    """
    key = "bytes_requested" if direction == "read" else "bytes_written"
    total_bytes = sum(int(result.get(key, 0)) for result in results)
    spans = [float(result.get("elapsed_s", 0.0)) for result in results]
    span = max(spans) if spans else 0.0
    overlap = concurrent_window(results)

    per_core = [
        {
            "core": result.get("core"),
            "gbps": result.get("analytic_gbps"),
            "bytes": result.get(key),
            "verified_ratio": result.get("read_verified_ratio")
            or result.get("write_verified_ratio"),
        }
        for result in results
    ]

    warning = None
    if failures:
        warning = "; ".join(failures)
    elif len(results) < core_count:
        warning = (
            f"only {len(results)} of {core_count} cores reported, so this is "
            "not an aggregate over the whole part"
        )
    else:
        warning = (verify_workers_overlapped(results, span)
                   or verify_cores_scaled(per_core))

    return {
        "cores": core_count,
        "cores_reporting": len(results),
        # Zero overlap is not a weak aggregate, it is not an aggregate.
        # The summed bandwidth of workers that never ran together is the
        # sum of independent single-core runs, which is a real number for
        # a different question -- and the workload's whole question is
        # whether the cores contend for a shared path to memory.
        #
        # llm_prefill, llm_decode and speculative_decode were published as
        # PASS with Scores while every one of them had produced a NaN: the
        # check fired, the message reached the row's Detail, and nothing
        # acted on it. This is that shape exactly, so it acts.
        #
        # Partial overlap stays a warning. A 19% overlap still measures
        # something; it just is not what the name says, and the row saying
        # so is the honest response. Only "no overlapping window" makes the
        # number mean nothing at all.
        "score_invalid": _no_overlap_at_all(results, span),
        "bytes_moved": total_bytes,
        "elapsed_s": elapsed,
        "worker_span_s": span,
        # The window in which every worker was inside its timed loop. An
        # aggregate is a claim about cores loading memory *at the same
        # time*, and summed bytes over the longest span cannot tell a
        # concurrent run from a sequential one.
        "concurrent_window_s": overlap,
        "analytic_gbps": total_bytes / span / 1e9 if span else 0.0,
        "per_core": per_core,
        "score_method": "workload",
        "analytic_basis": "summed bytes / longest worker span",
        "warning": warning,
    }


def verify_cores_scaled(per_core: typing.Sequence[typing.Mapping],
                        floor: float = 0.5) -> typing.Optional[str]:
    """Flag cores that did far less than their peers.

    A core reporting a fraction of the others usually means it never got a
    device rather than that it is slow, and averaging that into an aggregate
    understates the part while looking like a real measurement. Reported
    rather than failed: an uneven part is a finding, not an error.
    """
    rates = [core.get("gbps") or 0.0 for core in per_core]
    if len(rates) < 2:
        return None
    fastest = max(rates)
    if fastest <= 0:
        return "no core moved any bytes"
    slowest = min(rates)
    if slowest < fastest * floor:
        return (
            f"slowest core managed {slowest:.1f} GB/s against {fastest:.1f} -- "
            "the aggregate is uneven, so read it as a per-core problem rather "
            "than a bandwidth figure"
        )
    return None


def _worker_main(argv: typing.Optional[typing.Sequence[str]] = None) -> int:
    """One worker: run the single-core kernel and write its result out.

    Runs as ``python -m kernels.memory_agg`` in its own process with one
    core visible, which is the only way the per-core figures stay separate.
    """
    import argparse

    from . import memory_read, memory_write

    parser = argparse.ArgumentParser(prog="pantheon-neuron-agg-worker")
    parser.add_argument("--direction", required=True, choices=["read", "write"])
    parser.add_argument("--core", type=int, required=True)
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)

    problem = json.loads(args.problem)
    module = memory_read if args.direction == "read" else memory_write

    # Wall-clock, not monotonic: these have to be comparable across
    # processes, and the whole question is whether the workers were moving
    # bytes at the same time as each other.
    started_at = time.time()
    result = module.run(problem, args.duration)
    result["core"] = args.core
    # The kernel's own elapsed_s covers its timed loop; the window this
    # brackets also covers the compile, so the overlap is computed from the
    # loop's share of it rather than from the whole subprocess lifetime.
    result["finished_at"] = time.time()
    result["started_at"] = started_at

    # Only what is JSON-serialisable: the kernels return plans and warnings,
    # not tensors, but a future field that is not serialisable should fail
    # this worker rather than corrupt the aggregate.
    with open(args.result, "w", encoding="utf-8") as handle:
        json.dump(result, handle, default=str)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(_worker_main())
