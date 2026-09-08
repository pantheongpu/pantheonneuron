#!/usr/bin/env python3
"""Pantheon Neuron -- a stress and validation suite for AWS Neuron devices.

Targets Trainium (trn1, trn1n, trn2) and Inferentia2 (inf2), which share one
software stack.  Inf1 is out of scope: it uses the legacy neuron-cc
toolchain and torch-neuron on PyTorch 1.x.
"""

import argparse
import datetime
import json
import os
import platform
import statistics
import sys
import time
import typing

import neuron_device
import neuron_monitor
from kernels import (allocation_fragmentation, collectives, cores,
                     encoders, graph_replay, inference_mix, llm_inference,
                     memory_agg, memory_read, memory_write, nki_backend,
                     omni_virus, pcie_bandwidth, pulse_virus, registry,
                     tensor_virus, transformer_compute)

try:
    import psutil
except ImportError:
    psutil = None


PANTHEON_NEURON_VERSION = "0.1.0"
DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")


# --- Reporting --------------------------------------------------------------

def get_system_snapshot(devices) -> dict:
    """Aggregate run context for the report.

    Reports are written to ``database/``, which is gitignored -- they are
    not committed. The invariant does not rest on that: a report is the
    artifact that gets pasted into an issue, attached to a mail, or copied
    into a public write-up, and it is produced on a rented instance whose
    identifiers are somebody's account. Being one paste away from public is
    the same requirement as being public.

    So the snapshot must never contain host identifiers -- no hostname, no
    IP, no EC2 instance ID, no availability zone.
    ``tests/test_report_privacy.py`` enforces this; if you add a field here,
    assume it will be published.
    """
    snapshot = {
        "pantheon_neuron_version": PANTHEON_NEURON_VERSION,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "os_info": {
            "system": platform.system(),
            "release": platform.release(),
            "arch": platform.machine(),
        },
        "toolchain": nki_backend.probe(),
        "devices": [
            {
                "index": device.index,
                "arch": device.arch,
                "neuroncore_version": device.neuroncore_version,
                "neuroncores": device.neuroncores,
                "hbm_bytes": device.hbm_bytes,
                "supports_training": device.supports_training,
            }
            for device in devices
        ],
        "cpu_info": "psutil_missing",
        "ram_info": "psutil_missing",
    }

    if psutil:
        vm = psutil.virtual_memory()
        snapshot["cpu_info"] = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        }
        snapshot["ram_info"] = {"total_bytes": vm.total}

    return snapshot


def write_report(snapshot: dict, results: typing.List[dict], run_id: str) -> str:
    os.makedirs(DATABASE_DIR, exist_ok=True)
    payload = dict(snapshot)
    payload["run_id"] = run_id
    payload["completed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload["test_results"] = results

    target = os.path.join(DATABASE_DIR, f"pantheon_neuron_report_{run_id}.json")
    temporary = f"{target}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, target)
    return target


# --- Execution --------------------------------------------------------------

def reservation_cost(workloads) -> typing.Tuple[typing.List[str], typing.List[str]]:
    """What a selection costs the profiler: (aggregate names, workloads billed).

    The first list is the workloads that force the reservation off by
    declaring ``cores: "all"``. The second is the workloads that pay for it
    -- the ones whose registry entry names ``neuron-profile``, which are
    exactly the ones that will fall back to the analytic figure.

    Split out from ``reserve_profiler_core`` so the cost can be named in the
    message, asserted by a test, and rendered in the workload reference
    without re-deriving the rule in three places.
    """
    aggregate = [w.name for w in workloads
                 if (w.problem or {}).get("cores") == "all"]
    if not aggregate:
        return [], []
    billed = [
        w.name for w in workloads
        if w.name not in aggregate
        and w.score_source is not None
        and w.score_source.source == registry.PROFILER
    ]
    return aggregate, billed


def reserve_profiler_core(devices, workloads=()) -> typing.Optional[str]:
    """Keep one NeuronCore free so the profiler can replay a NEFF.

    Must run before the first workload, because the Neuron runtime reads
    ``NEURON_RT_VISIBLE_CORES`` when it initialises and ignores later
    changes. That is also why this is all-or-nothing for a run: the split
    cannot be renegotiated per workload from inside one process.

    Returns the reserved core, or None when nothing was reserved.

    An explicit setting from the caller wins: someone who pinned cores by
    hand is answering a question we should not overrule, and silently
    re-pinning their run would change what it measures.

    A workload that declares ``cores: "all"`` wins too, and for the same
    reason. ``memory_read_agg`` measures aggregate bandwidth across every
    NeuronCore; holding one back would still produce a number, and that
    number would quietly be the aggregate of all-but-one core under a name
    that says otherwise. A missing profiler Score announces itself in the
    row; a Score over the wrong core count does not.

    That rule has a consequence worth stating plainly, because it applies to
    the invocation the README puts first: ``--test all`` and ``--test
    memory`` both select an aggregate workload, so neither can reach the
    profiler for ``memory_read`` or ``memory_write``. The declared source is
    available only to a selection with no ``cores: "all"`` workload in it.
    The message below names which workloads are paying, rather than saying
    "these Scores" and leaving the reader to work out which.
    """
    if nki_backend.mock_mode():
        return None
    if os.environ.get(cores.VISIBLE_CORES):
        return None

    aggregate, billed = reservation_cost(workloads)
    if aggregate:
        print(
            f"[PANTHEON-NEURON] no core reserved: {aggregate[0]} measures all "
            "cores"
        )
        if billed:
            print(
                f"[PANTHEON-NEURON]   {', '.join(billed)} will report the "
                f"analytic fallback, not {registry.PROFILER}; run them in a "
                "selection with no cores:all workload to reach it"
            )
        return None

    total = sum(device.neuroncores for device in devices)
    plan = cores.split(total)
    if plan is None:
        return None

    os.environ[cores.VISIBLE_CORES] = plan["workload"]
    os.environ[cores.RESERVED_CORE] = plan["profiler"]
    print(
        f"[PANTHEON-NEURON] cores {plan['workload']} to the workload, "
        f"{plan['profiler']} reserved for neuron-profile"
    )
    return plan["profiler"]


def _measure_once(workload, devices, duration: int, monitor_period: float) -> dict:
    """One execution of one workload, scored. See ``run_workload``."""
    skip = workload.skip_reason(devices)
    if skip is not None:
        # A skipped row still declares its Unit and Problem, so a
        # cross-platform comparison shows an explicit gap for this workload
        # rather than silently dropping the row.
        return {
            "Test Name": workload.name,
            "Suite": workload.suite,
            "Status": "SKIPPED",
            "Detail": skip,
            "Duration (s)": 0.0,
            "Devices": [device.index for device in devices],
            "Score": None,
            "Unit": workload.unit,
            "Score Method": None,
            # Declared and empty, like Score above: a cross-platform
            # comparison reads an explicit gap, not a missing key.
            "Measurement": None,
            "Repeats": None,
            "Problem": dict(workload.problem) if workload.problem else None,
            "Telemetry": {"samples": 0},
        }

    monitor = neuron_monitor.NeuronMonitor(period_seconds=monitor_period)
    telemetry_started = monitor.start([device.index for device in devices])

    started = time.time()
    status, detail, score = "PASS", "", None
    try:
        score = _execute(workload, devices, duration)
    except nki_backend.BackendUnavailable as error:
        status, detail = "SKIPPED", str(error)
    except Exception as error:  # noqa: BLE001 - a failing workload is a result
        status, detail = "FAIL", f"{type(error).__name__}: {error}"
    elapsed = time.time() - started

    run = _LAST_RUN.get(workload.name)
    if run and run.get("warning") and status == "PASS":
        detail = run["warning"]
        if run.get("score_invalid"):
            # The kernel says its own output could not be verified, so the
            # throughput beside it is not a measurement of anything. The
            # 2026-09-08 full-coverage run reported llm_prefill, llm_decode
            # and speculative_decode as PASS with published Scores while
            # every one of them had produced a NaN: the check fired, the
            # message reached the row's Detail, and nothing acted on it.
            #
            # An unverifiable output is indistinguishable from a graph that
            # never ran, which is the definition of a failed workload.
            status = "FAIL"
            score = None

    metrics = monitor.stop() if telemetry_started else {"samples": 0}
    if metrics.get("execution_errors", 0) > 0 and status == "PASS":
        status = "FAIL"
        detail = f"{metrics['execution_errors']} Neuron execution error(s)"

    # The compute workloads declare neuron-monitor as their Score source, and
    # the monitor has only just stopped -- its counters do not exist while the
    # kernel is still running, so this cannot happen inside _execute. A
    # kernel-side figure, where one exists, stays as the fallback.
    if status == "PASS":
        declared = monitor_score(workload, metrics)
        if declared is not None:
            score = declared
            _LAST_RUN.setdefault(workload.name, {})["score_method"] = (
                registry.MONITOR
            )
        elif score is None and (_wants_monitor_score(workload)
                                or _wants_execution_rate(workload)):
            counter = ("effective_flops" if _wants_monitor_score(workload)
                       else "execution rate")
            detail = detail or (
                f"neuron-monitor reported no {counter}, so this run has "
                "no Score from its declared source"
            )

    # "Score" and "Unit" mirror the pantheongpu report schema exactly so a
    # cross-platform comparison can join on (Test Name, Unit). "Problem"
    # records the pinned shape/dtype, because a Score is only comparable if
    # both platforms ran the same problem.
    return {
        "Test Name": workload.name,
        "Suite": workload.suite,
        "Status": status,
        "Detail": detail,
        "Duration (s)": round(elapsed, 2),
        "Devices": [device.index for device in devices],
        "Score": round(score, 4) if isinstance(score, (int, float)) else None,
        "Unit": workload.unit,
        "Score Method": _score_method(workload, score),
        "Measurement": _provenance(workload),
        # Filled in by run_workload when more than one repeat ran; declared
        # here so every row has the same shape.
        "Repeats": None,
        "Problem": dict(workload.problem) if workload.problem else None,
        "Telemetry": metrics,
    }


def run_workload(workload, devices, duration: int, monitor_period: float,
                 repeat: int = 1) -> dict:
    """Execute one workload ``repeat`` times and return its result row.

    **A single sample is not a measurement, and this suite spent a day
    finding that out.** memory_read's declared profiler Score read 256.17,
    178.7 and 119.19 GB/s on three separate runs of the same pinned problem
    -- a 2x spread nobody would have seen, because every run reported one
    number and moved on. The cause was real and is fixed, but the reason it
    went unnoticed for so long is that nothing ever ran a workload twice.

    So the row now carries the spread alongside the Score. ``Score`` is the
    median of the successful repeats, which is what a reader should quote;
    ``Repeats`` records how many ran, the range, and the coefficient of
    variation, which is what tells them whether to trust it.

    Repeats are separate executions with separate telemetry, not one
    execution measured twice: a compile is amortised across them the way a
    real run amortises it, and a monitor-sourced Score is read per repeat
    from the counters that repeat produced.

    A failure in any repeat fails the row. A workload that works four times
    in five is not a workload that works.
    """
    rows = [_measure_once(workload, devices, duration, monitor_period)
            for _ in range(max(1, repeat))]

    row = rows[-1]
    if len(rows) == 1:
        return row

    # A skip is a property of the hardware, not of the run: repeating it
    # says nothing, so the first answer stands.
    if row["Status"] == "SKIPPED":
        return row

    failed = [r for r in rows if r["Status"] == "FAIL"]
    if failed:
        row = dict(failed[0])
        row["Detail"] = (
            f"{len(failed)} of {len(rows)} repeats failed: {failed[0]['Detail']}"
        )

    scores = [r["Score"] for r in rows
              if isinstance(r.get("Score"), (int, float))]
    row["Repeats"] = _spread(scores, len(rows))
    if scores and row["Status"] == "PASS":
        row["Score"] = round(statistics.median(scores), 4)
        unstable = _unstable(row["Repeats"])
        if unstable:
            row["Detail"] = "; ".join(filter(None, [row.get("Detail"), unstable]))
    return row


# Above this, repeats of the same pinned problem disagree enough that the
# median is not a summary of them. Chosen to be loud rather than strict:
# memory_read's three runs spanned 256.17 to 119.19 GB/s, a coefficient of
# variation near 0.4, and that is the kind of thing a row has to say out
# loud rather than average away.
UNSTABLE_CV = 0.10


def _spread(scores, attempted: int) -> dict:
    """What the repeats actually did, so a Score can be judged."""
    summary = {"attempted": attempted, "scored": len(scores)}
    if not scores:
        return summary
    summary["min"] = round(min(scores), 4)
    summary["max"] = round(max(scores), 4)
    summary["median"] = round(statistics.median(scores), 4)
    if len(scores) > 1:
        mean = statistics.fmean(scores)
        deviation = statistics.stdev(scores)
        summary["stdev"] = round(deviation, 4)
        # Relative, because these Scores span GB/s and tokens/s and a
        # absolute threshold would mean something different for each.
        summary["cv"] = round(deviation / mean, 4) if mean else None
    return summary


def _unstable(spread: typing.Mapping[str, typing.Any]) -> typing.Optional[str]:
    """Say so when repeats of one problem do not agree."""
    cv = spread.get("cv")
    if not isinstance(cv, (int, float)) or cv <= UNSTABLE_CV:
        return None
    return (
        f"repeats disagree: {spread['min']} to {spread['max']} "
        f"(cv {cv:.2f} over {spread['scored']} runs), so the median is a "
        "summary of unlike numbers rather than a measurement"
    )


# Workloads whose Score does not yet come from the source the registry
# declares. The row records the discrepancy rather than hiding it, so a
# report is never read as though the declared contract was honoured.
# Populated by kernels that report how their Score was obtained, so the
# report records the method actually used rather than the one declared.
_LAST_RUN: typing.Dict[str, dict] = {}


# The counter string exactly as the registry declares it. Matching on the
# counter rather than on the source is the point: graph_replay is also
# neuron-monitor-sourced, but its formula is delta(completed) / period and
# its unit is graph-steps/s. Gating on the source alone would apply the
# FLOPS arithmetic to it and publish a TFLOPS number wearing a
# graph-steps/s label -- precisely the unlike-quantity comparison
# NOT_COMPARABLE_WITH_GPU exists to prevent.
FLOPS_COUNTER = "neuroncore_counters.*.effective_flops"

# The other monitor-sourced formula. graph_replay declares
# delta(completed) / period in graph-steps/s, which is a rate over the
# execution counter rather than an average of a throughput counter -- the
# reason the flops gate matches on its counter and not on the source.
EXECUTIONS_COUNTER = "execution_stats.execution_summary.completed"


def _wants_execution_rate(workload) -> bool:
    """Is this workload scored from the monitor's execution counter?"""
    source = workload.score_source
    return bool(
        source
        and source.source == registry.MONITOR
        and EXECUTIONS_COUNTER in source.counters
    )


def _wants_monitor_score(workload) -> bool:
    """Is this workload scored from the monitor's effective_flops counter?"""
    source = workload.score_source
    return bool(
        source
        and source.source == registry.MONITOR
        and FLOPS_COUNTER in source.counters
    )


def monitor_score(workload, metrics: typing.Mapping[str, typing.Any]):
    """The Score its registry entry declares, read from monitor telemetry.

    The compute workloads declare ``mean(effective_flops) / 1e12``. That
    counter exists only in the neuron-monitor stream: it is absent from the
    CloudWatch metric set, and sysfs leaves ``flop_count`` at zero. So the
    figure has to be taken from the telemetry the run just collected rather
    than computed by the kernel.

    ``mean`` is over NeuronCores. A part reports one ``effective_flops``
    series per core, and a workload that saturates the device is running on
    all of them; summing would make a two-core part look twice as fast as
    the same silicon reported per core, while the mean keeps the number
    per-core and comparable across parts with different core counts.

    Returns None when the counter is absent, which is the honest answer for
    a mock run, a run with telemetry disabled, or a kernel that never
    reached the Tensor Engine. A fabricated Score would flow into a report
    and be compared against real GPU results.
    """
    if _wants_execution_rate(workload):
        # delta(completed) / period, computed by the monitor over the span it
        # actually observed. Absent when fewer than two samples carried the
        # counter, which is the honest answer for a run too short to measure
        # a rate over.
        rate = metrics.get("executions_per_s")
        return rate if isinstance(rate, (int, float)) else None

    if not _wants_monitor_score(workload):
        return None

    flops = metrics.get("effective_flops") or {}
    means = [
        core["mean"]
        for core in flops.values()
        if isinstance(core, dict) and isinstance(core.get("mean"), (int, float))
    ]
    if not means:
        return None
    return sum(means) / len(means) / 1e12


def _score_method(workload, score) -> typing.Optional[str]:
    """What actually produced this Score, which may not be what was declared.

    A kernel can fall back -- memory_read degrades to an analytic figure
    when the profiler is unavailable. Recording the fallback is the whole
    point: a provisional number must never read as though the declared
    contract held.
    """
    if score is None:
        return None
    run = _LAST_RUN.get(workload.name)
    if run and run.get("score_method"):
        method = run["score_method"]
        if method == "analytic":
            # Both halves of this label have to come from the workload. The
            # basis differs by kernel -- bytes for the bandwidth kernels,
            # FLOPs for the compute ones -- and so does the source that was
            # missed, which is neuron-profile for one and neuron-monitor for
            # the other. A fixed string would misdescribe whichever workload
            # it was not written for, and a provisional number wearing a
            # confident label is the failure this function exists to prevent.
            basis = run.get("analytic_basis") or "wall-clock arithmetic"
            return f"analytic ({basis}); declared source is {_declared(workload)}"
        return method
    return workload.score_source.source if workload.score_source else None


# What a kernel measured, beyond the Score itself, that a reader needs in
# order to judge the Score. A whitelist rather than "everything the kernel
# returned": these rows are published, and a kernel result also carries
# filesystem paths and plan dicts that have no business in a report.
#
# Added after the 2026-09-08 validation, where memory_read and memory_write
# finally scored from neuron-profile and the report could not say how hard
# the NEFF search had to look. `profiler_candidates_tried` is the number
# that says whether mtime ranking is still weak, and it was invisible.
_PROVENANCE_KEYS = (
    # Which graph the profiler actually read, and how sure we are it was
    # ours. A basename, never a path -- compiler workdirs carry usernames.
    "profiler_neff",
    "profiler_plan_coverage",
    "profiler_candidates_tried",
    "profiler_candidates_available",
    # The counters the declared formula divides, so a Score can be
    # recomputed from the report rather than trusted.
    "hbm_read_bytes",
    "hbm_write_bytes",
    "profiler_total_time_s",
    # The cross-check the profiler figure is meant to be compared against.
    "analytic_gbps",
    # A raw ops/s rate nobody can read, restated at a human scale.
    "quantized_tops",
    # kv_cache_churn: the bandwidth its update rate actually achieved,
    # which is what says whether the rate measured memory or dispatch.
    "cache_gbps",
    # MoE dispatch: slots per expert, which is what the arithmetic scales
    # with once routing is balanced.
    "capacity",
    # serving_mix: a request is many scheduler steps, so both rates are
    # wanted, and implied_tflops is what a request count cannot contradict.
    "scheduler_steps_per_s",
    "blocks_executed",
    "implied_tflops",
    "read_verified_ratio",
    "write_verified_ratio",
    "product_verified_ratio",
    # pcie_bandwidth: says whether the row predates the preallocated-buffer
    # fix, which is the difference between two incomparable methodologies.
    "buffers",
    "per_direction",
)


def _provenance(workload) -> typing.Optional[dict]:
    """The measured detail behind a Score, for the report row.

    A Score that cannot be recomputed or attributed is a number the reader
    has to take on faith, which is the thing this suite exists not to ask.
    """
    run = _LAST_RUN.get(workload.name)
    if not run:
        return None
    found = {key: run[key] for key in _PROVENANCE_KEYS
             if run.get(key) is not None}
    return found or None


def _declared(workload) -> str:
    """Name the Score source the registry declares, with its lead counter."""
    source = workload.score_source
    if source is None:
        return "unspecified"
    counter = source.counters[0] if source.counters else ""
    # Counter paths are namespaced in the registry ('neuroncore_counters.*.
    # effective_flops'); the leaf is what a reader recognises.
    leaf = counter.rsplit(".", 1)[-1]
    return f"{source.source} {leaf}".rstrip()


def _execute(workload, devices, duration: int) -> typing.Optional[float]:
    """Dispatch to the workload implementation and return its Score.

    The Score is in ``workload.unit`` and is what a cross-platform
    comparison actually reads. Real NKI kernels land here per workload;
    until then mock mode exercises the full orchestrator, telemetry and
    reporting path, and hardware runs fail loudly rather than reporting a
    meaningless PASS.

    Mock mode returns None, never a synthetic number -- a fabricated Score
    would flow into a report and be compared against real GPU results.
    """
    if workload.name == "baseline_metrics":
        time.sleep(min(duration, 2) if nki_backend.mock_mode() else duration)
        return None

    if nki_backend.mock_mode():
        time.sleep(min(duration, 2))
        return None

    if workload.name == "memory_read":
        return _execute_bandwidth(workload, duration, memory_read)

    if workload.name == "memory_write":
        return _execute_bandwidth(workload, duration, memory_write)

    if workload.name in ("tensor_virus", "int_virus"):
        # One kernel, two workloads: int_virus is the same GEMM over int8
        # operands, which the registry declares by dtype rather than by
        # naming a different kernel.
        #
        # Returns the analytic cross-check, not the declared Score: that one
        # is mean(effective_flops) and does not exist until the monitor
        # stops, so run_workload reads it and overrides this figure. Keeping
        # the analytic number here means a run whose telemetry came back
        # empty still reports what the kernel issued, labelled as analytic.
        result = tensor_virus.run(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["analytic_tflops"]

    if workload.name == "pulse_virus":
        # Same GEMM, switched on and off. Its analytic figure spans the idle
        # halves too, so it lines up with the monitor's average rather than
        # with tensor_virus's sustained number.
        result = pulse_virus.run(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["analytic_tflops"]

    # The workloads below report their own Score: no hardware counter
    # measures allocator behaviour, and the device's DMA counters cannot see
    # a host transfer. The registry declares both as INTERNAL, so what the
    # kernel returns is the Score itself rather than a cross-check.
    if workload.name == "allocation_fragmentation":
        result = allocation_fragmentation.run(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["allocation_events_per_s"]

    if workload.name == "pcie_bandwidth":
        result = pcie_bandwidth.run(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["analytic_gbps"]

    # The transformer family. Same building blocks, deliberately different
    # computations: prefill runs a whole prompt through every layer, decode
    # runs one token against a cache, and churn never runs the model at all.
    if workload.name == "llm_prefill":
        result = llm_inference.run_prefill(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["prompt_tokens_per_s"]

    if workload.name == "llm_decode":
        result = llm_inference.run_decode(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["tokens_per_s"]

    if workload.name == "kv_cache_churn":
        result = llm_inference.run_cache_churn(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["cache_updates_per_s"]

    if workload.name == "transformer_virus":
        result = transformer_compute.run_virus(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["analytic_tflops"]

    if workload.name == "transformer_train_step":
        result = transformer_compute.run_train_step(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["train_steps_per_s"]

    if workload.name == "fused_attention":
        result = inference_mix.run_fused_attention(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["attention_tiles_per_s"]

    if workload.name == "quantized_gemm":
        result = inference_mix.run_quantized_gemm(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["quantized_ops_per_s"]

    if workload.name == "moe_router":
        result = inference_mix.run_moe_router(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["routed_tokens_per_s"]

    if workload.name == "speculative_decode":
        result = inference_mix.run_speculative_decode(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["verified_tokens_per_s"]

    if workload.name == "serving_mix":
        result = inference_mix.run_serving_mix(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["requests_per_s"]

    if workload.name == "rag_embedding":
        result = encoders.run_rag_embedding(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["embedding_vectors_per_s"]

    if workload.name == "vision_encoder":
        result = encoders.run_vision_encoder(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["image_tiles_per_s"]

    if workload.name == "omni_virus":
        result = omni_virus.run(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["analytic_tflops"]

    # Aggregate bandwidth: one process per core, because the runtime binds a
    # process to its visible cores at initialisation and two threads would
    # share one allocation and measure the same core twice.
    if workload.name in ("memory_read_agg", "memory_write_agg"):
        direction = "read" if workload.name.startswith("memory_read") else "write"
        core_count = sum(device.neuroncores for device in devices)
        result = memory_agg.run(workload.problem, duration, direction, core_count)
        _LAST_RUN[workload.name] = result
        return result["analytic_gbps"]

    # Collectives come from AWS's own benchmark rather than a counter. Both
    # need two or more devices, so skip_reason keeps them off single-device
    # parts before execution reaches here.
    if workload.name == "all_reduce":
        result = collectives.run_all_reduce(workload.problem, devices)
        _LAST_RUN[workload.name] = result
        return result["busbw_gbps"]

    if workload.name == "p2p_thrasher":
        result = collectives.run_p2p(workload.problem, devices)
        _LAST_RUN[workload.name] = result
        return result["busbw_gbps"]

    if workload.name == "graph_replay":
        # Its declared Score is the monitor's execution rate; this figure
        # counts submissions instead, and run_workload prefers the counter.
        # The two disagreeing means the runtime accepted more replays than
        # the device finished, which is worth seeing rather than smoothing.
        result = graph_replay.run(workload.problem, duration)
        _LAST_RUN[workload.name] = result
        return result["graph_steps_per_s"]

    nki_backend.require_toolchain()
    raise NotImplementedError(
        f"Workload '{workload.name}' has no NKI implementation yet."
    )


# The workloads _execute can actually run. Kept beside the dispatch it
# describes so the two cannot drift: tests assert that everything absent from
# this set raises rather than reporting a silent PASS, and naming a specific
# workload there instead would quietly stop testing anything the day that
# workload got a kernel.
IMPLEMENTED = frozenset({"baseline_metrics", "memory_read", "memory_write",
                         "tensor_virus", "int_virus", "pulse_virus",
                         "allocation_fragmentation", "pcie_bandwidth",
                         "graph_replay", "llm_prefill", "llm_decode",
                         "kv_cache_churn", "transformer_virus",
                         "transformer_train_step", "fused_attention",
                         "quantized_gemm", "moe_router",
                         "speculative_decode", "serving_mix",
                         "rag_embedding", "vision_encoder", "omni_virus",
                         "memory_read_agg", "memory_write_agg",
                         "all_reduce", "p2p_thrasher"})


def _execute_bandwidth(workload, duration: int, module) -> float:
    """Run an HBM bandwidth kernel and return GB/s.

    Prefers the profiler figure, which is the source the registry declares
    and the only one that reflects traffic the hardware actually performed.
    Falls back to the analytic figure -- bytes moved over wall time -- when
    the profiler is unavailable, and records which was used so a
    provisional number is never read as the declared one.
    """
    result = module.run(workload.problem, duration)
    _LAST_RUN[workload.name] = result
    if result.get("profiler_gbps") is not None:
        return result["profiler_gbps"]
    return result["analytic_gbps"]


# --- CLI --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pantheon-neuron",
        description="Stress and validation suite for AWS Trainium and Inferentia.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {PANTHEON_NEURON_VERSION}"
    )
    parser.add_argument(
        "--test",
        default="all",
        help="Workload name, suite (baseline, core, memory, interconnect), or 'all'",
    )
    parser.add_argument(
        "--duration", type=int, default=30, help="Seconds per workload (default: 30)"
    )
    parser.add_argument(
        "--device", default="all", help="Comma-separated device indices or 'all'"
    )
    parser.add_argument(
        "--monitor-period",
        type=float,
        default=1.0,
        help="neuron-monitor sampling period in seconds",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force the CPU mock backend (same as PANTHEON_NEURON_MOCK=1)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run each workload this many times and report the spread "
             "(default: 1). A single sample cannot show that a Score is "
             "irreproducible, which is how a 2x swing in memory_read went "
             "unnoticed.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List workloads and exit"
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Do not write a report file"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.mock:
        os.environ["PANTHEON_NEURON_MOCK"] = "1"

    if args.list:
        for workload in registry.WORKLOADS:
            requires = ", ".join(sorted(workload.requires)) or "-"
            print(f"{workload.name:24} {workload.suite:13} requires: {requires}")
            print(f"{'':24} {workload.summary}")
        return 0

    # Resolve the target before touching hardware: asking for a workload
    # that has no Neuron equivalent should say so, not report missing
    # devices.
    try:
        workloads = registry.resolve(args.test)
    except KeyError as error:
        print(f"[PANTHEON-NEURON] {error}", file=sys.stderr)
        return 2

    try:
        discovered = neuron_device.discover()
        devices = neuron_device.select(discovered, args.device)
    except neuron_device.NeuronUnavailable as error:
        print(f"[PANTHEON-NEURON] {error}", file=sys.stderr)
        return 2

    arches = sorted({device.arch for device in devices})
    print(
        f"[PANTHEON-NEURON] {len(devices)} device(s) [{', '.join(arches)}], "
        f"{len(workloads)} workload(s)"
    )

    reserve_profiler_core(devices, workloads)

    snapshot = get_system_snapshot(devices)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for workload in workloads:
        print(f"[PANTHEON-NEURON] -> {workload.name}")
        row = run_workload(workload, devices, args.duration,
                           args.monitor_period, repeat=args.repeat)
        results.append(row)
        detail = f" ({row['Detail']})" if row.get("Detail") else ""
        print(f"[PANTHEON-NEURON]    {row['Status']}{detail}")
        spread = row.get("Repeats") or {}
        if spread.get("scored", 0) > 1:
            print(f"[PANTHEON-NEURON]    {spread['scored']} repeats: "
                  f"{spread['min']} to {spread['max']}, "
                  f"median {spread['median']}, cv {spread.get('cv')}")

    if not args.no_report:
        path = write_report(snapshot, results, run_id)
        print(f"[PANTHEON-NEURON] report: {path}")

    return 1 if any(row["Status"] == "FAIL" for row in results) else 0


if __name__ == "__main__":
    sys.exit(main())
