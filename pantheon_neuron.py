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
import sys
import time
import typing

import neuron_device
import neuron_monitor
from kernels import nki_backend, registry

try:
    import psutil
except ImportError:
    psutil = None


PANTHEON_NEURON_VERSION = "0.1.0"
DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")


# --- Reporting --------------------------------------------------------------

def get_system_snapshot(devices) -> dict:
    """Aggregate run context for the report.

    This repository is public and reports are committed to it, so the
    snapshot must never contain host identifiers -- no hostname, no IP, no
    EC2 instance ID, no availability zone.  ``tests/test_report_privacy.py``
    enforces this; if you add a field here, assume it will be published.
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

def run_workload(workload, devices, duration: int, monitor_period: float) -> dict:
    """Execute one workload and return its result row."""
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

    metrics = monitor.stop() if telemetry_started else {"samples": 0}
    if metrics.get("execution_errors", 0) > 0 and status == "PASS":
        status = "FAIL"
        detail = f"{metrics['execution_errors']} Neuron execution error(s)"

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
        "Problem": dict(workload.problem) if workload.problem else None,
        "Telemetry": metrics,
    }


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

    nki_backend.require_toolchain()
    raise NotImplementedError(
        f"Workload '{workload.name}' has no NKI implementation yet."
    )


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

    snapshot = get_system_snapshot(devices)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for workload in workloads:
        print(f"[PANTHEON-NEURON] -> {workload.name}")
        row = run_workload(workload, devices, args.duration, args.monitor_period)
        results.append(row)
        detail = f" ({row['Detail']})" if row.get("Detail") else ""
        print(f"[PANTHEON-NEURON]    {row['Status']}{detail}")

    if not args.no_report:
        path = write_report(snapshot, results, run_id)
        print(f"[PANTHEON-NEURON] report: {path}")

    return 1 if any(row["Status"] == "FAIL" for row in results) else 0


if __name__ == "__main__":
    sys.exit(main())
