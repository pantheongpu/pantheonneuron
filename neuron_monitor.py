"""Telemetry collection for Neuron devices.

Wraps the ``neuron-monitor`` binary, which streams one JSON object per
sampling period on stdout.  This is a cleaner source than scraping
``nvidia-smi`` text, but it carries a hazard the GPU suite also had: the
payload includes an ``instance_info`` block with the EC2 instance ID,
availability zone and region.

Reports from this suite are published publicly, so ``instance_info`` and
every other host identifier is dropped at ingest -- never at write time.
Stripping late is how identifiers end up in a report.
"""

import collections
import json
import os
import shutil
import statistics
import subprocess
import threading
import typing


# Dropped from every sample before it can reach a report.  Keyed by the
# top-level neuron-monitor field name.
_HOST_IDENTIFIER_FIELDS = frozenset(
    {
        "instance_info",
        "instance_id",
        "instance_type",
        "availability_zone",
        "region",
        "ami_id",
        "subnet_id",
        "hostname",
        "ip_address",
    }
)

_DEFAULT_PERIOD_SECONDS = 1.0


def _scrub(sample: dict) -> dict:
    """Remove host identifiers from one neuron-monitor sample, recursively."""
    if not isinstance(sample, dict):
        return sample
    clean = {}
    for key, value in sample.items():
        if key in _HOST_IDENTIFIER_FIELDS:
            continue
        if isinstance(value, dict):
            clean[key] = _scrub(value)
        elif isinstance(value, list):
            clean[key] = [_scrub(item) for item in value]
        else:
            clean[key] = value
    return clean


class NeuronMonitor:
    """Samples neuron-monitor in a background thread for the run's duration."""

    def __init__(self, period_seconds: float = _DEFAULT_PERIOD_SECONDS, mock: bool = False):
        self.period_seconds = period_seconds
        self.mock = mock or os.environ.get("PANTHEON_NEURON_MOCK") == "1"
        self._samples: typing.List[dict] = []
        self._process: typing.Optional[subprocess.Popen] = None
        self._thread: typing.Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._warned: typing.Set[str] = set()

    # -- lifecycle ---------------------------------------------------------

    def available(self) -> bool:
        return self.mock or shutil.which("neuron-monitor") is not None

    def start(self, device_indices: typing.Sequence[int]) -> bool:
        """Begin sampling. Returns False when telemetry is unavailable."""
        self._samples = []
        self._stop.clear()
        self._device_indices = list(device_indices)

        if self.mock:
            self._thread = threading.Thread(target=self._mock_loop, daemon=True)
            self._thread.start()
            return True

        binary = shutil.which("neuron-monitor")
        if binary is None:
            self._warn_once(
                "missing",
                "neuron-monitor not found; run will proceed without telemetry.",
            )
            return False

        config = json.dumps(
            {
                "period": f"{self.period_seconds}s",
                "neuron_runtimes": [
                    {
                        "tag_filter": ".*",
                        "metrics": [
                            {"type": "neuroncore_counters"},
                            {"type": "memory_used"},
                            {"type": "execution_stats"},
                        ],
                    }
                ],
                "system_metrics": [{"type": "neuron_hw_counters"}],
            }
        )

        try:
            self._process = subprocess.Popen(
                [binary, "--json-config", config],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            self._warn_once("spawn", f"could not start neuron-monitor: {error}")
            return False

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> dict:
        """Stop sampling and return the aggregated, scrubbed metrics."""
        self._stop.set()
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        return self.aggregate()

    # -- sampling ----------------------------------------------------------

    def _loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                self._warn_once("parse", "skipped malformed neuron-monitor sample")
                continue
            self._samples.append(_scrub(sample))

    def _mock_loop(self) -> None:
        """Synthesise plausible samples so CI exercises the same code path."""
        tick = 0
        while not self._stop.wait(min(self.period_seconds, 0.05)):
            tick += 1
            self._samples.append(
                _scrub(
                    {
                        "instance_info": {"instance_id": "i-fffffffffffffffff"},
                        "neuron_runtime_data": [
                            {
                                "report": {
                                    "neuroncore_counters": {
                                        "neuroncores_in_use": {
                                            str(index): {
                                                "neuroncore_utilization": 90.0
                                            }
                                            for index in self._device_indices
                                        }
                                    },
                                    "memory_used": {
                                        "neuron_runtime_used_bytes": {
                                            "device": 8 * 1024**3
                                        }
                                    },
                                    "execution_stats": {
                                        "error_summary": {"generic": 0},
                                        "total_executions": tick * 100,
                                    },
                                }
                            }
                        ],
                    }
                )
            )

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(f"[PANTHEON-NEURON] Warning: {message}")

    # -- aggregation -------------------------------------------------------

    def aggregate(self) -> dict:
        """Reduce raw samples to the summary a report records."""
        if not self._samples:
            return {"samples": 0}

        utilisation = collections.defaultdict(list)
        memory_bytes: typing.List[int] = []
        errors = 0
        executions = 0

        for sample in self._samples:
            for runtime in sample.get("neuron_runtime_data", []):
                report = runtime.get("report", {})
                cores = report.get("neuroncore_counters", {}).get(
                    "neuroncores_in_use", {}
                )
                for core_id, counters in cores.items():
                    value = counters.get("neuroncore_utilization")
                    if isinstance(value, (int, float)):
                        utilisation[str(core_id)].append(float(value))

                used = (
                    report.get("memory_used", {})
                    .get("neuron_runtime_used_bytes", {})
                    .get("device")
                )
                if isinstance(used, (int, float)):
                    memory_bytes.append(int(used))

                stats = report.get("execution_stats", {})
                for count in stats.get("error_summary", {}).values():
                    if isinstance(count, (int, float)):
                        errors += int(count)
                total = stats.get("total_executions")
                if isinstance(total, (int, float)):
                    executions = max(executions, int(total))

        summary = {
            "samples": len(self._samples),
            "execution_errors": errors,
            "total_executions": executions,
            "neuroncore_utilization": {
                core_id: {
                    "mean": round(statistics.fmean(values), 2),
                    "peak": round(max(values), 2),
                }
                for core_id, values in sorted(utilisation.items())
            },
        }
        if memory_bytes:
            summary["device_memory_used_bytes"] = {
                "mean": int(statistics.fmean(memory_bytes)),
                "peak": max(memory_bytes),
            }
        return summary
