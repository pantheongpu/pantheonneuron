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
import tempfile
import threading
import time
import typing


# Dropped from every sample before it can reach a report.  Keyed by the
# top-level neuron-monitor field name.
# Observed verbatim in neuron-monitor output on an inf2.xlarge running
# Neuron runtime 2.30.51.  The whole instance_info block is dropped, but
# each field is listed so a schema change cannot quietly reintroduce one.
_HOST_IDENTIFIER_FIELDS = frozenset(
    {
        "instance_info",
        "instance_id",
        "instance_name",
        "instance_type",
        "instance_region",
        "instance_availability_zone",
        "instance_availability_zone_id",
        "availability_zone",
        "region",
        "ami_id",
        "subnet_id",
        "hostname",
        "ip_address",
        "serial_number",
        # neuron-ls reports the launching command line, which carries
        # filesystem paths and therefore usernames.
        "command",
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


def execution_rate(series: typing.Sequence[typing.Tuple[int, int, float]],
                   ) -> typing.Dict[str, typing.Any]:
    """Executions per second from neuron-monitor's per-period completion counts.

    ``series`` is (sample index, completed, period) for every sample that
    carried ``execution_stats.execution_summary.completed``.

    **``completed`` is a count per period, not a running total.** Measured
    on trn1.2xlarge 2026-09-10 with graph_replay, the monitor sampling
    every ~5s through compile, 60,000 replays, and an idle tail:

        completed per period:  0 ... 0, 6657, 15409, 15273, 15199, 7465, 0, 0
        sum:                   60,003   (60,000 replays + 3 setup graphs)
        interior / period:     3082, 3055, 3040 per s   (loop's clock: 3057)

    It falls back to zero when the work stops, which a running total cannot
    do. This function used to take last minus first over the samples, which
    on per-period counts is the difference between two periods' tallies:
    zero on a run whose last sample is idle, and 1012 or 1506 or 729 on
    others, depending only on where the samples fell. The "about four
    replays per NEFF execution" graph_replay documented came from the same
    misreading -- one period's 14,737 set against the run's 60,000.

    The rate is taken over the **interior** of the active run: the first
    and last periods with completions are only partly busy (6657 and 7465
    above, against ~15,300 for a full one), and including them dilutes
    the rate by the idle part of each. Interior periods are wholly inside
    the execution, so their completions over their periods is the rate.

    When there is nothing to measure it says which nothing it is:

    - **No samples carried the counter** -- telemetry is missing or the
      counter does not exist on this part.
    - **No period completed anything** -- the samples are there and the
      device finished nothing. A failing workload, not a slow one.
    - **Too few whole periods** -- the device was busy across so few
      sampling periods that none was busy throughout. Run longer.
    """
    if not series:
        return {"execution_samples": 0,
                "execution_rate_absent": "no sample carried the completion counter"}

    total = sum(count for _, count, _ in series)
    summary: typing.Dict[str, typing.Any] = {
        "execution_samples": len(series),
        # The run's executions, which the old maximum over samples was not:
        # it reported one period's tally as the total.
        "executions_total": total,
    }
    if total <= 0:
        summary["execution_rate_absent"] = (
            f"no period across {len(series)} samples completed an execution, "
            "so the device completed nothing measurable")
        return summary

    active = [i for i, (_, count, _) in enumerate(series) if count > 0]
    interior = [series[i] for i in active[1:-1]]
    usable = [(count, period) for _, count, period in interior
              if isinstance(period, (int, float)) and period > 0]
    summary["execution_active_periods"] = len(active)
    summary["execution_samples_used"] = len(usable)
    if not usable:
        summary["execution_rate_absent"] = (
            f"the device was busy across {len(active)} sampling period(s) and "
            "none of them throughout, so no period measures the rate -- run "
            "longer")
        return summary

    span = sum(period for _, period in usable)
    # The rate's denominator: whole periods the device was executing in.
    summary["execution_span_s"] = round(span, 4)
    summary["executions_per_s"] = sum(count for count, _ in usable) / span
    return summary


def _completed_in(sample: dict) -> typing.Optional[int]:
    """The completion tally one sample carries, summed over runtimes."""
    total = None
    for runtime in (sample.get("neuron_runtime_data") or []):
        stats = (runtime.get("report") or {}).get("execution_stats") or {}
        count = (stats.get("execution_summary") or {}).get("completed")
        if isinstance(count, (int, float)):
            total = (total or 0) + int(count)
    return total


class NeuronMonitor:
    """Samples neuron-monitor in a background thread for the run's duration."""

    def __init__(self, period_seconds: float = _DEFAULT_PERIOD_SECONDS, mock: bool = False):
        self.period_seconds = period_seconds
        self.mock = mock or os.environ.get("PANTHEON_NEURON_MOCK") == "1"
        self._samples: typing.List[dict] = []
        # Arrival times, parallel to _samples. The monitor's own
        # stream carries no wall clock, and a rate needs one.
        self._sample_times: typing.List[float] = []
        self._process: typing.Optional[subprocess.Popen] = None
        self._thread: typing.Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._warned: typing.Set[str] = set()
        self._config_path: typing.Optional[str] = None
        self._stderr: typing.Optional[typing.IO[str]] = None

    # -- lifecycle ---------------------------------------------------------

    def available(self) -> bool:
        return self.mock or shutil.which("neuron-monitor") is not None

    def start(self, device_indices: typing.Sequence[int]) -> bool:
        """Begin sampling. Returns False when telemetry is unavailable."""
        self._samples = []
        # Cleared with the samples it indexes: a monitor reused across
        # workloads would otherwise carry the previous run's timestamps and
        # date this run's executions against them.
        self._sample_times = []
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

        # neuron-monitor takes a config FILE via -c/--config-file. There is no
        # --json-config flag: passing one makes it print usage to stdout and
        # exit, which the sample loop then reports as a malformed sample while
        # collecting nothing for the whole run.
        try:
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".json", prefix="pantheon-neuron-monitor-", delete=False
            )
            handle.write(config)
            handle.close()
            self._config_path = handle.name
        except OSError as error:
            self._warn_once("config", f"could not write neuron-monitor config: {error}")
            return False

        try:
            # stderr is captured, not discarded: it is the only place
            # neuron-monitor explains why it refused to start.
            self._stderr = tempfile.TemporaryFile("w+")
            self._process = subprocess.Popen(
                [binary, "-c", self._config_path],
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            self._warn_once("spawn", f"could not start neuron-monitor: {error}")
            return False

        # A bad invocation dies immediately; surface that now rather than
        # reporting "samples: 0" at the end of a five-minute workload.
        time.sleep(0.5)
        if self._process.poll() is not None:
            self._stderr.seek(0)
            why = (self._stderr.read() or "").strip().splitlines()
            self._warn_once(
                "earlyexit",
                "neuron-monitor exited immediately "
                f"({self._process.returncode}): {why[-1] if why else 'no stderr'}",
            )
            self._process = None
            return False

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def await_idle_period(self, timeout: float = 15.0,
                          poll: float = 0.25) -> bool:
        """Keep sampling until a period with no completions arrives.

        ``completed`` is a tally per period, and the period in which the
        work ended is only reported when that period closes. Stopping the
        monitor as soon as the workload returns loses it: on trn1.2xlarge
        2026-09-10 graph_replay ran 60,000 replays and the row's total
        read 45,709 -- which a reader would take as replays lost, the very
        misreading execution_rate exists to correct. One idle period after
        the work means every busy one has been reported.

        Returns whether it arrived. False on timeout, or when no sample
        after the call carries the counter at all.
        """
        if self.mock:
            return True
        seen = len(self._samples)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            fresh = self._samples[seen:]
            for sample in fresh:
                if _completed_in(sample) == 0:
                    return True
            time.sleep(poll)
        return False

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
        if self._config_path:
            try:
                os.unlink(self._config_path)
            except OSError:
                pass
            self._config_path = None
        if self._stderr is not None:
            try:
                self._stderr.close()
            except OSError:
                pass
            self._stderr = None
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
            self._sample_times.append(time.monotonic())

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
            self._sample_times.append(time.monotonic())

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
        flops = collections.defaultdict(list)
        memory_bytes: typing.List[int] = []
        latency_p50: typing.List[float] = []
        latency_p99: typing.List[float] = []
        errors = 0
        executions = 0
        ecc = {
            "mem_ecc_corrected": 0,
            "mem_ecc_uncorrected": 0,
            "sram_ecc_corrected": 0,
            "sram_ecc_uncorrected": 0,
        }

        completed_series: typing.List[typing.Tuple[int, int, typing.Any]] = []

        for index, sample in enumerate(self._samples):
            for runtime in (sample.get("neuron_runtime_data") or []):
                report = runtime.get("report") or {}
                cores = (report.get("neuroncore_counters") or {}).get(
                    "neuroncores_in_use"
                ) or {}
                for core_id, counters in cores.items():
                    value = counters.get("neuroncore_utilization")
                    if isinstance(value, (int, float)):
                        utilisation[str(core_id)].append(float(value))
                    # effective_flops is absent from the CloudWatch metric
                    # set and from sysfs (where flop_count stays 0), but
                    # neuron-monitor reports it per NeuronCore.
                    achieved = counters.get("effective_flops")
                    if isinstance(achieved, (int, float)) and achieved > 0:
                        flops[str(core_id)].append(float(achieved))

                used = (
                    (report.get("memory_used") or {})
                    .get("neuron_runtime_used_bytes") or {}
                ).get("device")
                if isinstance(used, (int, float)):
                    memory_bytes.append(int(used))

                stats = report.get("execution_stats") or {}
                for count in (stats.get("error_summary") or {}).values():
                    if isinstance(count, (int, float)):
                        errors += int(count)

                # execution_summary carries the failure modes that matter
                # for a stress run; anything that is not "completed" is a
                # defect signal.
                summary = stats.get("execution_summary", {})
                completed = summary.get("completed")
                if isinstance(completed, (int, float)):
                    # A count for this period, so the run's total is the
                    # sum -- see execution_rate for the measurement.
                    executions += int(completed)
                    completed_series.append(
                        (index, int(completed), stats.get("period")))
                for key in (
                    "completed_with_err",
                    "completed_with_num_err",
                    "failed_to_queue",
                    "incorrect_input",
                    "timed_out",
                ):
                    value = (summary or {}).get(key)
                    if isinstance(value, (int, float)):
                        errors += int(value)

                latency = (stats.get("latency_stats") or {}).get("device_latency") or {}
                for source, sink in (("p50", latency_p50), ("p99", latency_p99)):
                    value = latency.get(source)
                    if isinstance(value, (int, float)):
                        sink.append(float(value))

            hw = (sample.get("system_data") or {}).get("neuron_hw_counters") or {}
            for device in (hw.get("neuron_devices") or []):
                for key in ecc:
                    value = device.get(key)
                    if isinstance(value, (int, float)):
                        ecc[key] = max(ecc[key], int(value))

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
        if flops:
            summary["effective_flops"] = {
                core_id: {
                    "mean": int(statistics.fmean(values)),
                    "peak": int(max(values)),
                    # How many samples the mean is over. The declared
                    # formula is mean(effective_flops), and the monitor
                    # samples across the whole run -- compile included,
                    # where the counter reads nothing and the sample is
                    # dropped. A short run therefore averages a handful of
                    # samples, and one caught mid-ramp moves the mean a
                    # long way.
                    #
                    # Measured on trn1.2xlarge 2026-09-10 at DURATION=10:
                    # tensor_virus repeated 17.74, 26.14, 26.13 TFLOPS.
                    # The low one is not a slow run, it is a mean over
                    # fewer good samples.
                    "samples": len(values),
                }
                for core_id, values in sorted(flops.items())
            }
        if latency_p50:
            summary["device_latency_seconds"] = {
                "p50_mean": round(statistics.fmean(latency_p50), 6),
                "p99_peak": round(max(latency_p99), 6) if latency_p99 else None,
            }
        # graph_replay's declared Score. The 729.3 and 1174.8 graph-steps/s
        # of 2026-09-08, once read as compile time diluting a span, were
        # differences of per-period tallies -- see execution_rate.
        rate = execution_rate(completed_series)
        summary.update(rate)

        summary["ecc_events"] = ecc
        summary["ecc_events_total"] = sum(ecc.values())
        return summary
