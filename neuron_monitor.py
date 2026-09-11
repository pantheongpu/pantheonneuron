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


def period_string(seconds: float) -> str:
    """The period as neuron-monitor will honour it: whole seconds, at least 1.

    **It ignores anything else and falls back to its 5-second default.**
    Measured with this module's own config on inf2.xlarge 2026-09-11, the
    period each sample reported:

        "1.0s"   5.0 s   (3 samples in 16 s)    <- what this module sent
        "1s"     1.0 s   (13 samples in 16 s)
        "0.5s"   5.0 s
        "5.0s"   5.0 s
        "500ms"  no samples at all

    This built the string as f"{seconds}s", so the default of 1.0 went out
    as "1.0s" and every run since sampled five times more coarsely than it
    asked. The suite recorded the result as a property of the tool --
    "neuron-monitor floors around two seconds whatever --monitor-period
    asks for" -- while the 2026-08-26 schema probe, which wrote "1s", had
    been receiving 1.0 s periods all along.
    """
    return f"{max(1, round(seconds))}s"


def _longest_busy_block(counts: typing.Sequence[float]) -> typing.List[int]:
    """Indices of the longest uninterrupted run of nonzero readings.

    The index-level form of what whole_periods does with values, for a
    series whose readings are counts rather than rates.
    """
    best: typing.List[int] = []
    current: typing.List[int] = []
    for index, count in enumerate(counts):
        if count > 0:
            current.append(index)
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
    return best


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

    The rate is taken over the **interior of the longest uninterrupted run
    of busy periods**: the first and last periods with completions are only
    partly busy (6657 and 7465 above, against ~15,300 for a full one), and
    including them dilutes the rate by the idle part of each. Interior
    periods are wholly inside the execution, so their completions over
    their periods is the rate.

    Longest run rather than first-to-last busy, which is what this did
    until 2026-09-11. Setup work completes before the timed loop and a
    compile sits between them, so a run can read

        3 (setup), 0 x 46 (compile), 6657, 15409, 15273, 15199, 7465, 0

    and first-to-last hands the "first" slot to the setup blip, leaving
    the loop's partly busy 6657 inside the average. That is the same
    defect whole_periods was fixed for on the flops and utilisation
    series, where it moved memory_read's utilisation from 88.7% to 99.4%,
    and the two series now follow the same rule.

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

    block = _longest_busy_block(
        [count for _, count, _ in series])
    active = [i for i, (_, count, _) in enumerate(series) if count > 0]
    interior = [series[i] for i in block[1:-1]]
    usable = [(count, period) for _, count, period in interior
              if isinstance(period, (int, float)) and period > 0]
    summary["execution_active_periods"] = len(active)
    # The block the rate comes from, beside the total busy count: the two
    # differing is a run that paused, and says so rather than hiding it.
    summary["execution_block_periods"] = len(block)
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


def whole_periods(values: typing.Sequence[float]) -> typing.Tuple[list, list]:
    """(busy block, whole periods) of one core's per-period readings.

    The busy block is the **longest uninterrupted run of nonzero readings**;
    the whole periods are that block without its first and last, which the
    work only partly filled.

    Longest block rather than first-to-last nonzero, measured on
    trn1.2xlarge 2026-09-11: memory_read's core 0 read

        1.5 (setup), 0 x 46 (a 230 s compile), 35.3, 99.6, 99.6, 99.6,
        98.3, 100, 66.8 (the 30 s loop), 0, 0

    First-to-last put the compile inside the span: 11.55%. Dropping every
    zero and then the two ends let the setup blip take the "first" slot, so
    the loop's partly busy first period stayed in: 88.7%. The longest block
    is the loop, and its interior reads 99.4%.

    The trade-off, stated rather than hidden: a workload that paused for a
    whole sampling period mid-run would be measured over its longest
    uninterrupted stretch. This used to say none does because pulse_virus
    cycles every 2 s "inside the monitor's ~5 s period" -- a period that
    was 5 s only because the config was ignored. At the honoured 1 s its
    idle halves can fill a period, so the orchestrator samples it over
    whole cycles instead (pantheon_neuron.monitor_period_for).
    """
    best, current = [], []
    for value in values:
        if value > 0:
            current.append(value)
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
    return best, best[1:-1]


def whole_period_utilisation(values: typing.Sequence[float]) -> typing.Dict[str, typing.Any]:
    """neuroncore_utilization over the periods the core was busy throughout.

    Same rule as whole_period_flops. ``mean_all_samples`` is the old
    figure -- every sample, compile included -- kept beside it so the
    size of the difference stays visible.
    """
    summary: typing.Dict[str, typing.Any] = {
        "peak": round(max(values), 2) if values else 0.0,
        "mean_all_samples": round(statistics.fmean(values), 2) if values else 0.0,
    }
    _, whole = whole_periods(values)
    summary["samples"] = len(whole)
    summary["mean"] = round(statistics.fmean(whole), 2) if whole else 0.0
    return summary


def whole_period_flops(values: typing.Sequence[float]) -> typing.Dict[str, typing.Any]:
    """mean(effective_flops) over the periods the core was busy throughout.

    ``values`` are one core's nonzero readings in arrival order. Each is a
    rate over its sampling period, so the first and last busy periods --
    which the work only partly filled -- read low. Measured on
    trn1.2xlarge 2026-09-10, tensor_virus 8192^3 for 30 s, one sample per
    ~5 s:

        18.25, 72.34, 72.35, 71.02, 72.35, 72.57, 53.43 TFLOPS
        utilisation 25%, then 97-99.5%, then 73%

    Mean of all seven: 61.76. Mean of the five whole periods: 72.13,
    against the kernel's own analytic 71.93 from the same run. The Score
    was the first, low by as much as the edges happened to cover -- which
    varies with where the monitor's grid falls against the run, and is
    what the repeats' spreads were (65.05 to 71.94 on one run, rising).

    The same rule execution_rate applies to the completion tallies. The
    all-sample mean is kept beside it so the difference stays visible.
    """
    span, whole = whole_periods(values)
    summary: typing.Dict[str, typing.Any] = {
        "peak": int(max(values)) if values else 0,
        # The busy span, edges included: what the Score used to average.
        "samples_all": len(span),
        "mean_all_samples": int(statistics.fmean(span)) if span else 0,
    }
    # How many whole periods the mean is over.
    summary["samples"] = len(whole)
    if whole:
        summary["mean"] = int(statistics.fmean(whole))
    else:
        summary["absent"] = (
            f"the core was busy across {len(span)} sampling period(s) and "
            "none of them throughout, so no period measures its rate -- run "
            "longer")
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

        honoured = period_string(self.period_seconds)
        if float(honoured[:-1]) != self.period_seconds:
            self._warn_once(
                "period",
                f"neuron-monitor honours whole seconds only; sampling every "
                f"{honoured} rather than {self.period_seconds}s.",
            )

        binary = shutil.which("neuron-monitor")
        if binary is None:
            self._warn_once(
                "missing",
                "neuron-monitor not found; run will proceed without telemetry.",
            )
            return False

        config = json.dumps(
            {
                "period": period_string(self.period_seconds),
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
        periods: typing.List[float] = []

        for index, sample in enumerate(self._samples):
            # One reading per core per sample, the largest any runtime
            # reported. Each process is its own runtime and each reports
            # *every* core, the ones it does not drive at 0: on
            # trn1.2xlarge 2026-09-11 memory_read_agg's two workers sent
            # pid5349{0:99.6, 1:0} and pid5350{0:0, 1:99.4} in the same
            # sample. Appending both interleaved a busy core's readings
            # with zeros, so no uninterrupted busy block could form (0 and
            # 1 whole periods for two 30 s loops), and the old all-sample
            # mean halved.
            sample_util: typing.Dict[str, float] = {}
            sample_flops: typing.Dict[str, float] = {}
            sample_completed: typing.Optional[int] = None
            sample_period: typing.Optional[float] = None
            for runtime in (sample.get("neuron_runtime_data") or []):
                report = runtime.get("report") or {}
                period = (report.get("neuroncore_counters") or {}).get("period")
                if isinstance(period, (int, float)) and period > 0.5:
                    periods.append(float(period))
                cores = (report.get("neuroncore_counters") or {}).get(
                    "neuroncores_in_use"
                ) or {}
                for core_id, counters in cores.items():
                    value = counters.get("neuroncore_utilization")
                    if isinstance(value, (int, float)):
                        sample_util[str(core_id)] = max(
                            sample_util.get(str(core_id), 0.0), float(value))
                    # effective_flops is absent from the CloudWatch metric
                    # set and from sysfs (where flop_count stays 0), but
                    # neuron-monitor reports it per NeuronCore.
                    # Zeros kept: trimming happens in whole_periods, which
                    # needs to see an idle period *inside* the run to keep
                    # it -- dropping every zero here would read a run that
                    # paused as one that never did.
                    achieved = counters.get("effective_flops")
                    if isinstance(achieved, (int, float)):
                        sample_flops[str(core_id)] = max(
                            sample_flops.get(str(core_id), 0.0), float(achieved))

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
                    # One series entry per sample, summed over runtimes --
                    # the rule _completed_in uses and await_idle_period
                    # waits on. This appended one entry per runtime, so a
                    # second runtime on the device (a neuron-profile
                    # capture, an aggregate's worker) put two entries at
                    # the same instant: execution_rate then divided that
                    # sample's completions by two periods' worth of time.
                    sample_completed = (sample_completed or 0) + int(completed)
                    if isinstance(stats.get("period"), (int, float)):
                        sample_period = max(sample_period or 0.0,
                                            float(stats["period"]))
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

            if sample_completed is not None:
                completed_series.append((index, sample_completed, sample_period))
            for core_id, value in sample_util.items():
                utilisation[core_id].append(value)
            for core_id, value in sample_flops.items():
                flops[core_id].append(value)

            hw = (sample.get("system_data") or {}).get("neuron_hw_counters") or {}
            # max() is right only if these are totals since driver load --
            # and then a device's past events are charged to this run. If
            # they are per-period tallies, as execution_summary.completed
            # turned out to be, max() undercounts and the sum is right.
            # The AWS guide does not say, and no ECC event has been
            # observed on these parts to settle it the way completed was
            # settled (by watching it at a boundary). Recorded, not acted
            # on: nothing in the harness reads these to pass or fail a row.
            for device in (hw.get("neuron_devices") or []):
                for key in ecc:
                    value = device.get(key)
                    if isinstance(value, (int, float)):
                        ecc[key] = max(ecc[key], int(value))

        summary = {
            "samples": len(self._samples),
            # The period the monitor actually delivered, from the samples'
            # own `period` field -- the second quantity that would have
            # shown "1.0s" being ignored (requested 1, delivered 5).
            "sample_period_s": (round(statistics.median(periods), 3)
                                if periods else None),
            "execution_errors": errors,
            "total_executions": executions,
            # Over whole busy periods, like effective_flops. The mean was
            # over every sample -- the compile's zeros included -- so on the
            # 2026-09-11 full pass tensor_virus published 42.25% for a core
            # 97-99.5% busy in every whole period it ran, and
            # memory_read_agg 2.5%. That measured compile time.
            "neuroncore_utilization": {
                core_id: whole_period_utilisation(values)
                for core_id, values in sorted(utilisation.items())
            },
        }
        if memory_bytes:
            summary["device_memory_used_bytes"] = {
                "mean": int(statistics.fmean(memory_bytes)),
                "peak": max(memory_bytes),
            }
        # Only cores that retired a FLOP at all -- an idle core has no busy
        # periods to average, and the zeros are kept only so whole_periods
        # can see a pause inside a run.
        busy_flops = {core_id: values for core_id, values in flops.items()
                      if any(value > 0 for value in values)}
        if busy_flops:
            summary["effective_flops"] = {
                core_id: whole_period_flops(values)
                for core_id, values in sorted(busy_flops.items())
            }
            if not any("mean" in core for core in summary["effective_flops"].values()):
                summary["effective_flops_absent"] = next(
                    core["absent"] for core in summary["effective_flops"].values())
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
