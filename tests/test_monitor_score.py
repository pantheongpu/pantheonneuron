"""The Score path for workloads that declare neuron-monitor as their source.

Five core workloads -- tensor_virus, int_virus, pulse_virus,
transformer_virus, omni_virus -- declare ``mean(effective_flops over whole busy periods) / 1e12``.
That counter lives only in the neuron-monitor stream, so unlike the
bandwidth kernels the Score cannot come from the kernel: it has to be read
out of telemetry after the monitor stops. These tests cover that arithmetic
and, more importantly, the cases where the counter is missing -- where the
only correct answer is no Score at all.
"""

import pantheon_neuron
import sourcecheck
from kernels import registry
from neuron_device import NeuronDevice


TRN1 = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]


def _workload(name):
    return next(w for w in registry.WORKLOADS if w.name == name)


def _telemetry(*core_means):
    return {
        "samples": 10,
        "effective_flops": {
            str(index): {"mean": int(value), "peak": int(value)}
            for index, value in enumerate(core_means)
        },
    }


# -- which workloads use this path -------------------------------------------

def test_every_compute_workload_declares_the_monitor_source():
    """If one of these stops declaring it, its Score silently goes missing."""
    for name in ("tensor_virus", "int_virus", "pulse_virus",
                 "transformer_virus", "omni_virus"):
        assert pantheon_neuron._wants_monitor_score(_workload(name)), name


def test_bandwidth_workloads_do_not_use_the_monitor_path():
    """memory_read's Score comes from neuron-profile; this must not shadow it."""
    for name in ("memory_read", "memory_write"):
        assert not pantheon_neuron._wants_monitor_score(_workload(name)), name
        assert pantheon_neuron.monitor_score(
            _workload(name), _telemetry(8e11)
        ) is None


def test_graph_replay_is_monitor_sourced_but_not_scored_in_flops():
    """It reads neuron-monitor too, but its formula and unit are different.

    graph_replay declares sum(completed) / sum(period) in graph-steps/s. A gate
    on the source alone would hand it the FLOPS arithmetic and publish a
    TFLOPS magnitude under a graph-steps/s label. effective_flops is
    non-zero during a graph replay, so this would fire on real hardware
    rather than staying a theoretical mistake.
    """
    graph_replay = _workload("graph_replay")

    assert graph_replay.score_source.source == registry.MONITOR
    assert graph_replay.unit == "graph-steps/s"
    assert not pantheon_neuron._wants_monitor_score(graph_replay)
    assert pantheon_neuron.monitor_score(graph_replay, _telemetry(2e12)) is None


def test_the_gate_follows_the_registry_rather_than_a_hardcoded_list():
    """Any workload declaring the flops counter is scored through this path.

    A new compute workload that declares effective_flops must not need a
    second edit here to be scored, and one that stops declaring it must not
    keep being scored from it.
    """
    for workload in registry.WORKLOADS:
        declares = (
            workload.score_source is not None
            and workload.score_source.source == registry.MONITOR
            and pantheon_neuron.FLOPS_COUNTER in workload.score_source.counters
        )
        assert pantheon_neuron._wants_monitor_score(workload) is declares, (
            workload.name
        )


# -- the arithmetic ----------------------------------------------------------

def test_score_is_flops_over_one_trillion():
    """875,590,474,948 FLOP/s was the observed inf2 probe figure: 0.8756 TFLOPS."""
    score = pantheon_neuron.monitor_score(
        _workload("tensor_virus"), _telemetry(875_590_474_948)
    )
    assert round(score, 4) == 0.8756


def test_mean_is_taken_across_cores_not_sum():
    """A two-core part running the same work per core is not twice as fast."""
    score = pantheon_neuron.monitor_score(
        _workload("tensor_virus"), _telemetry(1e12, 1e12)
    )
    assert score == 1.0


def test_cores_are_averaged_when_they_differ():
    score = pantheon_neuron.monitor_score(
        _workload("tensor_virus"), _telemetry(1e12, 3e12)
    )
    assert score == 2.0


# -- absent counters produce no Score, never a fabricated one ----------------

def test_no_score_when_the_counter_is_absent():
    """The mock path and a kernel that never reached the engine land here."""
    assert pantheon_neuron.monitor_score(
        _workload("tensor_virus"), {"samples": 12}
    ) is None


def test_no_score_when_telemetry_never_started():
    assert pantheon_neuron.monitor_score(
        _workload("tensor_virus"), {"samples": 0}
    ) is None


def test_no_score_when_the_counter_is_present_but_empty():
    assert pantheon_neuron.monitor_score(
        _workload("tensor_virus"), {"samples": 5, "effective_flops": {}}
    ) is None


def test_non_numeric_counter_values_are_ignored():
    """Rather than crashing the run or coercing junk into a number."""
    metrics = {"samples": 5, "effective_flops": {"0": {"mean": None}}}
    assert pantheon_neuron.monitor_score(_workload("tensor_virus"), metrics) is None


# -- end to end through run_workload -----------------------------------------

def test_mock_run_records_no_score(monkeypatch):
    """Mock mode exercises the whole path and must still produce no number."""
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    row = pantheon_neuron.run_workload(
        _workload("tensor_virus"), TRN1, duration=1, monitor_period=0.1
    )
    assert row["Score"] is None
    assert row["Unit"] == "TFLOPS"


def test_a_failing_workload_gets_no_score_from_telemetry(monkeypatch):
    """Telemetry keeps sampling through a failure; that is not a result.

    Without the status gate a workload that raised on the first pass would
    still be scored from whatever the monitor happened to catch, and a FAIL
    row would carry a number that looks like a measurement.
    """
    def explode(*_args, **_kwargs):
        raise RuntimeError("kernel raised")

    monkeypatch.setattr(pantheon_neuron, "_execute", explode)
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "start",
        lambda self, indices: True,
    )
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "stop",
        lambda self: _telemetry(2e12),
    )

    row = pantheon_neuron.run_workload(
        _workload("tensor_virus"), TRN1, duration=1, monitor_period=0.1
    )
    assert row["Status"] == "FAIL"
    assert row["Score"] is None


def test_a_passing_run_is_scored_from_telemetry(monkeypatch):
    monkeypatch.setattr(pantheon_neuron, "_execute", lambda *a, **k: None)
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "start",
        lambda self, indices: True,
    )
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "stop",
        lambda self: _telemetry(2e12, 2e12),
    )

    row = pantheon_neuron.run_workload(
        _workload("tensor_virus"), TRN1, duration=1, monitor_period=0.1
    )
    assert row["Status"] == "PASS"
    assert row["Score"] == 2.0
    assert row["Score Method"] == registry.MONITOR


def test_a_passing_run_without_the_counter_says_so(monkeypatch):
    """A PASS with no Score must explain itself, not leave an empty cell."""
    monkeypatch.setattr(pantheon_neuron, "_execute", lambda *a, **k: None)
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "start",
        lambda self, indices: True,
    )
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "stop",
        lambda self: {"samples": 8},
    )

    row = pantheon_neuron.run_workload(
        _workload("tensor_virus"), TRN1, duration=1, monitor_period=0.1
    )
    assert row["Status"] == "PASS"
    assert row["Score"] is None
    assert "effective_flops" in row["Detail"]


# -- the number the declared Score replaced ----------------------------------

def test_close_figures_say_nothing():
    """tensor_virus: 26.19 issued against 26.06 counted. Either would do."""
    assert pantheon_neuron.override_disagreement(
        26.19, 26.06, "effective_flops") is None


def test_a_factor_of_four_is_reported():
    """graph_replay: 3051.2 submitted against 729.3 completed.

    The dispatch source already called this "worth seeing rather than
    smoothing" and then nothing reported it -- the reader got one number
    and never learned the other existed.
    """
    message = pantheon_neuron.override_disagreement(
        3051.206, 729.3231, "the completion counter")
    assert message is not None
    assert "729.3" in message and "3051" in message
    assert "4.2" in message


def test_it_fires_in_both_directions():
    """Which figure is larger is not the question."""
    assert pantheon_neuron.override_disagreement(10.0, 100.0, "c") is not None
    assert pantheon_neuron.override_disagreement(100.0, 10.0, "c") is not None


def test_a_missing_figure_is_not_a_disagreement():
    """Absent is not zero. Nothing to compare is not a comparison."""
    for absent in (None, "n/a", True, False):
        assert pantheon_neuron.override_disagreement(absent, 5.0, "c") is None
        assert pantheon_neuron.override_disagreement(5.0, absent, "c") is None


def test_a_zero_figure_is_its_own_statement():
    """These two cases came from tensor_virus.verify_against_monitor, a
    function written to make exactly these checks and never called from
    anywhere. A check that exists and does not run.

    A zero is not "a large ratio" -- it is a different claim, and each
    side means something different. The kernel at zero issued nothing; the
    counter at zero saw nothing while the kernel claimed work, which is
    what an eliminated matmul looks like.
    """
    issued_nothing = pantheon_neuron.override_disagreement(0, 5.0, "c")
    assert issued_nothing is not None
    assert "issued no arithmetic" in issued_nothing

    saw_nothing = pantheon_neuron.override_disagreement(26.19, 0, "c")
    assert saw_nothing is not None
    assert "probably eliminated" in saw_nothing

    # Negative is nonsense from either side, and reads as the same
    # failure: there is no throughput there.
    assert pantheon_neuron.override_disagreement(-1.0, 5.0, "c") is not None
    assert pantheon_neuron.override_disagreement(5.0, -1.0, "c") is not None


def test_the_threshold_is_wide_on_purpose():
    """The two quantities are genuinely different; only a large gap says
    something a reader can act on."""
    assert pantheon_neuron.OVERRIDE_DISAGREEMENT >= 1.5
    just_under = pantheon_neuron.OVERRIDE_DISAGREEMENT - 0.01
    assert pantheon_neuron.override_disagreement(1.0, just_under, "c") is None
    just_over = pantheon_neuron.OVERRIDE_DISAGREEMENT + 0.01
    assert pantheon_neuron.override_disagreement(1.0, just_over, "c") is not None


# -- the five cases that used to test an unreachable function ----------------
#
# tensor_virus.verify_against_monitor had these and was never called. They
# are here because this function is.

def test_cross_check_accepts_agreement():
    assert pantheon_neuron.override_disagreement(100.0, 90.0, "c") is None


def test_cross_check_flags_an_idle_engine():
    """The signal that the matmuls were folded away."""
    message = pantheon_neuron.override_disagreement(100.0, 0.0, "c")
    assert message is not None
    assert "eliminated" in message


def test_cross_check_flags_order_of_magnitude_disagreement():
    message = pantheon_neuron.override_disagreement(100.0, 5.0, "c")
    assert message is not None
    assert "factor of 20" in message


def test_cross_check_is_quiet_when_the_monitor_said_nothing():
    """Absent telemetry is handled by monitor_score, not called divergence."""
    assert pantheon_neuron.override_disagreement(100.0, None, "c") is None


def test_cross_check_flags_zero_analytic_throughput():
    message = pantheon_neuron.override_disagreement(0.0, 10.0, "c")
    assert message is not None
    assert "issued no arithmetic" in message


# -- a rate divided by more time than the run took ---------------------------

def test_a_span_inside_the_window_says_nothing():
    """The control. The monitor brackets the workload, so its trimmed
    span should sit inside the kernel's own measured window."""
    assert pantheon_neuron.span_outran_the_kernel(19.8, 20.47) is None
    assert pantheon_neuron.span_outran_the_kernel(20.47, 20.47) is None


def test_a_span_that_outran_the_window_is_reported():
    """graph_replay, trn1.2xlarge 2026-09-10: a 24.99s span against a
    20.47s window, with execution_idle_fraction at 0.0 -- nothing was
    trimmed at either end. Every extra second is time no replay ran,
    divided into a count that had stopped growing.
    """
    message = pantheon_neuron.span_outran_the_kernel(24.9954, 20.4657)
    assert message is not None
    # 24.9954 rounds to 25.00, which is what a reader sees. The first
    # version of this asserted "24.99" -- the test expecting truncation
    # where the code rounds, which is the test being wrong about the code
    # rather than the other way round.
    assert "25.00s" in message and "20.47s" in message
    assert "1.22x" in message


def test_a_small_overhang_is_tolerated():
    """The two brackets differ by design; only a real gap is a finding."""
    assert pantheon_neuron.span_outran_the_kernel(21.0, 20.0) is None
    assert pantheon_neuron.span_outran_the_kernel(22.5, 20.0) is not None


def test_missing_figures_are_not_an_overhang():
    for absent in (None, 0, -1.0, "n/a"):
        assert pantheon_neuron.span_outran_the_kernel(absent, 20.0) is None
        assert pantheon_neuron.span_outran_the_kernel(20.0, absent) is None


def test_the_overhang_is_far_too_small_to_explain_the_coalescing_gap():
    """Which is why both are reported rather than one being offered as
    the cause of the other.

    graph_replay's two figures differ by about four. A denominator
    inflated by a fifth is exactly the kind of thing that gets proposed
    as the explanation for a discrepancy it cannot account for.
    """
    inflation = 24.9954 / 20.4657
    assert inflation < 1.3
    assert 60000 / 14737 > 3.5
    assert inflation * 1.5 < 60000 / 14737


# -- the thinness warning has to describe the published counter --------------

def _graph_replay():
    return next(w for w in registry.WORKLOADS if w.name == "graph_replay")


def _tensor_virus():
    return next(w for w in registry.WORKLOADS if w.name == "tensor_virus")


def test_a_flops_scored_workload_is_told_about_effective_flops():
    metrics = {"effective_flops": {"0": {"samples": 1, "mean": 1.0}}}
    message = pantheon_neuron.thin_monitor_sample(metrics, _tensor_virus())
    assert message is not None
    assert "effective_flops" in message


def test_graph_replay_is_told_about_the_counter_it_publishes():
    """It was told about effective_flops, which it does not publish.

    trn1.2xlarge 2026-09-10: its row carried "effective_flops averaged
    over 3 sample(s)" while its Score came from the completion counter,
    whose own thinness went unreported. A warning naming the wrong
    quantity is worse than none -- it invites a reader to discount the
    Score for a reason that has nothing to do with it.
    """
    metrics = {"effective_flops": {"0": {"samples": 3, "mean": 1.0}},
               "execution_samples_used": 1}
    message = pantheon_neuron.thin_monitor_sample(metrics, _graph_replay())
    assert message is not None
    assert "completion rate" in message
    assert "effective_flops" not in message
    assert "1 whole sampling period(s)" in message


def test_graph_replay_with_enough_counter_samples_is_quiet():
    """Even when effective_flops is thin -- which is not its Score."""
    metrics = {"effective_flops": {"0": {"samples": 1, "mean": 1.0}},
               "execution_samples_used": 3}
    assert pantheon_neuron.thin_monitor_sample(
        metrics, _graph_replay()) is None


def test_the_flops_path_still_applies_without_a_workload():
    """The old signature keeps working, which is what every caller but
    the orchestrator uses."""
    metrics = {"effective_flops": {"0": {"samples": 1, "mean": 1.0}}}
    assert "effective_flops" in pantheon_neuron.thin_monitor_sample(metrics)


def test_a_missing_counter_sample_count_is_not_thinness():
    assert pantheon_neuron.thin_monitor_sample(
        {"execution_samples_used": None}, _graph_replay()) is None
    assert pantheon_neuron.thin_monitor_sample({}, _graph_replay()) is None


def test_the_thinness_advice_is_run_longer_and_blames_nothing_false():
    """It said neuron-monitor "floors around 2s whatever --monitor-period
    asks for". That was a malformed period string ("1.0s", ignored in
    favour of the 5 s default), not the tool -- "1s" delivers 1 s periods
    (inf2.xlarge 2026-09-11)."""
    flops = pantheon_neuron.thin_monitor_sample(
        {"effective_flops": {"0": {"samples": 1, "mean": 1.0}}})
    counter = pantheon_neuron.thin_monitor_sample(
        {"execution_samples_used": 1}, _graph_replay())
    for message in (flops, counter):
        assert message is not None
        assert "run longer" in message
        assert "floors around" not in message


def test_the_span_overhang_is_only_reported_for_a_rate():
    """A mean has no denominator for an inflated span to inflate.

    span_outran_the_kernel fired on pulse_virus on trn1.2xlarge
    2026-09-10 -- "the declared rate is divided by 4.50x the time the
    workload actually ran" -- and pulse_virus is scored from
    mean(effective_flops). Nothing divides a mean by the span, so the
    caveat described arithmetic the row never performed.

    Same defect as thin_monitor_sample two commits earlier, gated the
    same way: a row must describe the counter it publishes.
    """
    code = sourcecheck.flat_function_code(pantheon_neuron._measure_started)
    marker = code.index("span_outran_the_kernel")
    preceding = code[:marker]
    assert "_wants_execution_rate ( workload )" in preceding, (
        "the span check is not gated on the Score being a rate")


def test_the_gate_admits_graph_replay_and_excludes_the_compute_family():
    """graph_replay's Score is sum(completed)/sum(period) -- a rate over
    exactly the span this measures, so the overhang is real there."""
    assert pantheon_neuron._wants_execution_rate(_graph_replay())
    for name in ("tensor_virus", "pulse_virus", "omni_virus",
                 "int_virus", "transformer_virus"):
        workload = next(w for w in registry.WORKLOADS if w.name == name)
        assert not pantheon_neuron._wants_execution_rate(workload), name



def test_the_rate_threshold_rests_on_the_measured_periods():
    """Each whole period is a tally, not a reading: 3082, 3055 and 3040 per
    second against the loop's 3057 on trn1.2xlarge 2026-09-10."""
    periods, loop = (3082.1, 3054.6, 3040.0), 3057.1
    assert max(abs(p - loop) / loop for p in periods) < 0.015
    assert pantheon_neuron.MIN_RATE_PERIODS == 2
    assert pantheon_neuron.thin_monitor_sample(
        {"execution_samples_used": 2}, _graph_replay()) is None


# -- a pulsed workload is sampled over whole cycles ------------------------

def _pulse():

    return next(w for w in registry.WORKLOADS if w.name == "pulse_virus")


def test_a_pulsed_workload_is_sampled_over_whole_cycles():
    """At 1 s a sample covers half of pulse_virus's 2 s cycle, and one that
    falls inside an idle half reads zero and splits the busy block."""
    assert pantheon_neuron.monitor_period_for(_pulse(), 1.0) == 2.0


def test_the_whole_cycle_period_is_at_least_the_request():
    assert pantheon_neuron.monitor_period_for(_pulse(), 2.0) == 2.0
    assert pantheon_neuron.monitor_period_for(_pulse(), 3.0) == 4.0
    assert pantheon_neuron.monitor_period_for(_pulse(), 5.0) == 6.0


def test_a_steady_workload_keeps_the_request():

    steady = next(w for w in registry.WORKLOADS if w.name == "tensor_virus")
    assert pantheon_neuron.monitor_period_for(steady, 1.0) == 1.0


def test_every_pinned_pulse_period_is_whole_seconds():
    """neuron-monitor takes whole seconds; a fractional pulse would get no
    whole-cycle period and fall back to half-cycle samples."""

    for workload in registry.WORKLOADS:
        pulse = (workload.problem or {}).get("period_s")
        if pulse is not None:
            assert pulse == int(pulse), workload.name


# -- the figure the declared Score replaced ---------------------------------

def _row_with_monitor_score(monkeypatch, kernel_figure, monitor_flops):
    """Run one workload whose kernel counted `kernel_figure` while the
    monitor reports `monitor_flops`, and return the row."""
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    monkeypatch.setattr(pantheon_neuron, "_execute", lambda *a: kernel_figure)
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor, "stop",
        lambda self: {"samples": 4, "effective_flops": {
            "0": {"samples": 3, "mean": monitor_flops}}})
    workload = next(w for w in registry.WORKLOADS if w.name == "tensor_virus")
    row = pantheon_neuron.run_workload(workload, TRN1, 0.02, 0.01)
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    return row


def test_the_row_keeps_the_figure_the_monitor_replaced(monkeypatch):
    """The pair is the second quantity for a monitor Score. Reading it
    across a pass is what showed pulse_virus was sampled wrong -- 0.963
    where every other compute row sat within 0.2% of its kernel, and
    0.963 is far below the 1.5x that warns."""
    row = _row_with_monitor_score(monkeypatch, 40.0, 38.0e12)
    assert row["Score"] == 38.0
    assert row["Measurement"]["kernel_figure"] == 40.0
    assert row["Measurement"]["declared_over_kernel"] == 0.95


def test_agreement_is_recorded_too_not_just_disagreement(monkeypatch):
    row = _row_with_monitor_score(monkeypatch, 77.74, 77.59e12)
    assert row["Measurement"]["declared_over_kernel"] == 0.9981
    assert not row["Detail"], "agreement is recorded, not complained about"


def test_a_kernel_with_no_figure_of_its_own_records_no_ratio(monkeypatch):
    """A workload whose _execute returns None has nothing to divide -- and
    must not inherit the ratio of the run before it, which _LAST_RUN keeps
    until a kernel replaces it."""
    row = _row_with_monitor_score(monkeypatch, None, 50.0e12)
    assert row["Score"] == 50.0
    assert "declared_over_kernel" not in (row["Measurement"] or {})
