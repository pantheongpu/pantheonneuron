"""A single sample cannot show that a Score is irreproducible.

memory_read's declared profiler Score read 256.17, 178.7 and 119.19 GB/s on
three runs of the same pinned problem. Nothing noticed for a day, because
every run reported one number and moved on. The cause was real and is fixed;
the reason it went unseen is that nothing ever ran a workload twice.

So `--repeat` runs each workload N times and the row carries the spread
beside the Score: `Score` is the median, which is what to quote, and
`Repeats` is the range and coefficient of variation, which is what says
whether to trust it.
"""

import statistics

import pytest

import pantheon_neuron
from kernels import registry
from neuron_device import NeuronDevice

MOCK = [NeuronDevice(i, "mock", "v2", 2, 32 * 1024**3, True) for i in range(2)]


def _workload(name="memory_read"):
    return next(w for w in registry.WORKLOADS if w.name == name)


# -- the spread summary ------------------------------------------------------

def test_spread_reports_what_the_repeats_did():
    spread = pantheon_neuron._spread([100.0, 110.0, 120.0], attempted=3)
    assert spread["attempted"] == 3
    assert spread["scored"] == 3
    assert spread["min"] == 100.0
    assert spread["max"] == 120.0
    assert spread["median"] == 110.0
    assert spread["cv"] == pytest.approx(statistics.stdev(
        [100.0, 110.0, 120.0]) / 110.0, rel=1e-3)


def test_spread_of_one_score_has_no_variation():
    """One sample says nothing about spread, so it must not claim to."""
    spread = pantheon_neuron._spread([42.0], attempted=1)
    assert spread["scored"] == 1
    assert "cv" not in spread and "stdev" not in spread
    assert spread["median"] == 42.0


def test_spread_with_no_scores_still_records_the_attempt():
    spread = pantheon_neuron._spread([], attempted=3)
    assert spread == {"attempted": 3, "scored": 0}


def test_a_zero_mean_does_not_divide_by_zero():
    spread = pantheon_neuron._spread([0.0, 0.0], attempted=2)
    assert spread["cv"] is None


# -- the instability verdict -------------------------------------------------

def test_the_memory_read_swing_would_have_been_flagged():
    """The three figures that started this, run through the check."""
    observed = [256.17, 178.7, 119.19]
    spread = pantheon_neuron._spread(observed, attempted=3)
    message = pantheon_neuron._unstable(spread)

    assert message is not None
    assert "119.19" in message and "256.17" in message
    assert spread["cv"] > 0.3


def test_tight_repeats_are_not_flagged():
    """memory_write read 226.50 and 226.77 across two parts -- that is fine."""
    spread = pantheon_neuron._spread([226.50, 226.77, 226.11], attempted=3)
    assert pantheon_neuron._unstable(spread) is None


def test_a_single_score_is_never_called_unstable():
    assert pantheon_neuron._unstable(
        pantheon_neuron._spread([42.0], attempted=1)) is None


@pytest.mark.parametrize("cv_target,flagged", [(0.05, False), (0.25, True)])
def test_the_threshold_is_where_it_is_documented(cv_target, flagged):
    mean = 100.0
    offset = cv_target * mean * (2 ** 0.5) / 2
    spread = pantheon_neuron._spread([mean - offset, mean + offset], attempted=2)
    assert (pantheon_neuron._unstable(spread) is not None) == flagged


# -- the row -----------------------------------------------------------------

def test_a_single_run_carries_no_repeat_summary(monkeypatch):
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    row = pantheon_neuron.run_workload(_workload(), MOCK, 1, 0.01)
    assert row["Repeats"] is None
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_repeats_run_the_workload_that_many_times(monkeypatch):
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    calls = []
    original = pantheon_neuron._measure_once

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pantheon_neuron, "_measure_once", counted)
    pantheon_neuron.run_workload(_workload(), MOCK, 1, 0.01, repeat=3)
    assert len(calls) == 3
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_every_row_has_the_same_shape(monkeypatch):
    """Skipped, single and repeated rows must all declare the same keys."""
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    inf2 = [NeuronDevice(i, "inf2", "v2", 2, 32 * 1024**3, False) for i in range(2)]

    single = pantheon_neuron.run_workload(_workload(), inf2, 1, 0.01)
    repeated = pantheon_neuron.run_workload(_workload(), inf2, 1, 0.01, repeat=2)
    skipped = pantheon_neuron.run_workload(
        _workload("transformer_train_step"), inf2, 1, 0.01, repeat=2)

    assert sorted(single) == sorted(repeated) == sorted(skipped)
    assert "Repeats" in single
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_a_skip_is_not_repeated(monkeypatch):
    """A skip is a property of the hardware; running it again says nothing."""
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    inf2 = [NeuronDevice(0, "inf2", "v2", 2, 32 * 1024**3, False)]
    row = pantheon_neuron.run_workload(
        _workload("transformer_train_step"), inf2, 1, 0.01, repeat=3)
    assert row["Status"] == "SKIPPED"
    assert row["Repeats"] is None
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_the_median_is_the_published_score(monkeypatch):
    """Not the last run, and not the best one."""
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    scores = iter([10.0, 100.0, 20.0])
    original = pantheon_neuron._measure_once

    def scripted(*args, **kwargs):
        row = original(*args, **kwargs)
        row["Score"] = next(scores)
        return row

    monkeypatch.setattr(pantheon_neuron, "_measure_once", scripted)
    row = pantheon_neuron.run_workload(_workload(), MOCK, 1, 0.01, repeat=3)

    assert row["Score"] == 20.0
    assert row["Repeats"]["min"] == 10.0 and row["Repeats"]["max"] == 100.0
    assert "repeats disagree" in row["Detail"]
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_one_failed_repeat_fails_the_row(monkeypatch):
    """A workload that works four times in five is not a workload that works."""
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    outcomes = iter(["PASS", "FAIL", "PASS"])
    original = pantheon_neuron._measure_once

    def scripted(*args, **kwargs):
        row = original(*args, **kwargs)
        status = next(outcomes)
        row["Status"] = status
        row["Detail"] = "exploded" if status == "FAIL" else ""
        return row

    monkeypatch.setattr(pantheon_neuron, "_measure_once", scripted)
    row = pantheon_neuron.run_workload(_workload(), MOCK, 1, 0.01, repeat=3)

    assert row["Status"] == "FAIL"
    assert "1 of 3 repeats failed" in row["Detail"]
    assert "exploded" in row["Detail"]
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_the_cli_accepts_repeat():
    args = pantheon_neuron.build_parser().parse_args(["--repeat", "5"])
    assert args.repeat == 5
    assert pantheon_neuron.build_parser().parse_args([]).repeat == 1


# -- drift is not scatter ----------------------------------------------------
#
# Repeats run in one process, so a workload that leaves device memory
# allocated makes every later repeat measure a fuller device. Measured on
# trn1.2xlarge 2026-09-10: allocation_fragmentation's three repeats spanned
# 0.91 to 2,511.80 allocation-events/s, ordered. Noise does not do that.

def test_monotonic_repeats_are_reported_as_drift():
    falling = pantheon_neuron._spread([2511.8, 100.0, 0.9088], attempted=3)
    assert falling["trend"] == "falling"
    assert "drift rather than scatter" in pantheon_neuron._unstable(falling)

    rising = pantheon_neuron._spread([10.0, 20.0, 30.0], attempted=3)
    assert rising["trend"] == "rising"


def test_scattered_repeats_are_not_called_drift():
    """The tensor_virus shape: one low reading among two that agree."""
    scattered = pantheon_neuron._spread([26.1, 17.7, 26.0], attempted=3)
    assert scattered["trend"] is None
    message = pantheon_neuron._unstable(scattered)
    assert message is not None and "drift" not in message


def test_two_repeats_cannot_show_a_trend():
    """Every pair is monotonic; it means nothing until there are three."""
    assert pantheon_neuron._spread([1.0, 2.0], attempted=2)["trend"] is None


# -- a thin monitor mean says so ---------------------------------------------

def test_a_mean_over_too_few_samples_is_flagged():
    """The declared formula is mean(effective_flops), and the monitor drops
    samples taken while the workload compiles. A short run averages a
    handful, and one caught mid-ramp moves it a long way.

    trn1.2xlarge 2026-09-10 at DURATION=10: tensor_virus repeated 17.74,
    26.14, 26.13 TFLOPS. The low one is a mean over fewer good samples.
    """
    thin = pantheon_neuron.thin_monitor_sample(
        {"effective_flops": {"0": {"mean": 2e13, "samples": 2}}})
    assert thin is not None and "2 sample(s)" in thin

    assert pantheon_neuron.thin_monitor_sample(
        {"effective_flops": {"0": {"mean": 2e13, "samples": 40}}}) is None


def test_the_worst_core_decides():
    """One core starved of samples makes the mean over cores unstable too."""
    mixed = pantheon_neuron.thin_monitor_sample({"effective_flops": {
        "0": {"mean": 2e13, "samples": 40},
        "1": {"mean": 2e13, "samples": 1},
    }})
    assert mixed is not None and "1 sample(s)" in mixed


def test_no_flops_reported_is_not_a_thin_sample():
    """Absent is a different thing from thin, and monitor_score handles it."""
    assert pantheon_neuron.thin_monitor_sample({}) is None
    assert pantheon_neuron.thin_monitor_sample({"effective_flops": {}}) is None


# -- the row must describe the run it publishes ------------------------------

def _scripted(monkeypatch, scores, provenances):
    values, provs = iter(scores), iter(provenances)
    original = pantheon_neuron._measure_once

    def scripted(*args, **kwargs):
        row = original(*args, **kwargs)
        row["Score"] = next(values)
        row["Measurement"] = next(provs)
        return row

    monkeypatch.setattr(pantheon_neuron, "_measure_once", scripted)


def test_measurement_belongs_to_the_repeat_that_set_the_score(monkeypatch):
    """The Score is the median; the Measurement used to be the last run.

    Two different executions in one row, with nothing saying so -- a reader
    checking which NEFF was captured, or at what coverage, would be reading
    provenance for a run whose number was discarded.
    """
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    _scripted(monkeypatch, [10.0, 20.0, 30.0],
              [{"from": 1}, {"from": 2}, {"from": 3}])

    row = pantheon_neuron.run_workload(_workload(), MOCK, 1, 0.01, repeat=3)
    assert row["Score"] == 20.0
    assert row["Measurement"] == {"from": 2}, "not the last repeat"
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_an_even_median_belongs_to_no_repeat(monkeypatch):
    """It is an average of two runs, so no single Measurement describes it.

    Reporting one anyway would attach provenance to an execution that did
    not produce the number, which is worse than none.
    """
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    _scripted(monkeypatch, [10.0, 20.0], [{"from": 1}, {"from": 2}])

    row = pantheon_neuron.run_workload(_workload(), MOCK, 1, 0.01, repeat=2)
    assert row["Score"] == 15.0
    assert row["Measurement"] is None
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_a_single_run_keeps_its_own_measurement(monkeypatch):
    """No repeats, no ambiguity -- the row is that one run."""
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    _scripted(monkeypatch, [42.0], [{"from": 1}])

    row = pantheon_neuron.run_workload(_workload(), MOCK, 1, 0.01)
    assert row["Measurement"] == {"from": 1}
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
