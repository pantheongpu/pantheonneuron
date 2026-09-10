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


# Mock mode sleeps the requested duration, so every `1` below was a real
# second per repeat -- six tests at three repeats came to seventeen of the
# suite's fifty-nine seconds. None of them is testing the duration; they
# test how repeats are summarised, which holds at any duration.
DURATION = 0.02


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
    row = pantheon_neuron.run_workload(_workload(), MOCK, DURATION, 0.01)
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
    pantheon_neuron.run_workload(_workload(), MOCK, DURATION, 0.01, repeat=3)
    assert len(calls) == 3
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_every_row_has_the_same_shape(monkeypatch):
    """Skipped, single and repeated rows must all declare the same keys."""
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    inf2 = [NeuronDevice(i, "inf2", "v2", 2, 32 * 1024**3, False) for i in range(2)]

    single = pantheon_neuron.run_workload(_workload(), inf2, DURATION, 0.01)
    repeated = pantheon_neuron.run_workload(_workload(), inf2, DURATION, 0.01, repeat=2)
    skipped = pantheon_neuron.run_workload(
        _workload("transformer_train_step"), inf2, DURATION, 0.01, repeat=2)

    assert sorted(single) == sorted(repeated) == sorted(skipped)
    assert "Repeats" in single
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_a_skip_is_not_repeated(monkeypatch):
    """A skip is a property of the hardware; running it again says nothing."""
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    inf2 = [NeuronDevice(0, "inf2", "v2", 2, 32 * 1024**3, False)]
    row = pantheon_neuron.run_workload(
        _workload("transformer_train_step"), inf2, DURATION, 0.01, repeat=3)
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
    row = pantheon_neuron.run_workload(_workload(), MOCK, DURATION, 0.01, repeat=3)

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
    row = pantheon_neuron.run_workload(_workload(), MOCK, DURATION, 0.01, repeat=3)

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

    row = pantheon_neuron.run_workload(_workload(), MOCK, DURATION, 0.01, repeat=3)
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

    row = pantheon_neuron.run_workload(_workload(), MOCK, DURATION, 0.01, repeat=2)
    assert row["Score"] == 15.0
    assert row["Measurement"] is None
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def test_a_single_run_keeps_its_own_measurement(monkeypatch):
    """No repeats, no ambiguity -- the row is that one run."""
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    _scripted(monkeypatch, [42.0], [{"from": 1}])

    row = pantheon_neuron.run_workload(_workload(), MOCK, DURATION, 0.01)
    assert row["Measurement"] == {"from": 1}
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


# -- agreeing more closely than the Score can resolve ------------------------

def test_a_cv_above_the_resolution_is_real_agreement():
    """The control. Most Scores are continuous and this never fires."""
    assert pantheon_neuron.quantised_agreement({"cv": 0.02}, 0.007) is None
    assert pantheon_neuron.quantised_agreement({"cv": 0.007}, 0.007) is None


def test_a_cv_far_below_the_resolution_is_quantisation():
    """serving_mix: cv 0.0001 on a Score that steps by 1 in 149.

    Its Score is decode_tokens // decode, so the request count advances
    once per 32 decode steps and not at all in between. Three repeats at
    DURATION=60 landed on the same integer, and the only thing varying was
    the wall clock in the denominator -- read at the time as the most
    reproducible Score in the suite.
    """
    message = pantheon_neuron.quantised_agreement({"cv": 0.0001}, 1 / 149)
    assert message is not None
    assert "resolution" in message
    assert "same integer" in message


def test_a_continuous_score_is_never_accused():
    """No resolution declared means the counter is not an integer count."""
    assert pantheon_neuron.quantised_agreement({"cv": 0.0001}, None) is None
    assert pantheon_neuron.quantised_agreement({"cv": 0.0001}, 0) is None


def test_a_single_repeat_has_no_cv_to_compare():
    assert pantheon_neuron.quantised_agreement(None, 0.5) is None
    assert pantheon_neuron.quantised_agreement({}, 0.5) is None
    assert pantheon_neuron.quantised_agreement({"cv": None}, 0.5) is None


def test_this_is_the_opposite_check_to_unstable_and_they_cannot_both_fire():
    """UNSTABLE_CV catches a Score disagreeing with itself; this catches
    one that cannot disagree with itself. A Score is not both.
    """
    resolution = 1 / 149
    unstable_cv = pantheon_neuron.UNSTABLE_CV
    assert resolution < unstable_cv, (
        "if a Score's resolution exceeded the instability threshold, both "
        "checks could fire on the same row and each would be right")
    assert pantheon_neuron.quantised_agreement(
        {"cv": unstable_cv + 0.01}, resolution) is None


# -- the resolution comes from the declared formula --------------------------

def _named(name):
    """Not _workload: this file already has one, with a different signature.

    Shadowing it made eight unrelated tests fail with a TypeError -- a
    reminder that appending to a test file is editing it.
    """
    return next(w for w in registry.WORKLOADS if w.name == name)


def test_a_counted_score_resolves_to_one_of_the_things_it_counts():
    """No kernel has to remember to say so; the formula already does.

    transformer_train_step declares `steps_completed / elapsed_s`, so a
    run of 109 steps cannot resolve anything finer than 1 in 109.
    """
    resolution = pantheon_neuron.score_resolution(
        _named("transformer_train_step"), {"steps_completed": 109})
    assert resolution == pytest.approx(1 / 109)


def test_a_kernel_declaration_wins_over_the_formula():
    """serving_mix counts completed requests, which advance once per 32
    decode steps. The formula cannot know that; the kernel does.
    """
    resolution = pantheon_neuron.score_resolution(
        _named("serving_mix"),
        {"requests_completed": 149, "score_resolution": 0.25})
    assert resolution == 0.25


def test_a_continuous_score_has_no_resolution():
    """A bandwidth is not a count of anything, so the question does not
    apply -- and answering it anyway would flag every steady one."""
    assert pantheon_neuron.score_resolution(
        _named("memory_read"), {"bytes_moved": 8 << 30}) is None
    assert pantheon_neuron.score_resolution(
        _named("tensor_virus"), {"passes": 598}) is None


def test_a_missing_or_nonsense_count_resolves_to_nothing():
    workload = _named("transformer_train_step")
    assert pantheon_neuron.score_resolution(workload, {}) is None
    assert pantheon_neuron.score_resolution(
        workload, {"steps_completed": 0}) is None
    assert pantheon_neuron.score_resolution(
        workload, {"steps_completed": 2.5}) is None
    # bool is an int in Python, and a flag would resolve to 1.0.
    assert pantheon_neuron.score_resolution(
        workload, {"steps_completed": True}) is None


def test_every_internal_workload_either_resolves_or_says_why():
    """A sweep, so a new counted Score cannot quietly skip the check.

    Each INTERNAL workload's declared numerator either names a counter the
    kernel returns -- in which case the resolution is derivable -- or the
    formula is not a count over wall time, which this asserts explicitly
    rather than leaving as a silent None.
    """
    counted, continuous = [], []
    for workload in registry.WORKLOADS:
        source = workload.score_source
        if not source or source.source != registry.INTERNAL:
            continue
        formula = source.formula or ""
        numerator, _, denominator = formula.partition("/")
        if denominator.split("#")[0].strip() == "elapsed_s":
            counted.append((workload.name, numerator.strip()))
        else:
            continuous.append(workload.name)

    assert counted, "no counted Scores found -- the parse is broken"
    for name, numerator in counted:
        resolution = pantheon_neuron.score_resolution(
            _named(name), {numerator: 100})
        assert resolution == pytest.approx(0.01), (name, numerator)


def test_every_declared_numerator_is_a_key_some_kernel_returns():
    """The parse being right is not the same as the counter existing.

    test_every_internal_workload_either_resolves_or_says_why feeds a
    synthetic {numerator: 100} and checks the arithmetic. It would pass
    just as well for a numerator no kernel has ever returned, and
    score_resolution would then quietly return None for that workload
    forever -- a check present, running, and answering nothing, which is
    the twelfth shape in docs/checks_that_pass_by_accident.md.

    Read through the comment filter, so a counter named only in the
    registry's own formula string does not vouch for itself.
    """
    import os
    import sourcecheck

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ""
    for name in sorted(os.listdir(os.path.join(root, "kernels"))):
        if name.endswith(".py") and name != "registry.py":
            with open(os.path.join(root, "kernels", name),
                      encoding="utf-8") as handle:
                code += sourcecheck.code_only(handle.read())
    with open(os.path.join(root, "pantheon_neuron.py"),
              encoding="utf-8") as handle:
        code += sourcecheck.code_only(handle.read())

    assert code, "no kernel source read -- the sweep is broken"

    missing = []
    for workload in registry.WORKLOADS:
        source = workload.score_source
        if not source or source.source != registry.INTERNAL:
            continue
        numerator, _, denominator = (source.formula or "").partition("/")
        if denominator.split("#")[0].strip() != "elapsed_s":
            continue
        numerator = numerator.strip()
        if f'"{numerator}"' not in code:
            missing.append(f"{workload.name}: {numerator}")

    assert not missing, (
        f"declared numerators no kernel returns: {missing} -- "
        "score_resolution answers None for these forever")
