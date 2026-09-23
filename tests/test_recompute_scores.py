"""Every Score must be reproducible from its own row. Now it is checked.

`pantheon_neuron._provenance` calls the declared counter list "the row's
promise that the Score can be recomputed from it", and nothing tested the
promise. Applied to the 2026-09-21 dataset it held for 21 of 23 scored rows
and failed for two: `memory_read` and `memory_write` declared
`hbm_read_bytes / total_active_time / 1e9` while publishing
`profiler_total_time_s`, a longer window, so following the formula gave
255.96 against a published 272.94.
"""

import glob
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import recompute_scores  # noqa: E402
from kernels import profiler, registry  # noqa: E402

DATASET = os.path.join(ROOT, "data", "publication-2026-09-21")
REPORTS = sorted(glob.glob(os.path.join(DATASET, "**", "*.json"), recursive=True))
RESULTS = recompute_scores.check(REPORTS)

# The reports in the dataset were written before profiler_active_time_s
# existed, so they cannot carry the window their Score divided by. New runs
# can; test_the_fix_closes_it proves the key does it.
PRE_FIX = {"memory_read", "memory_write"}


def test_the_dataset_has_rows_to_check():
    assert len(RESULTS) > 100, "the dataset should carry many scored rows"


@pytest.mark.parametrize("row", [r for r in RESULTS if r["workload"] not in PRE_FIX],
                         ids=lambda r: f"{r['workload']}-{r['report'][-9:-5]}")
def test_every_row_reproduces_its_score(row):
    assert row["ratio"] is not None, f"{row['workload']}: {row['how']}"
    assert row["agrees"], (
        f"{row['workload']} published {row['score']} and its own row "
        f"recomputes {row['recomputed']} ({row['how']})")


@pytest.mark.parametrize("name", sorted(PRE_FIX))
def test_the_pre_fix_rows_say_exactly_why_they_cannot(name):
    """Not "it failed": the reason has to name the missing key, or the rule.

    Two reasons are legitimate here. Most of these rows predate
    profiler_active_time_s. The sweep's rows have no provenance at all,
    because they ran two repeats and an even count has no median repeat --
    the harness withholds Measurement rather than attributing the Score to an
    execution that did not produce it.
    """
    rows = [r for r in RESULTS if r["workload"] == name]
    assert rows, f"{name} is absent from the dataset"
    for row in rows:
        assert row["ratio"] is None
        assert ("profiler_active_time_s" in row["how"]
                or "provenance withheld by design" in row["how"]), row["how"]


def test_an_even_repeat_count_is_reported_as_withheld_not_as_missing():
    """"The row carries no X" would read as a defect; this one is a rule."""
    workload = next(w for w in registry.WORKLOADS if w.name == "memory_read")
    row = {"Score": 273.0, "Measurement": None,
           "Repeats": {"attempted": 2, "scored": 2}}
    got, how = recompute_scores.recompute(workload, row)
    assert got is None
    assert "withheld by design" in how and "median" in how


def test_an_odd_repeat_count_is_not_excused():
    """With a median repeat the row must carry that repeat's provenance."""
    workload = next(w for w in registry.WORKLOADS if w.name == "memory_read")
    row = {"Score": 273.0, "Measurement": None,
           "Repeats": {"attempted": 3, "scored": 3}}
    _got, how = recompute_scores.recompute(workload, row)
    assert "withheld by design" not in how


def test_the_fix_closes_it():
    """With the key the kernels now publish, the formula reproduces the Score.

    Counters from memory_read's own docstring table: the 8 GiB graph on
    trn1.2xlarge 2026-09-10, whose profile opens with 2.08 ms in which no
    byte moves.
    """
    counters = {"hbm_read_bytes": 8 * 1024 ** 3,
                "total_time": 0.033540, "total_active_time": 0.031463}
    score = profiler.bandwidth_gbps(counters, "read")
    row = {"Measurement": {
        "hbm_read_bytes": counters["hbm_read_bytes"],
        "profiler_total_time_s": counters["total_time"],
        "profiler_active_time_s": profiler.execution_window(counters)[0],
    }}
    workload = next(w for w in registry.WORKLOADS if w.name == "memory_read")
    got, how = recompute_scores.recompute(workload, row)
    assert got == pytest.approx(score, rel=1e-9), how
    # And the old key would have missed by the documented 6%.
    stale = counters["hbm_read_bytes"] / counters["total_time"] / 1e9
    assert stale / score < 0.95


def test_a_wrong_score_is_caught():
    """The check has to fail on the thing it exists to find."""
    workload = next(w for w in registry.WORKLOADS if w.name == "kv_cache_churn")
    row = {"Status": "PASS", "Test Name": "kv_cache_churn", "Score": 1000.0,
           "Measurement": {"cache_updates": 2000.0, "elapsed_s": 1.0}}
    got, _how = recompute_scores.recompute(workload, row)
    assert got == 2000.0
    assert abs(got / row["Score"] - 1.0) > recompute_scores.TOLERANCE


@pytest.mark.parametrize("workload", [w for w in registry.WORKLOADS if w.unit],
                         ids=lambda w: w.name)
def test_every_scored_workload_is_covered(workload):
    """A new workload cannot arrive without a way to check its Score."""
    if workload.name in recompute_scores.CANNOT_RECOMPUTE:
        assert len(recompute_scores.CANNOT_RECOMPUTE[workload.name]) > 40
        return
    assert workload.score_source and workload.score_source.formula
    row = next((r for r in RESULTS if r["workload"] == workload.name), None)
    if row is None:
        pytest.skip("not present in the committed dataset")
    assert row["how"] and "no declared formula" not in row["how"]


def test_no_alias_is_dead():
    """An alias for a term no formula uses is a note about nothing."""
    formulas = " ".join(w.score_source.formula for w in registry.WORKLOADS
                        if w.score_source and w.score_source.formula)
    for term in recompute_scores.ALIASES:
        assert term in formulas, f"{term!r} appears in no declared formula"


def test_every_alias_target_is_a_key_some_row_carries():
    keys = set()
    for path in REPORTS:
        with open(path, encoding="utf-8") as handle:
            for row in json.load(handle).get("test_results") or []:
                keys |= set(row.get("Measurement") or {})
                keys |= set(row.get("Telemetry") or {})
    # profiler_active_time_s postdates the dataset; the rest must be real.
    for target in set(recompute_scores.ALIASES.values()) - {"profiler_active_time_s"}:
        assert target in keys, f"{target!r} is in no report"


# -- the kernels must publish the window their Score divided by --------------

@pytest.mark.parametrize("module_name,counter", [
    ("memory_read", "hbm_read_bytes"), ("memory_write", "hbm_write_bytes")])
def test_the_profile_row_carries_the_window_the_score_used(module_name, counter,
                                                           monkeypatch, tmp_path):
    """The defect, pinned: publishing total_time while dividing by active time.

    Both are real counters and the row used to carry only the longer one, so
    the declared formula over the row came out 6% short.
    """
    from kernels import profiler as prof
    module = __import__(f"kernels.{module_name}", fromlist=["_profile"])

    counters = {counter: 8 * 1024 ** 3, "total_time": 0.033540,
                "total_active_time": 0.031463}
    monkeypatch.setattr(prof, "candidate_search", lambda *a, **k: ([str(tmp_path)], 1))
    monkeypatch.setattr(prof, "select_by_plan",
                        lambda *a, **k: {"counters": counters, "neff": "model.neff",
                                         "plan_coverage": 1.0,
                                         "candidates_tried": 1,
                                         "candidates_available": 1})

    result = module._profile(str(tmp_path), 0.0, counters[counter])
    assert result["profiler_active_time_s"] == counters["total_active_time"]
    assert result["profiler_total_time_s"] == counters["total_time"]
    # And the published Score divides by the active window, not the longer one.
    direction = "read" if module_name == "memory_read" else "write"
    assert result["profiler_gbps"] == pytest.approx(
        counters[counter] / result["profiler_active_time_s"] / 1e9, rel=1e-9)
    assert result["profiler_gbps"] != pytest.approx(
        counters[counter] / result["profiler_total_time_s"] / 1e9, rel=1e-3), direction
