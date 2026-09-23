"""tools/check_against_reference.py: the question a validation suite exists for.

This suite could measure a part and compare two vendors, and could not say
whether the part in front of it was performing like the reference fleet.
These cover the comparison and, as much as they cover anything, the refusals
-- a ratio between two different quantities is worse than no ratio.
"""

import glob
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import check_against_reference as checker  # noqa: E402

DATASET = os.path.join(ROOT, "data", "publication-2026-09-21")
_PINNED = {"bytes": 8589934592, "dtype": "bf16", "cores": 1}


def _report(rows, arch="trn1"):
    return {"devices": [{"index": 0, "arch": arch}], "test_results": rows}


def _row(name="memory_read", score=272.98, unit="GB/s", problem=None,
         status="PASS", duration=300.0):
    return {"Test Name": name, "Status": status, "Score": score, "Unit": unit,
            "Duration (s)": duration, "Problem": dict(problem or _PINNED)}


# -- the band ----------------------------------------------------------------

def test_a_tight_reference_spread_gets_the_floor():
    """memory_read_agg's hosts agree to 0.002%; three times that would flag
    a healthy part for rounding."""
    band, because = checker.band_for({"between_host_cv": 0.00002})
    assert band == checker.BAND_FLOOR
    assert "floor" in because


def test_a_wide_reference_spread_widens_the_band():
    band, because = checker.band_for({"between_host_cv": 0.0254})
    assert band == pytest.approx(3 * 0.0254)
    assert "2.54%" in because


def test_a_row_with_no_measured_spread_gets_the_floor():
    assert checker.band_for({"between_host_cv": None})[0] == checker.BAND_FLOOR


# -- what it refuses to compare ----------------------------------------------

def test_a_mock_report_has_no_reference():
    part, why, _ = checker.check(_report([_row()], arch="mock"))
    assert part is None and "mock" in why


def test_a_report_mixing_parts_is_refused():
    report = _report([_row()])
    report["devices"].append({"index": 1, "arch": "inf2"})
    part, why, _ = checker.check(report)
    assert part is None and "mixes parts" in why


def test_a_different_pinned_problem_is_not_a_disagreement():
    """The sweep ran memory_read at seven sizes. Every one of them would
    join on (Test Name, Unit) and mean something else."""
    swept = dict(_PINNED, bytes=1073741824)
    _part, _hosts, results = checker.check(_report([_row(problem=swept)]))
    assert results[0]["verdict"] == "not comparable"
    assert "bytes=" in results[0]["note"]
    assert results[0]["ratio"] is None


def test_a_different_unit_is_refused():
    _part, _hosts, results = checker.check(_report([_row(unit="TFLOPS")]))
    assert results[0]["verdict"] == "not comparable"
    assert "unit" in results[0]["note"]


def test_a_workload_the_reference_never_scored_says_so():
    _part, _hosts, results = checker.check(
        _report([_row(name="all_reduce", unit="GB/s")]))
    assert results[0]["verdict"] == "no reference"


def test_a_failed_row_is_not_compared():
    _part, _hosts, results = checker.check(_report([_row(status="FAIL")]))
    assert results == []


# -- what it catches ---------------------------------------------------------

def test_a_part_at_half_speed_is_flagged():
    _part, _hosts, results = checker.check(_report([_row(score=136.0)]))
    assert results[0]["verdict"] == "below"
    assert "50.2% below" in results[0]["note"]


def test_a_score_above_the_reference_is_flagged_too():
    """The largest wrong number this suite ever produced was 14,513 GB/s
    from a kernel XLA had deleted."""
    _part, _hosts, results = checker.check(_report([_row(score=14513.0)]))
    assert results[0]["verdict"] == "above"


def test_a_healthy_part_passes():
    _part, _hosts, results = checker.check(_report([_row(score=273.5)]))
    assert results[0]["verdict"] == "ok"


def test_a_long_run_says_it_was_a_long_run():
    _part, _hosts, results = checker.check(_report([_row(duration=3600.0)]))
    assert "3600s against the reference's 300s" in results[0]["note"]


def test_the_reference_window_is_not_annotated():
    _part, _hosts, results = checker.check(_report([_row(duration=302.0)]))
    assert "against the reference's" not in results[0]["note"]


# -- against the committed dataset -------------------------------------------

def _committed(group):
    for path in sorted(glob.glob(os.path.join(DATASET, "*", "*", group, "*.json"))):
        with open(path, encoding="utf-8") as handle:
            yield path, json.load(handle)


def test_every_reference_run_agrees_with_the_reference():
    """The 300 s reports are what the reference is made of, so each must
    land inside the band the others produced."""
    checked = 0
    for path, report in _committed("300x3"):
        _part, _hosts, results = checker.check(report)
        for result in results:
            assert result["verdict"] in ("ok", "no reference"), (
                f"{os.path.basename(path)}: {result['workload']} "
                f"{result['verdict']} -- {result['note']}")
            checked += result["verdict"] == "ok"
    assert checked > 40, f"only {checked} rows compared"


def test_the_sweep_compares_only_at_the_pinned_size():
    """Ten rows at seven sizes, all named memory_read or memory_write. Only
    the two that ran the reference's own size are a comparison; the rest
    would join on (Test Name, Unit) and mean something else."""
    compared, refused = [], []
    for _path, report in _committed("sweep"):
        _part, _hosts, results = checker.check(report)
        for result in results:
            (compared if result["verdict"] == "ok" else refused).append(result)
    assert {result["verdict"] for result in refused} == {"not comparable"}
    assert len(compared) == 2, "only the pinned size is comparable"
    assert len(refused) > 5


def test_the_one_hour_pcie_outlier_is_what_this_check_found():
    """trn1-a's 3600 s pcie_bandwidth is 17.9% below the 300 s reference,
    where the other two hosts moved 0.02% and 4.6% over the same change.
    The check exists to surface exactly this, so it is pinned."""
    path = os.path.join(DATASET, "trn1", "trn1-a", "3600",
                        "pantheon_neuron_report_20260922_071327.json")
    with open(path, encoding="utf-8") as handle:
        _part, _hosts, results = checker.check(json.load(handle))
    row = next(r for r in results if r["workload"] == "pcie_bandwidth")
    assert row["verdict"] == "below"
    assert row["ratio"] == pytest.approx(0.8212, abs=1e-4)


def test_no_other_committed_row_is_outside_its_band():
    """One finding, not a tool that flags everything."""
    flagged = []
    for path, report in _committed("*"):
        _part, _hosts, results = checker.check(report)
        flagged += [(os.path.basename(path), r["workload"]) for r in results
                    if r["verdict"] in ("below", "above")]
    assert flagged == [("pantheon_neuron_report_20260922_071327.json",
                        "pcie_bandwidth")]
