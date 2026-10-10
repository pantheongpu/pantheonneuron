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


def test_a_skipped_row_is_not_applicable_rather_than_failed():
    """all_reduce needs two devices and one was selected: nothing is wrong
    with the part, and reporting it as a failure would cry wolf on every
    single-device run."""
    row = _row(status="SKIPPED", score=None)
    _part, _hosts, results = checker.check(_report([row]))
    assert results == []


def test_a_pass_with_no_score_has_nothing_to_compare():
    """baseline_metrics passes and publishes no Score."""
    _part, _hosts, results = checker.check(_report([_row(score=None)]))
    assert results == []


# -- a row that did not pass is a finding ------------------------------------

def _failed(name="memory_read", detail="core 0 timed out; core 1 timed out",
            status="FAIL"):
    return dict(_row(name=name, score=None, status=status), Detail=detail)


def test_a_failed_row_is_named_not_skipped():
    """It used to be skipped. The tool answers "is this part broken?", and a
    FAILed row has no Score to divide -- which is the case where it is."""
    _part, _hosts, results = checker.check(_report([_failed()]))
    assert [r["workload"] for r in results] == ["memory_read"]
    assert results[0]["verdict"] == "failed"
    assert results[0]["ratio"] is None
    assert "FAIL" in results[0]["note"]
    assert "core 0 timed out" in results[0]["note"]
    assert "core 1" not in results[0]["note"], "first clause of the Detail only"


def test_any_status_other_than_pass_or_skipped_is_a_failure():
    for status in ("ERROR", "TIMEOUT", None):
        _part, _hosts, results = checker.check(_report([_failed(status=status)]))
        assert [r["verdict"] for r in results] == ["failed"], status


def test_a_failed_row_with_a_leftover_score_is_still_failed():
    """A FAIL that also carries a Score is not a measurement of a working
    part; the Score must not be divided and found healthy."""
    row = _row(score=273.5, status="FAIL")
    _part, _hosts, results = checker.check(_report([row]))
    assert results[0]["verdict"] == "failed"


def _write(tmp_path, rows, name="report.json"):
    path = tmp_path / name
    path.write_text(json.dumps(_report(rows)), encoding="utf-8")
    return str(path)


def test_a_report_where_every_row_failed_exits_nonzero(tmp_path, capsys):
    """The defect: every row FAILed, the result list was empty, and main
    printed "every comparable row is within the reference band", exit 0."""
    path = _write(tmp_path, [_failed("memory_read"), _failed("memory_write"),
                             _failed("tensor_virus")])
    assert checker.main([path]) == 1
    out = capsys.readouterr().out
    for name in ("memory_read", "memory_write", "tensor_virus"):
        assert name in out
    assert "every comparable row is within the reference band" not in out
    assert "3 row(s)" in out


def test_one_failed_row_among_healthy_ones_exits_nonzero(tmp_path, capsys):
    path = _write(tmp_path, [_row(score=273.5), _failed("memory_write")])
    assert checker.main([path]) == 1
    assert "memory_write" in capsys.readouterr().out


def test_skipped_and_scoreless_rows_alone_still_exit_zero(tmp_path, capsys):
    """The control: the old handling stands for rows that are not failures."""
    path = _write(tmp_path, [_row(score=273.5),
                             _row(name="all_reduce", status="SKIPPED", score=None),
                             _row(name="baseline_metrics", score=None)])
    assert checker.main([path]) == 0
    assert "every comparable row is within the reference band" in (
        capsys.readouterr().out)


def test_the_old_skip_is_what_made_an_all_failed_report_pass(tmp_path, monkeypatch, capsys):
    """Mutation check. Put the old filter back -- drop every row that is not
    a PASS with a numeric Score before checking -- and the same report that
    exits 1 above exits 0 and claims health. So the all-FAIL report is an
    input that tells the two behaviours apart, not one both would reject."""
    path = _write(tmp_path, [_failed("memory_read"), _failed("memory_write")])
    assert checker.main([path]) == 1, "fixed behaviour"
    capsys.readouterr()

    real = checker.check

    def old_check(report):
        kept = [row for row in report["test_results"]
                if row.get("Status") == "PASS"
                and isinstance(row.get("Score"), (int, float))]
        return real(dict(report, test_results=kept))

    monkeypatch.setattr(checker, "check", old_check)
    assert checker.main([path]) == 0, "the reintroduced defect must pass"
    assert "every comparable row is within the reference band" in (
        capsys.readouterr().out)


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
    compared, refused, failed = [], [], []
    for _path, report in _committed("sweep"):
        _part, _hosts, results = checker.check(report)
        for result in results:
            if result["verdict"] == "failed":
                failed.append(result)
            else:
                (compared if result["verdict"] == "ok" else refused).append(result)
    assert {result["verdict"] for result in refused} == {"not comparable"}
    assert len(compared) == 2, "only the pinned size is comparable"
    assert len(refused) > 5
    # The sweep's own failures (memory_write ran out of device memory at the
    # largest size) are named, not dropped: as many as the raw reports hold.
    assert len(failed) == _raw_failures("sweep")


def _raw_failures(group):
    """FAIL rows in the committed reports of one group, read straight from
    the JSON so the count does not depend on the code under test."""
    return sum(1 for _path, report in _committed(group)
               for row in report["test_results"] if row.get("Status") == "FAIL")


def test_the_committed_one_hour_failures_are_named():
    """memory_read_agg and memory_write_agg timed out on all three hosts'
    3600 s passes. The tool skipped all six, so the one-hour reports read as
    healthy apart from a single pcie_bandwidth row."""
    failed = [result["workload"] for _path, report in _committed("3600")
              for result in checker.check(report)[2]
              if result["verdict"] == "failed"]
    assert len(failed) == _raw_failures("3600")
    assert failed, "the committed one-hour reports hold FAIL rows"
    assert {"memory_read_agg", "memory_write_agg"} <= set(failed)


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


def test_an_equivalent_old_pin_compares():
    """serving_mix was refused on 2026-09-26 over a renamed key that changed
    nothing (#26). registry.EQUIVALENT_PINS says so, and the checker reads it."""
    old_pin = checker.registry.EQUIVALENT_PINS["serving_mix"][0]
    current = checker.registry.resolve("serving_mix")[0].problem
    row = {"Test Name": "serving_mix", "Status": "PASS", "Score": 2.48,
           "Unit": "requests/s", "Duration (s)": 300.0, "Problem": dict(current)}
    _part, _hosts, results = checker.check(_report([row]))
    assert results[0]["verdict"] == "ok"
    assert checker.registry.same_pin("serving_mix", old_pin, current)


def test_a_pin_that_really_differs_is_still_refused():
    current = dict(checker.registry.resolve("serving_mix")[0].problem)
    row = {"Test Name": "serving_mix", "Status": "PASS", "Score": 2.48,
           "Unit": "requests/s", "Duration (s)": 300.0,
           "Problem": dict(current, decode=512)}
    _part, _hosts, results = checker.check(_report([row]))
    assert results[0]["verdict"] == "not comparable"
