"""tools/summarise_hosts.py: two spreads, kept apart, and nothing averaged that should not be."""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import summarise_hosts  # noqa: E402


def _row(name, score, cv=0.001, status="PASS", problem=None, unit="GB/s"):
    return {"Test Name": name, "Status": status, "Score": score, "Unit": unit,
            "Score Method": "neuron-profile",
            "Problem": problem or {"bytes": 8},
            "Repeats": {"attempted": 3, "scored": 3, "cv": cv}}


def _write(root, host, group, stamp, rows):
    directory = root / host / group
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"pantheon_neuron_report_{stamp}.json").write_text(
        json.dumps({"test_results": rows}), encoding="utf-8")


def _only(summary, **match):
    found = [r for r in summary["rows"]
             if all(r[k] == v for k, v in match.items())]
    assert len(found) == 1, found
    return found[0]


def test_between_host_spread_is_over_host_medians(tmp_path):
    for host, score in (("a", 270.0), ("b", 273.0), ("c", 276.0)):
        _write(tmp_path, host, "300x3", "1", [_row("memory_read", score, cv=0.002)])
    row = _only(summarise_hosts.summarise(str(tmp_path)), workload="memory_read")
    assert row["median_of_hosts"] == 273.0
    assert row["between_host_cv"] == pytest.approx(3.0 / 273.0)
    assert row["worst_within_host_cv"] == 0.002
    assert row["scores"] == {"a": 270.0, "b": 273.0, "c": 276.0}


def test_worst_within_host_cv_is_the_largest_not_the_mean(tmp_path):
    for host, cv in (("a", 0.001), ("b", 0.2)):
        _write(tmp_path, host, "300x3", "1", [_row("memory_read", 270.0, cv=cv)])
    row = _only(summarise_hosts.summarise(str(tmp_path)), workload="memory_read")
    assert row["worst_within_host_cv"] == 0.2


def test_groups_are_never_merged(tmp_path):
    """A 3600 s row and a 300 s row of the same workload are two rows."""
    _write(tmp_path, "a", "300x3", "1", [_row("memory_read", 270.0)])
    _write(tmp_path, "a", "3600", "2", [_row("memory_read", 250.0)])
    summary = summarise_hosts.summarise(str(tmp_path))
    assert _only(summary, group="300x3")["median_of_hosts"] == 270.0
    assert _only(summary, group="3600")["median_of_hosts"] == 250.0


def test_a_swept_problem_does_not_land_beside_the_pinned_one(tmp_path):
    _write(tmp_path, "a", "sweep", "1", [_row("memory_read", 100.0, problem={"bytes": 1})])
    _write(tmp_path, "a", "sweep", "2", [_row("memory_read", 270.0, problem={"bytes": 8})])
    summary = summarise_hosts.summarise(str(tmp_path))
    assert sorted(r["problem"]["bytes"] for r in summary["rows"]) == [1, 8]


def test_a_host_that_failed_is_listed_not_dropped(tmp_path):
    _write(tmp_path, "a", "300x3", "1", [_row("moe_router", 5.0)])
    _write(tmp_path, "b", "300x3", "1", [_row("moe_router", None, status="FAIL")])
    row = _only(summarise_hosts.summarise(str(tmp_path)), workload="moe_router")
    assert row["hosts"] == {"a": "PASS", "b": "FAIL"}
    assert row["scores"] == {"a": 5.0}
    # One scored host is not a spread.
    assert row["between_host_cv"] is None
    assert "1/2" in summarise_hosts.render({"hosts": ["a", "b"], "rows": [row]})


def test_a_rerun_on_one_host_keeps_the_later_report(tmp_path):
    _write(tmp_path, "a", "300x3", "20260921_100000", [_row("memory_read", 100.0)])
    _write(tmp_path, "a", "300x3", "20260921_110000", [_row("memory_read", 270.0)])
    row = _only(summarise_hosts.summarise(str(tmp_path)), workload="memory_read")
    assert row["scores"] == {"a": 270.0}


def test_an_unreadable_file_is_skipped_not_fatal(tmp_path):
    _write(tmp_path, "a", "300x3", "1", [_row("memory_read", 270.0)])
    (tmp_path / "a" / "300x3" / "partial.json").write_text("{not json", encoding="utf-8")
    assert len(summarise_hosts.summarise(str(tmp_path))["rows"]) == 1
