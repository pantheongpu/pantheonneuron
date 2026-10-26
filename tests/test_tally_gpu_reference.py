"""tools/tally_gpu_reference.py, held against the file it produced.

The tool needs the pantheongpu report database, which is not vendored, so
its output -- data/publication-2026-09-21/gpu-reference.json -- is committed
instead and CI never runs it. That left two things unchecked: its tallying,
and whether its table of vendor peaks still matches the committed file every
published percentage is divided by.
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import tally_gpu_reference as tally  # noqa: E402

COMMITTED = os.path.join(ROOT, "data", "publication-2026-09-21", "gpu-reference.json")


def _committed():
    with open(COMMITTED, encoding="utf-8") as handle:
        return json.load(handle)["parts"]


@pytest.mark.parametrize("name", sorted(tally.PUBLISHED_PEAKS))
def test_every_peak_in_the_tool_is_the_one_committed(name):
    peak, source = tally.PUBLISHED_PEAKS[name]
    committed = _committed()[name]
    assert committed["published_peak_gbps"] == peak
    assert committed["published_peak_source"] == source


def test_no_committed_peak_lacks_a_source_in_the_tool():
    """A peak in the file the tool cannot reproduce was typed in by hand."""
    for name, entry in _committed().items():
        if entry["published_peak_gbps"] is not None:
            assert name in tally.PUBLISHED_PEAKS, name


def _report(root, name, rows, stem):
    path = root / f"{stem}.json"
    path.write_text(json.dumps({
        "gpu_static_info": [{"name": name}],
        "pantheon_version": "1.2.2",
        "test_results": rows,
    }))


def test_the_tally_is_a_median_per_part_and_row(tmp_path):
    for index, score in enumerate((100.0, 300.0, 200.0)):
        _report(tmp_path, "NVIDIA A10", [
            {"Test Name": "memory_read", "Score": score, "Unit": "GB/s"}], f"a{index}")
    result = tally.tally(str(tmp_path))
    row = result["NVIDIA A10"]["rows"]["memory_read"]
    assert row == {"median_gbps": 200.0, "samples": 3}
    assert result["NVIDIA A10"]["reports"] == 3
    assert result["NVIDIA A10"]["published_peak_gbps"] == 600.0


def test_rows_in_another_unit_or_without_a_score_are_left_out(tmp_path):
    _report(tmp_path, "NVIDIA A10", [
        {"Test Name": "memory_read", "Score": 500.0, "Unit": "GB/s"},
        {"Test Name": "memory_read", "Score": 9.0, "Unit": "ai-ops/s"},
        {"Test Name": "memory_read", "Score": None, "Unit": "GB/s"},
        {"Test Name": "memory_read", "Score": 0, "Unit": "GB/s"},
        {"Test Name": "tensor_virus", "Score": 70.0, "Unit": "TFLOPS"},
    ], "a")
    rows = tally.tally(str(tmp_path))["NVIDIA A10"]["rows"]
    assert rows == {"memory_read": {"median_gbps": 500.0, "samples": 1}}


def test_a_part_with_no_published_peak_says_so(tmp_path):
    _report(tmp_path, "NVIDIA RTX 9999", [
        {"Test Name": "memory_read", "Score": 1.0, "Unit": "GB/s"}], "a")
    entry = tally.tally(str(tmp_path))["NVIDIA RTX 9999"]
    assert entry["published_peak_gbps"] is None
    assert entry["published_peak_source"] == "not looked up"


def test_files_that_are_not_reports_are_skipped(tmp_path):
    (tmp_path / "notes.json").write_text("[1, 2, 3]")
    (tmp_path / "broken.json").write_text("{not json")
    assert tally.tally(str(tmp_path)) == {}


def test_main_refuses_without_a_database(tmp_path):
    assert tally.main([]) == 2
    assert tally.main([str(tmp_path / "missing")]) == 2
