"""The collectives parser, held against the one real nccom-test capture.

all_reduce and p2p_thrasher have never run: they need two devices, and this
account's quota reaches none. So everything the module does with nccom-test's
output was tested against a table written for the tests -- whose header
("count", "time(us)") is not the header nccom-test prints ("count(elems)",
"time:avg(us)"). The 2026-08-26 probe committed a real capture, 11 rows from
inf2.xlarge, and nothing parsed it.

It also adds the check the parse could not make on itself: nccom-test prints
its own average, and a misread column would parse every row, count every
row, and still disagree with it.
"""

import os

import pytest

from kernels import collectives

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURE = os.path.join(ROOT, "data", "probe-2026-08-26-tools",
                       "04-collectives-nccom-test.txt")


def _capture():
    with open(CAPTURE, encoding="utf-8") as handle:
        return handle.read()


# -- the real output ---------------------------------------------------------

def test_every_row_of_the_real_capture_parses():
    rows = collectives.parse_busbw(_capture())
    assert len(rows) == 11
    assert rows[0] == (1048576, 44.62)
    assert rows[-1] == (8388606, 52.43)
    assert collectives.unparsed_rows(_capture()) == 0


def test_nccom_tests_own_average_is_read():
    assert collectives.reported_average(_capture()) == pytest.approx(50.6552)


def test_the_real_capture_agrees_with_its_own_average():
    """50.6545 parsed against 50.6552 printed: the rows are rounded to two
    decimals, and the check must be silent on real output."""
    output = _capture()
    rows = collectives.parse_busbw(output)
    assert collectives.verify_average_matches_nccom(rows, output) is None


def test_the_real_sweep_passes_every_all_reduce_check():
    output = _capture()
    rows = collectives.parse_busbw(output)
    assert collectives.verify_sweep_was_fully_parsed(output) is None
    assert collectives.verify_sweep_covers_both_regimes(
        rows, 1 << 20, 8 << 20) is None


# -- what the cross-check exists to catch ------------------------------------

def test_a_column_appended_after_busbw_is_caught():
    """Every row still parses and counts -- the last decimal is simply the
    wrong column. Only nccom-test's own average can say so."""
    shifted = "\n".join(
        line + "      0.93" if line.strip()[:1].isdigit() else line
        for line in _capture().splitlines())
    rows = collectives.parse_busbw(shifted)
    assert len(rows) == 11
    assert collectives.unparsed_rows(shifted) == 0
    message = collectives.verify_average_matches_nccom(rows, shifted)
    assert message is not None and "not its bus bandwidth" in message


def test_output_without_an_average_line_is_not_accused():
    output = _capture().replace("Avg bus bandwidth: 50.6552 GB/s", "")
    rows = collectives.parse_busbw(output)
    assert collectives.reported_average(output) is None
    assert collectives.verify_average_matches_nccom(rows, output) is None


# -- p2p ran no check at all -------------------------------------------------

def test_a_single_size_run_that_measured_its_size_passes():
    assert collectives.verify_single_size_was_measured(
        [(67108864, 66.09)], 67108864) is None


def test_a_single_size_run_that_swept_anyway_is_caught():
    """run_p2p reports the last row. If nccom-test swept, that is whatever
    size the sweep ended on, published under the pinned bytes."""
    rows = collectives.parse_busbw(_capture())
    message = collectives.verify_single_size_was_measured(rows, 67108864)
    assert message is not None and "not the pinned message size" in message


def test_run_p2p_now_carries_its_checks(monkeypatch):
    monkeypatch.setattr(collectives, "_run", lambda args: _capture())
    result = collectives.run_p2p({"bytes": 67108864, "dtype": "fp32"},
                                 [object()])
    assert result["warning"] is not None
    assert "not the pinned message size" in result["warning"]


def test_run_all_reduce_on_the_real_capture_is_clean(monkeypatch):
    monkeypatch.setattr(collectives, "_run", lambda args: _capture())
    monkeypatch.setattr(collectives, "ranks_for", lambda devices: 2)
    result = collectives.run_all_reduce(
        {"bytes_min": 1 << 20, "bytes_max": 8 << 20, "dtype": "fp32"}, [object()])
    assert result["warning"] is None
    assert result["busbw_gbps"] == pytest.approx(50.6545, abs=1e-4)


def test_run_all_reduce_reports_a_misread_column(monkeypatch):
    shifted = "\n".join(
        line + "      0.93" if line.strip()[:1].isdigit() else line
        for line in _capture().splitlines())
    monkeypatch.setattr(collectives, "_run", lambda args: shifted)
    monkeypatch.setattr(collectives, "ranks_for", lambda devices: 2)
    result = collectives.run_all_reduce(
        {"bytes_min": 1 << 20, "bytes_max": 8 << 20, "dtype": "fp32"}, [object()])
    assert "not its bus bandwidth" in (result["warning"] or "")
