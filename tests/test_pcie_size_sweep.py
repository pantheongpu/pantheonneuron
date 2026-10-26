"""tools/pcie_size_sweep.py: the parts that run before, and after, the hardware.

The sweep itself needs a device. What does not: reading SIZES_MIB, refusing a
size the kernel cannot run, and the summary under the table. The summary had
the pin as a literal (`1 << 30`), so the one row the sweep exists to set
beside its neighbours would have stopped being reported the day the pin
moved, with nothing to say so. And a typo in SIZES_MIB was a traceback on a
rented instance.
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import pcie_size_sweep as sweep  # noqa: E402

MIB = 1024 ** 2


def _row(mib, h2d, d2h):
    return {"bytes": mib * MIB, "h2d_gbps": h2d, "d2h_gbps": d2h,
            "combined_gbps": (h2d + d2h) / 2}


def test_sizes_are_read_as_whole_mib():
    assert sweep.sizes_from("1, 16,1024") == [MIB, 16 * MIB, 1024 * MIB]


@pytest.mark.parametrize("text, words", [
    ("abc", "separated by commas"),
    ("1,x", "separated by commas"),
    ("0", "at least 1"),
    ("", "no sizes"),
])
def test_a_bad_setting_is_a_sentence(text, words):
    with pytest.raises(ValueError, match=words):
        sweep.sizes_from(text)


@pytest.mark.parametrize("value", ["abc", "0", ""])
def test_the_tool_exits_2_rather_than_raising(value):
    done = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "pcie_size_sweep.py")],
        env=dict(os.environ, SIZES_MIB=value), capture_output=True,
        text=True, timeout=60)
    assert done.returncode == 2
    assert "Traceback" not in done.stderr
    assert "pcie_size_sweep:" in done.stderr


def test_the_pinned_size_comes_from_the_registry():
    registry = sweep.registry
    assert sweep.pinned_bytes() == registry.resolve("pcie_bandwidth")[0].problem["bytes"]


def test_the_default_sweep_includes_the_pin():
    """Otherwise the summary's reason for existing is not in the table."""
    assert sweep.pinned_bytes() in sweep.sizes_from(sweep.DEFAULT_SIZES_MIB)


def test_the_summary_reports_the_pinned_row(monkeypatch):
    rows = [_row(16, 8.89, 3.93), _row(1024, 6.84, 1.11)]
    lines = sweep.summary(rows)
    assert "h2d peaks at 16 MiB" in lines[0]
    assert "The pinned 1024 MiB reads h2d 6.84" in lines[1]


def test_the_summary_follows_the_pin_when_it_moves(monkeypatch):
    """The literal it replaced would have reported nothing here."""
    monkeypatch.setattr(sweep, "pinned_bytes", lambda: 16 * MIB)
    lines = sweep.summary([_row(16, 8.89, 3.93), _row(1024, 6.84, 1.11)])
    assert "The pinned 16 MiB reads h2d 8.89" in lines[1]


def test_a_sweep_that_missed_the_pin_says_so(monkeypatch):
    lines = sweep.summary([_row(16, 8.89, 3.93)])
    assert "was not in this sweep" in lines[1]


def test_an_empty_sweep_has_no_summary():
    assert sweep.summary([]) == []
