"""Making sure profiler counters came from the kernel we measured.

`find_neff` picks a graph out of directories the compiler shares with every
other compile on the machine. On inf2.xlarge 2026-09-07 it picked wrong: a
capture of the newest NEFF reported `hbm_write_bytes: 2` for a kernel whose
plan moves 8 GiB. Nothing downstream would have noticed -- two bytes divide
by a real total_time and produce a real-looking GB/s, which is worse than an
error because it is publishable.
"""

import os

import pytest

from kernels import profiler


def _neff(path, mtime):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("x")
    os.utime(path, (mtime, mtime))
    return path


# -- narrowing the search ----------------------------------------------------

def test_since_excludes_graphs_compiled_before_the_kernel(tmp_path):
    """A cache full of other people's graphs must not win on recency."""
    old = _neff(str(tmp_path / "old" / "model.neff"), 1000)
    new = _neff(str(tmp_path / "new" / "model.neff"), 5000)

    assert profiler.find_neff(str(tmp_path), since=4000) == new
    assert profiler.find_neff(str(tmp_path), since=900) == new  # newest of both
    assert old  # kept for clarity


def test_without_since_the_newest_wins(tmp_path):
    """Unchanged behaviour when no compile timestamp was recorded."""
    _neff(str(tmp_path / "a" / "model.neff"), 1000)
    newest = _neff(str(tmp_path / "b" / "model.neff"), 9000)

    assert profiler.find_neff(str(tmp_path)) == newest


def test_since_falls_back_when_nothing_is_newer(tmp_path):
    """A cache hit leaves the kernel's NEFF with an old mtime.

    Narrowing to nothing would be worse than the old guess, so the search
    widens again rather than raising -- and the counter guard is what
    catches a wrong pick.
    """
    stale = _neff(str(tmp_path / "cached" / "model.neff"), 1000)

    assert profiler.find_neff(str(tmp_path), since=8000) == stale


def test_missing_neff_still_raises(tmp_path):
    with pytest.raises(profiler.ProfilerUnavailable):
        profiler.find_neff(str(tmp_path), since=1)


# -- the guard that makes a wrong pick unpublishable -------------------------

def test_guard_accepts_a_profile_that_moved_the_planned_bytes():
    counters = {"hbm_read_bytes": 8 << 30, "total_time": 0.03}
    assert profiler.verify_profile_covers_plan(
        counters, "read", 8 << 30
    ) is None


def test_guard_rejects_the_two_byte_graph():
    """The exact failure observed: 2 bytes against an 8 GiB plan."""
    counters = {"hbm_write_bytes": 2, "total_time": 0.002}
    with pytest.raises(profiler.ProfilerUnavailable) as excinfo:
        profiler.verify_profile_covers_plan(counters, "write", 8 << 30)
    assert "not the kernel that was measured" in str(excinfo.value)


def test_guard_allows_traffic_the_compiler_coalesced():
    """Real HBM traffic can be under the request without being another graph.

    The floor is deliberately loose: the point is to catch a different
    graph, not to second-guess the hardware.
    """
    planned = 8 << 30
    counters = {"hbm_read_bytes": int(planned * 0.75), "total_time": 0.03}
    assert profiler.verify_profile_covers_plan(counters, "read", planned) is None


def test_guard_reports_a_missing_counter():
    with pytest.raises(profiler.ProfilerUnavailable):
        profiler.verify_profile_covers_plan({"total_time": 1.0}, "read", 1 << 30)


def test_guard_is_inert_without_a_plan():
    """Nothing to compare against is not evidence of a wrong graph."""
    counters = {"hbm_read_bytes": 2, "total_time": 1.0}
    assert profiler.verify_profile_covers_plan(counters, "read", 0) is None
