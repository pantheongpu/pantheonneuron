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

import sourcecheck
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


def test_a_cache_hit_stays_reachable_behind_fresher_graphs(tmp_path):
    """The trn1 2026-09-08 failure: `since` used to drop the right answer.

    The run hit the compile cache, so its own NEFF kept an old mtime while
    the workloads before it left dozens of fresher graphs. Filtering to
    "newer than the compile" removed the one file that mattered, and the
    search exhausted its budget on graphs that were never candidates.

    Fresh still ranks first -- a cold compile is the common case -- but
    stale must appear after it rather than not at all.
    """
    cached = _neff(str(tmp_path / "cached" / "model.neff"), 1000)
    for index in range(3):
        _neff(str(tmp_path / f"other{index}" / "model.neff"), 5000 + index)

    found = profiler.find_neffs(str(tmp_path), since=4000)
    assert cached in found, "a cache hit must stay reachable"
    assert found[-1] == cached, "but fresher graphs still rank ahead of it"
    assert len(found) == 4


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


# -- searching instead of guessing -------------------------------------------
#
# mtime ranking picked wrong on both parts this suite has run on: inf2
# captured a graph that moved 2 bytes against an 8 GiB plan, trn1 one that
# moved 4. Both times the kernel's own graph was in the candidate list and
# was not first, and both times the declared Score was lost to the analytic
# fallback. The plan check is now the selector, not only the rejector.

def test_find_neffs_ranks_newest_first(tmp_path):
    middle = _neff(str(tmp_path / "b" / "model.neff"), 5000)
    newest = _neff(str(tmp_path / "c" / "model.neff"), 9000)
    oldest = _neff(str(tmp_path / "a" / "model.neff"), 1000)

    assert profiler.find_neffs(str(tmp_path)) == [newest, middle, oldest]


def test_find_neffs_honours_the_candidate_limit(tmp_path):
    for index in range(profiler.MAX_CANDIDATES + 4):
        _neff(str(tmp_path / f"d{index}" / "model.neff"), 1000 + index)

    assert len(profiler.find_neffs(str(tmp_path))) == profiler.MAX_CANDIDATES
    assert len(profiler.find_neffs(str(tmp_path), limit=2)) == 2


def test_the_candidate_limit_is_raisable(tmp_path, monkeypatch):
    """A run that exhausts the search says so; this is how you answer it."""
    for index in range(profiler.MAX_CANDIDATES + 4):
        _neff(str(tmp_path / f"d{index}" / "model.neff"), 1000 + index)

    monkeypatch.setenv(profiler.CANDIDATES_ENV, "3")
    assert len(profiler.find_neffs(str(tmp_path))) == 3

    monkeypatch.setenv(profiler.CANDIDATES_ENV, "not-a-number")
    assert len(profiler.find_neffs(str(tmp_path))) == profiler.MAX_CANDIDATES


def test_find_neff_is_still_the_head_of_the_ranking(tmp_path):
    _neff(str(tmp_path / "a" / "model.neff"), 1000)
    newest = _neff(str(tmp_path / "b" / "model.neff"), 9000)

    assert profiler.find_neff(str(tmp_path)) == newest


def test_find_neffs_does_not_list_the_same_file_twice(tmp_path, monkeypatch):
    """The caller's workdir can sit inside a compiler default."""
    inner = tmp_path / "cache" / "work"
    path = _neff(str(inner / "model.neff"), 5000)
    monkeypatch.setattr(profiler, "DEFAULT_WORKDIRS", (str(tmp_path / "cache"),))

    assert profiler.find_neffs(str(inner)) == [path]


class _FakeCaptures:
    """Stands in for read_counters: maps a NEFF path to its counters."""

    def __init__(self, by_path):
        self.by_path = by_path
        self.captured = []

    def __call__(self, neff_path, session_path):
        self.captured.append(neff_path)
        result = self.by_path[neff_path]
        if isinstance(result, Exception):
            raise result
        return result


def test_the_search_skips_the_epilogue_and_finds_the_real_graph(monkeypatch):
    """The trn1 2026-09-08 failure, replayed: 4 bytes against an 8 GiB plan."""
    planned = 8 << 30
    fake = _FakeCaptures({
        "/tmp/newest.neff": {"hbm_read_bytes": 4, "total_time": 1e-4},
        "/tmp/real.neff": {"hbm_read_bytes": planned, "total_time": 0.03},
    })
    monkeypatch.setattr(profiler, "read_counters", fake)

    found = profiler.select_by_plan(
        ["/tmp/newest.neff", "/tmp/real.neff"], "/tmp/s.ntff", "read", planned)

    assert found["neff"] == "/tmp/real.neff"
    assert found["plan_coverage"] == 1.0
    assert found["candidates_tried"] == 2
    assert found["candidates_available"] == 2
    # It really did try the wrong one first, rather than being handed the answer.
    assert fake.captured == ["/tmp/newest.neff", "/tmp/real.neff"]


def test_the_first_candidate_wins_when_it_covers_the_plan(monkeypatch):
    """A correct first guess must not pay for extra captures."""
    planned = 1 << 30
    fake = _FakeCaptures({
        "/tmp/a.neff": {"hbm_read_bytes": planned, "total_time": 0.01},
        "/tmp/b.neff": {"hbm_read_bytes": planned, "total_time": 0.01},
    })
    monkeypatch.setattr(profiler, "read_counters", fake)

    found = profiler.select_by_plan(
        ["/tmp/a.neff", "/tmp/b.neff"], "/tmp/s.ntff", "read", planned)

    assert found["candidates_tried"] == 1
    assert fake.captured == ["/tmp/a.neff"]


def test_a_candidate_that_fails_to_capture_does_not_end_the_search(monkeypatch):
    planned = 1 << 30
    fake = _FakeCaptures({
        "/tmp/broken.neff": profiler.ProfilerUnavailable("capture exited 1"),
        "/tmp/real.neff": {"hbm_write_bytes": planned, "total_time": 0.02},
    })
    monkeypatch.setattr(profiler, "read_counters", fake)

    found = profiler.select_by_plan(
        ["/tmp/broken.neff", "/tmp/real.neff"], "/tmp/s.ntff", "write", planned)

    assert found["neff"] == "/tmp/real.neff"


def test_a_candidate_missing_the_counter_does_not_end_the_search(monkeypatch):
    """trn1 reports 90 counters where inf2 reports 108."""
    planned = 1 << 30
    fake = _FakeCaptures({
        "/tmp/nocounter.neff": {"total_time": 0.01},
        "/tmp/real.neff": {"hbm_read_bytes": planned, "total_time": 0.02},
    })
    monkeypatch.setattr(profiler, "read_counters", fake)

    found = profiler.select_by_plan(
        ["/tmp/nocounter.neff", "/tmp/real.neff"], "/tmp/s.ntff", "read", planned)

    assert found["neff"] == "/tmp/real.neff"


def test_exhausting_the_search_says_what_every_candidate_reported(monkeypatch):
    """The diagnosis the single-guess version could never give.

    It could say one graph was wrong. It could not say none was right, nor
    how close any of them came, which is the difference between "retry" and
    "the kernel's NEFF is not on this machine".
    """
    planned = 8 << 30
    fake = _FakeCaptures({
        "/tmp/a.neff": {"hbm_read_bytes": 4, "total_time": 1e-4},
        "/tmp/b.neff": {"hbm_read_bytes": 2, "total_time": 1e-4},
    })
    monkeypatch.setattr(profiler, "read_counters", fake)

    with pytest.raises(profiler.ProfilerUnavailable) as raised:
        profiler.select_by_plan(
            ["/tmp/a.neff", "/tmp/b.neff"], "/tmp/s.ntff", "read", planned)

    message = str(raised.value)
    assert "none of 2 candidate" in message
    assert "a.neff" in message and "b.neff" in message
    assert fake.captured == ["/tmp/a.neff", "/tmp/b.neff"]


def test_exhausting_a_capped_search_says_how_to_look_further(monkeypatch):
    planned = 8 << 30
    paths = [f"/tmp/{index}.neff" for index in range(profiler.MAX_CANDIDATES)]
    fake = _FakeCaptures({p: {"hbm_read_bytes": 4, "total_time": 1e-4}
                          for p in paths})
    monkeypatch.setattr(profiler, "read_counters", fake)

    with pytest.raises(profiler.ProfilerUnavailable) as raised:
        profiler.select_by_plan(paths, "/tmp/s.ntff", "read", planned)

    assert profiler.CANDIDATES_ENV in str(raised.value)


def test_an_empty_candidate_list_is_refused():
    with pytest.raises(profiler.ProfilerUnavailable, match="no NEFF candidates"):
        profiler.select_by_plan([], "/tmp/s.ntff", "read", 1 << 30)


def test_a_partial_but_sufficient_profile_is_accepted(monkeypatch):
    """The floor is 0.5, not 1.0: the compiler may legitimately coalesce."""
    planned = 1 << 30
    fake = _FakeCaptures({
        "/tmp/a.neff": {"hbm_read_bytes": int(planned * 0.75), "total_time": 0.01},
    })
    monkeypatch.setattr(profiler, "read_counters", fake)

    found = profiler.select_by_plan(
        ["/tmp/a.neff"], "/tmp/s.ntff", "read", planned)
    assert found["plan_coverage"] == 0.75


# -- coverage, as its own judgement ------------------------------------------

def test_plan_coverage_is_the_ratio_it_claims():
    counters = {"hbm_read_bytes": 512, "total_time": 1.0}
    assert profiler.plan_coverage(counters, "read", 1024) == 0.5


def test_plan_coverage_is_none_without_a_plan():
    counters = {"hbm_read_bytes": 512, "total_time": 1.0}
    assert profiler.plan_coverage(counters, "read", 0) is None


def test_plan_coverage_raises_when_the_counter_is_absent():
    with pytest.raises(profiler.ProfilerUnavailable, match="hbm_read_bytes"):
        profiler.plan_coverage({"total_time": 1.0}, "read", 1 << 30)


# -- best match, not first acceptable ----------------------------------------
#
# Taking the first candidate over the floor made the declared Score
# irreproducible: three runs of memory_read returned 256.17, 178.7 and
# 119.19 GB/s, and the last cleared a 0.5 floor while diverging from its own
# analytic cross-check by 56%.

def test_the_best_covering_candidate_wins_not_the_first_acceptable(monkeypatch):
    planned = 8 << 30
    fake = _FakeCaptures({
        # Clears the floor, but is plainly not the kernel's graph.
        "/tmp/partial.neff": {"hbm_read_bytes": int(planned * 0.55),
                              "total_time": 0.02},
        "/tmp/real.neff": {"hbm_read_bytes": planned, "total_time": 0.03},
    })
    monkeypatch.setattr(profiler, "read_counters", fake)

    found = profiler.select_by_plan(
        ["/tmp/partial.neff", "/tmp/real.neff"], "/tmp/s.ntff", "read", planned)

    assert found["neff"] == "/tmp/real.neff"
    assert found["plan_coverage"] == 1.0
    assert fake.captured == ["/tmp/partial.neff", "/tmp/real.neff"]


def test_an_exact_match_stops_the_search_immediately():
    """Each further attempt is a real NEFF replay, so nothing beats 1.0."""
    planned = 1 << 30
    fake = _FakeCaptures({
        "/tmp/a.neff": {"hbm_read_bytes": planned, "total_time": 0.01},
        "/tmp/b.neff": {"hbm_read_bytes": planned, "total_time": 0.01},
    })
    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(profiler, "read_counters", fake)
        found = profiler.select_by_plan(
            ["/tmp/a.neff", "/tmp/b.neff"], "/tmp/s.ntff", "read", planned)
    assert found["candidates_tried"] == 1
    assert fake.captured == ["/tmp/a.neff"]


def test_a_near_miss_is_still_returned_but_says_how_near(monkeypatch):
    """Nothing matched exactly; the closest above the floor is the answer.

    The row carries plan_coverage, so a near miss reads as a near miss
    rather than as a clean identification.
    """
    planned = 1 << 30
    fake = _FakeCaptures({
        "/tmp/a.neff": {"hbm_read_bytes": int(planned * 0.60), "total_time": 0.01},
        "/tmp/b.neff": {"hbm_read_bytes": int(planned * 0.85), "total_time": 0.01},
    })
    monkeypatch.setattr(profiler, "read_counters", fake)

    found = profiler.select_by_plan(
        ["/tmp/a.neff", "/tmp/b.neff"], "/tmp/s.ntff", "read", planned)
    assert found["neff"] == "/tmp/b.neff"
    assert 0.84 < found["plan_coverage"] < 0.86
    assert found["candidates_tried"] == 2


def test_overshoot_is_ranked_by_distance_from_one(monkeypatch):
    """A graph moving twice the plan is as wrong as one moving half."""
    planned = 1 << 30
    fake = _FakeCaptures({
        "/tmp/double.neff": {"hbm_read_bytes": planned * 2, "total_time": 0.01},
        "/tmp/close.neff": {"hbm_read_bytes": int(planned * 0.9), "total_time": 0.01},
    })
    monkeypatch.setattr(profiler, "read_counters", fake)

    found = profiler.select_by_plan(
        ["/tmp/double.neff", "/tmp/close.neff"], "/tmp/s.ntff", "read", planned)
    assert found["neff"] == "/tmp/close.neff"


def test_everything_below_the_floor_still_raises(monkeypatch):
    """A best-of-a-bad-lot must not become a Score."""
    planned = 8 << 30
    fake = _FakeCaptures({
        "/tmp/a.neff": {"hbm_read_bytes": int(planned * 0.38), "total_time": 0.01},
        "/tmp/b.neff": {"hbm_read_bytes": int(planned * 0.06), "total_time": 0.01},
    })
    monkeypatch.setattr(profiler, "read_counters", fake)

    with pytest.raises(profiler.ProfilerUnavailable, match="none of 2 candidate"):
        profiler.select_by_plan(
            ["/tmp/a.neff", "/tmp/b.neff"], "/tmp/s.ntff", "read", planned)


# -- coverage is necessary and not sufficient --------------------------------
#
# trn1.2xlarge 2026-09-08, warm cache: the selector published 23.58 GB/s for
# memory_read against an analytic 271.7. The captured graph cleared the
# coverage floor -- it moved about the planned bytes -- because two
# workloads in this suite pin 8 GiB and byte-coverage cannot tell their
# graphs apart. The exact-match early exit could not help: another graph
# also covers the plan.

def test_a_diverging_profile_is_refused_when_the_kernel_read_everything():
    """read_verified_ratio adjudicates, because it comes from the kernel.

    It is measured from the accumulator rather than from any capture, so
    when it says the plan was fully read, the analytic figure is the
    trustworthy one and the profile belongs to another graph.
    """
    from kernels import memory_read

    assert memory_read._touched_the_whole_plan(1.0)
    assert memory_read._touched_the_whole_plan(0.995)
    assert not memory_read._touched_the_whole_plan(0.5)
    assert not memory_read._touched_the_whole_plan(None)


def test_both_bandwidth_kernels_adjudicate_the_same_way():
    from kernels import memory_read, memory_write

    for module in (memory_read, memory_write):
        assert module._touched_the_whole_plan(1.0)
        assert not module._touched_the_whole_plan(0.0)


def test_the_divergence_that_triggered_this_would_now_degrade():
    """23.58 against 271.7 is a ratio of 0.09, far outside tolerance."""
    from kernels import memory_read

    message = memory_read.verify_against_analytic(23.58, 271.7)
    assert message is not None
    assert "0.09" in message


# -- searching the caller's directory exclusively ----------------------------
#
# The docstring always said "the caller's workdir first, then the compiler's
# defaults". The code searched both at once, so a warm shared cache buried
# the kernel's own graph: candidate 13 of 14 in one measurement, and outside
# the budget entirely in a fuller run, which published a wrong graph at
# 23.58 GB/s against an analytic 271.7.

def test_the_callers_directory_is_searched_exclusively(tmp_path, monkeypatch):
    """When the workdir holds graphs, the shared cache is not consulted."""
    shared = tmp_path / "shared"
    mine = tmp_path / "mine"
    for index in range(5):
        _neff(str(shared / f"s{index}" / "model.neff"), 9000 + index)
    ours = _neff(str(mine / "model.neff"), 1000)   # older than all of them
    monkeypatch.setattr(profiler, "DEFAULT_WORKDIRS", (str(shared),))

    found = profiler.find_neffs(str(mine))
    assert found == [ours], "an isolated workdir makes this a confirmation"


def test_the_defaults_are_still_the_fallback(tmp_path, monkeypatch):
    """An empty workdir must not mean no candidates at all.

    An @nki.jit kernel ignores compiler_workdir and writes to the
    compiler's own tree, which is why the fallback exists.
    """
    shared = tmp_path / "shared"
    empty = tmp_path / "empty"
    empty.mkdir()
    theirs = _neff(str(shared / "model.neff"), 5000)
    monkeypatch.setattr(profiler, "DEFAULT_WORKDIRS", (str(shared),))

    assert profiler.find_neffs(str(empty)) == [theirs]


def test_a_missing_workdir_still_falls_back(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    theirs = _neff(str(shared / "model.neff"), 5000)
    monkeypatch.setattr(profiler, "DEFAULT_WORKDIRS", (str(shared),))

    assert profiler.find_neffs(str(tmp_path / "does-not-exist")) == [theirs]


def test_the_bandwidth_kernels_point_the_compiler_at_that_directory():
    """Isolation only works if the graphs actually land there."""
    import sourcecheck
    from kernels import cores, memory_read, memory_write

    assert cores.COMPILE_CACHE == "NEURON_COMPILE_CACHE_URL"
    for module in (memory_read, memory_write):
        code = sourcecheck.function_code(module.run)
        assert "os . environ . setdefault ( cores . COMPILE_CACHE , workdir )" in code


def test_an_explicit_compile_cache_is_not_overridden(monkeypatch):
    """setdefault, not assignment: someone who pinned it is answering a
    question we should not overrule."""
    import sourcecheck
    from kernels import memory_read

    code = sourcecheck.function_code(memory_read.run)
    assert "setdefault" in code
    assert "os . environ [ cores . COMPILE_CACHE ] =" not in code


# -- the bound was one-sided ------------------------------------------------

def _counters(read_bytes):
    return {"hbm_read_bytes": read_bytes, "total_time": 1.0}


def test_a_graph_far_larger_than_the_plan_is_refused():
    """It was accepted, and published as a bandwidth ten times too fast.

    select_by_plan tested `plan_coverage >= floor`, which bounds only the
    low side. A profile carrying ten times the planned bytes divides by a
    real total_time and produces a real-looking GB/s -- worse than an
    error, because it is publishable.

    The one-sided bound survived because verify_profile_covers_plan was
    dead: nothing exercised it against a graph larger than the plan, and
    the inline copy inherited the same gap.
    """
    planned = 1 << 30
    with pytest.raises(profiler.ProfilerUnavailable) as raised:
        profiler.verify_profile_covers_plan(
            _counters(planned * 10), "read", planned)
    assert "coverage 10" in str(raised.value)
    assert "someone else's graph" in str(raised.value)


def test_a_graph_far_smaller_than_the_plan_is_still_refused():
    """The side that always worked, kept as a control."""
    planned = 1 << 30
    with pytest.raises(profiler.ProfilerUnavailable) as raised:
        profiler.verify_profile_covers_plan(_counters(4), "read", planned)
    assert "only 4 bytes" in str(raised.value)


def test_a_graph_matching_the_plan_passes():
    planned = 1 << 30
    for moved in (planned, int(planned * 0.9), int(planned * 1.5)):
        assert profiler.verify_profile_covers_plan(
            _counters(moved), "read", planned) is None


def test_the_bounds_bracket_one():
    """Or the check would refuse the graph it is meant to accept."""
    assert profiler.CEILING > 1.0
    # The default floor is below 1.0 by the same argument.
    planned = 1 << 30
    assert profiler.verify_profile_covers_plan(
        _counters(planned), "read", planned) is None


def test_select_by_plan_refuses_an_oversized_best_rather_than_returning_it():
    """The gate has to be on the way out, not only in the ranking.

    Ranking already preferred |coverage - 1| smallest, so an oversized
    graph could be `best` while still being the wrong graph -- and the
    acceptance test never looked at the upper side.
    """
    source = sourcecheck.function_code(profiler.select_by_plan)
    assert "verify_profile_covers_plan" in source
    assert 'best [ "plan_coverage" ] >= floor' not in source


def test_the_ceiling_clears_every_coverage_ever_observed():
    """Every profiler_plan_coverage on record reads exactly 1.0.

    Checked against the reports on the trn1.2xlarge, 2026-09-10. So the
    ceiling has a factor of two of headroom against the only values the
    capture has produced -- which is the evidence for the number, and the
    reason it is 2.0 rather than something tighter.

    A ceiling that is too tight refuses a real capture and sends the row
    to the analytic fallback, undoing the core reservation and
    compile-cache isolation that made the declared profiler source work
    at all.
    """
    observed = 1.0
    assert profiler.CEILING >= 2 * observed
    planned = 1 << 30
    assert profiler.verify_profile_covers_plan(
        _counters(int(planned * observed)), "read", planned) is None


# -- a copy covers the plan too ---------------------------------------------

def test_a_copy_graph_is_passed_over_for_the_kernel(monkeypatch):
    """trn1.2xlarge 2026-09-10 at 1 GiB: a buffer-copy graph read 1 GiB and
    wrote 1 GiB beside memory_read's kernel, which read 1 GiB and wrote
    nothing. Coverage alone gave both 1.0 and took whichever came first."""
    planned = 1 << 30
    fake = _FakeCaptures({
        "/tmp/copy.neff": {"hbm_read_bytes": planned, "hbm_write_bytes": planned,
                           "total_time": 0.008167},
        "/tmp/kernel.neff": {"hbm_read_bytes": planned, "hbm_write_bytes": 512,
                             "total_time": 0.006020},
    })
    monkeypatch.setattr(profiler, "read_counters", fake)
    found = profiler.select_by_plan(
        ["/tmp/copy.neff", "/tmp/kernel.neff"], "/tmp/s.ntff", "read", planned)
    assert found["neff"] == "/tmp/kernel.neff"
    assert fake.captured == ["/tmp/copy.neff", "/tmp/kernel.neff"]


def test_the_write_kernels_own_source_read_is_not_mistaken_for_a_copy(monkeypatch):
    """memory_write reads one 2 MiB tile per pass of a 4 GiB plan."""
    planned = 4 << 30
    fake = _FakeCaptures({
        "/tmp/kernel.neff": {"hbm_write_bytes": planned, "hbm_read_bytes": 2 << 20,
                             "total_time": 0.019},
    })
    monkeypatch.setattr(profiler, "read_counters", fake)
    found = profiler.select_by_plan(["/tmp/kernel.neff"], "/tmp/s.ntff", "write", planned)
    assert found["neff"] == "/tmp/kernel.neff"


def test_only_copies_means_no_selection_and_says_why(monkeypatch):
    planned = 1 << 30
    fake = _FakeCaptures({
        "/tmp/copy.neff": {"hbm_read_bytes": planned, "hbm_write_bytes": planned,
                           "total_time": 0.008},
    })
    monkeypatch.setattr(profiler, "read_counters", fake)
    with pytest.raises(profiler.ProfilerUnavailable, match="a copy, not this kernel"):
        profiler.select_by_plan(["/tmp/copy.neff"], "/tmp/s.ntff", "read", planned)
