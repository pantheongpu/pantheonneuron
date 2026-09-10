"""Read hardware counters out of ``neuron-profile``.

This is how a Score reaches its declared source. ``neuron-monitor`` streams
telemetry but has no HBM byte counters; only the profiler does, and only as
a per-execution capture.

Five environment traps are encoded here, each one observed on hardware and
each one costing a round trip to find. The first four came out of the
2026-08-26 probes; the fifth out of the 2026-09-07 bring-up:

1. ``neuron-profile view`` exits with "$HOME is not defined" when HOME is
   unset. Anything running under SSM or a bare service manager hits this.
2. The Neuron bin directory must be on PATH, not merely referenced by
   absolute path -- the tools shell out to each other.
3. ``capture`` writes NTFF v6, which ``show-session`` and ``view`` can read.
   ``inspect`` writes v115, which the tooling shipped in the same AMI
   cannot read at all ("supported: 1 - 6"). Use capture.
4. The tools emit klog lines on stdout alongside the JSON, so the payload
   starts at the first ``{`` and everything before it is noise.
5. ``capture`` replays the NEFF and so needs a NeuronCore of its own. The
   workload process holds every visible core, so without a reserved core
   this fails with "Requested:2 Available:0" and the Score degrades to the
   analytic figure -- which, until 2026-09-07, it always had. See
   kernels/cores.py.

STATUS: VERIFIED END TO END, trn1.2xlarge 2026-09-10. The counter-reading
path is no longer exercised only by hand-taken captures: memory_read
(256.0888 GB/s) and memory_write (226.6205 GB/s) both scored through it in
a full pass, which is the first time the declared profiler source produced
Scores in a scored run rather than a standalone probe session.

Capture alone was verified earlier, on inf2.xlarge 2026-09-07. The gap
between the two dates is NEFF identification: a capture that runs against
the wrong graph produces a plausible bandwidth from four bytes, and
verify_profile_covers_plan refusing it is what made the difference visible
rather than silent. See docs/neuron_counters.md for the raw output this
parses.
"""

import json
import os
import shutil
import subprocess
import typing

from . import cores


NEURON_BIN = "/opt/aws/neuron/bin"
_TIMEOUT = 600


class ProfilerUnavailable(RuntimeError):
    """Raised when the profiler cannot produce counters."""


def _environment() -> typing.Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("HOME", "/root")  # trap 1
    path = env.get("PATH", "")
    if NEURON_BIN not in path.split(os.pathsep):  # trap 2
        env["PATH"] = os.pathsep.join([NEURON_BIN, path]) if path else NEURON_BIN

    # trap 5: capture replays the NEFF, so it needs a NeuronCore of its own.
    # The workload process holds every core the runtime made visible to it,
    # so without a reserved core this fails with "Requested:2 Available:0"
    # and the Score silently falls back to the analytic figure. The
    # orchestrator reserves one and names it here; see kernels/cores.py.
    reserved = env.get(cores.RESERVED_CORE)
    if reserved:
        env[cores.VISIBLE_CORES] = reserved
    return env


def available() -> bool:
    return shutil.which("neuron-profile", path=_environment()["PATH"]) is not None


# nrt_infodump ends its diagnostic dump with a "cut to here" separator, so the
# LAST line of a failing run is decoration, not the error. Prefer the lines
# that actually say something.
_NOISE = ("-----8<-----", "cut to here", "====", "----")

# nrt_infodump tags its ENTIRE environment dump at ERROR level, so "contains
# ERROR" selects dozens of NEURON_* variable assignments and buries the one
# line that says what actually went wrong. The dump is never the cause.
_DUMP = "nrt_infodump"


def _explain(stderr: typing.Optional[str], stdout: typing.Optional[str]) -> str:
    """Summarise why a neuron-profile invocation failed."""
    lines = [
        line.strip()
        for line in ((stderr or "") + "\n" + (stdout or "")).splitlines()
        if line.strip() and not any(n in line for n in _NOISE)
    ]
    if not lines:
        return "no output"
    signal = [line for line in lines if _DUMP not in line]
    errors = [line for line in signal if "ERROR" in line or "error" in line.lower()]
    # The FIRST real error is the cause; everything after it is fallout.
    chosen = errors[:3] or signal[:3] or lines[:2]
    return " | ".join(chosen)[:600]


def _run(args: typing.Sequence[str]) -> str:
    binary = shutil.which("neuron-profile", path=_environment()["PATH"])
    if binary is None:
        raise ProfilerUnavailable(
            f"neuron-profile not found; expected it in {NEURON_BIN}"
        )
    try:
        completed = subprocess.run(
            [binary, *args],
            capture_output=True, text=True, timeout=_TIMEOUT,
            env=_environment(), check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProfilerUnavailable(f"neuron-profile failed: {error}") from error
    if completed.returncode != 0:
        raise ProfilerUnavailable(
            f"neuron-profile exited {completed.returncode}: "
            + _explain(completed.stderr, completed.stdout)
        )
    return completed.stdout


# Where neuronx-cc leaves NEFFs when nothing sets compiler_workdir.
# Observed on trn1.2xlarge, 2026-08-27: an @nki.jit kernel writes to
# /tmp/no-user/neuroncc_compile_workdir/<uuid>/model.MODULE_<hash>.neff
# rather than anywhere the caller chose.
DEFAULT_WORKDIRS = (
    "/tmp/no-user/neuroncc_compile_workdir",
    "/var/tmp/neuron-compile-cache",
)


# How many NEFFs the plan search will capture before giving up. Each
# attempt is a real replay of the graph, so this is a time budget, not a
# correctness limit: the right NEFF is usually within the first few by
# mtime, and a machine whose compile cache holds hundreds of unrelated
# graphs should not be walked through all of them. Raise it with
# PANTHEON_NEURON_NEFF_CANDIDATES if a run reports exhausting the search.
MAX_CANDIDATES = 16
CANDIDATES_ENV = "PANTHEON_NEURON_NEFF_CANDIDATES"


def _candidate_limit() -> int:
    try:
        value = int(os.environ.get(CANDIDATES_ENV, ""))
    except ValueError:
        return MAX_CANDIDATES
    return value if value > 0 else MAX_CANDIDATES


def find_neffs(workdir: str, since: typing.Optional[float] = None,
               limit: typing.Optional[int] = None) -> typing.List[str]:
    """Every NEFF that could be the kernel's own graph, newest first.

    Searches the caller's workdir first, then the compiler's own default
    locations -- an @nki.jit kernel ignores compiler_workdir and writes to
    its own tree, so looking only where the caller asked finds nothing.

    ``since`` is a timestamp taken before the kernel compiled, and narrows
    the search to graphs built after it. It widens again when nothing is
    newer, because a compile-cache hit leaves the kernel's own NEFF with an
    old mtime and narrowing to nothing is worse than an unnarrowed guess.

    **Ordering is a ranking, not an answer.** mtime is the best cheap
    signal and it is not a reliable one: a single run compiles several
    graphs and the last is often a trivial epilogue. On inf2.xlarge
    2026-09-07 the newest reported ``hbm_write_bytes: 2`` against an 8 GiB
    plan, and on trn1.2xlarge 2026-09-08 the newest moved 4 bytes against
    the same plan. Both times the kernel's real graph was in this list and
    was not first. That is why callers should hand the list to
    ``select_by_plan`` rather than taking the head of it.
    """
    def scan(roots):
        found = []
        seen = set()
        for root in roots:
            if not root or not os.path.isdir(root):
                continue
            for base, _, names in os.walk(root):
                for name in names:
                    if not name.endswith(".neff"):
                        continue
                    path = os.path.join(base, name)
                    # The caller's workdir can sit inside a default one, so
                    # the same file is reachable by two roots.
                    real = os.path.realpath(path)
                    if real in seen:
                        continue
                    try:
                        found.append((os.path.getmtime(path), path))
                    except OSError:
                        continue
                    seen.add(real)
        return found

    # The caller's own directory *exclusively*, when it holds anything.
    # The docstring has always said "first, then the defaults"; the code
    # searched both at once, which is how a warm shared cache buried the
    # kernel's own graph. Setting NEURON_COMPILE_CACHE_URL to this same
    # directory makes it hold only this run's graphs -- measured on
    # trn1.2xlarge 2026-09-08: 3 NEFFs there against 14 in the shared
    # cache -- so searching it alone turns identification from a search
    # into a confirmation.
    candidates = scan([workdir])
    if not candidates:
        candidates = scan(DEFAULT_WORKDIRS)

    if not candidates:
        raise ProfilerUnavailable(
            f"no .neff under {workdir} -- trace with compiler_workdir set, "
            "otherwise torch_neuronx removes it"
        )

    # `since` ranks, it does not exclude. It used to filter, and on
    # trn1.2xlarge 2026-09-08 that filter removed the right answer: the run
    # hit the compile cache ("Using a cached neff at ..."), so the kernel's
    # own NEFF kept its original mtime while the 24 workloads before it left
    # 78 fresher graphs on the machine. Everything newer than `since` was
    # kept, the one graph we wanted was dropped, and the search exhausted
    # six candidates whose best coverage was 0.38 of the plan.
    #
    # Fresh-first is still the right order -- a cold compile is the common
    # case and its NEFF really is the newest -- but a cache hit must leave
    # the search able to reach the older file rather than never seeing it.
    if since is not None:
        fresh = sorted([e for e in candidates if e[0] >= since], reverse=True)
        stale = sorted([e for e in candidates if e[0] < since], reverse=True)
        ordered = [path for _, path in fresh + stale]
    else:
        ordered = [path for _, path in sorted(candidates, reverse=True)]
    return ordered[:(limit if limit is not None else _candidate_limit())]


def find_neff(workdir: str, since: typing.Optional[float] = None) -> str:
    """The single best guess at the kernel's NEFF: the newest candidate.

    Kept because the ranking is still useful on its own -- and because a
    caller with no plan to check against has nothing better to go on. A
    caller that *can* check should use ``select_by_plan``, which is the
    difference between guessing and identifying.
    """
    return find_neffs(workdir, since=since, limit=1)[0]


def capture(neff_path: str, session_path: str) -> str:
    """Execute the NEFF under the profiler, writing an NTFF session."""
    _run(["capture", "-n", neff_path, "-s", session_path])  # trap 3
    if not os.path.exists(session_path):
        raise ProfilerUnavailable(f"capture produced no session at {session_path}")
    return session_path


def summary(neff_path: str, session_path: str) -> typing.Dict[str, typing.Any]:
    """Decode a session into its flat counter dictionary."""
    raw = _run([
        "view", "-n", neff_path, "-s", session_path,
        "--output-format", "summary-json",
    ])
    start = raw.find("{")  # trap 4
    if start < 0:
        raise ProfilerUnavailable("no JSON in neuron-profile view output")
    try:
        payload = json.loads(raw[start:])
    except json.JSONDecodeError as error:
        raise ProfilerUnavailable(f"could not parse view output: {error}") from error

    # Counters sit under a single content-hash key, e.g. "n_7cf2c3ef...".
    counters: typing.Dict[str, typing.Any] = {}
    for value in payload.values():
        if isinstance(value, dict):
            counters.update(value)
    if not counters:
        raise ProfilerUnavailable("view returned no counters")
    return counters


def read_counters(neff_path: str, session_path: str) -> typing.Dict[str, typing.Any]:
    capture(neff_path, session_path)
    return summary(neff_path, session_path)


def plan_coverage(counters: typing.Mapping[str, typing.Any],
                  direction: str,
                  expected_bytes: int) -> typing.Optional[float]:
    """How much of the planned traffic this profile actually accounts for.

    1.0 means the captured graph moved exactly the bytes the plan
    describes. Returns None when there is no plan to check against, which
    is not a failure -- it is the honest answer for a caller that did not
    supply one.

    One NEFF execution moves the plan's bytes once, so the comparison is
    against a single pass, not the whole timed loop.
    """
    key = {"read": "hbm_read_bytes", "write": "hbm_write_bytes"}[direction]
    measured = counters.get(key)
    if not isinstance(measured, (int, float)):
        raise ProfilerUnavailable(f"{key} missing from profiler output")
    if expected_bytes <= 0:
        return None
    return measured / expected_bytes


# A profile may move this many times the planned bytes and still be
# accepted as the kernel's own graph. Symmetric with ``floor`` in spirit:
# a graph that moved ten times the plan is as certainly the wrong graph as
# one that moved a twentieth of it, and until 2026-09-10 only the low side
# was checked. The bytes then divided by a real total_time and produced a
# real-looking GB/s -- ten times too fast, and publishable.
CEILING = 2.0


def verify_profile_covers_plan(
    counters: typing.Mapping[str, typing.Any],
    direction: str,
    expected_bytes: int,
    floor: float = 0.5,
    ceiling: float = CEILING,
) -> None:
    """Reject a profile that did not come from the kernel we measured.

    The counters are checked against the work the plan describes, and a
    profile carrying less than ``floor`` or more than ``ceiling`` of the
    planned bytes is refused. Nothing downstream would otherwise notice:
    the bytes divide by a real ``total_time`` and produce a real-looking
    GB/s, which is worse than an error because it is publishable.

    **This function was dead for some time.** ``select_by_plan`` grew its
    own inline floor test and stopped calling it, while three docstrings,
    the generated reference and five tests went on crediting it with the
    2026-09-08 refusal. It is now the single gate ``select_by_plan``
    returns through, so the message the documents quote is the message the
    code produces.

    Being dead is also how it kept a one-sided bound: nothing exercised it
    against a graph larger than the plan, and the inline copy inherited
    the same gap.
    """
    coverage = plan_coverage(counters, direction, expected_bytes)
    if coverage is None:
        return
    if floor <= coverage <= ceiling:
        return
    key = {"read": "hbm_read_bytes", "write": "hbm_write_bytes"}[direction]
    moved = int(counters[key])
    side = "only " if coverage < floor else ""
    raise ProfilerUnavailable(
        f"profiled graph moved {side}{moved} bytes against a plan "
        f"of {expected_bytes} (coverage {coverage:.4g}) -- this is not the "
        "kernel that was measured, so its counters describe someone "
        "else's graph"
    )


def select_by_plan(candidates: typing.Sequence[str],
                   session_path: str,
                   direction: str,
                   expected_bytes: int,
                   floor: float = 0.5,
                   exact: float = 0.05) -> typing.Dict[str, typing.Any]:
    """Find which of ``candidates`` is the graph the kernel actually ran.

    The reason this exists: ``find_neffs`` ranks by mtime, and mtime picked
    wrong on both parts this suite has run on. inf2.xlarge 2026-09-07
    captured a graph that moved 2 bytes against an 8 GiB plan;
    trn1.2xlarge 2026-09-08 captured one that moved 4. Both times the
    kernel's own graph was in the candidate list and was not the newest
    entry in it, and both times the run degraded to the analytic figure
    with a declared source that has still never produced a number.

    So the plan check stops being only a rejector and becomes the selector:
    capture candidates and keep the one that accounts for the planned
    traffic.

    **The best match, not the first acceptable one.** Taking the first
    candidate over ``floor`` made the declared Score irreproducible: three
    runs of memory_read returned 256.17, 178.7 and 119.19 GB/s, and the
    last cleared a 0.5 floor while diverging from its own analytic
    cross-check by 56%. A partial match is not a coalesced version of the
    right graph, it is a different graph -- memory_read's own
    ``read_verified_ratio`` already confirms the kernel touched every
    planned byte, so the right NEFF reports coverage at 1.0 and anything
    well under it is somebody else's work.

    A candidate inside ``exact`` of 1.0 is taken immediately, because
    nothing can beat it and each extra attempt is a real NEFF replay.
    Otherwise the search continues and returns the closest to 1.0 it found,
    so a wrong first guess costs captures rather than correctness.

    Exhausting the cap raises with what every candidate reported, which is
    the diagnosis the single-guess version could never give: it said one
    graph was wrong, not that none was right.

    Returns the counters, the NEFF they came from, and how hard it looked.
    """
    if not candidates:
        raise ProfilerUnavailable("no NEFF candidates to profile")

    attempts = []
    best = None
    for position, neff in enumerate(candidates, start=1):
        try:
            counters = read_counters(neff, session_path)
        except ProfilerUnavailable as error:
            attempts.append(f"{os.path.basename(neff)}: capture failed ({error})")
            continue

        try:
            coverage = plan_coverage(counters, direction, expected_bytes)
        except ProfilerUnavailable as error:
            attempts.append(f"{os.path.basename(neff)}: {error}")
            continue

        found = {
            "counters": counters,
            "neff": neff,
            "plan_coverage": coverage,
            "candidates_tried": position,
            "candidates_available": len(candidates),
        }

        # No plan to check against: nothing can rank these, so the first
        # capture is the answer.
        if coverage is None:
            return found

        if abs(coverage - 1.0) <= exact:
            # This is the kernel's own graph. Stop paying for replays.
            return found

        if best is None or abs(coverage - 1.0) < abs(best["plan_coverage"] - 1.0):
            best = found
        attempts.append(
            f"{os.path.basename(neff)}: covered {coverage:.4g} of the plan"
        )

    if best is not None:
        # Nothing matched exactly, but something may still be within the
        # bounds. It is the closest to the plan of everything on the
        # machine, and the row's plan_coverage says how close, so a reader
        # can see that this was a near miss rather than a clean
        # identification.
        #
        # Checked through verify_profile_covers_plan rather than against
        # `floor` inline. The inline test compared one side only, so a
        # graph that moved ten times the planned bytes was accepted and
        # published as a bandwidth ten times too fast.
        try:
            verify_profile_covers_plan(
                best["counters"], direction, expected_bytes, floor)
            return best
        except ProfilerUnavailable as error:
            attempts.append(str(error))

    raise ProfilerUnavailable(
        f"none of {len(candidates)} candidate NEFF(s) moved the planned "
        f"{expected_bytes} bytes -- " + "; ".join(attempts[:4])
        + (f" (raise {CANDIDATES_ENV} to search further)"
           if len(candidates) >= _candidate_limit() else "")
    )


def bandwidth_gbps(counters: typing.Mapping[str, typing.Any],
                   direction: str = "read") -> float:
    """Compute HBM bandwidth exactly as the registry's formula declares.

    ``hbm_read_bytes / total_time / 1e9``. Both counters are per-execution,
    so this is the bandwidth of one NEFF run, not of the whole workload.
    """
    key = {"read": "hbm_read_bytes", "write": "hbm_write_bytes"}[direction]
    total_time = counters.get("total_time")
    measured = counters.get(key)

    if not isinstance(measured, (int, float)):
        raise ProfilerUnavailable(f"{key} missing from profiler output")
    if not isinstance(total_time, (int, float)) or total_time <= 0:
        raise ProfilerUnavailable("total_time missing or non-positive")
    return measured / total_time / 1e9
