"""Read hardware counters out of ``neuron-profile``.

This is how a Score reaches its declared source. ``neuron-monitor`` streams
telemetry but has no HBM byte counters; only the profiler does, and only as
a per-execution capture.

Four environment traps are encoded here, each one observed during the
2026-08-26 probes and each one costing a round trip to find:

1. ``neuron-profile view`` exits with "$HOME is not defined" when HOME is
   unset. Anything running under SSM or a bare service manager hits this.
2. The Neuron bin directory must be on PATH, not merely referenced by
   absolute path -- the tools shell out to each other.
3. ``capture`` writes NTFF v6, which ``show-session`` and ``view`` can read.
   ``inspect`` writes v115, which the tooling shipped in the same AMI
   cannot read at all ("supported: 1 - 6"). Use capture.
4. The tools emit klog lines on stdout alongside the JSON, so the payload
   starts at the first ``{`` and everything before it is noise.

STATUS: UNTESTED. Written from captures taken by hand on an inf2.xlarge;
this code path has never run. See docs/neuron_counters.md for the raw
output it parses.
"""

import json
import os
import shutil
import subprocess
import typing


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


def find_neff(workdir: str) -> str:
    """Locate the compiled NEFF a capture needs.

    Searches the caller's workdir first, then the compiler's own default
    locations -- an @nki.jit kernel ignores compiler_workdir and writes to
    its own tree, so looking only where the caller asked finds nothing.

    Returns the most recently modified NEFF: a session may compile several
    (one per distinct input shape), and the newest is the one just run.
    """
    candidates = []
    for root in (workdir, *DEFAULT_WORKDIRS):
        if not root or not os.path.isdir(root):
            continue
        for base, _, names in os.walk(root):
            for name in names:
                if name.endswith(".neff"):
                    path = os.path.join(base, name)
                    try:
                        candidates.append((os.path.getmtime(path), path))
                    except OSError:
                        continue
    if candidates:
        return max(candidates)[1]

    for base, _, names in os.walk(workdir):
        for name in names:
            if name.endswith(".neff"):
                return os.path.join(base, name)
    raise ProfilerUnavailable(
        f"no .neff under {workdir} -- trace with compiler_workdir set, "
        "otherwise torch_neuronx removes it"
    )


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
