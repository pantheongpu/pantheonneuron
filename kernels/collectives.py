"""NeuronLink collectives, measured with AWS's own benchmark.

Score: **GB/s**, from ``nccom-test`` busbw as the registry declares. This is
the fourth and last Score source in the suite, and the only one that is an
external benchmark rather than a counter: AWS ships `nccom-test` alongside
the runtime, it reports bus bandwidth directly, and reimplementing a
collective benchmark to get a worse version of the same number would be
indefensible.

``all_reduce``    sweeps sizes and averages busbw, because a collective's
                  efficiency is size-dependent -- small messages are
                  latency-bound and large ones bandwidth-bound, and one
                  size would characterise only one regime.
``p2p_thrasher``  sends between two devices at a fixed large size, which
                  isolates the link itself from the reduction arithmetic
                  all_reduce also performs.

**Both need two or more devices and will skip on every part this account can
currently rent.** `trn1.32xlarge` is the smallest instance with
device-to-device NeuronLink and needs 128 vCPUs against a granted 64, so
these are written against the documented output format rather than against
observed output. The 2026-08-26 probe did run `nccom-test` on a single
inf2, where it reported 50.66 GB/s core-to-core -- which is *not* NeuronLink
and must not be recorded as such.

STATUS: UNTESTED ON HARDWARE, and unusually so: unlike every other kernel
here, it cannot be tested until the quota lands.
"""

import os
import re
import shutil
import subprocess
import typing


NEURON_BIN = "/opt/aws/neuron/bin"
_TIMEOUT = 900


class CollectivesUnavailable(RuntimeError):
    """Raised when nccom-test cannot produce a bandwidth figure."""


def _environment() -> typing.Dict[str, str]:
    """Same PATH discipline the profiler needs: the tools shell out."""
    env = dict(os.environ)
    env.setdefault("HOME", "/root")
    path = env.get("PATH", "")
    if NEURON_BIN not in path.split(os.pathsep):
        env["PATH"] = os.pathsep.join([NEURON_BIN, path]) if path else NEURON_BIN
    return env


def available() -> bool:
    return shutil.which("nccom-test", path=_environment()["PATH"]) is not None


# nccom-test prints a table whose last numeric column is busbw in GB/s, with
# one row per message size. Matching the size and the trailing float keeps
# the parse anchored to the row shape rather than to column positions, which
# differ between versions.
_ROW = re.compile(
    r"^\s*(?P<bytes>\d+)\s+.*?(?P<busbw>\d+\.\d+)\s*$", re.MULTILINE
)


# A line that starts with a size and has at least one more field. Every
# real data row looks like this; only some of them also end in a float.
_DATA_LINE = re.compile(r"^\s*\d+\s+\S", re.MULTILINE)


def parse_busbw(output: str) -> typing.List[typing.Tuple[int, float]]:
    """Extract (bytes, busbw GB/s) rows from nccom-test output.

    Returns every row rather than a single figure: a sweep's shape is the
    interesting part, and averaging is the caller's decision.
    """
    rows = []
    for match in _ROW.finditer(output):
        size = int(match.group("bytes"))
        busbw = float(match.group("busbw"))
        if size > 0 and busbw > 0:
            rows.append((size, busbw))
    return rows


def unparsed_rows(output: str) -> int:
    r"""Data-shaped lines the bandwidth pattern did not match.

    ``_ROW`` requires the trailing field to be ``\d+\.\d+``. A version of
    nccom-test that prints an integer bus bandwidth on some rows -- or
    puts a different column last -- drops those rows silently, and the
    average is then over a biased subset of the sweep with nothing saying
    so.

    This module cannot be run on any part this account can reach, so the
    regex is deliberately not loosened to guess at a format nobody here
    has seen. What can be done without guessing is to notice the mismatch:
    a caller that parsed 3 of 9 data lines is not looking at a sweep.
    """
    return max(0, len(_DATA_LINE.findall(output)) - len(parse_busbw(output)))


def verify_sweep_was_fully_parsed(output: str) -> typing.Optional[str]:
    """Flag a sweep where the parse dropped rows."""
    dropped = unparsed_rows(output)
    if not dropped:
        return None
    parsed = len(parse_busbw(output))
    return (
        f"parsed {parsed} of {parsed + dropped} data rows -- the bandwidth "
        "pattern did not match the rest, so this average is over part of "
        "the sweep"
    )


def _run(args: typing.Sequence[str]) -> str:
    binary = shutil.which("nccom-test", path=_environment()["PATH"])
    if binary is None:
        raise CollectivesUnavailable(
            f"nccom-test not found; expected it in {NEURON_BIN}"
        )
    try:
        completed = subprocess.run(
            [binary, *args], capture_output=True, text=True,
            timeout=_TIMEOUT, env=_environment(), check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CollectivesUnavailable(f"nccom-test failed: {error}") from error
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise CollectivesUnavailable(
            f"nccom-test exited {completed.returncode}: "
            f"{tail[-1] if tail else 'no output'}"
        )
    return completed.stdout


def run_all_reduce(problem: typing.Mapping[str, typing.Any],
                   devices: typing.Sequence) -> dict:
    """Sweep all-reduce sizes and average the bus bandwidth."""
    bytes_min = int(problem["bytes_min"])
    bytes_max = int(problem["bytes_max"])
    ranks = len(devices)

    output = _run([
        "-r", str(ranks),
        "-b", str(bytes_min),
        "-e", str(bytes_max),
        "-f", "2",              # double the size each step
        "-n", "20",
        "-c", "all_reduce",
        "-d", str(problem.get("dtype", "fp32")),
    ])
    rows = parse_busbw(output)
    if not rows:
        raise CollectivesUnavailable(
            "nccom-test produced no parsable bandwidth rows"
        )

    average = sum(busbw for _, busbw in rows) / len(rows)
    return {
        "busbw_gbps": average,
        "sweep": {str(size): busbw for size, busbw in rows},
        "ranks": ranks,
        "score_method": "nccom-test",
        "analytic_basis": "nccom-test all_reduce busbw, averaged over the sweep",
        "warning": (verify_sweep_was_fully_parsed(output)
                    or verify_sweep_covers_both_regimes(
                        rows, bytes_min, bytes_max)),
    }


def run_p2p(problem: typing.Mapping[str, typing.Any],
            devices: typing.Sequence) -> dict:
    """Send between devices at one large size and report bus bandwidth."""
    size = int(problem["bytes"])
    ranks = len(devices)

    output = _run([
        "-r", str(ranks),
        "-b", str(size),
        "-e", str(size),
        "-n", "20",
        "-c", "sendrecv",
        "-d", str(problem.get("dtype", "fp32")),
    ])
    rows = parse_busbw(output)
    if not rows:
        raise CollectivesUnavailable(
            "nccom-test produced no parsable bandwidth rows"
        )

    return {
        "busbw_gbps": rows[-1][1],
        "bytes": size,
        "ranks": ranks,
        "score_method": "nccom-test",
        "analytic_basis": "nccom-test sendrecv busbw",
        "warning": None,
    }


def verify_sweep_covers_both_regimes(
    rows: typing.Sequence[typing.Tuple[int, float]],
    bytes_min: int,
    bytes_max: int,
) -> typing.Optional[str]:
    """Check the sweep spans the range the pinned problem asked for.

    An average over rows that all sit at one end describes one regime while
    claiming to describe the collective. Small messages are latency-bound
    and large ones bandwidth-bound; an average of only the small end is a
    latency figure wearing a bandwidth label.
    """
    if not rows:
        return "no rows to average"
    sizes = [size for size, _ in rows]
    if min(sizes) > bytes_min * 2:
        return (
            f"sweep started at {min(sizes)} bytes rather than {bytes_min} -- "
            "the small-message regime is missing from this average"
        )
    if max(sizes) < bytes_max // 2:
        return (
            f"sweep stopped at {max(sizes)} bytes rather than {bytes_max} -- "
            "the bandwidth-bound regime is missing from this average"
        )
    return None
