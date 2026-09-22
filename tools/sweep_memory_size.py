#!/usr/bin/env python3
"""Does the HBM bandwidth Score depend on how many bytes the problem pins?

memory_read pins 8 GiB and memory_write 4 GiB. pantheongpu's workloads of the
same names pin nothing: they size themselves from ``--mem``, a share of
whatever the card has free, so an A100 moves about 39 GiB a pass and a T4
about 15. Both report bytes over seconds, which is why these rows are the ones
the two suites can compare at all -- but only if the rate does not move with
the size. A bandwidth that is flat from 1 GiB to 12 is the same quantity at
either suite's size. One that climbs or falls is a different quantity at each
size, and the comparison needs the curve beside it.

Nothing here has measured that. This tool does, by running the real harness --
reserved core, profiler capture, repeats, report file and all -- once per size,
with the workload's pinned ``bytes`` replaced for that process only. Going
through the harness rather than calling the kernel is the point: the figure
that would be compared is the harness's, from the declared source.

**The report tells the truth about what ran.** A row's ``Problem`` is
``dict(workload.problem)``, and the override replaces the workload the registry
resolves, so a 2 GiB run publishes ``bytes: 2147483648`` rather than the 8 GiB
it did not run. A size that is not a whole number of tiles is refused for the
same reason: the kernel would round it down and the row would name bytes it
never moved.

One process per size. The Neuron runtime holds its cores for the life of the
process, and the harness reserves a profiler core at start-up.

    python tools/sweep_memory_size.py                    # per-workload defaults
    SIZES_GIB=2,8 DURATION=60 REPEAT=2 python tools/sweep_memory_size.py
"""

import dataclasses
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kernels import registry, tiling  # noqa: E402

SWEEPABLE = ("memory_read", "memory_write")
GIB = 1024 ** 3

# Default sizes per workload, because the two kernels cannot hold the same
# buffers. A read allocates its source and nothing else, so it runs to 12 GiB
# on a 16 GiB NeuronCore. A write's output *is* its buffer, and a pass's
# destination is still resident when the next pass allocates its own -- the
# constraint memory_write's own comments record, and the reason it pins 4 GiB.
#
# The first sweep used 1,2,4,8,12 GiB for both, on trn1.2xlarge 2026-09-22.
# memory_read passed at every size, flat from 272.8 to 273.1 GB/s.
# memory_write passed to 4 GiB and failed at 8 and 12 with "Not enough Neuron
# memory on core 0 for size=8589934592" -- a constraint the kernel already
# documented, asked for by a default that had not read it, at twenty minutes
# of instance time per failed size.
DEFAULT_SIZES_GIB = {
    "memory_read": "1,2,4,8,12",
    "memory_write": "1,2,4",
}


def sizes_from(text: str):
    """``"1,2,4"`` -> byte counts. Whole GiB only, and at least one."""
    sizes = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        gib = int(part)
        if gib < 1:
            raise ValueError(f"sizes are whole GiB of at least 1, got {part!r}")
        sizes.append(gib * GIB)
    if not sizes:
        raise ValueError("no sizes given")
    return sizes


def swept_problem(workload, total_bytes: int) -> dict:
    """The workload's problem with ``bytes`` replaced, or a refusal.

    Refuses a size the kernel would round: the row publishes this dict as
    what ran.
    """
    if workload.name not in SWEEPABLE:
        raise ValueError(
            f"{workload.name} does not pin a byte count this tool can sweep; "
            f"choose from {', '.join(SWEEPABLE)}"
        )
    plan = tiling.tile_plan(total_bytes, str(workload.problem["dtype"]))
    if plan["actual_bytes"] != total_bytes:
        raise ValueError(
            f"{total_bytes} bytes is not a whole number of "
            f"{plan['tile_bytes']}-byte tiles; the kernel would move "
            f"{plan['actual_bytes']} and the row would name the larger figure"
        )
    problem = dict(workload.problem)
    problem["bytes"] = total_bytes
    return problem


def override(name: str, total_bytes: int):
    """Make the registry resolve ``name`` to the swept problem. This process only."""
    workload = registry.resolve(name)[0]
    swept = dataclasses.replace(workload, problem=swept_problem(workload, total_bytes))
    registry._BY_NAME[name] = swept
    return swept


def run_one(name: str, total_bytes: int, duration: int, repeat: int) -> int:
    """Child mode: override, then hand over to the real harness."""
    import pantheon_neuron

    override(name, total_bytes)
    return pantheon_neuron.main(
        ["--test", name, "--duration", str(duration), "--repeat", str(repeat)])


def _reports():
    import pantheon_neuron
    return set(glob.glob(os.path.join(pantheon_neuron.DATABASE_DIR, "*.json")))


def summarise(path: str, name: str) -> dict:
    """The fields of one report that answer the question."""
    with open(path, encoding="utf-8") as handle:
        row = next(r for r in json.load(handle)["test_results"]
                   if r["Test Name"] == name)
    repeats = row.get("Repeats") or {}
    return {
        "bytes": (row.get("Problem") or {}).get("bytes"),
        "status": row.get("Status"),
        "score": row.get("Score"),
        "method": row.get("Score Method"),
        "cv": repeats.get("cv"),
        "report": os.path.basename(path),
    }


def main() -> int:
    if len(sys.argv) == 6 and sys.argv[1] == "--one":
        _, _, name, total_bytes, duration, repeat = sys.argv
        return run_one(name, int(total_bytes), int(duration), int(repeat))

    override = os.environ.get("SIZES_GIB")
    duration = int(os.environ.get("DURATION", "60"))
    repeat = int(os.environ.get("REPEAT", "2"))
    names = [n.strip() for n in
             os.environ.get("WORKLOADS", ",".join(SWEEPABLE)).split(",")]

    sizes = {name: sizes_from(override or DEFAULT_SIZES_GIB[name]) for name in names}

    # Refuse the whole sweep up front rather than an hour in.
    for name in names:
        for total_bytes in sizes[name]:
            swept_problem(registry.resolve(name)[0], total_bytes)

    rows = []
    for name in names:
        for total_bytes in sizes[name]:
            before = _reports()
            print(f"=== {name} at {total_bytes // GIB} GiB, {duration}s x {repeat}",
                  flush=True)
            code = subprocess.call(
                [sys.executable, os.path.abspath(__file__), "--one", name,
                 str(total_bytes), str(duration), str(repeat)], cwd=ROOT)
            written = sorted(_reports() - before)
            if not written:
                rows.append((name, {"bytes": total_bytes, "status": f"exit {code}",
                                    "score": None, "method": None, "cv": None,
                                    "report": None}))
                continue
            rows.append((name, summarise(written[-1], name)))

    print(f"\n{'workload':14} {'GiB':>4} {'status':>8} {'GB/s':>10} {'cv':>8}  method")
    for name, row in rows:
        score = "-" if row["score"] is None else f"{row['score']:.2f}"
        cv = "-" if row["cv"] is None else f"{row['cv']:.4f}"
        print(f"{name:14} {row['bytes'] // GIB:>4} {row['status']!s:>8} "
              f"{score:>10} {cv:>8}  {row['method']}")
    return 0 if all(r["status"] == "PASS" for _, r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
