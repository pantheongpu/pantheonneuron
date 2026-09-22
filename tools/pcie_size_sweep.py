#!/usr/bin/env python3
"""Which transfer size should pcie_bandwidth pin?

`pcie_bandwidth` pins 1 GiB and scores 3.68 GB/s. Its own docstring records why
that is the degraded regime, from a size sweep on trn1.2xlarge 2026-09-10:

    MiB     h2d      d2h
      1    5.57     2.92
     16   11.55     2.83
     64   11.22     0.88
    256   10.09     1.01
   1024    7.37     1.09

d2h holds ~2.9 GB/s to 16 MiB and collapses past it -- the signature of a
staging buffer -- and h2d peaks at 16 MiB too, at 11.55 GB/s against roughly 16
GB/s theoretical for the Gen4 x8 link the probes recorded. So 16 MiB measures
the link and 1 GiB measures what a large transfer costs through the staging
path. The kernel calls the choice "a registry decision and is left open".

This re-measures the curve through the current kernel so the decision rests on
fresh data rather than a fortnight-old note, and so whichever size is pinned
has a sweep beside it showing what the neighbours do.

Each size runs in its own process: the Neuron runtime holds its cores for the
life of a process, and this way one failure cannot take the rest of the sweep
with it.

    python tools/pcie_size_sweep.py                 # 1..1024 MiB, 10 s per size
    SIZES_MIB=16,1024 SECONDS=20 python tools/pcie_size_sweep.py
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kernels import pcie_bandwidth, registry  # noqa: E402

MIB = 1024 ** 2
DEFAULT_SIZES_MIB = "1,4,16,32,64,256,1024"


def sizes_from(text):
    sizes = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        mib = int(part)
        if mib < 1:
            raise ValueError(f"sizes are whole MiB of at least 1, got {part!r}")
        sizes.append(mib * MIB)
    if not sizes:
        raise ValueError("no sizes given")
    return sizes


def swept_problem(total_bytes):
    """The pinned problem with `bytes` replaced, checked against the kernel."""
    workload = registry.resolve("pcie_bandwidth")[0]
    problem = dict(workload.problem)
    problem["bytes"] = total_bytes
    pcie_bandwidth.transfer_plan(problem)      # refuses a size the kernel cannot run
    return problem


def run_one(total_bytes, seconds):
    """Child mode: one size, printing the per-direction result as JSON."""
    result = pcie_bandwidth.run(swept_problem(total_bytes), seconds)
    per = result.get("per_direction") or {}
    print("RESULT " + json.dumps({
        "bytes": total_bytes,
        "combined_gbps": round(result.get("analytic_gbps") or 0.0, 3),
        "h2d_gbps": round((per.get("h2d") or {}).get("gbps") or 0.0, 3),
        "d2h_gbps": round((per.get("d2h") or {}).get("gbps") or 0.0, 3),
        "h2d_passes": (per.get("h2d") or {}).get("passes"),
        "d2h_passes": (per.get("d2h") or {}).get("passes"),
        "warning": result.get("warning"),
    }))
    return 0


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--one":
        return run_one(int(sys.argv[2]), int(sys.argv[3]))

    sizes = sizes_from(os.environ.get("SIZES_MIB", DEFAULT_SIZES_MIB))
    seconds = int(os.environ.get("SECONDS", "10"))
    for total in sizes:
        swept_problem(total)                   # refuse the whole sweep up front

    print(f"{'MiB':>6} {'h2d GB/s':>9} {'d2h GB/s':>9} {'combined':>9}  "
          f"{'h2d passes':>10} {'d2h passes':>10}  warning", flush=True)
    rows = []
    for total in sizes:
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--one", str(total), str(seconds)],
            cwd=ROOT, capture_output=True, text=True)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if not line:
            tail = (out.stderr or out.stdout).strip().splitlines()[-1:] or ["no output"]
            print(f"{total // MIB:>6} {'-':>9} {'-':>9} {'-':>9}  "
                  f"{'-':>10} {'-':>10}  failed: {tail[0][:120]}", flush=True)
            continue
        row = json.loads(line[len("RESULT "):])
        rows.append(row)
        print(f"{row['bytes'] // MIB:>6} {row['h2d_gbps']:>9.2f} {row['d2h_gbps']:>9.2f} "
              f"{row['combined_gbps']:>9.2f}  {row['h2d_passes']!s:>10} "
              f"{row['d2h_passes']!s:>10}  {row['warning'] or ''}", flush=True)

    if rows:
        best_h2d = max(rows, key=lambda r: r["h2d_gbps"])
        best_d2h = max(rows, key=lambda r: r["d2h_gbps"])
        pinned = next((r for r in rows if r["bytes"] == 1 << 30), None)
        print(f"\nh2d peaks at {best_h2d['bytes'] // MIB} MiB "
              f"({best_h2d['h2d_gbps']} GB/s); d2h peaks at "
              f"{best_d2h['bytes'] // MIB} MiB ({best_d2h['d2h_gbps']} GB/s).", flush=True)
        if pinned:
            print(f"The pinned 1024 MiB reads h2d {pinned['h2d_gbps']}, "
                  f"d2h {pinned['d2h_gbps']}, combined {pinned['combined_gbps']} GB/s.",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
