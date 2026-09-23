#!/usr/bin/env python3
"""Is this part performing like the reference fleet, or is something wrong?

This suite can measure a part and it can compare two vendors. It could not
answer the question it exists for -- *this card is slow, is it broken?* --
because nothing held a fresh report against the published reference. A
reader had to open `summary-trn1.json`, find the row, and divide.

So this divides. For every scored row it prints the ratio to the reference
median, and calls out the rows that cannot be explained by the spread the
reference itself shows.

    python tools/check_against_reference.py database/pantheon_neuron_report_*.json
    python tools/check_against_reference.py path/to/reports/

Exit status is 1 when a row falls outside its band, in either direction. A
Score *above* the reference is not good news here: the largest wrong number
this suite has ever produced was 14,513 GB/s from a kernel XLA had deleted.

**What the band is made of.** Both terms are measured, not chosen:

* **The reference hosts disagree with each other.** Three trn1.2xlarge and
  two inf2.xlarge ran the same commit on the same day; 21 of 23 trn1 rows
  agree within 0.2%, and `between_host_cv` in the summary records what each
  row's spread actually was. The band is `BAND_SPREADS` times that.
* **A floor, because most of those spreads are far too tight to be a
  tolerance.** `memory_read_agg` has a between-host CV of 0.002%, and a band
  of 0.006% would flag a healthy part for rounding. The floor is 5%.

The floor is also wide enough for the one systematic effect the dataset can
measure: running twelve times longer. Each host's own 3600 s figure against
its own 300 s figure, from the committed reports:

    memory_read, memory_write, and both aggregates   within 0.13%
    transformer_virus                                within 0.76%
    pcie_bandwidth              0.8212, 1.0002, 0.9544 on the three hosts

**The last line is a finding, not a tolerance to widen.** Running this check
over the committed dataset flags exactly one row: trn1-a's 3600 s
pcie_bandwidth, 17.9% below the 300 s reference, where the other two hosts
moved 0.02% and 4.6% over the same change. That is either a link that
degrades over an hour on one host or a figure too unstable to compare, and
neither is something a band should be stretched to cover. It is the first
thing this check found, which is the argument for having it.

**Duration is reported, not corrected for.** The reference ran 300 s per
workload; a row measured over a materially different window says so in its
note, so a reader can see when they are comparing across durations.

**What it refuses.** A ratio is only a comparison if both sides ran the same
thing, and this checks rather than assumes: the part must match, the unit
must match, and the pinned `Problem` must match key for key. A row that
fails any of those is reported as not comparable, with the difference shown.
That is the failure mode this repo keeps meeting -- a join on a name that
silently pairs two different quantities.
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kernels import registry  # noqa: E402

REFERENCE_DIR = os.path.join(ROOT, "data", "publication-2026-09-21")

# The reference pass: 300 s per workload, three repeats, medians across
# hosts. The other committed groups (3600, sweep) are not reference material
# -- they answer different questions and one of them is a single host.
REFERENCE_GROUP = "300x3"

# What "300x3" means, for the note that says when a row was measured over a
# different window. Not a threshold to compare against -- only a number to
# print beside the fresh row's own.
REFERENCE_SECONDS = 300.0

# How far a measured window may sit from the reference's before the note
# mentions it. Twice, because that is where "the same run, longer" becomes
# "a different run" -- and because the 3600 s reports are the case this
# exists to label.
WINDOW_FACTOR = 2.0

# See the module docstring: three times the row's own measured between-host
# spread, floored at 5%.
BAND_SPREADS = 3.0
BAND_FLOOR = 0.05


def reference(part):
    """The reference rows for a part, by workload name."""
    path = os.path.join(REFERENCE_DIR, f"summary-{part}.json")
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    rows = {row["workload"]: row for row in summary["rows"]
            if row["group"] == REFERENCE_GROUP and row["median_of_hosts"]}
    return summary["hosts"], rows


def part_of(report):
    """Which part wrote this report, or (None, why not)."""
    devices = report.get("devices")
    if not isinstance(devices, list) or not devices:
        return None, "the report names no devices"
    arches = {str(device.get("arch")) for device in devices
              if isinstance(device, dict)}
    if len(arches) != 1:
        return None, f"the report mixes parts: {', '.join(sorted(arches))}"
    arch = arches.pop()
    if arch not in registry.PART_PEAKS:
        # "mock" lands here, which is the point: a mock run has no Score and
        # must never be held against hardware as though it did.
        return None, f"no reference for a {arch!r} device"
    return arch, None


def band_for(row):
    """The fraction a Score may differ by, and where that fraction came from."""
    spread = row.get("between_host_cv")
    if isinstance(spread, (int, float)):
        widened = BAND_SPREADS * spread
        if widened > BAND_FLOOR:
            return widened, (f"{BAND_SPREADS:g}x the reference's own "
                             f"between-host spread of {spread:.2%}")
    return BAND_FLOOR, f"the {BAND_FLOOR:.0%} floor"


def _problem_difference(ours, theirs):
    """How two pinned problems differ, in a sentence, or None."""
    ours, theirs = dict(ours or {}), dict(theirs or {})
    if ours == theirs:
        return None
    keys = sorted(set(ours) | set(theirs))
    differing = [f"{key}={ours.get(key, '-')!r} against {theirs.get(key, '-')!r}"
                 for key in keys if ours.get(key) != theirs.get(key)]
    return "; ".join(differing)


def window_note(row):
    """Say when this row's window is not the reference's, or nothing."""
    measured = row.get("Duration (s)")
    if not isinstance(measured, (int, float)) or measured <= 0:
        return ""
    if (measured > REFERENCE_SECONDS * WINDOW_FACTOR
            or measured < REFERENCE_SECONDS / WINDOW_FACTOR):
        return (f" [measured over {measured:.0f}s against the reference's "
                f"{REFERENCE_SECONDS:.0f}s]")
    return ""


def compare(row, reference_row):
    """(verdict, ratio, note) for one scored row against its reference."""
    if reference_row is None:
        return "no reference", None, (
            "the reference pass has no median for this workload")
    if row.get("Unit") != reference_row.get("unit"):
        return "not comparable", None, (
            f"unit {row.get('Unit')!r} against the reference's "
            f"{reference_row.get('unit')!r}")
    difference = _problem_difference(row.get("Problem"), reference_row.get("problem"))
    if difference:
        return "not comparable", None, f"a different pinned problem: {difference}"

    ratio = row["Score"] / reference_row["median_of_hosts"]
    band, because = band_for(reference_row)
    window = window_note(row)
    if abs(ratio - 1.0) <= band:
        return "ok", ratio, because + window
    direction = "below" if ratio < 1.0 else "above"
    return direction, ratio, (
        f"{abs(ratio - 1.0):.1%} {direction} the reference, outside {because} "
        f"({band:.1%})" + window)


def check(report):
    """Every scored row of one report, against the reference for its part."""
    part, why_not = part_of(report)
    if part is None:
        return None, why_not, []
    hosts, rows = reference(part)
    results = []
    for row in report.get("test_results") or []:
        score = row.get("Score")
        if row.get("Status") != "PASS" or not isinstance(score, (int, float)):
            continue
        name = row.get("Test Name")
        verdict, ratio, note = compare(row, rows.get(name))
        results.append({"workload": name, "score": score, "unit": row.get("Unit"),
                        "reference": (rows.get(name) or {}).get("median_of_hosts"),
                        "verdict": verdict, "ratio": ratio, "note": note})
    return part, hosts, results


def _figure(value):
    """A Score at a width a column can hold; quantized_gemm reports 1.8e13."""
    return f"{value:.4g}" if abs(value) >= 1e6 else f"{value:.4f}"


def _reports(argv):
    paths = []
    for argument in argv:
        if os.path.isdir(argument):
            paths += glob.glob(os.path.join(argument, "**", "*.json"), recursive=True)
        else:
            paths += glob.glob(argument)
    return sorted(set(paths))


def main(argv) -> int:
    paths = _reports(argv or [os.path.join(ROOT, "database")])
    if not paths:
        print("no reports found; pass a report file or a directory")
        return 2

    failed, checked = 0, 0
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(report, dict) or "test_results" not in report:
            continue
        part, hosts, results = check(report)
        if part is None:
            print(f"{os.path.basename(path)}: not checked -- {hosts}\n")
            continue
        checked += 1
        print(f"{os.path.basename(path)}: {part} against "
              f"{', '.join(hosts)} ({REFERENCE_GROUP})")
        print(f"  {'workload':26} {'score':>14} {'reference':>14} {'ratio':>8}  note")
        for result in results:
            ratio = f"{result['ratio']:.4f}" if result["ratio"] else "--"
            shown = _figure(result["reference"]) if result["reference"] else "--"
            mark = " " if result["verdict"] == "ok" else "!"
            print(f" {mark}{result['workload']:26} {_figure(result['score']):>14} "
                  f"{shown:>14} {ratio:>8}  {result['note'][:78]}")
            if result["verdict"] in ("below", "above"):
                failed += 1
        print()

    if not checked:
        print("no report carried devices with a published reference")
        return 2
    if failed:
        print(f"{failed} row(s) outside the reference band.")
        return 1
    print("every comparable row is within the reference band.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
