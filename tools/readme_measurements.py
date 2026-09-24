#!/usr/bin/env python3
"""Keep the README's Measured column equal to the published dataset.

The Kernel status table is the first set of numbers a reader of this repo
meets, and every figure in it was typed. Checked against the 2026-09-21
pass -- three trn1.2xlarge hosts, 300 s x 3 repeats each, medians across
hosts -- 22 of 24 agreed within 0.3%, and two did not:

    pcie_bandwidth             4.01 GB/s     dataset 3.678    9.0% high
    allocation_fragmentation   2,265.7       dataset 2,397.2  5.5% low

Neither said which run it came from, so a reader had no way to know they
were looking at a different measurement from the one the dataset publishes.
Three more named a unit the registry does not use -- "events/s" for
allocation-events/s, "vectors/s" for embedding-vectors/s, "TOPS" for
quantized-ops/s -- and a comparison joins on (Test Name, Unit), so those
rows would not have joined.

This renders the column from `summary-trn1.json` in the registry's own unit,
and checks it. It is the same move as `compare_with_gpu.py --render` for the
dataset README: a figure a reader will quote is generated, not transcribed.

    python tools/readme_measurements.py            # check; exit 1 on drift
    python tools/readme_measurements.py --write    # rewrite the column
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kernels import registry  # noqa: E402

README = os.path.join(ROOT, "README.md")
SUMMARY = os.path.join(ROOT, "data", "publication-2026-09-21", "summary-trn1.json")
GROUP = "300x3"
HEADER = "| Workload | Status | Measured | Score source |"


def display(value):
    """One figure, at a precision the table can hold and a reader can quote."""
    if value >= 1e6:
        return f"{value:.4g}"
    if value >= 1000:
        return f"{value:,.1f}"
    return f"{value:.2f}"


def read_readme():
    with open(README, encoding="utf-8") as handle:
        return handle.read()


def reference():
    """(figure, unit, score methods) per workload from the reference pass."""
    with open(SUMMARY, encoding="utf-8") as handle:
        rows = json.load(handle)["rows"]
    return {row["workload"]: row for row in rows if row["group"] == GROUP}


def expected_cell(name, rows):
    """What the Measured cell should say for this workload, or '—'."""
    row = rows.get(name)
    if not row or not row.get("median_of_hosts"):
        return "—"
    unit = registry.resolve(name)[0].unit
    return f"{display(row['median_of_hosts'])} {unit}"


def _table_span(text):
    start = text.index(HEADER)
    end = text.index("\n\n", start)
    return start, end


def cells(text):
    """(line index within table, workload, measured cell, source cell)."""
    start, end = _table_span(text)
    out = []
    for index, line in enumerate(text[start:end].splitlines()):
        if index < 2:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        out.append((index, parts[0].strip("`"), parts[2], parts[3]))
    return out


def drift(text=None):
    """Every row whose Measured or Score source cell disagrees with the data."""
    text = text if text is not None else read_readme()
    rows = reference()
    found = []
    for _index, name, measured, source in cells(text):
        want = expected_cell(name, rows)
        if measured != want:
            found.append((name, "Measured", measured, want))
        methods = (rows.get(name) or {}).get("score_methods") or []
        if len(methods) == 1 and methods[0] not in ("None", None):
            shown = source.strip("`")
            if shown != methods[0]:
                found.append((name, "Score source", source, methods[0]))
    return found


def rewrite(text):
    """The README with the Measured column regenerated."""
    rows = reference()
    start, end = _table_span(text)
    lines = text[start:end].splitlines()
    for index, name, _measured, _source in cells(text):
        parts = lines[index].strip().strip("|").split("|")
        parts[2] = f" {expected_cell(name, rows)} "
        lines[index] = "|" + "|".join(parts) + "|"
    return text[:start] + "\n".join(lines) + text[end:]


def main(argv) -> int:
    with open(README, encoding="utf-8") as handle:
        text = handle.read()
    if "--write" in argv:
        updated = rewrite(text)
        with open(README, "w", encoding="utf-8") as handle:
            handle.write(updated)
        print(f"rewrote the Measured column from {os.path.relpath(SUMMARY, ROOT)}")
        return 0
    found = drift(text)
    for name, column, shown, want in found:
        print(f"{name:26} {column:13} README {shown!r}, dataset {want!r}")
    if found:
        print("\nrun: python tools/readme_measurements.py --write")
        return 1
    print("the README's Measured column matches the published dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
