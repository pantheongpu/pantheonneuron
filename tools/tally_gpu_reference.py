#!/usr/bin/env python3
"""Tally pantheongpu's published memory figures into a committed reference.

The cross-platform comparison needs a number from the other suite, and this
repo has been bitten twice by taking one from memory. `PANTHEONGPU_UNITS` was
a transcription that went stale without anyone noticing (#20), and the
publication README's GPU column was typed from medians computed in a shell
with nothing to check it against.

So the figures are tallied here, once, from pantheongpu's own report database,
and committed as `data/publication-2026-09-21/gpu-reference.json`. The
comparison tool reads that file; nothing reads a number out of prose.

**This tool needs the pantheongpu report database**, which lives in the
website repository and is not vendored here -- so it does not run in CI, and
its output is committed instead. Point it at that checkout:

    python tools/tally_gpu_reference.py ~/tools/pantheon/pantheongpu_website

Published peaks are not tallied: they come from each vendor's own document,
are entered in PUBLISHED_PEAKS below with the source, and were read on
2026-09-22. A part with no published figure gets None and is reported as such
rather than given a borrowed one -- see the A10G.
"""

import collections
import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The rows a device-level comparison uses. pantheongpu's memory_read is a
# whole GPU; its `_agg` is the same kernel over a stress data pattern, which
# its own reports put within 0.3%. See docs/cross_platform_comparability.md.
ROWS = ("memory_read", "memory_write", "memory_read_agg", "memory_write_agg")

# Each read from the vendor's own document on 2026-09-22. None means the
# vendor publishes no figure for that part, which is a fact about the part
# and not a gap to fill with a neighbour's number.
PUBLISHED_PEAKS = {
    "NVIDIA A10": (600.0, "NVIDIA A10 datasheet, 'GPU Memory Bandwidth 600 GB/s'"),
    "NVIDIA A10G": (None, "no published figure: an AWS-specific part (300 W against "
                          "the A10's 150 W); NVIDIA documents the A10, AWS gives only "
                          "its 24 GB"),
    "NVIDIA L40S": (864.0, "NVIDIA L40S product page, 'Memory Bandwidth 864GB/s'"),
    "NVIDIA A100-SXM4-40GB": (
        1555.0, "NVIDIA A100 datasheet, SXM 40GB column, '1,555GB/s'"),
    "NVIDIA A100-SXM4-80GB": (
        2039.0, "NVIDIA A100 datasheet, SXM 80GB column, '2,039GB/s'"),
    "NVIDIA H100 PCIe": (
        2000.0, "NVIDIA H100 PCIe product brief PB-11133 Table 2, 'Peak memory "
                "bandwidth 2,000 GB/s' (its own 1,593 MHz x 5,120 bits gives 2,039)"),
    "NVIDIA H100 80GB HBM3": (
        3350.0, "NVIDIA H100 product page, H100 SXM 'GPU Memory Bandwidth 3.35TB/s'"),
    "NVIDIA L4": (300.0, "NVIDIA L4 product page, 'GPU memory bandwidth 300GB/s'"),
    "NVIDIA GH200 480GB": (
        None, "not established for this variant: the product page quotes 10 TB/s for "
              "the two-superchip NVL2 and 'up to 288GB', while this part pairs 96GB "
              "HBM3 with 480GB of CPU memory. A single-GPU figure was not found on "
              "the vendor's page, so none is used"),
}


def tally(database_dir):
    """Median of each part's published Scores, per row, with the counts."""
    parts = collections.defaultdict(lambda: collections.defaultdict(list))
    versions = collections.defaultdict(collections.Counter)
    reports = collections.Counter()
    pattern = os.path.join(database_dir, "**", "*.json")
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(report, dict) or not isinstance(report.get("test_results"), list):
            continue
        static = report.get("gpu_static_info")
        first = static[0] if isinstance(static, list) and static else {}
        name = first.get("name") if isinstance(first, dict) else None
        if not name:
            continue
        reports[name] += 1
        versions[name][str(report.get("pantheon_version"))] += 1
        for row in report["test_results"]:
            if not isinstance(row, dict) or row.get("Test Name") not in ROWS:
                continue
            score, unit = row.get("Score"), row.get("Unit")
            if isinstance(score, (int, float)) and score > 0 and unit == "GB/s":
                parts[name][row["Test Name"]].append(float(score))

    reference = {}
    for name, rows in sorted(parts.items()):
        peak, source = PUBLISHED_PEAKS.get(name, (None, "not looked up"))
        reference[name] = {
            "reports": reports[name],
            "versions": dict(sorted(versions[name].items())),
            "published_peak_gbps": peak,
            "published_peak_source": source,
            "rows": {
                row: {"median_gbps": round(statistics.median(values), 3),
                      "samples": len(values)}
                for row, values in sorted(rows.items())
            },
        }
    return reference


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 2
    database = argv[0]
    if not os.path.isdir(database):
        print(f"not a directory: {database}")
        return 2
    reference = tally(database)
    if not reference:
        print(f"no pantheongpu reports found under {database}")
        return 1
    payload = {
        "what": ("Median memory bandwidth pantheongpu published per GPU, tallied from "
                 "its report database, with each part's published peak read from the "
                 "vendor's own document. The comparison tool reads this file so no "
                 "figure is transcribed into prose."),
        "tallied": "2026-09-22",
        "rows_tallied": list(ROWS),
        "parts": reference,
    }
    destination = os.path.join(ROOT, "data", "publication-2026-09-21", "gpu-reference.json")
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {destination}: {len(reference)} parts")
    for name, entry in sorted(reference.items()):
        read = entry["rows"].get("memory_read", {})
        print(f"  {name:38} reports={entry['reports']:5} "
              f"read={read.get('median_gbps')} peak={entry['published_peak_gbps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
