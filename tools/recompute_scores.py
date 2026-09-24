#!/usr/bin/env python3
"""Recompute every Score from its own row, the way the registry says it can be.

`_provenance` calls a declared counter list "the row's promise that the Score
can be recomputed from it". The promise was never tested. Applying it to the
2026-09-21 dataset, 21 of 23 scored rows reproduce their Score exactly -- and
two do not:

    memory_read   declared hbm_read_bytes / total_active_time / 1e9
                  row published profiler_total_time_s = total_time
                  recomputed 255.96 against a published 272.94, 6.2% short

The row named `total_active_time` as its basis and published the other
counter beside it, so a reader following the formula divided by a window that
includes a 2.08 ms startup in which no byte moves -- landing exactly on the
biased "over total_time" figure `bandwidth_gbps` documents and rejects. Fixed
by publishing `profiler_active_time_s`, the denominator the Score used; this
tool is what would have caught it.

    python tools/recompute_scores.py                      # the committed dataset
    python tools/recompute_scores.py path/to/reports      # any report tree

Exit status is 1 when a row that should recompute does not.
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kernels import registry  # noqa: E402

# How close a recomputation has to land. The arithmetic is the same
# arithmetic, so this is rounding in the published figures (Scores are stored
# to four decimals), not measurement noise.
TOLERANCE = 0.005

# Formula terms whose name in the row differs from the name in the declared
# formula, and why. Each is a real key in the report, not a synonym invented
# here.
ALIASES = {
    # The aggregates sum what their workers moved and divide by the longest
    # worker's loop; the row carries that sum and that span under its own
    # names. See kernels/memory_agg.py.
    "sum(bytes_requested over cores)": "bytes_moved",
    "sum(bytes_written over cores)": "bytes_moved",
    "max(elapsed_s over cores)": "worker_span_s",
    # The profiled window the Score divided by. The row also carries
    # total_time, which is a different and longer window.
    "total_active_time": "profiler_active_time_s",
}

# Rows a single report cannot carry the parts for, with the reason. Not a
# place to put a row that merely fails: every entry here says why no report
# could ever reproduce it.
CANNOT_RECOMPUTE = {
    "all_reduce": "nccom-test reports a bus bandwidth averaged over a size "
                  "sweep the row does not carry; and this has never run",
    "p2p_thrasher": "same: nccom-test's own figure, and this has never run",
}


def _first_core(series):
    """The monitor reports per NeuronCore; a single-core workload uses one."""
    if isinstance(series, dict):
        for value in series.values():
            if isinstance(value, dict):
                return value
    return {}


class MissingProvenance(Exception):
    """A versioned report lacks a key its schema says the row carries."""


def recompute(workload, row, schema=None):
    """(value, how) recomputed from this row, or (None, why not)."""
    source = workload.score_source
    if workload.name in CANNOT_RECOMPUTE:
        return None, CANNOT_RECOMPUTE[workload.name]
    if source is None or not source.formula:
        return None, "no declared formula"

    measurement = row.get("Measurement") or {}
    telemetry = row.get("Telemetry") or {}

    # An even number of repeats has no median repeat, so the Score is an
    # average of two executions and the harness withholds provenance rather
    # than attributing it to one of them (_median_provenance). Nothing is
    # wrong with such a row; it simply cannot be recomputed from itself.
    repeats = row.get("Repeats") or {}
    scored = repeats.get("scored")
    if not measurement and isinstance(scored, int) and scored > 1 and scored % 2 == 0:
        return None, (f"provenance withheld by design: {scored} repeats have no "
                      f"median repeat, so the Score belongs to no single execution")
    formula = source.formula.split("#")[0].strip()

    def value(name):
        key = ALIASES.get(name, name)
        found = measurement.get(key, telemetry.get(key))
        return found if isinstance(found, (int, float)) else None

    if source.source == registry.MONITOR:
        if "effective_flops" in formula:
            mean = _first_core(telemetry.get("effective_flops")).get("mean")
            if isinstance(mean, (int, float)):
                return mean / 1e12, "mean(effective_flops) / 1e12"
            return None, "the row carries no effective_flops mean"
        rate = telemetry.get("executions_per_s")
        if isinstance(rate, (int, float)):
            return rate, "executions_per_s, the monitor's per-period rate"
        return None, "the row carries no executions_per_s"

    if source.source == registry.PROFILER:
        counter = formula.split("/")[0].strip()
        bytes_moved, window = value(counter), value("total_active_time")
        if bytes_moved is None:
            return None, f"the row carries no {counter}"
        if window is None:
            # A report's schema_version says whether the key could be there
            # at all. Before it existed this was a guess from the date.
            if schema is None:
                return None, ("the row carries no profiler_active_time_s: the "
                              "report has no schema_version, so it predates "
                              "that key and the window the Score divided by "
                              "is not in it")
            raise MissingProvenance(
                f"schema {schema} report, and the row carries no "
                "profiler_active_time_s: the kernel stopped publishing the "
                "window its Score divided by")
        return bytes_moved / window / 1e9, f"{counter} / profiler_active_time_s / 1e9"

    # Everything else divides a counted quantity by a measured window.
    parts = [p.strip() for p in formula.split("/")]
    numerator = value(parts[0]) if parts else None
    if numerator is None:
        return None, f"the row carries no {parts[0] if parts else 'numerator'}"
    denominator = value(parts[1]) if len(parts) > 1 else None
    if denominator is None and len(parts) > 1:
        return None, f"the row carries no {parts[1]}"
    result = numerator / denominator
    if len(parts) > 2 and parts[2] == "1e9":
        result /= 1e9
    return result, f"{parts[0]} / {parts[1]}" + (" / 1e9" if len(parts) > 2 else "")


def check(report_paths):
    """Every scored row in these reports, recomputed. Returns the rows."""
    by_name = {w.name: w for w in registry.WORKLOADS}
    results = []
    for path in report_paths:
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, ValueError):
            continue
        for row in report.get("test_results") or []:
            workload = by_name.get(row.get("Test Name"))
            score = row.get("Score")
            if workload is None or row.get("Status") != "PASS":
                continue
            if not isinstance(score, (int, float)) or score <= 0:
                continue
            defect = False
            try:
                got, how = recompute(workload, row, report.get("schema_version"))
            except MissingProvenance as error:
                got, how, defect = None, str(error), True
            ratio = got / score if got else None
            results.append({
                "report": os.path.basename(path),
                "workload": workload.name,
                "score": score,
                "recomputed": got,
                "ratio": ratio,
                "how": how,
                "agrees": ratio is not None and abs(ratio - 1.0) <= TOLERANCE,
                # Not recomputable because the row broke its schema's
                # promise, which is a failure -- as against not recomputable
                # because the report predates the promise, which is not.
                "defect": defect,
            })
    return results


def main(argv) -> int:
    where = argv[0] if argv else os.path.join(ROOT, "data", "publication-2026-09-21")
    paths = sorted(glob.glob(os.path.join(where, "**", "*.json"), recursive=True))
    results = check(paths)
    if not results:
        print(f"no scored rows found under {where}")
        return 1

    seen, disagreed, unrecomputable = {}, [], {}
    for row in results:
        seen.setdefault(row["workload"], row)
        if row["defect"]:
            disagreed.append(row)
        elif row["ratio"] is None:
            unrecomputable.setdefault(row["workload"], row["how"])
        elif not row["agrees"]:
            disagreed.append(row)

    print(f"{len(results)} scored rows in {len(paths)} reports under {where}\n")
    print(f"{'workload':26} {'score':>14} {'recomputed':>14} {'ratio':>8}  how")
    for name, row in sorted(seen.items()):
        got = f"{row['recomputed']:.4f}" if row["recomputed"] else "--"
        ratio = f"{row['ratio']:.4f}" if row["ratio"] else "--"
        print(f"{name:26} {row['score']:>14.4f} {got:>14} {ratio:>8}  {row['how'][:60]}")

    if unrecomputable:
        print(f"\nnot recomputable from a row ({len(unrecomputable)}):")
        for name, why in sorted(unrecomputable.items()):
            print(f"  {name:26} {why}")
    if disagreed:
        print(f"\nDISAGREED ({len(disagreed)}): a Score its own row does not reproduce")
        for row in disagreed:
            if row["defect"]:
                print(f"  {row['workload']:26} {row['how']} in {row['report']}")
                continue
            print(f"  {row['workload']:26} {row['score']:.4f} against "
                  f"{row['recomputed']:.4f} ({row['ratio']:.4f}) in {row['report']}")
        return 1
    print("\nevery recomputable row reproduces its Score.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
