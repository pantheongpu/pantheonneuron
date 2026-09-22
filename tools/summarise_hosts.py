#!/usr/bin/env python3
"""Put the same pass from several hosts side by side, and say how far apart they are.

Every hardware figure this suite had published before 2026-09-21 came from one
rented instance at a time. Repeats on that instance say how steady a Score is
*there*; they say nothing about whether the next instance would agree, and a
reader comparing against another vendor's part is implicitly assuming it would.

This reads one directory per host, each holding the reports that host wrote,
and reports two spreads per workload and keeps them apart because they answer
different questions:

* **within a host** -- the coefficient of variation the harness computed over
  that host's repeats, worst host shown;
* **between hosts** -- the coefficient of variation of the hosts' medians.

A between-host spread well above the within-host one means the instance is part
of the measurement, and a single-host figure should not be quoted as the part's.

Rows are matched on (group, Test Name, Unit, Problem). The group is a directory
level the caller chooses -- ``300x3``, ``3600``, ``sweep`` -- because a report
written before 2026-09-21 does not record the ``--duration`` it ran at, and the
measured window cannot stand in for it: allocation_fragmentation stops at its
pinned count after about 18 s whatever the flag said. So a 3600 s run is never
averaged with a 300 s one, and a size-sweep row never lands beside the pinned
problem, without this tool guessing at either.

A workload that did not PASS on a host is listed as such rather than dropped: a
Score from two hosts of three is a different claim from three of three.

    python tools/summarise_hosts.py RESULTS_DIR            # a table
    python tools/summarise_hosts.py RESULTS_DIR --json     # the same, as data

RESULTS_DIR/<host>/<group>/ holds report JSON files. Hosts are named however
the caller labels them -- never an instance id, since these names reach the
output.
"""

import glob
import json
import os
import statistics
import sys


def load_rows(group_dir: str):
    """Every result row in one host's group directory."""
    rows = []
    for path in sorted(glob.glob(os.path.join(group_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(report, dict):
            continue
        for row in report.get("test_results") or []:
            if isinstance(row, dict) and row.get("Test Name"):
                rows.append(row)
    return rows


def _subdirectories(path: str):
    return sorted(name for name in os.listdir(path)
                  if os.path.isdir(os.path.join(path, name)))


def key_of(group: str, row):
    problem = json.dumps(row.get("Problem"), sort_keys=True)
    return (group, row["Test Name"], row.get("Unit"), problem)


def _order(item):
    """Group, workload, then Problem values numerically: 2 GiB before 12."""
    (group, name, unit, problem), _ = item
    values = json.loads(problem) or {}
    return (group, name, str(unit), [
        (key, 0, value) if isinstance(value, (int, float)) and not isinstance(value, bool)
        else (key, 1, str(value))
        for key, value in sorted(values.items())])


def summarise(results_dir: str):
    hosts = _subdirectories(results_dir)
    table = {}
    for host in hosts:
        host_dir = os.path.join(results_dir, host)
        for group in _subdirectories(host_dir):
            for row in load_rows(os.path.join(host_dir, group)):
                entry = table.setdefault(key_of(group, row), {})
                # A host that ran the same thing twice keeps the later row;
                # the glob is sorted and report names are timestamps.
                entry[host] = row

    summary = []
    for (group, name, unit, problem), per_host in sorted(table.items(), key=_order):
        scored = {
            host: row["Score"] for host, row in per_host.items()
            if row.get("Status") == "PASS"
            and isinstance(row.get("Score"), (int, float))}
        within = [
            (row.get("Repeats") or {}).get("cv") for row in per_host.values()]
        within = [cv for cv in within if isinstance(cv, (int, float))]
        medians = list(scored.values())
        between = None
        if len(medians) > 1 and statistics.fmean(medians):
            between = statistics.stdev(medians) / statistics.fmean(medians)
        summary.append({
            "group": group,
            "workload": name,
            "unit": unit,
            "problem": json.loads(problem),
            "hosts": {host: per_host[host].get("Status") for host in sorted(per_host)},
            "scores": {host: scored[host] for host in sorted(scored)},
            "score_methods": sorted({
                str(row.get("Score Method")) for row in per_host.values()
                if row.get("Status") == "PASS"}),
            "median_of_hosts": statistics.median(medians) if medians else None,
            "worst_within_host_cv": max(within) if within else None,
            "between_host_cv": between,
        })
    return {"hosts": hosts, "rows": summary}


def _distinguishers(rows):
    """For rows sharing (group, workload), the Problem keys that differ.

    Rows are keyed on Problem so a size sweep never averages 1 GiB with 12,
    and the first rendering dropped the Problem -- five memory_read sweep rows
    printed as five identical lines. Only the keys that actually differ are
    shown, so a table of pinned problems stays as narrow as it was.
    """
    by_name = {}
    for row in rows:
        by_name.setdefault((row["group"], row["workload"]), []).append(row)
    labels = {}
    for siblings in by_name.values():
        if len(siblings) < 2:
            continue
        keys = sorted({key for row in siblings for key in (row["problem"] or {})})
        differing = [key for key in keys
                     if len({repr((row["problem"] or {}).get(key)) for row in siblings}) > 1]
        for row in siblings:
            labels[id(row)] = " ".join(
                f"{key}={_compact((row['problem'] or {}).get(key))}" for key in differing)
    return labels


def _compact(value):
    """Byte counts as GiB when they divide evenly; anything else as is."""
    if isinstance(value, int) and value >= 1 << 30 and value % (1 << 30) == 0:
        return f"{value >> 30}GiB"
    return value


def render(summary) -> str:
    lines = [f"hosts: {', '.join(summary['hosts'])}", ""]
    labels = _distinguishers(summary["rows"])
    header = (f"{'group':8} {'workload':26} {'unit':20} {'median':>14} "
              f"{'within cv':>10} {'between cv':>11}  passed")
    lines.append(header)
    for row in summary["rows"]:
        passed = sum(1 for status in row["hosts"].values() if status == "PASS")
        median = row["median_of_hosts"]
        lines.append(
            f"{row['group']:8} {(row['workload'] + ' ' + labels.get(id(row), '')).strip():26} "
            f"{row['unit'] or '-':20} "
            f"{'-' if median is None else format(median, '.4g'):>14} "
            f"{_cv(row['worst_within_host_cv']):>10} "
            f"{_cv(row['between_host_cv']):>11}  {passed}/{len(row['hosts'])}")
    return "\n".join(lines)


def _cv(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def main(argv) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    summary = summarise(argv[0])
    if "--json" in argv[1:]:
        print(json.dumps(summary, indent=2))
    else:
        print(render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
