"""The dataset README's prose figures are recomputed from the data they describe.

The comparison table in data/publication-2026-09-21/README.md is rendered by
tools/compare_with_gpu.py and tested. The prose around it was typed, and one
figure had drifted: "reads run 83.5-98.4%" beside a table saying 83.6% for
the same A10 row (83.556%, truncated in one place and rounded in the other).
This is the #50 lesson for the Kernel status table, applied to the prose:
every number a reader might quote is derived here from the committed files,
and the test names the sentence when they part.
"""

import glob
import json
import os
import re
import statistics

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "publication-2026-09-21")


def _load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as handle:
        return json.load(handle)


def _readme():
    with open(os.path.join(DATA, "README.md"), encoding="utf-8") as handle:
        return re.sub(r"\s+", " ", handle.read())


def _rows(part, group="300x3"):
    return [row for row in _load(f"summary-{part}.json")["rows"]
            if row["group"] == group and row["median_of_hosts"]]


def _cv(part, name, group="300x3"):
    row = next(r for r in _load(f"summary-{part}.json")["rows"]
               if r["group"] == group and r["workload"] == name)
    return row["between_host_cv"]


def _gpu(name, row):
    return _load("gpu-reference.json")["parts"][name]["rows"][row]["median_gbps"]


def _peak(name):
    return _load("gpu-reference.json")["parts"][name]["published_peak_gbps"]


def _shares(row):
    parts = _load("gpu-reference.json")["parts"]
    return [100 * entry["rows"][row]["median_gbps"] / entry["published_peak_gbps"]
            for entry in parts.values()
            if entry["published_peak_gbps"] and row in entry["rows"]]


def _trn1_read_agg():
    return next(r for r in _rows("trn1")
                if r["workload"] == "memory_read_agg")["median_of_hosts"]


def _pcie_per_host():
    """(host, direction) -> {group: gbps} from the committed reports."""
    found = {}
    for path in glob.glob(os.path.join(DATA, "trn1", "*", "*", "*.json")):
        host, group = path.split(os.sep)[-3], path.split(os.sep)[-2]
        if group not in ("300x3", "3600"):
            continue
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        for row in report["test_results"]:
            if row["Test Name"] == "pcie_bandwidth" and row["Status"] == "PASS":
                for leg, value in row["Measurement"]["per_direction"].items():
                    found.setdefault((host, leg), {})[group] = value["gbps"]
    return found


def _claims():
    trn1, inf2 = _rows("trn1"), _rows("inf2")
    trn1_within = sum(1 for r in trn1 if r["between_host_cv"] <= 0.002)
    inf2_tight = sum(1 for r in inf2 if r["between_host_cv"] <= 0.001)
    inf2_loose = sum(1 for r in inf2 if r["between_host_cv"] <= 0.005)
    reads, writes = _shares("memory_read"), _shares("memory_write")
    read_agg = _trn1_read_agg()
    h100_pcie_nominal = 2039.0     # the brief's own clock x bus width
    yield ("trn1 agreement",
           f"{trn1_within} of {len(trn1)} scored workloads agree across the "
           "three hosts within 0.2%")
    yield ("inf2 agreement",
           f"{inf2_tight} of {len(inf2)} agree across its two hosts within 0.1% "
           f"and all {inf2_loose} within 0.5%")
    yield ("inf2 exceptions",
           f"(`allocation_fragmentation` {100 * _cv('inf2', 'allocation_fragmentation'):.2f}%, "
           f"`pcie_bandwidth` {100 * _cv('inf2', 'pcie_bandwidth'):.2f}%, "
           f"`graph_replay` {100 * _cv('inf2', 'graph_replay'):.2f}%)")
    yield ("trn1 allocation spread",
           f"| `allocation_fragmentation` | {100 * _cv('trn1', 'allocation_fragmentation'):.1f}% |")
    yield ("trn1 pcie spread",
           f"| `pcie_bandwidth` | {100 * _cv('trn1', 'pcie_bandwidth'):.1f}% at 300 s, "
           f"{100 * _cv('trn1', 'pcie_bandwidth', '3600'):.1f}% at 3600 s |")
    yield ("Trainium1 share of each AWS reading",
           f"{read_agg:.1f} GB/s at either **{100 * read_agg / 880.5:.1f}%** or "
           f"**{100 * read_agg / 820.0:.1f}%**")
    yield ("H100 PCIe against its own clock arithmetic",
           f"{100 * _gpu('NVIDIA H100 PCIe', 'memory_read') / h100_pcie_nominal:.1f}% read and "
           f"{100 * _gpu('NVIDIA H100 PCIe', 'memory_write') / h100_pcie_nominal:.1f}% write")
    yield ("L40S writes",
           f"{_gpu('NVIDIA L40S', 'memory_write'):.1f} GB/s against "
           f"{_peak('NVIDIA L40S'):.0f}, while its reads reach "
           f"{100 * _gpu('NVIDIA L40S', 'memory_read') / _peak('NVIDIA L40S'):.1f}%")
    yield ("NVIDIA read and write ranges",
           f"{min(reads):.1f}-{max(reads):.1f}%; writes run "
           f"{min(writes):.1f}-{max(writes):.1f}%")
    yield ("NVIDIA read range, whole percent",
           f"the NVIDIA parts reach {min(reads):.0f}-{max(reads):.0f}%")
    for (host, leg), value in sorted(_pcie_per_host().items()):
        if "300x3" in value and "3600" in value:
            change = 100 * (value["3600"] / value["300x3"] - 1)
            sign = "\u2212" if change < 0 else "+"   # the README uses a real minus
            yield (f"pcie {host} {leg}",
                   f"{value['300x3']:.3f} | {value['3600']:.3f} | "
                   f"{'**' if abs(change) > 10 else ''}{sign}{abs(change):.1f}%")


@pytest.mark.parametrize("label, text", [
    pytest.param(label, text, id=label) for label, text in _claims()])
def test_the_readme_says_what_the_data_says(label, text):
    assert text in _readme(), f"{label}: the data gives {text!r}"


def test_there_are_claims_to_check():
    """Eighteen, counting the six per-host pcie cells; a helper returning
    nothing would pass every one of them."""
    assert len(list(_claims())) >= 16


def test_statistics_is_the_source_not_a_shortcut():
    """between_host_cv is what the prose quotes; confirm it is the population
    cv of the hosts' medians, so the summary cannot drift from its meaning."""
    row = next(r for r in _rows("trn1") if r["workload"] == "pcie_bandwidth")
    scores = list(row["scores"].values())
    cv = statistics.pstdev(scores) / statistics.fmean(scores)
    assert row["between_host_cv"] == pytest.approx(cv, rel=0.02) or \
        row["between_host_cv"] == pytest.approx(
            statistics.stdev(scores) / statistics.fmean(scores), rel=0.02)
