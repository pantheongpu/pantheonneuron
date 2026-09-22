#!/usr/bin/env python3
"""Compare this suite's Scores with pantheongpu's, and refuse where it cannot.

`registry.NOT_COMPARABLE_WITH_GPU` and `registry.SAME_UNIT_DIFFERENT_QUANTITY`
exist, in the registry's own words, so "the comparison tooling can render an
honest 'not comparable' rather than a row that quietly never joins, or worse,
one that joins and misleads". **There was no such tooling.** The registers were
consulted by the documentation generator and by nothing that compares
anything, so the only cross-platform table this repo has published was typed
by hand.

This is that tool. It reads:

* `data/publication-2026-09-21/summary-{trn1,inf2}.json` -- this suite's
  medians across hosts, from `tools/summarise_hosts.py`;
* `data/publication-2026-09-21/gpu-reference.json` -- pantheongpu's medians
  and each part's published peak, from `tools/tally_gpu_reference.py`;
* the two registers, for what must not be compared and why.

and prints the comparison, then the refusals with their reasons. No figure is
read from prose, which is the failure mode this repo keeps meeting: a
transcription that was right when written and silently stopped being right.

**The one pairing that holds crosses names.** This suite's `memory_read` is a
single NeuronCore of two; pantheongpu's is a whole GPU. The whole-device row
here is `memory_read_agg`, and in pantheongpu `_agg` means the same whole-GPU
kernel over a stress data pattern -- a different thing under the same suffix,
which its own reports put within 0.3% of the plain row. So device-level
Trainium is compared against plain GPU `memory_read`, a pairing no join on
(Test Name, Unit) can produce. See #28 and docs/cross_platform_comparability.md.

    python tools/compare_with_gpu.py            # table, then the refusals
    python tools/compare_with_gpu.py --render   # just the markdown table
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kernels import registry  # noqa: E402

DATA = os.path.join(ROOT, "data", "publication-2026-09-21")

# (this suite's row, pantheongpu's row, why the names differ). Declared here
# rather than inferred, because the whole point is that the names do not line
# up and a reader has to be told which quantity is on each side.
DEVICE_PAIRING = (
    ("memory_read_agg", "memory_read",
     "every NeuronCore against a whole GPU: both are the device's read bandwidth"),
    ("memory_write_agg", "memory_write",
     "every NeuronCore against a whole GPU: both are the device's write bandwidth"),
)

# The parts a Trainium1 buyer is choosing between, in the order a reader
# should meet them. Anything in the reference but not here is still tallied
# and simply not tabled; --all shows the rest.
HEADLINE_GPUS = (
    "NVIDIA A10", "NVIDIA A10G", "NVIDIA L4", "NVIDIA L40S",
    "NVIDIA A100-SXM4-40GB", "NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3",
)


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as handle:
        return json.load(handle)


def neuron_devices():
    """This suite's device-level medians, per part, from the summaries."""
    parts = []
    for arch, summary in (("trn1", "summary-trn1.json"), ("inf2", "summary-inf2.json")):
        try:
            rows = load(summary)["rows"]
        except (OSError, ValueError, KeyError):
            continue
        peak = registry.PART_PEAKS[arch]
        measured = {}
        for ours, _theirs, _why in DEVICE_PAIRING:
            row = next((r for r in rows
                        if r["group"] == "300x3" and r["workload"] == ours
                        and r["median_of_hosts"]), None)
            if row:
                measured[ours] = (row["median_of_hosts"], len(row["scores"]))
        if measured:
            parts.append({
                "label": peak["device_name"],
                "peak_gbps": peak["hbm_gbps"],
                "peak_source": "AWS Neuron architecture docs (820 GiB/s); its NKI "
                               "guide says 820 GB/s, a 7.4% disagreement -- "
                               "registry.PART_PEAKS keeps the lower-flattering GiB "
                               "reading",
                "measured": measured,
            })
    return parts


def gpu_devices(reference, only_headline=True):
    """The GPU rows, in HEADLINE_GPUS order so the table reads by capability."""
    names = [n for n in HEADLINE_GPUS if n in reference["parts"]]
    if not only_headline:
        names += sorted(set(reference["parts"]) - set(HEADLINE_GPUS))
    parts = []
    for name in names:
        entry = reference["parts"][name]
        measured = {}
        for _ours, theirs, _why in DEVICE_PAIRING:
            row = entry["rows"].get(theirs)
            if row:
                measured[theirs] = (row["median_gbps"], row["samples"])
        if measured:
            parts.append({
                "label": name,
                "peak_gbps": entry["published_peak_gbps"],
                "peak_source": entry["published_peak_source"],
                "measured": measured,
                "reports": entry["reports"],
            })
    return parts


def _share(value, peak):
    return f"{100 * value / peak:.1f}%" if value and peak else "--"


def render(reference, only_headline=True):
    """The comparison, as the markdown the dataset README carries."""
    lines = [
        "| Part | Device read (GB/s) | Device write (GB/s) | Published peak (GB/s) | Read % | Write % |",
        "|---|---|---|---|---|---|",
    ]
    for part in neuron_devices():
        read, _ = part["measured"].get("memory_read_agg", (None, 0))
        write, _ = part["measured"].get("memory_write_agg", (None, 0))
        peak = part["peak_gbps"]
        lines.append(
            f"| **{part['label']}** | **{read:.1f}** | **{write:.1f}** | {peak:.1f} | "
            f"**{_share(read, peak)}** | **{_share(write, peak)}** |")
    for part in gpu_devices(reference, only_headline):
        read, _ = part["measured"].get("memory_read", (None, 0))
        write, _ = part["measured"].get("memory_write", (None, 0))
        peak = part["peak_gbps"]
        shown = f"{peak:,.0f}" if peak else "*none published*"
        lines.append(
            f"| {part['label'].replace('NVIDIA ', '')} | {read:.1f} | {write:.1f} | "
            f"{shown} | {_share(read, peak)} | {_share(write, peak)} |")
    return "\n".join(lines)


def refusals():
    """Every name that joins and must not be compared, with the reason."""
    out = []
    for name, unit in sorted(registry.NOT_COMPARABLE_WITH_GPU.items()):
        out.append((name, "units diverge",
                    f"this suite reports {unit}; pantheongpu does not"))
    for name, why in sorted(registry.SAME_UNIT_DIFFERENT_QUANTITY.items()):
        out.append((name, "same unit, different quantity", why))
    return out


def main(argv) -> int:
    try:
        reference = load("gpu-reference.json")
    except OSError:
        print("gpu-reference.json is missing; run tools/tally_gpu_reference.py")
        return 2
    only_headline = "--all" not in argv
    table = render(reference, only_headline)
    if "--render" in argv:
        print(table)
        return 0

    print("Device memory bandwidth, the one comparison that holds:\n")
    print(table)
    print(f"\nPairing: {DEVICE_PAIRING[0][2]}.")
    print(f"pantheongpu medians from {reference['tallied']}; "
          f"Neuron from the 2026-09-21 pass.\n")
    print(f"Refused, {len(refusals())} names that join on (Test Name, Unit) and "
          f"must not be read as comparisons:\n")
    for name, kind, why in refusals():
        print(f"  {name:26} {kind}")
        print(f"  {'':26} {why[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
