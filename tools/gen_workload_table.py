#!/usr/bin/env python3
"""Regenerate docs/workload_counter_map.md from the registry.

The registry is the source of truth; this renders the readable view.
Run after changing kernels/registry.py:

    python tools/gen_workload_table.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels import registry  # noqa: E402
from neuron_device import NeuronDevice  # noqa: E402

FLEET = [
    ("inf2.xlarge", "inf2", 1, 2, False),
    ("inf2.24xlarge", "inf2", 6, 2, False),
    ("trn1.2xlarge", "trn1", 1, 2, True),
    ("trn1.32xlarge", "trn1", 16, 2, True),
]

SOURCE_SHORT = {
    registry.PROFILER: "profile",
    registry.MONITOR: "monitor",
    registry.NCCOM: "nccom",
    registry.INTERNAL: "kernel",
}


def _devices(arch, count, cores, training):
    return [
        NeuronDevice(i, arch, "v2", cores, 32 * 1024**3, training)
        for i in range(count)
    ]


def main() -> None:
    out = [
        "# Workload reference",
        "",
        "Generated from `kernels/registry.py` by `tools/gen_workload_table.py`.",
        "Do not hand-edit.",
        "",
        "`Score` column: where the number comes from.",
        "`profile` = neuron-profile, `monitor` = neuron-monitor,",
        "`nccom` = nccom-test, `kernel` = counted by the workload itself.",
        "",
        "Instance columns show whether the capability gate admits the workload —",
        "**not** whether a kernel exists. Only `baseline_metrics` is implemented;",
        "everything else raises `NotImplementedError` on hardware.",
        "",
        "| Workload | Suite | Unit | Score | Formula | inf2.xl | inf2.24xl | trn1.2xl | trn1.32xl |",
        "|---|---|---|---|---|:--:|:--:|:--:|:--:|",
    ]

    for workload in registry.WORKLOADS:
        source = workload.score_source
        cells = []
        for _, arch, count, cores, training in FLEET:
            ok = workload.runnable_on(_devices(arch, count, cores, training))
            cells.append("✅" if ok else "—")
        formula = source.formula.split("#")[0].strip() if source else "—"
        if len(formula) > 46:
            formula = formula[:43] + "…"
        out.append(
            f"| `{workload.name}` | {workload.suite} | {workload.unit or '—'} "
            f"| {SOURCE_SHORT.get(source.source, '—') if source else '—'} "
            f"| `{formula}` | " + " | ".join(cells) + " |"
        )

    out += [
        "",
        "A `—` in an instance column means the capability gate skips it: "
        "`all_reduce` and `p2p_thrasher` need 2+ devices for NeuronLink, and "
        "`transformer_train_step` needs a Trainium part.",
        "",
        "## Pinned problems",
        "",
        "A Score is comparable across platforms only if both ran the same "
        "problem, so shape and dtype travel with the score into the report.",
        "",
        "| Workload | Problem |",
        "|---|---|",
    ]
    for workload in registry.WORKLOADS:
        if workload.problem:
            params = ", ".join(f"{k}={v}" for k, v in workload.problem.items())
            out.append(f"| `{workload.name}` | {params} |")

    out += [
        "",
        "## Counters referenced",
        "",
        "| Workload | Counters |",
        "|---|---|",
    ]
    for workload in registry.WORKLOADS:
        if workload.score_source:
            counters = "<br>".join(f"`{c}`" for c in workload.score_source.counters)
            out.append(f"| `{workload.name}` | {counters} |")

    out += [
        "",
        "## No Neuron equivalent",
        "",
        f"{len(registry.NO_NEURON_EQUIVALENT)} pantheongpu workloads have no "
        "counterpart here. Asking for one by name explains why rather than "
        "reporting an unknown test.",
        "",
        "| pantheongpu workload | Reason |",
        "|---|---|",
    ]
    for name, reason in sorted(registry.NO_NEURON_EQUIVALENT.items()):
        out.append(f"| `{name}` | {reason} |")

    out += [
        "",
        "## Measured values",
        "",
        "`data/baselines.json` records what each counter actually read during "
        "the probes. **Those are observations, not benchmark results** — the "
        "probe load was an untuned matmul at 0.0049% MFU. They prove each "
        "counter is readable and catch plumbing regressions; they are not "
        "Inferentia2's throughput.",
        "",
    ]

    target = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs",
        "workload_counter_map.md",
    )
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out))
    print(f"wrote {target} ({len(out)} lines)")


if __name__ == "__main__":
    main()
