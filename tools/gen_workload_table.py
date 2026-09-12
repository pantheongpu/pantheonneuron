#!/usr/bin/env python3
"""Regenerate docs/workload_counter_map.md from the registry.

The registry is the source of truth; this renders the readable view.
Run after changing kernels/registry.py:

    python tools/gen_workload_table.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels import registry
from neuron_device import NeuronDevice

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


def _indicative():
    """Formula applied to counters measured during the probe, where available."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "baselines.json")
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get("derived_indicative", {})
    except OSError:
        return {}


def _devices(arch, count, cores, training):
    return [
        NeuronDevice(i, arch, "v2", cores, 32 * 1024**3, training)
        for i in range(count)
    ]


def _not_comparable() -> str:
    """Name the workloads that share a GPU name but not a quantity.

    Read from ``registry.NOT_COMPARABLE_WITH_GPU`` rather than typed out, so
    adding a workload to that dict cannot leave this paragraph listing the
    old set. The count is derived for the same reason.
    """
    names = sorted(registry.NOT_COMPARABLE_WITH_GPU)
    listed = ", ".join(f"`{name}`" for name in names)
    return (
        f"{len(names)} workloads exist on both platforms under the same name "
        f"and must **not** be compared: {listed}."
    )


def _prose():
    """The narrative sections of the reference.

    These live here, not in the Markdown. The generated file says "do not
    hand-edit" and means it: prose added to the output was deleted the next
    time anyone ran this script, which is exactly the sort of silent loss the
    rest of this suite is built to avoid. Anything that belongs in the
    reference belongs in this function.
    """
    return [
        "## How a monitor-sourced Score is read",
        "",
        "The five compute workloads declare `mean(effective_flops over whole busy periods) / 1e12`. "
        "That counter exists only in the neuron-monitor stream — it is absent "
        "from the CloudWatch metric set, and sysfs leaves `flop_count` at zero "
        "— so unlike the bandwidth kernels, their Score cannot come from the "
        "kernel. `pantheon_neuron.monitor_score` reads it from the telemetry "
        "the run just collected, after the monitor stops.",
        "",
        "`mean` is across NeuronCores. A part reports one series per core, and "
        "a workload that saturates the device runs on all of them; summing "
        "would make a two-core part look twice as fast as the same silicon "
        "reported per core.",
        "",
        "Two cases deliberately produce no Score rather than a number:",
        "",
        "- **The counter is absent** — a mock run, telemetry disabled, or a "
        "kernel that never reached the Tensor Engine. The row records a PASS "
        "with no Score and says why.",
        "- **The workload failed** — telemetry keeps sampling through a "
        "failure, so without a status gate a FAIL row would carry whatever the "
        "monitor caught and read as a measurement.",
        "",
        "`graph_replay` is also neuron-monitor-sourced but is **not** on this "
        "path: its formula is `sum(completed) / sum(period)` over whole busy periods, in graph-steps/s. "
        "The gate matches on the declared counter, not on the source, so the "
        "FLOPS arithmetic cannot reach it.",
        "",
        "## The declared profiler Score has never been produced by a run",
        "",
        "`memory_read` and `memory_write` declare `neuron-profile` as their "
        "Score source, and every run so far has degraded to the analytic "
        "fallback instead. Two causes, found in that order:",
        "",
        "- **inf2.xlarge 2026-09-07** — `neuron-profile capture` replays the "
        "NEFF, which needs NeuronCores, and the workload process held them all "
        "(`Logical Neuron Core(s) not available - Requested:2 Available:0`). "
        "That is why the profiler figures in `data/baselines.json` exist at "
        "all: they came from standalone probe sessions, never from a scored "
        "run. `kernels/cores.py` now reserves a core to close it.",
        "- **trn1.2xlarge 2026-09-08** — the reservation worked and the "
        "capture ran for the first time, against the wrong graph. "
        "`verify_profile_covers_plan` refused it: *profiled graph moved 4 "
        "bytes against a plan of 8589934592*. Narrowing NEFF selection by "
        "compile timestamp is not enough to identify the kernel's own graph. "
        "Scores from a declared hardware source that run: 0 of 4.",
        "",
        "So the fallback is not a rare degradation, it is the only path these "
        "Scores have ever taken — but it is now a loud one. The failure is a "
        "refusal rather than a plausible bandwidth computed from four bytes, "
        "and the row's `Score Method` names the method actually used.",
        "",
        "## Reserving the core costs a selection",
        "",
        "The Neuron runtime reads `NEURON_RT_VISIBLE_CORES` once at "
        "initialisation, so the workload/profiler split is fixed for a whole "
        "run and cannot be renegotiated per workload. `memory_read_agg` and "
        "`memory_write_agg` declare `cores: \"all\"`, and holding a core back "
        "from them would report the aggregate of all-but-one core under a name "
        "that says otherwise. Until 2026-09-11 their presence turned the "
        "reservation off for the entire run, so `--test all` and "
        "`--test memory` could not reach the profiler for `memory_read` or "
        "`memory_write`.",
        "",
        "They now run first, in their own worker processes, and the "
        "reservation is made after them (`pantheon_neuron.reservation_point`), "
        "so every selection reaches the declared source. The console names "
        "where the reservation lands.",
        "",
        "**The aggregates themselves never reach neuron-profile, and no "
        "longer claim to.** A capture replays the NEFF and needs a "
        "NeuronCore of its own; an aggregate gives every core to a worker "
        "and each worker holds the one core it can see. They are counted by "
        "the kernel — summed bytes over the longest worker's loop — which "
        "is what their rows have always reported. The single-core "
        "`memory_read` and `memory_write` keep the profiler, and the "
        "kernels now skip a capture attempt entirely unless a core was "
        "reserved for it.",
        "",
        "## Where the comparison does not hold",
        "",
        _not_comparable(),
        "",
        "pantheongpu v1.0.19 replaced their units with a single "
        f"`{registry.GPU_SYNTHETIC_AI_UNIT}`. Ten of its AI workloads shared "
        "one kernel body and six compiled to byte-identical SASS, so what it "
        "reports is generic synthetic throughput rather than the quantity each "
        "name suggests. The Neuron implementations count the real thing — "
        "tokens generated, cache updates applied, training steps completed.",
        "",
        "## Where the units match and the quantities do not",
        "",
        f"{len(registry.SAME_UNIT_DIFFERENT_QUANTITY)} workloads join "
        "cleanly on (Test Name, Unit) and should not be read as a "
        "comparison. This is the worse of the two failure modes: a failed "
        "join is visible, a successful join between unlike quantities is "
        "not.",
        "",
        "| Workload | Why the two numbers differ |",
        "|---|---|",
    ] + [
        f"| `{name}` | {reason} |"
        for name, reason in sorted(
            registry.SAME_UNIT_DIFFERENT_QUANTITY.items())
    ] + [
        "",
        "Nothing here changes what joins. See "
        "`docs/cross_platform_comparability.md` for the evidence and the "
        "three options, none of them taken.",
        "",
        f"Copying `{registry.GPU_SYNTHETIC_AI_UNIT}` here would restore the "
        "join and compare unlike quantities, so these keep their own units and "
        "are listed in `registry.NOT_COMPARABLE_WITH_GPU`. "
        "`tests/test_score_schema.py` fails if a unit diverges without being "
        "declared there.",
        "",
    ]


TARGET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs",
    "workload_counter_map.md",
)


def render() -> str:
    """Build the whole reference as a string.

    Split from ``main`` so a test can compare the rendered text against the
    committed file without writing to it. That guard is the only thing that
    makes "do not hand-edit" enforceable: a registry change that nobody
    regenerated for now fails CI instead of leaving the reference describing
    a suite that no longer exists.
    """
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
        "**not** whether its kernel has met hardware. Every workload in the "
        f"registry has an implementation ({len(registry.WORKLOADS)} of "
        f"{len(registry.WORKLOADS)}); a name absent from "
        "`pantheon_neuron.IMPLEMENTED` raises `NotImplementedError` on "
        "hardware rather than reporting a silent PASS. See the README for "
        "which kernels have actually run on a device.",
        "",
        "| Workload | Suite | Unit | Score | Measured | inf2.xl | inf2.24xl | trn1.2xl | trn1.32xl |",
        "|---|---|---|---|--:|:--:|:--:|:--:|:--:|",
    ]
    indicative = _indicative()

    for workload in registry.WORKLOADS:
        source = workload.score_source
        cells = []
        for _, arch, count, cores, training in FLEET:
            ok = workload.runnable_on(_devices(arch, count, cores, training))
            cells.append("✅" if ok else "—")
        got = indicative.get(workload.name)
        measured = f"**{got['value']:,.4g}**" if got else "—"
        out.append(
            f"| `{workload.name}` | {workload.suite} | {workload.unit or '—'} "
            f"| {SOURCE_SHORT.get(source.source, '—') if source else '—'} "
            f"| {measured} | " + " | ".join(cells) + " |"
        )

    out += [
        "",
        "**Measured** applies each workload's declared formula to the counters "
        "actually read during the probe. Only five workloads have one, because "
        "only their counters were captured. These are **not Scores** — no "
        "kernel ran, and the load was an untuned matmul at 0.0049% MFU rather "
        "than the pinned problem each workload declares. A real Score will "
        "differ by orders of magnitude.",
        "",
        "A `—` in an instance column means the capability gate skips it: "
        "`all_reduce` and `p2p_thrasher` need 2+ devices for NeuronLink, and "
        "`transformer_train_step` needs a Trainium part.",
        "",
    ]

    out += _prose()

    out += [
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

    return "\n".join(out)


def main() -> None:
    text = render()
    with open(TARGET, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"wrote {TARGET} ({text.count(chr(10)) + 1} lines)")


if __name__ == "__main__":
    main()
