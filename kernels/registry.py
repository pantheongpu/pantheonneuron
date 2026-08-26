"""Workload registry with capability gating.

A workload declares the device capabilities it needs; the registry filters
against what discovery actually found.  This is what keeps Trainium-only
work (collectives over NeuronLink, long training soaks) and Inferentia in
one repo without either pretending to be the other.
"""

import dataclasses
import typing


@dataclasses.dataclass(frozen=True)
class Workload:
    name: str
    suite: str
    summary: str
    requires: typing.FrozenSet[str] = frozenset()
    min_devices: int = 1

    def runnable_on(self, devices) -> bool:
        if len(devices) < self.min_devices:
            return False
        return all(
            self.requires <= device.capabilities() for device in devices
        )

    def skip_reason(self, devices) -> typing.Optional[str]:
        if len(devices) < self.min_devices:
            return (
                f"needs {self.min_devices} devices, {len(devices)} selected"
            )
        for device in devices:
            missing = self.requires - device.capabilities()
            if missing:
                return (
                    f"device {device.index} ({device.arch}) lacks: "
                    + ", ".join(sorted(missing))
                )
        return None


WORKLOADS: typing.Tuple[Workload, ...] = (
    Workload(
        name="baseline_metrics",
        suite="baseline",
        summary="Idle telemetry baseline; no load applied.",
    ),
    Workload(
        name="gemm_stress",
        suite="core",
        summary="Sustained large matmuls to load the tensor engine.",
        requires=frozenset({"compute"}),
    ),
    Workload(
        name="hbm_bandwidth",
        suite="memory",
        summary="Large-tensor streaming to saturate device HBM bandwidth.",
        requires=frozenset({"hbm"}),
    ),
    Workload(
        name="multicore_saturation",
        suite="core",
        summary="Concurrent work on every NeuronCore of each device.",
        requires=frozenset({"compute", "multicore"}),
    ),
    Workload(
        name="collective_allreduce",
        suite="interconnect",
        summary="All-reduce across devices over NeuronLink.",
        requires=frozenset({"training", "multicore"}),
        min_devices=2,
    ),
)

SUITES = ("baseline", "core", "memory", "interconnect")

_BY_NAME = {workload.name: workload for workload in WORKLOADS}


def resolve(target: str) -> typing.List[Workload]:
    """Resolve a --test value: a workload name, a suite name, or 'all'."""
    key = target.strip().lower()
    if key == "all":
        return list(WORKLOADS)
    if key in _BY_NAME:
        return [_BY_NAME[key]]
    matched = [w for w in WORKLOADS if w.suite == key]
    if matched:
        return matched
    known = ", ".join(sorted(_BY_NAME) + list(SUITES))
    raise KeyError(f"Unknown test '{target}'. Known targets: {known}")
