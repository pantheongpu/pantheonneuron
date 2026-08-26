"""Neuron device discovery and capability model.

Trainium and Inferentia share one software stack (the AWS Neuron SDK), so
this suite treats them as one platform with per-architecture capabilities
rather than as two separate backends.  Workloads declare what they need;
the registry filters by what the discovered hardware actually offers.

Topology is read from ``neuron-ls`` at runtime.  The static table below is a
fallback for the rare case where ``neuron-ls`` is present but returns
something we cannot parse -- it is deliberately small, because hardcoded
device tables rot with every instance launch.
"""

import dataclasses
import json
import os
import re
import shutil
import subprocess
import typing


MOCK_ENV = "PANTHEON_NEURON_MOCK"

# NeuronCore generations.  Inf1 (NeuronCore-v1, legacy neuron-cc toolchain)
# is deliberately out of scope: it is a different compiler and a different
# framework package, and supporting it would double the backend surface.
NEURONCORE_V2 = "v2"
NEURONCORE_V3 = "v3"

# arch -> (neuroncore version, cores per device, supports training)
#
# VALIDATION: only the inf2 row has been confirmed against real hardware
# (inf2.xlarge, runtime 2.30.51 -- neuron_hardware_info reported
# neuroncore_version "v2", neuroncore_per_device_count 2). The Trainium rows
# are assumed and have never been observed; trn2's core count in particular
# is the least confident value here.
#
# These are fallbacks. Core count and memory come from neuron-ls at runtime
# when present, so a wrong row here surfaces through supports_training --
# which gates the whole training suite.
_ARCH_TABLE = {
    "trn1": (NEURONCORE_V2, 2, True),    # assumed
    "trn1n": (NEURONCORE_V2, 2, True),   # assumed
    "trn2": (NEURONCORE_V3, 8, True),    # assumed, low confidence
    "inf2": (NEURONCORE_V2, 2, False),   # verified on hardware
}

_UNSUPPORTED_ARCH = {
    "inf1": "Inf1 uses the legacy neuron-cc toolchain and is not supported.",
}


class NeuronUnavailable(RuntimeError):
    """Raised when no Neuron hardware or mock backend is usable."""


@dataclasses.dataclass(frozen=True)
class NeuronDevice:
    """One NeuronDevice (a physical chip), which hosts several NeuronCores."""

    index: int
    arch: str
    neuroncore_version: str
    neuroncores: int
    hbm_bytes: int
    supports_training: bool

    @property
    def is_mock(self) -> bool:
        return self.arch == "mock"

    def capabilities(self) -> typing.Set[str]:
        caps = {"compute", "hbm", f"neuroncore_{self.neuroncore_version}"}
        if self.supports_training:
            caps.add("training")
        if self.neuroncores > 1:
            caps.add("multicore")
        # NeuronLink is present on every in-scope architecture, Inferentia2
        # included -- inf2 shards large models across devices for inference,
        # so collectives must not be gated behind the training capability.
        if self.arch in _ARCH_TABLE or self.is_mock:
            caps.add("collectives")
        return caps


def _mock_devices(count: int = 2) -> typing.List[NeuronDevice]:
    """CPU-backed stand-ins so CI can exercise the orchestrator without hardware."""
    return [
        NeuronDevice(
            index=i,
            arch="mock",
            neuroncore_version=NEURONCORE_V2,
            neuroncores=2,
            hbm_bytes=32 * 1024**3,
            supports_training=True,
        )
        for i in range(count)
    ]


def _run_neuron_ls() -> typing.Optional[dict]:
    binary = shutil.which("neuron-ls")
    if binary is None:
        return None
    try:
        completed = subprocess.run(
            [binary, "--json-output"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def _normalise_arch(raw: str) -> str:
    """Map whatever neuron-ls reports onto an arch key in _ARCH_TABLE."""
    text = (raw or "").strip().lower()
    # neuron-ls has reported this field variously as "trn1", "Trainium",
    # "Trainium2" and as a full instance type; match the useful part.
    if "trainium2" in text or "trn2" in text:
        return "trn2"
    if "trainium" in text or "trn1" in text:
        return "trn1n" if "trn1n" in text else "trn1"
    if "inferentia2" in text or "inf2" in text:
        return "inf2"
    if "inferentia" in text or "inf1" in text:
        return "inf1"
    match = re.match(r"(trn1n|trn1|trn2|inf1|inf2)", text)
    return match.group(1) if match else text


SYSFS_ROOT = "/sys/devices/virtual/neuron_device"


def _read_sysfs(relative: str) -> typing.Optional[str]:
    try:
        with open(os.path.join(SYSFS_ROOT, relative), encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _arch_from_sysfs() -> typing.Optional[str]:
    """Read the architecture from the driver.

    ``neuron-ls --json-output`` reports topology (index, nc_count,
    memory_size) but carries no architecture field at all -- verified on an
    inf2.xlarge running Neuron runtime 2.30.51.  Without this, every device
    would fall back to the table default and report supports_training=False,
    silently skipping the training suite on real Trainium hardware.

    The driver exposes it here instead:
        info/architecture/instance_type  -> "Inf2"
        info/architecture/device_name    -> "Inferentia2"
        info/architecture/arch_type      -> "NDv3"  (device gen, not core gen)
    """
    for field in ("instance_type", "device_name"):
        value = _read_sysfs(f"neuron0/info/architecture/{field}")
        if value:
            arch = _normalise_arch(value)
            if arch in _ARCH_TABLE or arch in _UNSUPPORTED_ARCH:
                return arch
    return None


def _arch_from_instance_type() -> typing.Optional[str]:
    """Env hint, for containers where the sysfs tree is not mounted."""
    hint = os.environ.get("PANTHEON_NEURON_ARCH", "").strip().lower()
    return _normalise_arch(hint) if hint else None


def _parse_neuron_ls(payload) -> typing.List[NeuronDevice]:
    if isinstance(payload, dict):
        entries = payload.get("neuron_devices") or payload.get("devices") or []
    elif isinstance(payload, list):
        entries = payload
    else:
        return []

    devices = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        index = entry.get("neuron_device") or entry.get("nd_index") or position
        raw_arch = (
            entry.get("neuron_device_type")
            or entry.get("device_type")
            or entry.get("arch")
            or _arch_from_sysfs()
            or _arch_from_instance_type()
            or ""
        )
        arch = _normalise_arch(str(raw_arch))
        if arch in _UNSUPPORTED_ARCH:
            raise NeuronUnavailable(_UNSUPPORTED_ARCH[arch])

        version, default_cores, trains = _ARCH_TABLE.get(
            arch, (NEURONCORE_V2, 2, False)
        )
        cores = entry.get("nc_count") or entry.get("neuroncore_count") or default_cores
        hbm = entry.get("memory_size") or entry.get("hbm_size") or 32 * 1024**3

        devices.append(
            NeuronDevice(
                index=int(index),
                arch=arch,
                neuroncore_version=version,
                neuroncores=int(cores),
                hbm_bytes=int(hbm),
                supports_training=trains,
            )
        )
    return devices


def discover(force_mock: bool = False) -> typing.List[NeuronDevice]:
    """Return the usable Neuron devices, or mock devices in CI.

    Raises NeuronUnavailable when real hardware is required but absent.
    """
    if force_mock or os.environ.get(MOCK_ENV) == "1":
        return _mock_devices()

    payload = _run_neuron_ls()
    if payload is not None:
        devices = _parse_neuron_ls(payload)
        if devices:
            return devices

    raise NeuronUnavailable(
        "No Neuron devices found. Install the Neuron driver and run on a "
        f"trn1/trn2/inf2 instance, or set {MOCK_ENV}=1 for a CPU mock run."
    )


def select(devices, spec: str) -> typing.List[NeuronDevice]:
    """Filter discovered devices by an ``--device`` spec ('all' or '0,2')."""
    if spec.strip().lower() == "all":
        return list(devices)
    wanted = {int(part) for part in spec.split(",") if part.strip()}
    chosen = [device for device in devices if device.index in wanted]
    missing = wanted - {device.index for device in chosen}
    if missing:
        raise NeuronUnavailable(
            f"Requested device(s) not present: {sorted(missing)}"
        )
    return chosen
