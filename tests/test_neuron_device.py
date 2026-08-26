"""Device discovery and the capability model."""

import pytest

import neuron_device
from neuron_device import NeuronDevice


def test_mock_discovery_without_hardware(monkeypatch):
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    devices = neuron_device.discover()
    assert len(devices) == 2
    assert all(device.is_mock for device in devices)


def test_discovery_raises_without_hardware_or_mock(monkeypatch):
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.setattr(neuron_device.shutil, "which", lambda _: None)
    with pytest.raises(neuron_device.NeuronUnavailable):
        neuron_device.discover()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("trn1", "trn1"),
        ("Trainium", "trn1"),
        ("trn1n.32xlarge", "trn1n"),
        ("Trainium2", "trn2"),
        ("trn2", "trn2"),
        ("Inferentia2", "inf2"),
        ("inf2.48xlarge", "inf2"),
    ],
)
def test_arch_normalisation(raw, expected):
    assert neuron_device._normalise_arch(raw) == expected


def test_inf1_is_rejected_as_out_of_scope():
    """Inf1 needs the legacy neuron-cc toolchain; fail loudly, not silently."""
    payload = {"neuron_devices": [{"neuron_device": 0, "device_type": "Inferentia"}]}
    with pytest.raises(neuron_device.NeuronUnavailable, match="Inf1"):
        neuron_device._parse_neuron_ls(payload)


def test_parse_neuron_ls_reads_topology():
    payload = {
        "neuron_devices": [
            {"neuron_device": 0, "device_type": "trn1", "nc_count": 2},
            {"neuron_device": 1, "device_type": "trn1", "nc_count": 2},
        ]
    }
    devices = neuron_device._parse_neuron_ls(payload)
    assert [device.index for device in devices] == [0, 1]
    assert all(device.arch == "trn1" for device in devices)
    assert all(device.supports_training for device in devices)


def test_inferentia_does_not_claim_training_capability():
    device = NeuronDevice(0, "inf2", "v2", 2, 32 * 1024**3, False)
    assert "training" not in device.capabilities()
    assert "compute" in device.capabilities()


def test_select_by_index():
    devices = [NeuronDevice(i, "trn1", "v2", 2, 1, True) for i in range(4)]
    assert [d.index for d in neuron_device.select(devices, "all")] == [0, 1, 2, 3]
    assert [d.index for d in neuron_device.select(devices, "0,2")] == [0, 2]


def test_select_rejects_absent_device():
    devices = [NeuronDevice(0, "trn1", "v2", 2, 1, True)]
    with pytest.raises(neuron_device.NeuronUnavailable, match="not present"):
        neuron_device.select(devices, "0,7")
