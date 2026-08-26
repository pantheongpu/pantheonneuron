"""End-to-end orchestrator behaviour."""

import json
import os

import pytest

import pantheon_neuron
from kernels import nki_backend, registry
from neuron_device import NeuronDevice


TRN1 = [NeuronDevice(i, "trn1", "v2", 2, 32 * 1024**3, True) for i in range(2)]
INF2 = [NeuronDevice(i, "inf2", "v2", 2, 32 * 1024**3, False) for i in range(2)]


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    yield
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def _workload(name):
    return next(w for w in registry.WORKLOADS if w.name == name)


def test_skipped_workload_reports_reason(mock_env):
    """Training work is genuinely unavailable on Inferentia."""
    row = pantheon_neuron.run_workload(
        _workload("transformer_train_step"), INF2, duration=1, monitor_period=0.01
    )
    assert row["Status"] == "SKIPPED"
    assert "training" in row["Detail"]


def test_collectives_are_not_skipped_on_inferentia(mock_env):
    """inf2 has NeuronLink; all_reduce must actually run there."""
    row = pantheon_neuron.run_workload(
        _workload("all_reduce"), INF2, duration=1, monitor_period=0.01
    )
    assert row["Status"] == "PASS"


def test_workload_runs_in_mock_mode(mock_env):
    row = pantheon_neuron.run_workload(
        _workload("tensor_virus"), TRN1, duration=1, monitor_period=0.01
    )
    assert row["Status"] == "PASS"
    assert row["Telemetry"]["samples"] > 0


def test_unimplemented_workload_does_not_silently_pass_on_hardware(monkeypatch):
    """A missing NKI kernel must never be reported as a successful run."""
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.setattr(
        nki_backend, "require_toolchain", lambda: {"neuronxcc": "2.x"}
    )
    with pytest.raises(NotImplementedError):
        pantheon_neuron._execute(_workload("tensor_virus"), TRN1, duration=1)


def test_execution_errors_flip_a_pass_to_fail(mock_env, monkeypatch):
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor,
        "stop",
        lambda self: {"samples": 3, "execution_errors": 2},
    )
    row = pantheon_neuron.run_workload(
        _workload("tensor_virus"), TRN1, duration=1, monitor_period=0.01
    )
    assert row["Status"] == "FAIL"
    assert "execution error" in row["Detail"]


def test_report_is_written_atomically(mock_env, tmp_path, monkeypatch):
    monkeypatch.setattr(pantheon_neuron, "DATABASE_DIR", str(tmp_path))
    snapshot = pantheon_neuron.get_system_snapshot(TRN1)
    path = pantheon_neuron.write_report(snapshot, [{"Test Name": "x"}], "runid")

    assert os.path.exists(path)
    assert not os.path.exists(f"{path}.tmp")
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle)["run_id"] == "runid"


def test_cli_list_exits_clean(capsys):
    assert pantheon_neuron.main(["--list"]) == 0
    assert "tensor_virus" in capsys.readouterr().out


def test_cli_reports_missing_hardware(monkeypatch, capsys):
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.setattr(
        pantheon_neuron.neuron_device.shutil, "which", lambda _: None
    )
    assert pantheon_neuron.main(["--test", "baseline_metrics"]) == 2
    assert "No Neuron devices" in capsys.readouterr().err


def test_cli_rejects_unknown_test(mock_env, capsys):
    assert pantheon_neuron.main(["--test", "nope"]) == 2
    assert "Unknown test" in capsys.readouterr().err


def test_full_mock_run_writes_clean_report(mock_env, tmp_path, monkeypatch):
    monkeypatch.setattr(pantheon_neuron, "DATABASE_DIR", str(tmp_path))
    assert pantheon_neuron.main(
        ["--test", "baseline_metrics", "--duration", "1", "--monitor-period", "0.01"]
    ) == 0

    reports = list(tmp_path.glob("*.json"))
    assert len(reports) == 1
    blob = reports[0].read_text().lower()
    for forbidden in ("instance_id", "hostname", "availability_zone"):
        assert forbidden not in blob
