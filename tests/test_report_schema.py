"""A report says which shape it has, and the shape cannot move without saying so.

Reports carried no version while the keys they can hold kept growing --
profiler_active_time_s, ecc_events_observed, tiling and the pinned-memory
controls all arrived within four days -- so the only way to tell "this
report predates the field" from "this kernel did not report it" was the
date. The README said so, and recompute_scores.py guessed the same way.

The fingerprint below is the set of keys a report can carry: the row, the
Measurement whitelist, the monitor's summary, and the report's top level.
Change that set without bumping REPORT_SCHEMA and this fails, naming what to
do. That is what keeps the number meaning something.
"""

import copy
import hashlib
import json
import os
import sys

import pytest

import neuron_monitor
import pantheon_neuron
from kernels import registry
from neuron_device import NeuronDevice

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import recompute_scores  # noqa: E402

# schema -> fingerprint of the key set that schema describes. Append, never
# edit: an old entry is what an old report's number means.
FINGERPRINTS = {
    1: 'e21f505394e676a3',
}

_DEVICES = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]
_ECC = ("mem_ecc_corrected", "mem_ecc_uncorrected",
        "sram_ecc_corrected", "sram_ecc_uncorrected")


def _monitor_keys():
    sample = {"system_data": {"neuron_hw_counters": {"neuron_devices": [
        {"neuron_device_index": 0, **dict.fromkeys(_ECC, 0)}]}}}
    monitor = neuron_monitor.NeuronMonitor()
    monitor._samples = [copy.deepcopy(sample), copy.deepcopy(sample)]
    return sorted(monitor.aggregate())


def _row_keys(monkeypatch):
    workload = next(w for w in registry.WORKLOADS if w.name == "tensor_virus")
    monkeypatch.setattr(pantheon_neuron, "_execute", lambda *a, **k: 78.6)
    monkeypatch.setattr(neuron_monitor.NeuronMonitor, "start",
                        lambda self, indices: True)
    monkeypatch.setattr(neuron_monitor.NeuronMonitor, "stop",
                        lambda self: {"samples": 2})
    monkeypatch.setattr(neuron_monitor.NeuronMonitor, "shutdown",
                        lambda self: None)
    monkeypatch.setitem(pantheon_neuron._LAST_RUN, workload.name,
                        {"elapsed_s": 1.0, "score_method": "analytic"})
    return sorted(pantheon_neuron._measure_once(workload, _DEVICES, 1, 0.5))


def _report(tmp_path, monkeypatch):
    monkeypatch.setattr(pantheon_neuron, "DATABASE_DIR", str(tmp_path))
    path = pantheon_neuron.write_report({"devices": [], "invocation": {}},
                                        [], "20260924_000000")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fingerprint(monkeypatch, tmp_path):
    shape = {
        "row": _row_keys(monkeypatch),
        "measurement": sorted(pantheon_neuron._PROVENANCE_KEYS),
        "telemetry": _monitor_keys(),
        "report": sorted(_report(tmp_path, monkeypatch)),
    }
    return hashlib.sha256(
        json.dumps(shape, sort_keys=True).encode()).hexdigest()[:16]


def test_the_shape_has_not_moved_without_a_bump(monkeypatch, tmp_path):
    current = fingerprint(monkeypatch, tmp_path)
    schema = pantheon_neuron.REPORT_SCHEMA
    recorded = FINGERPRINTS.get(schema)
    assert recorded == current, (
        f"the keys a report can carry changed (fingerprint {current}, schema "
        f"{schema} recorded {recorded}). Bump pantheon_neuron.REPORT_SCHEMA, "
        "say what changed in REPORT_SCHEMA_CHANGES, and add "
        f"{{{schema + 1}: {current!r}}} to FINGERPRINTS here.")


def test_every_schema_says_what_it_changed():
    changes = pantheon_neuron.REPORT_SCHEMA_CHANGES
    assert pantheon_neuron.REPORT_SCHEMA == max(changes)
    assert sorted(changes) == list(range(1, max(changes) + 1))
    assert set(changes) == set(FINGERPRINTS)
    assert all(text.strip() for text in changes.values())


def test_a_written_report_says_its_schema_first(tmp_path, monkeypatch):
    report = _report(tmp_path, monkeypatch)
    assert report["schema_version"] == pantheon_neuron.REPORT_SCHEMA
    assert next(iter(report)) == "schema_version"


def test_the_snapshot_cannot_overwrite_it(tmp_path, monkeypatch):
    monkeypatch.setattr(pantheon_neuron, "DATABASE_DIR", str(tmp_path))
    path = pantheon_neuron.write_report({"schema_version": 999}, [], "x")
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle)["schema_version"] == pantheon_neuron.REPORT_SCHEMA


# -- what the number is for --------------------------------------------------

def _profile_row():
    return {"Test Name": "memory_read", "Status": "PASS", "Score": 272.98,
            "Measurement": {"hbm_read_bytes": 8 * 1024**3}}


def test_an_unversioned_report_missing_the_window_predates_it():
    workload = registry.resolve("memory_read")[0]
    got, how = recompute_scores.recompute(workload, _profile_row(), None)
    assert got is None and "no schema_version" in how


def test_a_versioned_report_missing_the_window_is_a_defect():
    """Before the version this could only be excused, never caught."""
    workload = registry.resolve("memory_read")[0]
    with pytest.raises(recompute_scores.MissingProvenance, match="stopped publishing"):
        recompute_scores.recompute(workload, _profile_row(), 1)


def test_the_checker_fails_a_versioned_report_that_broke_its_promise(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"schema_version": 1,
                                "test_results": [_profile_row()]}))
    rows = recompute_scores.check([str(path)])
    assert rows[0]["defect"] is True
    assert recompute_scores.main([str(tmp_path)]) == 1
