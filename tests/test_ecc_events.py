"""ECC errors during a run: the evidence this suite collected and never read.

`neuron-monitor` reports four ECC counters per device, and every report this
suite has written carries them. Nothing read them. A run in which memory
changed underneath the kernel published a clean PASS with a Score, and the
number saying so sat in the row's Telemetry -- which is the shape this repo
keeps finding, except that here there was not even a check to ignore.

It was not an oversight but a stalled decision, and the monitor said so: a
max across samples is a total since driver load, or a per-period tally, and
nobody knows which, so acting on its value would either charge a device's
past to this run or undercount. The way past it is to stop reading the
value and read the *rise*, which means the same thing under both.
"""

import copy

import pytest

import neuron_monitor
import pantheon_neuron
from kernels import registry
from neuron_device import NeuronDevice

_DEVICES = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]
_KEYS = ("mem_ecc_corrected", "mem_ecc_uncorrected",
         "sram_ecc_corrected", "sram_ecc_uncorrected")


def _sample(**counters):
    device = {"neuron_device_index": counters.pop("index", 0)}
    device.update(dict.fromkeys(_KEYS, 0))
    device.update(counters)
    return {"system_data": {"neuron_hw_counters": {"neuron_devices": [device]}}}


def _aggregate(*samples):
    monitor = neuron_monitor.NeuronMonitor()
    monitor._samples = [copy.deepcopy(s) for s in samples]
    return monitor.aggregate()


# -- the rise, under either reading of the counter ---------------------------

def test_a_rising_total_is_counted_as_events_now():
    """If the counters are totals since driver load, the rise is new."""
    metrics = _aggregate(_sample(mem_ecc_corrected=100),
                         _sample(mem_ecc_corrected=103))
    assert metrics["ecc_events_observed"]["mem_ecc_corrected"] == 3
    assert metrics["ecc_events_observed_total"] == 3


def test_a_per_period_tally_is_counted_too():
    """If they are per-period, a period that counted more is also now."""
    metrics = _aggregate(_sample(), _sample(sram_ecc_uncorrected=3), _sample())
    assert metrics["ecc_events_observed"]["sram_ecc_uncorrected"] == 3


def test_a_device_that_arrived_with_a_history_is_not_charged_for_it():
    """The whole reason nothing could act on the raw counter."""
    metrics = _aggregate(_sample(mem_ecc_corrected=4096),
                         _sample(mem_ecc_corrected=4096))
    assert metrics["ecc_events"]["mem_ecc_corrected"] == 4096
    assert metrics["ecc_events_observed_total"] == 0


def test_one_sample_cannot_show_a_rise_and_does_not_claim_to():
    """Zero would report a clean run that was never observed."""
    metrics = _aggregate(_sample(mem_ecc_corrected=7))
    assert metrics["ecc_events_observed"] is None
    assert metrics["ecc_events_observed_total"] is None


def test_two_devices_are_summed():
    monitor = neuron_monitor.NeuronMonitor()
    both = {"system_data": {"neuron_hw_counters": {"neuron_devices": [
        {"neuron_device_index": 0, **dict.fromkeys(_KEYS, 0)},
        {"neuron_device_index": 1, **dict.fromkeys(_KEYS, 0)},
    ]}}}
    after = copy.deepcopy(both)
    after["system_data"]["neuron_hw_counters"]["neuron_devices"][0][
        "mem_ecc_uncorrected"] = 1
    after["system_data"]["neuron_hw_counters"]["neuron_devices"][1][
        "mem_ecc_uncorrected"] = 1
    monitor._samples = [both, after]
    assert monitor.aggregate()["ecc_events_observed"]["mem_ecc_uncorrected"] == 2


def test_a_counter_that_falls_is_not_a_negative_event_count():
    metrics = _aggregate(_sample(mem_ecc_corrected=5), _sample())
    assert metrics["ecc_events_observed"]["mem_ecc_corrected"] == 5
    assert metrics["ecc_events_observed_total"] >= 0


def test_the_raw_counter_is_untouched():
    """Existing readers of ecc_events see exactly what they saw before."""
    metrics = _aggregate(_sample(mem_ecc_corrected=2),
                         _sample(mem_ecc_corrected=9))
    assert metrics["ecc_events"]["mem_ecc_corrected"] == 9
    assert metrics["ecc_events_total"] == 9


# -- what the row does about it ----------------------------------------------

def test_an_uncorrected_error_fails_the_row():
    verdict, note = pantheon_neuron.ecc_verdict(
        {"mem_ecc_corrected": 0, "mem_ecc_uncorrected": 1,
         "sram_ecc_corrected": 0, "sram_ecc_uncorrected": 0})
    assert verdict == "FAIL"
    assert "not a measurement of a working part" in note


def test_a_corrected_error_keeps_the_score_and_says_so():
    verdict, note = pantheon_neuron.ecc_verdict(
        {"mem_ecc_corrected": 4, "mem_ecc_uncorrected": 0,
         "sram_ecc_corrected": 0, "sram_ecc_uncorrected": 0})
    assert verdict == "PASS"
    assert "the Score stands" in note
    assert "producing errors" in note


def test_a_clean_run_says_nothing():
    verdict, note = pantheon_neuron.ecc_verdict(dict.fromkeys(_KEYS, 0))
    assert (verdict, note) == ("PASS", "")


def test_an_unobserved_run_is_not_treated_as_clean_or_as_broken():
    assert pantheon_neuron.ecc_verdict(None) == ("PASS", "")


def test_the_breakdown_names_which_memory():
    _verdict, note = pantheon_neuron.ecc_verdict(
        {"mem_ecc_corrected": 0, "mem_ecc_uncorrected": 2,
         "sram_ecc_corrected": 1, "sram_ecc_uncorrected": 0})
    assert "mem_ecc_uncorrected 2" in note and "sram_ecc_corrected 1" in note
    assert "mem_ecc_corrected" not in note.split("(")[1]


# -- end to end through the row builder --------------------------------------

def _row(observed, monkeypatch):
    workload = next(w for w in registry.WORKLOADS if w.name == "tensor_virus")
    monkeypatch.setattr(pantheon_neuron, "_execute", lambda *a, **k: 78.6)
    monkeypatch.setattr(neuron_monitor.NeuronMonitor, "start",
                        lambda self, indices: True)
    monkeypatch.setattr(neuron_monitor.NeuronMonitor, "stop",
                        lambda self: {"samples": 2,
                                      "ecc_events_observed": observed})
    monkeypatch.setattr(neuron_monitor.NeuronMonitor, "shutdown",
                        lambda self: None)
    monkeypatch.setitem(pantheon_neuron._LAST_RUN, workload.name,
                        {"elapsed_s": 1.0, "score_method": "analytic"})
    return pantheon_neuron._measure_once(workload, _DEVICES, 1, 0.5)


def test_a_row_with_an_uncorrected_error_does_not_publish_a_score(monkeypatch):
    row = _row({"mem_ecc_corrected": 0, "mem_ecc_uncorrected": 1,
                "sram_ecc_corrected": 0, "sram_ecc_uncorrected": 0}, monkeypatch)
    assert row["Status"] == "FAIL"
    assert row["Score"] is None
    assert "uncorrected ECC" in row["Detail"]


def test_a_row_with_a_corrected_error_publishes_its_score(monkeypatch):
    row = _row({"mem_ecc_corrected": 2, "mem_ecc_uncorrected": 0,
                "sram_ecc_corrected": 0, "sram_ecc_uncorrected": 0}, monkeypatch)
    assert row["Status"] == "PASS"
    assert row["Score"] == pytest.approx(78.6)
    assert "corrected ECC" in row["Detail"]


def test_a_clean_row_carries_no_ecc_sentence(monkeypatch):
    row = _row(dict.fromkeys(_KEYS, 0), monkeypatch)
    assert row["Status"] == "PASS"
    assert "ECC" not in row["Detail"]


def test_a_report_written_before_this_existed_is_not_failed(monkeypatch):
    """Every committed row predates ecc_events_observed; none may change."""
    row = _row(None, monkeypatch)
    assert row["Status"] == "PASS"
    assert "ECC" not in row["Detail"]
