"""The neuron-profile ``_percent`` counters are fractions.

Every counter ending in ``_percent`` reports a 0-1 fraction. Measured on
trn1.2xlarge 2026-09-10 against a torch.matmul at 4096^3 bf16 that the
same run clocked at 51.8 TFLOPS.

A probe read them as percentages, printed "tensor engine active: NKI
0.379%", and concluded the engines were comparable. On the right scale
they said the opposite. Nothing in production consumes these counters
yet, which is exactly when a units guard is cheapest to write.
"""

import os

from kernels import omni_virus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Measured values, torch.matmul 4096^3 bf16, trn1.2xlarge 2026-09-10.
MEASURED_XLA = {
    "tensor_engine_active_time_percent": 0.4510375568931359,
    "dma_active_time_percent": 0.2046196032555904,
    "mfu_estimated_percent": 0.3783779707219885,
    "mfu_max_achievable_estimated_percent": 1,
}


def test_every_percent_counter_measured_is_on_a_zero_to_one_scale():
    """If any of these were a real percentage, a maximum-achievable MFU
    of 1 would mean one percent -- and a matmul at 51.8 TFLOPS would
    have had its tensor engine busy 0.45% of the time."""
    for name, value in MEASURED_XLA.items():
        assert name.endswith("_percent"), name
        assert 0.0 <= value <= 1.0, (name, value)


def test_a_maximum_achievable_of_one_is_the_giveaway():
    """The single reading that settles it: nothing is 1% achievable."""
    assert MEASURED_XLA["mfu_max_achievable_estimated_percent"] == 1


def test_the_one_consumer_is_told_the_scale():
    """omni_virus.engine_activity is the function that will first read
    these, whenever it is wired to a profiler capture. Its docstring has
    to carry the scale, or the first reader repeats the probe's error."""
    doc = " ".join(omni_virus.engine_activity.__doc__.split())
    assert "0-1 fractions" in doc
    assert "Multiply by 100" in doc


def test_the_counter_reference_records_it():
    with open(os.path.join(ROOT, "docs", "neuron_counters.md"),
              encoding="utf-8") as handle:
        doc = " ".join(handle.read().split())
    assert "The `_percent` counters are fractions" in doc
    # \u2013 rather than a literal en dash: the doc uses one, and ruff
    # (RUF001) rightly flags an ambiguous dash in source.
    assert "0\u20131 fraction, not a percentage" in doc
