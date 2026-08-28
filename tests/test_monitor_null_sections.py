"""neuron-monitor emits explicit JSON null for sections with nothing to report.

Observed on trn1.2xlarge 2026-08-27: once telemetry actually collected samples,
aggregation crashed with
    AttributeError: 'NoneType' object has no attribute 'get'
because dict.get(key, {}) only defaults when the key is ABSENT -- a null value
comes back as None. Every nested lookup must tolerate null, not just missing.
"""

import neuron_monitor


def _monitor_with(samples):
    mon = neuron_monitor.NeuronMonitor(mock=True)
    mon._samples = samples
    return mon


def test_null_latency_stats_do_not_crash_aggregation():
    mon = _monitor_with([{
        "neuron_runtime_data": [{
            "report": {
                "execution_stats": {
                    "latency_stats": {"device_latency": None},
                    "error_summary": None,
                },
                "neuroncore_counters": None,
                "memory_used": None,
            }
        }],
        "system_data": {"neuron_hw_counters": None},
    }])
    out = mon.aggregate()
    assert out["samples"] == 1


def test_every_nested_section_may_be_null():
    mon = _monitor_with([{
        "neuron_runtime_data": [{"report": None}],
        "system_data": None,
    }])
    out = mon.aggregate()
    assert out["samples"] == 1


def test_null_runtime_list_is_treated_as_empty():
    mon = _monitor_with([{"neuron_runtime_data": None, "system_data": None}])
    assert mon.aggregate()["samples"] == 1


def test_real_values_still_aggregate():
    mon = _monitor_with([{
        "neuron_runtime_data": [{
            "report": {
                "execution_stats": {
                    "latency_stats": {"device_latency": {"p50": 0.5, "p99": 0.9}},
                },
            }
        }],
        "system_data": {"neuron_hw_counters": {"neuron_devices": [
            {"mem_ecc_corrected": 3}
        ]}},
    }])
    out = mon.aggregate()
    assert out["samples"] == 1
    assert out["device_latency_seconds"]["p50_mean"] == 0.5
    assert out["ecc_events"]["mem_ecc_corrected"] == 3
