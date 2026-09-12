"""Aggregation against a neuron-monitor sample captured on real hardware.

Recorded on an inf2.xlarge (1 NeuronDevice, 2 NeuronCores-v2, Neuron
runtime 2.30.51, DLAMI 20260227) under a sustained torch-neuronx matmul
load.  Values are verbatim except the identifiers, which are the point of
half these assertions.

Keeping a real sample here means a neuron-monitor schema change breaks a
test instead of silently zeroing a counter in every report.
"""

import json
import pathlib

import neuron_monitor


# Trimmed to one runtime and the fields the aggregator reads, but the
# structure and key names are exactly as emitted.
REAL_SAMPLE = {
    "instance_info": {
        "ami_id": "ami-0fd664467b3cf8dfd",
        "instance_availability_zone": "us-east-1d",
        "instance_availability_zone_id": "use1-az6",
        "instance_id": "i-0dd3c47dea213fd30",
        "instance_name": "",
        "instance_region": "us-east-1",
        "instance_type": "inf2.xlarge",
        "subnet_id": "subnet-07d275b496c32dd6f",
    },
    "neuron_hardware_info": {
        "logical_neuroncore_config": 1,
        "neuron_device_count": 1,
        "neuron_device_memory_size": 34359738368,
        "neuron_device_type": "inferentia2",
        "neuron_device_version": "v3",
        "neuroncore_per_device_count": 2,
        "neuroncore_version": "v2",
    },
    "neuron_runtime_data": [
        {
            "pid": 2199,
            "report": {
                "neuroncore_counters": {
                    "neuroncores_in_use": {
                        "0": {
                            "effective_flops": 875590474948,
                            "neuroncore_utilization": 89.40087251059674,
                        },
                        "1": {"effective_flops": 0, "neuroncore_utilization": 0},
                    },
                    "period": 1.000665759,
                },
                "memory_used": {
                    "neuron_runtime_used_bytes": {
                        "host": 1166249984,
                        # As data/probe-2026-08-26/07-neuron-monitor-schema.txt
                        # records it. This fixture said "device", the name the
                        # parser read -- the same number under a key the
                        # hardware does not use, so the test proved the parser
                        # handled a sample it had been reshaped to fit.
                        "neuron_device": 141649980,
                    }
                },
                "execution_stats": {
                    "error_summary": {
                        "hardware": 0,
                        "model": 0,
                        "numerical": 0,
                        "runtime": 0,
                        "transient": 0,
                    },
                    "execution_summary": {
                        "completed": 816,
                        "completed_with_err": 0,
                        "completed_with_num_err": 0,
                        "failed_to_queue": 0,
                        "incorrect_input": 0,
                        "timed_out": 0,
                    },
                    "latency_stats": {
                        "device_latency": {
                            "p0": 0.0011093616485595703,
                            "p50": 0.0011217594146728516,
                            "p99": 0.0011413097381591797,
                            "p100": 0.0012009143829345703,
                        }
                    },
                },
            },
        }
    ],
    "system_data": {
        "neuron_hw_counters": {
            "neuron_devices": [
                {
                    "neuron_device_index": 0,
                    "mem_ecc_corrected": 0,
                    "mem_ecc_uncorrected": 0,
                    "sram_ecc_corrected": 0,
                    "sram_ecc_uncorrected": 0,
                }
            ]
        }
    },
}


def _aggregate(sample):
    monitor = neuron_monitor.NeuronMonitor()
    monitor._samples = [neuron_monitor._scrub(sample)]
    return monitor.aggregate()


def test_real_sample_is_fully_scrubbed():
    blob = json.dumps(neuron_monitor._scrub(REAL_SAMPLE))
    for identifier in (
        "i-0dd3c47dea213fd30",
        "ami-0fd664467b3cf8dfd",
        "subnet-07d275b496c32dd6f",
        "us-east-1d",
        "use1-az6",
        "inf2.xlarge",
    ):
        assert identifier not in blob, f"leaked {identifier}"


def test_scrub_keeps_the_hardware_description():
    """Device topology is not an identifier and must survive."""
    scrubbed = neuron_monitor._scrub(REAL_SAMPLE)
    hardware = scrubbed["neuron_hardware_info"]
    assert hardware["neuroncore_version"] == "v2"
    assert hardware["neuron_device_version"] == "v3"
    assert hardware["neuron_device_memory_size"] == 34359738368


def test_utilization_and_flops_are_extracted():
    metrics = _aggregate(REAL_SAMPLE)
    assert metrics["neuroncore_utilization"]["0"]["peak"] == 89.4
    assert metrics["effective_flops"]["0"]["peak"] == 875590474948
    # Core 1 was idle; a zero reading must not be recorded as achieved FLOPs.
    assert "1" not in metrics["effective_flops"]


def test_execution_and_latency_are_extracted():
    metrics = _aggregate(REAL_SAMPLE)
    assert metrics["total_executions"] == 816
    assert metrics["execution_errors"] == 0
    assert metrics["device_latency_seconds"]["p50_mean"] == 0.001122


def test_ecc_counters_are_extracted():
    metrics = _aggregate(REAL_SAMPLE)
    assert metrics["ecc_events"] == {
        "mem_ecc_corrected": 0,
        "mem_ecc_uncorrected": 0,
        "sram_ecc_corrected": 0,
        "sram_ecc_uncorrected": 0,
    }
    assert metrics["ecc_events_total"] == 0


def test_ecc_and_execution_failures_are_surfaced():
    """The failure path is what a stress run actually cares about."""
    import copy

    bad = copy.deepcopy(REAL_SAMPLE)
    bad["system_data"]["neuron_hw_counters"]["neuron_devices"][0][
        "sram_ecc_uncorrected"
    ] = 3
    stats = bad["neuron_runtime_data"][0]["report"]["execution_stats"]
    stats["execution_summary"]["timed_out"] = 2
    stats["error_summary"]["hardware"] = 1

    metrics = _aggregate(bad)
    assert metrics["ecc_events_total"] == 3
    assert metrics["execution_errors"] == 3  # 1 hardware + 2 timed_out


# -- the device memory field, which no report ever carried -------------------

def test_the_real_sample_yields_device_memory():
    """It never did. The parser read `neuron_runtime_used_bytes.device`,
    hardware writes `neuron_device`, and nothing asserted the field came
    out -- so device_memory_used_bytes was absent from every report the
    suite has written, while mock runs filled it in because the mock
    emitted the key the parser wanted."""
    metrics = _aggregate(REAL_SAMPLE)
    assert metrics["device_memory_used_bytes"]["peak"] == 141649980


def test_the_older_spelling_still_parses():
    """Kept as a fallback: a runtime that writes `device` is not wrong,
    it is just not what these parts write."""
    monitor = neuron_monitor.NeuronMonitor()
    monitor._samples = [{"neuron_runtime_data": [{"report": {"memory_used": {
        "neuron_runtime_used_bytes": {"device": 4242}}}}]}]
    assert monitor.aggregate()["device_memory_used_bytes"]["peak"] == 4242


def test_the_fixture_uses_the_name_the_probe_recorded():
    """The fixture is only evidence if it matches the capture it came from.

    This one held the probe's number under the parser's key, which is how
    a renamed field passed for the life of the suite.
    """
    probe = pathlib.Path("data/probe-2026-08-26/07-neuron-monitor-schema.txt")
    text = probe.read_text(encoding="utf-8")
    assert "neuron_runtime_used_bytes.neuron_device" in text, (
        "the probe no longer records this field; re-read it before trusting "
        "the fixture")
    used = REAL_SAMPLE["neuron_runtime_data"][0]["report"]["memory_used"]
    assert "neuron_device" in used["neuron_runtime_used_bytes"]


# -- every path the aggregator reads, against the capture --------------------

# The dotted paths neuron_monitor.aggregate walks, in the spelling the
# 2026-08-26 schema probe recorded. Keeping the list here rather than
# deriving it is the point: a rename in the parser has to be reflected
# against the capture by hand, which is the step that was skipped when
# `neuron_device` became `device` and stayed wrong for the life of the
# suite.
MONITOR_PATHS = (
    "neuron_runtime_data[].report.neuroncore_counters.period",
    "neuron_runtime_data[].report.neuroncore_counters.neuroncores_in_use.0.effective_flops",
    "neuron_runtime_data[].report.neuroncore_counters.neuroncores_in_use.0.neuroncore_utilization",
    "execution_stats.period",
    "execution_stats.execution_summary.completed",
    "execution_stats.execution_summary.completed_with_err",
    "execution_stats.execution_summary.failed_to_queue",
    "execution_stats.execution_summary.incorrect_input",
    "latency_stats.device_latency.p50",
    "latency_stats.device_latency.p99",
    "memory_used.neuron_runtime_used_bytes.neuron_device",
    "system_data.neuron_hw_counters.neuron_devices[].mem_ecc_corrected",
    "system_data.neuron_hw_counters.neuron_devices[].sram_ecc_uncorrected",
)


def test_every_path_the_aggregator_reads_is_in_the_capture():
    probe = pathlib.Path(
        "data/probe-2026-08-26/07-neuron-monitor-schema.txt"
    ).read_text(encoding="utf-8")
    missing = [path for path in MONITOR_PATHS if path not in probe]
    assert not missing, (
        f"paths the parser reads that the capture does not record: {missing}")


def test_the_listed_paths_are_the_ones_the_parser_uses():
    """The list is only a check if the parser really reads these names."""
    source = pathlib.Path("neuron_monitor.py").read_text(encoding="utf-8")
    for path in MONITOR_PATHS:
        leaf = path.rsplit(".", 1)[-1]
        assert f'"{leaf}"' in source, f"{leaf} is listed but never read"
