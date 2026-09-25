"""Which device each part of tools/validate_hardware.sh runs on.

Until 2026-09-25 no part this account could launch had two chips, so the
script never passed --device and every run took them all -- which on one
chip is the same thing. On inf2.24xlarge (six Inferentia2) it is not:
memory_read_agg and memory_write_agg declare cores "all" and would report
the whole instance's bandwidth under the name the reference dataset uses
for one chip. So single-chip workloads are pinned to one device, and the
collectives -- the only workloads whose subject is the link between chips --
span all of them, and only where there are two or more to span.

The collectives section is run for real here, against a stub harness and
a stated device count.
"""

import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tools", "validate_hardware.sh")


def _text():
    with open(SCRIPT, encoding="utf-8") as handle:
        return handle.read()


def _collectives_section():
    text = _text()
    start = text.index("COLLECTIVES=${COLLECTIVES:-")
    end = text.index("\nfi\n", start) + len("\nfi\n")
    return text[start:end]


def _run_section(device_count):
    script = "\n".join([
        'hr() { echo "== $*"; }',
        'run_one() { echo "RUN $*"; }',
        f"DEVICE_COUNT={device_count}",
        "DURATION=30", "REPEAT=1", "WORKLOAD_TIMEOUT=60",
        _collectives_section(),
    ])
    return subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, timeout=30).stdout


def test_on_a_multi_chip_part_both_collectives_span_every_chip():
    out = _run_section(6)
    runs = [line for line in out.splitlines() if line.startswith("RUN ")]
    assert [run.split()[1] for run in runs] == ["all_reduce", "p2p_thrasher"]
    assert all("--device all" in run for run in runs)
    assert "across all 6 devices" in out


def test_on_a_single_chip_part_they_are_skipped_and_the_pass_says_so():
    out = _run_section(1)
    assert "RUN " not in out
    assert "skipped: 1 Neuron device(s) here" in out


def test_with_no_neuron_device_found_they_are_skipped_too():
    assert "RUN " not in _run_section(0)


def test_every_single_chip_run_is_pinned_to_one_device():
    text = _text()
    assert "DEVICE=${DEVICE:-0}" in text
    # Join shell line continuations first, so a call is one line.
    joined = text.replace("\\\n", " ")
    calls = re.findall(r"^\s*run_one [^\n]*", joined, re.M)
    single = [call for call in calls if "--device all" not in call]
    assert len(single) >= 2, "expected the orchestrated and warm-cache runs"
    assert all('--device "$DEVICE"' in call for call in single), single


def test_the_device_count_reads_the_sysfs_tree_the_part_section_reads():
    count_line = next(line for line in _text().splitlines()
                      if line.startswith("DEVICE_COUNT="))
    assert "/sys/devices/virtual/neuron_device" in count_line
