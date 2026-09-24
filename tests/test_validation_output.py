"""What tools/validate_hardware.sh shows from each run, executed rather than read.

It piped each run through `tail -6`, and the warm-cache search through
`tail -4` -- a count of lines in a stream whose length depends on the row.
When the harness started printing the Score and a closing tally, `tail -4`
began dropping the status line, which on the warm-cache search carries the
Detail that says whether the profiled graph was ours. Nothing failed; the
evidence stopped arriving, on a pass that costs money to run.

These pull `run_one` out of the script itself and run it against a stub in
place of the harness, so they test the function the pass runs rather than a
description of it.
"""

import os
import re
import stat
import subprocess
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tools", "validate_hardware.sh")


def _function_and_noise():
    with open(SCRIPT, encoding="utf-8") as handle:
        text = handle.read()
    noise = re.search(r"^NOISE=.*$", text, re.M).group(0)
    body = re.search(r"^run_one\(\) \{.*?^\}", text, re.M | re.S).group(0)
    return noise, body


def _stub(tmp_path, body):
    """An executable standing in for the venv's python."""
    path = tmp_path / "python"
    path.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _run(tmp_path, stub_body, limit=10):
    noise, function = _function_and_noise()
    stub = _stub(tmp_path, stub_body)
    logs = tmp_path / "logs"
    logs.mkdir()
    script = "\n".join([
        "set -uo pipefail",
        f"PY={stub}",
        f"LOGDIR={logs}",
        noise,
        function,
        f"run_one memory_read {limit} --test memory_read",
    ])
    done = subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, cwd=tmp_path, timeout=60)
    return done.stdout, logs / "memory_read.log"


_HARNESS = """
echo "[PANTHEON-NEURON] 1 device(s) [trn1], 1 workload(s)"
for i in $(seq 1 40); do echo "neuronx-cc: compiling graph $i"; done
echo "CCOM WARN something the filter drops"
echo "[PANTHEON-NEURON] -> memory_read"
echo "[PANTHEON-NEURON]    PASS (the profiled graph could not be attributed to this kernel)"
echo "[PANTHEON-NEURON]    272.9800 GB/s"
echo "[PANTHEON-NEURON]    31.0% of 880.5 GB/s across 1 of 2 core(s)"
echo "[PANTHEON-NEURON]    3 repeats: 272.9 to 273.1, median 272.98, cv 0.0003"
echo "[PANTHEON-NEURON] 1 workload(s): 1 PASS"
echo "[PANTHEON-NEURON] report: database/pantheon_neuron_report_X.json"
"""


def test_the_status_line_survives_however_long_the_row_is(tmp_path):
    """The line tail -4 dropped."""
    shown, _log = _run(tmp_path, _HARNESS + "exit 0\n")
    assert "PASS (the profiled graph could not be attributed" in shown
    assert "272.9800 GB/s" in shown
    assert "1 workload(s): 1 PASS" in shown


def test_compiler_output_stays_in_the_log_not_on_the_terminal(tmp_path):
    shown, log = _run(tmp_path, _HARNESS + "exit 0\n")
    assert "compiling graph" not in shown
    assert "CCOM WARN" not in shown
    assert "compiling graph 40" in log.read_text()


def test_a_fail_row_is_reported_without_a_raw_dump(tmp_path):
    """Exit 1 is a FAIL row, which the harness lines explain."""
    shown, _log = _run(tmp_path, textwrap.dedent("""
        echo "[PANTHEON-NEURON]    FAIL (the Score was zero GB/s)"
        echo "neuronx-cc: noise"
        exit 1
    """))
    assert "FAIL (the Score was zero GB/s)" in shown
    assert "exited 1 (a FAIL row" in shown
    assert "its last lines" not in shown


def test_a_crash_shows_the_lines_that_explain_it(tmp_path):
    """A traceback prints no harness line, so without this a crashed run
    would show nothing at all."""
    shown, _log = _run(tmp_path, textwrap.dedent("""
        echo "Traceback (most recent call last):"
        echo "  File \\"pantheon_neuron.py\\", line 1"
        echo "CCOM WARN noise"
        echo "RuntimeError: NRT init failed"
        exit 3
    """))
    assert "exited 3; its last lines" in shown
    assert "RuntimeError: NRT init failed" in shown
    assert "CCOM WARN" not in shown


def test_a_run_that_hangs_is_reported_as_timed_out(tmp_path):
    shown, _log = _run(tmp_path, "sleep 30\n", limit=1)
    assert "TIMED OUT after 1s" in shown


def test_the_log_path_is_printed_so_the_evidence_can_be_found(tmp_path):
    shown, log = _run(tmp_path, _HARNESS + "exit 0\n")
    assert f"full output: {log}" in shown
    assert log.exists()


def test_no_run_is_trimmed_by_line_count_any_more():
    """Neither pipe survives anywhere a harness run is shown."""
    with open(SCRIPT, encoding="utf-8") as handle:
        text = handle.read()
    runs = [line for line in text.splitlines()
            if "pantheon_neuron.py" in line and not line.lstrip().startswith("#")]
    assert runs, "no harness invocation found"
    assert all("tail" not in line for line in runs)
    assert text.count("run_one ") >= 2


@pytest.mark.parametrize("label", ["orchestrated", "warm-cache"])
def test_both_passes_go_through_run_one(label):
    with open(SCRIPT, encoding="utf-8") as handle:
        text = handle.read()
    marker = ("for workload in $ORCHESTRATED" if label == "orchestrated"
              else "NEFF search against a warm compile cache")
    section = text[text.index(marker):]
    assert "run_one" in section[:800]
