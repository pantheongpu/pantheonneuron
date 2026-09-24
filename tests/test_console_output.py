"""What a reader watching the run actually sees.

The suite printed fourteen lines and not one of them was a Score. A
forty-minute pass showed `PASS` per workload, a share of peak for the third
of the suite that has a published peak, and a path -- so the number the
reader ran the suite to get was only in the JSON. There was no summary
either: answering "did this part pass?" meant scrolling back through
twenty-odd rows, for a run whose exit status already knew.

These drive the real loop rather than a copy of it: `run_workload` is
replaced with canned rows and `_run_selection` is called, which is the
function main() delegates to.
"""

import types

import pytest

import pantheon_neuron
from neuron_device import NeuronDevice

_DEVICES = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]
_ARGS = types.SimpleNamespace(duration=1, monitor_period=1.0, repeat=1)


def _run(rows, monkeypatch, capsys):
    """Run the console loop over canned rows and return what it printed."""
    queue = list(rows)
    monkeypatch.setattr(pantheon_neuron, "run_workload",
                        lambda *a, **k: queue.pop(0))
    workloads = pantheon_neuron.registry.WORKLOADS[:len(rows)]
    results = []
    pantheon_neuron._run_selection(workloads, None, _DEVICES, _ARGS, results)
    return capsys.readouterr().out, results


def _row(name="tensor_virus", status="PASS", score=78.6013, unit="TFLOPS",
         detail="", peak=None, repeats=None):
    return {"Test Name": name, "Status": status, "Score": score, "Unit": unit,
            "Detail": detail, "Percent Of Peak": None, "Peak": peak,
            "Repeats": repeats}


# -- the measurement reaches the console -------------------------------------

def test_the_score_is_printed_with_its_unit(monkeypatch, capsys):
    printed, _ = _run([_row()], monkeypatch, capsys)
    assert "78.6013 TFLOPS" in printed


def test_a_workload_with_no_published_peak_still_shows_its_number(
        monkeypatch, capsys):
    """Most of the suite has no peak, so the share line never fires for it
    and `PASS` was the whole of what those rows printed."""
    printed, _ = _run([_row(name="moe_router", score=255242.28,
                            unit="tokens/s")], monkeypatch, capsys)
    assert "255242.2800 tokens/s" in printed


def test_a_row_without_a_score_prints_no_number(monkeypatch, capsys):
    printed, _ = _run([_row(score=None)], monkeypatch, capsys)
    assert "TFLOPS" not in printed


def test_a_skipped_row_prints_no_number(monkeypatch, capsys):
    printed, _ = _run([_row(status="SKIPPED", score=None,
                            detail="needs 2 devices")], monkeypatch, capsys)
    assert "SKIPPED" in printed
    assert "TFLOPS" not in printed


@pytest.mark.parametrize("value, shown", [
    (1.845e13, "1.845e+13"),      # quantized_gemm, eleven orders up
    (2.4805, "2.4805"),           # serving_mix
    (272.9804, "272.9804"),
    (0.0, "0.0000"),
])
def test_the_figure_fits_a_console_line(value, shown):
    assert pantheon_neuron.figure(value) == shown


def test_a_figure_that_is_not_a_number_says_so():
    assert pantheon_neuron.figure(None) == "--"


# -- the verdict at the end --------------------------------------------------

def test_the_summary_tallies_every_status(capsys):
    pantheon_neuron.print_summary([
        _row(name="a"), _row(name="b"),
        _row(name="c", status="FAIL", score=None, detail="the Score was zero"),
        _row(name="d", status="SKIPPED", score=None, detail="needs 2 devices"),
    ])
    printed = capsys.readouterr().out
    assert "4 workload(s): 2 PASS, 1 FAIL, 1 SKIPPED" in printed


def test_a_failure_is_named_not_counted(capsys):
    pantheon_neuron.print_summary(
        [_row(name="pcie_bandwidth", status="FAIL", score=None,
              detail="the landing buffer still reads 0; and more")])
    printed = capsys.readouterr().out
    assert "FAIL pcie_bandwidth: the landing buffer still reads 0" in printed
    # Only the first clause; a Detail can run to several sentences.
    assert "and more" not in printed


def test_skips_are_named_too(capsys):
    pantheon_neuron.print_summary([
        _row(name="all_reduce", status="SKIPPED", score=None),
        _row(name="p2p_thrasher", status="SKIPPED", score=None),
    ])
    assert "skipped: all_reduce, p2p_thrasher" in capsys.readouterr().out


def test_a_clean_pass_says_so_without_naming_anything(capsys):
    pantheon_neuron.print_summary([_row(name="a"), _row(name="b")])
    printed = capsys.readouterr().out
    assert "2 workload(s): 2 PASS" in printed
    assert "FAIL" not in printed and "skipped" not in printed


def test_an_empty_run_prints_nothing(capsys):
    pantheon_neuron.print_summary([])
    assert capsys.readouterr().out == ""


def test_a_failure_with_no_detail_still_names_the_workload(capsys):
    pantheon_neuron.print_summary(
        [_row(name="graph_replay", status="FAIL", score=None, detail="")])
    assert "FAIL graph_replay: no detail" in capsys.readouterr().out


# -- and it is the run that does it, not only these tests --------------------

def test_the_summary_runs_for_a_finished_pass_and_an_interrupted_one():
    """Both exits from main(): a pass that completed and one stopped by
    hand, which already wrote a partial report and printed nothing about
    what was in it."""
    import sourcecheck
    code = sourcecheck.flat_function_code(pantheon_neuron.main)
    assert code.count("print_summary ( results )") == 2
    assert code.index("print_summary ( results )") < code.index("raise")
