"""The comparison tooling the registers were written for.

`registry.NOT_COMPARABLE_WITH_GPU` and `SAME_UNIT_DIFFERENT_QUANTITY` say they
exist so "the comparison tooling can render an honest 'not comparable'". For
most of this repo's life nothing compared anything: the registers were read by
the documentation generator and by no tool, and the one cross-platform table
published was typed by hand from medians computed in a shell.

These check the tool that closes that, and -- the part that matters -- that
the published table stays what the tool renders.
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import compare_with_gpu  # noqa: E402
from kernels import registry  # noqa: E402

REFERENCE = compare_with_gpu.load("gpu-reference.json")
README = os.path.join(ROOT, "data", "publication-2026-09-21", "README.md")


def test_the_published_table_is_what_the_tool_renders():
    """The transcription failure, closed: prose cannot drift from the data."""
    with open(README, encoding="utf-8") as handle:
        published = handle.read()
    assert compare_with_gpu.render(REFERENCE) in published, (
        "data/publication-2026-09-21/README.md is out of date with the data. "
        "Run: python tools/compare_with_gpu.py --render")


def test_rendering_twice_gives_the_same_table():
    assert compare_with_gpu.render(REFERENCE) == compare_with_gpu.render(REFERENCE)


def test_the_tool_runs_and_renders_the_same_table():
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "compare_with_gpu.py"), "--render"],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip() == compare_with_gpu.render(REFERENCE)


def test_every_flagged_name_is_refused_with_its_reason():
    """Whatever the registers hold, the tool must refuse it and say why."""
    refused = {name: (kind, why) for name, kind, why in compare_with_gpu.refusals()}
    for name in registry.NOT_COMPARABLE_WITH_GPU:
        assert refused[name][0] == "units diverge"
    for name, declared in registry.SAME_UNIT_DIFFERENT_QUANTITY.items():
        assert refused[name][0] == "same unit, different quantity"
        assert refused[name][1] == declared, "the reason must be the registry's own"
    assert len(refused) == (len(registry.NOT_COMPARABLE_WITH_GPU)
                            + len(registry.SAME_UNIT_DIFFERENT_QUANTITY))


def test_no_flagged_workload_appears_in_the_comparison():
    """A refused name must not reach the table by another route."""
    table = compare_with_gpu.render(REFERENCE)
    flagged = set(registry.NOT_COMPARABLE_WITH_GPU) | set(registry.SAME_UNIT_DIFFERENT_QUANTITY)
    for name in flagged:
        assert name not in table


def test_the_pairing_crosses_names_and_says_so():
    """The comparison joins memory_*_agg here to memory_* there, deliberately."""
    ours = {pair[0] for pair in compare_with_gpu.DEVICE_PAIRING}
    theirs = {pair[1] for pair in compare_with_gpu.DEVICE_PAIRING}
    assert ours == {"memory_read_agg", "memory_write_agg"}
    assert theirs == {"memory_read", "memory_write"}
    assert ours & theirs == set(), "if the names matched, a plain join would do"
    for _ours, _theirs, why in compare_with_gpu.DEVICE_PAIRING:
        assert "whole GPU" in why and len(why) > 40


def test_a_part_with_no_published_peak_gets_no_percentage():
    """The A10G: AWS-specific, no vendor figure. A borrowed one would be a lie."""
    entry = REFERENCE["parts"]["NVIDIA A10G"]
    assert entry["published_peak_gbps"] is None
    assert "no published figure" in entry["published_peak_source"]
    row = next(line for line in compare_with_gpu.render(REFERENCE).splitlines()
               if line.startswith("| A10G "))
    assert "*none published*" in row and "%" not in row


@pytest.mark.parametrize("name", sorted(compare_with_gpu.HEADLINE_GPUS))
def test_every_headline_part_is_tallied_with_a_peak_or_a_stated_reason(name):
    entry = REFERENCE["parts"][name]
    assert entry["reports"] > 0 and entry["rows"], f"{name} has no measured rows"
    peak, source = entry["published_peak_gbps"], entry["published_peak_source"]
    assert source and source != "not looked up", f"{name}: cite the figure or say none"
    if peak is None:
        assert "no published figure" in source or "not established" in source
    else:
        assert peak > 0 and ("NVIDIA" in source or "AWS" in source)


def test_the_neuron_side_comes_from_the_registry_peaks():
    """Not a number typed into the tool: the same peak the Scores are judged by."""
    parts = {p["label"]: p for p in compare_with_gpu.neuron_devices()}
    assert parts["Trainium1"]["peak_gbps"] == registry.PART_PEAKS["trn1"]["hbm_gbps"]
    assert parts["Inferentia2"]["peak_gbps"] == registry.PART_PEAKS["inf2"]["hbm_gbps"]
    # And the disagreement between AWS's own pages travels with it.
    assert "7.4%" in parts["Trainium1"]["peak_source"]


def test_the_reference_records_how_it_was_tallied():
    assert REFERENCE["tallied"] == "2026-09-22"
    assert set(REFERENCE["rows_tallied"]) >= {"memory_read", "memory_write"}
    for name, entry in REFERENCE["parts"].items():
        assert entry["versions"], f"{name}: which pantheongpu versions produced this?"
        for row in entry["rows"].values():
            assert row["samples"] > 0 and row["median_gbps"] > 0
