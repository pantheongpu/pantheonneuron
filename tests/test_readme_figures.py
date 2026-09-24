"""The README's Measured column is the published dataset, not a transcription.

Two figures in the Kernel status table had been typed from earlier single
runs and sat 9.0% (pcie_bandwidth) and 5.5% (allocation_fragmentation) from
the 2026-09-21 medians, with no date to say they were a different
measurement. Three named units the registry does not use, so they would not
join on (Test Name, Unit). tools/readme_measurements.py now renders the
column; this keeps it rendered.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import readme_measurements as figures  # noqa: E402


def test_the_measured_column_matches_the_dataset():
    found = figures.drift()
    assert not found, "\n".join(
        f"{name}: README {shown!r}, dataset {want!r}"
        for name, _column, shown, want in found
    ) + "\nrun: python tools/readme_measurements.py --write"


def test_every_registry_workload_has_a_row():
    rows = {name for _i, name, _m, _s in figures.cells(
        figures.read_readme())}
    registered = {w.name for w in figures.registry.WORKLOADS}
    assert rows == registered


def test_a_hand_edited_figure_is_caught():
    text = figures.read_readme()
    edited = text.replace("| 3.68 GB/s |", "| 4.01 GB/s |", 1)
    assert edited != text, "the pcie_bandwidth cell was not where expected"
    names = [name for name, *_ in figures.drift(edited)]
    assert names == ["pcie_bandwidth"]


def test_a_unit_the_registry_does_not_use_is_caught():
    text = figures.read_readme()
    edited = text.replace("allocation-events/s |", "events/s |", 1)
    assert edited != text
    assert [n for n, *_ in figures.drift(edited)] == ["allocation_fragmentation"]


def test_a_wrong_score_source_is_caught():
    text = figures.read_readme()
    # The Kernel status table's row; the README has three other tables with
    # a memory_read row, and editing the first one tests nothing.
    start = text.index("| `memory_read` |", text.index(figures.HEADER))
    line_end = text.index("\n", start)
    line = text[start:line_end]
    edited = text[:start] + line.replace("`neuron-profile`", "workload") + text[line_end:]
    found = figures.drift(edited)
    assert ("memory_read", "Score source") in [(n, c) for n, c, *_ in found]


def test_a_workload_the_reference_never_scored_shows_a_dash():
    assert figures.expected_cell("all_reduce", figures.reference()) == "—"


@pytest.mark.parametrize("value, shown", [
    (1.845e13, "1.845e+13"),
    (97348.37, "97,348.4"),
    (272.98, "272.98"),
    (2.4805, "2.48"),
])
def test_display(value, shown):
    assert figures.display(value) == shown


def test_rewriting_is_idempotent():
    text = figures.read_readme()
    assert figures.rewrite(text) == text
