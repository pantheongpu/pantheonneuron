"""A row has to carry the label that decides whether its number is comparable.

tensor_virus says it plainly: the GEMM tilings differ by ~2.7x at the pinned
shape, `PANTHEON_NEURON_GEMM_TILING` selects between them, and "a number
without this label is not comparable with one that has it". No published row
carried the label -- the word `tiling` appears nowhere in any of the 2026-09-21
tensor_virus rows.

The same held for the controls pcie_bandwidth runs and then reports: whether
the host pages were really pinned decides whether the staging-buffer
explanation for the d2h asymmetry is tested or still open, and the answer
reached no report.
"""

import pytest

import pantheon_neuron
from kernels import pcie_bandwidth, pulse_virus, tensor_virus


@pytest.mark.parametrize("key", [
    "tiling", "row_tiles_checked", "row_tiles_wrong",
    "host_source_pinned", "host_landing_pinned", "h2d_sources_alternate",
])
def test_the_label_reaches_the_row(key):
    assert key in pantheon_neuron._PROVENANCE_KEYS


def _provenance(workload_name, result, monkeypatch):
    workload = next(w for w in pantheon_neuron.registry.WORKLOADS
                    if w.name == workload_name)
    monkeypatch.setitem(pantheon_neuron._LAST_RUN, workload_name, result)
    return pantheon_neuron._provenance(workload) or {}


def test_the_tiling_a_run_used_is_what_the_row_publishes(monkeypatch):
    """Not the registry's default: the variable can change it, and then the
    row must say what ran rather than what was pinned."""
    row = _provenance("tensor_virus", {"tiling": "streaming"}, monkeypatch)
    assert row["tiling"] == "streaming"


def test_every_tiling_the_kernel_accepts_can_be_published(monkeypatch):
    for strategy in tensor_virus.STRATEGIES:
        row = _provenance("tensor_virus", {"tiling": strategy}, monkeypatch)
        assert row["tiling"] == strategy


def test_a_clean_row_tile_check_still_shows_its_count(monkeypatch):
    """Zero wrong is the interesting case: it says the check ran."""
    row = _provenance("tensor_virus",
                      {"row_tiles_checked": 64, "row_tiles_wrong": 0},
                      monkeypatch)
    assert row["row_tiles_checked"] == 64
    assert row["row_tiles_wrong"] == 0


def test_unpinned_host_memory_is_reported_as_such(monkeypatch):
    """False is the value that matters -- it says the bounce-buffer
    explanation for the d2h asymmetry is still untested."""
    row = _provenance("pcie_bandwidth",
                      {"host_source_pinned": False, "host_landing_pinned": False,
                       "h2d_sources_alternate": True},
                      monkeypatch)
    assert row["host_source_pinned"] is False
    assert row["host_landing_pinned"] is False
    assert row["h2d_sources_alternate"] is True


@pytest.mark.parametrize("module, key", [
    (tensor_virus, "tiling"),
    (pulse_virus, "tiling"),
    (tensor_virus, "row_tiles_wrong"),
    (pulse_virus, "row_tiles_wrong"),
    (pcie_bandwidth, "host_source_pinned"),
])
def test_the_kernel_actually_reports_the_key(module, key):
    """A whitelist entry for a key no kernel writes is a key that never
    appears; the whitelist is checked against the kernels for exactly that."""
    import sourcecheck
    with open(module.__file__, encoding="utf-8") as handle:
        code = sourcecheck.code_only(handle.read())
    assert f'"{key}"' in code
