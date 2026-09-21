"""tools/sweep_memory_size.py: the swept size must be the size the row names.

The sweep exists to be compared against pantheongpu's occupancy-sized runs, so
the one thing it cannot do is run one byte count and publish another. A row's
``Problem`` is ``dict(workload.problem)``; these check that the override
changes exactly that and refuses anything the kernel would round.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import sweep_memory_size  # noqa: E402
from kernels import registry, tiling  # noqa: E402

GIB = 1024 ** 3


@pytest.fixture
def restore_registry(monkeypatch):
    """override() writes registry._BY_NAME; put every entry back afterwards."""
    for name in sweep_memory_size.SWEEPABLE:
        monkeypatch.setitem(registry._BY_NAME, name, registry._BY_NAME[name])


def test_sizes_parse_as_whole_gib():
    assert sweep_memory_size.sizes_from("1, 2,12") == [GIB, 2 * GIB, 12 * GIB]


@pytest.mark.parametrize("text", ["", "0", "-1", "1.5", "big"])
def test_sizes_that_are_not_whole_positive_gib_are_refused(text):
    with pytest.raises(ValueError):
        sweep_memory_size.sizes_from(text)


@pytest.mark.parametrize("name", sweep_memory_size.SWEEPABLE)
def test_override_changes_what_the_registry_resolves(name, restore_registry):
    pinned = dict(registry.resolve(name)[0].problem)
    sweep_memory_size.override(name, 2 * GIB)
    swept = registry.resolve(name)[0].problem
    assert swept["bytes"] == 2 * GIB
    # Only the size moves; dtype and cores are still the declared ones.
    assert {k: v for k, v in swept.items() if k != "bytes"} == \
        {k: v for k, v in pinned.items() if k != "bytes"}


@pytest.mark.parametrize("name", sweep_memory_size.SWEEPABLE)
def test_every_default_size_is_a_whole_number_of_tiles(name):
    """Otherwise the default sweep would refuse itself on hardware."""
    workload = registry.resolve(name)[0]
    for total in sweep_memory_size.sizes_from("1,2,4,8,12"):
        assert sweep_memory_size.swept_problem(workload, total)["bytes"] == total


def test_a_size_the_kernel_would_round_is_refused():
    workload = registry.resolve("memory_read")[0]
    tile = tiling.tile_plan(GIB, workload.problem["dtype"])["tile_bytes"]
    with pytest.raises(ValueError, match="whole number"):
        sweep_memory_size.swept_problem(workload, GIB + tile // 2)


def test_a_workload_that_pins_no_byte_count_is_refused():
    with pytest.raises(ValueError, match="does not pin"):
        sweep_memory_size.swept_problem(registry.resolve("tensor_virus")[0], GIB)


def test_the_pinned_problems_are_untouched_by_importing_the_tool():
    assert registry.resolve("memory_read")[0].problem["bytes"] == 8 * GIB
    assert registry.resolve("memory_write")[0].problem["bytes"] == 4 * GIB
