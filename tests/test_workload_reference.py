"""The generated workload reference must match the registry it claims to render.

`docs/workload_counter_map.md` says "Generated from kernels/registry.py ... do
not hand-edit". Nothing enforced either half of that until this file existed:

- A registry change that nobody regenerated for left the reference describing
  a suite that no longer exists. It did: the header claimed only
  `baseline_metrics` was implemented long after all 26 workloads had kernels.
- Prose written *into* the Markdown was silently deleted by the next person to
  run the generator. It was: four hand-written sections, ~27 lines, gone on
  the next regeneration with no error anywhere.

So the rendered text is compared against the committed file here, and the
narrative sections live in `tools/gen_workload_table.py` where regeneration
preserves them.
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import gen_workload_table  # noqa: E402

from kernels import registry  # noqa: E402


def _committed() -> str:
    with open(gen_workload_table.TARGET, encoding="utf-8") as handle:
        return handle.read()


def test_the_committed_reference_is_what_the_generator_renders():
    """Regenerating must be a no-op, or the reference has drifted."""
    assert gen_workload_table.render() == _committed(), (
        "docs/workload_counter_map.md is out of date with kernels/registry.py. "
        "Run: python tools/gen_workload_table.py"
    )


def test_rendering_is_idempotent():
    assert gen_workload_table.render() == gen_workload_table.render()


def test_running_the_generator_leaves_the_file_unchanged(tmp_path):
    """The script itself, not just render(), agrees with what is committed."""
    before = _committed()
    completed = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "gen_workload_table.py")],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert _committed() == before


def test_every_workload_appears_in_the_table():
    rendered = gen_workload_table.render()
    for workload in registry.WORKLOADS:
        assert f"| `{workload.name}` |" in rendered, workload.name


def test_the_narrative_sections_survive_regeneration():
    """The prose lives in the generator, so it cannot be regenerated away.

    Asserted by heading rather than by wording: this guards the mechanism,
    not the phrasing, so editing a paragraph in the generator does not fail
    a test that has nothing to say about it.
    """
    rendered = gen_workload_table.render()
    for heading in (
        "## How a monitor-sourced Score is read",
        "## The declared profiler Score has never been produced by a run",
        "## Reserving the core costs a selection",
        "## Where the comparison does not hold",
    ):
        assert heading in rendered, heading


def test_the_header_does_not_understate_what_is_implemented():
    """The stale claim this file was written for: catch it if it comes back."""
    rendered = gen_workload_table.render()
    assert "Only `baseline_metrics` is implemented" not in rendered


@pytest.mark.parametrize("name", sorted(registry.NOT_COMPARABLE_WITH_GPU))
def test_not_comparable_workloads_are_named_in_the_reference(name):
    """The list is derived from the registry, so it cannot list the old set."""
    assert f"`{name}`" in gen_workload_table.render()
