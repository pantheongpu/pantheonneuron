"""A check's evidence has to reach the row the check decided.

The whitelist already says so for kv_cache_churn's ring counts ("the check
acts on these; the row has to show them") and, since #46, for the GEMM
row-tile counts. Three more checks acted on counts no report carried:

* memory_*_agg's two worker checks read `per_core` -- each worker's
  bandwidth, bytes and coverage -- and the aggregate's own docstring said
  "the row carries the evidence". It did not: the 2026-09-21 rows publish
  the summed bytes and nothing per core.
* graph_replay compares replays run against replays requested, and
  published neither.
* serving_mix's Score is prefills plus completed decode requests, flagged
  in the README as quantised to one request per 32 decode steps, and the
  counts that quantisation is made of never reached a row.

The test below derives the list rather than restating it: every argument a
kernel passes to a verify_* function by a name that is also one of that
kernel's result keys must be published, unless it is declared derivable.
"""

import ast
import os

import pytest

import pantheon_neuron

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Check inputs the row does not need because a published field already
# determines them. Declared with the field that does, so the exemption is
# checkable rather than a shrug.
DERIVABLE = {
    ("graph_replay", "plan"): "passed to the check and not read by it",
    ("inference_mix", "period"): "Problem carries prefill_ratio, which fixes the period",
    ("pulse_virus", "loaded_s"): "observed_duty is loaded_s over the published elapsed_s",
}


def _check_inputs():
    """(module, argument) for every verify_* call argument that is a result key."""
    found = set()
    kernels = os.path.join(ROOT, "kernels")
    for name in sorted(os.listdir(kernels)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(kernels, name), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        keys = {key.value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            label = (getattr(node.func, "id", None)
                     or getattr(node.func, "attr", "") or "")
            if not label.startswith("verify_"):
                continue
            for argument in node.args:
                if isinstance(argument, ast.Name) and argument.id in keys:
                    found.add((name[:-3], argument.id))
    return sorted(found)


@pytest.mark.parametrize("module, key", _check_inputs())
def test_every_check_input_reaches_the_row(module, key):
    if (module, key) in DERIVABLE:
        return
    assert key in pantheon_neuron._PROVENANCE_KEYS, (
        f"{module} passes {key} to a check but the row never shows it; "
        "whitelist it in _PROVENANCE_KEYS (and bump REPORT_SCHEMA), or "
        "declare in DERIVABLE which published field determines it")


def test_the_scan_finds_the_checks_it_was_written_for():
    """A scan that finds nothing passes everything."""
    inputs = set(_check_inputs())
    for expected in [("memory_agg", "per_core"), ("graph_replay", "replays"),
                     ("inference_mix", "decode_requests"),
                     ("llm_inference", "steps")]:
        assert expected in inputs


def test_no_exemption_is_stale():
    inputs = set(_check_inputs())
    assert set(DERIVABLE) <= inputs, set(DERIVABLE) - inputs


def test_per_core_survives_to_the_report(monkeypatch):
    workload = pantheon_neuron.registry.resolve("memory_read_agg")[0]
    per_core = [{"core": 0, "gbps": 272.1, "bytes": 10, "verified_ratio": 1.0},
                {"core": 1, "gbps": 271.6, "bytes": 10, "verified_ratio": 1.0}]
    monkeypatch.setitem(pantheon_neuron._LAST_RUN, workload.name,
                        {"per_core": per_core, "bytes_moved": 20})
    assert pantheon_neuron._provenance(workload)["per_core"] == per_core


def test_serving_mix_publishes_what_its_score_is_made_of(monkeypatch):
    workload = pantheon_neuron.registry.resolve("serving_mix")[0]
    monkeypatch.setitem(pantheon_neuron._LAST_RUN, workload.name,
                        {"prefills": 60, "decode_requests": 14,
                         "decode_steps": 448, "requests_completed": 74})
    row = pantheon_neuron._provenance(workload)
    assert (row["prefills"], row["decode_requests"], row["decode_steps"]) == (60, 14, 448)


def test_graph_replay_publishes_run_against_requested(monkeypatch):
    workload = pantheon_neuron.registry.resolve("graph_replay")[0]
    monkeypatch.setitem(pantheon_neuron._LAST_RUN, workload.name,
                        {"replays": 90000, "requested_replays": 100000})
    row = pantheon_neuron._provenance(workload)
    assert (row["replays"], row["requested_replays"]) == (90000, 100000)
