"""Every runnable workload must be in the hardware validation pass.

`tools/validate_hardware.sh` is the only thing that ever runs this suite on a
device, and until 2026-09-08 its list held four names while the registry held
26. Eighteen workloads had therefore never executed on hardware at all --
which is how `omni_virus` shipped a call to a `_read_back` it did not have,
a NameError that only its first real run could have found.

A workload absent from that list is a workload nobody is checking. So the
list is compared against the registry here, and adding a workload without
adding it to the pass fails the build rather than quietly going unrun.
"""

import os
import re

import pytest

import pantheon_neuron
from kernels import registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tools", "validate_hardware.sh")


def _script() -> str:
    with open(SCRIPT, encoding="utf-8") as handle:
        return handle.read()


def orchestrated_names() -> set:
    """The workloads the validation pass actually runs."""
    text = _script()
    match = re.search(r'ORCHESTRATED=\$\{ORCHESTRATED:-"(.*?)"\}', text, re.S)
    assert match, "could not find the ORCHESTRATED list in validate_hardware.sh"
    return set(match.group(1).split())


# Workloads that cannot run on any instance this project can currently get,
# with the reason. Kept as data so a name here is a decision rather than an
# omission -- and so the day the quota lands, the test tells us to add them.
UNREACHABLE = {
    "all_reduce": "needs 2+ devices; trn1.32xlarge is 128 vCPU against a granted 64",
    "p2p_thrasher": "needs 2+ devices; same quota",
    "baseline_metrics": "idle telemetry, no load -- runs as part of any --test all",
}


def test_every_reachable_workload_is_validated():
    listed = orchestrated_names()
    expected = {w.name for w in registry.WORKLOADS} - set(UNREACHABLE)
    missing = expected - listed
    assert not missing, (
        f"not in validate_hardware.sh, so never run on hardware: {sorted(missing)}"
    )


def test_the_list_names_only_real_workloads():
    """A typo here silently drops a workload from the pass."""
    known = {w.name for w in registry.WORKLOADS}
    unknown = orchestrated_names() - known
    assert not unknown, f"not workloads in the registry: {sorted(unknown)}"


def test_unreachable_workloads_are_excluded_for_a_stated_reason():
    listed = orchestrated_names()
    for name, reason in UNREACHABLE.items():
        assert name not in listed, f"{name} cannot run: {reason}"


@pytest.mark.parametrize("name", sorted(UNREACHABLE))
def test_unreachable_names_are_still_real_workloads(name):
    """The exclusion list must not outlive the workloads it excuses."""
    assert name in {w.name for w in registry.WORKLOADS}


def test_the_collectives_are_excluded_only_while_the_quota_blocks_them():
    """Both need min_devices > 1. If that ever changes, so must this list."""
    by_name = {w.name for w in registry.WORKLOADS}
    for name in ("all_reduce", "p2p_thrasher"):
        workload = next(w for w in registry.WORKLOADS if w.name == name)
        assert workload.min_devices > 1, (
            f"{name} no longer needs multiple devices -- it should be validated"
        )
    assert "all_reduce" in by_name


def test_failures_do_not_end_the_run():
    """A workload that fails is a result; it must not cost the ones after it.

    The loop checks PIPESTATUS and reports, rather than running under `set
    -e` semantics that would abandon the remaining workloads.
    """
    text = _script()
    assert "PIPESTATUS" in text
    assert "TIMED OUT" in text
    # set -e would end the pass on the first non-zero exit.
    assert "set -uo pipefail" in text and "set -euo" not in text


def test_the_pass_exercises_the_neff_search_against_a_warm_cache():
    """The one thing the 2026-09-08 run could not test.

    It scored from candidate 1 of 1, because a fresh instance holds exactly
    one NEFF. Running a profiler-sourced workload again after everything
    else has compiled is what puts more than one graph in the ranking.
    """
    text = _script()
    assert "NEFF search against a warm compile cache" in text
    search = text[text.index("NEFF search against a warm compile cache"):]
    assert "--test memory_read" in search, (
        "the warm-cache pass must run a profiler-sourced workload"
    )


def test_the_pass_can_repeat_every_workload():
    """One run is a sample, not a measurement.

    memory_read's declared Score read 256.17, 178.7 and 119.19 GB/s on
    three runs of the same problem before its cause was found, and a
    single-pass harness could never have shown that. The default stays 1 --
    repeats multiply an already hour-long pass -- but a pass whose numbers
    will be quoted should set REPEAT.
    """
    text = _script()
    assert "REPEAT=${REPEAT:-1}" in text
    assert '--repeat "$REPEAT"' in text
    # Every orchestrated invocation carries it, not just the first.
    assert text.count('--repeat "$REPEAT"') >= 2


def test_the_summary_prints_the_spread():
    """A repeated run whose spread is invisible is a repeated run wasted."""
    text = _script()
    assert "repeats:" in text
    assert "cv" in text


# -- a finding nobody can find is not a finding ------------------------------

def test_every_doc_is_reachable_from_the_readme():
    """Six documents existed and the README linked to none of them.

    Each records a measurement that cost a hardware run to establish, and
    an unlinked file is one nobody reads before repeating the work. The
    check is against the whole directory rather than a list, so a new
    document cannot be orphaned by forgetting to add it here.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "README.md"), encoding="utf-8") as handle:
        readme = handle.read()
    docs = sorted(name for name in os.listdir(os.path.join(root, "docs"))
                  if name.endswith(".md"))
    assert docs, "the check would pass vacuously with no documents"
    unlinked = [name for name in docs if f"docs/{name}" not in readme]
    assert not unlinked, unlinked


def test_the_records_are_named_where_a_reader_will_look():
    """data/hardware_runs.json is the answer to "has this actually run"."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "README.md"), encoding="utf-8") as handle:
        readme = handle.read()
    assert "data/hardware_runs.json" in readme
    assert "data/baselines.json" in readme


def test_no_test_module_defines_the_same_name_twice():
    """Twice in one session, appending to a test file shadowed a helper
    already in it, and the second definition silently won.

    Both times the failure was loud -- eight unrelated tests raised
    TypeError -- but only because the signatures differed. Two helpers
    with the *same* signature and different behaviour would have made
    earlier tests quietly assert against the later one's semantics, which
    is the version of this that does not announce itself.
    """
    import ast

    root = os.path.dirname(os.path.abspath(__file__))
    offenders = []
    for name in sorted(os.listdir(root)):
        if not name.startswith("test_") or not name.endswith(".py"):
            continue
        with open(os.path.join(root, name), encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=name)
        seen = set()
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            # A parametrised or conditionally defined name is not a
            # redefinition worth flagging; a plain duplicate is.
            if node.name in seen:
                offenders.append(f"{name}:{node.lineno} redefines {node.name}")
            seen.add(node.name)
    assert not offenders, offenders


# Production functions that legitimately have no production call site,
# each with the reason it is an exception rather than an oversight.
#
# The list is deliberately short and deliberately explicit. Restricting
# the sweep to names starting with "verify_" would shrink it to nothing
# and was the first version of this check -- which found two dead
# functions and missed four more, including two added the same afternoon
# by the author of the check.
CALLED_ONLY_BY_TESTS = {
    # Registry guards. tests/test_score_schema.py walks every pinned
    # problem and asks whether the engine will run its dtype. That is a
    # policy assertion about the registry, not a runtime decision, so
    # there is nothing at runtime for it to gate.
    "engine_accepts",
    "refusal",
    # A property of the pinned inputs, asserted rather than computed: the
    # inputs are built so every expert receives exactly `capacity` tokens.
    # Nothing at runtime needs the answer, because the inputs cannot make
    # it come out otherwise -- and a test proving that is the point.
    "routing_balance",
    # Aspirational, and honestly so. Its four counters --
    # tensor/vector/scalar/gpsimd_engine_active_time_percent -- are real
    # and are in neuron-profile's 108-counter set. They are NOT in the
    # neuron-monitor stream, and omni_virus declares neuron-monitor as its
    # Score source, so nothing on that workload's path ever holds them.
    #
    # Wiring it to the monitor metrics was tried on 2026-09-10 and
    # reverted: it type-checked, ran, and returned an empty dict every
    # time. A call site that reports nothing is worse than no call site,
    # because it looks answered.
    #
    # The workload whose entire premise is per-engine behaviour is scored
    # by the reader that cannot see any engine. Reaching those counters
    # needs a profiler capture of omni_virus's own NEFF, which needs a
    # reserved core, which is off for any selection containing a
    # cores: "all" workload. That is a real piece of work, not an
    # oversight, and it is not done.
    "engine_activity",
}


def test_every_production_function_is_called_from_production_code():
    """Two were not, and both had passing tests.

    tensor_virus.verify_against_monitor and
    profiler.verify_profile_covers_plan were each written, documented and
    covered -- five tests apiece -- and neither was called from anywhere.
    Their tests said the functions were correct, and they were. Nothing
    said whether they ran.

    Being dead is not a neutral state either. verify_profile_covers_plan
    kept a one-sided bound the whole time it was unreachable, because
    nothing exercised it against a graph larger than the plan, and the
    inline copy that replaced it inherited the same gap.

    Read through the comment filter, so a mention in a docstring does not
    count as a call site -- three docstrings credited
    verify_profile_covers_plan with a refusal it was no longer making.
    """
    import ast
    import sourcecheck

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources = {}
    for directory in (root, os.path.join(root, "kernels")):
        for name in sorted(os.listdir(directory)):
            if name.endswith(".py"):
                path = os.path.join(directory, name)
                with open(path, encoding="utf-8") as handle:
                    sources[path] = handle.read()

    assert sources, "no production modules found -- the sweep is broken"

    code = "".join(sourcecheck.code_only(text) for text in sources.values())

    unreachable = []
    for path, text in sources.items():
        for node in ast.parse(text).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("__"):
                continue
            if node.name in CALLED_ONLY_BY_TESTS:
                continue
            # One occurrence is the definition itself.
            if code.count(node.name) <= 1:
                unreachable.append(f"{os.path.basename(path)}:{node.name}")

    assert not unreachable, (
        f"defined and never called: {unreachable} -- either wire it in, "
        "delete it, or name it in CALLED_ONLY_BY_TESTS with a reason")


def test_the_readme_status_table_lists_every_workload():
    """It is hand-maintained, and a workload missing from it is invisible.

    The figures in that table cannot be derived -- they are dated single
    runs -- but its *membership* can, and a new workload silently absent
    from the status table is a workload nobody knows has no status.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "README.md"), encoding="utf-8") as handle:
        readme = handle.read()

    start = readme.index("| Workload | Status | Measured | Score source |")
    end = readme.index("\n\n", start)
    table = readme[start:end]

    listed = set(re.findall(r"^\| `([^`]+)` \|", table, re.MULTILINE))
    assert listed, "no rows parsed -- the table shape changed"

    known = {workload.name for workload in registry.WORKLOADS}
    assert not known - listed, f"missing from the status table: {known - listed}"
    assert not listed - known, f"in the table but not the registry: {listed - known}"


def test_the_accident_catalogue_numbering_matches_its_own_count():
    """A document that miscounts its own list is a small joke at its own
    expense, and this file is where it would land.

    The intro and the closing line both quote a total. Deriving it from
    the headings means adding an entry cannot leave either stale -- which
    has already happened once, in the commit that added the eleventh.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "docs", "checks_that_pass_by_accident.md")
    with open(path, encoding="utf-8") as handle:
        doc = handle.read()

    numbered = re.findall(r"^### (\d+)\. ", doc, re.MULTILINE)
    assert numbered, "no numbered entries found -- the heading shape changed"

    # Numbered from 1 with no gaps, so a renumber cannot silently collide.
    assert [int(n) for n in numbered] == list(range(1, len(numbered) + 1)), \
        numbered

    words = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
             14: "fourteen", 15: "fifteen", 16: "sixteen"}
    total = words.get(len(numbered))
    assert total, f"add {len(numbered)} to the words map"
    assert f"produced {total} of them" in doc, (
        f"the intro does not say 'produced {total} of them'")
    assert f"{total} above were found" in doc, (
        f"the closing line does not say '{total} above were found'")


def test_every_implemented_workload_has_a_dispatch_branch():
    """A name in IMPLEMENTED with no branch falls through to
    NotImplementedError -- which is the honest outcome, but the name
    being in IMPLEMENTED is then a claim the code does not keep.

    The pattern deliberately allows digits. The first version of this
    check used `[a-z_]+` and reported p2p_thrasher as unhandled, because
    the name has a 2 in it. A check whose pattern cannot express its
    subject is the shape in docs/checks_that_pass_by_accident.md that
    matches a phrasing rather than a claim -- here it produced a false
    positive rather than a false negative, which is the lucky direction.
    """
    import re
    import sourcecheck

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pantheon_neuron.py"),
              encoding="utf-8") as handle:
        code = sourcecheck.code_only(handle.read())

    named = set(re.findall(r'workload \. name == "([a-z0-9_]+)"', code))
    for group in re.findall(r'workload \. name in \(([^)]*)\)', code):
        named.update(re.findall(r'"([a-z0-9_]+)"', group))

    assert named, "no dispatch branches parsed -- the shape changed"
    missing = set(pantheon_neuron.IMPLEMENTED) - named
    assert not missing, f"in IMPLEMENTED with no dispatch branch: {missing}"


def test_every_dispatch_branch_names_a_real_workload():
    """A branch for a workload the registry does not have is dead code
    that reads as coverage."""
    import re
    import sourcecheck

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pantheon_neuron.py"),
              encoding="utf-8") as handle:
        code = sourcecheck.code_only(handle.read())

    named = set(re.findall(r'workload \. name == "([a-z0-9_]+)"', code))
    for group in re.findall(r'workload \. name in \(([^)]*)\)', code):
        named.update(re.findall(r'"([a-z0-9_]+)"', group))

    known = {workload.name for workload in registry.WORKLOADS}
    assert not named - known, f"dispatched but not in the registry: {named - known}"


def test_the_readme_has_no_dangling_internal_links():
    """A cross-reference to a heading that moved reads as a broken page.

    The findings sections cross-link each other, and headings here get
    rewritten as measurements replace guesses.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "README.md"), encoding="utf-8") as handle:
        doc = handle.read()

    headings = re.findall(r"^#+ (.+)$", doc, re.MULTILINE)
    assert headings, "no headings parsed -- the shape changed"
    slugs = {re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-")
             for h in headings}

    anchors = re.findall(r"\]\(#([a-z0-9-]+)\)", doc)
    dangling = [a for a in anchors if a not in slugs]
    assert not dangling, dangling


def test_every_readme_file_link_points_at_something_that_exists():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "README.md"), encoding="utf-8") as handle:
        doc = handle.read()

    targets = re.findall(r"\]\(((?:docs|data|tools|kernels)/[^)#]+)\)", doc)
    assert targets, "no file links parsed -- the shape changed"
    missing = [t for t in targets
               if not os.path.exists(os.path.join(root, t))]
    assert not missing, missing


def test_the_readme_headings_are_unique():
    """Two identical headings give one of them an unreachable anchor."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "README.md"), encoding="utf-8") as handle:
        headings = re.findall(r"^#+ (.+)$", handle.read(), re.MULTILINE)
    duplicates = sorted({h for h in headings if headings.count(h) > 1})
    assert not duplicates, duplicates


def test_the_validation_summary_only_counts_this_run():
    """It counted every report on the machine, from any run, ever.

    The 2026-09-10 pass reported "Workloads run: 26, PASS 26" for a run
    that ran 24. all_reduce, p2p_thrasher and baseline_metrics were rows
    from earlier runs whose reports were still on disk -- and the first
    two cannot run on a single-device part at all, so the summary
    credited the run with passing two workloads the hardware refuses.

    A summary that mixes runs is worse than none: every figure in it
    reads as a statement about the run that just finished, and that one
    was copied into the README before anyone noticed.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "tools", "validate_hardware.sh"),
              encoding="utf-8") as handle:
        script = handle.read()

    assert "RUN_STARTED=$(date +%s)" in script
    assert "export RUN_STARTED" in script
    # The filter itself, and the count of what it discarded -- silently
    # ignoring reports would trade one wrong number for a missing one.
    assert "os.path.getmtime(p) >= started" in script
    assert "from earlier runs on this machine ignored" in script


def test_no_test_parametrises_only_over_gitignored_paths():
    """An empty parameter set is a skip, and a skip is not a check.

    test_committed_reports_are_clean parametrised over `database/*.json`.
    database/ is in .gitignore, so a fresh checkout has none, pytest
    reported "got empty parameter set" and skipped -- in the job
    deliberately split out of the matrix so a single green check could
    not hide the privacy guard.

    This looks for the shape rather than that one case: a glob rooted at
    a directory the repository does not track cannot find anything in CI.
    """
    import ast

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, ".gitignore"), encoding="utf-8") as handle:
        ignored = {line.strip().rstrip("/") for line in handle
                   if line.strip() and not line.startswith("#")}
    assert ignored, "no .gitignore entries parsed -- the shape changed"

    offenders = []
    for name in sorted(os.listdir(os.path.dirname(os.path.abspath(__file__)))):
        if not name.startswith("test_") or not name.endswith(".py"):
            continue
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source, filename=name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = getattr(node.func, "attr", None)
            if target != "glob":
                continue
            literals = [n.value for n in ast.walk(node)
                        if isinstance(n, ast.Constant)
                        and isinstance(n.value, str)]
            for literal in literals:
                if literal.rstrip("/") in ignored:
                    # Only a finding if it is the *sole* source of cases.
                    offenders.append(f"{name}:{node.lineno} globs {literal!r}")

    # database/ is still globbed on purpose, by a test that says so and is
    # backed by one which generates its own report. Anything else is new.
    unexpected = [o for o in offenders
                  if "test_report_privacy.py" not in o]
    assert not unexpected, unexpected


def test_the_privacy_guard_does_not_depend_on_finding_a_file():
    """The version that runs in CI has to make its own evidence."""
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "test_report_privacy.py"),
              encoding="utf-8") as handle:
        source = handle.read()
    assert "def test_a_freshly_written_report_is_clean" in source
    assert "write_report(" in source
    assert "the run wrote no report, so nothing was checked" in source
