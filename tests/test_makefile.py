"""The Makefile advertises targets; this checks they exist.

``make lint`` was in ``.PHONY`` and in the help text and had no recipe.
``make`` answers that with "Nothing to be done for 'lint'" and **exits
0**, so a CI step calling it, or a contributor running it before opening
a PR, would have seen success while nothing was checked.

That is the same defect this repo catalogues in
docs/checks_that_pass_by_accident.md, in its purest form: not a check
that was weak, or scoped wrongly, or matched the wrong string, but a
check that did not exist and reported success anyway.

It was found by running ``make lint`` to see what it caught.
"""

import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAKEFILE = os.path.join(ROOT, "Makefile")


def _makefile():
    with open(MAKEFILE, encoding="utf-8") as handle:
        return handle.read()


def _targets_with_recipes():
    """Targets that actually do something when invoked.

    A target does something if it has a tab-indented recipe **or** at
    least one prerequisite: `ci: lint test mock` runs three targets
    without a recipe of its own.

    The first version of this counted only recipes and reported `ci` as
    missing -- a checker for empty targets that was wrong about what
    empty means. Which is the point of the file: `make` answers a target
    with neither recipe nor prerequisite by printing "Nothing to be done"
    and exiting 0, and that is the condition worth detecting, not the
    absence of a tab.
    """
    targets = set()
    current = None
    for line in _makefile().split("\n"):
        if line.startswith("\t"):
            if current:
                targets.add(current)
            continue
        match = re.match(r"^([a-zA-Z0-9_.-]+):(?!=)(.*)$", line)
        if not match:
            current = None
            continue
        current = match.group(1)
        # .PHONY is a directive whose "prerequisites" are the names it
        # declares, not a target anyone invokes.
        if current.startswith("."):
            current = None
            continue
        if match.group(2).split():          # has prerequisites
            targets.add(current)
    return targets


def _phony_targets():
    match = re.search(r"^\.PHONY:\s*(.+)$", _makefile(), re.MULTILINE)
    assert match, ".PHONY line not found -- the Makefile shape changed"
    return set(match.group(1).split())


def _advertised_targets():
    """Targets named in the help text, which is what a reader will try."""
    return set(re.findall(r'@echo "make (\S+)', _makefile()))


def test_the_makefile_declares_some_targets():
    """So the three checks below cannot pass over an empty set."""
    assert _phony_targets()
    assert _advertised_targets()
    assert _targets_with_recipes()


def test_every_phony_target_has_a_recipe():
    """A .PHONY name with no recipe is a target that silently does
    nothing and exits 0."""
    missing = _phony_targets() - _targets_with_recipes() - {"help"}
    assert not missing, f"declared in .PHONY with no recipe: {sorted(missing)}"


def test_every_advertised_target_exists():
    """`make help` is the discovery path. A target it names that does not
    run is worse than one it omits."""
    missing = _advertised_targets() - _targets_with_recipes()
    assert not missing, f"advertised by `make help` but not defined: {sorted(missing)}"


def test_every_real_target_is_advertised():
    """The other direction: a target nobody is told about is a target
    nobody runs, which is how `lint` stayed empty for so long."""
    unadvertised = (_targets_with_recipes() - _advertised_targets()
                    - {"help"})
    assert not unadvertised, (
        f"defined but absent from `make help`: {sorted(unadvertised)}")


@pytest.mark.skipif(sys.platform == "win32", reason="no make on Windows CI")
def test_make_lint_actually_runs_something():
    """The specific regression, asserted against make's own output.

    "Nothing to be done for X" is what an empty target prints, and it is
    the string that would have appeared for two weeks. Checking for its
    absence is narrow on purpose: a recipe that exists but is a no-op
    would still pass the tests above.
    """
    result = subprocess.run(
        ["make", "-n", "lint"], cwd=ROOT, capture_output=True, text=True,
        check=False)
    assert result.returncode == 0, result.stderr
    assert "Nothing to be done" not in result.stdout
    assert "ruff" in result.stdout, (
        "make lint no longer invokes a linter: " + result.stdout)


def test_clean_removes_the_bytecode_that_hides_compile_warnings():
    """`filterwarnings = error` catches a SyntaxWarning only on a cold
    compile.

    A compile-time warning is emitted when a module is compiled, not when
    a cached .pyc is loaded, so a warm __pycache__ hides it. CI checks out
    fresh and is always cold; locally, `make clean test` is what
    reproduces CI, and that only holds if clean actually removes the
    caches.
    """
    recipe = _makefile()
    marker = recipe.index("\nclean:")
    body = recipe[marker:marker + 400]
    for cache in ("__pycache__", "kernels/__pycache__", "tests/__pycache__"):
        assert cache in body, f"make clean leaves {cache} behind"


def test_the_coverage_floor_sits_below_the_measured_figure():
    """A floor set above what the suite achieves fails on introduction,
    and a check that fails on day one gets waived rather than fixed.

    Measured 78.64% with branch coverage on 2026-09-10; the floor is 78.
    """
    recipe = _makefile()
    match = re.search(r"COVERAGE_FLOOR:-(\d+)", recipe)
    assert match, "the coverage floor is no longer set in the Makefile"
    floor = int(match.group(1))
    assert 50 <= floor <= 78, (
        f"floor is {floor}: above the measured 78.64% it fails on "
        "introduction, and below 50 it is not a floor")


def test_the_coverage_config_explains_why_kernels_are_low():
    """Otherwise the obvious way to raise the number is to write tests
    asserting a kernel body raises BackendUnavailable -- coverage of the
    import guard, not of the kernel.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, ".coveragerc"), encoding="utf-8") as handle:
        config = " ".join(handle.read().split())
    assert "cannot execute without a Neuron device" in config
    assert "do not raise this number" in config.lower()
    assert "data/hardware_runs.json" in config


def test_ci_installs_tooling_from_the_pinned_file():
    """CI ran `pip install ruff`, unpinned, and got 0.16.7; a local 0.5.6
    passed the same code. The lint job's verdict depended on the day it
    ran, and the PR sat red for several commits while I reported it green
    from a local run.

    So every CI job that installs tooling does it from
    requirements-dev.txt, and nothing installs ruff by bare name.
    """
    # Read as text, not parsed as YAML. The first version imported yaml,
    # which is not a declared dependency -- it was on the machine that
    # wrote the test and not on the runner that ran it, so this test, the
    # one written to stop works-locally-fails-in-CI drift, failed all four
    # matrix jobs through exactly that drift. The property is textual
    # anyway: does any line install a tool by bare name.
    with open(os.path.join(ROOT, ".github", "workflows", "ci.yml"),
              encoding="utf-8") as handle:
        workflow = handle.read()
    installs = [line for line in workflow.split("\n")
                if "pip install" in line]
    assert installs, "no install steps found -- the workflow shape changed"
    for command in installs:
        assert not re.search(r"pip install[^\n]*\bruff\b", command), command
        assert not re.search(r"pip install[^\n]* pytest\b(?!-)", command), command


def test_the_linter_is_pinned_to_an_exact_version():
    """A range would reintroduce the drift one release at a time."""
    with open(os.path.join(ROOT, "requirements-dev.txt"),
              encoding="utf-8") as handle:
        pins = [line.strip() for line in handle
                if line.strip() and not line.startswith("#")]
    assert pins, "requirements-dev.txt pins nothing"
    ruff = [p for p in pins if p.startswith("ruff")]
    assert ruff and "==" in ruff[0], ruff


# Modules imported by this repository that are deliberately not declared:
# the Neuron toolchain installs from the AWS pip index rather than PyPI and
# only exists on a Neuron instance, and every import of it is behind the
# mock-mode guard. Local modules are the repo's own files.
_UNDECLARED_BY_DESIGN = {
    "torch", "torch_xla", "torch_neuronx", "neuronxcc",
}


def test_every_third_party_import_is_a_declared_dependency():
    """test_ci_installs_tooling_from_the_pinned_file imported yaml.

    PyYAML was on the machine that wrote the test and not on the runner
    that ran it, so it failed all four matrix jobs and the coverage job --
    a test written to stop works-locally-fails-in-CI drift, failing
    through exactly that drift. It passed locally, because locally is
    where the undeclared package was.

    This asserts the general case: every top-level module the repository
    imports is the standard library, the repository itself, the Neuron
    toolchain by design, or declared in a requirements file.
    """
    import ast

    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is None:
        pytest.skip("sys.stdlib_module_names needs Python 3.10+")

    declared = set()
    for name in ("requirements.txt", "requirements-dev.txt"):
        with open(os.path.join(ROOT, name), encoding="utf-8") as handle:
            for line in handle:
                line = line.split("#")[0].strip()
                if line:
                    declared.add(re.split(r"[=<>!~\\[]", line)[0]
                                 .strip().lower().replace("-", "_"))
    # pytest-cov is imported as pytest_cov and pulled in as a plugin; its
    # distribution name is declared under the hyphenated spelling.
    assert declared, "no requirements parsed -- the sweep is vacuous"

    local = {name[:-3] for name in os.listdir(ROOT) if name.endswith(".py")}
    local |= {"kernels", "tests", "tools", "sourcecheck"}
    for sub in ("tests", "tools", "kernels"):
        local |= {name[:-3]
                  for name in os.listdir(os.path.join(ROOT, sub))
                  if name.endswith(".py")}

    undeclared = set()
    for directory in (ROOT, os.path.join(ROOT, "kernels"),
                      os.path.join(ROOT, "tests"), os.path.join(ROOT, "tools")):
        for name in os.listdir(directory):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(directory, name), encoding="utf-8") as h:
                tree = ast.parse(h.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    top = module.split(".")[0]
                    if (not top or top in stdlib or top in local
                            or top in _UNDECLARED_BY_DESIGN
                            or top.lower() in declared):
                        continue
                    undeclared.add(f"{top} ({name})")

    assert not undeclared, (
        f"imported but not declared in a requirements file: "
        f"{sorted(undeclared)}")
