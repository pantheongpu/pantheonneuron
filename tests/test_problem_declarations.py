"""A declared ``problem`` is a claim about the run. Read the kernel, not the prose.

``Workload.problem`` says what the measured run executed, and every report row
publishes it verbatim: ``"Problem": dict(workload.problem)``. The registry's own
words are that "a Score is only comparable across platforms if both ran the same
problem". Nothing checked that the run and the claim agreed, and they did not:

* ``kv_cache_churn`` declared ``heads: 16``. ``cache_plan`` never read it -- the
  caches are shaped ``(layers, context, hidden)`` with no head dimension -- so
  the key could have said 64 and no byte of the run would move.
* ``serving_mix`` ran ``problem.get("layers", 32)`` and
  ``problem.get("hidden", 4096)`` while declaring neither. The stack depth and
  width that set the Score sat outside the published problem, so two runs at
  different depths would publish the same "pinned" problem.

Both directions are the same defect: the row names something other than what
ran. This module checks both by parsing the kernels.

**Why parse rather than transcribe.** A hand-kept table of which key each kernel
reads would be a second copy of the belief, and this repo has already had two
tests pass because both sides of a comparison were transcribed from the same
wrong assumption. The entry point for each workload is read out of the dispatch
in ``pantheon_neuron.py``, so a kernel that is rewired cannot leave this check
pointing at the old function.

Keys that are honoured without being read -- a literal equal to the declaration,
the shape of the run, a worker subprocess -- are declared in
``registry.PROBLEM_KEYS_NOT_READ_FROM_THE_DICT`` with the reason, and that
register is itself checked for entries that have gone stale.
"""

import ast
import importlib
import inspect
import os
import pkgutil

import pytest

import kernels
from kernels import registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_MODULES = {module.name for module in pkgutil.iter_modules(kernels.__path__)}

# How deep to follow ``f(problem)`` from the entry point. Three hops covers
# run -> plan -> helper, which is the deepest chain in the suite today; a
# kernel that buries a read deeper will show up as an unread key rather than
# passing quietly.
_MAX_DEPTH = 3


def _dispatch_tree():
    with open(os.path.join(ROOT, "pantheon_neuron.py"), encoding="utf-8") as handle:
        return ast.parse(handle.read())


def _entry_points():
    """Workload name -> [(kernel module, function)], read from the dispatch.

    Two shapes appear there and both are handled: a direct
    ``module.run(workload.problem, ...)``, and the indirection
    ``_execute_bandwidth(workload, duration, memory_read)``, which binds the
    kernel module at the call site and calls ``module.run`` inside.
    """
    found = {}
    for node in ast.walk(_dispatch_tree()):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Attribute)
                and test.left.attr == "name"):
            continue
        compared = test.comparators[0]
        if isinstance(test.ops[0], ast.Eq) and isinstance(compared, ast.Constant):
            names = [compared.value]
        elif (isinstance(test.ops[0], ast.In)
                and isinstance(compared, (ast.Tuple, ast.List))):
            names = [element.value for element in compared.elts
                     if isinstance(element, ast.Constant)]
        else:
            continue

        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            takes_problem = any(
                isinstance(argument, ast.Attribute) and argument.attr == "problem"
                for argument in call.args)
            if (takes_problem and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id in KERNEL_MODULES):
                for name in names:
                    found.setdefault(name, []).append(
                        (call.func.value.id, call.func.attr))
            elif isinstance(call.func, ast.Name):
                for argument in call.args:
                    if (isinstance(argument, ast.Name)
                            and argument.id in KERNEL_MODULES):
                        for name in names:
                            found.setdefault(name, []).append((argument.id, "run"))
    return found


def _keys_read(module, function, seen=None, depth=0):
    """(keys read, keys read with a default) for one kernel function.

    Follows calls that pass ``problem`` along, so a key read in ``cache_plan``
    counts for the ``run_cache_churn`` that calls it.
    """
    if seen is None:
        seen = set()
    if (module, function) in seen or depth > _MAX_DEPTH:
        return set(), set()
    seen.add((module, function))
    try:
        imported = importlib.import_module(f"kernels.{module}")
        source = inspect.getsource(getattr(imported, function))
    except (ImportError, AttributeError, OSError, TypeError):
        return set(), set()

    read, defaulted = set(), set()
    for node in ast.walk(ast.parse(source.lstrip())):
        # problem["key"]
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "problem"
                and isinstance(node.slice, ast.Constant)):
            read.add(node.slice.value)
        if not isinstance(node, ast.Call):
            continue
        # problem.get("key") and problem.get("key", default)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "problem" and node.args
                and isinstance(node.args[0], ast.Constant)):
            read.add(node.args[0].value)
            if len(node.args) > 1:
                defaulted.add(node.args[0].value)
        # f(problem) / other_module.f(problem)
        if any(isinstance(a, ast.Name) and a.id == "problem" for a in node.args):
            if isinstance(node.func, ast.Name):
                deeper = _keys_read(module, node.func.id, seen, depth + 1)
            elif (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in KERNEL_MODULES):
                deeper = _keys_read(node.func.value.id, node.func.attr,
                                    seen, depth + 1)
            else:
                continue
            read |= deeper[0]
            defaulted |= deeper[1]
    return read, defaulted


def _scored(workload):
    return workload.problem is not None


SCORED = [w for w in registry.WORKLOADS if _scored(w)]
ENTRY_POINTS = _entry_points()


def _keys_for(workload):
    read, defaulted = set(), set()
    for module, function in ENTRY_POINTS.get(workload.name, []):
        found, defaults = _keys_read(module, function)
        read |= found
        defaulted |= defaults
    return read, defaulted


@pytest.mark.parametrize("workload", SCORED, ids=lambda w: w.name)
def test_every_workload_with_a_problem_has_an_entry_point(workload):
    """A workload the dispatch does not reach would skip every check below."""
    assert ENTRY_POINTS.get(workload.name), (
        f"{workload.name}: no kernel entry point found in the dispatch, so its "
        "declared problem is checked against nothing"
    )


@pytest.mark.parametrize("workload", SCORED, ids=lambda w: w.name)
def test_every_declared_key_is_read_by_the_kernel(workload):
    """The row publishes this dict as what ran, so the run has to read it."""
    read, _ = _keys_for(workload)
    excused = set(registry.PROBLEM_KEYS_NOT_READ_FROM_THE_DICT.get(
        workload.name, {}))
    unread = set(workload.problem) - read - excused
    assert not unread, (
        f"{workload.name} declares {sorted(unread)} and the kernel never reads "
        f"{'it' if len(unread) == 1 else 'them'}. Either the run should use "
        f"the declared value, or the key does not describe this run and should "
        f"go; if the kernel honours it another way, declare it in "
        f"registry.PROBLEM_KEYS_NOT_READ_FROM_THE_DICT with the reason."
    )


@pytest.mark.parametrize("workload", SCORED, ids=lambda w: w.name)
def test_no_undeclared_default_shapes_the_run(workload):
    """``problem.get("layers", 32)`` on an undeclared key is a hidden pin.

    The value sets the work and the published problem does not mention it, so
    two runs at different depths publish the same claim. There is no excuse
    register for this one: declaring the key costs nothing and the default
    stays as the value.
    """
    _, defaulted = _keys_for(workload)
    hidden = defaulted - set(workload.problem)
    assert not hidden, (
        f"{workload.name}: the kernel reads {sorted(hidden)} with a default, "
        f"but {'it is' if len(hidden) == 1 else 'they are'} not in the declared "
        f"problem. Add {'it' if len(hidden) == 1 else 'them'} to the registry "
        f"with the default as the value."
    )


def test_the_constant_register_names_real_workloads_and_keys():
    """An entry for a key nobody declares is a note about nothing."""
    by_name = {w.name: w for w in registry.WORKLOADS}
    for name, excuses in registry.PROBLEM_KEYS_NOT_READ_FROM_THE_DICT.items():
        workload = by_name.get(name)
        assert workload is not None, f"{name} is not a workload"
        assert workload.problem, f"{name} declares no problem"
        for key, reason in excuses.items():
            assert key in workload.problem, (
                f"{name}: excused key {key!r} is not declared")
            assert len(reason) > 20, (
                f"{name}.{key}: the reason must say how it is honoured")


@pytest.mark.parametrize("workload", SCORED, ids=lambda w: w.name)
def test_the_constant_register_has_no_stale_entries(workload):
    """A key the kernel now reads must not stay excused.

    Otherwise the register becomes a place where a key can hide after the
    reason for excusing it has gone away.
    """
    excused = set(registry.PROBLEM_KEYS_NOT_READ_FROM_THE_DICT.get(
        workload.name, {}))
    if not excused:
        pytest.skip("nothing excused for this workload")
    read, _ = _keys_for(workload)
    stale = excused & read
    assert not stale, (
        f"{workload.name}: {sorted(stale)} is excused in "
        f"PROBLEM_KEYS_NOT_READ_FROM_THE_DICT but the kernel does read it now; "
        f"drop the entry"
    )


def test_the_dispatch_parse_finds_the_kernels_it_should():
    """If the parse silently found nothing, every check above would pass.

    The failure this guards against is structural: a rename in the dispatch
    that makes ``_entry_points`` return an empty dict would turn the whole
    module green while checking nothing.
    """
    assert len(ENTRY_POINTS) >= len(SCORED), (
        f"the dispatch parse resolved {len(ENTRY_POINTS)} workloads for "
        f"{len(SCORED)} scored ones"
    )
    # A spot check with a known shape: kv_cache_churn is dispatched directly,
    # memory_read through the _execute_bandwidth indirection.
    assert ("llm_inference", "run_cache_churn") in ENTRY_POINTS["kv_cache_churn"]
    assert ("memory_read", "run") in ENTRY_POINTS["memory_read"]
