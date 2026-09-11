"""Splitting NeuronCores between the workload and the profiler.

``neuron-profile capture`` does not read counters from a running process --
it replays the NEFF, which means it needs NeuronCores of its own. The Neuron
runtime hands a process every visible core by default, so a workload that
declares ``cores: 1`` still holds all of them, and the profiler that runs
afterwards finds none:

    Logical Neuron Core(s) not available - Requested:2 Available:0

Observed on inf2.xlarge 2026-09-07, where it made the declared Score source
unreachable for every scored run in the suite's history: `memory_read` and
`memory_write` both name `neuron-profile`, and both have only ever reported
the analytic fallback. The same capture succeeds against the same NEFF once
no workload is holding the device.

So the run reserves a core. The workload gets every core but the last, the
profiler gets the last one, and both fit on the two-core parts this suite
targets. That also makes the process honour the ``cores: 1`` its own pinned
problems declare, which it never did before.

What this does not change: the profiler's numbers come from a replay of the
NEFF, not from the timed loop. Same graph and same work, but a separate
execution -- which is inherent to how the declared source is defined, not a
consequence of reserving a core.
"""

import contextlib
import os
import typing


# The runtime's own variable, read at process start. Setting it after the
# runtime has initialised has no effect, which is why the orchestrator sets
# it before dispatching rather than the kernel setting it inside run().
VISIBLE_CORES = "NEURON_RT_VISIBLE_CORES"

# How the orchestrator tells the profiler which core it kept free. An env
# var rather than an argument so profiler.py needs no device discovery.
RESERVED_CORE = "PANTHEON_NEURON_PROFILE_CORE"

# Where the Neuron compiler keeps its cache. Pointing it at the run's own
# workdir is what makes a captured profile attributable: the shared cache
# at /var/tmp/neuron-compile-cache accumulates every graph ever compiled on
# the machine, and on a warm one the kernel's own NEFF ranks last by mtime
# because a cache hit leaves its timestamp alone. Measured on trn1.2xlarge
# 2026-09-08: the right graph was candidate 13 of 14, and in a fuller run
# it fell outside the search budget entirely and a wrong graph was
# published at 23.58 GB/s against an analytic 271.7.
#
# With this set, the run's directory holds only the run's graphs -- 3
# against the shared cache's 14 in the same measurement -- and
# profiler.find_neffs searches it exclusively.
COMPILE_CACHE = "NEURON_COMPILE_CACHE_URL"

# **The run's directory stopped holding only the run's graphs.** The
# kernels set the variable with os.environ.setdefault and never unset it,
# at a fixed path. Every workload after memory_read in the same process
# then compiled into the directory the profiler searches, and the path
# outlived the process: on trn1.2xlarge 2026-09-11 it held 63 NEFFs after
# one full pass, and the next `--test memory_read --repeat 3` searched 16
# candidates, newest first, ran out of budget before its own graph, and
# fell back to the analytic Score. The compiler reads the variable at
# every compile, not once per process (measured the same day: graphs
# compiled under /tmp/a, /tmp/b, /tmp/a landed in a, b and a), so the
# fix is to set it only while the kernel runs, to a directory of its own.
WORKDIR = "PANTHEON_NEURON_WORKDIR"
DEFAULT_WORKDIR = "/tmp/pantheon_ccwork"


def kernel_workdir(name: str) -> str:
    """The compile and profile directory for one kernel, created if needed.

    One per kernel, under PANTHEON_NEURON_WORKDIR, so it holds only that
    kernel's graphs across every run on the machine -- and a later run
    still gets compile-cache hits from it.
    """
    workdir = os.path.join(os.environ.get(WORKDIR, DEFAULT_WORKDIR), name)
    os.makedirs(workdir, exist_ok=True)
    return workdir


@contextlib.contextmanager
def compile_cache(directory: str):
    """Compile into ``directory`` for the duration of the block, then stop.

    A cache someone set before the block is theirs and is left alone, as
    setdefault left it: they pinned it for a reason we should not overrule.
    Anything set here is removed on the way out, so the next workload in
    the process compiles wherever it would have.
    """
    if os.environ.get(COMPILE_CACHE):
        yield os.environ[COMPILE_CACHE]
        return
    os.environ[COMPILE_CACHE] = directory
    try:
        yield directory
    finally:
        os.environ.pop(COMPILE_CACHE, None)


def split(total_cores: int) -> typing.Optional[typing.Dict[str, str]]:
    """Divide ``total_cores`` between the workload and the profiler.

    Returns the two ``NEURON_RT_VISIBLE_CORES`` values, or None when there
    is nothing to divide. One core cannot be split: the workload takes it
    and the profiler goes without, which is the honest outcome -- the Score
    then degrades to the analytic figure and the row records that, rather
    than the run pretending the declared source was consulted.
    """
    if total_cores < 2:
        return None
    last = total_cores - 1
    workload = "0" if last == 1 else f"0-{last - 1}"
    return {"workload": workload, "profiler": str(last)}
