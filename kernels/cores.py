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

import typing


# The runtime's own variable, read at process start. Setting it after the
# runtime has initialised has no effect, which is why the orchestrator sets
# it before dispatching rather than the kernel setting it inside run().
VISIBLE_CORES = "NEURON_RT_VISIBLE_CORES"

# How the orchestrator tells the profiler which core it kept free. An env
# var rather than an argument so profiler.py needs no device discovery.
RESERVED_CORE = "PANTHEON_NEURON_PROFILE_CORE"


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
