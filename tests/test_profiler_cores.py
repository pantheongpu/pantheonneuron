"""Reserving a NeuronCore so the profiler can reach its declared Score.

`neuron-profile capture` replays the NEFF and needs a core. The runtime
hands a process every visible core, so the workload took all of them and
the profiler found none -- which is why `memory_read` and `memory_write`
have only ever reported the analytic fallback despite both declaring
`neuron-profile`. These tests cover the split and the environment it
produces; whether the capture then succeeds is a hardware question.
"""

import os

import pantheon_neuron
from kernels import cores, profiler
from neuron_device import NeuronDevice


def _devices(count, cores_each=2):
    return [
        NeuronDevice(i, "inf2", "v2", cores_each, 32 * 1024**3, False)
        for i in range(count)
    ]


# -- the split ---------------------------------------------------------------

def test_two_cores_split_one_each():
    """Both parts this suite targets have exactly two cores per device."""
    assert cores.split(2) == {"workload": "0", "profiler": "1"}


def test_more_cores_leave_only_the_last_reserved():
    """The profiler needs one core; the workload should keep the rest."""
    assert cores.split(4) == {"workload": "0-2", "profiler": "3"}
    assert cores.split(8) == {"workload": "0-6", "profiler": "7"}


def test_a_single_core_cannot_be_split():
    """The workload takes it and the Score honestly degrades to analytic."""
    assert cores.split(1) is None
    assert cores.split(0) is None


# -- the environment the profiler runs under ---------------------------------

def test_profiler_runs_on_the_reserved_core(monkeypatch):
    monkeypatch.setenv(cores.RESERVED_CORE, "1")
    assert profiler._environment()[cores.VISIBLE_CORES] == "1"


def test_profiler_leaves_cores_alone_when_none_was_reserved(monkeypatch):
    """A single-core part, or a caller who pinned cores by hand."""
    monkeypatch.delenv(cores.RESERVED_CORE, raising=False)
    monkeypatch.delenv(cores.VISIBLE_CORES, raising=False)
    assert cores.VISIBLE_CORES not in profiler._environment()


def test_reservation_overrides_an_inherited_workload_pin(monkeypatch):
    """The profiler must not inherit the cores the workload is holding.

    This is the whole failure: the capture ran with the workload's own core
    list and asked for cores that were already taken.
    """
    monkeypatch.setenv(cores.VISIBLE_CORES, "0")
    monkeypatch.setenv(cores.RESERVED_CORE, "1")
    assert profiler._environment()[cores.VISIBLE_CORES] == "1"


# -- the orchestrator's side -------------------------------------------------

def test_orchestrator_reserves_the_last_core(monkeypatch):
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.delenv(cores.VISIBLE_CORES, raising=False)
    monkeypatch.delenv(cores.RESERVED_CORE, raising=False)

    reserved = pantheon_neuron.reserve_profiler_core(_devices(1))

    assert reserved == "1"
    assert os.environ[cores.VISIBLE_CORES] == "0"
    assert os.environ[cores.RESERVED_CORE] == "1"


def test_reservation_counts_cores_across_devices(monkeypatch):
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.delenv(cores.VISIBLE_CORES, raising=False)
    monkeypatch.delenv(cores.RESERVED_CORE, raising=False)

    reserved = pantheon_neuron.reserve_profiler_core(_devices(2))

    assert reserved == "3"
    assert os.environ[cores.VISIBLE_CORES] == "0-2"


def test_an_explicit_core_pin_is_not_overruled(monkeypatch):
    """Someone who pinned cores by hand answered this question already."""
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.setenv(cores.VISIBLE_CORES, "1")
    monkeypatch.delenv(cores.RESERVED_CORE, raising=False)

    assert pantheon_neuron.reserve_profiler_core(_devices(1)) is None
    assert os.environ[cores.VISIBLE_CORES] == "1"
    assert cores.RESERVED_CORE not in os.environ


def test_an_all_cores_workload_blocks_the_reservation(monkeypatch):
    """An aggregate measurement must not quietly lose a core.

    memory_read_agg declares cores: "all". Reserving one would still
    produce a number, and it would be the aggregate of all-but-one core
    under a name that claims otherwise. A missing profiler Score announces
    itself in the row; a Score over the wrong core count does not.
    """
    from kernels import registry

    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.delenv(cores.VISIBLE_CORES, raising=False)
    monkeypatch.delenv(cores.RESERVED_CORE, raising=False)

    agg = [w for w in registry.WORKLOADS if w.name == "memory_read_agg"]
    assert agg and agg[0].problem["cores"] == "all"

    assert pantheon_neuron.reserve_profiler_core(_devices(1), agg) is None
    assert cores.VISIBLE_CORES not in os.environ


def test_single_core_workloads_still_reserve(monkeypatch):
    """The bandwidth workloads declare cores: 1, so the split is faithful."""
    from kernels import registry

    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.delenv(cores.VISIBLE_CORES, raising=False)
    monkeypatch.delenv(cores.RESERVED_CORE, raising=False)

    single = [w for w in registry.WORKLOADS
              if w.name in ("memory_read", "memory_write")]
    assert all(w.problem["cores"] == 1 for w in single)

    assert pantheon_neuron.reserve_profiler_core(_devices(1), single) == "1"


def test_mock_mode_reserves_nothing(monkeypatch):
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    monkeypatch.delenv(cores.VISIBLE_CORES, raising=False)
    monkeypatch.delenv(cores.RESERVED_CORE, raising=False)

    assert pantheon_neuron.reserve_profiler_core(_devices(1)) is None
    assert cores.VISIBLE_CORES not in os.environ


def test_single_core_part_reserves_nothing(monkeypatch):
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.delenv(cores.VISIBLE_CORES, raising=False)
    monkeypatch.delenv(cores.RESERVED_CORE, raising=False)

    assert pantheon_neuron.reserve_profiler_core(_devices(1, cores_each=1)) is None
    assert cores.VISIBLE_CORES not in os.environ


def test_the_default_selection_cannot_reach_the_profiler(monkeypatch, capsys):
    """`--test all` is the README's first example and it pays this cost.

    The reservation is all-or-nothing for a run, and `all` selects
    memory_read_agg, so memory_read and memory_write cannot reach the
    neuron-profile Score they declare. That is a deliberate trade -- an
    aggregate over the wrong core count is worse than a labelled fallback
    -- but it is the documented invocation, so it is pinned here rather
    than rediscovered on an instance that bills by the hour.
    """
    from kernels import registry

    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.delenv(cores.VISIBLE_CORES, raising=False)
    monkeypatch.delenv(cores.RESERVED_CORE, raising=False)

    everything = registry.resolve("all")
    assert pantheon_neuron.reserve_profiler_core(_devices(1), everything) is None
    assert cores.VISIBLE_CORES not in os.environ

    # The cost is named, not left as "these Scores".
    printed = capsys.readouterr().out
    assert "memory_read" in printed and "memory_write" in printed
    assert registry.PROFILER in printed


def test_the_memory_suite_pays_the_same_cost(monkeypatch):
    """`--test memory` selects the aggregates too, so it is no way round it."""
    from kernels import registry

    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.delenv(cores.VISIBLE_CORES, raising=False)
    monkeypatch.delenv(cores.RESERVED_CORE, raising=False)

    memory = registry.resolve("memory")
    assert pantheon_neuron.reserve_profiler_core(_devices(1), memory) is None
    assert cores.VISIBLE_CORES not in os.environ


def test_reservation_cost_names_who_pays():
    """The billed list is the profiler-sourced workloads, not the aggregates."""
    from kernels import registry

    aggregate, billed = pantheon_neuron.reservation_cost(registry.resolve("all"))
    # The collectives too: nccom-test starts a worker on every core.
    assert set(aggregate) == {"memory_read_agg", "memory_write_agg",
                              "all_reduce", "p2p_thrasher"}
    assert set(billed) == {"memory_read", "memory_write"}

    # Every billed workload really does declare the profiler, and no
    # aggregate is billed for a reservation it refused.
    by_name = {w.name: w for w in registry.WORKLOADS}
    assert all(by_name[n].score_source.source == registry.PROFILER
               for n in billed)
    assert not set(aggregate) & set(billed)


def test_a_selection_without_aggregates_is_charged_nothing():
    from kernels import registry

    single = [w for w in registry.WORKLOADS
              if w.name in ("memory_read", "memory_write")]
    assert pantheon_neuron.reservation_cost(single) == ([], [])
