"""The conftest restores process state between tests; these prove it does.

Definition order within a file is the order pytest runs, so the second test
of each pair sees exactly what the first one left behind.
"""

import os

import pantheon_neuron
from kernels import cores

_MARK = "PANTHEON_NEURON_ISOLATION_PROBE"


def test_1_a_test_leaves_state_behind():
    # Written directly, the way reserve_profiler_core writes it -- not via
    # monkeypatch, which would clean up after itself and prove nothing.
    os.environ[_MARK] = "leaked"
    os.environ[cores.VISIBLE_CORES] = _MARK
    pantheon_neuron._LAST_RUN["tensor_virus"] = {_MARK: True}


def test_2_the_next_test_does_not_inherit_it():
    assert _MARK not in os.environ
    assert os.environ.get(cores.VISIBLE_CORES) != _MARK
    assert _MARK not in (pantheon_neuron._LAST_RUN.get("tensor_virus") or {})
