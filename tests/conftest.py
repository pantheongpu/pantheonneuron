"""Suite-wide isolation: every test gets the environment and run state it started with.

**The suite was order-dependent, and forward order hid it.** Run with the test
files reversed, ``test_a_workload_with_no_duty_cycle_gets_the_full_ceiling``
failed -- tensor_virus's ceiling read 95 TFLOPS against the part's 190 --
because a test in test_profiler_cores.py had left ``NEURON_RT_VISIBLE_CORES``
pinned to one core, and peak_share caps its core count by that variable.

The tests were not careless. They called ``monkeypatch.delenv`` before
reserving a core, which reads like cleanup and is not: when the variable is
absent, ``delenv`` records nothing to restore, and ``reserve_profiler_core``
then writes ``os.environ`` directly. The write outlived the test. Forward
order happened to run every reader of that variable first.

``pantheon_neuron._LAST_RUN`` is the other shared state tests write. The ones
that seed it do pop it afterwards -- but inline, not in a finally, so an
assertion that fails between the seed and the pop leaves it for every later
test, and the next failure is reported against a test that did nothing wrong.

Restoring both around every test closes the class, including the leaks a
failing test would cause, rather than patching the one that was found.
"""

import os

import pytest

import pantheon_neuron


@pytest.fixture(autouse=True)
def _restore_process_state():
    environment = dict(os.environ)
    last_run = {name: dict(result) if isinstance(result, dict) else result
                for name, result in pantheon_neuron._LAST_RUN.items()}
    try:
        yield
    finally:
        # In place, not rebinding: code under test holds references to
        # os.environ and to the _LAST_RUN dict itself.
        os.environ.clear()
        os.environ.update(environment)
        pantheon_neuron._LAST_RUN.clear()
        pantheon_neuron._LAST_RUN.update(last_run)
