"""registry.EQUIVALENT_PINS: an old pin is the same run only with the proof.

A refusal to compare different pins is the right default, and this is the one
way past it, so each entry has to earn its place: a real workload, a pin that
really is different from today's, and a reason citing what changed and the
measurement showing it changed nothing.
"""

import pytest

from kernels import registry


@pytest.mark.parametrize("name", sorted(registry.EQUIVALENT_PINS))
def test_every_entry_is_a_real_workload_with_a_real_difference(name):
    old_pin, why = registry.EQUIVALENT_PINS[name]
    current = registry.resolve(name)[0].problem
    assert dict(old_pin) != dict(current), "an identical pin needs no entry"
    assert "#" in why and len(why) > 40, "cite the change and the evidence"


def test_same_pin_is_symmetric_and_scoped():
    old_pin = registry.EQUIVALENT_PINS["serving_mix"][0]
    current = registry.resolve("serving_mix")[0].problem
    assert registry.same_pin("serving_mix", old_pin, current)
    assert registry.same_pin("serving_mix", current, old_pin)
    # Scoped to its own workload: the same dicts prove nothing elsewhere.
    assert not registry.same_pin("moe_router", old_pin, current)


def test_a_changed_value_is_not_covered_by_the_rename():
    current = dict(registry.resolve("serving_mix")[0].problem)
    assert not registry.same_pin("serving_mix", dict(current, decode=512), current)
