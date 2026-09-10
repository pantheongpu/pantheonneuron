PYTHON ?= python3

.PHONY: help test lint fmt cov mock list clean ci

help:
	@echo "make test   - run the test suite (no hardware required)"
	@echo "make lint   - static checks (ruff); the same ones CI runs"
	@echo "make fmt    - apply the fixes lint can make automatically"
	@echo "make cov    - test suite with coverage, against the floor CI enforces"
	@echo "make ci     - everything CI runs, in the order it runs it"
	@echo "make mock   - full mock run of every workload"
	@echo "make list   - list workloads and their required capabilities"
	@echo "make clean  - remove caches and generated reports"

test:
	$(PYTHON) -m pytest tests/ -q

# `lint` was listed in .PHONY and in help above with no recipe, so it
# printed "Nothing to be done for 'lint'" and exited 0. A CI step calling
# it would have been green forever while checking nothing.
#
# tests/test_makefile.py now asserts every target this help text advertises
# has a recipe, because the failure was silent in both directions: nothing
# ran, and nothing said nothing ran.
lint:
	$(PYTHON) -m ruff check .

fmt:
	$(PYTHON) -m ruff check . --fix

# 78, because the measured figure on 2026-09-10 was 79%. The floor is set
# below what the suite achieves so it ratchets against a real drop rather
# than failing the day it is introduced -- a coverage number picked as an
# aspiration is a check that gets waived the first time it is
# inconvenient, and a waived check is the kind this repo catalogues.
#
# Raise it when the figure rises. Do not raise it to make a point.
cov:
	$(PYTHON) -m pytest tests/ -q \
		--cov=. --cov-report=term-missing:skip-covered \
		--cov-fail-under=$${COVERAGE_FLOOR:-78}

ci: lint test mock

mock:
	PANTHEON_NEURON_MOCK=1 $(PYTHON) pantheon_neuron.py \
		--duration 2 --monitor-period 0.05

list:
	$(PYTHON) pantheon_neuron.py --list

clean:
	rm -rf .pytest_cache __pycache__ kernels/__pycache__ tests/__pycache__
	rm -rf .ruff_cache .coverage htmlcov
	rm -rf database results
