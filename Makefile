PYTHON ?= python3

.PHONY: help test lint mock list clean

help:
	@echo "make test   - run the test suite (no hardware required)"
	@echo "make mock   - full mock run of every workload"
	@echo "make list   - list workloads and their required capabilities"
	@echo "make clean  - remove caches and generated reports"

test:
	$(PYTHON) -m pytest tests/ -q

mock:
	PANTHEON_NEURON_MOCK=1 $(PYTHON) pantheon_neuron.py \
		--duration 2 --monitor-period 0.05

list:
	$(PYTHON) pantheon_neuron.py --list

clean:
	rm -rf .pytest_cache __pycache__ kernels/__pycache__ tests/__pycache__
	rm -rf database results
