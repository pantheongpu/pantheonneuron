"""The two tools that only mean anything on a device: what can be held off it.

tools/dma_levers.py and tools/probe_load.py had no test at all. Their
measurements need a NeuronCore, so the honest coverage is what precedes and
surrounds them: dma_levers' tile plan and its handling of a bad setting (a
traceback on a rented instance until 2026-09-27), and probe_load compiling
and naming the same venv the validation script uses, so its documented
command is not a path that stopped existing.
"""

import os
import py_compile
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)

import dma_levers  # noqa: E402


def test_dma_levers_plans_aws_rule_of_thumb_rows():
    """128 partitions, at least 4 KiB per partition row -- AWS's DMA advice."""
    assert dma_levers.PARTITION == 128
    assert dma_levers.FREE * 2 >= 4096          # bf16 is two bytes


def test_dma_levers_plan_is_whole_tiles():
    rows, actual = dma_levers._tile_rows(2 * 1024 ** 3)
    assert actual == rows * dma_levers.PARTITION * dma_levers.FREE * 2
    assert actual <= 2 * 1024 ** 3


@pytest.mark.parametrize("env", [{"GIB": "x"}, {"SECONDS": "ten"},
                                 {"GIB": "0"}, {"SECONDS": "-5"}])
def test_dma_levers_refuses_a_bad_setting_in_a_sentence(env):
    done = subprocess.run([sys.executable, os.path.join(TOOLS, "dma_levers.py")],
                          env=dict(os.environ, **env), capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 2
    assert "Traceback" not in done.stderr
    assert "dma_levers:" in done.stdout


def test_dma_levers_without_a_device_says_so_and_exits_2():
    env = {key: value for key, value in os.environ.items()
           if key != "PANTHEON_NEURON_MOCK"}
    done = subprocess.run([sys.executable, os.path.join(TOOLS, "dma_levers.py")],
                          env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 2
    assert "Neuron device" in done.stdout


def test_probe_load_compiles():
    """It imports torch_neuronx at the top, so it cannot be imported here;
    compiling it is what catches it rotting between hardware runs."""
    py_compile.compile(os.path.join(TOOLS, "probe_load.py"), doraise=True)


def test_probe_load_documents_the_venv_the_validation_script_uses():
    with open(os.path.join(TOOLS, "probe_load.py"), encoding="utf-8") as handle:
        documented = set(re.findall(r"/opt/aws_neuronx_venv_\w+", handle.read()))
    with open(os.path.join(TOOLS, "validate_hardware.sh"), encoding="utf-8") as handle:
        used = set(re.findall(r"/opt/aws_neuronx_venv_\w+", handle.read()))
    assert documented and used
    assert documented == used, (documented, used)
