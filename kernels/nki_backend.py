"""NKI availability and the CPU fallback used for mock runs.

There is no hand-written device C++ path on Neuron the way there is on CUDA.
Deliberate stress patterns go through NKI (the Neuron Kernel Interface), a
tile-based Python DSL that ``neuronx-cc`` lowers to NeuronCore instructions.
Graph-level workloads built with torch-neuronx are subject to compiler
optimisation and will happily have a stress loop folded away, so anything
that must actually keep the hardware busy belongs in NKI.
"""

import os
import typing


class BackendUnavailable(RuntimeError):
    pass


def mock_mode() -> bool:
    return os.environ.get("PANTHEON_NEURON_MOCK") == "1"


def probe() -> typing.Dict[str, typing.Optional[str]]:
    """Report which pieces of the Neuron toolchain are importable."""
    found: typing.Dict[str, typing.Optional[str]] = {
        "neuronxcc": None,
        "torch_neuronx": None,
        "nki": None,
    }
    if mock_mode():
        return found

    try:
        import neuronxcc  # type: ignore

        found["neuronxcc"] = getattr(neuronxcc, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import torch_neuronx  # type: ignore

        found["torch_neuronx"] = getattr(torch_neuronx, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import neuronxcc.nki  # type: ignore  # noqa: F401

        found["nki"] = found["neuronxcc"]
    except ImportError:
        pass

    return found


def require_toolchain() -> typing.Dict[str, typing.Optional[str]]:
    """Fail loudly rather than silently degrading to a meaningless run."""
    if mock_mode():
        return probe()
    found = probe()
    if found["neuronxcc"] is None:
        raise BackendUnavailable(
            "neuronx-cc not importable. Install the Neuron SDK "
            "(pip install neuronx-cc torch-neuronx) or set "
            "PANTHEON_NEURON_MOCK=1 for a CPU mock run."
        )
    return found
