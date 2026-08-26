"""Generate sustained NeuronCore load for counter probing.

Used to produce data/probe-2026-08-26. Run on a Neuron instance with the
venv bin directory on PATH -- torch_neuronx shells out to libneuronpjrt-path
and fails with FileNotFoundError if it is not there:

    export PATH=/opt/aws_neuronx_venv_pytorch_2_8/bin:/opt/aws/neuron/bin:$PATH
    /opt/aws_neuronx_venv_pytorch_2_8/bin/python tools/probe_load.py
"""

import time

import torch
import torch.nn as nn
import torch_neuronx


class Block(nn.Module):
    def __init__(self, n=2048):
        super().__init__()
        self.a = nn.Linear(n, n, bias=False)
        self.b = nn.Linear(n, n, bias=False)

    def forward(self, x):
        for _ in range(8):
            x = torch.relu(self.b(self.a(x)))
        return x


def main(seconds: int = 120) -> None:
    print("[load] tracing...", flush=True)
    model = Block().eval()
    example = torch.randn(8, 2048)

    started = time.time()
    traced = torch_neuronx.trace(model, example)
    print(f"[load] compiled in {time.time() - started:.1f}s", flush=True)

    print(f"[load] running {seconds}s...", flush=True)
    deadline = time.time() + seconds
    iterations = 0
    while time.time() < deadline:
        traced(example)
        iterations += 1
    print(f"[load] done, {iterations} iterations", flush=True)


if __name__ == "__main__":
    main()
