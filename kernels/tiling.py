"""Tile geometry shared by the HBM bandwidth kernels.

Confirmed on hardware: ``nl.tile_size.pmax`` reports 128 on both
NeuronCore-v2 parts probed (inf2.xlarge 2026-08-26, trn1.2xlarge
2026-08-27). The partition dimension is a hardware limit, not a tunable.
"""

import typing


PARTITION = 128
FREE_ELEMENTS = 2048
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "int8": 1, "uint8": 1}

# Operands the Tensor Engine treats as integers, so a workload can ask
# "is this integer arithmetic?" without enumerating widths at every call
# site. It matters for three separate decisions -- the accumulator type,
# the unit (TOPS rather than TFLOPS), and the analytic basis -- and each
# was written as `dtype == "int8"` until uint8 arrived.
INTEGER_DTYPES = frozenset({"int8", "uint8"})


def is_integer(dtype: str) -> bool:
    return dtype in INTEGER_DTYPES


# Which dtypes the part will run, by the path the kernel takes to reach it.
#
# These are two different questions and the first draft of this table
# collapsed them, which is the same mistake it exists to prevent. The
# NKI operand list was transcribed from a prose comment rather than
# measured, and a probe on trn1.2xlarge 2026-09-10 falsified it for the
# XLA path within the hour.

# NKI path -- kernels calling nc_matmul directly (tensor_virus and the
# workloads built on it).
#
# `nc_matmul does not support stationary.dtype=int8`, trn1.2xlarge
# 2026-09-08. That refusal is why int_virus pins uint8.
NKI_OPERANDS = frozenset({"bf16", "fp16", "fp32", "uint8"})

# XLA path -- kernels expressing a matmul in torch and letting
# neuronx-cc compile it (everything in inference_mix, llm_inference,
# encoders, transformer_compute).
#
# int8 is here and absent from NKI_OPERANDS, which is the finding: the
# two paths do not accept the same set, so "the engine supports X" is
# not a well-formed claim without saying which path asked.
#
# fp8_e4m3 was in the first draft of this table on the strength of a
# prose comment and is not here, because neuronx-cc refuses it:
# `[NCC_ESPP047] Data type F8E4M3FN is not supported`, trn1.2xlarge
# 2026-09-10, libneuronxla 2.2.15515.0.
XLA_OPERANDS = frozenset({"bf16", "fp16", "fp32", "int8", "uint8"})

# Measured T-ops/s at 4096^3, all five in one process so nothing differs
# but the dtype (trn1.2xlarge, 2026-09-10, 20s each, product verified
# exact against all-ones arithmetic in every case):
#
#     int8 -> int32     18.46      0.26x bf16
#     int8 direct       18.45      0.26x bf16
#     uint8 -> int32    72.78      1.03x bf16
#     bf16              70.38      1.00x
#     fp8_e4m3          refused by neuronx-cc
#
# Two things follow, and both contradict what a reader assumes about a
# workload named "quantized":
#
# 1. int8 through XLA is **the slowest path on the part**, not the
#    fastest. Eight-bit is an accuracy and footprint decision here, not
#    a throughput one.
# 2. uint8 merely matches bf16. There is no 8-bit speedup to measure in
#    either direction, because the operands are promoted to int32 before
#    the matmul and int32 is the width the arithmetic actually runs at.
#
# `int8 direct` and `int8 -> int32` agreeing to three digits is the
# evidence for that promotion: an explicit `.to(torch.int32)` on the
# operands changes nothing, because XLA had already inserted it.
OPERAND_RATES_4096 = {
    "int8": 18.46, "uint8": 72.78, "bf16": 70.38,
}

OPERAND_REFUSALS = {
    ("nki", "int8"): ("nc_matmul does not support stationary.dtype=int8 "
                      "(trn1.2xlarge, 2026-09-08) -- use uint8"),
    ("xla", "fp8_e4m3"): ("[NCC_ESPP047] Data type F8E4M3FN is not "
                          "supported (trn1.2xlarge, 2026-09-10)"),
}


def engine_accepts(dtype: str, path: str = "xla") -> bool:
    """Whether ``path`` will run ``dtype`` as a matmul operand.

    ``path`` is required in spirit and defaulted in practice: every
    workload with a pinned dtype except int_virus reaches the engine
    through XLA.
    """
    table = NKI_OPERANDS if path == "nki" else XLA_OPERANDS
    return dtype in table


def refusal(dtype: str, path: str = "xla") -> typing.Optional[str]:
    """The measurement that rejected ``dtype`` on ``path``, if any."""
    return OPERAND_REFUSALS.get((path, dtype))


def tile_plan(total_bytes: int, dtype: str,
              free: int = FREE_ELEMENTS) -> typing.Dict[str, int]:
    """Split a requested byte count into whole tiles ``free`` elements wide.

    ``actual_bytes`` is what the kernel will really move: the request
    rounded *down* to a whole number of tiles. A Score must be computed
    from the actual figure -- dividing by the requested figure when the
    kernel moved less inflates the result.
    """
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    element = DTYPE_BYTES[dtype]
    tile_bytes = PARTITION * free * element

    tiles = total_bytes // tile_bytes
    if tiles < 1:
        raise ValueError(
            f"requested {total_bytes} bytes is smaller than one "
            f"{tile_bytes}-byte tile"
        )
    return {
        "tiles": tiles,
        "tile_bytes": tile_bytes,
        "actual_bytes": tiles * tile_bytes,
        "partition": PARTITION,
        "free": free,
        "element_bytes": element,
    }


# int8 is here for the integer compute workloads. It is deliberately absent
# from the bandwidth kernels' usable set in practice: their tile geometry is
# expressed in elements, so a 1-byte dtype quarters the bytes moved per tile
# and would make a GB/s figure incomparable with the bf16 runs beside it.
TORCH_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32",
                "int8": "int8", "uint8": "uint8"}


def torch_dtype(dtype: str):
    """Map our dtype name onto a torch dtype, imported lazily."""
    import torch  # type: ignore

    if dtype not in TORCH_DTYPES:
        raise ValueError(f"no torch dtype for {dtype!r}")
    return getattr(torch, TORCH_DTYPES[dtype])
