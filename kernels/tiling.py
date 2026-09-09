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


def tile_plan(total_bytes: int, dtype: str) -> typing.Dict[str, int]:
    """Split a requested byte count into whole tiles.

    ``actual_bytes`` is what the kernel will really move: the request
    rounded *down* to a whole number of tiles. A Score must be computed
    from the actual figure -- dividing by the requested figure when the
    kernel moved less inflates the result.
    """
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    element = DTYPE_BYTES[dtype]
    tile_bytes = PARTITION * FREE_ELEMENTS * element

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
        "free": FREE_ELEMENTS,
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
