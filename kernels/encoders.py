"""Retrieval embedding and vision encoding.

Score: **workload-counted**. Nothing in hardware counts vectors or image
tiles.

Both are encoders and neither is the other:

``rag_embedding``   wide, shallow, and batched. A retrieval pipeline embeds
                    hundreds of short texts at once, so the shape is a large
                    batch of small vectors and the cost is dominated by the
                    projection plus the L2 normalisation every vector store
                    requires. Skipping the normalisation would measure a
                    matmul, not an embedding.
``vision_encoder``  patchification first, then a transformer over the
                    patches. At 224 pixels and patch 14 that is 256 patches
                    per image, so the sequence is fixed by geometry rather
                    than chosen -- and the patch projection is a distinct
                    cost no text model pays.

STATUS: UNTESTED ON HARDWARE.
"""

import time
import typing

from . import nki_backend, tiling, transformer_ops


def run_rag_embedding(problem: typing.Mapping[str, typing.Any],
                      duration: int) -> dict:
    """Embed a batch of vectors and normalise them. Count vectors."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    dim = int(problem["dim"])
    batch = int(problem["batch"])
    dtype = tiling.torch_dtype(str(problem["dtype"]))

    device = xm.xla_device()
    inputs = torch.ones((batch, dim), dtype=dtype, device=device)
    projection = torch.ones((dim, dim), dtype=dtype, device=device)
    xm.mark_step()

    def embed():
        hidden = torch.matmul(inputs, projection)
        activated = torch.nn.functional.gelu(hidden)
        # L2 normalisation in fp32. Every vector store expects unit vectors,
        # and doing it in bf16 would both misreport the cost and lose enough
        # precision that the norm is not one.
        vectors = torch.matmul(activated, projection).float()
        norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        return vectors / norm.clamp_min(1e-6)

    warm = embed()
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    sink = None
    vectors = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        sink = embed()
        xm.mark_step()
        vectors += batch
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    passes = vectors // batch if batch else 0
    flops = passes * 2 * (2 * batch * dim * dim)

    return {
        "vectors_embedded": vectors,
        "elapsed_s": elapsed,
        "embedding_vectors_per_s": vectors / elapsed if elapsed else 0.0,
        "flops_issued": flops,
        "score_method": "workload",
        "analytic_basis": "vectors embedded / wall time",
        "warning": transformer_ops.verify_output_is_a_number(
            transformer_ops.read_back(sink), "embedding"),
    }


def patch_plan(problem: typing.Mapping[str, typing.Any]) -> typing.Dict[str, int]:
    """Patch geometry for the pinned resolution.

    The sequence length of a vision encoder is not a free parameter: it is
    (resolution / patch)^2. A resolution that does not divide by the patch
    size would silently drop a strip of every image, so it is rejected.
    """
    resolution = int(problem["resolution"])
    patch = int(problem["patch"])
    batch = int(problem["batch"])

    if patch <= 0 or resolution <= 0:
        raise ValueError(f"invalid geometry: {resolution}px, patch {patch}")
    if resolution % patch:
        raise ValueError(
            f"resolution {resolution} is not divisible by patch {patch} -- "
            "the encoder would drop a strip of every image"
        )
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")

    per_side = resolution // patch
    return {
        "resolution": resolution,
        "patch": patch,
        "batch": batch,
        "per_side": per_side,
        "patches": per_side * per_side,
        # Three channels, flattened per patch: the projection's input width.
        "patch_dim": patch * patch * 3,
    }


def run_vision_encoder(problem: typing.Mapping[str, typing.Any],
                       duration: int) -> dict:
    """Patchify, project, then run a transformer block. Count image tiles."""
    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    plan = patch_plan(problem)
    dtype = tiling.torch_dtype(str(problem["dtype"]))
    hidden = 1024

    device = xm.xla_device()
    # Images as flattened patches. Starting from patches rather than from a
    # [B, C, H, W] image is deliberate: an unfold on the host would put data
    # movement the device never does inside the timed region.
    patches = torch.ones(
        (plan["batch"], plan["patches"], plan["patch_dim"]),
        dtype=dtype, device=device,
    )
    patch_projection = torch.ones(
        (plan["patch_dim"], hidden), dtype=dtype, device=device
    )
    params = transformer_ops.weights(hidden, dtype, device, heads=16)
    xm.mark_step()

    def encode():
        embedded = torch.matmul(patches, patch_projection)
        return transformer_ops.block(embedded, params)

    warm = encode()
    xm.mark_step()
    xm.wait_device_ops()
    del warm

    sink = None
    tiles = 0
    images = 0
    started = time.perf_counter()
    deadline = started + duration
    while time.perf_counter() < deadline:
        sink = encode()
        xm.mark_step()
        images += plan["batch"]
        tiles += plan["batch"] * plan["patches"]
    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    return {
        "image_tiles": tiles,
        "images": images,
        "elapsed_s": elapsed,
        "image_tiles_per_s": tiles / elapsed if elapsed else 0.0,
        "patches_per_image": plan["patches"],
        "score_method": "workload",
        "analytic_basis": "image tiles / wall time",
        "warning": transformer_ops.verify_output_is_a_number(
            transformer_ops.read_back(sink), "encoder output"),
    }
