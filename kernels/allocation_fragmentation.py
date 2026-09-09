"""Device allocator under fragmentation pressure.

Score: **allocation-events/s**, counted by the workload itself, as the
registry declares. No hardware counter measures allocator behaviour, so
this is one of the workloads where the kernel is the only possible source.

What it looks for is not throughput but failure. A device allocator that
never coalesces freed blocks will serve a long run of same-sized requests
happily and then refuse a large one it has the free bytes for. So the
sizes are deliberately mixed and the frees deliberately interleaved: small
blocks are kept while large ones churn, which leaves the free list
punctured rather than tidy. The rate is what is reported; the interesting
result is an allocation that fails while the accounting says there is room.

No NKI here. This is the runtime's allocator, reached through ordinary
device tensors, so nothing in this file depends on a compiled kernel.

STATUS: UNTESTED ON HARDWARE.
"""

import time
import typing


# Enough live bytes to force reuse rather than a walk up fresh address
# space, and far enough below a NeuronCore's 16 GB that a healthy run never
# fails on capacity alone -- a failure here should mean fragmentation, not
# that the workload asked for more than the part has.
LIVE_BUDGET_BYTES = 2 << 30

# Every fourth block is retained. Small enough that the run makes progress,
# frequent enough that the retained blocks sit between the churning ones
# instead of clustering at one end.
KEEP_EVERY = 4


def size_sequence(problem: typing.Mapping[str, typing.Any]) -> typing.List[int]:
    """The allocation sizes, cycling between the pinned bounds.

    Deterministic rather than random: a fragmentation result that cannot be
    reproduced exactly is not evidence of anything, and a seeded RNG would
    still make the sequence depend on the Python version's implementation.
    The doubling walk covers the range in a way that mixes block classes,
    which is what punctures the free list.
    """
    low = int(problem["size_min"])
    high = int(problem["size_max"])
    count = int(problem["allocations"])

    if low <= 0 or high < low:
        raise ValueError(f"invalid size bounds: {low}..{high}")
    if count <= 0:
        raise ValueError(f"allocations must be positive, got {count}")

    sizes = []
    size = low
    for _ in range(count):
        sizes.append(size)
        size *= 2
        if size > high:
            size = low
    return sizes


def run(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Churn device allocations and return the event rate.

    ``duration`` bounds the run; the pinned allocation count bounds it the
    other way, so a fast part finishes early rather than padding the clock.
    """
    from . import nki_backend

    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    sizes = size_sequence(problem)
    device = xm.xla_device()

    retained: typing.List[typing.Any] = []
    live_bytes = 0
    events = 0
    failure = None

    started = time.perf_counter()
    deadline = started + duration

    for index, size in enumerate(sizes):
        if time.perf_counter() >= deadline:
            break

        elements = max(size // 2, 1)  # bf16
        try:
            block = torch.ones(elements, dtype=torch.bfloat16, device=device)
            # The allocation is lazy until the graph is cut, so without this
            # the loop would count intentions rather than allocations.
            xm.mark_step()
        except RuntimeError as error:
            failure = (
                f"allocation {index} of {size} bytes failed with "
                f"{live_bytes} bytes live: {error}"
            )
            break

        events += 1

        if index % KEEP_EVERY == 0:
            # The size travels with the block. Popping subtracts what was
            # actually freed, and the sizes here span 4 KiB to 16 MiB, so
            # subtracting the size of the block just allocated instead --
            # which is what this did -- lets the tally drift from reality.
            # The eviction loop hides most of it by running until the tally
            # drops under budget, but the tally is what the failure message
            # reports when an allocation fails, and at the pinned problem it
            # was overstating by about 19 MiB against 2 GiB held.
            retained.append((block, size))
            live_bytes += size
        # Everything else falls out of scope here, freeing a block from
        # between two retained ones.

        while live_bytes > LIVE_BUDGET_BYTES and retained:
            _, freed = retained.pop(0)
            live_bytes -= freed
            xm.mark_step()

    xm.wait_device_ops()
    elapsed = time.perf_counter() - started

    return {
        "events": events,
        "elapsed_s": elapsed,
        "allocation_events_per_s": events / elapsed if elapsed else 0.0,
        "retained_blocks": len(retained),
        "live_bytes": live_bytes,
        "score_method": "workload",
        "analytic_basis": "allocation events / wall time",
        "warning": failure,
        "requested": len(sizes),
    }
