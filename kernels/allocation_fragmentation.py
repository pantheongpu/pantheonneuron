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

    # Warm every distinct size before the clock starts. Each is its own
    # graph shape, so the first pass over the sequence compiles all of
    # them, and without this the compile lands inside the measurement.
    #
    # Measured on trn1.2xlarge 2026-09-10: three repeats read 549, then
    # 2,646, then 2,750 allocation-events/s -- monotonically rising, which
    # is warm-up, not noise. The first repeat was paying for thirteen
    # compiles and the later ones were hitting the cache.
    #
    # The same defect serving_mix had, and memory_read documents: a
    # measurement that includes its own compile is measuring the compiler.
    for size in sorted(set(sizes)):
        try:
            warm = torch.ones(max(size // 2, 1), dtype=torch.bfloat16,
                              device=device)
            xm.mark_step()
            del warm
        except RuntimeError:
            # A size that cannot be allocated at all is the run's own
            # business to discover and report; warming is best effort.
            break
    xm.wait_device_ops()

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

    # Which limit stopped the run. The pinned allocation count usually
    # does, long before the clock: 10,000 allocations take under four
    # seconds on trn1, so --duration 30 and --duration 60 both measure the
    # same four seconds. That is the documented design -- a fast part
    # should finish early rather than pad the clock -- but a row that
    # reports a Score without saying the window was four seconds invites
    # the reader to assume it was thirty.
    bounded_by = "allocations" if events >= len(sizes) else "duration"

    return {
        "events": events,
        "elapsed_s": elapsed,
        "allocation_events_per_s": events / elapsed if elapsed else 0.0,
        "retained_blocks": len(retained),
        "live_bytes": live_bytes,
        "score_method": "workload",
        "analytic_basis": "allocation events / wall time",
        "warning": failure or verify_window_is_long_enough(
            elapsed, duration, bounded_by),
        "requested": len(sizes),
        "bounded_by": bounded_by,
        "measured_window_s": round(elapsed, 3),
    }


# Below this, a rate over the window is dominated by whatever happened to
# happen in it. Measured on trn1.2xlarge 2026-09-10: the pinned problem
# runs for about 3.8 seconds however long a duration is asked for, and its
# repeats sit at a coefficient of variation around 0.15 -- the drift is
# gone since the warm-up, and this is what is left.
MIN_WINDOW_SECONDS = 5.0


def verify_window_is_long_enough(elapsed: float, duration: int,
                                 bounded_by: str) -> typing.Optional[str]:
    """Say so when the run measured far less time than was asked for.

    The allocation count bounds this workload, not the clock, so
    ``--duration`` mostly does not do what a reader expects. Raising it
    changes nothing; the way to a steadier number here is ``--repeat``, or
    a larger pinned count.
    """
    if bounded_by != "allocations" or elapsed >= MIN_WINDOW_SECONDS:
        return None
    return (
        f"measured {elapsed:.1f}s of a requested {duration}s: the pinned "
        "allocation count bounds this run, not the clock, so --duration "
        "does not lengthen it. A window this short is why the repeats "
        "scatter; use --repeat, or pin more allocations"
    )
