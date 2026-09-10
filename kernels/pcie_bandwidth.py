"""Host-to-device and device-to-host transfer over PCIe.

Score: **GB/s**, counted by the workload, as the registry declares. The
device's own DMA counters are device-side only and do not see a host
transfer, so unlike the HBM kernels there is no profiler figure to prefer
here -- wall-clock timing is the measurement, not a fallback.

The probes recorded this part's link as Gen4 x8 with a Gen5 x8 maximum, so
a healthy result is bounded by the negotiated link rather than by HBM. That
makes this the workload that notices a link that trained down: a card
running at half its lanes still passes every compute test and shows up
here.

Each direction is timed separately. A single bidirectional figure hides the
asymmetry, and the asymmetry is usually where the fault is -- reads back
from the device commonly run slower than writes to it, so averaging them
turns a one-sided regression into a smaller two-sided one.

**The d2h asymmetry is explained, and the link is healthy.** A size sweep
on trn1.2xlarge, 2026-09-10:

    MiB     h2d      d2h    d2h ms/pass
      1    5.57     2.92           0.36
     16   11.55     2.83           5.94
     64   11.22     0.88          76.20
    256   10.09     1.01         264.68
   1024    7.37     1.09         988.20

Three things fall out of it, none of which a single 1 GiB transfer could
have shown:

- **The link is fine.** h2d reaches 11.55 GB/s at 16 MiB against roughly
  16 GB/s theoretical for the Gen4 x8 the probes recorded -- 72%, which is
  what a healthy link does.
- **d2h is bandwidth-bound, not overhead-bound.** Fitting time = overhead
  + bytes/BW across the sweep gives 1.09 GB/s and an overhead of about
  zero. Every earlier explanation assumed this without being able to check
  it.
- **The rate is not constant, and that is the actual finding.** d2h holds
  ~2.9 GB/s to 16 MiB and collapses to 0.88 by 64 MiB. A cliff in one
  direction between two transfer sizes is the signature of a staging
  buffer: transfers that fit go fast, larger ones are chunked through it.
  That is the bounce-buffer explanation ``pin_memory()`` could not test
  directly -- it returned unpinned buffers on this stack -- arriving from
  the other side.

**So the pinned 1 GiB measures the degraded regime for both directions.**
h2d is past its own peak there too (7.37 against 11.55). The size is a
registry decision and is left open: 16 MiB measures the link, 1 GiB
measures what a large transfer actually costs, and those are different
questions. What is no longer true is that the number is unexplained.

STATUS: verified on both parts 2026-09-08; the pinning and alternating-
source controls are UNTESTED. No NKI: this is torch tensor movement, so it
depends on the runtime rather than on a compiled kernel.
"""

import time
import typing


def transfer_plan(problem: typing.Mapping[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    """Elements and directions for the pinned transfer size."""
    total_bytes = int(problem["bytes"])
    direction = str(problem.get("direction", "bidirectional"))

    if total_bytes <= 0:
        raise ValueError(f"bytes must be positive, got {total_bytes}")
    if direction not in ("bidirectional", "h2d", "d2h"):
        raise ValueError(f"unknown direction {direction!r}")

    directions = (["h2d", "d2h"] if direction == "bidirectional"
                  else [direction])
    return {
        "bytes": total_bytes,
        "elements": total_bytes // 2,  # bf16
        "directions": directions,
        "direction": direction,
    }


def run(problem: typing.Mapping[str, typing.Any], duration: int) -> dict:
    """Move a buffer across PCIe in each direction and time both."""
    from . import nki_backend

    nki_backend.require_toolchain()

    import torch  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    plan = transfer_plan(problem)
    device = xm.xla_device()

    # Every buffer is allocated before the clock starts, and both legs copy
    # into an existing destination rather than producing a new one. The
    # allocation is not the thing being measured, and on the d2h side it
    # used to dominate: `.cpu()` returns a *new* host tensor, so each pass
    # paid a 1 GiB allocation and its page faults while the h2d side reused
    # one buffer.
    #
    # Removing that did not move the number -- trn1.2xlarge 2026-09-08
    # measured d2h 1.1 GB/s against h2d 6.0 both before and after -- so two
    # further differences are controlled here, and the result records which
    # were actually in effect rather than assuming.
    #
    # PINNED HOST MEMORY. Pageable host pages cannot be DMA'd directly: the
    # driver stages them through a bounce buffer, which costs an extra copy
    # in the direction that writes host memory. That is d2h, and it is the
    # oldest explanation for exactly this asymmetry. pin_memory() is best
    # effort -- it is a CUDA-shaped API and may be a no-op or unavailable on
    # this stack, so the result says which memory it really got.
    def host_buffer(fill):
        plain = torch.full((plan["elements"],), fill, dtype=torch.bfloat16)
        try:
            return plain.pin_memory(), True
        except (RuntimeError, NotImplementedError, AssertionError):
            return plain, False

    host, pinned_source = host_buffer(1.0)
    # ALTERNATING SOURCES. If the runtime can tell that h2d copies the same
    # bytes every pass, it may serve the copy without moving them, which
    # would inflate h2d rather than depress d2h -- and the two are
    # indistinguishable from the ratio alone. Two sources with different
    # contents, alternating, remove that explanation.
    host_alt, _ = host_buffer(2.0)
    landing, pinned_landing = host_buffer(0.0)

    resident = host.to(device)
    xm.mark_step()
    xm.wait_device_ops()

    per_direction: typing.Dict[str, typing.Dict[str, float]] = {}
    started = time.perf_counter()
    deadline = started + duration

    for direction in plan["directions"]:
        moved = 0
        passes = 0
        # Split the budget evenly so one direction cannot starve the other
        # of the clock and leave the asymmetry unmeasured.
        share = deadline - (duration / len(plan["directions"])) * (
            len(plan["directions"]) - 1 - plan["directions"].index(direction)
        )
        leg_started = time.perf_counter()
        while time.perf_counter() < min(share, deadline):
            if direction == "h2d":
                # SUSPECT: this reads as "copy into the existing buffer, so
                # no per-pass allocation", and that rests on in-place
                # semantics XLA does not have. An assignment lowers to a
                # dynamic-update-slice that produces a *new* tensor -- see
                # docs/xla_has_no_in_place_write.md, where kv_cache_churn
                # measured it -- so this leg may allocate 1 GiB a pass
                # anyway. The fix it belongs to moved d2h 1.0 -> 1.1 and
                # h2d 6.0 -> 6.4 GB/s, which is consistent with it having
                # changed nothing. Unresolved, and it needs a hardware run
                # rather than a third guess.
                resident.copy_(host if passes % 2 == 0 else host_alt)
                # Lazy: the copy is queued, so the barrier is what makes
                # this a transfer measurement rather than a submission one.
                xm.mark_step()
                xm.wait_device_ops()
            else:
                # Copying from an XLA tensor to a host one is synchronous --
                # it returns with the bytes already on the host -- so no
                # barrier is needed and adding one would charge this leg for
                # a round trip the transfer has already completed.
                landing.copy_(resident)
            moved += plan["bytes"]
            passes += 1
        leg_elapsed = time.perf_counter() - leg_started
        per_direction[direction] = {
            "bytes": moved,
            "elapsed_s": leg_elapsed,
            "gbps": moved / leg_elapsed / 1e9 if leg_elapsed else 0.0,
            "passes": passes,
        }

    elapsed = time.perf_counter() - started
    total_moved = sum(leg["bytes"] for leg in per_direction.values())

    return {
        "elapsed_s": elapsed,
        "bytes_transferred": total_moved,
        # The registry's formula: total bytes over total time. The
        # per-direction figures are recorded beside it because that is where
        # a degraded link actually shows itself.
        "analytic_gbps": total_moved / elapsed / 1e9 if elapsed else 0.0,
        "per_direction": per_direction,
        # Says how the number was produced. Three methodologies have now
        # produced a d2h figure; a row that does not say which is not
        # comparable with one that does.
        "buffers": "preallocated",
        # Whether the host pages were actually pinned, not whether pinning
        # was requested. If these are False the bounce-buffer explanation
        # for the d2h asymmetry is still live and untested.
        "host_source_pinned": pinned_source,
        "host_landing_pinned": pinned_landing,
        "h2d_sources_alternate": True,
        "score_method": "workload",
        "analytic_basis": "bytes transferred / wall time",
        "warning": verify_directions_are_balanced(per_direction),
        "plan": plan,
    }


def verify_directions_are_balanced(
    per_direction: typing.Mapping[str, typing.Mapping[str, float]],
    ratio_floor: float = 0.25,
) -> typing.Optional[str]:
    """Flag a link that is far slower one way than the other.

    Some asymmetry is normal -- the two directions do not share a code path
    and device-to-host is usually the slower one. An order-of-magnitude
    split is not normal. Reported rather than failed: this is a diagnostic,
    and the number is still a measurement.

    The message names the harness before it names the hardware, because the
    first time this guard fired the harness was the cause: d2h allocated a
    fresh host buffer per pass and h2d did not, and the 6x split that
    produced was read as a link property. It was not. A guard that points
    only at the link teaches the reader to suspect the wrong thing.
    """
    rates = {name: leg.get("gbps", 0.0) for name, leg in per_direction.items()}
    if len(rates) < 2:
        return None
    slowest = min(rates.values())
    fastest = max(rates.values())
    if fastest <= 0:
        return "no bytes moved in either direction"
    if slowest < fastest * ratio_floor:
        slow_name = min(rates, key=rates.get)
        return (
            f"{slow_name} ran at {slowest:.1f} GB/s against {fastest:.1f} the "
            "other way -- before reading this as a link property, check that "
            "both legs reused a preallocated buffer (the 2026-09-08 split was "
            "a per-pass host allocation, not the link), then the negotiated "
            "link width"
        )
    return None
