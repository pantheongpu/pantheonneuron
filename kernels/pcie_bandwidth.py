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

**So the pinned 1 GiB measured the degraded regime for both directions.**
h2d was past its own peak there too (7.37 against 11.55).

**Resolved 2026-09-22: the pin stays 1 GiB, because 16 MiB cannot be
run.** The sweep was re-measured through this kernel that day on a
trn1.2xlarge (tools/pcie_size_sweep.py, 10 s per size, one process each):

    MiB     h2d      d2h   combined
      1    5.80     2.28       4.04
      4    8.05     3.41       5.73
     16    8.89     3.93       6.41
     32    6.59     1.08       3.83
     64    6.65     1.08       3.86
    256    5.79     1.11       3.43
   1024    6.84     1.11       3.78

Both directions peak at 16 MiB and the cliff sits immediately past it, now
between 16 and 32 MiB rather than 16 and 64. On that evidence alone the pin
should move: this is the row that notices a link that trained down, and
that wants the size at which the link is what binds.

**It cannot move, and the reason is host memory.** At 16 MiB the runtime's
host footprint grows about 5 MB per transfer and does not come back:
measured 3.61, 6.20, 8.63 ... 28.92 GiB across 6,000 passes in 15 s, and a
60 s harness run at that pin was OOM-killed -- "Out of memory: Killed
process ... anon-rss:31175284kB" on a 30 GB host. At the 1 GiB pin the
same loop is flat: 3.98 GiB across 97 passes, unchanged to two decimals.

The difference is the pass rate, not the code. 1 GiB moves about 5 passes
a second and 16 MiB about 400, and whatever the runtime holds per transfer
is reclaimed fast enough at the former and not at the latter. Both
`resident.copy_(host)` and `host.to(device)` with an explicit `del` grow
identically, so this is not the in-place-write question the SUSPECT note
below asks about -- that note stands, and this is not it.

So a 16 MiB pin would OOM this suite's own 300 s and 3600 s runs, and the
pinned 1 GiB survives them (8,610 to 9,956 passes per hour-long leg on
2026-09-21, no failures). A pin that cannot survive the durations the
suite uses is not an improvement, however much better it measures in 10
seconds. The sweep above ships beside the pin so the cost of that choice
is visible: the Score is about 3.7 GB/s where the link can do about 8.9
one way.

(The whole curve is lower than 2026-09-10's, which read 11.55 at 16 MiB
against 8.89 here. Different instance, same shape; the conclusion rests
on the shape.)

STATUS: VERIFIED ON HARDWARE, both parts 2026-09-08 and trn1.2xlarge
2026-09-10, including the preallocation and alternating-source controls --
the size sweep above was taken through this kernel with both in place. No
NKI: this is torch tensor movement, so it depends on the runtime rather
than on a compiled kernel.
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

    host, pinned_source = host_buffer(SOURCE_FILL)
    # ALTERNATING SOURCES. If the runtime can tell that h2d copies the same
    # bytes every pass, it may serve the copy without moving them, which
    # would inflate h2d rather than depress d2h -- and the two are
    # indistinguishable from the ratio alone. Two sources with different
    # contents, alternating, remove that explanation.
    host_alt, _ = host_buffer(ALT_FILL)
    landing, pinned_landing = host_buffer(LANDING_FILL)

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

    try:
        landing_value = float(landing.reshape(-1)[0])
    except Exception:  # materialisation failed; leave unverified
        landing_value = None

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
        # What actually arrived, not just how fast. Read once at the end
        # rather than per pass, so the check costs nothing the measurement
        # would notice.
        "landing_value": landing_value,
        "warning": "; ".join(part for part in (
            verify_transfer_arrived(landing_value, plan["directions"]),
            verify_directions_are_balanced(per_direction,
                                           transfer_bytes=plan["bytes"]),
        ) if part) or None,
        "plan": plan,
    }


# The three buffer fills, named so the arrival check can derive what a
# completed transfer must leave behind rather than repeating a literal.
#
# They were literals, and nothing read any of them back: the kernel timed
# copies and never looked at what arrived. A d2h leg that moved no bytes
# would leave `landing` at its initial fill and report full bandwidth,
# because bandwidth here is bytes-requested over wall time and the bytes
# requested are a constant.
SOURCE_FILL = 1.0
ALT_FILL = 2.0
LANDING_FILL = 0.0


def verify_transfer_arrived(
    landing_value: typing.Optional[float],
    directions: typing.Sequence[str],
) -> typing.Optional[str]:
    """Check the host landing buffer holds bytes the device sent.

    ``landing`` starts at ``LANDING_FILL`` and the two device-side sources
    are ``SOURCE_FILL`` and ``ALT_FILL``, so after a d2h leg it must hold
    one of the latter. Still holding its own initial fill means nothing
    arrived -- and every figure in the row would be unchanged, because
    the Score counts bytes requested rather than bytes observed.

    Says nothing when d2h did not run: a plan of h2d alone leaves the
    landing buffer untouched by design, and accusing it then would be a
    check that fires on a correct run.
    """
    if "d2h" not in directions:
        return None
    if landing_value is None:
        return "the landing buffer could not be read back to verify"
    if landing_value != landing_value:
        return "the landing buffer is NaN"
    if landing_value == LANDING_FILL:
        return (
            f"the landing buffer still reads {LANDING_FILL:g} after the "
            "d2h leg, where a completed transfer leaves "
            f"{SOURCE_FILL:g} or {ALT_FILL:g} -- no bytes arrived, and the "
            "Score counts bytes requested rather than bytes observed"
        )
    if landing_value not in (SOURCE_FILL, ALT_FILL):
        return (
            f"the landing buffer reads {landing_value:g}, which is neither "
            f"source fill ({SOURCE_FILL:g} or {ALT_FILL:g}) -- the bytes "
            "that arrived are not the bytes that were sent"
        )
    return None


# Largest d2h transfer measured on the fast side of the staging cliff: the
# 2026-09-10 size sweep held ~2.9 GB/s through 16 MiB and 0.88 by 64 MiB.
D2H_FAST_UP_TO = 16 << 20


def verify_directions_are_balanced(
    per_direction: typing.Mapping[str, typing.Mapping[str, float]],
    ratio_floor: float = 0.25,
    transfer_bytes: typing.Optional[int] = None,
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
        # The explained case says so. With both legs preallocated -- which
        # run() now always does -- the harness-first advice below sent every
        # reader to re-check a cause fixed on 2026-09-08, on every run, while
        # the measured cause went unnamed.
        if (slow_name == "d2h" and isinstance(transfer_bytes, int)
                and transfer_bytes > D2H_FAST_UP_TO):
            return (
                f"d2h ran at {slowest:.1f} GB/s against {fastest:.1f} the "
                f"other way at {transfer_bytes / (1 << 20):.0f} MiB per "
                "transfer -- past the staging cliff a size sweep measured on "
                "trn1.2xlarge 2026-09-10, where d2h held ~2.9 GB/s up to 16 MiB "
                "and ~1 GB/s past 64 MiB: a transfer-size effect, not the link"
            )
        return (
            f"{slow_name} ran at {slowest:.1f} GB/s against {fastest:.1f} the "
            "other way -- before reading this as a link property, check that "
            "both legs reused a preallocated buffer (the 2026-09-08 split was "
            "a per-pass host allocation, not the link), then the negotiated "
            "link width"
        )
    return None
