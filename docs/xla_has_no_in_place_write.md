# XLA has no in-place write, and this suite assumed it twice

A kernel that mutates a resident device tensor does not do what the source
says. XLA is functional: an assignment into a slice lowers to a
dynamic-update-slice, which **produces a new tensor**. There is no in-place
write to lower it to.

That is measured, not inferred — see "The evidence" below. This note exists
because the same assumption is baked into more than one kernel, and because
its symptoms look like three different problems depending on where you meet
them.

## The evidence

`kv_cache_churn`, trn1.2xlarge, 2026-09-08. Three attempts at "append to a
resident KV cache", each of which first looked like something else:

| attempt | measured | first diagnosis | actual |
|---|---|---|---|
| 1 layer, 1 token, `index_copy_` | 115 µs for 16 KiB — 1,794× what HBM needs | dispatch overhead | wrong |
| 32 layers, 64 tokens, `index_copy_` | 455 ms for 32 MiB — 54× a full-slice rewrite | a slow scatter | wrong |
| 8 static ring slots, slice assign | **~7 minutes to compile each slot's graph** | — | the graph handles the whole 2 GiB cache |

The third is the one that settles it. A graph that takes seven minutes to
compile a 256 MiB slice write is not compiling a slice write. Once the graph
is understood to carry the entire 2 GiB tensor, the first two follow from it,
and both earlier diagnoses were wrong.

**Consequence:** the cost of appending to a KV cache is proportional to the
size of the cache, not to the number of tokens appended. For anyone serving
on this hardware that is a larger finding than the workload it came from.

### Counted, 2026-09-11

The evidence above is timing and compile time. neuron-profile counts HBM
bytes per execution directly. For `kv_cache_churn`'s eight ring-slot write
graphs at the pinned problem (134.22 MB of K and V resident), trn1.2xlarge:

| per slot write | bytes | × resident |
|---|--:|--:|
| read | 268.47 MB | 2.00 |
| written | 247.46 MB | 1.84 |

That confirms the whole-cache copy, and shows the copy happens **twice**.
The workload had charged one copy (2 × resident, read plus write), so its
`cache_gbps` reported 102 GB/s for 196 GB/s of actual traffic. Why twice is
not measured. A plausible reading is torch-xla copying the functional
result back into the mutated tensor's buffer. Appending 16.8 MB of K and
V moves 516 MB.

## Where else the assumption is made

### `pcie_bandwidth` — suspected, unverified

The h2d leg reads:

```python
resident.copy_(host if passes % 2 == 0 else host_alt)
```

with a comment claiming this avoids the per-pass allocation that
`resident = host.to(device)` would incur. **That claim rests on in-place
semantics that do not exist.** If `copy_` into an XLA tensor also produces a
new tensor, the leg allocates 1 GiB per pass regardless and the fix changed
nothing.

The measurement is consistent with the fix having done nothing:

| | before | after |
|---|--:|--:|
| d2h | 1.0 GB/s | 1.1 GB/s |
| h2d | 6.0 GB/s | 6.4 GB/s |

That is the same shape as `kv_cache_churn`'s 0.1029 → 0.0740: a change
justified by in-place reasoning that moved the number by noise or the wrong
way. It does **not** prove the d2h asymmetry is explained — only that one
more explanation for it has been ruled unproven, and that the workload's
comments assert something about the runtime that this suite has since
measured to be false.

Not fixed here. Fixing it needs a hardware run, and shipping a third
in-place-flavoured guess without one is how the last two went.

### Ruled out

Every other kernel was checked for mutation of a device tensor inside a
timed loop. `pcie_bandwidth` is the only other one. `inference_mix`'s
scattered writes (`host_states[positions, ...] = 1.0`) are on **host**
tensors built before the clock starts, and `memory_agg`, `collectives` and
`nki_backend` only mutate Python dicts.

## What to do about it in a new kernel

1. **Never assume a write lands in place.** If a kernel's cost model says
   "this step moves N bytes" and the tensor being written is much larger
   than N, the step probably moves the tensor.
2. **Suspect the compile time.** A graph whose compile time scales with a
   buffer it should only be slicing is handling the whole buffer. That was
   the signal that broke this open, and it costs nothing to look at.
3. **Prefer a cost model the kernel can report.** `cache_plan` now returns
   `slice_bytes` alongside `bytes_per_step` precisely so the two can
   disagree visibly, rather than the smaller one being quoted.
4. **A rate cannot audit itself.** `cache-updates/s` looks identical whether
   an update moved a cache or a register. Whatever a workload claims to
   measure, it should also report the derived quantity that would be absurd
   if the claim were false — bandwidth, here.

## Addendum: kernels contaminate each other in one process

A diagnostic script that ran five kernels in sequence, in one process,
measured `allocation_fragmentation` at **0.9 allocation-events/s**. The same
kernel in a clean process measures **1,371.8/s** — a factor of 1,500.

Nothing was wrong with the kernel. The kernels before it had left device
memory allocated, and the allocator was working against a nearly-full
NeuronCore.

`validate_hardware.sh` is unaffected: it invokes `pantheon_neuron.py --test
<name>` per workload, so each gets a fresh process. But anyone writing a
one-off diagnostic should run one kernel per process, or measure something
that is not sensitive to how much HBM is already spoken for.

The near-miss is worth recording: the slow number appeared immediately after
a change to that kernel's eviction accounting, and the obvious inference was
that the change caused it. It did not. Running the old accounting and the new
one in clean processes is what separated them.
