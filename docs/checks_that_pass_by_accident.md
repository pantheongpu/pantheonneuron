# Checks that pass by accident

A failing check is a good day. It says what is wrong and where.

A check that passes for a reason unrelated to what it asserts is worse
than no check at all, because it also occupies the space where a real one
would go. This repo has now produced seventeen of them, and they are collected
here because they rhyme — the same three or four shapes keep recurring,
and knowing the shapes is the only defence.

## The shapes

### 1. The check reads prose, and the prose quotes the code

A textual assertion about a kernel — that a loop is not unrolled, that a
buffer is not reallocated per pass — matched the very comment explaining
the defect it was written to catch. Comments quote the code they warn
about, so `assert "expert_weights[chosen]" not in source` is satisfied by
deleting the code and satisfied equally by nothing at all.

Happened **twice** before `tests/sourcecheck.py` existed to strip comments
and docstrings first.

### 2. The filter that made the checks trustworthy had its own bugs

Two, and the second was caused by fixing the first.

`code_only` treated `tokenize.NL` as a statement boundary, so every dict
key on a continuation line looked like a docstring and vanished:

```
{"key": 1,        ->    { : 1 ,
 "other": 2}             : 2 }
```

Every check written against a dict key was affected. An `in` assertion
could not pass; a `not in` assertion passed for entirely the wrong reason.
Found when an assertion about `"implied_tflops"` failed against source
that demonstrably contained it.

Removing `NL` fixed that and broke module docstrings preceded by a shebang
or licence comment: those follow `COMMENT` then `NL`, so with `NL` no
longer a separator they stopped looking like the first statement of
anything and survived into the filtered output. An ordering assertion over
`tools/compare_matmul_paths.py` then matched text inside its docstring.

The question was never "what was the previous token" but "have we seen a
real token since the last statement boundary".

### 3. The assertion cannot fail

```python
assert "not comparable across pins" in source.lower() or True
```

Written while adding a *different* check about accidental passes.

The fix was not to remove the `or True` but to stop asserting against a
comment at all: `registry.SCORE_DEPENDS_ON_PIN` is a dict a test can read,
and a dict cannot be satisfied by prose.

### 4. The threshold is a constant that a legitimate change flips

A regression test asserted that an old accounting bug drifted by more than
10 MiB. Repinning `allocation_fragmentation` from 10,000 allocations to
40,000 made the drift 7.3 MiB — still real, still a bug, and now
undetected.

A threshold that a change to the pinned problem can flip is not measuring
what it claims. It is now a fraction of the live budget.

### 5. The ordering assertion orders the wrong two things

`assert source.index(A) < source.index(B)` over raw file text is satisfied
by the first *mention* of A — a line of prose above the code makes it pass
while saying nothing about the code.

Routed through the filter, the same assertion then matched `ratio =`
against a *different function's* local variable rather than the division
it was written about. An ordering assertion is only as good as the two
things it orders, and one of these was the wrong statement in the wrong
place.

### 6. The check is keyed on something the target does not have

A test held every kernel's `STATUS:` line against a record of hardware
runs, keyed on which workloads a module owns. `transformer_ops` owns no
workload, so it was filtered out of the check and went on claiming
`UNTESTED ON HARDWARE` while every kernel importing it had passed.

**It was found by reading the list the check had discarded**, not by the
check.

### 7. The check matches a phrasing rather than a claim

The replacement asked every module for the literal string `STATUS:
UNTESTED ON HARDWARE`. Four modules saying `STATUS: verified ... but X is
UNTESTED` went straight past it, and all four were stale — two of them
understating a solved problem that had taken real work.

An exact-phrase match only catches the phrasing it was written against.
Three iterations of this one guard, three holes, each found by reading
what the previous version filtered out.

### 8. The measurement cannot disagree with itself

`serving_mix` reported cv 0.0001 over three repeats — by a wide margin the
steadiest Score in the suite, and read as evidence of an exceptionally
stable workload.

Its Score is an integer division: `decode_tokens // decode`, advancing
once per 32 decode steps and not at all in between. Three runs landed on
the same integer, and the only thing varying was the wall clock in the
denominator. Variation in the work done was below the resolution of the
number reporting it.

This is the complement of an unstable Score, and much harder to notice,
because it looks like the best row in the table.

The same defect had already been seen from the other side and misread.
An earlier status row recorded `serving_mix` as *"completes no decode
request at 20 s — 0.0312 requests/s"*, filed as a duration problem. It is
the same integer division: at 20 seconds the quotient had barely
advanced. One symptom looked like an unusable Score and the other like an
unusually good one, and they are the same arithmetic.

### 9. The comment describes a check that does not exist

```python
# The two disagreeing means the runtime accepted more replays than the
# device finished, which is worth seeing rather than smoothing.
```

Nothing reported the disagreement. `graph_replay` published one of two
numbers that differ by a factor of 4.2 and the reader never learned the
other existed.

A comment describing a check is not a check.

### 10. The check is correct, complete, tested — and unreachable

`tensor_virus.verify_against_monitor` compared the declared Score against
the analytic cross-check and would have caught a monitor reading of zero
against a kernel claiming throughput. Written, documented, and covered by
**five passing tests**.

Nothing called it. Not the kernel, not the orchestrator, not anything.

The five tests passing said the function was correct, and it was. They
said nothing whatever about whether it ran, and nothing else did either.
**Coverage of a function is not evidence that the function is reachable.**

Its job is now `pantheon_neuron.override_disagreement`, which is called,
covers every monitor-scored workload rather than one family, and carries
both zero cases.

**And there was a second one.** `profiler.verify_profile_covers_plan`
rejects a capture whose counters do not match the plan — the check that
produced *"profiled graph moved 4 bytes against a plan of 8589934592"*,
quoted in three docstrings, in the generated reference and in the README.
`select_by_plan` had grown an inline floor test and stopped calling it.
Five more tests, all green, all on unreachable code.

Being dead is not a neutral state. While nothing exercised it, it kept a
**one-sided bound**: it refused a graph that moved too few of the planned
bytes and accepted one that moved ten times too many, which divides by a
real `total_time` and publishes a bandwidth an order of magnitude too
fast. The inline copy inherited the same gap. Nobody found it because
nobody ran it, and nobody ran it because the tests passed.

Both are now guarded by a sweep asserting every function in production
code is called from production code, read through the comment filter so a
docstring crediting a function does not count as calling it.

**The first version of that sweep was itself one of these.** It restricted
itself to names starting with `verify_`, which is the shape the two known
cases happened to have. Widening it to every production function found
four more — including `tiling.engine_accepts` and `tiling.refusal`, added
the same afternoon by the author of this document, in the commit that
added the entry above.

A check written to catch a class of defect, scoped to the instances that
had already been found, is a check that can only ever confirm what is
already known. There is now an explicit `CALLED_ONLY_BY_TESTS` allowlist
naming each legitimate exception with its reason, so the exceptions are
visible rather than implied by a prefix.

### 11. The call site exists, runs, and returns nothing

The fix for #10 wired `omni_virus.engine_activity` into the orchestrator.
It type-checked. It ran on every `omni_virus` row. It returned an empty
dict every time, because its four counters live in `neuron-profile`'s
108-counter set and `omni_virus` is scored from `neuron-monitor`, which
does not carry them.

The tests passed, because they asserted the call happened and nothing
asserted it produced anything. **A call site that reports nothing is
worse than no call site, because it looks answered** — the function is no
longer on the dead list, and the gap is now invisible instead of
enumerable.

Caught by asking which reader emits those counter names. One number held
against another, again.

The underlying fact was the better finding: the workload whose entire
premise is per-engine behaviour is scored by the reader that cannot see
any engine.

### 12. The check is coupled to the formatting, not the content

Two tests asserted `"cause unknown" in doc`. An edit rewrapped the
paragraph, the phrase landed across a line break, and both failed — for
no reason connected to what the document says.

The same coupling fails the other way just as easily: a phrase that
happens to survive a rewrap keeps a test green while the surrounding
sentence is rewritten to mean the opposite. Collapsing whitespace before
matching is what makes such a check about the prose.

Related, and the same root: an ordering assertion over raw source (#5),
and a status check matching an exact phrase (#7). All three test how
something is written rather than what it says.

### 13. The summary is over a wider scope than it claims

`validate_hardware.sh` ends by printing what the run proved. It built
that from `glob("database/*.json")` with no date filter, keeping the
newest row per workload name — so on an instance carrying reports from a
fortnight earlier it announced **"Workloads run: 26, PASS 26"** for a run
that ran 24.

Two of the extra three cannot run on a single-device part at all. The
summary credited the pass with workloads the hardware refuses.

Nothing about it looked wrong. The count was plausible, every row was a
real row from a real run, and the figure was copied into the README
before the discrepancy surfaced — in a test written for something else
entirely, which refused to record `all_reduce` as both passed and never
run.

An aggregate is a claim about a scope. This one never stated its scope,
so nothing could check it.

### 14. The advertised check has no recipe

`make lint` was in `.PHONY` and in `make help`, with no recipe under it.
`make` answers that by printing **"Nothing to be done for 'lint'"** and
**exiting 0** — so a CI step calling it, or a contributor running it
before opening a PR, saw success while nothing was checked.

Not a check that was weak, or scoped wrongly, or matched the wrong
string. One that did not exist and reported success anyway.

The first real lint run found entry 15 below, plus two unused imports.

### 15. The parameter set is empty, so the test skips

`test_committed_reports_are_clean` parametrised over `database/*.json`.
`database/` is in `.gitignore`, so a fresh checkout contains none, pytest
reports *"got empty parameter set"* and skips.

**The privacy guard on a public repository had never run in CI** — in the
job deliberately split out of the matrix precisely so a single green
check could not hide it.

A skip is not a failure and not a pass; it is the check declining to
answer, and a job of skips exits 0. It now generates a report with the
code under test rather than hoping to find one, and asserts it wrote
something before checking it.

Found by deleting some local reports for an unrelated reason and noticing
the skip count go up by one.

### 16. The correction was wrong and the original was right

`kernels/memory_read.py` quoted "the part's ~820 GB/s HBM" in prose for
weeks. Building a table of peak specs, I judged it unverified — correctly
— and then *replaced* it with 613 GB/s, derived by dividing the AWS
instance page's "9.8 TB/s" by 16 chips, on a suspicion that 820 was
really Inferentia2's number.

The architecture documentation gives both parts in identical words: two
NeuronCore-v2, 32 GiB HBM at **820 GiB/sec**, 190 TFLOPS. The prose was
right. The correction was wrong. And because the substituted ceiling was
smaller, it reported the bandwidth kernels at **83–88% of peak when they
reach about 60%** — an error in the flattering direction, published in a
table with a `source` field and a `verified` flag that made it look
checked.

Three things should have stopped it, and each was available:

- **The repo already knew.** Both chips report NeuronCore-v2, which it
  documents in two places. Same core, same generation — the per-chip
  figures being different was the surprising claim, not the boring one.
- **The arithmetic didn't reconcile.** 16 × 820 GiB/s is 14.1 TB/s, not
  9.8. I noticed that inf2 reconciled and trn1 didn't, and read it as
  evidence *for* the substitution rather than against the source I was
  substituting from.
- **Units.** 820 **GiB**/s is 880.5 GB/s decimal, and the Scores are
  decimal. Even the right figure was 7% wrong until converted.

The lesson is not "verify your numbers" — the flag said `verified:
False` and the console printed *"peak unverified"*, which is the process
working. It is that **an unverified number replaced by a derived one is
worse than the unverified number**, because a derivation carries an air
of work that a recalled figure does not.

### 17. The check passes because the collection is empty

Not yet caught in the wild here, and guarded against on the way in: a
coverage check over a directory, a sweep over "every INTERNAL workload",
a parse that finds no matches — all pass silently when they find nothing.
Every such test in this suite now asserts the collection is non-empty
before asserting anything about its contents.

## The defence

Nothing here was caught by a linter or by a careful reading. Every one was
caught by a **second quantity** — one number held against another that
could disagree with it:

- The filter bug: an assertion failing against source that contained the
  string.
- The stale statuses: a hardware record read next to a docstring.
- The quantised Score: a cv read next to the Score's own resolution.
- The tiling diagnosis: 4.7× less traffic buying 1.06×.

So the practice that finds these is the same one that finds kernel
defects, and for the same reason. **A number on its own cannot be wrong.**

The corollary is uncomfortable and worth stating plainly: a green test run
is evidence about the checks that exist, not about the code. Six of the
seventeen above were found by reading what a passing check had filtered out.
