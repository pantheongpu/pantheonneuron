# 2026-09-08 validation, both parts

`tools/validate_hardware.sh` on trn1.2xlarge and inf2.xlarge, same commit,
run in parallel. See the README's "What the 2026-09-08 rerun found" for what
these logs mean.

`instance-id` is scrubbed. `neuron-ls` prints it in its human-readable
header, which `docs/neuron_counters.md` records as one of the identifiers
this repository must not publish — the same invariant
`tests/test_report_privacy.py` enforces for reports applies to anything else
committed here. Home paths are rewritten to `~`.

The warning text quoted in `trn1.log` for `pcie_bandwidth` names a
preallocated buffer as the first thing to check. It was checked: both legs
were already preallocated in this run, which is what makes the asymmetry
interesting rather than explained.
