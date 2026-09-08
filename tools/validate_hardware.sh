#!/usr/bin/env bash
# Validate everything that has never run on a Neuron device, in one pass.
#
# Capacity for these parts is scarce and billed by the hour, so this exists
# to make a window cheap: one command, no rediscovery. The 2026-09-07
# session spent roughly a quarter of its instance time working out that the
# DLAMI venv's bin must be on PATH before torch_neuronx can import at all,
# which is encoded below rather than learned again.
#
# Run it ON the instance, from the repo root:
#
#     bash tools/validate_hardware.sh 2>&1 | tee validation.log
#
# Exit status is 0 if it completed, regardless of what it found -- the
# findings are the output, and a workload that fails is a result.
set -uo pipefail

VENV=${PANTHEON_NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_2_8}
PY=$VENV/bin/python
# The venv's bin, not just its python: torch_neuronx shells out to
# libneuronpjrt-path, which lives there, and fails at import without it.
export PATH="$VENV/bin:/opt/aws/neuron/bin:$PATH"
export PANTHEON_NEURON_WORKDIR=${PANTHEON_NEURON_WORKDIR:-/tmp/pantheon_ccwork}
DURATION=${DURATION:-30}

# The GEMM workloads pin 8192^3, which unrolls to 65,536 matmul calls and
# has never compiled -- four times the graph that already took about seven
# minutes. They are exercised at reduced shapes through the kernel instead,
# so this script never blocks on an unbounded compile.
GEMM_SHAPE=${GEMM_SHAPE:-2048}

hr() { printf '\n========== %s ==========\n' "$*"; }

hr "part"
neuron-ls 2>/dev/null | sed -n '1,10p'
cat /sys/devices/virtual/neuron_device/neuron0/info/architecture/arch_type 2>/dev/null
cat /sys/devices/virtual/neuron_device/neuron0/info/architecture/device_name 2>/dev/null
$PY -c "import neuronxcc, torch; print('neuronxcc', neuronxcc.__version__, '| torch', torch.__version__)" 2>&1 | tail -1

# ---------------------------------------------------------------------------
# Orchestrated runs. These use the registry's pinned problems and go through
# the whole reporting path, which is the only way to exercise the Score
# sources: the profiler needs the reserved core the orchestrator sets up,
# and the monitor Score is read after the monitor stops.
# ---------------------------------------------------------------------------
for workload in memory_read memory_write pcie_bandwidth allocation_fragmentation; do
  hr "orchestrated: $workload (pinned problem)"
  timeout 2400 $PY pantheon_neuron.py --test "$workload" --duration "$DURATION" 2>&1 \
    | grep -vE 'CCOM WARN|nccl_net_ofi|OFI plugin|neuronpjrt.cc' \
    | tail -6
done

# ---------------------------------------------------------------------------
# Reduced-shape kernel runs. Direct calls, so no Score is produced and none
# should be read as one -- this asks only whether the kernel computes what
# it claims at a shape that compiles.
# ---------------------------------------------------------------------------
hr "reduced shape: the GEMM family (${GEMM_SHAPE}^3)"
timeout 1800 $PY - <<PYEOF 2>&1 | grep -E '^(tensor_virus|int_virus|pulse_virus|  )'
from kernels import tensor_virus, pulse_virus

shape = [$GEMM_SHAPE, $GEMM_SHAPE, $GEMM_SHAPE]

for name, module, problem in (
    ("tensor_virus", tensor_virus,
     {"op": "matmul", "shape": shape, "dtype": "bf16"}),
    # uint8, not int8: trn1's Tensor Engine rejects signed int8 outright
    # ("nc_matmul does not support stationary.dtype=int8", 2026-09-08), and
    # the registry now pins the dtype the part will actually run.
    ("int_virus", tensor_virus,
     {"op": "matmul", "shape": shape, "dtype": "uint8"}),
    ("pulse_virus", pulse_virus,
     {"op": "matmul", "shape": shape, "dtype": "bf16",
      "duty_cycle": 0.5, "period_s": 2}),
):
    try:
        r = module.run(problem, duration=$DURATION)
        unit = r.get("analytic_unit", "TFLOPS")
        print(f"{name}: OK")
        print(f"  analytic          {r['analytic_tflops']:.2f} {unit}")
        if "loaded_tflops" in r:
            print(f"  loaded-only       {r['loaded_tflops']:.2f} {unit}")
            print(f"  pulses            {r.get('pulses')}")
        print(f"  passes            {r['passes']}")
        print(f"  product verified  {r.get('product_verified_ratio')}")
        print(f"  warning           {r.get('warning')}")
    except Exception as exc:
        print(f"{name}: FAIL {type(exc).__name__}: {str(exc)[:200]}")
PYEOF

# ---------------------------------------------------------------------------
# The NEFF search, which is the whole reason a profiler Score might land this
# time. Ranking by mtime picked wrong on both parts: inf2 captured a graph
# that moved 2 bytes against an 8 GiB plan, trn1 one that moved 4. The plan
# check is now the selector, so a wrong first guess costs another capture
# rather than the Score. This reports how hard it had to look.
# ---------------------------------------------------------------------------
hr "NEFF selection: how many candidates the search needed"
$PY - <<'PYEOF'
import glob, json, os

reports = sorted(glob.glob("database/pantheon_neuron_report_*.json"),
                 key=os.path.getmtime)[-6:]
for path in reports:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    for row in payload.get("test_results", []):
        if row["Test Name"] not in ("memory_read", "memory_write"):
            continue
        print(f"  {row['Test Name']:14} via {row.get('Score Method')}")
        detail = (row.get("Detail") or "").strip()
        if detail:
            print(f"    {detail[:200]}")
print("  (a run that needed a late candidate means mtime ranking is weak;")
print("   raise PANTHEON_NEURON_NEFF_CANDIDATES if a search reports exhaustion)")
PYEOF

# ---------------------------------------------------------------------------
# What the run actually proved. The Score Method field is the point: it
# records the source that produced each number, so "neuron-profile" here is
# the first time the declared bandwidth source has ever fired, and
# "neuron-monitor" the first time the compute one has.
# ---------------------------------------------------------------------------
hr "summary: which Score sources fired"
$PY - <<'PYEOF'
import glob, json, os

reports = sorted(glob.glob("database/pantheon_neuron_report_*.json"),
                 key=os.path.getmtime)[-4:]
if not reports:
    print("  no reports written")
seen = {}
for path in reports:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    for row in payload.get("test_results", []):
        seen[row["Test Name"]] = row

for name, row in sorted(seen.items()):
    score = row.get("Score")
    print(f"  {name:26} {row['Status']:8} "
          f"{score if score is not None else '-':>14} {row.get('Unit') or '':14} "
          f"via {row.get('Score Method')}")
    detail = (row.get("Detail") or "").strip()
    if detail:
        print(f"    {detail[:150]}")

declared = [r for r in seen.values()
            if r.get("Score Method") in ("neuron-profile", "neuron-monitor")]
print(f"\n  Scores from a declared hardware source: {len(declared)} of {len(seen)}")
PYEOF

hr "done"
