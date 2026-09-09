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
#
# Every single-device workload is here. Until 2026-09-08 this list held four
# names and 18 of 26 workloads had never touched hardware at all -- which is
# how omni_virus shipped a NameError that only its first real run could have
# found. A workload absent from this list is a workload nobody is checking.
#
# The two collectives are absent because they cannot run: they need 2+
# devices and trn1.32xlarge needs 128 vCPUs against a granted 64. They skip
# themselves at runtime, so including them would print a skip rather than
# tell us anything.
#
# Each name is timed out and failures are recorded rather than fatal: a
# workload that fails is a result, and it must not cost the run every
# workload after it.
# ---------------------------------------------------------------------------
ORCHESTRATED=${ORCHESTRATED:-"
  memory_read memory_write pcie_bandwidth allocation_fragmentation
  tensor_virus int_virus pulse_virus transformer_virus omni_virus
  graph_replay
  llm_prefill llm_decode kv_cache_churn
  fused_attention quantized_gemm moe_router speculative_decode serving_mix
  rag_embedding vision_encoder
  transformer_train_step
  memory_read_agg memory_write_agg
"}

# Per-workload ceiling. The pinned 8192^3 compiles in about 80s now that the
# accumulation loop is rolled, but the transformer family has never compiled
# at all and its graphs are deeper, so this is generous on purpose. It is a
# ceiling, not a budget: a workload that hits it is telling us something.
WORKLOAD_TIMEOUT=${WORKLOAD_TIMEOUT:-2400}

for workload in $ORCHESTRATED; do
  hr "orchestrated: $workload (pinned problem)"
  timeout "$WORKLOAD_TIMEOUT" $PY pantheon_neuron.py \
      --test "$workload" --duration "$DURATION" 2>&1 \
    | grep -vE 'CCOM WARN|nccl_net_ofi|OFI plugin|neuronpjrt.cc' \
    | tail -6
  status=${PIPESTATUS[0]}
  if [ "$status" -eq 124 ]; then
    echo "[VALIDATE] $workload TIMED OUT after ${WORKLOAD_TIMEOUT}s"
  elif [ "$status" -ne 0 ]; then
    echo "[VALIDATE] $workload exited $status"
  fi
done

# ---------------------------------------------------------------------------
# The NEFF search, against a cache that is no longer empty.
#
# The 2026-09-08 run scored from neuron-profile on the first candidate, from
# a candidate list of one: a fresh instance with PANTHEON_NEURON_WORKDIR set
# holds exactly one NEFF, so the ranking had nothing to rank and the search
# -- the actual fix -- went unexercised. Every workload above has now
# compiled into the same tree, so running memory_read again here asks the
# question the first pass could not: with many graphs to choose between,
# does the plan check still find ours?
# ---------------------------------------------------------------------------
hr "NEFF search against a warm compile cache"
echo "NEFFs now on this machine:"
find "$PANTHEON_NEURON_WORKDIR" /tmp/no-user/neuroncc_compile_workdir \
     /var/tmp/neuron-compile-cache -name '*.neff' 2>/dev/null | wc -l
timeout "$WORKLOAD_TIMEOUT" $PY pantheon_neuron.py \
    --test memory_read --duration "$DURATION" 2>&1 \
  | grep -vE 'CCOM WARN|nccl_net_ofi|OFI plugin|neuronpjrt.cc' | tail -4

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
                 key=os.path.getmtime)
for path in reports:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    for row in payload.get("test_results", []):
        if row["Test Name"] not in ("memory_read", "memory_write"):
            continue
        print(f"  {row['Test Name']:14} via {row.get('Score Method')}")
        telemetry = row.get("Telemetry") or {}
        idle = telemetry.get("execution_idle_fraction")
        if idle is not None and idle > 0.1:
            print(f"    {idle:.0%} of the observed window was not executing "
                  f"(compile), span {telemetry.get('execution_span_s')}s")
        measured = row.get("Measurement") or {}
        tried = measured.get("profiler_candidates_tried")
        if tried is not None:
            print(f"    candidate {tried} of "
                  f"{measured.get('profiler_candidates_available')}, "
                  f"coverage {measured.get('profiler_plan_coverage')}, "
                  f"graph {measured.get('profiler_neff')}")
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

# Every report this run wrote, not the last four: the orchestrated list is
# now the whole registry, so a window that truncates hides workloads.
reports = sorted(glob.glob("database/pantheon_neuron_report_*.json"),
                 key=os.path.getmtime)
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
passed = [r for r in seen.values() if r["Status"] == "PASS"]
failed = [r for r in seen.values() if r["Status"] == "FAIL"]
print(f"\n  Workloads run: {len(seen)}  "
      f"PASS {len(passed)}  FAIL {len(failed)}  "
      f"SKIP {len(seen) - len(passed) - len(failed)}")
print(f"  Scores from a declared hardware source: {len(declared)} of {len(seen)}")
if failed:
    print("\n  Failures:")
    for r in failed:
        print(f"    {r['Test Name']:26} {(r.get('Detail') or '')[:110]}")
PYEOF

hr "done"
