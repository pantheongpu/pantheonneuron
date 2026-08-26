#!/usr/bin/env bash
# Enumerate every performance counter a Neuron device actually exposes.
#
# Run on a trn1/trn2/inf2 instance with the Neuron driver installed (the
# Deep Learning AMI Neuron has it preinstalled).  Writes a JSON summary to
# stdout and a human-readable log to stderr.
#
#   ./probe_counters.sh > neuron_counters.json
#
# The point of this script is to answer, on real hardware, what the
# documented interfaces do NOT tell us: whether device temperature, power
# draw, clocks or throttle state are reachable through sysfs, neuron-ls or
# undocumented neuron-monitor fields.  CloudWatch's published Neuron metric
# set has none of them.
set -uo pipefail

NEURON_BIN=/opt/aws/neuron/bin
log() { echo "[probe] $*" >&2; }

log "=== neuron-ls ==="
"$NEURON_BIN/neuron-ls" --json-output 2>/dev/null | tee /tmp/neuron_ls.json >&2 \
  || log "neuron-ls unavailable"

log "=== driver / runtime versions ==="
cat /proc/version >&2
modinfo neuron 2>/dev/null | grep -E '^(version|srcversion):' >&2 || log "neuron module info unavailable"
apt list --installed 2>/dev/null | grep -i neuron >&2 || true

log "=== sysfs tree (this is where temp/power would live, if anywhere) ==="
SYSFS=/sys/devices/virtual/neuron_device
if [ -d "$SYSFS" ]; then
  find "$SYSFS" -maxdepth 3 -type f 2>/dev/null | head -200 >&2
  log "--- readable scalar values ---"
  find "$SYSFS" -maxdepth 3 -type f 2>/dev/null | while read -r f; do
    v=$(head -c 120 "$f" 2>/dev/null | tr -d '\n')
    [ -n "$v" ] && echo "  ${f#$SYSFS/} = $v" >&2
  done
else
  log "no $SYSFS -- driver may expose counters elsewhere"
  find /sys -maxdepth 4 -iname '*neuron*' 2>/dev/null | head -40 >&2
fi

log "=== hwmon (standard Linux temp/power interface) ==="
for h in /sys/class/hwmon/hwmon*; do
  [ -e "$h/name" ] || continue
  echo "  $h -> $(cat "$h/name" 2>/dev/null)" >&2
  ls "$h" 2>/dev/null | grep -E '^(temp|power|curr|in)[0-9]+_' | head -10 | sed 's/^/      /' >&2
done

log "=== full neuron-monitor sample (all documented metric groups) ==="
CFG=$(mktemp)
cat > "$CFG" <<'JSONEOF'
{
  "period": "1s",
  "neuron_runtimes": [
    {"tag_filter": ".*",
     "metrics": [
       {"type": "neuroncore_counters"},
       {"type": "memory_used"},
       {"type": "neuron_runtime_vcpu_usage"},
       {"type": "execution_stats"}
     ]}
  ],
  "system_metrics": [
    {"type": "vcpu_usage"},
    {"type": "memory_info"},
    {"period": "2s", "type": "neuron_hw_counters"}
  ]
}
JSONEOF

timeout 8 "$NEURON_BIN/neuron-monitor" -c "$CFG" 2>/dev/null | head -3 > /tmp/nm_raw.json
log "captured $(wc -l < /tmp/nm_raw.json) sample(s)"

# Emit every leaf key path so we can see the true schema, not the doc's.
python3 - <<'PYEOF'
import json, sys

paths = set()
def walk(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            walk(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        for item in node[:1]:
            walk(item, f"{prefix}[]")
    else:
        paths.add(f"{prefix} = {type(node).__name__}")

samples = []
try:
    for line in open("/tmp/nm_raw.json"):
        line = line.strip()
        if line:
            samples.append(json.loads(line))
except Exception as error:
    print(f"[probe] could not parse neuron-monitor output: {error}", file=sys.stderr)

for sample in samples:
    walk(sample)

interesting = [p for p in sorted(paths)
               if any(k in p.lower() for k in
                      ("temp", "power", "watt", "clock", "freq", "throttle", "volt", "fan"))]

print(json.dumps({
    "neuron_monitor_key_paths": sorted(paths),
    "thermal_power_related_paths": interesting,
    "sample_count": len(samples),
}, indent=2))

print(f"[probe] {len(paths)} distinct key paths", file=sys.stderr)
print(f"[probe] thermal/power/clock related: {interesting or 'NONE FOUND'}", file=sys.stderr)
PYEOF
rm -f "$CFG"
