#!/usr/bin/env bash
# Trainium probe -- resolves the four things Inferentia2 could not answer.
#
# Run on a trn1/trn1n/trn2 instance with the Neuron driver installed.
# Targets exactly the assumptions recorded as unverified in README.md:
#
#   1. _ARCH_TABLE row for trn1: NeuronCore version, cores/device, training
#   2. What info/architecture/instance_type returns on a Trainium part
#      (_arch_from_sysfs is only verified to return "Inf2")
#   3. Whether the training capability path actually works
#   4. Whether neuron-profile reports the same 108 counters as on inf2
set -uo pipefail
B=/opt/aws/neuron/bin
export PATH=$B:$PATH
log() { echo "[trn-probe] $*" >&2; }

echo "########## 1. ARCH DETECTION (resolves _ARCH_TABLE) ##########"
S=/sys/devices/virtual/neuron_device/neuron0
for f in core_count connected_devices \
         info/architecture/arch_type \
         info/architecture/device_name \
         info/architecture/instance_type; do
  printf "  %-40s = %s\n" "$f" "$(cat $S/$f 2>&1 | head -1)"
done
echo "  (serial_number deliberately not printed -- device identifier)"

echo
echo "########## 2. neuron-ls ##########"
$B/neuron-ls --json-output 2>&1 | head -40

echo
echo "########## 3. neuron_hardware_info (authoritative core/device versions) ##########"
timeout 8 $B/neuron-monitor 2>/dev/null | head -1 | python3 -c "
import json,sys
try:
    d=json.loads(sys.stdin.readline())
except Exception as e:
    print('  parse failed:', e); raise SystemExit
for k,v in sorted(d.get('neuron_hardware_info',{}).items()):
    print(f'  {k:36} = {v}')
"

echo
echo "########## 4. THERMAL / POWER (confirm inf2 findings hold) ##########"
echo "  hwmon:      $(for h in /sys/class/hwmon/hwmon*; do cat $h/name 2>/dev/null; done | tr '\n' ' ')"
echo "  thermal:    $(ls /sys/class/thermal/ 2>/dev/null | tr '\n' ' ')"
echo "  power/util: $(cat $S/stats/power/utilization 2>&1)"
echo "  grep for temp/volt/fan in neuron sysfs:"
find /sys -path '*neuron*' -type f 2>/dev/null \
  | grep -iE 'temp|volt|fan' | head -10 || echo "    (none)"

echo
echo "########## 5. TRAINING PATH (the capability that gates the suite) ##########"
V=$(ls -d /opt/aws_neuronx_venv_pytorch_2_8 2>/dev/null | head -1)
if [ -z "$V" ]; then V=$(ls -d /opt/aws_neuronx_venv_pytorch* 2>/dev/null | head -1); fi
export PATH=$V/bin:$PATH
cat > /tmp/train_probe.py <<'PYEOF'
import time, torch, torch.nn as nn
import torch_xla.core.xla_model as xm

dev = xm.xla_device()
print(f"[train] xla device: {dev}", flush=True)

model = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024)).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
x = torch.randn(16, 1024, device=dev)
y = torch.randn(16, 1024, device=dev)

t0 = time.time()
for step in range(20):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()          # <-- the thing Inferentia cannot do
    xm.optimizer_step(opt, barrier=True)
elapsed = time.time() - t0
print(f"[train] 20 steps in {elapsed:.2f}s -> {20/elapsed:.3f} train-steps/s", flush=True)
print(f"[train] final loss {loss.item():.6f}", flush=True)
PYEOF
timeout 900 $V/bin/python /tmp/train_probe.py 2>&1 | tail -12

echo
echo "########## 6. PROFILER COUNTER SET (compare against inf2's 108) ##########"
export HOME=${HOME:-/root}
cat > /tmp/mkneff.py <<'PYEOF'
import torch, torch.nn as nn, torch_neuronx
class B(nn.Module):
    def __init__(s, n=1024):
        super().__init__(); s.a=nn.Linear(n,n,bias=False); s.b=nn.Linear(n,n,bias=False)
    def forward(s,x):
        for _ in range(4): x=torch.relu(s.b(s.a(x)))
        return x
torch_neuronx.trace(B().eval(), torch.randn(4,1024), compiler_workdir='/tmp/ccwork')
print("[ok] traced")
PYEOF
timeout 900 $V/bin/python /tmp/mkneff.py 2>&1 | tail -3
NEFF=$(find /tmp/ccwork -name '*.neff' 2>/dev/null | head -1)
echo "  NEFF=$NEFF"
if [ -n "$NEFF" ]; then
  timeout 300 $B/neuron-profile capture -n "$NEFF" -s /tmp/capture.ntff 2>&1 | grep -v 'level=info' | tail -3
  timeout 300 $B/neuron-profile view -n "$NEFF" -s /tmp/capture.ntff \
    --output-format summary-json --json-pretty-print 2>&1 | grep -v 'level=' > /tmp/summary.json
  python3 - <<'PYEOF'
import json
raw=open('/tmp/summary.json').read(); i=raw.find('{')
d=json.loads(raw[i:])
def flat(n,p=""):
    o={}
    if isinstance(n,dict):
        for k,v in n.items(): o.update(flat(v,f"{p}.{k}" if p else k))
    elif isinstance(n,list):
        if n: o.update(flat(n[0],p+"[]"))
    else: o[p]=n
    return o
f=flat(d)
keys={k.split('.')[-1] for k in f}
print(f"  TOTAL COUNTERS: {len(f)}  (inf2 reported 108)")
for grp,pat in (("throttle","throttle"),("cycle","cycle"),("mfu/flops","flop"),("collectives","cc_")):
    hits=sorted(k for k in keys if pat in k.lower())
    print(f"  {grp:12} ({len(hits)}): {', '.join(hits[:6])}")
PYEOF
fi
echo
log "probe complete"
