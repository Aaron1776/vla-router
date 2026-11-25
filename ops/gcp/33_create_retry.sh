#!/usr/bin/env bash
# Create the Spot L4 VM despite stockouts (the user, 2026-11-25): try us-central1-a, -b, -c in turn every 10 minutes
# for up to 6 hours; no other region. Once a zone starts the VM:
#   1. env.sh's ZONE is pointed at that zone (every Mac script that addresses the VM reads it);
#   2. the dry run of the spend guard's stop, against that VM, before any held step is queued: the instance resolves
#      in that zone (the Mac's stop command `gcloud compute instances stop $VM_NAME --zone=<zone>`), it is a Spot VM
#      that stops on preemption, the guard daemon on it is ticking in the ledger at the re-based price (it stops the
#      VM with its local `shutdown_command`), and `spend check --phase <phase>` passes;
#   3. only then the held step scripts (arguments) are copied into the bucket queue. The runner on the VM works
#      through 010-090 first (~1 h), so a held step queued within that time runs after them; a failed dry run queues
#      nothing and the runner stops the VM after 090 (queue empty).
# Then it watches (log only, no restarts): VM status, the runner's STATE.json and the ledger every 5 minutes, until the
# VM stops or 20 hours pass. Run detached so it outlives the session that started it:
#   nohup caffeinate -i -s bash ops/gcp/33_create_retry.sh --phase phase1a ops/vm/postfreeze/720-phase1a.sh \
#     >> outputs/l4-retry/retry.log 2>&1 &
# Exit: 0 VM created and held steps queued (then watched); 2 an error other than a stockout (e.g. sign-in): nothing
# created, the user decides; 3 the dry run failed (VM created, nothing queued); 4 no capacity in 6 hours.
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
source ops/gcp/env.sh
: "${IMAGES_COMMIT:=03f311c4cd18a392285709987426bdfc42413fcd}"
PHASE=""
if [[ "${1:-}" == "--phase" ]]; then PHASE="$2"; shift 2; fi
HELD=("$@")
for f in "${HELD[@]}"; do [[ -f "${f}" && "${f}" == *.sh ]] || { echo "not a step script: ${f}"; exit 1; }; done
ZONES=(us-central1-a us-central1-b us-central1-c)
TRY_S="${RETRY_TRY_S:-600}"
DEADLINE=$(( $(date +%s) + ${RETRY_HOURS:-6} * 3600 ))
PRICE="$(sed -n 's/^  vm_hourly_usd: *\([0-9.]*\).*/\1/p' ops/ops.example.yaml)"
log() { echo "$(date -u +%FT%TZ) $*"; }
mask() { sed -e "s/${VM_NAME}/<VM>/g" -e "s/${PROJECT_ID}/<PROJECT>/g" -e "s#${BUCKET}#<BUCKET>#g"; }

dry_run() {   # dry_run <zone> <unix time the create started>
  local z="$1" t0="$2" st i
  st="$(gcloud compute instances describe "${VM_NAME}" --project="${PROJECT_ID}" --zone="${z}" \
        --format='value(status,zone.basename(),scheduling.provisioningModel,scheduling.instanceTerminationAction)' 2>&1)"
  log "dry run 1/4: the Mac's stop command 'gcloud compute instances stop <VM> --zone=${z}' resolves to: ${st}" | mask
  [[ "${st}" == *"${z}"*SPOT*STOP* ]] || { log "dry run FAILED: the VM does not resolve as a Spot VM in ${z}"; return 1; }
  log "dry run 2/4: the guard's stop on the VM: $(grep '^shutdown_command' ops/ops.example.yaml) (host source $(gcloud storage cat "gs://${BUCKET}/vla-router/phase0/src/CURRENT" 2>/dev/null | cut -c1-7))"
  for i in $(seq 1 30); do                    # the guard starts first at boot; give the boot 15 minutes
    if "${REPO}/.venv-ops/bin/python" - "${t0}" "${PRICE}" <<'PY'
import json, os, sys, time
from google.cloud import storage
t0, price = float(sys.argv[1]), float(sys.argv[2])
b = storage.Client(project=os.environ["PROJECT_ID"]).bucket(os.environ["BUCKET"])
led = json.loads(b.blob("vla-router/ops/ledger.json").download_as_text())
s = [x for x in led["sessions"] if x["started_at"] >= t0 - 600 and x.get("last_tick") and time.time() - x["last_tick"] < 300]
print(json.dumps({"guard_sessions_since_create": len(s), "hourly_usd": [x["hourly_usd"] for x in s]}))
sys.exit(0 if s and all(abs(x["hourly_usd"] - price) < 1e-9 for x in s) else 1)
PY
    then
      log "dry run 3/4: the guard daemon on this VM is ticking in the ledger at ${PRICE} USD/h"
      break
    fi
    (( i == 30 )) && { log "dry run FAILED: no ticking guard session from this boot after 15 minutes"; return 1; }
    sleep 30
  done
  if [[ -n "${PHASE}" ]]; then
    local out rc
    out="$("${REPO}/.venv-ops/bin/vlar-ops" spend check --phase "${PHASE}" --ops-config ops/ops.yaml 2>&1 | tail -n 1)"; rc=$?
    log "dry run 4/4: spend check --phase ${PHASE}: ${out}"
    (( rc == 0 )) || { log "dry run FAILED: the spend check would stop ${PHASE}"; return 1; }
  fi
  return 0
}

watch_vm() {   # watch_vm <zone>: log only, every 5 minutes, until the VM stops or 20 hours pass
  local z="$1" end=$(( $(date +%s) + 20 * 3600 )) st state
  while (( $(date +%s) < end )); do
    st="$(gcloud compute instances describe "${VM_NAME}" --project="${PROJECT_ID}" --zone="${z}" --format='value(status)' 2>&1 | tail -n 1)"
    state="$(gcloud storage cat "gs://${BUCKET}/vla-router/phase0/STATE.json" 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("state"), d.get("step"), str(d.get("reason",""))[:160])' 2>/dev/null)"
    log "watch: VM ${st}; runner: ${state}; ledger: $("${REPO}/.venv-ops/bin/vlar-ops" spend status --ops-config ops/ops.yaml 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); p=d.get("phase") or {}; print("total", d["total_usd"], "phase", p.get("name"), round(p.get("vm_usd",0),3), "/ stop", p.get("stop_usd"))' 2>/dev/null)" | mask
    [[ "${st}" == TERMINATED || "${st}" == STOPPED ]] && { log "watch: the VM has stopped; watcher ends"; return 0; }
    sleep 300
  done
  log "watch: 20 hours passed; watcher ends"
}

log "retry: zones ${ZONES[*]} every ${TRY_S} s until $(date -u -r "${DEADLINE}" +%FT%TZ); held: ${HELD[*]:-none}; phase ${PHASE:-none}"
while (( $(date +%s) < DEADLINE )); do
  for z in "${ZONES[@]}"; do
    t0="$(date +%s)"
    out="$(ZONE="${z}" IMAGES_COMMIT="${IMAGES_COMMIT}" ops/gcp/30_create_vm.sh 2>&1)"; rc=$?
    if (( rc != 0 )) && ! grep -q "already exists" <<<"${out}"; then
      if grep -qiE "stockout|does not have enough resources|ZONE_RESOURCE_POOL_EXHAUSTED" <<<"${out}"; then
        log "${z}: stockout"; continue
      fi
      log "${z}: create failed (not a stockout), stopping: $(tail -n 3 <<<"${out}" | tr '\n' ' ')" | mask
      exit 2
    fi
    log "${z}: VM created"
    sed -i '' "s/^export ZONE=\"[a-z0-9-]*\"/export ZONE=\"${z}\"/" ops/gcp/env.sh
    log "env.sh now points at $(sed -n 's/^export ZONE="\(.*\)"/\1/p' ops/gcp/env.sh)"
    if ! dry_run "${z}" "${t0}"; then
      log "held steps NOT queued; the runner stops the VM after 090 (queue empty). The user decides."
      watch_vm "${z}"; exit 3
    fi
    for f in "${HELD[@]}"; do
      gcloud storage cp -q "${f}" "gs://${BUCKET}/vla-router/phase0/queue/$(basename "${f}")" && log "queued $(basename "${f}")"
    done
    watch_vm "${z}"; exit 0
  done
  (( $(date +%s) + TRY_S < DEADLINE )) || break
  sleep "${TRY_S}"
done
log "no capacity in ${ZONES[*]} for ${RETRY_HOURS:-6} hours: stopped retrying; the user decides (no other region)"
exit 4
