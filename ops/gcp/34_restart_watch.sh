#!/usr/bin/env bash
# Phase 1 restart watcher (the user, 2026-11-25; SPEC §17 amendments 67 (5), 70; a stand-in until the restarter): runs
# detached on the Mac for one step's spend phase.
#  1. Starts the Spot VM in us-central1-a (unless NO_START=1): a stockout is retried every 10 minutes for up to 6
#     hours (no zone move: the user's decision); `vlar-ops spend check --phase PHASE` must pass before every start.
#  2. Polls every 3 minutes (VM status, the runner's STATE.json, the ledger). When the VM has stopped:
#     - a compute.instances.preempted operation since its last start => restart the same way, at most MAX_AUTO times
#       (8 per step); one more preemption => no restart, the user decides;
#     - any other stop (queue empty, guard stop, exit 76, a manual stop) => no restart, the watcher ends.
#  3. A status it cannot read is UNKNOWN and never counts as running (2026-11-25: an expired sign-in left the old
#     watcher reading empty statuses as "not stopped" for 3.4 hours). After 3 unreadable polls in a row it logs a
#     warning and raises a macOS notification with sound, and again every 20 unreadable polls (~1 h).
# Reads and starts go through application-default credentials (ops/gcp/gce_adc.py), not the gcloud CLI.
#   PHASE=phase1b nohup caffeinate -i -s bash ops/gcp/34_restart_watch.sh >> outputs/l4-retry/retry.log 2>&1 &
# Test hooks: ENV_SH, GCE, VLAR_OPS, NOTIFY, POLL_S, TRY_S, RECHECK_S, MAX_POLLS, NO_START, LIB_ONLY (functions only).
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
source "${ENV_SH:-ops/gcp/env.sh}"
Z=us-central1-a
PHASE="${PHASE:-phase1a}"
MAX_AUTO="${MAX_AUTO:-8}"
TRY_S="${TRY_S:-600}"; RETRY_HOURS="${RETRY_HOURS:-6}"; POLL_S="${POLL_S:-180}"; WATCH_HOURS="${WATCH_HOURS:-30}"
read -r -a GCE <<< "${GCE:-${REPO}/.venv-ops/bin/python ${REPO}/ops/gcp/gce_adc.py}"
read -r -a VOPS <<< "${VLAR_OPS:-${REPO}/.venv-ops/bin/vlar-ops}"
log() { echo "$(date -u +%FT%TZ) restarter: $*"; }
mask() { sed -e "s/${VM_NAME}/<VM>/g" -e "s/${PROJECT_ID}/<PROJECT>/g" -e "s#${BUCKET}#<BUCKET>#g"; }
notify() {   # a macOS notification with sound (no-op where osascript is missing)
  local msg="${1//\"/\'}"
  "${NOTIFY:-osascript}" -e "display notification \"${msg}\" with title \"vla-router watcher\" sound name \"Sosumi\"" \
    >/dev/null 2>&1 || true
}

classify() {   # a read status -> running | starting | stopping | stopped | unknown; only "running" counts as running
  case "$1" in
    RUNNING) echo running ;;
    PROVISIONING|STAGING|REPAIRING) echo starting ;;
    STOPPING|SUSPENDING) echo stopping ;;
    TERMINATED|STOPPED|SUSPENDED) echo stopped ;;
    *) echo unknown ;;
  esac
}
status() { "${GCE[@]}" status 2>/dev/null | tail -n 1; }
preempted_since_last_start() { "${GCE[@]}" preempted-since-start >/dev/null 2>&1; }

start_vm() {   # 0 started; 1 do not start / failed (the user decides)
  local out rc deadline=$(( $(date +%s) + RETRY_HOURS * 3600 ))
  out="$("${VOPS[@]}" spend check --phase "${PHASE}" --ops-config ops/ops.yaml 2>&1 | tail -n 1)"; rc=$?
  log "spend check --phase ${PHASE}: ${out}"
  (( rc == 0 )) || { log "spend check fails: NOT starting the VM"; return 1; }
  while (( $(date +%s) < deadline )); do
    out="$("${GCE[@]}" start 2>&1)"; rc=$?
    if (( rc == 0 )); then log "${Z}: start issued; VM $(status)"; return 0; fi
    if (( rc == 3 )); then log "${Z}: stockout; retrying in ${TRY_S} s"; sleep "${TRY_S}"; continue; fi
    log "${Z}: start failed (not a stockout): $(tail -n 3 <<<"${out}" | tr '\n' ' ')" | mask; return 1
  done
  log "${Z}: no capacity for ${RETRY_HOURS} h; the user decides (no zone move)"; return 1
}

[[ "${LIB_ONLY:-0}" == 1 ]] && return 0 2>/dev/null

log "start: phase ${PHASE}, VM ${VM_NAME:+<VM>} in ${Z}, automatic restarts allowed: ${MAX_AUTO}"
if [[ "${NO_START:-0}" != 1 ]]; then start_vm || exit 1; fi
auto=0; blind=0; polls=0
end=$(( $(date +%s) + WATCH_HOURS * 3600 ))
while (( $(date +%s) < end )); do
  sleep "${POLL_S}"
  polls=$(( polls + 1 )); [[ -n "${MAX_POLLS:-}" ]] && (( polls > MAX_POLLS )) && { log "test: ${MAX_POLLS} polls done"; exit 0; }
  st="$(status)"; cls="$(classify "${st}")"
  state="$("${GCE[@]}" state 2>/dev/null)"
  ledger="$("${VOPS[@]}" spend status --ops-config ops/ops.yaml 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); p=d.get("phase") or {}; print("total", d["total_usd"], "phase", p.get("name"), round(p.get("vm_usd",0),3), "/ stop", p.get("stop_usd"))' 2>/dev/null)"
  if [[ "${cls}" == unknown ]]; then
    blind=$(( blind + 1 ))
    log "watch: VM UNKNOWN (status not readable: '${st}'); runner: ${state}; ledger: ${ledger}" | mask
    if (( blind == 3 || (blind > 3 && (blind - 3) % 20 == 0) )); then
      log "status NOT readable ${blind} polls in a row (credentials?): no restart can happen until it is"
      notify "VM status unreadable ${blind} polls in a row: renew gcloud auth / application-default login"
    fi
    continue
  fi
  blind=0
  log "watch: VM ${st}; runner: ${state}; ledger: ${ledger}" | mask
  [[ "${cls}" == stopped ]] || continue
  if ! preempted_since_last_start; then sleep "${RECHECK_S:-60}"; fi   # the operation can lag the stop
  if preempted_since_last_start; then
    if (( auto >= MAX_AUTO )); then
      log "preempted again after ${auto} automatic restarts: NOT restarting; the user decides"
      notify "VM preempted again after ${auto} restarts: not restarted"; exit 5
    fi
    auto=$(( auto + 1 )); log "preempted: automatic restart ${auto}/${MAX_AUTO}"
    start_vm || { notify "VM restart failed: see retry.log"; exit 1; }
  else
    log "the VM stopped and it was NOT a preemption (runner: ${state}): no restart; the user decides"; exit 0
  fi
done
log "watched ${WATCH_HOURS} h; watcher ends (VM $(status))"
