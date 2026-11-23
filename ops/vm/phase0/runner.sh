#!/usr/bin/env bash
# Phase 0 step runner on the L4 VM (systemd unit ops/vm/vlar-phase0.service; started by ops/vm/phase0/startup.sh on
# every boot). Long jobs run here, never in an SSH session from the Mac.
#
# The queue is a folder in the bucket, <VLAR_PHASE0_URL>/queue/, of step scripts run one at a time in name order
# (ops/vm/phase0/steps/; queued from the Mac by ops/gcp/31_phase0_queue.sh, or by a step, e.g. the parity gate
# queues the base rollouts). Each step runs under a time cap (`# cap-seconds: N` in the script, else
# VLAR_STEP_CAP_S), its log goes to logs/<step>.log, and after every step the logs, the step history (with the spend
# guard's running total) and STATE.json are mirrored to the bucket.
#
# Exit status of a step:   0 done (moved to done/)
#                         75 needs the user: stays queued (it re-runs on the next boot, once the input is there)
#                         76 a gate failed: moved to failed/, stop, no debugging (e.g. π0.5 parity)
#                        124 the time cap (timeout's TERM): moved to failed/
#                        137 SIGKILL: the time cap if the cap had passed (timeout's KILL after an ignored TERM), else
#                            an out-of-memory kill if the kernel log since the step's start says so, else "killed"
#                      other failed: moved to failed/
# Steps write a one-line reason to $VLAR_REASON_FILE.
#
# The VM never sits idle on the GPU: when the queue is empty, or a step fails or needs the user, the runner stops
# the model servers and workers, syncs the job queue's results and its own files to the bucket, writes the reason to
# STATE.json there, and stops the VM (shutdown -h: the instance is STOPPED, its disk kept). The spend daemon
# (vlar-spend.service) runs independently and stops everything at the hard limit; while its STOP flag is up the
# runner starts no step.
set -uo pipefail
: "${VLAR_PHASE0_URL:?gs://<bucket>/vla-router/phase0}"
D="${VLAR_PHASE0_DIR:-/var/lib/vlar/phase0}"
STOP_FLAG="${VLAR_STOP_FLAG:-/var/lib/vlar/STOP}"
STEP_CAP_S="${VLAR_STEP_CAP_S:-14400}"
SHUTDOWN_CMD="${VLAR_SHUTDOWN_CMD:-shutdown -h now}"
NEEDS_INPUT=75
GATE_FAILED=76
mkdir -p "$D/queue" "$D/done" "$D/failed" "$D/logs"
export VLAR_PHASE0_URL VLAR_PHASE0_DIR="$D"
CURRENT=""
CHILD=""

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(now)] $*"; }

spend_json() {   # the spend guard's running total, as one JSON object (null if unreadable)
  vlar-ops spend status 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(json.dumps({k: d.get(k) for k in ("total_usd", "vm_usd", "disk_usd", "storage_usd", "vm_hours",
                                             "hard_limit_usd")}))
except Exception:
    print("null")' 2>/dev/null || echo null
}

oom_killed() {   # oom_killed <epoch>: the kernel log since then records an out-of-memory kill (amendment 41)
  local log
  log="$(journalctl -k --no-pager --since "@$1" 2>/dev/null)"
  grep -qiE 'out of memory|oom-kill|oom_reaper|killed process' <<<"$log"
}

write_state() {  # write_state <state> <step> <reason>
  python3 - "$D/STATE.json" "$1" "$2" "$3" "$(now)" "$(spend_json)" <<'EOF'
import json, sys
path, state, step, reason, at, spend = sys.argv[1:7]
json.dump({"state": state, "step": step or None, "reason": reason, "at": at, "spend": json.loads(spend)},
          open(path, "w"), indent=1)
EOF
  gcloud storage cp -q "$D/STATE.json" "$VLAR_PHASE0_URL/STATE.json" >/dev/null 2>&1 || say "warning: STATE.json not uploaded"
}

history() {      # history <step> <outcome> <exit status> <seconds>
  python3 - "$D/history.jsonl" "$1" "$2" "$3" "$4" "$(now)" "$(spend_json)" <<'EOF'
import json, sys
path, step, outcome, rc, secs, at, spend = sys.argv[1:8]
with open(path, "a") as f:
    f.write(json.dumps({"step": step, "outcome": outcome, "exit": int(rc), "seconds": int(secs), "at": at,
                        "spend": json.loads(spend)}) + "\n")
EOF
}

sync_files() {   # mirror logs, step outcomes and history to the bucket (never back into its queue)
  local sub
  for sub in logs done failed; do
    gcloud storage rsync -r -q "$D/$sub" "$VLAR_PHASE0_URL/$sub" >/dev/null 2>&1 || say "warning: $sub not synced"
  done
  [[ -f "$D/history.jsonl" ]] && { gcloud storage cp -q "$D/history.jsonl" "$VLAR_PHASE0_URL/history.jsonl" \
    >/dev/null 2>&1 || say "warning: history not synced"; }
  return 0
}

pull_queue() {   # new steps from the bucket's queue (the bucket is the queue's source of truth)
  gcloud storage rsync -q "$VLAR_PHASE0_URL/queue" "$D/queue" >/dev/null 2>&1 || true
}

move_step() {    # move_step <file> <done|failed>: locally and in the bucket
  mv "$D/queue/$1" "$D/$2/$1"
  gcloud storage cp -q "$D/$2/$1" "$VLAR_PHASE0_URL/$2/$1" >/dev/null 2>&1 || say "warning: $1 not copied to $2/ in the bucket"
  gcloud storage rm -q "$VLAR_PHASE0_URL/queue/$1" >/dev/null 2>&1 || say "warning: $1 still in the bucket's queue"
}

finish() {       # finish <state> <reason>: stop GPU work, sync everything, record why, stop the VM
  say "stopping: $2"
  systemctl stop 'vlar-worker@*' vlar-xvla vlar-pi05 >/dev/null 2>&1 || true
  if [[ -e /var/lib/vlar/queue.sqlite ]]; then
    vlar-ops queue sync >>"$D/logs/runner-sync.log" 2>&1 || say "warning: vlar-ops queue sync failed"
  fi
  sync_files
  write_state "$1" "$CURRENT" "$2"
  say "state $1 written to $VLAR_PHASE0_URL/STATE.json; stopping the VM"
  sync_files
  trap '' TERM INT          # the shutdown stops this service too: keep the reason just written
  $SHUTDOWN_CMD
  exit 0
}

on_term() {      # preemption or a manual stop: abandon the step (it stays queued and re-runs on the next boot)
  say "SIGTERM during ${CURRENT:-no step}: abandoning it (it stays queued)"
  [[ -n "$CURRENT" ]] && history "$CURRENT" interrupted 143 0
  sync_files
  [[ -n "$CHILD" ]] && kill -TERM "$CHILD" 2>/dev/null
  if [[ -e "$STOP_FLAG" ]]; then
    write_state stopped "$CURRENT" "the spend guard reached its hard limit and stopped the VM during ${CURRENT:-no step}; the step stays queued"
    exit 143
  fi
  write_state interrupted "$CURRENT" "the VM was stopped from outside (preemption or a manual stop) during ${CURRENT:-no step}; the step stays queued"
  exit 143
}
trap on_term TERM INT

run_step() {     # run_step <file>
  local f="$1" name="${1%.sh}" cap rc reason t0
  CURRENT="$name"
  cap="$(sed -n 's/^# cap-seconds: *\([0-9][0-9]*\) *$/\1/p' "$D/queue/$f" | head -n 1)"
  cap="${cap:-$STEP_CAP_S}"
  export VLAR_STEP_NAME="$name" VLAR_REASON_FILE="$D/logs/$name.reason"
  rm -f "$VLAR_REASON_FILE"
  write_state running "$name" "running (cap ${cap} s)"
  say "start $name (cap ${cap} s)"
  t0="$(date +%s)"
  timeout --kill-after=120 "$cap" bash "$D/queue/$f" >>"$D/logs/$name.log" 2>&1 &
  CHILD=$!
  wait "$CHILD"; rc=$?
  CHILD=""
  reason="$(head -c 2000 "$VLAR_REASON_FILE" 2>/dev/null | tr '\n' ' ')"
  say "end $name: exit $rc ${reason}"
  case "$rc" in
    0)  move_step "$f" done; history "$name" done 0 $(( $(date +%s) - t0 )); sync_files ;;
    "$NEEDS_INPUT")
        history "$name" needs_input "$rc" $(( $(date +%s) - t0 )); sync_files
        finish needs_input "$name needs input: ${reason:-see logs/$name.log}" ;;
    "$GATE_FAILED")
        move_step "$f" failed; history "$name" gate_failed "$rc" $(( $(date +%s) - t0 )); sync_files
        finish gate_failed "$name: gate failed: ${reason:-see logs/$name.log}" ;;
    124)
        move_step "$f" failed; history "$name" time_cap "$rc" $(( $(date +%s) - t0 )); sync_files
        finish failed "$name hit its time cap (${cap} s)" ;;
    137)
        local elapsed=$(( $(date +%s) - t0 )) kind why
        if (( elapsed >= cap )); then
          kind=time_cap; why="hit its time cap (${cap} s) and was killed after ignoring SIGTERM"
        elif oom_killed "$t0"; then
          kind=oom_killed; why="was killed by the kernel's out-of-memory killer after ${elapsed} s (exit 137)"
        else
          kind=killed; why="was killed (SIGKILL, exit 137) after ${elapsed} s, within its ${cap} s cap; no out-of-memory kill in the kernel log"
        fi
        move_step "$f" failed; history "$name" "$kind" "$rc" "$elapsed"; sync_files
        finish failed "$name $why" ;;
    *)  move_step "$f" failed; history "$name" failed "$rc" $(( $(date +%s) - t0 )); sync_files
        finish failed "$name failed (exit $rc): ${reason:-see logs/$name.log}" ;;
  esac
  CURRENT=""
}

say "phase 0 runner started; queue $VLAR_PHASE0_URL/queue"
# Evidence for restarts without a reboot (2026-11-23: the runner restarted at 04:35:50Z mid-step, no reboot; suspected
# unattended-upgrades/needrestart): who touched the runner and what apt changed, kept in logs/ with every start.
{ journalctl --no-pager -u vlar-phase0 -u apt-daily-upgrade -u unattended-upgrades --since -2days 2>&1 | tail -n 120
  echo "== /var/log/apt/history.log"; tail -n 60 /var/log/apt/history.log 2>&1
  echo "== needrestart"; journalctl --no-pager --since -2days 2>/dev/null | grep -i needrestart | tail -n 40
} > "$D/logs/system-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 || true
while true; do
  [[ -e "$STOP_FLAG" ]] && finish stopped "the spend guard's STOP flag is up (hard limit); no step started"
  pull_queue
  next="$(ls "$D/queue" 2>/dev/null | grep -E '\.sh$' | LC_ALL=C sort | head -n 1)"
  [[ -z "$next" ]] && finish idle "out of queued work: every queued step is done"
  run_step "$next"
done
