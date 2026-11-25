#!/usr/bin/env bash
# cap-seconds: 21600
# spend-phase: phase1b-screen
# Phase 1, Tranche B, the screen (SPEC §17 amendments 56, 62 (2), 65; vlar_ops/phase1b.py): ~1,150 exploration
# episodes, record off, profile none, each on a fresh simulator (setup count 2, rule (A); the worker replays the spec's
# setup_count and every marker records the measured one). Boots only with the user's approval; the runner starts it
# only while the spend phase `phase1b-screen` is open. In order:
#  1. The screen plan built on the Mac (`vlar-ops phase1 plan-b-screen`), from <storage>/phase1b/screen/, with
#     params.json (per_task, seed, k for steps 740's selection); SHA-256s into the reports.
#  2. Both production servers, the image cross-check and the golden-set replay: any mismatch stops (exit 76).
#  3. The screen through `run_phase explore` (7 workers by amendment 64's RAM rule, at most 2 jobs per process).
# Every report goes to <storage>/phase1b/reports/ when the step ends, however it ends.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
: "${VLAR_STORAGE_URL:?}"
PH=phase1b
W="/var/lib/vlar/${PH}"
PD="${W}/screen"
REP="${W}/reports"
RURL="${VLAR_STORAGE_URL}/${PH}/reports"
BOOT="$(cat /proc/sys/kernel/random/boot_id)"
mkdir -p "${PD}" "${REP}"

upload_reports() {
  local rc=$?
  cp -p "${P0D}/logs/explore-supervisor.json" "${REP}/screen-supervisor.json" 2>/dev/null
  cp -p /var/lib/vlar/golden-pass.json "${REP}/golden-pass-${BOOT}.json" 2>/dev/null
  echo "{\"boot_id\": \"${BOOT}\", \"step\": \"730\", \"step_exit\": ${rc}, \"at\": $(date +%s)}" >> "${REP}/step-exits.jsonl"
  gcloud storage rsync -r "${REP}" "${RURL}" >/dev/null 2>&1 || echo "WARNING: ${PH} reports not uploaded"
}
trap upload_reports EXIT
trap 'exit 143' TERM INT

# ---- 1. the plan
gcloud storage cp -q "${VLAR_STORAGE_URL}/${PH}/screen/*.json" "${PD}/" || fail "${PH} screen plan not downloaded"
for f in plan params; do [[ -s "${PD}/${f}.json" ]] || fail "${PH} screen: ${f}.json missing"; done
(cd "${PD}" && sha256sum ./*.json) > "${REP}/screen-sha256-${BOOT}.txt"
python3 - "${PD}" <<'PY' || fail "${PH} screen plan is not a fresh-start record-off exploration plan"
import json, sys
head, *jobs = [json.loads(line) for line in open(f"{sys.argv[1]}/plan.json") if line.strip()]
assert head["phase"] == "explore" and head["n_jobs"] == len(jobs) and head["params"]["tranche"] == "phase1b"
assert all(j["phase"] == "explore" and j["kind"] == "episode" and j["spec"]["record"] == "off"
           and j["spec"]["setup_count"] == 2 for j in jobs)
print(f"screen plan {head['plan_id']}: {len(jobs)} episodes")
PY

# ---- 2. servers, image cross-check, golden replay (the gate)
[[ -s /etc/vlar/batching.env ]] || write_batching_env
calibration_frozen || fail "the calibration file is not frozen"
systemctl start vlar-xvla vlar-pi05 || fail "model servers did not start"
wait_servers || { systemctl stop vlar-xvla vlar-pi05; fail "model servers not answering /info after 10 minutes"; }
golden_check || { systemctl stop vlar-xvla vlar-pi05; gate_fail "golden-set replay or image cross-check failed: stopped, no debugging"; }
python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m["passed"] and m.get("image_check") else 1)' \
  /var/lib/vlar/golden-pass.json || { systemctl stop vlar-xvla vlar-pi05; gate_fail "the golden-pass marker records no image check"; }
export VLAR_GOLDEN_DONE=1

# ---- 3. the screen
vlar-ops queue add "${PD}/plan.json" || fail "${PH} screen queue add"
vlar-ops queue resume >/dev/null || fail "${PH} queue resume"
echo "{\"boot_id\": \"${BOOT}\", \"screen_started\": $(date +%s.%N)}" >> "${REP}/timing.jsonl"
VLAR_SUPERVISE_ARGS="--limit-gib 5.5 --servers-peak-gb 20.45 --max-jobs 2" run_phase explore "--phase explore"
echo "{\"boot_id\": \"${BOOT}\", \"screen_ended\": $(date +%s.%N)}" >> "${REP}/timing.jsonl"
reason "${PH} screen done: golden set and image check passed, every screened episode done"
