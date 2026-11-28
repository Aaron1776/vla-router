#!/usr/bin/env bash
# cap-seconds: 54000
# spend-phase: phase1a
# Phase 1, Tranche A (SPEC §17 amendment 65; docs/PHASE1-PLAN-2026-11-25.md): b1 and b2, 3 seeds each, at a stratified
# draw of base points (failed-parent share 2/3, uniform within strata), restored from the stored base snapshots: no
# screening and no re-runs. The runner starts it only while the spend phase `phase1a` is open (opened from the Mac with
# the plan's allocation and line, which hold the new disk's setup steps 010-090 too); the spend service stops claims
# at the phase's stop point and shuts the VM down at its line. The pilot's step (710) adapted; in order:
#  1. The branch plan built on the Mac (`vlar-ops phase1 plan-a`), from <storage>/phase1a/plan/; its files' SHA-256s
#     go into the reports, and plan.json must be the plan selection.json describes.
#  2. Both production servers, the image cross-check (amendment 54) and the golden-set replay: any mismatch stops the
#     step (exit 76), no debugging.
#  3. The branch jobs through `run_phase` (the supervisor: amendment 64's RAM rule, 7 workers at 5.5 GiB beside the
#     servers' measured 20.45 GB peak; at most 2 jobs per worker process), open jobs point by point
#     (`vlar-ops pilot reorder`, which reads the plan's phase), with the phase's wall times in reports/timing.jsonl.
# The pilot's cheap checks (i)-(iii) are not repeated: they passed on the pilot's boots (amendment 62, the corrected
# (iii) included) and branch runs restore snapshots, which the setup count does not touch.
# Every report goes to <storage>/phase1a/reports/ when the step ends, however it ends.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
: "${VLAR_STORAGE_URL:?}"
PH=phase1a
W="/var/lib/vlar/${PH}"
PD="${W}/plan"
REP="${W}/reports"
RURL="${VLAR_STORAGE_URL}/${PH}/reports"
BOOT="$(cat /proc/sys/kernel/random/boot_id)"
mkdir -p "${PD}" "${REP}"

upload_reports() {
  local rc=$?
  cp -p "${P0D}/logs/${PH}-supervisor.json" "${REP}/" 2>/dev/null
  cp -p /var/lib/vlar/golden-pass.json "${REP}/golden-pass-${BOOT}.json" 2>/dev/null
  echo "{\"boot_id\": \"${BOOT}\", \"step_exit\": ${rc}, \"at\": $(date +%s)}" >> "${REP}/step-exits.jsonl"
  gcloud storage rsync -r "${REP}" "${RURL}" >/dev/null 2>&1 || echo "WARNING: ${PH} reports not uploaded"
}
trap upload_reports EXIT
trap 'exit 143' TERM INT

# ---- 1. the plan
gcloud storage cp -q "${VLAR_STORAGE_URL}/${PH}/plan/*.json" "${PD}/" || fail "${PH} plan not downloaded"
for f in plan labels selection; do [[ -s "${PD}/${f}.json" ]] || fail "${PH} plan: ${f}.json missing"; done
(cd "${PD}" && sha256sum ./*.json) > "${REP}/plan-sha256-${BOOT}.txt"
python3 - "${PD}" "${PH}" <<'PY' || fail "${PH} plan: plan.json is not the plan selection.json describes"
import json, sys
d, ph = sys.argv[1:3]
head, *jobs = [json.loads(line) for line in open(f"{d}/plan.json") if line.strip()]   # Plan.write: header, jobs
sel = json.load(open(f"{d}/selection.json"))
assert head["plan_id"] == sel["plan_id"] and head["phase"] == ph == sel["phase"], (head["plan_id"], sel["plan_id"])
assert head["n_jobs"] == len(jobs) == sel["n_jobs"] and all(j["phase"] == ph for j in jobs), len(jobs)
assert len({j["subject"] for j in jobs}) == sel["n_points"] == len(json.load(open(f"{d}/labels.json")))
assert {j["kind"] for j in jobs} == {"b1_continue", "b2_one_call"}, {j["kind"] for j in jobs}
print(f"{ph} plan {head['plan_id']}: {len(jobs)} jobs, {sel['n_points']} points")
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
echo "golden set and image cross-check passed on boot ${BOOT}"

# ---- 3. the branches: open jobs point by point, workers by the RAM rule (amendment 64)
vlar-ops queue add "${PD}/plan.json" || fail "${PH} queue add"
vlar-ops queue resume >/dev/null || fail "${PH} queue resume"
vlar-ops pilot reorder --plan-dir "${PD}" --out "${REP}/reorder-${BOOT}.json" || fail "${PH} reorder"
echo "{\"boot_id\": \"${BOOT}\", \"phase_started\": $(date +%s.%N)}" >> "${REP}/timing.jsonl"
VLAR_SUPERVISE_ARGS="--limit-gib 5.5 --servers-peak-gb 20.45 --max-jobs 2" run_phase "${PH}" "--phase ${PH}"
echo "{\"boot_id\": \"${BOOT}\", \"phase_ended\": $(date +%s.%N)}" >> "${REP}/timing.jsonl"
echo "${PH} branches done: $(vlar-ops queue status --phase "${PH}" | tail -n 1)"
reason "${PH} done: golden set and image check passed, every branch job done"
