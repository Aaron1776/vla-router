#!/usr/bin/env bash
# cap-seconds: 14400
# spend-phase: phase1f
# Phase 1, the final tier, version (a) (the user, 2026-11-26; SPEC §17 amendments 65, 67 (3), 75): more base points,
# b1 and b2 with 3 seeds, sized to the money left when 740 ended (the disk/storage line released beyond the end plus
# one day; `phase1.final_tier_points` at the phase's stop point). A and F are ONE stratified draw: Tranche A's seed and
# settings at A's 64 + F's n points per task, A's points left out (`vlar-ops phase1 plan-f`, built on the Mac), so
# every A and F point's inclusion probability is the combined draw's. Tranche A's step (720) otherwise, in order:
#  1. The branch plan from <storage>/phase1f/plan/; its files' SHA-256s go into the reports; plan.json must be the plan
#     selection.json describes, and selection.json must name Tranche A as the draw it continues.
#  2. Both production servers, the image cross-check (amendment 54) and the golden-set replay: any mismatch stops the
#     step (exit 76), no debugging.
#  3. The branch jobs through `run_phase` (amendment 64's RAM rule, 7 workers at 5.5 GiB beside the servers' measured
#     20.45 GB peak; at most 2 jobs per worker process), open jobs point by point (`vlar-ops pilot reorder`).
# Every report goes to <storage>/phase1f/reports/ when the step ends, however it ends; its exit is recorded with
# "step": "750" in reports/step-exits.jsonl (the Mac's step check reads it).
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
: "${VLAR_STORAGE_URL:?}"
PH=phase1f
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
  echo "{\"boot_id\": \"${BOOT}\", \"step\": \"750\", \"step_exit\": ${rc}, \"at\": $(date +%s)}" >> "${REP}/step-exits.jsonl"
  gcloud storage rsync -r "${REP}" "${RURL}" >/dev/null 2>&1 || echo "WARNING: ${PH} reports not uploaded"
}
trap upload_reports EXIT
trap 'exit 143' TERM INT

# ---- 1. the plan
gcloud storage cp -q "${VLAR_STORAGE_URL}/${PH}/plan/*.json" "${PD}/" || fail "${PH} plan not downloaded"
for f in plan labels selection; do [[ -s "${PD}/${f}.json" ]] || fail "${PH} plan: ${f}.json missing"; done
(cd "${PD}" && sha256sum ./*.json) > "${REP}/plan-sha256-${BOOT}.txt"
python3 - "${PD}" "${PH}" <<'PY' || fail "${PH} plan: plan.json is not the plan selection.json describes, or not one draw with Tranche A"
import json, sys
d, ph = sys.argv[1:3]
head, *jobs = [json.loads(line) for line in open(f"{d}/plan.json") if line.strip()]   # Plan.write: header, jobs
sel = json.load(open(f"{d}/selection.json"))
assert head["plan_id"] == sel["plan_id"] and head["phase"] == ph == sel["phase"], (head["plan_id"], sel["plan_id"])
assert head["n_jobs"] == len(jobs) == sel["n_jobs"] and all(j["phase"] == ph for j in jobs), len(jobs)
assert len({j["subject"] for j in jobs}) == sel["n_points"] == len(json.load(open(f"{d}/labels.json")))
assert {j["kind"] for j in jobs} == {"b1_continue", "b2_one_call"}, {j["kind"] for j in jobs}
assert sel.get("combined_with", {}).get("phase") == "phase1a", sel.get("combined_with")   # A and F one draw
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
