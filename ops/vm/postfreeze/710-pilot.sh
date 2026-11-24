#!/usr/bin/env bash
# cap-seconds: 18000
# spend-phase: pilot
# The pilot (docs/PROPOSALS-2026-11-24.md §4 and §0.4; SPEC §17 amendments 56-59), approved up to $2.50. The runner
# starts it only while the spend phase `pilot` is open (`vlar-ops spend phase-open --phase pilot --allocation-usd 1.60
# --line-usd 2.50`, from the Mac); the spend service stops claims at the phase's stop point (~$2.34) and shuts the VM
# down at $2.50. In order:
#  1. The branch plan built on the Mac (`vlar-ops pilot plan`: 150 base points, 1,638 jobs), from <storage>/pilot/plan/;
#     its files' SHA-256s go into the reports, and plan.json must be the plan selection.json describes.
#  2. Both production servers, the image cross-check (amendment 54) and the golden-set replay: any mismatch stops the
#     step (exit 76), no debugging.
#  3. The cheap checks (`vlar-ops pilot check`, in the sim image, nothing written to the dataset): (i) /features
#     against the recorded probe features, (ii) xvla_only against routed always-local, (iii) record off against
#     record full, bit for bit. A mismatch is reported and does not stop the branches (the boot is paid for); the
#     step then ends with exit 76. Done once: a re-run after a preemption reads the earlier report.
#  4. The branch jobs through `run_phase` (so through the supervisor: memory limits, --max-jobs recycling, restarts),
#     with the phase's wall times in reports/timing.jsonl and the supervisor's summary (peak memory, recycles).
# Every report goes to <storage>/pilot/reports/ when the step ends, however it ends.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
: "${VLAR_STORAGE_URL:?}"
PILOT=/var/lib/vlar/pilot
PD="${PILOT}/plan"
REP="${PILOT}/reports"
RURL="${VLAR_STORAGE_URL}/pilot/reports"
BOOT="$(cat /proc/sys/kernel/random/boot_id)"
mkdir -p "${PD}" "${REP}" "${PILOT}/staging"

upload_reports() {
  local rc=$?
  cp -p "${P0D}/logs/pilot-supervisor.json" "${REP}/" 2>/dev/null
  cp -p /var/lib/vlar/golden-pass.json "${REP}/golden-pass-${BOOT}.json" 2>/dev/null
  echo "{\"boot_id\": \"${BOOT}\", \"step_exit\": ${rc}, \"at\": $(date +%s)}" >> "${REP}/step-exits.jsonl"
  gcloud storage rsync -r "${REP}" "${RURL}" >/dev/null 2>&1 || echo "WARNING: pilot reports not uploaded"
}
trap upload_reports EXIT
trap 'exit 143' TERM INT

# ---- 1. the plan
gcloud storage cp -q "${VLAR_STORAGE_URL}/pilot/plan/*.json" "${PD}/" || fail "pilot plan not downloaded"
for f in plan labels selection checks; do [[ -s "${PD}/${f}.json" ]] || fail "pilot plan: ${f}.json missing"; done
(cd "${PD}" && sha256sum plan.json labels.json selection.json checks.json) > "${REP}/plan-sha256-${BOOT}.txt"
python3 - "${PD}" <<'PY' || fail "pilot plan: plan.json is not the plan selection.json describes"
import json, sys
d = sys.argv[1]
head, *jobs = [json.loads(line) for line in open(f"{d}/plan.json") if line.strip()]   # Plan.write: header, jobs
sel = json.load(open(f"{d}/selection.json"))
assert head["plan_id"] == sel["plan_id"] and head["phase"] == "pilot", (head["plan_id"], sel["plan_id"])
assert head["n_jobs"] == len(jobs) == sel["n_jobs"] and all(j["phase"] == "pilot" for j in jobs), len(jobs)
assert len({j["subject"] for j in jobs}) == sel["n_points"] == len(json.load(open(f"{d}/labels.json")))
print(f"pilot plan {head['plan_id']}: {len(jobs)} jobs, {sel['n_points']} points")
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

# ---- 3. the cheap checks, once
if [[ ! -s "${REP}/checks.json" ]]; then
  t0="$(date +%s)"
  docker run --rm --gpus all --network host -v /etc/vlar:/etc/vlar:ro -v /var/lib/vlar:/var/lib/vlar \
    -v "${R}/ops/ops.yaml:/etc/vlar-ops/ops.yaml:ro" -e VLAR_OPS_CONFIG=/etc/vlar-ops/ops.yaml -e MUJOCO_GL=egl \
    -e PYOPENGL_PLATFORM=egl "${CAL_ENV[@]}" "${OPS_RUN[@]}" "vlar-sim:${C}" \
    vlar-ops pilot check --plan-dir "${PD}" --staging "${PILOT}/staging" --out "${REP}/checks.json.tmp" \
    > "${REP}/checks-${BOOT}.log" 2>&1
  echo "pilot check exit $? after $(( $(date +%s) - t0 )) s: $(tail -n 2 "${REP}/checks-${BOOT}.log" | tr '\n' ' ')"
  [[ -s "${REP}/checks.json.tmp" ]] && mv "${REP}/checks.json.tmp" "${REP}/checks.json"
  rm -rf "${PILOT}/staging"; mkdir -p "${PILOT}/staging"
fi
CHECKS="$(python3 -c 'import json,sys
try: r=json.load(open(sys.argv[1]))
except Exception: print("no report (the check did not finish)"); sys.exit()
print("passed" if r["passed"] else "MISMATCH: " + json.dumps({"features": r["features"]["problems"], "features_equal_probe": r["features"]["features_equal_probe"], "xvla_only": [x["problems"] for x in r["xvla_only"]], "record_off": [x["problems"] for x in r["record_off"]]})[:1500])' \
  "${REP}/checks.json")"
echo "pilot checks: ${CHECKS}"

# ---- 4. the branches
vlar-ops queue add "${PD}/plan.json" || fail "pilot queue add"
echo "{\"boot_id\": \"${BOOT}\", \"phase_started\": $(date +%s.%N)}" >> "${REP}/timing.jsonl"
run_phase pilot "--phase pilot"
echo "{\"boot_id\": \"${BOOT}\", \"phase_ended\": $(date +%s.%N)}" >> "${REP}/timing.jsonl"
echo "pilot branches done: $(vlar-ops queue status --phase pilot | tail -n 1)"
[[ "${CHECKS}" == passed ]] || gate_fail "pilot branches done; checks: ${CHECKS} (reports in ${RURL}): no debugging on the VM"
reason "pilot done: golden set and image check passed, checks (i)-(iii) passed, every branch job done"
