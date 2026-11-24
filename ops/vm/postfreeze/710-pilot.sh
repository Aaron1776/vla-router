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
#     step then ends with exit 76. Done once: a re-run after a preemption reads the earlier report. When the plan has
#     checks-iii.json, the corrected (iii) runs once too (every re-run on a fresh simulator at the base run's model
#     load count: LIBERO's fixture placement depends on it, the finding of 2026-11-24) and supersedes the first (iii).
#  4. The branch jobs through `run_phase` (so through the supervisor: memory limits, --max-jobs recycling, restarts),
#     open jobs point by point (`vlar-ops pilot reorder`), workers of at least 5 GiB each (as many as fit), with the
#     phase's wall times in reports/timing.jsonl and the supervisor's summary (peak memory, recycles).
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
(cd "${PD}" && sha256sum ./*.json) > "${REP}/plan-sha256-${BOOT}.txt"
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

# ---- 3. the cheap checks, each once: checks.json ((i)-(iii), first boot), then checks-iii.json when the plan has it
# (the corrected (iii), the user 2026-11-24: every re-run on a fresh simulator at the base run's load count; it
# supersedes the first (iii))
pilot_check() {   # pilot_check <subjects file> <report name>
  [[ -s "${REP}/$2" ]] && return 0
  local t0; t0="$(date +%s)"
  docker run --rm --gpus all --network host -v /etc/vlar:/etc/vlar:ro -v /var/lib/vlar:/var/lib/vlar \
    -v "${R}/ops/ops.yaml:/etc/vlar-ops/ops.yaml:ro" -e VLAR_OPS_CONFIG=/etc/vlar-ops/ops.yaml -e MUJOCO_GL=egl \
    -e PYOPENGL_PLATFORM=egl "${CAL_ENV[@]}" "${OPS_RUN[@]}" "vlar-sim:${C}" \
    vlar-ops pilot check --plan-dir "${PD}" --subjects "$1" --staging "${PILOT}/staging" --out "${REP}/$2.tmp" \
    > "${REP}/${2%.json}-${BOOT}.log" 2>&1
  echo "pilot check $1 exit $? after $(( $(date +%s) - t0 )) s: $(tail -n 2 "${REP}/${2%.json}-${BOOT}.log" | tr '\n' ' ')"
  [[ -s "${REP}/$2.tmp" ]] && mv "${REP}/$2.tmp" "${REP}/$2"
  rm -rf "${PILOT}/staging"; mkdir -p "${PILOT}/staging"
}
pilot_check checks.json checks.json
[[ -s "${PD}/checks-iii.json" ]] && pilot_check checks-iii.json checks-iii.json
python3 - "${REP}" "${PD}" > "${REP}/checks-summary.txt" <<'PY'
import json, os, sys
rep, pd = sys.argv[1:3]
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
first, iii = load(f"{rep}/checks.json"), load(f"{rep}/checks-iii.json")
want_iii = os.path.exists(f"{pd}/checks-iii.json")
if first is None or (want_iii and iii is None):
    print("no report (a check did not finish)")
    sys.exit()
bad = {}
if first["features"]["problems"]:
    bad["features"] = first["features"]["problems"]
x = [r["problems"] for r in first["xvla_only"] if not r["passed"]]
if x:
    bad["xvla_only"] = x
off = (iii if want_iii else first)["record_off"]
o = [{"subject": r["subject"], "loads": r.get("loads"), "problems": r["problems"]} for r in off if not r["passed"]]
if o:
    bad["record_off"] = o
print("passed" if not bad else "MISMATCH: " + json.dumps(bad)[:1500])
PY
CHECKS="$(cat "${REP}/checks-summary.txt")"
echo "pilot checks: ${CHECKS}"

# ---- 4. the branches: the open jobs point by point (`pilot reorder`), workers sized for the measured memory
vlar-ops queue add "${PD}/plan.json" || fail "pilot queue add"
vlar-ops queue resume >/dev/null || fail "pilot queue resume"
vlar-ops pilot reorder --plan-dir "${PD}" --out "${REP}/reorder-${BOOT}.json" || fail "pilot reorder"
echo "{\"boot_id\": \"${BOOT}\", \"phase_started\": $(date +%s.%N)}" >> "${REP}/timing.jsonl"
# The first boot's workers were killed at 4 GiB (peak sampled 4.07 GB): at least 5 GiB each, as many as fit beside
# the servers, and still at most 2 jobs per process (the user, 2026-11-24).
VLAR_SUPERVISE_ARGS="--min-limit-gib 5 --max-jobs 2" run_phase pilot "--phase pilot"
echo "{\"boot_id\": \"${BOOT}\", \"phase_ended\": $(date +%s.%N)}" >> "${REP}/timing.jsonl"
echo "pilot branches done: $(vlar-ops queue status --phase pilot | tail -n 1)"
[[ "${CHECKS}" == passed ]] || gate_fail "pilot branches done; checks: ${CHECKS} (reports in ${RURL}): no debugging on the VM"
reason "pilot done: golden set and image check passed, checks (i)-(iii) passed, every branch job done"
