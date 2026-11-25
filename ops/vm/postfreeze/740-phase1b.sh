#!/usr/bin/env bash
# cap-seconds: 64800
# spend-phase: phase1b
# Phase 1, Tranche B after the screen (SPEC §17 amendments 56, 57, 62 (2), 65; vlar_ops/phase1b.py). Boots only with
# the user's approval; the runner starts it only while the spend phase `phase1b` is open. In order:
#  1. Selection by episode from the screen's markers (`vlar-ops phase1 select-b`: per task n/k episodes, 2/3 from
#     failed screened episodes), written to <storage>/phase1b/rerun/ before any re-run.
#  2. The golden gate (exit 76 on any mismatch).
#  3. The re-runs (record full, at the screen's setup count) through `run_phase explore`, compared with their screened
#     runs every 2 minutes while they run (`compare-b`): any difference writes the spend guard's stop flag, so all
#     re-runs stop (amendment 56; max-effort checkpoint), and the step stops with exit 76.
#  4. Every re-run compared once more (all must have passed), then the branch plan over their points
#     (`plan-b`: k points per episode, b1 and b2, 3 seeds), uploaded to <storage>/phase1b/plan/.
#  5. The branch jobs through `run_phase phase1b`, open jobs point by point (`pilot reorder`).
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
: "${VLAR_STORAGE_URL:?}"
PH=phase1b
W="/var/lib/vlar/${PH}"
SD="${W}/screen"; RD="${W}/rerun"; PD="${W}/plan"; REP="${W}/reports"
RURL="${VLAR_STORAGE_URL}/${PH}/reports"
BOOT="$(cat /proc/sys/kernel/random/boot_id)"
STOP_FLAG="${VLAR_STOP_FLAG:-/var/lib/vlar/STOP}"
mkdir -p "${SD}" "${RD}" "${PD}" "${REP}"

upload_reports() {
  local rc=$?
  [[ -n "${WATCH_PID:-}" ]] && kill "${WATCH_PID}" 2>/dev/null
  cp -p "${P0D}/logs/explore-supervisor.json" "${REP}/rerun-supervisor.json" 2>/dev/null
  cp -p "${P0D}/logs/${PH}-supervisor.json" "${REP}/" 2>/dev/null
  cp -p /var/lib/vlar/golden-pass.json "${REP}/golden-pass-${BOOT}.json" 2>/dev/null
  echo "{\"boot_id\": \"${BOOT}\", \"step\": \"740\", \"step_exit\": ${rc}, \"at\": $(date +%s)}" >> "${REP}/step-exits.jsonl"
  gcloud storage rsync -r "${REP}" "${RURL}" >/dev/null 2>&1 || echo "WARNING: ${PH} reports not uploaded"
}
trap upload_reports EXIT
trap 'exit 143' TERM INT

ops_sim() {   # vlar-ops in the sim image (pyarrow, the harness config), the bucket and the work dir mounted
  docker run --rm --network host -v /etc/vlar:/etc/vlar:ro -v /var/lib/vlar:/var/lib/vlar \
    -v "${R}/ops/ops.yaml:/etc/vlar-ops/ops.yaml:ro" -e VLAR_OPS_CONFIG=/etc/vlar-ops/ops.yaml \
    "${CAL_ENV[@]}" "${OPS_RUN[@]}" "vlar-sim:${C}" vlar-ops "$@"
}
param() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "${SD}/params.json" "$1"; }

# ---- 1. selection (kept across boots: the re-run plan is made once)
gcloud storage cp -q "${VLAR_STORAGE_URL}/${PH}/screen/*.json" "${SD}/" || fail "${PH} screen plan not downloaded"
gcloud storage cp -q "${VLAR_STORAGE_URL}/${PH}/rerun/*.json" "${RD}/" 2>/dev/null
if [[ ! -s "${RD}/plan.json" ]]; then
  ops_sim phase1 select-b --plan-dir "${SD}" --per-task "$(param per_task)" --k "$(param k)" --seed "$(param seed)" \
    --out "${RD}" || fail "${PH} selection failed"
  gcloud storage cp -q "${RD}/plan.json" "${RD}/episodes.json" "${VLAR_STORAGE_URL}/${PH}/rerun/" || fail "${PH} re-run plan not uploaded"
fi
(cd "${RD}" && sha256sum ./*.json) > "${REP}/rerun-sha256-${BOOT}.txt"

# ---- 2. the gate
[[ -s /etc/vlar/batching.env ]] || write_batching_env
calibration_frozen || fail "the calibration file is not frozen"
systemctl start vlar-xvla vlar-pi05 || fail "model servers did not start"
wait_servers || { systemctl stop vlar-xvla vlar-pi05; fail "model servers not answering /info after 10 minutes"; }
golden_check || { systemctl stop vlar-xvla vlar-pi05; gate_fail "golden-set replay or image cross-check failed: stopped, no debugging"; }
export VLAR_GOLDEN_DONE=1

# ---- 3. the re-runs, compared as they finish
CMP="${REP}/compare.json"
vlar-ops queue add "${RD}/plan.json" || fail "${PH} re-run queue add"
( while sleep 120; do
    ops_sim phase1 compare-b --plan-dir "${RD}" --out "${CMP}" --stop-flag "${STOP_FLAG}" >/dev/null 2>&1
    [[ $? -eq 76 ]] && break
  done ) &
WATCH_PID=$!
echo "{\"boot_id\": \"${BOOT}\", \"reruns_started\": $(date +%s.%N)}" >> "${REP}/timing.jsonl"
VLAR_SUPERVISE_ARGS="--limit-gib 5.5 --servers-peak-gb 20.45 --max-jobs 2" run_phase explore "--phase explore"
kill "${WATCH_PID}" 2>/dev/null; WATCH_PID=""

# ---- 4. every comparison, then the branch plan
ops_sim phase1 compare-b --plan-dir "${RD}" --out "${CMP}" --stop-flag "${STOP_FLAG}" --require-all
rc=$?
[[ ${rc} -eq 76 ]] && gate_fail "a re-run differs from its screened run: all re-runs stopped, no branches (max-effort checkpoint)"
[[ ${rc} -ne 0 ]] && fail "${PH}: not every re-run compared"
if [[ ! -s "${PD}/plan.json" ]]; then
  ops_sim phase1 plan-b --plan-dir "${RD}" --compare "${CMP}" --per-task "$(param per_task)" --k "$(param k)" \
    --seed "$(param seed)" --out "${PD}" || fail "${PH} branch plan failed"
  gcloud storage cp -q "${PD}/*.json" "${VLAR_STORAGE_URL}/${PH}/plan/" || fail "${PH} branch plan not uploaded"
fi
(cd "${PD}" && sha256sum ./*.json) > "${REP}/plan-sha256-${BOOT}.txt"

# ---- 5. the branches
vlar-ops queue add "${PD}/plan.json" || fail "${PH} queue add"
vlar-ops queue resume >/dev/null || fail "${PH} queue resume"
vlar-ops pilot reorder --plan-dir "${PD}" --out "${REP}/reorder-${BOOT}.json" || fail "${PH} reorder"
echo "{\"boot_id\": \"${BOOT}\", \"phase_started\": $(date +%s.%N)}" >> "${REP}/timing.jsonl"
VLAR_SUPERVISE_ARGS="--limit-gib 5.5 --servers-peak-gb 20.45 --max-jobs 2" run_phase "${PH}" "--phase ${PH}"
echo "{\"boot_id\": \"${BOOT}\", \"phase_ended\": $(date +%s.%N)}" >> "${REP}/timing.jsonl"
reason "${PH} done: every re-run matched its screen, every branch job done"
