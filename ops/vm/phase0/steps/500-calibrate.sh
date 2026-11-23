#!/usr/bin/env bash
# cap-seconds: 7200
# Phase 0 step 6 (SPEC §5, §11 steps 3 and 6; amendments 9, 10, 38), queued by hand once the base rollouts (400) are
# done: from the base rollouts' and the π0.5 parity episodes' records,
#   1. the relative handoff check: X-VLA's first-target distances at ordinary chunk boundaries of the base rollouts,
#      then the VM row with both production servers up (amendment 10);
#   2. the per-task time limits and margin m, written to the calibration file (`vlar-ops finish calibrate`);
#   3. the grasp check (amendment 38): flagged -> exit 75, the user looks before the freeze (to go on anyway, the
#      user creates <phase0>/step6/grasps-accepted in the bucket and the step is re-run);
# then queues 600 (step 7, the freeze). Re-runnable: everything is recomputed from the records.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
calfile require-write-once >/dev/null || fail "no write-once calibration: run step 2 (200) first"
G="${P0D}/step6"; rm -rf "${G}"; mkdir -p "${G}/records"
for t in runs events steps; do
  for ph in base pi05_parity; do
    gcloud storage rsync -r "${VLAR_STORAGE_URL}/records/${t}/${ph}" "${G}/records/${t}/${ph}" >/dev/null \
      || fail "records/${t}/${ph} not downloaded"
  done
done
OPS=(docker run "${SIM_RUN[@]}" "${CAL_ENV[@]}" "vlar-sim:${C}" vlar-ops)
up6() { gcloud storage cp -q "$1" "${VLAR_PHASE0_URL}/step6/$(basename "$1")" >/dev/null 2>&1 || echo "warning: $1 not synced"; }

"${OPS[@]}" finish first-targets --root "${G}" --out "${G}/first_targets.json" || fail "first-target distances failed"
up6 "${G}/first_targets.json"
[[ -s /etc/vlar/batching.env ]] || write_batching_env
systemctl start vlar-xvla vlar-pi05 || fail "model servers did not start"
wait_servers || { systemctl stop vlar-xvla vlar-pi05; fail "model servers not answering /info after 10 minutes"; }
row handoff-relative-l4 docker run "${SIM_RUN[@]}" "${CAL_ENV[@]}" -e VLAR_XVLA_SERVER=127.0.0.1:8777 \
  -e VLAR_PI05_SERVER=127.0.0.1:8778 -e VLAR_BASE_FIRST_TARGETS="${G}/first_targets.json" "vlar-sim:${C}" \
  python -m pytest -q -rs -p no:cacheprovider tests/reference/test_handoff.py -k vm
systemctl stop vlar-xvla vlar-pi05
[[ "$(cat "${RESULTS}/handoff-relative-l4.status")" == passed ]] \
  || fail "relative handoff check failed (logs/results/${C}/handoff-relative-l4.log)"

docker run --rm -v "${CAL_DIR}:${CAL_DIR}" -v "${P0D}:${P0D}" "${CAL_ENV[@]}" "${OPS_RUN[@]}" "vlar-sim:${C}" \
  vlar-ops finish calibrate --root "${G}" --write --out "${G}/time_limits.json" || fail "time-limit calibration failed"
sync_calibration
up6 "${G}/time_limits.json"

"${OPS[@]}" grasps --root "${G}" --out "${G}/grasps.json"; rc=$?
up6 "${G}/grasps.json"
if (( rc == 75 )); then
  gcloud storage ls "${VLAR_PHASE0_URL}/step6/grasps-accepted" >/dev/null 2>&1 \
    || need_input "grasp check flagged (phase0/step6/grasps.json): the gripper thresholds do not separate the base rollouts' grasps cleanly; the user decides before the freeze"
  echo "grasp check flagged, accepted by the user (phase0/step6/grasps-accepted)"
elif (( rc != 0 )); then
  fail "grasp check failed (exit ${rc})"
fi
gcloud storage cp -q "${R}/ops/vm/phase0/steps/600-freeze.sh" "${VLAR_PHASE0_URL}/queue/600-freeze.sh" \
  || fail "step 6 done but 600-freeze could not be queued"
echo "step 6 done: time limits written; 600-freeze queued"
