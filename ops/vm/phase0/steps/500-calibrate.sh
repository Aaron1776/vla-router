#!/usr/bin/env bash
# cap-seconds: 7200
# Phase 0 step 6 (SPEC §5, §11 steps 3 and 6; amendments 9, 10, 38), queued by hand once the base rollouts (400) are
# done: from the base rollouts' and the π0.5 parity episodes' records,
#   1. the relative handoff check: X-VLA's first-target distances at ordinary chunk boundaries of the base rollouts,
#      then the VM row with both production servers up (amendment 10);
#   2. the per-task time limits and margin m, written to the calibration file (`vlar-ops finish calibrate`);
#   3. the grasp check (amendment 38); a flag goes into the summary;
#   4. the calibration summary (`vlar-ops finish summary`: per-task limits, m, π0.5's share within the limits, the
#      grasp check), printed and uploaded to <phase0>/step6/calibration-summary.{txt,json};
# then it stops for the user (exit 75; the user's rule, 2026-11-23): the freeze (600) is queued only with the user's
# explicit approval after reading the summary, never by this step. Exit 75 keeps the step queued, so it re-runs on
# the next boot: a re-run whose records and calibration file are the uploaded summary's does nothing and ends done
# (exit 0), so a 600 queued behind it runs; otherwise everything is recomputed from the records (to force that,
# delete the uploaded summary).
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
S="${G}/calibration-summary.json"
if gcloud storage cp -q "${VLAR_PHASE0_URL}/step6/calibration-summary.json" "${G}/previous-summary.json" >/dev/null 2>&1 \
   && "${OPS[@]}" finish unchanged --root "${G}" --previous "${G}/previous-summary.json"; then
  echo "step 6 already done for these records and this calibration file (phase0/step6/calibration-summary.txt)"
  exit 0
fi

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
(( rc == 0 || rc == 75 )) || fail "grasp check failed (exit ${rc})"      # 75: flagged, reported in the summary

"${OPS[@]}" finish summary --root "${G}" --time-limits "${G}/time_limits.json" --grasps "${G}/grasps.json" \
  --out "${S}" || fail "calibration summary failed"
for f in "${S%.json}.txt" "${S}"; do        # the JSON last: a re-run counts as done only once both are up
  gcloud storage cp -q "${f}" "${VLAR_PHASE0_URL}/step6/$(basename "${f}")" >/dev/null 2>&1 \
    || fail "$(basename "${f}") not uploaded: the user cannot review step 6"
done
need_input "step 6 done, calibration written: read phase0/step6/calibration-summary.txt; the freeze (600) is queued only with the user's explicit approval"
