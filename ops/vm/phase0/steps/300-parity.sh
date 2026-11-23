#!/usr/bin/env bash
# cap-seconds: 28800
# Phase 0 step 4 (SPEC §11), the π0.5 parity run with its automatic gate (the user's rule, 2026-11-22): pi05_only on
# initial states 0-9 of every task (100 episodes), scene seed 1000 + k, 520 steps, no time cutoff, event-logged, by
# the production workers (calibrated count) against the production servers; then the runs table against References/
# (exact McNemar p >= 0.05). Pass: a manifest check over the parity records, then the base rollouts are queued
# (400-base.sh). Fail: exit 76, the VM stops, nothing is debugged. Re-runnable: the job queue resumes.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
calfile require-write-once || fail "no write-once calibration: run step 2 (200-step2-measure) first"
PL="${P0D}/plans"; mkdir -p "${PL}"
vlar-ops plan episodes --phase pi05_parity --inits 0-9 --mode pi05_only --scene pi05_parity \
  --out "${PL}/pi05_parity.json" || fail "parity plan"
vlar-ops queue add "${PL}/pi05_parity.json" || fail "queue add"
run_phase pi05_parity "--phase pi05_parity --option scene_seed=pi05_parity"
G="${P0D}/parity"; rm -rf "${G}"; mkdir -p "${G}/records/runs" "${G}/m/records" "${G}/m/results"
gcloud storage rsync -r "${VLAR_STORAGE_URL}/records/runs/pi05_parity" "${G}/records/runs/pi05_parity" >/dev/null \
  || fail "parity runs not downloaded"
docker run "${SIM_RUN[@]}" "${CAL_ENV[@]}" "vlar-sim:${C}" vlar-ops parity gate \
  --records "${G}" --references /opt/vla-router/References --out "${G}/gate.json"; rc=$?
gcloud storage cp -q "${G}/gate.json" "${VLAR_PHASE0_URL}/parity/gate.json" >/dev/null 2>&1 || true
(( rc == 76 )) && gate_fail "π0.5 parity failed (exact McNemar p < 0.05; phase0/parity/gate.json): stop, no debugging"
(( rc == 0 )) || fail "parity gate gave no verdict (exit ${rc}); see logs/300-parity.log"
# The production worker's records through the Cloud Storage backend: a manifest over the parity jobs.
for t in runs steps events; do
  gcloud storage rsync -r "${VLAR_STORAGE_URL}/records/${t}/pi05_parity" "${G}/m/records/${t}/pi05_parity" >/dev/null 2>&1
done
gcloud storage rsync -r "${VLAR_STORAGE_URL}/results/pi05_parity" "${G}/m/results/pi05_parity" >/dev/null \
  || fail "parity markers not downloaded"
docker run "${SIM_RUN[@]}" "vlar-sim:${C}" vlar-ops dataset manifest --root "${G}/m" \
  || fail "manifest over the parity records failed"
gcloud storage cp -q "${R}/ops/vm/phase0/steps/400-base.sh" "${VLAR_PHASE0_URL}/queue/400-base.sh" \
  || fail "parity passed but the base step could not be queued"
echo "parity passed; base rollouts queued"
