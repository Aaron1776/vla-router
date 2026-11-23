#!/usr/bin/env bash
# cap-seconds: 7200
# Phase 0 step 7 (SPEC §11 step 7, §15; amendments 34, 35, 46): the golden set and the v1.0 freeze. Queued by hand,
# only with the user's explicit approval after reading step 6's calibration summary (500 stops for it).
#   1. five base-rollout snapshots with their recorded trajectories and outcomes (`vlar-ops golden select`), kept
#      beside the calibration file (golden/golden.json + .npz);
#   2. replayed through the production servers bit for bit (`golden check --mark-checked`), else stop;
#   3. `vlar-ops finish freeze`: the calibration file frozen, freeze.json with its lineage, the golden set and the
#      ops digest (amendment 46: every container that runs or checks ops code mounts the host's vlar_ops, OPS_RUN);
#      `provenance.verify_freeze` re-checks it;
#   4. the self-check again on the frozen file (as every later start runs it before any worker, lib.sh run_phase).
# The queue then runs empty and the VM stops: Phase 0 is complete.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
calfile require-write-once >/dev/null || fail "no write-once calibration: run step 2 (200) first"
if calibration_frozen; then echo "already frozen"; calfile show; exit 0; fi
G="${P0D}/step7"; rm -rf "${G}"; mkdir -p "${G}/records" "${CAL_DIR}/golden"
for t in runs points steps; do
  gcloud storage rsync -r "${VLAR_STORAGE_URL}/records/${t}/base" "${G}/records/${t}/base" >/dev/null \
    || fail "records/${t}/base not downloaded"
done
GOLD="${CAL_DIR}/golden/golden.json"
docker run --rm -v "${CAL_DIR}:${CAL_DIR}" -v "${P0D}:${P0D}" "${CAL_ENV[@]}" "${OPS_RUN[@]}" "vlar-sim:${C}" \
  vlar-ops golden select --root "${G}" --out "${GOLD}" || fail "golden set not chosen"
[[ -s /etc/vlar/batching.env ]] || write_batching_env
systemctl start vlar-xvla vlar-pi05 || fail "model servers did not start"
stop_servers() { systemctl stop vlar-xvla vlar-pi05; }
wait_servers || { stop_servers; fail "model servers not answering /info after 10 minutes"; }
golden_check -v "${CAL_DIR}:${CAL_DIR}" -- --mark-checked \
  || { stop_servers; sync_calibration; fail "the golden set does not replay bit for bit: no freeze (§15)"; }
docker run --rm -v "${CAL_DIR}:${CAL_DIR}" "${CAL_ENV[@]}" "${OPS_RUN[@]}" "vlar-sim:${C}" \
  vlar-ops finish freeze --golden "${GOLD}" || { stop_servers; fail "freeze failed"; }
docker run --rm -v /etc/vlar:/etc/vlar:ro "${CAL_ENV[@]}" "${OPS_RUN[@]}" "vlar-sim:${C}" python -c \
  'from vlar_harness import provenance; from vlar_harness.config import load_config; c = load_config(); assert c.frozen; provenance.verify_freeze(c); print("freeze verified")' \
  || { stop_servers; fail "the freeze does not verify"; }
sync_calibration
golden_check || { stop_servers; fail "the golden set does not replay on the frozen file (§15)"; }
stop_servers
gcloud storage cp -q /var/lib/vlar/golden-pass.json "${VLAR_PHASE0_URL}/step7/golden-pass.json" >/dev/null 2>&1 || true
calfile show
echo "harness v1.0 frozen; Phase 0 complete"
