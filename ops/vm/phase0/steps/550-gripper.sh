#!/usr/bin/env bash
# cap-seconds: 900
# Amendments 50-51 (the user, 2026-11-24): the gripper thresholds from ground truth, written on the VM as a new
# calibration version before the freeze (600): shut 3.0 mm (was 9.00 mm from step 2's empty-gripper procedure), open
# unchanged at 71.55 mm. They are not write-once. The version's note records the ground-truth validation and where
# the analysis files are. Queued by hand with 600, after 500, with the user's approval of the freeze. Written here,
# not on the Mac: the VM restores the calibration from the bucket only when its own file is missing (startup.sh), so a
# version written elsewhere would never reach it, and the VM's next sync would overwrite it in the bucket.
# Re-runnable: does nothing once the shut threshold is 3.0 mm.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
calfile require-write-once >/dev/null || fail "no write-once calibration: run step 2 (200) first"
if calibration_frozen; then fail "the calibration file is frozen: gripper thresholds not written"; fi
SHUT=0.003
cur="$(calfile get measurement.gripper_thresholds.shut_width_m)" || fail "cannot read the gripper thresholds"
if [[ "${cur}" == "${SHUT}" ]]; then
  echo "gripper shut threshold already ${SHUT} m"; calfile show; exit 0
fi
U="${P0D}/gripper-thresholds-update.json"
printf '{"measurement": {"gripper_thresholds": {"shut_width_m": %s}}}\n' "${SHUT}" > "${U}"
NOTE="Amendments 50-51 (the user, 2026-11-24): gripper shut threshold 3.0 mm (was ${cur} m), open unchanged. \
Ground-truth validation (replaces the retired 5%-band grasp check): 500 restored base-rollout snapshots at settled \
closed-gripper steps; at 3.0 mm, 0.6% of held read as shut, 39.3% of not-held read as holding, 6.5% total error \
(random 400: 339 held, 61 not held). Analysis files: ${VLAR_PHASE0_URL}/step6/grasp-review/"
calfile write "${U}" --note "${NOTE}" || fail "gripper thresholds not written"
[[ "$(calfile get measurement.gripper_thresholds.shut_width_m)" == "${SHUT}" ]] || fail "shut threshold not ${SHUT} after the write"
sync_calibration
calfile get measurement.gripper_thresholds
calfile show
