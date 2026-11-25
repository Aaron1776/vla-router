#!/usr/bin/env bash
# cap-seconds: 3600
# The Spark's relative handoff check (SPEC §11 step 3, amendment 10), the row step 500 ran on the L4, without 500's
# calibration writes: both production servers at the calibrated geometry (batching.env from the calibration file),
# then tests/reference/test_handoff.py -k vm with the L4 base rollouts' first-target distances
# (References/l4-phase0/step6/first_targets.json, copied to step6/ by the caller).
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
G="${P0D}/step6"
[[ -s "${G}/first_targets.json" ]] || need_input "no ${G}/first_targets.json (the L4's base first-target distances)"
[[ -s /etc/vlar/batching.env ]] || write_batching_env
systemctl start vlar-xvla vlar-pi05 || fail "model servers did not start"
wait_servers || { systemctl stop vlar-xvla vlar-pi05; fail "model servers not answering /info after 10 minutes"; }
row handoff-relative-spark docker run "${SIM_RUN[@]}" "${CAL_ENV[@]}" -e VLAR_XVLA_SERVER=127.0.0.1:8777 \
  -e VLAR_PI05_SERVER=127.0.0.1:8778 -e VLAR_BASE_FIRST_TARGETS="${G}/first_targets.json" "vlar-sim:${C}" \
  python -m pytest -q -rs -p no:cacheprovider tests/reference/test_handoff.py -k vm
systemctl stop vlar-xvla vlar-pi05
[[ "$(cat "${RESULTS}/handoff-relative-spark.status")" == passed ]] \
  || fail "relative handoff check failed (logs/results/${C}/handoff-relative-spark.log)"
