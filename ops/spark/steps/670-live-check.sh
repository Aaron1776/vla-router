#!/usr/bin/env bash
# cap-seconds: 14400
# Trained deciders live, and the check that their inputs equal what training builds from the records
# (ops/spark/live_check.py, live.py, evaluate.py). For each task T of LC_TASKS, fold T of a `vlar_train --variants 2x2`
# output (T is that fold's test task), two episode sets over LC_INITS:
#   set R: router_hist drives (profile router), router built on the side from the same observe();
#   set C: cascader_hist drives (profile cascader_nospread), cascader_nospread on the side.
# Each set runs twice: `evaluate.py run` (record off, as an evaluation) and `live_check.py run` (record full, points
# rows and built inputs kept), then `live_check.py compare` (exit 1 if a column differs; the step then exits 76).
#   LC_DIR    output folder under the Spark root;  LC_TRAIN  the training output folder (fold<i>/<decider>/)
#   LC_TASKS  default 0,5;  LC_INITS  default 0-3
# Start-up as step 650: servers at the calibrated geometry, the π0.5 gate on 8779, its /info and startup checks.
# vlar_train from VLAR_SPARK_VLAR_TRAIN, mounted over the image's copy.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
S="${LC_DIR:?output folder}"; TR="${LC_TRAIN:?training output folder}"; mkdir -p "${S}/obs"
CALF="${CAL_DIR}/calibration.yaml"; VT="${VLAR_SPARK_VLAR_TRAIN:?vlar_train source folder}"
EV=(-e VLAR_CALIBRATION="${CALF}" -e VLAR_SPARK_PLATFORM="${P0D}/logs/platform.json" -v "${R}/ops/spark:/spark:ro"
    -v "${VT}:/opt/vla-router/src/vlar_train:ro" -v "${S}:${S}" -v "${TR}:${TR}:ro")
stop_all() { docker rm -f vlar-pi05-gate >/dev/null 2>&1; systemctl stop vlar-xvla vlar-pi05; }
[[ -s /etc/vlar/batching.env ]] || write_batching_env
source /etc/vlar/batching.env
systemctl start vlar-xvla vlar-pi05 || fail "model servers did not start"
wait_servers || { stop_all; fail "model servers not answering /info after 10 minutes"; }
docker run -d --rm --name vlar-pi05-gate --network host -v "${R}/ops/spark:/spark:ro" -v "${S}:${S}" \
  vlar-aarch64-base python3 /spark/pi05_gate.py --listen 8779 --backend 127.0.0.1:8778 --log "${S}/gate.jsonl" >/dev/null
for _ in $(seq 1 30); do curl -sf http://127.0.0.1:8779/info | grep -q '"max_in_flight": 1' && break; sleep 2; done
curl -sf http://127.0.0.1:8779/info | python3 -c 'import json,sys; g=json.load(sys.stdin)["gate"]; print("gate:", g); sys.exit(0 if g["mode"]=="gate" and g["max_in_flight"]==1 else 1)' \
  || { stop_all; fail "the π0.5 gate is not on"; }
[[ -s "${S}/obs/pi05_requests.json" ]] || docker run "${SIM_RUN[@]}" "${EV[@]}" "vlar-sim:${C}" \
  vlar-ops step2 capture --out "${S}/obs" >/dev/null || { stop_all; fail "request capture failed"; }
docker run "${GPU_RUN[@]}" -v "${S}:${S}" -v "${R}/ops/spark:/spark:ro" "vlar-pi05:${C}" python /spark/gate_check.py \
  --obs "${S}/obs" --max-batch "${VLAR_MAX_BATCH}" --out "${S}/gate-check.json" | tail -n 1 \
  || { stop_all; fail "gate startup check failed (${S}/gate-check.json)"; }
T="${LC_TASKS:-0,5}"; I="${LC_INITS:-0-3}"
one() {   # one TASK SET DRIVER SIDE: plan, the evaluation run (record off), the check run (record full)
  local t="$1" set="$2" drive="$3" side="$4" tag="$2$1" d="${S}/$2$1"
  mkdir -p "${d}"
  docker run "${SIM_RUN[@]}" "${EV[@]}" "vlar-sim:${C}" python /spark/evaluate.py plan --method decider \
    --decider-dir "${TR}/fold${t}/${drive}" --tag "${tag}" --tasks "${t}" --inits "${I}" --out "${d}/plan" || return 1
  local jobs="${d}/plan/eval_decider_${tag}-jobs.jsonl"
  docker run "${SIM_RUN[@]}" "${EV[@]}" "vlar-sim:${C}" python /spark/evaluate.py run --jobs "${jobs}" \
    --out "${d}/eval" --pi05-port 8779 || return 1
  docker run "${SIM_RUN[@]}" "${EV[@]}" "vlar-sim:${C}" python /spark/live_check.py run --jobs "${jobs}" \
    --out "${d}/check" --side "${TR}/fold${t}/${side}" --pi05-port 8779 || return 1
}
pids=(); rc=0
for t in ${T//,/ }; do
  one "${t}" R router_hist router > "${S}/R${t}.log" 2>&1 & pids+=($!)
  one "${t}" C cascader_hist cascader_nospread > "${S}/C${t}.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "${p}" || rc=1; done
stop_all
(( rc == 0 )) || fail "an episode set failed (${S}/[RC]*.log)"
bad=0
for d in "${S}"/[RC]*/; do
  d="${d%/}"; echo "== $(basename "${d}")"
  docker run --rm -e VLAR_CALIBRATION="${CALF}" -v /etc/vlar:/etc/vlar:ro -v "${R}/ops/spark:/spark:ro" \
    -v "${VT}:/opt/vla-router/src/vlar_train:ro" -v "${S}:${S}" "vlar-sim:${C}" python /spark/live_check.py compare \
    --run "${d}/check" --eval "${d}/eval" --out "${d}/compare.json" | tail -n 1 || bad=1
done
python3 - "${S}" <<'PY'
import json, sys
log = [json.loads(l) for l in open(sys.argv[1] + "/gate.jsonl")]
print("gate log: /act", len(log), "non-200", sum(r["status"] != 200 for r in log))
PY
(( bad == 0 )) || gate_fail "live inputs differ from the recorded rows (${S}/*/compare.json)"
