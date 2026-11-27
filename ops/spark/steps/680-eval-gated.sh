#!/usr/bin/env bash
# cap-seconds: 64800
# The Spark's gated evaluation runs (SPEC §10; ops/spark/evaluate.py), for every method that may call π0.5:
#   EVAL_METHOD  pi05_only | fixed | random | decider;  EVAL_ARGS  extra `evaluate.py plan` options (e.g. --rate 0.1,
#   --decider-dir DIR);  EVAL_TAG  phase suffix;  EVAL_SHARDS  parallel simulators (8);  EVAL_LIMIT  episodes per shard
#   VLAR_SPARK_VLAR_TRAIN  vlar_train source to mount (deciders only)
# Startup as step 650: both production servers at the calibrated geometry, the π0.5 gate on 8779 (one /act in flight)
# and its startup check (gate_check.py); then the plan (evaluation seed namespace, setup count 2) and the run, which
# refuses without the gate. Frozen calibration. Output: P0D/eval/<phase>/ (results, records, the gate's log).
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
M="${EVAL_METHOD:?pi05_only, fixed, random or decider}"; N="${EVAL_SHARDS:-8}"
PH="eval_${M}${EVAL_TAG:+_${EVAL_TAG}}"; E="${P0D}/eval"; OUT="${E}/${PH}"; mkdir -p "${OUT}" "${E}/gate-obs"
CALF="${CAL_DIR}/calibration.yaml"
[[ -s "${CALF}" ]] || need_input "no frozen calibration file in ${CAL_DIR}"
EV=(-e VLAR_CALIBRATION="${CALF}" -v "${R}/ops/spark:/spark:ro")
[[ -n "${VLAR_SPARK_VLAR_TRAIN:-}" ]] && EV+=(-v "${VLAR_SPARK_VLAR_TRAIN}:/opt/vla-router/src/vlar_train:ro")
stop_all() { docker rm -f vlar-pi05-gate >/dev/null 2>&1; systemctl stop vlar-xvla vlar-pi05; }
[[ -s /etc/vlar/batching.env ]] || write_batching_env
source /etc/vlar/batching.env
systemctl start vlar-xvla vlar-pi05 || fail "model servers did not start"
wait_servers || { stop_all; fail "model servers not answering /info after 10 minutes"; }
docker run -d --rm --name vlar-pi05-gate --network host -v "${R}/ops/spark:/spark:ro" -v "${OUT}:${OUT}" \
  vlar-aarch64-base python3 /spark/pi05_gate.py --listen 8779 --backend 127.0.0.1:8778 --log "${OUT}/gate.jsonl" >/dev/null
for _ in $(seq 1 30); do curl -sf http://127.0.0.1:8779/info | grep -q '"max_in_flight": 1' && break; sleep 2; done
curl -sf http://127.0.0.1:8779/info | python3 -c 'import json,sys; g=json.load(sys.stdin)["gate"]; print("gate:", g); sys.exit(0 if g["mode"]=="gate" and g["max_in_flight"]==1 else 1)' \
  || { stop_all; fail "the π0.5 gate is not on"; }
[[ -s "${E}/gate-obs/pi05_requests.json" ]] || docker run "${SIM_RUN[@]}" -e VLAR_CALIBRATION="${CALF}" "vlar-sim:${C}" \
  vlar-ops step2 capture --out "${E}/gate-obs" >/dev/null || { stop_all; fail "request capture failed"; }
docker run "${GPU_RUN[@]}" -v "${E}:${E}" -v "${R}/ops/spark:/spark:ro" "vlar-pi05:${C}" python /spark/gate_check.py \
  --obs "${E}/gate-obs" --max-batch "${VLAR_MAX_BATCH}" --out "${OUT}/gate-check.json" | tail -n 1 \
  || { stop_all; fail "gate startup check failed (${OUT}/gate-check.json)"; }
# shellcheck disable=SC2086
docker run "${SIM_RUN[@]}" "${EV[@]}" "vlar-sim:${C}" python /spark/evaluate.py plan --method "${M}" ${EVAL_ARGS:-} \
  ${EVAL_TAG:+--tag "${EVAL_TAG}"} --out "${E}" | tail -n 1 | cut -c1-300 || { stop_all; fail "evaluation plan failed"; }
LIM=(); [[ -n "${EVAL_LIMIT:-}" ]] && LIM=(--limit "${EVAL_LIMIT}")
pids=()
for i in $(seq 0 $((N - 1))); do
  docker run "${SIM_RUN[@]}" "${EV[@]}" -e VLAR_SPARK_PLATFORM="${P0D}/logs/platform.json" "vlar-sim:${C}" \
    python /spark/evaluate.py run --jobs "${E}/${PH}-jobs.jsonl" --shard "${i}/${N}" --out "${OUT}" --pi05-port 8779 \
    "${LIM[@]}" > "${OUT}/shard-${i}.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
stop_all
(( rc == 0 )) || fail "a shard failed (${OUT}/shard-*.log)"
echo "episodes: $(cat "${OUT}"/*-results-*.jsonl | wc -l)"
