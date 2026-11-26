#!/usr/bin/env bash
# cap-seconds: 64800
# The Spark's scripted evaluation runs (SPEC §10; ops/spark/evaluate.py): EVAL_METHOD (xvla_only), EVAL_SHARDS (8),
# EVAL_LIMIT (episodes per shard, a smoke). Plans the method's 500 episodes in the evaluation seed namespace at setup
# count 2, then runs them with `repro.py run` against both production servers at the calibrated geometry, under the
# frozen calibration (b80124b2). No live time limit: scored offline from the event logs. Output: P0D/eval/<method>/.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
M="${EVAL_METHOD:?xvla_only}"; N="${EVAL_SHARDS:-8}"
E="${P0D}/eval"; OUT="${E}/${EVAL_OUT:-${M}}"; mkdir -p "${OUT}"
CALF="${CAL_DIR}/calibration.yaml"
[[ -s "${CALF}" ]] || need_input "no frozen calibration file in ${CAL_DIR}"
docker run --rm -v "${E}:${E}" -v "${R}/ops/spark:/spark:ro" "vlar-sim:${C}" python /spark/evaluate.py plan \
  --method "${M}" --out "${E}" || fail "evaluation plan failed"
LIM=(); [[ -n "${EVAL_LIMIT:-}" ]] && LIM=(--limit "${EVAL_LIMIT}")
[[ -s /etc/vlar/batching.env ]] || write_batching_env
systemctl start vlar-xvla vlar-pi05 || fail "model servers did not start"
wait_servers || { systemctl stop vlar-xvla vlar-pi05; fail "model servers not answering /info after 10 minutes"; }
pids=()
for i in $(seq 0 $((N - 1))); do
  docker run "${SIM_RUN[@]}" -e VLAR_CALIBRATION="${CALF}" -e VLAR_SPARK_PLATFORM="${P0D}/logs/platform.json" \
    -v "${R}/ops/spark:/spark:ro" "vlar-sim:${C}" python /spark/repro.py run --jobs "${E}/eval_${M}-jobs.jsonl" \
    --shard "${i}/${N}" --out "${OUT}" "${LIM[@]}" > "${OUT}/shard-${i}.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
systemctl stop vlar-xvla vlar-pi05
(( rc == 0 )) || fail "a shard failed (${OUT}/shard-*.log)"
echo "episodes: $(cat "${OUT}"/*-results-*.jsonl | wc -l)"
