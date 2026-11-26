#!/usr/bin/env bash
# cap-seconds: 14400
# Branch jobs on the Spark (ops/spark/branches.py): the planner's b1 and b2 jobs at a stratified draw of the Spark's
# own base decision points (P0D/repro/base), π0.5 through the gate (port 8779, one /act in flight), frozen calibration.
#   BR_DIR       output folder under the Spark root (plan/, branches/, parent-check/, dataset/)
#   BR_PHASE     the branch jobs' phase name (in every job, run id folder and records path)
#   BR_PER_TASK  points drawn per task (default 4, two thirds from failed parents);  BR_SEED  the draw's seed
#   BR_SHARDS    parallel simulator processes (default 8);  BR_LIMIT  jobs per shard (a smoke run: no dataset)
#   BR_PARENT_CHECK=1  also run b1 with seed index 1 at every drawn point (outside the plan and the dataset)
#   BR_NOTE      the dataset README's first line
# Start-up as step 650: both production servers at the calibrated geometry, the gate, its /info and startup checks.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
D="${BR_DIR:?output folder}"; PH="${BR_PHASE:?phase name}"; N="${BR_SHARDS:-8}"
BASE="${P0D}/repro/base"; PD="${D}/plan"; mkdir -p "${D}/obs" "${D}/branches" "${D}/parent-check"
CALF="${CAL_DIR}/calibration.yaml"
[[ -s "${CALF}" ]] || need_input "no frozen calibration file in ${CAL_DIR}"
[[ -d "${BASE}/records/base" ]] || need_input "no base reproduction records in ${BASE}"
EV=(-e VLAR_CALIBRATION="${CALF}" -e VLAR_SPARK_PLATFORM="${P0D}/logs/platform.json" -v "${R}/ops/spark:/spark:ro" -v "${D}:${D}")
stop_all() { docker rm -f vlar-pi05-gate >/dev/null 2>&1; systemctl stop vlar-xvla vlar-pi05; }
if [[ ! -s "${PD}/plan.json" ]]; then
  SEED=(); [[ -n "${BR_SEED:-}" ]] && SEED=(--seed "${BR_SEED}")
  docker run "${SIM_RUN[@]}" "${EV[@]}" "vlar-sim:${C}" python /spark/branches.py plan --base "${BASE}" --out "${PD}" \
    --phase "${PH}" --per-task "${BR_PER_TASK:-4}" "${SEED[@]}" | tail -n 1 || fail "branch plan failed"
fi
[[ -s /etc/vlar/batching.env ]] || write_batching_env
source /etc/vlar/batching.env
systemctl start vlar-xvla vlar-pi05 || fail "model servers did not start"
wait_servers || { stop_all; fail "model servers not answering /info after 10 minutes"; }
docker run -d --rm --name vlar-pi05-gate --network host -v "${R}/ops/spark:/spark:ro" -v "${D}:${D}" \
  vlar-aarch64-base python3 /spark/pi05_gate.py --listen 8779 --backend 127.0.0.1:8778 --log "${D}/gate.jsonl" >/dev/null
for _ in $(seq 1 30); do curl -sf http://127.0.0.1:8779/info | grep -q '"max_in_flight": 1' && break; sleep 2; done
curl -sf http://127.0.0.1:8779/info | python3 -c 'import json,sys; g=json.load(sys.stdin)["gate"]; print("gate:", g); sys.exit(0 if g["mode"]=="gate" and g["max_in_flight"]==1 else 1)' \
  || { stop_all; fail "the π0.5 gate is not on"; }
[[ -s "${D}/obs/pi05_requests.json" ]] || docker run "${SIM_RUN[@]}" "${EV[@]}" "vlar-sim:${C}" \
  vlar-ops step2 capture --out "${D}/obs" >/dev/null || { stop_all; fail "request capture failed"; }
docker run "${GPU_RUN[@]}" -v "${D}:${D}" -v "${R}/ops/spark:/spark:ro" "vlar-pi05:${C}" python /spark/gate_check.py \
  --obs "${D}/obs" --max-batch "${VLAR_MAX_BATCH}" --out "${D}/gate-check.json" | tail -n 1 \
  || { stop_all; fail "gate startup check failed (${D}/gate-check.json)"; }
LIM=(); [[ -n "${BR_LIMIT:-}" ]] && LIM=(--limit "${BR_LIMIT}")
shards() {   # shards PLAN OUT
  local pids=() rc=0 i
  for i in $(seq 0 $((N - 1))); do
    docker run "${SIM_RUN[@]}" "${EV[@]}" "vlar-sim:${C}" python /spark/branches.py run --plan "$1" --base "${BASE}" \
      --out "$2" --shard "${i}/${N}" --pi05-port 8779 "${LIM[@]}" > "$2/shard-${i}.log" 2>&1 &
    pids+=($!)
  done
  for i in "${pids[@]}"; do wait "${i}" || rc=1; done
  return "${rc}"
}
shards "${PD}/plan.json" "${D}/branches" || { stop_all; fail "a branch shard failed (${D}/branches/shard-*.log)"; }
if [[ "${BR_PARENT_CHECK:-0}" == 1 ]]; then
  shards "${PD}/parent-check.json" "${D}/parent-check" || { stop_all; fail "a parent-check shard failed (${D}/parent-check/shard-*.log)"; }
fi
stop_all
python3 - "${D}" <<'PY'
import glob, json, sys
for name in ("branches", "parent-check"):
    rows = [json.loads(l) for f in glob.glob(f"{sys.argv[1]}/{name}/results-*.jsonl") for l in open(f)]
    by = {}
    for x in rows:
        b = by.setdefault(x["run"]["branch"], {"runs": 0, "success": 0, "pi05_requests": 0})
        b["runs"] += 1; b["success"] += bool(x["run"]["success"]); b["pi05_requests"] += int(x["run"]["pi05_requests"] or 0)
    print(name, len(rows), "runs", by, "gates", sorted({json.dumps(x["pi05_gate"], sort_keys=True) for x in rows}))
log = [json.loads(l) for l in open(sys.argv[1] + "/gate.jsonl")]
print("gate log: /act", len(log), "max in flight on arrival", max((r["in_flight_on_arrival"] for r in log), default=0),
      "non-200", sum(r["status"] != 200 for r in log))
PY
if [[ -n "${BR_LIMIT:-}" ]]; then echo "smoke run (BR_LIMIT=${BR_LIMIT}): no dataset assembled"; exit 0; fi
docker run --rm -v "${P0D}:${P0D}" -v "${D}:${D}" -v "${R}/ops/spark:/spark:ro" vlar-aarch64-base python3 /spark/branches.py \
  assemble --base "${BASE}" --runs "${D}/branches" --plan-dir "${PD}" --out "${D}/dataset" \
  --note "${BR_NOTE:-Branch dataset from the Spark (phase ${PH})}" || fail "dataset assembly failed"
