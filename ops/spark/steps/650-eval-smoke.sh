#!/usr/bin/env bash
# cap-seconds: 14400
# Smoke test of the Spark's evaluation driver (ops/spark/evaluate.py, live.py, pi05_gate.py, gate_check.py): both
# production servers at the calibrated geometry, the π0.5 gate in front of the π0.5 server (port 8779, one /act in
# flight), the gate's startup check, then a few episodes per method (SMOKE_TASKS, default 2,5 x SMOKE_INITS 0-1) with
# a dummy random decider for the learned path. Frozen calibration (observe() needs it). vlar_train with the history
# hook from VLAR_SPARK_VLAR_TRAIN, mounted over the image's copy. Output: P0D/eval-smoke/.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
S="${P0D}/eval-smoke"; mkdir -p "${S}/obs"
CALF="${CAL_DIR}/calibration.yaml"; VT="${VLAR_SPARK_VLAR_TRAIN:?vlar_train source folder (history hook)}"
EV=(-e VLAR_CALIBRATION="${CALF}" -v "${R}/ops/spark:/spark:ro" -v "${VT}:/opt/vla-router/src/vlar_train:ro")
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
[[ -s "${S}/obs/pi05_requests.json" ]] || docker run "${SIM_RUN[@]}" -e VLAR_CALIBRATION="${CALF}" "vlar-sim:${C}" \
  vlar-ops step2 capture --out "${S}/obs" >/dev/null || { stop_all; fail "request capture failed"; }
docker run "${GPU_RUN[@]}" -v "${S}:${S}" -v "${R}/ops/spark:/spark:ro" "vlar-pi05:${C}" python /spark/gate_check.py \
  --obs "${S}/obs" --max-batch "${VLAR_MAX_BATCH}" --out "${S}/gate-check.json" | tail -n 1 \
  || { stop_all; fail "gate startup check failed (${S}/gate-check.json)"; }
T="${SMOKE_TASKS:-2,5}"; I="${SMOKE_INITS:-0-1}"
for m in "xvla_only" "pi05_only" "fixed --spacing 5" "random --rate 0.2" "decider --decider dummy:router" \
         "decider --decider dummy:cascader_nospread --tag nospread"; do
  for t in ${T//,/ }; do
    docker run "${SIM_RUN[@]}" "${EV[@]}" "vlar-sim:${C}" python /spark/evaluate.py plan --method ${m} \
      --tasks "${t}" --inits "${I}" --out "${S}/plans/t${t}" | tail -n 1 | cut -c1-160
  done
done
for f in "${S}"/plans/t*/eval_*-jobs.jsonl; do
  docker run "${SIM_RUN[@]}" "${EV[@]}" -e VLAR_SPARK_PLATFORM="${P0D}/logs/platform.json" "vlar-sim:${C}" \
    python /spark/evaluate.py run --jobs "${f}" --out "${S}/runs" --pi05-port 8779 2>&1 | grep -E '^eval_|Error|error' | tail -n 6
done
stop_all
python3 - "${S}" <<'PY'
import glob, json, sys
rows = [json.loads(l) for f in glob.glob(sys.argv[1] + "/runs/*-results-*.jsonl") for l in open(f)]
by = {}
for x in rows:
    m = x["run_id"].split("/")[0]
    b = by.setdefault(m, {"episodes": 0, "success": 0, "pi05_steps": 0, "escalations": 0, "gate": set()})
    b["episodes"] += 1; b["success"] += bool(x["run"]["success"]); b["pi05_steps"] += int(x["run"]["pi05_steps"] or 0)
    b["escalations"] += sum(d["choice"] == "escalate" for d in x["decisions"]); b["gate"].add(json.dumps(x["pi05_gate"], sort_keys=True))
for m, b in sorted(by.items()):
    print(m, {**b, "gate": sorted(b["gate"])})
log = [json.loads(l) for l in open(sys.argv[1] + "/gate.jsonl")]
print("gate log: /act", len(log), "max in flight on arrival", max((r["in_flight_on_arrival"] for r in log), default=0),
      "non-200", sum(r["status"] != 200 for r in log))
PY
