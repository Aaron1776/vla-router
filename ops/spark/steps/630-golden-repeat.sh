#!/usr/bin/env bash
# cap-seconds: 14400
# The Spark golden candidates' exact-replay rate (not a VM step; ops/spark/golden_spark.py): golden.select over the
# Spark's base reproduction with tasks GOLDEN_EXCLUDE (default 3,8) left out, then GOLDEN_REPEATS (default 10)
# golden.check passes, each in a fresh container, against both production servers at the calibrated geometry, under
# the calibration the reproduction used (aa38aac7). The servers start once and stay up across the repeats (as the golden
# check runs against already-running servers); each repeat is a new container (--pid=host, so the recorded pid is the
# host's), with a fresh Python process and fresh simulators. Results: GOLDEN_RESULTS (default repeats.jsonl).
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
RP="${P0D}/repro"; REC="${RP}/base/records"; G="${RP}/golden-spark"; mkdir -p "${G}"
CALV="${CAL_DIR}/versions/aa38aac7fb61850e3550a256885fe634f48810aaeeb7f182f00b9552cf5e3890.yaml"
GR=(docker run "${SIM_RUN[@]}" -e VLAR_CALIBRATION="${CALV}" -v "${R}/ops/spark:/spark:ro" "vlar-sim:${C}" python /spark/golden_spark.py)
[[ -s "${G}/golden.json" ]] || "${GR[@]}" select --records "${REC}" --out "${G}" --exclude-tasks "${GOLDEN_EXCLUDE:-3,8}" \
  || fail "golden select failed"
[[ -s /etc/vlar/batching.env ]] || write_batching_env
systemctl start vlar-xvla vlar-pi05 || fail "model servers did not start"
wait_servers || { systemctl stop vlar-xvla vlar-pi05; fail "model servers not answering /info after 10 minutes"; }
for k in $(seq 1 "${GOLDEN_REPEATS:-10}"); do
  docker run "${SIM_RUN[@]}" --pid=host -e VLAR_CALIBRATION="${CALV}" -v "${R}/ops/spark:/spark:ro" "vlar-sim:${C}" \
    python /spark/golden_spark.py check --records "${REC}" --golden "${G}" --repeat "${k}" \
    --results-file "${GOLDEN_RESULTS:-repeats.jsonl}" | tail -n 1 | cut -c1-300
done
systemctl stop vlar-xvla vlar-pi05
python3 - "${G}/${GOLDEN_RESULTS:-repeats.jsonl}" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
problems = [r["problem"] for x in rows for r in x["results"] if r["problem"]]
print(f"golden passes: {sum(x['passed'] for x in rows)}/{len(rows)}; distinct pids {len({x['pid'] for x in rows})}; "
      f"distinct containers {len({x.get('container') for x in rows})}; problems: {problems or 'none'}")
PY
