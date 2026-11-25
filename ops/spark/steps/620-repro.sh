#!/usr/bin/env bash
# cap-seconds: 43200
# The Spark's reproduction of the L4's Phase 0 (not a VM step; ops/spark/repro.py, ops/spark/README.md):
#   REPRO_PHASE   pi05_parity (100 episodes) or base (500)
#   REPRO_SHARDS  parallel simulator processes; default 1 for pi05_parity (π0.5 at max_batch 4 is not batch-invariant
#                 on the GB10, step 210, so one client keeps every request alone in its batch), 8 for base (X-VLA
#                 passed 210 at the calibrated geometry)
#   REPRO_LIMIT   episodes per shard (a smoke run: no compare);  REPRO_OUT  output folder name (default the phase)
# Every episode runs on a fresh simulator brought to the L4 run's reconstructed setup count, under the calibration the
# L4 runs were recorded with (aa38aac7), against both production servers at the calibrated geometry. Then the paired
# compare against the L4's per-episode results (exit 76 if the pass criterion fails).
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
PH="${REPRO_PHASE:?pi05_parity or base}"
N="${REPRO_SHARDS:-$([[ "${PH}" == base ]] && echo 8 || echo 1)}"
RP="${P0D}/repro"; OUT="${RP}/${REPRO_OUT:-${PH}}"; mkdir -p "${OUT}"
CSV="${RP}/$([[ "${PH}" == base ]] && echo base-runs.csv || echo pi05-parity-runs.csv)"
[[ -s "${RP}/${PH}-jobs.jsonl" && -s "${CSV}" ]] || need_input "run repro.py plan into ${RP} first"
CALV="${CAL_DIR}/versions/aa38aac7fb61850e3550a256885fe634f48810aaeeb7f182f00b9552cf5e3890.yaml"
[[ -s "${CALV}" ]] || need_input "no calibration version aa38aac7 in ${CAL_DIR}/versions"
OPT=(); [[ "${PH}" == pi05_parity ]] && OPT=(--option scene_seed=pi05_parity)
LIM=(); [[ -n "${REPRO_LIMIT:-}" ]] && LIM=(--limit "${REPRO_LIMIT}")
[[ -s /etc/vlar/batching.env ]] || write_batching_env
systemctl start vlar-xvla vlar-pi05 || fail "model servers did not start"
wait_servers || { systemctl stop vlar-xvla vlar-pi05; fail "model servers not answering /info after 10 minutes"; }
pids=()
for i in $(seq 0 $((N - 1))); do
  docker run "${SIM_RUN[@]}" -e VLAR_CALIBRATION="${CALV}" -e VLAR_SPARK_PLATFORM="${P0D}/logs/platform.json" \
    -v "${R}/ops/spark:/spark:ro" "vlar-sim:${C}" python /spark/repro.py run --jobs "${RP}/${PH}-jobs.jsonl" \
    --shard "${i}/${N}" --out "${OUT}" "${LIM[@]}" "${OPT[@]}" > "${OUT}/shard-${i}.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
systemctl stop vlar-xvla vlar-pi05
(( rc == 0 )) || fail "a shard failed (${OUT}/shard-*.log)"
if [[ -n "${REPRO_LIMIT:-}" ]]; then echo "smoke run, no compare: $(cat "${OUT}"/*-results-*.jsonl | wc -l) episodes"; exit 0; fi
docker run --rm -v "${RP}:${RP}" -v "${R}/ops/spark:/spark:ro" "vlar-sim:${C}" python /spark/repro.py compare \
  --phase "${PH}" --runs "${CSV}" --results "${OUT}" || gate_fail "reproduction of ${PH} failed its criterion"
