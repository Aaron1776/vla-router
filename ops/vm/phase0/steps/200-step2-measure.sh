#!/usr/bin/env bash
# cap-seconds: 10800
# Phase 0 step 2 on the L4 (SPEC §11; amendments 14, 24, 25, 35), before any π0.5-calling run. Writes the calibration
# file's write-once group (π0.5 latency, upload size, batch geometry: immutable afterwards) and the rest of step 2
# (X-VLA latencies, worker count, gripper thresholds), then /etc/vlar/batching.env, and syncs both to the bucket.
#   1. capture real requests from the simulator;  2. GPU time per batch at sizes 1-16, deterministic on and off, in
#   each model's image;  2b. the joint memory at each size: both production servers at that geometry, the planned
#   2 x B workers rendering, full batches to both, the card's peak from nvidia-smi (approved 2026-11-23);
#   3. the geometry rule (at least 2 GB free at the joint peak; exit 75 if fixed deterministic batching is too slow:
#   amendment 24);
#   4. servers unpadded (max_batch 1, X-VLA sample_rows K): the clock charges and the upload size;  5. gripper widths.
# Every sub-result is kept in step2/<image commit>/ (bucket: phase0/step2/), so a re-run after a preemption resumes.
# If the write-once group is already set, nothing is measured again (it cannot change).
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
S="${P0D}/step2/${C}"; mkdir -p "${S}"
sync_step2() { gcloud storage rsync -r "${S}" "${VLAR_PHASE0_URL}/step2/${C}" >/dev/null 2>&1 || echo "warning: step2 not synced"; }
if calfile require-write-once >/dev/null 2>&1; then
  echo "write-once group already set (amendment 35):"; calfile show
  write_batching_env; sync_calibration; exit 0
fi
fetch_tokenizer || need_input "the gated PaliGemma tokenizer is not on the VM or in the bucket (π0.5 needs it)"
SIM=(docker run "${SIM_RUN[@]}" "vlar-sim:${C}" vlar-ops step2)
[[ -s "${S}/obs/pi05_requests.json" ]] || "${SIM[@]}" capture --out "${S}/obs" || fail "step2 capture failed"
for m in xvla pi05; do
  [[ -s "${S}/bench-${m}.json" ]] && continue
  docker run "${GPU_RUN[@]}" -v "${S}:${S}" "vlar-${m}:${C}" python -m vlar_harness.servers.bench "${m}" \
    --obs "${S}/obs" --out "${S}/bench-${m}.json" || fail "bench ${m} failed"
done
K="$(docker run --rm "vlar-sim:${C}" python -c 'from vlar_harness.config import load_config; print(load_config(calibration=None)["measurement"]["K"])')"

# 2b. Joint memory at each size both benches ran (memory-<B>.json; vlar_ops/step2.py: render, load, memory).
stop_joint() { docker rm -f vlar-xvla-mem vlar-pi05-mem $(docker ps -aq --filter name=vlar-render-) >/dev/null 2>&1; }
up() { [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == true ]]; }
servers_ready() {   # both answer /info; gives up early if a server container has exited (e.g. out of memory)
  local _; for _ in $(seq 1 60); do
    up vlar-xvla-mem && up vlar-pi05-mem || return 1
    curl -sf http://127.0.0.1:8777/info >/dev/null && curl -sf http://127.0.0.1:8778/info >/dev/null && return 0
    sleep 10; done
  return 1
}
SIZES="$(python3 -c 'import json, sys
a, b = (json.load(open(f))["sizes"] for f in sys.argv[1:3])
print(" ".join(sorted((s for s in a if s in b and "error" not in a[s] and "error" not in b[s]), key=int)))' \
  "${S}/bench-xvla.json" "${S}/bench-pi05.json")"
for B in ${SIZES}; do
  [[ -s "${S}/memory-${B}.json" ]] && continue
  W=$(( 2 * B < 2 ? 2 : (2 * B > 12 ? 12 : 2 * B) )); err=""
  stop_joint; rm -f "${S}/memory-${B}.smi"
  docker run -d --name vlar-xvla-mem --gpus all --network host -v /opt/weights:/weights:ro "vlar-xvla:${C}" \
    python -m vlar_harness.servers.xvla_server --model-path /weights/X-VLA-Pt \
    --lora-path /weights/X-VLA-libero-long-peft --host 127.0.0.1 --port 8777 --max-batch "${B}" \
    --sample-rows "$(( B * K ))" >/dev/null
  docker run -d --name vlar-pi05-mem --gpus all --network host -v /opt/weights:/weights:ro -v /opt/hf:/hf:ro \
    -e HF_HOME=/hf "vlar-pi05:${C}" python -m vlar_harness.servers.pi05_server \
    --checkpoint /weights/pi05_libero_finetuned --host 127.0.0.1 --port 8778 --max-batch "${B}" >/dev/null
  if servers_ready; then
    for i in $(seq 1 "${W}"); do
      docker run -d --name "vlar-render-${i}" "${SIM_RUN[@]}" "vlar-sim:${C}" vlar-ops step2 render \
        --task $(( (i - 1) % 10 )) --seconds 600 --out /tmp/render.json >/dev/null
    done
    for _ in $(seq 1 36); do                    # every renderer has its first frame out (at most 6 minutes)
      n=0; for i in $(seq 1 "${W}"); do docker logs "vlar-render-${i}" 2>&1 | grep -q '^rendering' && n=$((n + 1)); done
      (( n == W )) && break; sleep 10
    done
    (( n == W )) || err="only ${n} of ${W} renderers started"
    if [[ -z "${err}" ]]; then
      nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -lms 250 > "${S}/memory-${B}.smi" &
      SMI=$!
      "${SIM[@]}" load --obs "${S}/obs" --size "${B}" --seconds 120 --out "${S}/load-${B}.json" \
        || err="full batches failed at size ${B} (a server may have run out of memory)"
      kill "${SMI}"; wait "${SMI}" 2>/dev/null
      n="$(docker ps -q --filter name=vlar-render- | wc -l)"
      (( n == W )) || err="${err:+${err}; }only ${n} of ${W} renderers still running at the end"
      up vlar-xvla-mem && up vlar-pi05-mem || err="${err:+${err}; }a server exited during the measurement"
    fi
  else
    err="servers at size ${B} did not come up (exited or not answering /info)"
  fi
  stop_joint
  vlar-ops step2 memory --smi "${S}/memory-${B}.smi" --size "${B}" --workers "${W}" ${err:+--error "${err}"} \
    --out "${S}/memory-${B}.json" || fail "joint memory result at size ${B} not written"
  sync_step2
done
"${SIM[@]}" choose --xvla "${S}/bench-xvla.json" --pi05 "${S}/bench-pi05.json" --memory-dir "${S}" \
  --out "${S}/geometry.json"; rc=$?
sync_step2
(( rc == 75 )) && need_input "fixed deterministic batching too slow (amendment 24): numbers in phase0/step2/${C}/geometry.json"
(( rc == 0 )) || fail "geometry rule failed (exit ${rc})"

# Unpadded servers: the clock charges (amendment 25) and the upload size (amendment 14).
stop_unpadded() { docker rm -f vlar-xvla-unpadded vlar-pi05-unpadded >/dev/null 2>&1; }
trap stop_unpadded EXIT
if [[ ! -s "${S}/latency.json" || ! -s "${S}/upload.json" ]]; then
  docker run -d --name vlar-xvla-unpadded --gpus all --network host -v /opt/weights:/weights:ro "vlar-xvla:${C}" \
    python -m vlar_harness.servers.xvla_server --model-path /weights/X-VLA-Pt \
    --lora-path /weights/X-VLA-libero-long-peft --host 127.0.0.1 --port 8777 --max-batch 1 --sample-rows "${K}" \
    || fail "unpadded X-VLA server did not start"
  docker run -d --name vlar-pi05-unpadded --gpus all --network host -v /opt/weights:/weights:ro -v /opt/hf:/hf:ro \
    -e HF_HOME=/hf "vlar-pi05:${C}" python -m vlar_harness.servers.pi05_server \
    --checkpoint /weights/pi05_libero_finetuned --host 127.0.0.1 --port 8778 --max-batch 1 \
    || fail "unpadded π0.5 server did not start"
  wait_servers || fail "unpadded servers not answering /info after 10 minutes"
  [[ -s "${S}/latency.json" ]] || "${SIM[@]}" latency --obs "${S}/obs" --out "${S}/latency.json" || fail "latency failed"
  [[ -s "${S}/upload.json" ]] || "${SIM[@]}" upload --out "${S}/upload.json" || fail "upload size failed"
  stop_unpadded
fi
[[ -s "${S}/gripper.json" ]] || "${SIM[@]}" gripper --out "${S}/gripper.json" || fail "gripper thresholds failed"
"${SIM[@]}" updates --dir "${S}" --out "${S}/updates.json" || fail "calibration update not assembled"
calfile write "${S}/updates.json" --note "Phase 0 step 2 on images ${C}" || fail "calibration write refused"
write_batching_env
sync_calibration; sync_step2
calfile show
