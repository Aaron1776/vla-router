# Sourced by the Phase 0 step scripts (ops/vm/phase0/steps/) that ops/vm/phase0/runner.sh runs on the L4 VM.
# Exit statuses the runner understands: 0 done, 75 needs the user (step stays queued), 76 gate failed, other failed.
set -uo pipefail
R=/opt/vla-router
P0D="${VLAR_PHASE0_DIR:-/var/lib/vlar/phase0}"
[[ -f /etc/vlar/images.env ]] && source /etc/vlar/images.env          # VLAR_IMAGE_COMMIT, once the images are loaded
C="${VLAR_IMAGE_COMMIT:-}"
RESULTS="${P0D}/logs/results/${C:-no-images}"   # one <row>.status file per test row, per image commit
mkdir -p "${RESULTS}"
# The calibration file (amendments 34, 35): read-only in every container through the /etc/vlar mount; written only
# by `calfile write` (the image's own writer, the only container that mounts it read-write); synced to the bucket.
CAL_DIR=/etc/vlar/calibration
CAL_ENV=(-e VLAR_CALIBRATION="${CAL_DIR}/calibration.yaml")

reason() { echo "$*" > "${VLAR_REASON_FILE:-/dev/null}"; echo "$*"; }
fail() { reason "$*"; exit 1; }
need_input() { reason "$*"; exit 75; }
gate_fail() { reason "$*"; exit 76; }

need_images() {
  [[ -n "${C}" ]] || fail "the images are not loaded (no /etc/vlar/images.env): run 010-load-images first"
  local t; for t in sim pi05 xvla; do
    docker image inspect "vlar-${t}:${C}" >/dev/null 2>&1 || fail "image vlar-${t}:${C} is not loaded"
  done
}

have_tokenizer() { [[ -d /opt/hf/hub/models--google--paligemma-3b-pt-224 ]]; }
fetch_tokenizer() {  # the gated tokenizer's cache, if it has reached the bucket (<storage>/hf-tokenizer/)
  have_tokenizer && return 0
  gcloud storage ls "${VLAR_STORAGE_URL}/hf-tokenizer/models--google--paligemma-3b-pt-224/" >/dev/null 2>&1 || return 1
  mkdir -p /opt/hf/hub && gcloud storage rsync -r "${VLAR_STORAGE_URL}/hf-tokenizer" /opt/hf/hub && have_tokenizer
}

# Container options for the GPU test rows (docs/HARNESS.md, VM re-run list): EGL rendering, CUDA, the weights and
# the tokenizer cache read-only, provenance from /etc/vlar.
GPU_RUN=(--rm --gpus all --network host -v /etc/vlar:/etc/vlar:ro -v /opt/weights:/weights:ro -v /opt/hf:/hf:ro
         -e HF_HOME=/hf -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e VLAR_TEST_DEVICE=cuda
         -e VLAR_XVLA_MODEL_PATH=/weights/X-VLA-Pt -e VLAR_XVLA_LORA_PATH=/weights/X-VLA-libero-long-peft
         -e VLAR_PI05_CHECKPOINT=/weights/pi05_libero_finetuned)

# row <name> <command...>: run one test row, keep going whatever its result, record passed/failed with the last
# lines of its output; the verdict step (090) reads them.
row() {
  local name="$1"; shift
  local log="${RESULTS}/${name}.log" t0=$(date +%s)
  echo "== row ${name}: $*"
  if "$@" > "${log}" 2>&1; then echo passed > "${RESULTS}/${name}.status"
  else echo "failed (exit $?)" > "${RESULTS}/${name}.status"; fi
  echo "   $(cat "${RESULTS}/${name}.status") in $(( $(date +%s) - t0 )) s; $(tail -n 1 "${log}")"
}

not_run() { echo "not run: $2" > "${RESULTS}/$1.status"; echo "== row $1 not run: $2"; }

calfile() {      # calfile <command> [args]: python -m vlar_harness.calfile in the sim image
  mkdir -p "${CAL_DIR}"
  docker run --rm -v "${CAL_DIR}:${CAL_DIR}" -v "${P0D}:${P0D}" "${CAL_ENV[@]}" "vlar-sim:${C}" \
    python -m vlar_harness.calfile "$@"
}
sync_calibration() {
  gcloud storage rsync -r "${CAL_DIR}" "${VLAR_STORAGE_URL}/calibration" >/dev/null || fail "calibration not synced to the bucket"
}
write_batching_env() {   # the servers' geometry from the write-once group
  calfile batching-env > /tmp/vlar-batching.env || fail "no write-once calibration: Phase 0 step 2 first"
  install -m 0644 /tmp/vlar-batching.env /etc/vlar/batching.env && rm -f /tmp/vlar-batching.env
}
SIM_RUN=(--rm --gpus all --network host -v /etc/vlar:/etc/vlar:ro -v "${P0D}:${P0D}" -e MUJOCO_GL=egl
         -e PYOPENGL_PLATFORM=egl)

wait_servers() {   # both production servers answer /info (they load the weights first)
  local _; for _ in $(seq 1 60); do
    curl -sf http://127.0.0.1:8777/info >/dev/null && curl -sf http://127.0.0.1:8778/info >/dev/null && return 0
    sleep 10; done
  return 1
}

# run_phase <phase> <worker args>: the production servers and the calibrated number of workers until no job of the
# phase is left; up to three rounds (a worker that died leaves its lease, which `queue resume` releases).
run_phase() {
  local phase="$1" args="$2" n round counts
  n="$(calfile get servers.workers)"; [[ "${n}" =~ ^[0-9]+$ ]] || fail "no worker count in the calibration file"
  [[ -s /etc/vlar/batching.env ]] || write_batching_env
  printf 'VLAR_WORKER_ARGS=%s\n' "${args}" > /etc/vlar/worker.env
  systemctl start vlar-xvla vlar-pi05 || fail "model servers did not start"
  wait_servers || { systemctl stop vlar-xvla vlar-pi05; fail "model servers not answering /info after 10 minutes"; }
  for round in 1 2 3; do
    vlar-ops queue resume >/dev/null
    for i in $(seq 1 "${n}"); do systemctl start "vlar-worker@${i}"; done
    while systemctl list-units 'vlar-worker@*' --state=active --no-legend | grep -q .; do
      sleep 120; echo "$(date -u +%FT%TZ) ${phase}: $(vlar-ops queue status --phase "${phase}" | tail -n 1)"
    done
    vlar-ops queue sync >/dev/null
    counts="$(vlar-ops queue status --phase "${phase}" | tail -n 1)"
    python3 -c 'import json,sys; c=json.loads(sys.argv[1]); sys.exit(0 if c["pending"]+c["leased"]+c["done_local"]==0 else 1)' "${counts}" && break
  done
  systemctl stop vlar-xvla vlar-pi05
  echo "${phase}: ${counts}"
  python3 -c 'import json,sys; c=json.loads(sys.argv[1]); sys.exit(0 if c["pending"]+c["leased"]+c["done_local"]+c["failed"]==0 else 1)' "${counts}" \
    || fail "${phase} incomplete after 3 rounds: ${counts}"
}
