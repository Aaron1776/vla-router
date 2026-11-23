# Sourced by the Phase 0 step scripts (ops/vm/phase0/steps/) that ops/vm/phase0/runner.sh runs on the L4 VM.
# Exit statuses the runner understands: 0 done, 75 needs the user (step stays queued), 76 gate failed, other failed.
set -uo pipefail
R=/opt/vla-router
RESULTS="${VLAR_PHASE0_DIR:-/var/lib/vlar/phase0}/logs/results"     # one <row>.status file per test row
mkdir -p "${RESULTS}"
[[ -f /etc/vlar/images.env ]] && source /etc/vlar/images.env          # VLAR_IMAGE_COMMIT, once the images are loaded
C="${VLAR_IMAGE_COMMIT:-}"

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
