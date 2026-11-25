#!/usr/bin/env bash
# ops/spark/run.sh STEP...: the local stand-in for ops/vm/phase0/runner.sh on the DGX Spark. Runs the named steps in
# order (e.g. `010 015 030 040 050 090`), each from ops/spark/steps/ if the Spark has its own version (010, 015, 020)
# else from ops/vm/phase0/steps/ or ops/vm/postfreeze/, rewritten by transform.py (VM paths -> folders under
# $VLAR_SPARK_ROOT on the host side) with the shims of shim.py on PATH (a local folder as the bucket, servers and
# workers as named containers, no spend guard, no shutdown). Stops at the first step that does not exit 0 (75 needs
# the user, 76 gate failed, other failed), as the VM's runner does. Run it in tmux: `tmux new -d -s vlar 'ops/spark/run.sh ...'`.
# Every step's log, history entry and results folder carry platform.json (machine, GPU, driver, image digests, commit).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "${HERE}/../.." && pwd)"
export VLAR_SPARK_ROOT="${VLAR_SPARK_ROOT:-$HOME/work/vlar-data/spark}" VLAR_SPARK_REPO="${REPO}"
export VLAR_SPARK_WHEELHOUSE="${VLAR_SPARK_WHEELHOUSE:-$HOME/work/vlar-data/wheelhouse-aarch64}"
export VLAR_SPARK_IMAGES="${VLAR_SPARK_IMAGES:-$HOME/work/vlar-data/etc/image-digests.json}"
export VLAR_SPARK_XVLA_WEIGHTS="${VLAR_SPARK_XVLA_WEIGHTS:-$HOME/work/vlar-data/weights}"
export VLAR_SPARK_HF_HUB="${VLAR_SPARK_HF_HUB:-$HOME/hf_cache/hub}"
SD="${VLAR_SPARK_ROOT}"; RUN="${SD}/run"; export VLAR_SPARK_RUN="${RUN}"
export VLAR_PHASE0_DIR="${SD}/var/phase0" VLAR_STORAGE_URL=gs://local/vla-router VLAR_PHASE0_URL=gs://local/vla-router/phase0
export PATH="${HERE}/shims:${PATH}"
mkdir -p "${SD}"/{etc/calibration,var/bucket/vla-router,var/phase0/logs,weights,hf/hub} "${RUN}"
OPS="${REPO}/ops/ops.yaml"          # git-ignored; the containers see it as /etc/vlar-ops/ops.yaml
[[ -f "${OPS}" ]] || cat > "${OPS}" <<'YAML'
# The Spark's ops config (written by ops/spark/run.sh; paths as the containers see them).
storage_url: /var/lib/vlar/bucket/vla-router
local_root: /var/lib/vlar
golden_marker: /var/lib/vlar/golden-pass.json
YAML
python3 "${HERE}/transform.py" "${REPO}/ops/vm/phase0/lib.sh" "${RUN}/lib.sh"

platform() {
  python3 - "${VLAR_SPARK_IMAGES}" "${REPO}" <<'PY'
import json, platform, subprocess, sys
q = lambda *c: subprocess.run(c, capture_output=True, text=True).stdout.strip()
imgs = json.load(open(sys.argv[1])) if __import__("os").path.exists(sys.argv[1]) else {}
print(json.dumps({"machine": "dgx-spark", "arch": platform.machine(), "kernel": platform.release(),
                  "gpu": q("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"),
                  "image_digests": imgs, "repo_commit": q("git", "-C", sys.argv[2], "rev-parse", "HEAD"),
                  "repo_dirty_paths": q("git", "-C", sys.argv[2], "status", "--porcelain").splitlines()}))
PY
}
PLATFORM="$(platform)"; echo "${PLATFORM}" > "${SD}/var/phase0/logs/platform.json"
for s in "$@"; do
  src="$(ls "${HERE}"/steps/"${s}"-*.sh "${REPO}"/ops/vm/phase0/steps/"${s}"-*.sh "${REPO}"/ops/vm/postfreeze/"${s}"-*.sh 2>/dev/null | head -1)"
  [[ -n "${src}" ]] || { echo "no step ${s}"; exit 2; }
  name="$(basename "${src}" .sh)"; python3 "${HERE}/transform.py" "${src}" "${RUN}/${name}.sh"
  cap="$(grep -m1 -oP '^# cap-seconds: \K[0-9]+' "${src}" || echo 10800)"
  log="${SD}/var/phase0/logs/${name}.log"; t0="$(date -u +%FT%TZ)"
  echo "== ${t0} ${name} (from ${src#${REPO}/}, cap ${cap} s)"
  { echo "platform: ${PLATFORM}"; VLAR_REASON_FILE="${SD}/var/phase0/reason" timeout "${cap}" bash "${RUN}/${name}.sh"; } > "${log}" 2>&1
  rc=$?
  [[ -f "${SD}/etc/images.env" ]] && source "${SD}/etc/images.env"
  res="${SD}/var/phase0/logs/results/${VLAR_IMAGE_COMMIT:-no-images}"; mkdir -p "${res}" && echo "${PLATFORM}" > "${res}/platform.json"
  python3 -c 'import json,sys; print(json.dumps({"step": sys.argv[1], "exit": int(sys.argv[2]), "start": sys.argv[3], "end": sys.argv[4], "platform": json.loads(sys.argv[5])}))' \
    "${name}" "${rc}" "${t0}" "$(date -u +%FT%TZ)" "${PLATFORM}" >> "${SD}/var/phase0/history.jsonl"
  echo "   exit ${rc}; $(tail -n 3 "${log}" | tr '\n' ' ' | cut -c1-300)"
  (( rc == 0 )) || { echo "stopped at ${name} (exit ${rc}; 75 needs the user, 76 gate failed; reason: $(cat "${SD}/var/phase0/reason" 2>/dev/null))"; exit "${rc}"; }
done
