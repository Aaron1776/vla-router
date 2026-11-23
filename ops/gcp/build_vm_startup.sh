#!/usr/bin/env bash
# Startup script of the CPU build VM (ops/gcp/35_build_images.sh; runs as root on Ubuntu 24.04, once).
# Builds the three images from the commit's source archive, runs the in-container tests that need no GPU
# (rendering with OSMesa instead of EGL), exports the images and writes DONE, or FAILED with the reason.
set -uo pipefail
md() { curl -fs -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
PREFIX="$(md vlar-prefix)"; COMMIT="$(md vlar-commit)"
W=/var/tmp/vlar-build
[[ -e "${W}/started" ]] && exit 0          # a reboot must not start a second build
mkdir -p "${W}" && touch "${W}/started" && cd "${W}"
exec > >(tee -a "${W}/build.log") 2>&1
upload_logs() { gcloud storage cp "${W}"/*.log "${PREFIX}/" || true; }
fail() { echo "FAILED: $*"; upload_logs; echo "$*" | gcloud storage cp - "${PREFIX}/FAILED"; shutdown -h now; exit 1; }

# The CPU first (amendment 33): x86 CPUs take different code paths in PyTorch's CPU math libraries, so the in-container
# tests' bit-level results depend on it. lscpu prints the model name and the instruction-set flags.
echo "== build VM CPU"; lscpu || cat /proc/cpuinfo; echo "== end of CPU"

command -v gcloud >/dev/null || fail "gcloud is not on the build VM image"
apt-get update && apt-get install -y --no-install-recommends docker.io docker-buildx zstd || fail "apt install"
gcloud storage cp "${PREFIX}/src.tar.gz" . && mkdir -p src && tar -xzf src.tar.gz -C src || fail "source archive"
cd src
for t in sim pi05 xvla; do
  DOCKER_BUILDKIT=1 docker build -f containers/Dockerfile --target "${t}" --build-arg GIT_COMMIT="${COMMIT}" \
    -t "vlar-${t}:${COMMIT}" . || fail "docker build ${t}"
done
for t in sim pi05 xvla; do      # the code path PyTorch's CPU kernels dispatch to here (e.g. AVX2, AVX512)
  docker run --rm "vlar-${t}:${COMMIT}" python -c \
    "import torch; print('torch CPU capability (${t}):', torch.backends.cpu.get_cpu_capability())" || true
done

# Small Hugging Face config files the CPU model tests use (no weights), at the pinned revisions.
docker run --rm -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 -v "${W}/hf:/hf-cfg" "vlar-pi05:${COMMIT}" python -c "
from huggingface_hub import snapshot_download as d
from vlar_harness.config import load_config
m = load_config()['models']
d(m['xvla']['base'], revision=m['xvla']['base_revision'], local_dir='/hf-cfg/X-VLA-Pt',
  allow_patterns=['config.json', 'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json'])
d(m['pi05']['checkpoint'], revision=m['pi05']['revision'], local_dir='/hf-cfg/pi05_libero_finetuned',
  allow_patterns=['config.json', 'policy_preprocessor.json', 'policy_postprocessor.json'])
" || fail "Hugging Face config download"

status=0
cpu=(-e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa -e VLAR_TEST_DEVICE=cpu -v "${W}/hf:/hf-cfg:ro"
     -e VLAR_XVLA_CONFIG_DIR=/hf-cfg/X-VLA-Pt -e VLAR_PI05_CONFIG_DIR=/hf-cfg/pi05_libero_finetuned)
docker run --rm "${cpu[@]}" "vlar-sim:${COMMIT}" python -m pytest -q -rs tests > "${W}/test-sim.log" 2>&1 || status=1
docker run --rm "${cpu[@]}" "vlar-pi05:${COMMIT}" python -m pytest -q -rs \
  tests/reference/test_batch_invariance.py -k pi05 > "${W}/test-pi05.log" 2>&1 || status=1
# X-VLA's tests need starlette's test client (httpx2, containers/xvla-test.lock), installed with hashes into this
# throwaway container only: the exported image keeps exactly xvla.lock.
docker run --rm "${cpu[@]}" "vlar-xvla:${COMMIT}" sh -c \
  "pip install -q --require-hashes --no-deps --only-binary :all: -r containers/xvla-test.lock && \
   python -m pytest -q -rs tests/reference/test_model_servers.py tests/reference/test_batch_invariance.py -k xvla" \
  > "${W}/test-xvla.log" 2>&1 || status=1
tail -n 3 "${W}"/test-*.log
(( status == 0 )) || fail "in-container tests (see test-*.log)"

cd "${W}"
for t in sim pi05 xvla; do
  docker save "vlar-${t}:${COMMIT}" | zstd -T0 -6 -q -o "vlar-${t}.tar.zst" || fail "export ${t}"
done
printf '{"sim": "%s", "pi05": "%s", "xvla": "%s", "commit": "%s"}\n' \
  "$(docker image inspect --format '{{.Id}}' "vlar-sim:${COMMIT}")" \
  "$(docker image inspect --format '{{.Id}}' "vlar-pi05:${COMMIT}")" \
  "$(docker image inspect --format '{{.Id}}' "vlar-xvla:${COMMIT}")" "${COMMIT}" > image-digests.json
sha256sum vlar-*.tar.zst image-digests.json > SHA256SUMS
gcloud storage cp vlar-*.tar.zst image-digests.json SHA256SUMS "${PREFIX}/" || fail "upload"
upload_logs
echo ok | gcloud storage cp - "${PREFIX}/DONE"
shutdown -h now
