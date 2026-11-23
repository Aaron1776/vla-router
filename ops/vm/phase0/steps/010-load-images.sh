#!/usr/bin/env bash
# cap-seconds: 3600
# Phase 0 step 0 on the L4 (SPEC §11, amendment 28): wait for the NVIDIA driver (the Deep Learning VM image installs
# it on first boot), then ops/vm/load_images.sh: driver 580.65.06+, images against SHA256SUMS and their recorded IDs,
# /etc/vlar provenance and batching files, check_env.py --gpu in each image. Skipped if those images are loaded.
source /opt/vla-router/ops/vm/phase0/lib.sh
: "${VLAR_IMAGES_URL:?gs://<bucket>/containers/<commit>}"
for _ in $(seq 1 80); do nvidia-smi >/dev/null 2>&1 && break; sleep 15; done
nvidia-smi || fail "no working NVIDIA driver after 20 minutes (nvidia-smi)"
# The Deep Learning VM image (common-cu129-ubuntu-2404-nvidia-580) has the driver and the NVIDIA container toolkit but
# not Docker (found 2026-11-22): install Ubuntu's docker.io, as the build VM does, and register the NVIDIA runtime.
command -v nvidia-ctk >/dev/null || need_input "the NVIDIA container toolkit (nvidia-ctk) is not on this VM image"
if ! command -v docker >/dev/null || ! command -v zstd >/dev/null; then
  apt-get update -q && apt-get install -y -q --no-install-recommends docker.io zstd || fail "cannot install docker.io and zstd"
fi
if ! docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
  nvidia-ctk runtime configure --runtime=docker && systemctl restart docker || fail "cannot register the NVIDIA runtime with docker"
fi
docker info 2>/dev/null | grep -i -e 'server version' -e 'runtimes'
want="${VLAR_IMAGES_URL##*/}"
if [[ "${C}" == "${want}" ]] && docker image inspect "vlar-sim:${C}" "vlar-pi05:${C}" "vlar-xvla:${C}" >/dev/null 2>&1; then
  echo "images ${C} already loaded"; exit 0
fi
"${R}/ops/vm/load_images.sh" "${VLAR_IMAGES_URL}" \
  || fail "ops/vm/load_images.sh failed (driver, checksums, image IDs or check_env --gpu): see logs/010-load-images.log"
