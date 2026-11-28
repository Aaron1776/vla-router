#!/usr/bin/env bash
# Phase 0, first step on the L4 VM (docs/HARNESS.md, VM re-run list; amendment 28): check the NVIDIA driver is new
# enough for CUDA 13.0, load the three images from the bucket, check them against SHA256SUMS and their image IDs,
# record the IDs for provenance (/etc/vlar/image-digests.json; mount /etc/vlar read-only into every container), and
# run each image's environment check with the GPU.
#   ops/vm/load_images.sh gs://<bucket>/containers/<commit>
set -euo pipefail
PREFIX="${1:?gs://<bucket>/containers/<commit>}"
MIN_DRIVER="580.65.06"                               # CUDA 13.0's minimum Linux driver
drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
[[ "$(printf '%s\n%s\n' "${MIN_DRIVER}" "${drv}" | sort -V | head -n 1)" == "${MIN_DRIVER}" ]] \
  || { echo "NVIDIA driver ${drv} is older than ${MIN_DRIVER} (CUDA 13.0): use a newer VM image"; exit 1; }
echo "driver ${drv} OK"
D="$(mktemp -d)"; cd "${D}"
gcloud storage cp "${PREFIX}/vlar-*.tar.zst" "${PREFIX}/image-digests.json" "${PREFIX}/SHA256SUMS" .
sha256sum -c SHA256SUMS
COMMIT="$(python3 -c "import json; print(json.load(open('image-digests.json'))['commit'])")"
for t in sim pi05 xvla; do
  zstd -dc "vlar-${t}.tar.zst" | docker load
  want="$(python3 -c "import json,sys; print(json.load(open('image-digests.json'))[sys.argv[1]])" "${t}")"
  got="$(docker image inspect --format '{{.Id}}' "vlar-${t}:${COMMIT}")"
  [[ "${got}" == "${want}" ]] || { echo "vlar-${t}: image ID ${got}, recorded ${want}"; exit 1; }
done
rm -f vlar-*.tar.zst                                  # 7 GB back before anything else (disk budget)
sudo install -D -m 0644 image-digests.json /etc/vlar/image-digests.json
echo "VLAR_IMAGE_COMMIT=${COMMIT}" | sudo tee /etc/vlar/images.env >/dev/null   # the tag the services run
# The servers' batch geometry lives in the calibration file's write-once group (amendment 35): written to
# /etc/vlar/batching.env once Phase 0 step 2 has chosen it; until then the production servers cannot start.
if docker run --rm -v /etc/vlar:/etc/vlar:ro -e VLAR_CALIBRATION=/etc/vlar/calibration/calibration.yaml \
    "vlar-sim:${COMMIT}" python -m vlar_harness.calfile batching-env > /tmp/vlar-batching.env 2>/dev/null; then
  sudo install -m 0644 /tmp/vlar-batching.env /etc/vlar/batching.env
else
  sudo rm -f /etc/vlar/batching.env; echo "no batch geometry yet (Phase 0 step 2 decides it)"
fi
rm -f /tmp/vlar-batching.env
for t in sim pi05 xvla; do
  lock=lerobot.lock; [[ "${t}" == xvla ]] && lock=xvla.lock
  docker run --rm --gpus all -v /etc/vlar:/etc/vlar:ro "vlar-${t}:${COMMIT}" \
    python containers/check_env.py --role "${t}" --lock "containers/${lock}" --gpu
done
cd / && rm -rf "${D}"
df -h / | tail -n 1
echo "images vlar-{sim,pi05,xvla}:${COMMIT} loaded and checked"
