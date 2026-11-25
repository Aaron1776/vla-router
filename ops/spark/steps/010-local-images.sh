#!/usr/bin/env bash
# cap-seconds: 1800
# The Spark's step 010 (instead of ops/vm/phase0/steps/010-load-images.sh): the aarch64 images built on this machine
# by containers/aarch64/build.sh, not loaded from the bucket. As load_images.sh: images.env, image-digests.json (with
# "platform": "linux/aarch64"), batching.env from the calibration file if it has the geometry, then check_env.py --gpu
# in each image (driver, torch sees the GPU, CUDA 13.0, MuJoCo EGL in sim) against the aarch64 locks.
source /opt/vla-router/ops/vm/phase0/lib.sh
D="${VLAR_SPARK_ROOT}/etc"; mkdir -p "${D}/calibration"
want="${VLAR_SPARK_IMAGES:?the image-digests.json written by containers/aarch64/build.sh}"
COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "${want}")"
for t in sim pi05 xvla; do
  got="$(docker image inspect --format '{{.Id}}' "vlar-${t}:${COMMIT}")" || fail "image vlar-${t}:${COMMIT} is not built"
  exp="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "${want}" "${t}")"
  [[ "${got}" == "${exp}" ]] || fail "vlar-${t}:${COMMIT} is ${got}, the build recorded ${exp}"
done
install -m 0644 "${want}" "${D}/image-digests.json"
echo "VLAR_IMAGE_COMMIT=${COMMIT}" > "${D}/images.env"
if docker run --rm -v /etc/vlar:/etc/vlar:ro -e VLAR_CALIBRATION=/etc/vlar/calibration/calibration.yaml \
    "vlar-sim:${COMMIT}" python -m vlar_harness.calfile batching-env > "${D}/batching.env.tmp" 2>/dev/null; then
  mv "${D}/batching.env.tmp" "${D}/batching.env"
else
  rm -f "${D}/batching.env.tmp" "${D}/batching.env"; echo "no batch geometry yet (no calibration file with it)"
fi
for t in sim pi05 xvla; do
  lock=lerobot.lock; [[ "${t}" == xvla ]] && lock=xvla.lock
  docker run --rm --gpus all -v /etc/vlar:/etc/vlar:ro "vlar-${t}:${COMMIT}" \
    python containers/check_env.py --role "${t}" --lock "containers/aarch64/${lock}" --gpu || fail "check_env --gpu failed in ${t}"
done
echo "images vlar-{sim,pi05,xvla}:${COMMIT} checked"
