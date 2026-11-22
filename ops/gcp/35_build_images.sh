#!/usr/bin/env bash
# Builds the three container images on a small CPU-only Spot VM (no GPU, so no GPU quota needed), runs the
# in-container tests that need no GPU, and exports the images as files to the bucket (SPEC §13, §15; amendment 28):
#   gs://$BUCKET/containers/<commit>/{vlar-sim,vlar-pi05,vlar-xvla}.tar.zst, image-digests.json, SHA256SUMS,
#   build.log, test-*.log, then DONE (or FAILED with the reason).
# Run from the repo root on a clean commit. Waits for the VM (polling the bucket), prints the result, then deletes
# the VM and its disk. The VM also deletes itself after BUILD_MAX_HOURS (--max-run-duration), so it cannot run on.
# Cost: an e2-standard-8 Spot VM for about an hour plus its disk, well under $1; the weekly ledger reconcile
# (`vlar-ops spend reconcile`) picks it up, since the spend daemon runs on the L4 VM only.
set -euo pipefail
: "${PROJECT_ID:?source ops/gcp/env.sh first}"
: "${BUCKET:?}" "${BUILD_VM_NAME:?}" "${BUILD_MACHINE:?}" "${BUILD_DISK_GB:?}" "${BUILD_MAX_HOURS:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[[ -z "$(git status --porcelain)" ]] || { echo "commit first: the images are built from a clean commit"; exit 1; }
COMMIT="$(git rev-parse HEAD)"
PREFIX="gs://${BUCKET}/containers/${COMMIT}"
if gcloud storage ls "${PREFIX}/DONE" >/dev/null 2>&1; then echo "already built: ${PREFIX}"; exit 0; fi

SRC="$(mktemp -d)/src.tar.gz"
git archive --format=tar.gz -o "${SRC}" "${COMMIT}"
gcloud storage cp "${SRC}" "${PREFIX}/src.tar.gz"
gcloud storage rm -q "${PREFIX}/FAILED" 2>/dev/null || true

gcloud compute instances create "${BUILD_VM_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" \
  --machine-type="${BUILD_MACHINE}" \
  --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration="${BUILD_MAX_HOURS}h" \
  --boot-disk-size="${BUILD_DISK_GB}GB" --boot-disk-type=pd-balanced \
  --image-family="${BUILD_IMAGE_FAMILY}" --image-project="${BUILD_IMAGE_PROJECT}" \
  --service-account="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" --scopes=cloud-platform \
  --metadata=vlar-prefix="${PREFIX}",vlar-commit="${COMMIT}" \
  --metadata-from-file=startup-script="${HERE}/build_vm_startup.sh"

cleanup() { gcloud compute instances delete "${BUILD_VM_NAME}" --zone="${ZONE}" --quiet 2>/dev/null || true; }
trap cleanup EXIT
deadline=$(( $(date +%s) + BUILD_MAX_HOURS * 3600 ))
while (( $(date +%s) < deadline )); do
  if gcloud storage ls "${PREFIX}/DONE" >/dev/null 2>&1; then
    echo "built and tested: ${PREFIX}"; gcloud storage cat "${PREFIX}/image-digests.json"; exit 0
  fi
  if gcloud storage ls "${PREFIX}/FAILED" >/dev/null 2>&1; then
    echo "build FAILED: $(gcloud storage cat "${PREFIX}/FAILED")"; echo "logs: ${PREFIX}/build.log, test-*.log"; exit 1
  fi
  if ! gcloud compute instances describe "${BUILD_VM_NAME}" --zone="${ZONE}" >/dev/null 2>&1; then
    echo "the build VM is gone without a result (Spot preemption or the time cap): run this script again"; exit 1
  fi
  sleep 60
done
echo "no result after ${BUILD_MAX_HOURS} h"; exit 1
