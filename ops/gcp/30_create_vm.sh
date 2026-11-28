#!/usr/bin/env bash
# One g2-standard-16 Spot VM (1x L4 24 GB, 16 vCPU, 64 GiB). G2 machine types come with their GPU
# attached, so no --accelerator flag. On preemption the VM stops (disk kept) and the shutdown script
# stops the step runner and the spend daemon so their last state and tick are recorded.
# The startup script (ops/vm/phase0/startup.sh) installs the host side from the bucket on every boot, starts the
# spend guard, records the disk in the ledger once, and starts the Phase 0 step runner, which runs the steps queued
# by ops/gcp/31_phase0_queue.sh (run it first) and stops the VM when it runs out of work or needs the user.
#   IMAGES_COMMIT=<full commit of the images in gs://$BUCKET/containers/> ops/gcp/30_create_vm.sh
# VLAR_RUNNER=off boots without the runner (manual work over SSH; stop the VM yourself).
set -euo pipefail
: "${PROJECT_ID:?source ops/gcp/env.sh first}"
: "${IMAGES_COMMIT:?the full commit of the images to load (gs://\$BUCKET/containers/<commit>)}"
[[ "${IMAGES_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "IMAGES_COMMIT must be a full 40-character commit hash"; exit 1; }
[[ "${IMAGE_FAMILY}" == REPLACE ]] && { echo "set IMAGE_FAMILY in env.sh"; exit 1; }
# torch 2.11.0 for CUDA 13.0 needs driver 580.65.06+ (amendment 29); load_images.sh checks the installed one.
if [[ "${IMAGE_FAMILY}" =~ nvidia-([0-9]+)$ ]]; then
  (( BASH_REMATCH[1] >= 580 )) || { echo "${IMAGE_FAMILY}: driver ${BASH_REMATCH[1]} is too old for CUDA 13.0 (needs 580+)"; exit 1; }
else
  echo "warning: cannot tell the driver version from ${IMAGE_FAMILY}; ops/vm/load_images.sh will check it on the VM"
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
gcloud compute instances create "${VM_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" \
  --machine-type=g2-standard-16 \
  --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE \
  --boot-disk-size="${DISK_GB}GB" --boot-disk-type="${DISK_TYPE}" \
  --image-family="${IMAGE_FAMILY}" --image-project="${IMAGE_PROJECT}" \
  --service-account="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --scopes=cloud-platform \
  --metadata=install-nvidia-driver=True,vlar-storage-url="gs://${BUCKET}/vla-router",vlar-images-url="gs://${BUCKET}/containers/${IMAGES_COMMIT}",vlar-disk-gb="${DISK_GB}",vlar-disk-usd-per-gb-month="${DISK_USD_PER_GB_MONTH}",vlar-runner="${VLAR_RUNNER:-on}" \
  --metadata-from-file=shutdown-script="${HERE}/../vm/shutdown.sh",startup-script="${HERE}/../vm/phase0/startup.sh"
# The disk bills for every hour it exists, running or not: the startup script records it in the spend ledger once
# (vlar-ops spend disk-created --name ${VM_NAME}-boot). Deleting the VM and disk: close it with
#   vlar-ops spend disk-deleted --name ${VM_NAME}-boot
echo "created ${VM_NAME}; follow it with ops/gcp/32_watch_vm.sh"
