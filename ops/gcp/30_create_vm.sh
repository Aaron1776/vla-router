#!/usr/bin/env bash
# One g2-standard-16 Spot VM (1x L4 24 GB, 16 vCPU, 64 GiB). G2 machine types come with their GPU
# attached, so no --accelerator flag. On preemption the VM stops (disk kept) and the shutdown script
# stops the spend daemon so its last tick is recorded.
set -euo pipefail
: "${PROJECT_ID:?source ops/gcp/env.sh first}"
[[ "${IMAGE_FAMILY}" == REPLACE ]] && { echo "set IMAGE_FAMILY in env.sh"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
gcloud compute instances create "${VM_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" \
  --machine-type=g2-standard-16 \
  --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE \
  --boot-disk-size="${DISK_GB}GB" --boot-disk-type="${DISK_TYPE}" \
  --image-family="${IMAGE_FAMILY}" --image-project="${IMAGE_PROJECT}" \
  --service-account="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --scopes=cloud-platform \
  --metadata=install-nvidia-driver=True \
  --metadata-from-file=shutdown-script="${HERE}/../vm/shutdown.sh"
# The disk bills for every hour it exists, running or not: record it in the spend ledger.
echo "Now record the disk in the ledger (from the VM or any machine with the ops config):"
echo "  vlar-ops spend disk-created --name ${VM_NAME}-boot --gb ${DISK_GB} --usd-per-gb-month ${DISK_USD_PER_GB_MONTH}"
