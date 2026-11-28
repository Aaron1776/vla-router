#!/usr/bin/env bash
# End of project (docs/SPEC.md §13): only after the archives are downloaded and verified locally.
# Deletes the VM and its disk, then the bucket and everything in it. Cannot be undone.
set -euo pipefail
: "${PROJECT_ID:?source ops/gcp/env.sh first}"
if [[ "${1:-}" != "--archives-verified" ]]; then
  echo "Refusing: run 'vlar-ops package verify --out <local dir>' on your computer first, then pass --archives-verified."
  exit 1
fi
read -r -p "Type the bucket name (${BUCKET}) to delete the VM, its disk and the bucket: " answer
[[ "${answer}" == "${BUCKET}" ]] || { echo "aborted"; exit 1; }
gcloud compute instances delete "${VM_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --delete-disks=all --quiet
# Clear soft delete first (SPEC §17 amendment 60): with the project's 30 days, the final deletion would otherwise
# keep, and bill, the whole bucket for 30 more days. Objects soft-deleted before this may stay listed until their
# own 30 days pass: check `gcloud storage ls --soft-deleted -r gs://<bucket>` afterwards.
gcloud storage buckets update "gs://${BUCKET}" --clear-soft-delete
gcloud storage rm --recursive "gs://${BUCKET}"
