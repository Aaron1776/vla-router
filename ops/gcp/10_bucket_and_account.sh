#!/usr/bin/env bash
# Bucket for all project data plus the VM's service account (docs/SPEC.md §12, §13).
set -euo pipefail
: "${PROJECT_ID:?source ops/gcp/env.sh first}"
# The APIs every script uses (free to enable; a new project has few of them on).
gcloud services enable compute.googleapis.com storage.googleapis.com iam.googleapis.com logging.googleapis.com \
  monitoring.googleapis.com billingbudgets.googleapis.com --project="${PROJECT_ID}"

gcloud storage buckets create "gs://${BUCKET}" --project="${PROJECT_ID}" --location="${REGION}" \
  --uniform-bucket-level-access --public-access-prevention
# Soft delete keeps deleted and overwritten objects (and bills them) for the retention period: the only undo for an
# accidental delete or overwrite. 30 days for the project's duration (SPEC §17 amendment 60, the user 2026-11-24,
# reversing the setup's "off"): the per-minute ledger rewrites are ~3 KB each, so the cost is the planned deletions
# (~$0.02/GB-month). 90_teardown.sh clears it before the final deletion, or that would be kept and billed 30 days.
gcloud storage buckets update "gs://${BUCKET}" --soft-delete-duration=30d

gcloud iam service-accounts create "${SA_NAME}" --project="${PROJECT_ID}" --display-name="vla-router VM"
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
# A new service account takes a few seconds to reach IAM ("does not exist"): retry for up to two minutes.
for i in $(seq 1 12); do
  gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member="serviceAccount:${SA}" \
    --role="roles/storage.objectAdmin" >/dev/null && break
  [[ $i == 12 ]] && { echo "service account not visible to IAM after 2 minutes"; exit 1; }
  sleep 10
done
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${SA}" \
  --role="roles/logging.logWriter" --condition=None
echo "bucket gs://${BUCKET} and service account ${SA} ready"
