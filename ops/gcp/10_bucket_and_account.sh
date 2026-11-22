#!/usr/bin/env bash
# Bucket for all project data plus the VM's service account (docs/SPEC.md §12, §13).
set -euo pipefail
: "${PROJECT_ID:?source ops/gcp/env.sh first}"
# The APIs every script uses (free to enable; a new project has few of them on).
gcloud services enable compute.googleapis.com storage.googleapis.com iam.googleapis.com logging.googleapis.com \
  monitoring.googleapis.com billingbudgets.googleapis.com --project="${PROJECT_ID}"

gcloud storage buckets create "gs://${BUCKET}" --project="${PROJECT_ID}" --location="${REGION}" \
  --uniform-bucket-level-access --public-access-prevention
# Soft delete keeps deleted objects (and bills them) for 7 days by default; this project rewrites
# the ledger every minute, so turn it off.
gcloud storage buckets update "gs://${BUCKET}" --clear-soft-delete

gcloud iam service-accounts create "${SA_NAME}" --project="${PROJECT_ID}" --display-name="vla-router VM"
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member="serviceAccount:${SA}" \
  --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${SA}" \
  --role="roles/logging.logWriter" --condition=None
echo "bucket gs://${BUCKET} and service account ${SA} ready"
