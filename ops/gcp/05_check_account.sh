#!/usr/bin/env bash
# Read-only checks before anything is created (SPEC §12, §18 item 6): who is signed in, billing on the project, the
# APIs the scripts use, the GPU quotas the L4 VM needs, and a Deep Learning VM image family whose NVIDIA driver
# supports CUDA 13.0 (580+). Creates nothing and costs nothing.
set -uo pipefail
: "${PROJECT_ID:?source ops/gcp/env.sh first}"
echo "== account";  gcloud auth list --filter=status:ACTIVE --format="value(account)"
echo "== billing";  gcloud billing projects describe "${PROJECT_ID}" --format="value(billingAccountName,billingEnabled)"
echo "== billing accounts visible to you"; gcloud billing accounts list --format="table(name,displayName,open)"
echo "== enabled APIs (compute, storage, iam, logging, monitoring, billingbudgets)"
gcloud services list --enabled --project="${PROJECT_ID}" --format="value(config.name)" \
  | grep -E '^(compute|storage|iam|logging|monitoring|billingbudgets)\.googleapis\.com$' || echo "(none of them yet)"
echo "== GPU quota, global (needs GPUS_ALL_REGIONS >= 1)"
gcloud compute project-info describe --project="${PROJECT_ID}" --flatten=quotas \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -E 'METRIC|GPUS_ALL_REGIONS'
echo "== GPU quota, ${REGION} (needs NVIDIA_L4_GPUS and PREEMPTIBLE_NVIDIA_L4_GPUS >= 1)"
gcloud compute regions describe "${REGION}" --project="${PROJECT_ID}" --flatten=quotas \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -E 'METRIC|NVIDIA_L4'
echo "== Deep Learning VM image families with driver 580+ (for IMAGE_FAMILY)"
gcloud compute images list --project=deeplearning-platform-release --no-standard-images \
  --format="value(family)" | grep -E 'nvidia-(5[89][0-9]|[6-9][0-9]{2})$' | sort -u
echo "== build VM image family ${BUILD_IMAGE_FAMILY}"
gcloud compute images describe-from-family "${BUILD_IMAGE_FAMILY}" --project="${BUILD_IMAGE_PROJECT}" --format="value(name)"
