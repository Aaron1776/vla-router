# Copy to ops/gcp/env.sh (git-ignored), fill in, then `source ops/gcp/env.sh`.
export PROJECT_ID="REPLACE"
export BILLING_ACCOUNT="REPLACE"          # gcloud billing accounts list
export REGION="us-central1"               # needs L4 Spot capacity and quota
export ZONE="us-central1-a"
export BUCKET="REPLACE"                   # globally unique, e.g. <project>-vla-router
export VM_NAME="vlar-l4"
export SA_NAME="vlar-vm"
export DISK_GB=200                        # container images + weights + local results buffer
export DISK_TYPE="pd-balanced"
export DISK_USD_PER_GB_MONTH=0.10         # pd-balanced list price: check for your region
# Deep Learning VM image with the NVIDIA driver, Docker and the NVIDIA container toolkit.
# Family names change; list current ones with:
#   gcloud compute images list --project deeplearning-platform-release --filter="family~common-cu12" --format="value(family)" | sort -u
export IMAGE_PROJECT="deeplearning-platform-release"
export IMAGE_FAMILY="REPLACE"
export ALERT_EMAIL="REPLACE"              # where the soft-limit alert goes; never committed
