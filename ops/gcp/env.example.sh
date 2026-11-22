# Copy to ops/gcp/env.sh (git-ignored), fill in, then `source ops/gcp/env.sh`.
export PROJECT_ID="REPLACE"
export BILLING_ACCOUNT="REPLACE"          # gcloud billing accounts list
export REGION="us-central1"               # needs L4 Spot capacity and quota
export ZONE="us-central1-a"
export BUCKET="REPLACE"                   # globally unique, e.g. <project>-vla-router
export VM_NAME="vlar-l4"
export SA_NAME="vlar-vm"
export DISK_GB=100                        # L4 boot disk, sized 2026-11-22 (ops/README.md, "Boot disk size"):
                                          # VM image 50 + images 14 + image files while loading 7 + weights 11
                                          # + working space 5 = 87 GB, plus a 13 GB margin. About $0.33/day.
export DISK_TYPE="pd-balanced"
export DISK_USD_PER_GB_MONTH=0.10         # pd-balanced list price: check for your region
# Deep Learning VM image with the NVIDIA driver, Docker and the NVIDIA container toolkit. The images use torch
# 2.11.0 built for CUDA 13.0, which needs driver 580.65.06 or newer: pick a family whose name ends in nvidia-580
# or later (30_create_vm.sh refuses older ones; ops/vm/load_images.sh checks the installed driver). Family names
# change; list current ones with:
#   gcloud compute images list --project deeplearning-platform-release --format="value(family)" | grep -E 'nvidia-(5[89][0-9]|[6-9][0-9]{2})$' | sort -u
export IMAGE_PROJECT="deeplearning-platform-release"
export IMAGE_FAMILY="REPLACE"
# CPU-only build VM for the container images (35_build_images.sh; amendment 28): no GPU, no GPU quota.
export BUILD_VM_NAME="vlar-build"
export BUILD_MACHINE="e2-standard-8"      # 8 vCPU, 32 GB
export BUILD_DISK_GB=150                  # three images, their build cache and the exported files
export BUILD_MAX_HOURS=4                  # the VM deletes itself after this (--max-run-duration)
export BUILD_IMAGE_PROJECT="ubuntu-os-cloud"
export BUILD_IMAGE_FAMILY="ubuntu-2404-lts-amd64"
export ALERT_EMAIL="REPLACE"              # where the soft-limit alert goes; never committed
