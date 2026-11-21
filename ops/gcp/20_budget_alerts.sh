#!/usr/bin/env bash
# Backup alerts at 50/80/100% of $50 (docs/SPEC.md §12). These only email; the spend guard stops the VM.
set -euo pipefail
: "${BILLING_ACCOUNT:?source ops/gcp/env.sh first}"
gcloud billing budgets create --billing-account="${BILLING_ACCOUNT}" --display-name="vla-router-50usd" \
  --budget-amount=50USD --filter-projects="projects/${PROJECT_ID}" \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.8 --threshold-rule=percent=1.0
