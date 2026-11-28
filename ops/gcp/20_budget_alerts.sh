#!/usr/bin/env bash
# Backup alerts at 50/80/100% of $50 (docs/SPEC.md §12). These only email; the spend guard stops the VM.
set -euo pipefail
: "${BILLING_ACCOUNT:?source ops/gcp/env.sh first}" "${BUDGET_START:?}" "${BUDGET_END:?}"
# One custom period for the whole project, not a calendar month: the $50 is a lifetime ceiling, and a monthly
# budget would reset and alert at $25/$40/$50 per month.
# Credits are excluded: on a coupon-funded account the cost after credits stays at $0 until the coupon is used up,
# so a budget on it would never alert. The thresholds track the gross cost, as the spend guard's ledger does.
gcloud billing budgets create --billing-account="${BILLING_ACCOUNT}" --display-name="vla-router-50usd" \
  --budget-amount=50USD --filter-projects="projects/${PROJECT_ID}" \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.8 --threshold-rule=percent=1.0 \
  --credit-types-treatment=exclude-all-credits \
  --start-date="${BUDGET_START}" --end-date="${BUDGET_END}"
