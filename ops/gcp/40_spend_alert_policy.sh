#!/usr/bin/env bash
# Email when the spend guard logs its soft-limit (or hard-limit) event (docs/SPEC.md §12).
# Uses a log-based alert, so no mail credentials live on the VM.
set -euo pipefail
: "${PROJECT_ID:?source ops/gcp/env.sh first}"
[[ "${ALERT_EMAIL}" == REPLACE ]] && { echo "set ALERT_EMAIL in env.sh"; exit 1; }
CHANNEL=$(gcloud beta monitoring channels create --project="${PROJECT_ID}" --display-name="vla-router spend" \
  --type=email --channel-labels="email_address=${ALERT_EMAIL}" --format="value(name)")
POLICY=$(mktemp)
cat > "${POLICY}" <<JSON
{
  "displayName": "vla-router spend guard",
  "combiner": "OR",
  "conditions": [{
    "displayName": "spend guard limit event",
    "conditionMatchedLog": {
      "filter": "logName=\"projects/${PROJECT_ID}/logs/vlar-spend\" AND (jsonPayload.event=\"spend_soft_limit\" OR jsonPayload.event=\"spend_hard_limit\")"
    }
  }],
  "alertStrategy": {"notificationRateLimit": {"period": "3600s"}, "autoClose": "86400s"},
  "notificationChannels": ["${CHANNEL}"]
}
JSON
gcloud alpha monitoring policies create --project="${PROJECT_ID}" --policy-from-file="${POLICY}"
rm -f "${POLICY}"
# Budget alerts (20_budget_alerts.sh) go to the billing account's admins by default, which on a university account may
# not include you: send them to this channel too.
BUDGET=$(gcloud billing budgets list --billing-account="${BILLING_ACCOUNT}" --filter="displayName=vla-router-50usd" \
  --format="value(name)" | head -n 1)
[[ -n "${BUDGET}" ]] && gcloud billing budgets update "${BUDGET}" --billing-account="${BILLING_ACCOUNT}" \
  --notifications-rule-monitoring-notification-channels="${CHANNEL}"
