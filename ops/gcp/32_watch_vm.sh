#!/usr/bin/env bash
# Watch the L4 VM from the Mac until it stops, printing each change of the step runner's state (STATE.json in the
# bucket), then the final state with its reason, the step history and the spend guard's running total.
# Network drops and sleep on the Mac are harmless: failed checks are retried, never read as a result.
#   ops/gcp/32_watch_vm.sh            (exit 0 once the VM is stopped or gone)
set -uo pipefail
: "${PROJECT_ID:?source ops/gcp/env.sh first}"
P0="gs://${BUCKET}/vla-router/phase0"
POLL_S="${VLAR_POLL_SECONDS:-120}"
last=""; ERR="$(mktemp)"; trap 'rm -f "${ERR}"' EXIT
while true; do
  st="$(gcloud compute instances describe "${VM_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" \
        --format='value(status)' 2>"${ERR}")"
  if [[ -z "${st}" ]] && grep -q 'was not found' "${ERR}"; then st=GONE; fi
  state="$(gcloud storage cat "${P0}/STATE.json" 2>/dev/null | tr -d '\n' | tr -s ' ')"
  if [[ -n "${state}" && "${state}" != "${last}" ]]; then echo "[$(date '+%m-%d %H:%M')] VM ${st:-?}: ${state}"; last="${state}"; fi
  case "${st}" in
    TERMINATED|STOPPED|SUSPENDED|GONE)
      echo "== VM ${VM_NAME} is ${st}"
      echo "== final state"; gcloud storage cat "${P0}/STATE.json" 2>/dev/null
      echo "== step history"; gcloud storage cat "${P0}/history.jsonl" 2>/dev/null
      exit 0 ;;
  esac
  sleep "${POLL_S}"
done
