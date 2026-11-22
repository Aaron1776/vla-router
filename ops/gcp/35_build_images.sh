#!/usr/bin/env bash
# Builds the three container images on a small CPU-only VM (no GPU, so no GPU quota needed), runs the
# in-container tests that need no GPU, and exports the images as files to the bucket (SPEC §13, §15; amendment 28):
#   gs://$BUCKET/containers/<commit>/{vlar-sim,vlar-pi05,vlar-xvla}.tar.zst, image-digests.json, SHA256SUMS,
#   build.log, test-*.log, then DONE (or FAILED with the reason).
# Run from the repo root on a clean commit, keeping the Mac awake for the whole run:
#   caffeinate -i -s ops/gcp/35_build_images.sh
# BUILD_PROVISIONING picks the VM's provisioning model: SPOT (default; cheapest, can be preempted at any time) or
# STANDARD (on-demand, about $0.27/h for e2-standard-8; not preempted), e.g.
#   BUILD_PROVISIONING=STANDARD caffeinate -i -s ops/gcp/35_build_images.sh
# Both get the same BUILD_MAX_HOURS cap (--max-run-duration, deleted at the cap), polling and confirmed deletion.
# Waits for the VM (polling the bucket), prints the result, then deletes the VM and its disk and confirms both gone.
#
# Network drops on this machine are expected and harmless: a check that fails is retried, never read as a result.
# The VM is deleted only after DONE or FAILED has actually been read from the bucket, or once the BUILD_MAX_HOURS
# cap has passed (the VM also deletes itself at the cap, --max-run-duration), or when it is confirmed already gone
# (then only a leftover disk is removed). If this script stops for any other reason (interrupted, a bug) it leaves
# the VM alone and says how to delete it.
# Exit status: 0 built and tested; 1 build failed or no result; 2 the deletion could not be confirmed.
#
# Cost: an e2-standard-8 VM (Spot or on-demand) for about an hour plus its disk, well under $1; the weekly ledger reconcile
# (`vlar-ops spend reconcile`) picks it up, since the spend daemon runs on the L4 VM only.
set -euo pipefail
: "${PROJECT_ID:?source ops/gcp/env.sh first}"
: "${BUCKET:?}" "${ZONE:?}" "${SA_NAME:?}" "${BUILD_VM_NAME:?}" "${BUILD_MACHINE:?}" "${BUILD_DISK_GB:?}"
: "${BUILD_MAX_HOURS:?}" "${BUILD_IMAGE_FAMILY:?}" "${BUILD_IMAGE_PROJECT:?}"
PROVISIONING="${BUILD_PROVISIONING:-SPOT}"
case "${PROVISIONING}" in
  SPOT|STANDARD) ;;
  *) echo "BUILD_PROVISIONING must be SPOT or STANDARD, not '${PROVISIONING}'"; exit 1 ;;
esac
POLL_S="${VLAR_POLL_SECONDS:-60}"                           # test hook
CAP_S="${VLAR_CAP_SECONDS:-$(( BUILD_MAX_HOURS * 3600 ))}"  # test hook; the cap is BUILD_MAX_HOURS
DELETE_TRIES="${VLAR_DELETE_TRIES:-90}"                     # about 1.5 h of retries at 60 s
HERE="$(cd "$(dirname "$0")" && pwd)"
[[ -z "$(git status --porcelain)" ]] || { echo "commit first: the images are built from a clean commit"; exit 1; }
COMMIT="$(git rev-parse HEAD)"
PREFIX="gs://${BUCKET}/containers/${COMMIT}"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Three-way checks: print the answer and return 0, or return 1 when the check itself failed (network, auth, ...).
vm_state() {    # present | gone
  local out; out="$(gcloud compute instances list --project="${PROJECT_ID}" --zones="${ZONE}" \
    --filter="name=${BUILD_VM_NAME}" --format="value(name)" 2>/dev/null)" || return 1
  if grep -qxF "${BUILD_VM_NAME}" <<<"${out}"; then echo present; else echo gone; fi
}
disk_state() {  # present | gone (the boot disk has the VM's name)
  local out; out="$(gcloud compute disks list --project="${PROJECT_ID}" --zones="${ZONE}" \
    --filter="name=${BUILD_VM_NAME}" --format="value(name)" 2>/dev/null)" || return 1
  if grep -qxF "${BUILD_VM_NAME}" <<<"${out}"; then echo present; else echo gone; fi
}
result_state() {  # done | failed | none; the prefix always holds src.tar.gz, so a failed listing is an error
  local out; out="$(gcloud storage ls "${PREFIX}/" 2>/dev/null)" || return 1
  if grep -qxF "${PREFIX}/DONE" <<<"${out}"; then echo done
  elif grep -qxF "${PREFIX}/FAILED" <<<"${out}"; then echo failed
  else echo none; fi
}
read_object() {   # prints the object, retrying; never fails the script
  local i; for (( i = 0; i < 5; i++ )); do gcloud storage cat "${PREFIX}/$1" 2>/dev/null && return 0; sleep "${POLL_S}"; done
  echo "(could not read ${PREFIX}/$1 just now; it is in the bucket)"
}
settle() {        # retries a three-way check until it answers, or returns 1 once the given time has passed
  local fn=$1 until=$2 ans
  while :; do
    if ans="$("${fn}")"; then echo "${ans}"; return 0; fi
    (( $(date +%s) < until )) || return 1
    log "${fn}: check failed (network?); retrying in ${POLL_S} s" >&2; sleep "${POLL_S}"
  done
}

delete_build_vm() {   # returns 0 only once the VM and its disk are both confirmed gone
  local i vm disk
  for (( i = 0; i < DELETE_TRIES; i++ )); do
    if vm="$(vm_state)"; then
      if [[ "${vm}" == present ]]; then
        gcloud compute instances delete "${BUILD_VM_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --quiet \
          || log "instance delete did not complete; checking again"
      elif disk="$(disk_state)"; then
        if [[ "${disk}" == gone ]]; then log "confirmed: the build VM and its disk are deleted"; return 0; fi
        gcloud compute disks delete "${BUILD_VM_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --quiet \
          || log "disk delete did not complete; checking again"
      else log "disk check failed (network?); retrying"; fi
    else log "VM check failed (network?); retrying"; fi
    sleep "${POLL_S}"
  done
  log "COULD NOT CONFIRM the deletion of ${BUILD_VM_NAME} and its disk in ${ZONE}: check and delete them by hand"
  return 2
}

if gcloud storage ls "${PREFIX}/DONE" >/dev/null 2>&1; then echo "already built: ${PREFIX}"; exit 0; fi
st="$(settle vm_state "$(( $(date +%s) + 1800 ))")" || { echo "cannot reach Compute Engine; nothing created"; exit 1; }
[[ "${st}" == gone ]] || { echo "a VM named ${BUILD_VM_NAME} already exists in ${ZONE}; delete it or wait for it"; exit 1; }

SRC="$(mktemp -d)/src.tar.gz"
git archive --format=tar.gz -o "${SRC}" "${COMMIT}"
gcloud storage cp "${SRC}" "${PREFIX}/src.tar.gz"
gcloud storage rm -q "${PREFIX}/FAILED" 2>/dev/null || true

may_delete=0
on_exit() {
  local rc=$?
  if (( may_delete == 0 )); then
    log "stopped before a result: ${BUILD_VM_NAME} was NOT deleted (it deletes itself ${BUILD_MAX_HOURS} h after start)"
    log "to delete it now: gcloud compute instances delete ${BUILD_VM_NAME} --zone=${ZONE}"
  fi
  exit "${rc}"
}
trap on_exit EXIT
trap 'exit 130' INT TERM HUP

deadline=$(( $(date +%s) + CAP_S ))
if ! gcloud compute instances create "${BUILD_VM_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" \
  --machine-type="${BUILD_MACHINE}" \
  --provisioning-model="${PROVISIONING}" --instance-termination-action=DELETE --max-run-duration="${BUILD_MAX_HOURS}h" \
  --boot-disk-size="${BUILD_DISK_GB}GB" --boot-disk-type=pd-balanced \
  --image-family="${BUILD_IMAGE_FAMILY}" --image-project="${BUILD_IMAGE_PROJECT}" \
  --service-account="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" --scopes=cloud-platform \
  --metadata=vlar-prefix="${PREFIX}",vlar-commit="${COMMIT}" \
  --metadata-from-file=startup-script="${HERE}/build_vm_startup.sh"; then
  # A dropped connection can fail the call after the VM was created: find out which before giving up.
  st="$(settle vm_state "${deadline}")" || st=unknown
  if [[ "${st}" == gone ]]; then may_delete=1; echo "the VM was not created; nothing to delete"; exit 1; fi
  [[ "${st}" == present ]] || { log "cannot tell whether the VM was created"; exit 1; }
  log "create reported an error, but the VM exists; waiting for its result"
fi
log "build VM created (${PROVISIONING}); polling ${PREFIX}/ every ${POLL_S} s (cap ${BUILD_MAX_HOURS} h)"

result=""
while :; do
  if r="$(result_state)"; then
    if [[ "${r}" == done || "${r}" == failed ]]; then result="${r}"; break; fi
  else log "bucket check failed (network?); retrying"; fi
  if (( $(date +%s) >= deadline )); then result=cap; break; fi
  if vm="$(vm_state)"; then
    if [[ "${vm}" == gone ]]; then
      # Preempted (Spot), deleted at the cap, or deleted elsewhere: the result may have landed just before, so read
      # the bucket for sure.
      r="$(settle result_state "$(( $(date +%s) + 1800 ))")" || r=unknown
      if [[ "${r}" == done || "${r}" == failed ]]; then result="${r}"; else result=gone; fi
      break
    fi
  else log "VM check failed (network?); retrying"; fi
  sleep "${POLL_S}"
done

may_delete=1
case "${result}" in
  done)   log "built and tested: ${PREFIX}"; read_object image-digests.json; rc=0 ;;
  failed) log "build FAILED: $(read_object FAILED)"; log "logs: ${PREFIX}/build.log, test-*.log"; rc=1 ;;
  cap)    log "no result after ${BUILD_MAX_HOURS} h"; rc=1 ;;
  *)      log "the build VM is gone without a result (Spot preemption, the time cap, or deleted elsewhere): run this script again"; rc=1 ;;
esac
delete_build_vm || rc=2
exit "${rc}"
