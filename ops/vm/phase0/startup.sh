#!/usr/bin/env bash
# GCE startup script of the L4 VM (ops/gcp/30_create_vm.sh; runs as root on every boot). Installs the host side of the
# harness from the source archive the bucket names in <storage>/phase0/src/CURRENT (repo at /opt/vla-router, .venv
# with the queue and spend commands, ops.yaml), starts the spend guard before anything else, records the boot disk in
# the spend ledger once, then starts the Phase 0 step runner (ops/vm/phase0/runner.sh), which stops the VM when it
# runs out of queued work or needs the user. Metadata `vlar-runner=off` boots without the runner (manual work).
set -uo pipefail
md() { curl -fs -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
L=/var/lib/vlar; R=/opt/vla-router
mkdir -p "${L}/phase0/logs" && exec >>"${L}/phase0/logs/startup.log" 2>&1
echo "== $(date -u +%FT%TZ) boot"
STORAGE_URL="$(md attributes/vlar-storage-url)"; IMAGES_URL="$(md attributes/vlar-images-url)"
DISK_GB="$(md attributes/vlar-disk-gb)"; DISK_RATE="$(md attributes/vlar-disk-usd-per-gb-month)"
RUNNER="$(md attributes/vlar-runner || echo on)"; NAME="$(md name)"
P0="${STORAGE_URL}/phase0"
fail() {   # never leave the VM running idle: record why in the bucket, then stop
  echo "startup FAILED: $*"
  printf '{"state": "failed", "step": "startup", "reason": "startup: %s", "at": "%s", "spend": null}\n' "$*" \
    "$(date -u +%FT%TZ)" | gcloud storage cp - "${P0}/STATE.json" || true
  gcloud storage cp "${L}/phase0/logs/startup.log" "${P0}/logs/startup.log" || true
  shutdown -h now; exit 1
}
[[ -n "${STORAGE_URL}" && -n "${IMAGES_URL}" ]] || fail "metadata vlar-storage-url / vlar-images-url missing"

SRC="$(gcloud storage cat "${P0}/src/CURRENT")" || fail "no ${P0}/src/CURRENT (run ops/gcp/31_phase0_queue.sh)"
if [[ "$(cat "${R}/.src" 2>/dev/null)" != "${SRC}" ]]; then
  echo "installing ${SRC}"
  rm -rf "${R}.new" && mkdir -p "${R}.new" && gcloud storage cp "${P0}/src/${SRC}" /tmp/vlar-src.tar.gz \
    && tar -xzf /tmp/vlar-src.tar.gz -C "${R}.new" || fail "source archive ${SRC}"
  rm -rf "${R}" && mv "${R}.new" "${R}"
  python3 -m venv "${R}/.venv" 2>/dev/null || { apt-get update -q && apt-get install -y -q python3-venv \
    && python3 -m venv "${R}/.venv"; } || fail "python3 -m venv"
  "${R}/.venv/bin/pip" install -q -e "${R}[gcs,dev]" || fail "pip install of the host package"
  echo "${SRC}" > "${R}/.src"
fi
sed "s#^storage_url: .*#storage_url: ${STORAGE_URL}#" "${R}/ops/ops.example.yaml" > "${R}/ops/ops.yaml"
mkdir -p /etc/vlar/calibration
# The calibration file and its versions (amendment 35) survive a VM rebuild through the bucket. Restored only when
# missing here: a local file is never replaced by an older copy.
if [[ ! -e /etc/vlar/calibration/calibration.yaml ]]; then
  gcloud storage rsync -r "${STORAGE_URL}/calibration" /etc/vlar/calibration >/dev/null 2>&1 \
    && echo "calibration restored from ${STORAGE_URL}/calibration" || echo "no calibration in the bucket yet"
fi
printf 'VLAR_PHASE0_URL=%s\nVLAR_STORAGE_URL=%s\nVLAR_IMAGES_URL=%s\n' "${P0}" "${STORAGE_URL}" "${IMAGES_URL}" \
  > /etc/vlar/phase0.env
cp "${R}"/ops/vm/vlar-{spend,phase0,xvla,pi05}.service "${R}/ops/vm/vlar-worker@.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now vlar-spend || fail "spend guard did not start"
export VLAR_OPS_CONFIG="${R}/ops/ops.yaml"
if [[ ! -e "${L}/.disk-recorded" ]]; then   # once per disk: it bills every hour it exists, running or not
  "${R}/.venv/bin/vlar-ops" spend disk-created --name "${NAME}-boot" --gb "${DISK_GB}" \
    --usd-per-gb-month "${DISK_RATE}" && touch "${L}/.disk-recorded" || fail "disk not recorded in the ledger"
fi
"${R}/.venv/bin/vlar-ops" spend status || true
if [[ "${RUNNER}" == off ]]; then echo "vlar-runner=off: runner not started"; exit 0; fi
systemctl start vlar-phase0 || fail "phase 0 runner did not start"
