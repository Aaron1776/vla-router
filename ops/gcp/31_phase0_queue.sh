#!/usr/bin/env bash
# Queue Phase 0 steps for the L4 VM's step runner (ops/vm/phase0/runner.sh), from the Mac, without SSH:
#   ops/gcp/31_phase0_queue.sh ops/vm/phase0/steps/010-load-images.sh ...
# Uploads `git archive` of the current clean commit as the host-side source the VM installs on its next boot
# (<storage>/phase0/src/<commit>.tar.gz, named in src/CURRENT), then copies the step scripts into
# <storage>/phase0/queue/. A running VM picks new steps up before its next step; a stopped VM on its next start.
set -euo pipefail
: "${BUCKET:?source ops/gcp/env.sh first}"
(( $# > 0 )) || { echo "usage: $0 <step script>..."; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "commit first: the VM installs a clean commit"; exit 1; }
P0="gs://${BUCKET}/vla-router/phase0"
COMMIT="$(git rev-parse HEAD)"
SRC="${COMMIT}.tar.gz"
if ! gcloud storage ls "${P0}/src/${SRC}" >/dev/null 2>&1; then
  T="$(mktemp -d)"; git archive --format=tar.gz -o "${T}/${SRC}" HEAD
  gcloud storage cp -q "${T}/${SRC}" "${P0}/src/${SRC}"; rm -rf "${T}"
fi
echo "${SRC}" | gcloud storage cp -q - "${P0}/src/CURRENT"
for f in "$@"; do
  [[ -f "${f}" && "${f}" == *.sh ]] || { echo "not a step script: ${f}"; exit 1; }
  gcloud storage cp -q "${f}" "${P0}/queue/$(basename "${f}")"
done
echo "host source ${COMMIT:0:7}; queue now:"
gcloud storage ls "${P0}/queue/" 2>/dev/null | sed 's#.*/#  #'
