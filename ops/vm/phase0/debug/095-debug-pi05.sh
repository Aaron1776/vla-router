#!/usr/bin/env bash
# cap-seconds: 3600
# Diagnosis of the π0.5 padded-vs-batch-of-one discrepancy (stage A row pi05-batch-invariance-l4; user's go-ahead
# 2026-11-23, max effort). Runs pi05_batch_debug.py (from <phase0>/debug/ in the bucket) in the vlar-pi05 image on the
# GPU with the stage A test rows' container options; everything lands in logs/debug-pi05/<time>/ (synced to the bucket).
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
fetch_tokenizer || fail "the gated PaliGemma tokenizer is not on the VM"
OUT="${P0D}/logs/debug-pi05/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUT}" /opt/vlar-debug
gcloud storage cp "${VLAR_PHASE0_URL}/debug/pi05_batch_debug.py" /opt/vlar-debug/pi05_batch_debug.py \
  || fail "no debug script at ${VLAR_PHASE0_URL}/debug/"
sha256sum /opt/vlar-debug/pi05_batch_debug.py | tee "${OUT}/script.sha256"
{ nvidia-smi; nvidia-smi -q | grep -iE "driver version|cuda version|product name|vbios|ecc mode" ; } > "${OUT}/nvidia-smi.txt" 2>&1
docker run "${GPU_RUN[@]}" -v /opt/vlar-debug:/debug:ro -v "${OUT}:/out" "vlar-pi05:${C}" \
  python /debug/pi05_batch_debug.py --out /out > "${OUT}/run.log" 2>&1
rc=$?
tail -n 5 "${OUT}/run.log"
(( rc == 0 )) || fail "debug script exit ${rc} (logs/debug-pi05/$(basename "${OUT}")/run.log)"
echo "debug outputs: logs/debug-pi05/$(basename "${OUT}")"
