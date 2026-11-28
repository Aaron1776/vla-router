#!/usr/bin/env bash
# cap-seconds: 3600
# Weights at the harness config's pinned revisions into /opt/weights (read-only mounts for the servers and tests),
# their SHA-256 recorded in logs/weights.SHA256SUMS (synced to the bucket); the gated PaliGemma tokenizer's cache from
# the bucket (<storage>/hf-tokenizer/, put there from a logged-in machine: envs/README.md) into /opt/hf/hub.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
mkdir -p /opt/weights /opt/hf/hub
docker run --rm -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 -v /opt/weights:/weights "vlar-pi05:${C}" python -c "
from huggingface_hub import snapshot_download as d
from vlar_harness.config import load_config
m = load_config()['models']
d(m['pi05']['checkpoint'], revision=m['pi05']['revision'], local_dir='/weights/pi05_libero_finetuned')
d(m['xvla']['base'], revision=m['xvla']['base_revision'], local_dir='/weights/X-VLA-Pt')
d(m['xvla']['lora'], revision=m['xvla']['lora_revision'], local_dir='/weights/X-VLA-libero-long-peft')
" || fail "weight download from Hugging Face failed"
(cd /opt/weights && find . -type f -not -path '*/.cache/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) \
  > "${VLAR_PHASE0_DIR}/logs/weights.SHA256SUMS" || fail "weight checksums"
if fetch_tokenizer; then
  echo "tokenizer cache copied"
else
  echo "the gated PaliGemma tokenizer is not in ${VLAR_STORAGE_URL}/hf-tokenizer/ yet (the π0.5 rows will not run)"
fi
du -sh /opt/weights/* /opt/hf; df -h /
