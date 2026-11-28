#!/usr/bin/env bash
# cap-seconds: 7200
# VM re-run list, π0.5 rows (need the gated PaliGemma tokenizer): batch invariance with the real weights at the
# production geometry, bit for bit (amendments 24, 26: blocking), and the server against lerobot-eval's processing.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
if ! fetch_tokenizer; then
  for r in pi05-batch-invariance-l4 pi05-vs-lerobot-eval; do not_run "${r}" "the gated PaliGemma tokenizer is not on the VM"; done
  exit 0
fi
row pi05-batch-invariance-l4 docker run "${GPU_RUN[@]}" "vlar-pi05:${C}" python -m pytest -q -rs -p no:cacheprovider \
  tests/reference/test_batch_invariance.py -k pi05_real
row pi05-vs-lerobot-eval docker run "${GPU_RUN[@]}" "vlar-pi05:${C}" python -m pytest -q -rs -p no:cacheprovider \
  tests/reference/test_model_servers.py -k pi05
exit 0
