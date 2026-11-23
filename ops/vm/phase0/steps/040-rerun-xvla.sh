#!/usr/bin/env bash
# cap-seconds: 7200
# VM re-run list, X-VLA rows: the server against the authors' handler with the real weights and LoRA, and batch
# invariance on the L4 at the production geometry, bit for bit (amendment 24: blocking). Starlette's test client
# (containers/xvla-test.lock) goes into the throwaway container only.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
XT="pip install -q --require-hashes --no-deps --only-binary :all: -r containers/xvla-test.lock"
row xvla-vs-authors docker run "${GPU_RUN[@]}" "vlar-xvla:${C}" sh -c \
  "${XT} && python -m pytest -q -rs -p no:cacheprovider tests/reference/test_model_servers.py -k xvla"
row xvla-batch-invariance-l4 docker run "${GPU_RUN[@]}" "vlar-xvla:${C}" sh -c \
  "${XT} && python -m pytest -q -rs -p no:cacheprovider tests/reference/test_batch_invariance.py -k xvla"
exit 0
