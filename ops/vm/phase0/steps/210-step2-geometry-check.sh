#!/usr/bin/env bash
# cap-seconds: 7200
# After step 2, before parity (amendments 24, 35): the L4's bit-for-bit batch-invariance rows at the geometry step 2
# chose (the tests read it from the calibration file). Skipped when it is the servers' default 8 requests / 40 sample
# rows, which the re-run rows on these images already checked. Any failure stops the VM (nothing is debugged).
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
calfile require-write-once || fail "no write-once calibration: run step 2 (200-step2-measure) first"
geo="$(calfile get servers.batching.max_batch) $(calfile get servers.batching.xvla_sample_rows)"
if [[ "${geo}" == "8 40" ]]; then echo "geometry 8/40: already checked by the re-run rows on ${C}"; exit 0; fi
fetch_tokenizer || need_input "the gated PaliGemma tokenizer is not on the VM"
XT="pip install -q --require-hashes --no-deps --only-binary :all: -r containers/xvla-test.lock"
row xvla-batch-invariance-geometry docker run "${GPU_RUN[@]}" "${CAL_ENV[@]}" "vlar-xvla:${C}" sh -c \
  "${XT} && python -m pytest -q -rs -p no:cacheprovider tests/reference/test_batch_invariance.py -k xvla"
row pi05-batch-invariance-geometry docker run "${GPU_RUN[@]}" "${CAL_ENV[@]}" "vlar-pi05:${C}" python -m pytest -q \
  -rs -p no:cacheprovider tests/reference/test_batch_invariance.py -k pi05_real
for r in xvla-batch-invariance-geometry pi05-batch-invariance-geometry; do
  [[ "$(cat "${RESULTS}/${r}.status")" == passed ]] || fail "${r} failed at geometry ${geo} (logs/results/${C}/${r}.log)"
done
echo "batch invariance holds bit for bit at geometry ${geo}"
