#!/usr/bin/env bash
# cap-seconds: 1800
# The Spark's step 020 (instead of ops/vm/phase0/steps/020-weights.sh): the weights already on this machine at the
# config's pinned revisions, hard-linked (no copy, no download) into /opt/weights's stand-in; the gated tokenizer's
# cache folder from the local Hugging Face cache (only that model folder: no token file); SHA-256 of every weight file
# in logs/weights.SHA256SUMS, comparable with the L4's.
source /opt/vla-router/ops/vm/phase0/lib.sh
W=/opt/weights; mkdir -p "${W}" /opt/hf/hub
: "${VLAR_SPARK_XVLA_WEIGHTS:?folder with X-VLA-Pt and X-VLA-libero-long-peft}" "${VLAR_SPARK_HF_HUB:?local HF hub cache}"
link() { [[ -e "${W}/$2" ]] || cp -rlL "$1" "${W}/$2" || fail "cannot link $1"; }
link "${VLAR_SPARK_XVLA_WEIGHTS}/X-VLA-Pt" X-VLA-Pt
link "${VLAR_SPARK_XVLA_WEIGHTS}/X-VLA-libero-long-peft" X-VLA-libero-long-peft
rev="$(python3 -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["models"]["pi05"]["revision"])' \
  /opt/vla-router/src/vlar_harness/configs/harness-v1.yaml 2>/dev/null || echo 7152f13e34c6c115d808aea0e5d08aca74eac14b)"
link "${VLAR_SPARK_HF_HUB}/models--lerobot--pi05_libero_finetuned/snapshots/${rev}" pi05_libero_finetuned
tok=models--google--paligemma-3b-pt-224
[[ -d "/opt/hf/hub/${tok}" ]] || cp -al "${VLAR_SPARK_HF_HUB}/${tok}" "/opt/hf/hub/${tok}" || fail "tokenizer cache"
(cd "${W}" && find . -type f -not -path '*/.cache/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) \
  > "${VLAR_PHASE0_DIR}/logs/weights.SHA256SUMS" || fail "weight checksums"
wc -l < "${VLAR_PHASE0_DIR}/logs/weights.SHA256SUMS" | sed 's/^/weight files: /'; du -sh "${W}"/* /opt/hf
