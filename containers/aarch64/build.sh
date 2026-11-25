#!/usr/bin/env bash
# containers/aarch64/build.sh COMMIT: the harness's three images for linux/aarch64, vlar-{sim,pi05,xvla}:<full commit>,
# built offline on the Spark. The context is `git archive` of COMMIT plus this folder from the working tree, the X-VLA
# checkout (third_party/X-VLA at 6c4ecfb8) and the LIBERO assets; Python packages come from the wheelhouse, checked
# against the aarch64 locks. Writes $VLAR_SPARK_DATA/etc/image-digests.json ({"commit", "sim", "pi05", "xvla"}: Docker
# image IDs, as ops/vm/load_images.sh records them) and prints it.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
C="$(git -C "$REPO" rev-parse "${1:?usage: build.sh COMMIT}^{commit}")"
DATA="${VLAR_SPARK_DATA:-$HOME/work/vlar-data}"
WH="$DATA/wheelhouse-aarch64"; ASSETS="$DATA/libero-assets"; X="$REPO/third_party/X-VLA"
for p in "$WH" "$ASSETS" "$X/.git"; do [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }; done
CTX="$DATA/build/$C"; rm -rf "$CTX"; mkdir -p "$CTX"
git -C "$REPO" archive "$C" | tar -x -C "$CTX"
rm -rf "$CTX/containers/aarch64"; cp -a "$REPO/containers/aarch64" "$CTX/containers/aarch64"
cp -a "$X" "$CTX/xvla-src"
cp -a "$ASSETS" "$CTX/libero-assets"; rm -rf "$CTX/libero-assets/.cache"
docker image inspect vlar-aarch64-base >/dev/null 2>&1 \
  || docker build -t vlar-aarch64-base -f "$REPO/containers/aarch64/Dockerfile.base" "$REPO/containers/aarch64"
for t in sim pi05 xvla; do
  echo "== vlar-$t:$C"
  docker build --network none --build-context wh="$WH" --build-arg GIT_COMMIT="$C" --target "$t" \
    -t "vlar-$t:$C" -f "$CTX/containers/aarch64/Dockerfile" "$CTX"
done
mkdir -p "$DATA/etc"
python3 - "$C" "$DATA/etc/image-digests.json" <<'PY'
import json, subprocess, sys
c, out = sys.argv[1:]
d = {"commit": c, "platform": "linux/aarch64",
     **{t: subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", f"vlar-{t}:{c}"], capture_output=True,
                          text=True, check=True).stdout.strip() for t in ("sim", "pi05", "xvla")}}
open(out, "w").write(json.dumps(d, indent=1) + "\n")
print(json.dumps(d, indent=1))
PY
rm -rf "$CTX"
