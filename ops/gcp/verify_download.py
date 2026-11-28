"""Check a download from the bucket file by file against the bucket's own checksums (the user, 2026-11-26): every
object under each --prefix must exist at <dest>/<object name> with the object's size and CRC32C, and its MD5 where the
bucket has one (composite objects have none); a local file under a prefix that the bucket does not hold is reported
as extra. Bucket reads only (object listing), application-default credentials. Run from the repo root after
`source ops/gcp/env.sh && export PROJECT_ID BUCKET`:
  .venv-ops/bin/python ops/gcp/verify_download.py --dest <dir> --prefix containers/<commit> --prefix vla-router/...
Prints one JSON line; exit 0 when every object matches and nothing is extra, else 1."""
import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path


def file_sums(path: Path, crc32c_cls) -> tuple[str, str]:
    """(CRC32C, MD5) of a file, both base64 as Cloud Storage reports them."""
    crc, md5 = crc32c_cls(), hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            crc.update(chunk)
            md5.update(chunk)
    return base64.b64encode(crc.digest()).decode(), base64.b64encode(md5.digest()).decode()


def compare(objects: list, dest: Path, prefixes: list, sums) -> dict:
    """`objects`: (name, size, crc32c, md5 or None) from the bucket; `sums(path)` -> (crc32c, md5)."""
    out = {"objects": len(objects), "bytes": sum(o[1] for o in objects), "md5_checked": 0, "missing": [],
           "size_mismatch": [], "crc32c_mismatch": [], "md5_mismatch": [], "extra": []}
    names = set()
    for name, size, crc, md5 in objects:
        names.add(name)
        p = dest / name
        if not p.is_file():
            out["missing"].append(name)
            continue
        if p.stat().st_size != size:
            out["size_mismatch"].append(name)
            continue
        c, m = sums(p)
        if c != crc:
            out["crc32c_mismatch"].append(name)
        if md5 is not None:
            out["md5_checked"] += 1
            if m != md5:
                out["md5_mismatch"].append(name)
    for pre in prefixes:
        root = dest / pre
        for dirpath, _, files in os.walk(root):
            for fn in files:
                rel = (Path(dirpath) / fn).relative_to(dest).as_posix()
                if rel not in names:
                    out["extra"].append(rel)
    bad = sum(len(out[k]) for k in ("missing", "size_mismatch", "crc32c_mismatch", "md5_mismatch", "extra"))
    out["ok"] = bad == 0 and out["objects"] > 0
    for k in ("missing", "size_mismatch", "crc32c_mismatch", "md5_mismatch", "extra"):
        out[f"n_{k}"] = len(out[k])
        out[k] = out[k][:20]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--prefix", action="append", required=True)
    a = ap.parse_args()
    import google_crc32c
    from google.cloud import storage
    bkt = storage.Client(project=os.environ["PROJECT_ID"]).bucket(os.environ["BUCKET"])
    objects = []
    for pre in a.prefix:
        for b in bkt.list_blobs(prefix=pre.rstrip("/") + "/"):
            if not b.name.endswith("/"):
                objects.append((b.name, b.size, b.crc32c, b.md5_hash))
    r = compare(objects, Path(a.dest), [p.rstrip("/") for p in a.prefix],
                lambda p: file_sums(p, google_crc32c.Checksum))
    print(json.dumps(r))
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
