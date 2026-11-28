"""Download bucket prefixes into <dest>/<object name> (the user, 2026-11-26: the end-of-Phase-1 download, overnight).
Bucket reads only. Copies only objects that are missing locally or differ in size or CRC32C, so running it again after
more objects land fetches just those; it never deletes a local file. --method auto (default): `gcloud storage rsync -r`
per prefix while the gcloud CLI's sign-in works, else (or if a CLI copy fails) application-default credentials through
the storage client; cli or adc forces one. --exclude PREFIX skips objects under it (prefixes already downloaded and
verified). Run from the repo root after `source ops/gcp/env.sh && export PROJECT_ID BUCKET`:
  .venv-ops/bin/python ops/gcp/sync_download.py --dest <dir> --prefix vla-router --exclude vla-router/images/base
Prints one JSON line per prefix, then a summary line; exit 0 when every prefix synced, else 1. Then check the download
with verify_download.py, which compares every file with the bucket."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def norm(prefix: str) -> str:
    return prefix.strip("/") + "/"


def excluded(name: str, excludes: list) -> bool:
    return any(name.startswith(norm(e)) for e in excludes)


def cli_exclude_regex(prefix: str, excludes: list) -> str | None:
    """`gcloud storage rsync -x` takes one Python regex matched against names relative to the source URL."""
    rel = [norm(e)[len(norm(prefix)):] for e in excludes if norm(e).startswith(norm(prefix))]
    return "|".join("^" + re.escape(r) for r in rel) or None


def to_fetch(objects: list, dest: Path, excludes: list, local_crc) -> list:
    """`objects`: (name, size, crc32c) from the bucket. Those not already at dest with the same size and CRC32C."""
    out = []
    for name, size, crc in objects:
        if name.endswith("/") or excluded(name, excludes):
            continue
        p = dest / name
        if p.is_file() and p.stat().st_size == size and local_crc(p) == crc:
            continue
        out.append(name)
    return out


def fetch_all(blobs: dict, names: list, dest: Path, workers: int = 8, tries: int = 3) -> dict:
    """Download each named blob to a temporary file beside its target (CRC32C-checked), then rename it into place."""
    def one(name: str):
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        for i in range(tries):
            try:
                blobs[name].download_to_filename(str(tmp), checksum="crc32c")
                os.replace(tmp, p)
                return None
            except Exception as e:                       # noqa: BLE001 - reported, retried, never swallowed
                err = f"{type(e).__name__}: {e}"[:300]
                tmp.unlink(missing_ok=True)
                time.sleep(2 * (i + 1))
        return name, err

    with ThreadPoolExecutor(max_workers=workers) as ex:
        failed = [r for r in ex.map(one, names) if r is not None]
    return {"fetched": len(names) - len(failed), "bytes": sum(blobs[n].size for n in names),
            "failed": len(failed), "failures": failed[:20]}


def local_crc32c(p: Path) -> str:
    import google_crc32c
    c = google_crc32c.Checksum()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            c.update(chunk)
    return base64.b64encode(c.digest()).decode()


def cli_signed_in() -> bool:
    try:
        r = subprocess.run(["gcloud", "auth", "print-access-token", "--quiet"], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
                           env={**os.environ, "CLOUDSDK_CORE_DISABLE_PROMPTS": "1"})
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def sync_cli(bucket: str, prefix: str, dest: Path, excludes: list) -> dict:
    t = time.time()
    (dest / prefix.strip("/")).mkdir(parents=True, exist_ok=True)
    cmd = ["gcloud", "storage", "rsync", "-r", f"gs://{bucket}/{prefix.strip('/')}", str(dest / prefix.strip("/"))]
    rx = cli_exclude_regex(prefix, excludes)
    if rx:
        cmd[4:4] = ["-x", rx]
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                       env={**os.environ, "CLOUDSDK_CORE_DISABLE_PROMPTS": "1"})
    lines = r.stderr.strip().splitlines() if r.returncode else []
    errors = [x[:300] for x in lines if "ERROR" in x or "rror:" in x][:5]     # gcloud prints them mid-stream
    return {"method": "cli", "rc": r.returncode, "s": round(time.time() - t), "errors": errors,
            "stderr_tail": lines[-3:]}


def sync_adc(bkt, prefix: str, dest: Path, excludes: list) -> dict:
    t = time.time()
    blobs = {b.name: b for b in bkt.list_blobs(prefix=norm(prefix))}
    names = to_fetch([(b.name, b.size, b.crc32c) for b in blobs.values()], dest, excludes, local_crc32c)
    r = fetch_all(blobs, names, dest)
    return {"method": "adc", "rc": 0 if r["failed"] == 0 else 1, "listed": len(blobs), **r, "s": round(time.time() - t)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--prefix", action="append", required=True)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--method", choices=["auto", "cli", "adc"], default="auto")
    a = ap.parse_args()
    dest = Path(a.dest)
    bkt = None
    use_cli = a.method == "cli" or (a.method == "auto" and cli_signed_in())
    ok = True
    for pre in a.prefix:
        r = sync_cli(os.environ["BUCKET"], pre, dest, a.exclude) if use_cli else None
        if r is None or (r["rc"] != 0 and a.method == "auto"):
            if bkt is None:
                from google.cloud import storage
                bkt = storage.Client(project=os.environ["PROJECT_ID"]).bucket(os.environ["BUCKET"])
            first = r
            r = sync_adc(bkt, pre, dest, a.exclude)
            if first is not None:
                r["after_cli_failure"] = first
        ok &= r["rc"] == 0
        print(json.dumps({"prefix": pre, **r}), flush=True)
    print(json.dumps({"ok": ok, "cli_first": use_cli}), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
