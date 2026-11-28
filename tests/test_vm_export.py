"""The VM export before deletion (ops/vm/postfreeze/760-vm-export.sh), run on the Mac with stand-ins for gcloud and
journalctl: it copies the queue database (WAL included) and the containers' journal to a new prefix, leaves the
source database byte for byte, writes correct checksums, and refuses a prefix that already holds anything."""
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "ops/vm/postfreeze/760-vm-export.sh"
pytestmark = pytest.mark.skipif(not (shutil.which("zstd") and shutil.which("sha256sum")), reason="zstd/sha256sum")

FAKE_GCLOUD = r"""#!/usr/bin/env bash
# gcloud stand-in: gs://fake/<key> lives at $FAKE_BUCKET/<key>
loc() { echo "${FAKE_BUCKET}/${1#gs://fake/}"; }
[[ "$1 $2" == "storage ls" ]] || [[ "$1 $2" == "storage cp" ]] || exit 9
if [[ "$2" == ls ]]; then
  p="$(loc "${3%\*\*}")"; [[ -d "$p" ]] || exit 1
  n=$(find "$p" -type f | wc -l); (( n > 0 )) || exit 1
  find "$p" -type f | sed "s#^${FAKE_BUCKET}/#gs://fake/#"; exit 0
fi
shift 2; args=(); for a in "$@"; do [[ "$a" == -* ]] || args+=("$a"); done
d="$(loc "${args[1]}")"; [[ -e "$d" ]] && exit 0          # -n: no-clobber
mkdir -p "$(dirname "$d")" && cp "${args[0]}" "$d"
"""

FAKE_JOURNALCTL = r"""#!/usr/bin/env bash
case " $* " in
  *" --list-boots "*) echo " -1 aaaa Mon 2026-11-25 06:29:00 UTC"; echo "  0 bbbb Thu 2026-11-27 11:40:00 UTC" ;;
  *" --disk-usage "*) echo "Archived and active journals take up 1.2G in the file system." ;;
  *" -o json "*)
    echo '{"_SYSTEMD_UNIT":"vlar-pi05.service","__REALTIME_TIMESTAMP":"1795700000000000","_BOOT_ID":"aaaa","MESSAGE":"batch 4"}'
    echo '{"_SYSTEMD_UNIT":"vlar-worker@3.service","__REALTIME_TIMESTAMP":"1795700000000001","_BOOT_ID":"aaaa","MESSAGE":"job"}'
    echo '{"_SYSTEMD_UNIT":"vlar-pi05.service","__REALTIME_TIMESTAMP":"1795790000000000","_BOOT_ID":"bbbb","MESSAGE":"batch 2"}' ;;
  *) exit 3 ;;
esac
"""


def make_queue(path: Path):
    db = sqlite3.connect(path, isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA wal_autocheckpoint=0")          # keep the rows below in the WAL only
    db.executescript("""
        CREATE TABLE jobs (job_id TEXT PRIMARY KEY, plan_id TEXT, priority INTEGER, tier INTEGER, ord INTEGER,
                           phase TEXT, body TEXT, status TEXT, lease_owner TEXT, lease_expires REAL,
                           attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT);
        CREATE TABLE plans (plan_id TEXT PRIMARY KEY, phase TEXT, priority INTEGER, params TEXT);
    """)
    db.execute("INSERT INTO plans VALUES ('p1', 'phase1b', 0, '{}')")
    db.executemany("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   [("j1", "p1", 0, 1, 0, "phase1b", "{}", "done", None, None, 0, None),
                    ("j2", "p1", 0, 1, 1, "phase1b", "{}", "done", None, None, 1, "Traceback: lease lost"),
                    ("j3", "p1", 0, 1, 2, "phase1b", "{}", "done", None, None, 0, None)])
    return db                                           # left open, as the VM's last writer left it


def run(tmp_path, bucket, db_path):
    bins = tmp_path / "bin"
    bins.mkdir(exist_ok=True)
    for name, body in (("gcloud", FAKE_GCLOUD), ("journalctl", FAKE_JOURNALCTL)):
        (bins / name).write_text(body)
        (bins / name).chmod(0o755)
    env = {**os.environ, "PATH": f"{bins}:{os.environ['PATH']}", "FAKE_BUCKET": str(bucket),
           "VLAR_STORAGE_URL": "gs://fake/vla-router", "VLAR_QUEUE_DB": str(db_path),
           "VLAR_EXPORT_DIR": str(tmp_path / "stage")}
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_the_export_copies_the_queue_and_the_journal_and_leaves_the_source_alone(tmp_path):
    vm = tmp_path / "vm"
    vm.mkdir()
    db = make_queue(vm / "queue.sqlite")
    assert (vm / "queue.sqlite-wal").stat().st_size > 0
    before = {p.name: sha(p) for p in vm.iterdir()}
    bucket = tmp_path / "bucket"
    r = run(tmp_path, bucket, vm / "queue.sqlite")
    assert r.returncode == 0, r.stdout + r.stderr
    assert {p.name: sha(p) for p in vm.iterdir()} == before                 # the source, byte for byte
    out = bucket / "vla-router/vm-export-2026-11-27"
    raw = out / "queue/raw"
    assert {p.name: sha(p) for p in raw.iterdir()} == before                # raw copies equal the source files
    c = sqlite3.connect(out / "queue/queue-consolidated.sqlite")
    assert c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 3        # rows only in the WAL are in
    s = json.loads((out / "queue/summary.json").read_text())
    assert s["integrity_check"] == "ok" and s["jobs_with_failed_attempts"] == 1
    assert [json.loads(l)["job_id"] for l in (out / "queue/retries.jsonl").read_text().splitlines()] == ["j2"]
    j = json.loads((out / "journal/containers-summary.json").read_text())
    assert j["units"]["vlar-pi05.service"]["entries"] == 2 and j["units"]["vlar-pi05.service"]["boots"] == ["aaaa", "bbbb"]
    lines = subprocess.run(["zstd", "-dc", str(out / "journal/containers.json.zst")], capture_output=True,
                           text=True).stdout.splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["MESSAGE"] == "batch 4"
    sums = dict(reversed(l.split(None, 1)) for l in (out / "SHA256SUMS").read_text().splitlines())
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    assert sorted(k.removeprefix("./") for k in sums) == files
    assert all(sha(out / k.removeprefix("./")) == v for k, v in sums.items())
    db.close()


def test_a_prefix_that_already_holds_anything_is_refused_and_left_as_it_was(tmp_path):
    vm = tmp_path / "vm"
    vm.mkdir()
    make_queue(vm / "queue.sqlite").close()
    bucket = tmp_path / "bucket"
    old = bucket / "vla-router/vm-export-2026-11-27/SHA256SUMS"
    old.parent.mkdir(parents=True)
    old.write_text("earlier\n")
    r = run(tmp_path, bucket, vm / "queue.sqlite")
    assert r.returncode == 1 and "already holds objects" in r.stdout
    assert old.read_text() == "earlier\n" and len(list(old.parent.rglob("*"))) == 1


def test_no_queue_database_fails_before_anything_is_uploaded(tmp_path):
    bucket = tmp_path / "bucket"
    r = run(tmp_path, bucket, tmp_path / "missing.sqlite")
    assert r.returncode == 1 and "no queue database" in r.stdout and not bucket.exists()
