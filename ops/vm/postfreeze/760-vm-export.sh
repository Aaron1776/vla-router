#!/usr/bin/env bash
# cap-seconds: 1800
# spend-phase: vmexport
# The VM export before its deletion (the user, 2026-11-27): COPIES ONLY, no other work: starts no container, no
# server, no worker, and never writes to the queue database or the journal.
#  1. The job queue's database (/var/lib/vlar/queue.sqlite with its -wal and -shm files, byte for byte), then, from
#     those copies in a scratch folder, one consolidated single-file copy, its integrity check, and a summary: jobs by
#     phase and status, attempt counts, and every job that had a failed attempt (its phase, attempts, status and last
#     error), i.e. the retry history (re-runs after preemptions show as released leases, not as attempts).
#  2. The system journal of every container unit: the workers (vlar-worker@*), both model servers (vlar-xvla,
#     vlar-pi05) and the spend guard (vlar-spend), all boots the journal still holds, as JSON lines (every field),
#     zstd-compressed, with each unit's entry count, first and last time and boots; plus the journal's boot list and
#     disk usage (what retention kept).
# Everything goes to a NEW prefix, <storage>/vm-export-2026-11-27/, with SHA256SUMS: the step refuses if that prefix
# already holds an object, copies each file with no-clobber, and reads the listing back.
set -uo pipefail
: "${VLAR_STORAGE_URL:?}"
DB="${VLAR_QUEUE_DB:-/var/lib/vlar/queue.sqlite}"
DEST="${VLAR_EXPORT_URL:-${VLAR_STORAGE_URL}/vm-export-2026-11-27}"
OUT="${VLAR_EXPORT_DIR:-/var/lib/vlar/vm-export-2026-11-27}"
JCTL="${VLAR_JOURNALCTL:-journalctl}"
fail() { echo "$*" > "${VLAR_REASON_FILE:-/dev/null}"; echo "$*"; exit 1; }

if gcloud storage ls "${DEST}/" >/dev/null 2>&1; then fail "${DEST}/ already holds objects: nothing copied"; fi
rm -rf "${OUT}"                                   # this step's own staging, from an interrupted attempt
mkdir -p "${OUT}/queue/raw" "${OUT}/journal" || fail "no staging folder ${OUT}"

# ---- 1. the queue database
[[ -s "${DB}" ]] || fail "no queue database at ${DB}"
for f in "${DB}" "${DB}-wal" "${DB}-shm"; do
  [[ -e "${f}" ]] && { cp -p "${f}" "${OUT}/queue/raw/" || fail "copy of ${f} failed"; }
done
python3 - "${OUT}/queue" "$(basename "${DB}")" <<'PY' || fail "queue database summary failed"
import json, os, shutil, sqlite3, sys, tempfile
out, name = sys.argv[1:3]
with tempfile.TemporaryDirectory() as tmp:          # work on a copy of the copies: raw/ stays byte for byte
    for f in os.listdir(f"{out}/raw"):
        shutil.copy2(f"{out}/raw/{f}", tmp)
    src = sqlite3.connect(f"{tmp}/{name}")           # the scratch copy (reads its WAL; may checkpoint it)
    dst = sqlite3.connect(f"{out}/queue-consolidated.sqlite")
    src.backup(dst)
    src.close()
dst.execute("PRAGMA journal_mode=DELETE")                 # one self-contained file, no -wal/-shm
integrity = dst.execute("PRAGMA integrity_check").fetchone()[0]
by = [dict(zip(("phase", "status", "jobs", "max_attempts"), r)) for r in dst.execute(
    "SELECT phase, status, COUNT(*), MAX(attempts) FROM jobs GROUP BY phase, status ORDER BY phase, status")]
retried = [dict(zip(("job_id", "plan_id", "phase", "tier", "status", "attempts", "last_error"), r)) for r in dst.execute(
    "SELECT job_id, plan_id, phase, tier, status, attempts, last_error FROM jobs "
    "WHERE attempts > 0 OR last_error IS NOT NULL ORDER BY phase, job_id")]
plans = [dict(zip(("plan_id", "phase", "priority"), r)) for r in dst.execute("SELECT plan_id, phase, priority FROM plans")]
dst.close()
with open(f"{out}/retries.jsonl", "w") as f:
    for r in retried:
        f.write(json.dumps(r) + "\n")
json.dump({"integrity_check": integrity, "plans": plans, "by_phase_status": by, "jobs_with_failed_attempts": len(retried),
           "raw_files": sorted(os.listdir(f"{out}/raw"))}, open(f"{out}/summary.json", "w"), indent=1)
print(f"queue: integrity {integrity}, {sum(r['jobs'] for r in by)} jobs, {len(retried)} with failed attempts")
sys.exit(0 if integrity == "ok" else 1)
PY

# ---- 2. the containers' journal
"${JCTL}" --no-pager --list-boots > "${OUT}/journal/boots.txt" 2>&1 || fail "journal boot list failed"
"${JCTL}" --no-pager --disk-usage > "${OUT}/journal/disk-usage.txt" 2>&1 || true
python3 - "${JCTL}" "${OUT}/journal" <<'PY' || fail "journal export failed"
import json, subprocess, sys
jctl, out = sys.argv[1:3]
units = ["vlar-worker@*", "vlar-xvla.service", "vlar-pi05.service", "vlar-spend.service"]
cmd = [jctl, "--no-pager", "-o", "json"] + [a for u in units for a in ("-u", u)]
j = subprocess.Popen(cmd, stdout=subprocess.PIPE)
z = subprocess.Popen(["zstd", "-q", "-6", "-T0", "-o", f"{out}/containers.json.zst"], stdin=subprocess.PIPE)
stats, bad = {}, 0
for line in j.stdout:
    z.stdin.write(line)
    try:
        e = json.loads(line)
    except ValueError:
        bad += 1
        continue
    u, t = e.get("_SYSTEMD_UNIT") or e.get("UNIT") or "?", int(e.get("__REALTIME_TIMESTAMP", 0))
    s = stats.setdefault(u, {"entries": 0, "first_us": t, "last_us": t, "boots": set()})
    s["entries"] += 1
    s["first_us"], s["last_us"] = min(s["first_us"], t), max(s["last_us"], t)
    s["boots"].add(e.get("_BOOT_ID", "?"))
z.stdin.close()
rj, rz = j.wait(), z.wait()
for s in stats.values():
    s["boots"] = sorted(s["boots"])
json.dump({"command": cmd, "journalctl_exit": rj, "zstd_exit": rz, "unparsed_lines": bad, "units": stats},
          open(f"{out}/containers-summary.json", "w"), indent=1)
print(f"journal: {sum(s['entries'] for s in stats.values())} entries from {len(stats)} units")
sys.exit(0 if rj == 0 and rz == 0 and stats else 1)
PY

# ---- 3. checksums, copy with no-clobber, read back
(cd "${OUT}" && find . -type f ! -name SHA256SUMS | LC_ALL=C sort | xargs sha256sum > SHA256SUMS) || fail "checksums failed"
while IFS= read -r f; do
  gcloud storage cp -n -q "${OUT}/${f#./}" "${DEST}/${f#./}" >/dev/null 2>&1 || fail "upload of ${f} failed"
done < <(cd "${OUT}" && find . -type f | LC_ALL=C sort)
want="$(cd "${OUT}" && find . -type f | wc -l | tr -d ' ')"
got="$(gcloud storage ls "${DEST}/**" 2>/dev/null | grep -c -v '/$')"
[[ "${got}" == "${want}" ]] || fail "the bucket lists ${got} export files, expected ${want}"
echo "exported ${want} files to ${DEST}/"
