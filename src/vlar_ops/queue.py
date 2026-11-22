"""Resumable job queue for parallel simulator workers on one VM (docs/SPEC.md §12).

Local state is a SQLite file shared by the worker processes. The durable record is the
bucket: a job is done when its result object exists there. Each result is written to
local disk first, then uploaded, then marked done, so a preempted VM loses at most the
jobs in flight. On restart, `resume()` releases stale leases, re-uploads local results
that never reached the bucket, and marks done every job whose result is in the bucket.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .plan import Job, Plan
from .storage import Storage

RESULTS_PREFIX = "results"


def result_key(phase: str, jid: str) -> str:
    return f"{RESULTS_PREFIX}/{phase}/{jid}.json"


class JobQueue:
    def __init__(self, db_path: Path | str, storage: Storage, local_dir: Path | str, max_attempts: int = 3):
        self.db_path = Path(db_path)
        self.storage = storage
        self.local_dir = Path(local_dir)
        self.max_attempts = max_attempts
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.db_path, timeout=60, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=60000")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                priority INTEGER NOT NULL,
                tier INTEGER NOT NULL,
                ord INTEGER NOT NULL,
                phase TEXT NOT NULL,
                body TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'leased', 'done_local', 'done', 'failed')),
                lease_owner TEXT,
                lease_expires REAL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
            );
            CREATE INDEX IF NOT EXISTS jobs_next ON jobs (status, priority, tier, ord);
            CREATE TABLE IF NOT EXISTS plans (plan_id TEXT PRIMARY KEY, phase TEXT, priority INTEGER, params TEXT);
        """)

    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------------------ setup

    def add_plan(self, plan: Plan, priority: Optional[int] = None) -> int:
        """Insert a plan's jobs; already-known job ids are left as they are. Returns jobs added."""
        pid = plan.plan_id
        with self._tx():
            if priority is None:
                row = self._db.execute("SELECT COALESCE(MAX(priority), -1) + 1 FROM plans").fetchone()
                priority = int(row[0])
            self._db.execute("INSERT OR IGNORE INTO plans VALUES (?, ?, ?, ?)",
                             (pid, plan.phase, priority, json.dumps(plan.params, sort_keys=True)))
            before = self._count()
            self._db.executemany(
                "INSERT OR IGNORE INTO jobs (job_id, plan_id, priority, tier, ord, phase, body) VALUES (?,?,?,?,?,?,?)",
                [(j.job_id, pid, priority, j.tier, j.order, j.phase, j.to_json()) for j in plan.jobs])
            return self._count() - before

    def resume(self) -> dict[str, int]:
        """Call once at VM start, before any worker runs."""
        released = self.release_all_leases()
        uploaded = 0
        for jid, phase in self._db.execute("SELECT job_id, phase FROM jobs WHERE status='done_local'").fetchall():
            if self._upload(phase, jid):
                uploaded += 1
        marked = 0
        for (phase,) in self._db.execute("SELECT DISTINCT phase FROM jobs").fetchall():
            remote = {Path(k).stem for k in self.storage.list(f"{RESULTS_PREFIX}/{phase}/") if k.endswith(".json")}
            pending = [jid for (jid,) in self._db.execute(
                "SELECT job_id FROM jobs WHERE phase=? AND status != 'done'", (phase,)).fetchall()]
            hits = [jid for jid in pending if jid in remote]
            with self._tx():
                self._db.executemany("UPDATE jobs SET status='done', lease_owner=NULL WHERE job_id=?",
                                     [(j,) for j in hits])
            marked += len(hits)
        return {"released_leases": released, "uploaded_local": uploaded, "marked_done_from_bucket": marked}

    # ------------------------------------------------------------ worker side

    def claim(self, worker_id: str, lease_s: float = 3 * 3600,
              phases: Optional[Sequence[str]] = None) -> Optional[Job]:
        """The next job in order, optionally only from `phases` (workers started for one phase's options)."""
        now = time.time()
        where, args = "(status='pending' OR (status='leased' AND lease_expires < ?))", [now]
        if phases:
            where += f" AND phase IN ({','.join('?' * len(phases))})"
            args += list(phases)
        with self._tx():
            row = self._db.execute(f"SELECT job_id, body FROM jobs WHERE {where} ORDER BY priority, tier, ord LIMIT 1",
                                   args).fetchone()
            if row is None:
                return None
            self._db.execute("UPDATE jobs SET status='leased', lease_owner=?, lease_expires=? WHERE job_id=?",
                             (worker_id, now + lease_s, row[0]))
        return Job.from_json(row[1])

    def complete(self, job: Job, result: Mapping[str, Any], meta: Optional[Mapping[str, Any]] = None) -> bool:
        """Persist a job's result locally, upload it, mark it done. Returns True once it is in the bucket."""
        record = {"job": json.loads(job.to_json()), "result": result, "meta": dict(meta or {}),
                  "finished_at": time.time()}
        path = self._local_path(job.phase, job.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(record, sort_keys=True, default=_json_default))
        tmp.replace(path)
        with self._tx():
            self._db.execute("UPDATE jobs SET status='done_local', lease_owner=NULL WHERE job_id=?", (job.job_id,))
        return self._upload(job.phase, job.job_id)

    def fail(self, job: Job, error: str) -> str:
        with self._tx():
            attempts = self._db.execute("SELECT attempts FROM jobs WHERE job_id=?", (job.job_id,)).fetchone()[0] + 1
            status = "failed" if attempts >= self.max_attempts else "pending"
            self._db.execute("UPDATE jobs SET status=?, attempts=?, last_error=?, lease_owner=NULL, "
                             "lease_expires=NULL WHERE job_id=?", (status, attempts, error[-4000:], job.job_id))
        return status

    def release_all_leases(self) -> int:
        """Return every leased job to pending (VM restart, or abandoning jobs at a hard spend stop)."""
        with self._tx():
            return self._db.execute("UPDATE jobs SET status='pending', lease_owner=NULL, lease_expires=NULL "
                                    "WHERE status='leased'").rowcount

    def release(self, job: Job) -> None:
        """Give a claimed job back untouched (clean abandon, e.g. on a hard spend stop)."""
        with self._tx():
            self._db.execute("UPDATE jobs SET status='pending', lease_owner=NULL, lease_expires=NULL "
                             "WHERE job_id=? AND status='leased'", (job.job_id,))

    # ------------------------------------------------------------ status

    def counts(self, phase: Optional[str] = None) -> dict[str, int]:
        q = "SELECT status, COUNT(*) FROM jobs" + (" WHERE phase=?" if phase else "") + " GROUP BY status"
        rows = self._db.execute(q, (phase,) if phase else ()).fetchall()
        out = {s: 0 for s in ("pending", "leased", "done_local", "done", "failed")}
        out.update({s: n for s, n in rows})
        return out

    def in_flight(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM jobs WHERE status='leased'").fetchone()[0]

    def done_ids(self, phase: Optional[str] = None) -> set[str]:
        q = "SELECT job_id FROM jobs WHERE status='done'" + (" AND phase=?" if phase else "")
        return {r[0] for r in self._db.execute(q, (phase,) if phase else ()).fetchall()}

    # ------------------------------------------------------------ internals

    def _local_path(self, phase: str, jid: str) -> Path:
        return self.local_dir / result_key(phase, jid)

    def _upload(self, phase: str, jid: str) -> bool:
        path = self._local_path(phase, jid)
        if not path.exists():
            return False
        try:
            self.storage.put_file(result_key(phase, jid), path)
        except Exception:  # network blip: stays done_local, retried by resume() or the next sync
            return False
        with self._tx():
            self._db.execute("UPDATE jobs SET status='done' WHERE job_id=?", (jid,))
        return True

    def sync(self) -> int:
        """Retry uploads of results that are only on local disk."""
        n = 0
        for jid, phase in self._db.execute("SELECT job_id, phase FROM jobs WHERE status='done_local'").fetchall():
            n += self._upload(phase, jid)
        return n

    def _count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    def _tx(self):
        return _Tx(self._db)


class _Tx:
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def __enter__(self):
        self.db.execute("BEGIN IMMEDIATE")
        return self.db

    def __exit__(self, exc_type, exc, tb):
        self.db.execute("ROLLBACK" if exc_type else "COMMIT")
        return False


def _json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def run_worker(queue: JobQueue, handler: Callable[[Job], Mapping[str, Any]], *,
               should_stop: Callable[[], bool], worker_id: Optional[str] = None,
               meta: Optional[Callable[[], Mapping[str, Any]]] = None) -> int:
    """Claim and run jobs until the queue is empty or `should_stop()` (spend guard) says stop.

    The stop check happens before each claim, so in-progress jobs always finish.
    """
    wid = worker_id or default_worker_id()
    n = 0
    while not should_stop():
        job = queue.claim(wid)
        if job is None:
            break
        started = time.time()
        try:
            result = handler(job)
        except Exception:
            queue.fail(job, traceback.format_exc())
            continue
        queue.complete(job, result, {"worker": wid, "started_at": started, **(meta() if meta else {})})
        n += 1
    return n
