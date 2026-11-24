"""Resumable job queue for parallel simulator workers on one VM (docs/SPEC.md §12).

Local state is a SQLite file shared by the worker processes. The durable record is the
bucket: a job is done when its result object exists there. Each result is written to
local disk first, then uploaded, then marked done, so a preempted VM loses at most the
jobs in flight. On restart, `resume()` releases stale leases, re-uploads local results
that never reached the bucket, and marks done every job whose result is in the bucket.

Leases (amendment 40): `complete`, `fail` and `release` change a job only while the caller still holds its lease
(`lease_owner`), and a worker renews its lease while a job runs (`keep_lease`). A job that outlived its lease and was
claimed by another worker therefore cannot be reset, failed or marked done by the first one.

Progress-gated leases (the user, 2026-11-24; PROPOSALS §0.6): a lease lasts LEASE_S (10 minutes) and is renewed every
RENEW_EVERY_S (60 s), but only when the job's progress token changed since the last renewal. A worker frozen, or
stuck on a request that never returns, stops renewing and loses the job within LEASE_S + RENEW_EVERY_S. `reap`
counts a dead or frozen worker's jobs as failed attempts (the supervisor, vlar_ops.supervisor).
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .plan import Job, Plan
from .storage import Storage

RESULTS_PREFIX = "results"
LEASE_S = 600.0
RENEW_EVERY_S = 60.0


class LeaseLost(RuntimeError):
    """The job's lease expired and another worker claimed it: this worker's result is not recorded."""


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

    def claim(self, worker_id: str, lease_s: float = LEASE_S,
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

    def holds(self, job: Job, worker_id: str) -> bool:
        return self._db.execute("SELECT 1 FROM jobs WHERE job_id=? AND status='leased' AND lease_owner=?",
                                (job.job_id, worker_id)).fetchone() is not None

    def complete(self, job: Job, result: Mapping[str, Any], meta: Optional[Mapping[str, Any]] = None, *,
                 worker_id: str) -> bool:
        """Persist a job's result locally, upload it, mark it done. Returns True once it is in the bucket. Raises
        LeaseLost (recording nothing) if `worker_id` no longer holds the job's lease."""
        if not self.holds(job, worker_id):
            raise LeaseLost(f"{worker_id} no longer holds {job.job_id}")
        record = {"job": json.loads(job.to_json()), "result": result, "meta": dict(meta or {}),
                  "finished_at": time.time()}
        path = self._local_path(job.phase, job.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(record, sort_keys=True, default=_json_default))
        with self._tx():
            if self._db.execute("UPDATE jobs SET status='done_local', lease_owner=NULL, lease_expires=NULL "
                                "WHERE job_id=? AND status='leased' AND lease_owner=?",
                                (job.job_id, worker_id)).rowcount != 1:
                tmp.unlink(missing_ok=True)
                raise LeaseLost(f"{worker_id} lost {job.job_id} while completing it")
            tmp.replace(path)
        return self._upload(job.phase, job.job_id)

    def fail(self, job: Job, error: str, *, worker_id: str) -> str:
        """Count a failed attempt (back to pending, or failed after max_attempts). Returns the new status, or "lost"
        (changing nothing) if `worker_id` no longer holds the job's lease."""
        with self._tx():
            row = self._db.execute("SELECT attempts FROM jobs WHERE job_id=? AND status='leased' AND lease_owner=?",
                                   (job.job_id, worker_id)).fetchone()
            if row is None:
                return "lost"
            attempts = row[0] + 1
            status = "failed" if attempts >= self.max_attempts else "pending"
            self._db.execute("UPDATE jobs SET status=?, attempts=?, last_error=?, lease_owner=NULL, "
                             "lease_expires=NULL WHERE job_id=?", (status, attempts, error[-4000:], job.job_id))
        return status

    def renew(self, job: Job, worker_id: str, lease_s: float = LEASE_S) -> bool:
        """Extend `worker_id`'s lease on the job to now + lease_s; False if it no longer holds it."""
        with self._tx():
            return self._db.execute("UPDATE jobs SET lease_expires=? WHERE job_id=? AND status='leased' AND "
                                    "lease_owner=?", (time.time() + lease_s, job.job_id, worker_id)).rowcount == 1

    def leases(self, worker_id: str) -> list[tuple[Job, float]]:
        """The jobs `worker_id` holds, with their lease expiry times."""
        rows = self._db.execute("SELECT body, lease_expires FROM jobs WHERE status='leased' AND lease_owner=?",
                                (worker_id,)).fetchall()
        return [(Job.from_json(b), float(e)) for b, e in rows]

    def owner(self, job_id: str) -> tuple[Optional[str], Optional[str]]:
        """(status, lease owner) of a job, (None, None) if unknown."""
        row = self._db.execute("SELECT status, lease_owner FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def reap(self, worker_id: str, reason: str) -> list[tuple[str, str]]:
        """Count every job `worker_id` still holds as a failed attempt (back to pending, or failed after max_attempts),
        for a worker that died or was stopped while frozen. Returns [(job_id, new status)]."""
        return [(job.job_id, self.fail(job, f"worker {worker_id}: {reason}", worker_id=worker_id))
                for job, _ in self.leases(worker_id)]

    @contextmanager
    def keep_lease(self, job: Job, worker_id: str, *, progress: Callable[[], Any], lease_s: Optional[float] = None,
                   every_s: Optional[float] = None):
        """Every `every_s` (default RENEW_EVERY_S), renew the lease to now + `lease_s` (default LEASE_S) if
        `progress()` returned something different from the last check; otherwise leave it to run out. Runs in a
        thread with its own database connection. Yields a dict: "lost" turns true if a renewal found the lease gone,
        "renewals" and "stalls" count the checks with and without progress."""
        lease_s = LEASE_S if lease_s is None else lease_s
        every_s = RENEW_EVERY_S if every_s is None else every_s
        state, stop = {"lost": False, "renewals": 0, "stalls": 0}, threading.Event()

        def loop():
            q = JobQueue(self.db_path, self.storage, self.local_dir, self.max_attempts)
            last = progress()
            try:
                while not stop.wait(every_s):
                    now = progress()
                    if now == last:
                        state["stalls"] += 1
                        continue
                    last = now
                    if q.renew(job, worker_id, lease_s):
                        state["renewals"] += 1
                    else:
                        state["lost"] = True
                        return
            finally:
                q.close()
        t = threading.Thread(target=loop, name=f"lease-{job.job_id}", daemon=True)
        t.start()
        try:
            yield state
        finally:
            stop.set()
            t.join()

    def release_all_leases(self) -> int:
        """Return every leased job to pending (VM restart, or abandoning jobs at a hard spend stop)."""
        with self._tx():
            return self._db.execute("UPDATE jobs SET status='pending', lease_owner=NULL, lease_expires=NULL "
                                    "WHERE status='leased'").rowcount

    def release(self, job: Job, *, worker_id: str) -> bool:
        """Give a claimed job back untouched (clean abandon, e.g. on a hard spend stop); only by its lease holder."""
        with self._tx():
            return self._db.execute("UPDATE jobs SET status='pending', lease_owner=NULL, lease_expires=NULL "
                                    "WHERE job_id=? AND status='leased' AND lease_owner=?",
                                    (job.job_id, worker_id)).rowcount == 1

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
               meta: Optional[Callable[[], Mapping[str, Any]]] = None,
               progress: Callable[[], Any] = lambda: None) -> int:
    """Claim and run jobs until the queue is empty or `should_stop()` (spend guard) says stop.

    The stop check happens before each claim, so in-progress jobs always finish. Leases are renewed only when
    `progress()` changes (default: never, so a job keeps only the lease its claim gave it).
    """
    wid = worker_id or default_worker_id()
    n = 0
    while not should_stop():
        job = queue.claim(wid)
        if job is None:
            break
        started = time.time()
        try:
            with queue.keep_lease(job, wid, progress=progress):
                result = handler(job)
        except Exception:
            queue.fail(job, traceback.format_exc(), worker_id=wid)
            continue
        try:
            queue.complete(job, result, {"worker": wid, "started_at": started, **(meta() if meta else {})},
                           worker_id=wid)
        except LeaseLost:
            continue
        n += 1
    return n
