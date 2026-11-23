"""The simulator worker loop (docs/SPEC.md §12): claim a job, run it, upload its files, then mark it done.

A runner returns (artifacts, summary): each artifact a mapping with `key` (bucket key), `path` (local file),
`sha256` and `size` (vlar_harness.collect.Runner). Every artifact is uploaded and checked before
`JobQueue.complete` writes the job's result marker, so a result in the bucket always means its snapshots, images
and records are there too (snapshot sync). A preempted worker leaves no marker; the job is re-run on resume and
rewrites the same keys (jobs are idempotent). A failing job is retried up to the queue's attempt limit.
The VM's worker process (vlar_ops.vm_worker) runs this loop with the production runner.
"""

from __future__ import annotations

import hashlib
import sys
import traceback
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from .plan import Job
from .queue import JobQueue, LeaseLost
from .storage import Storage


class UploadMismatch(RuntimeError):
    """A file in the bucket does not match what the job wrote."""


def upload(storage: Storage, artifacts: Sequence[Mapping]) -> None:
    for a in artifacts:
        data = Path(a["path"]).read_bytes()
        if len(data) != a["size"] or hashlib.sha256(data).hexdigest() != a["sha256"]:
            raise UploadMismatch(f"{a['path']} changed after the job listed it")
        storage.put_bytes(a["key"], data)
        back = storage.get_bytes(a["key"])
        if back is None or hashlib.sha256(back).hexdigest() != a["sha256"]:
            raise UploadMismatch(f"{a['key']} did not arrive intact")


def run_worker(queue: JobQueue, storage: Storage, runner: Callable, worker_id: str, *,
               stop: Callable[[], bool] = lambda: False, max_jobs: Optional[int] = None,
               phases: Optional[Sequence[str]] = None, refuse: Optional[Callable[[Job], Optional[str]]] = None,
               meta: Optional[Mapping] = None, cleanup: Optional[Callable[[Job], None]] = None) -> dict[str, int]:
    """Process jobs until none is left, `stop()` is true (spend guard, operator), `max_jobs` were taken, or a job
    this worker cannot run comes up.

    `phases` limits the claims to those phases. `refuse(job)` says why this worker cannot run a job (for example,
    the job needs other harness options): the job goes back to pending untouched, spending none of its attempts,
    and the worker stops instead of claiming it again and again. `meta` goes into every result marker (provenance,
    worker id). `cleanup(job)` runs after every job, whatever happened. A SystemExit or KeyboardInterrupt during
    a job (SIGTERM, Ctrl-C) releases the job, so it leaves no marker and no lease, and propagates. The lease is
    renewed while a job runs (amendment 40); a job whose lease was nevertheless lost to another worker is counted
    as "lost" and changes nothing in the queue.
    """
    stats = {"done": 0, "done_local": 0, "retry": 0, "failed": 0, "refused": 0, "lost": 0}
    taken = 0
    while not stop() and (max_jobs is None or taken < max_jobs):
        job = queue.claim(worker_id, phases=phases)
        if job is None:
            break
        reason = refuse(job) if refuse is not None else None
        if reason:
            queue.release(job, worker_id=worker_id)
            stats["refused"] += 1
            print(f"worker {worker_id}: refused job {job.job_id} ({job.phase}): {reason}", file=sys.stderr, flush=True)
            break
        taken += 1
        try:
            with queue.keep_lease(job, worker_id):
                artifacts, summary = runner(job)
                upload(storage, artifacts)
            record = {"summary": dict(summary),
                      "artifacts": [{k: a[k] for k in ("key", "sha256", "size")} for a in artifacts]}
            stats["done" if queue.complete(job, record, meta, worker_id=worker_id) else "done_local"] += 1
        except LeaseLost:
            stats["lost"] += 1
        except Exception:
            status = queue.fail(job, traceback.format_exc(), worker_id=worker_id)
            stats["lost" if status == "lost" else "failed" if status == "failed" else "retry"] += 1
        except BaseException:
            queue.release(job, worker_id=worker_id)   # only a job still leased by us: a completed one keeps its result
            raise
        finally:
            if cleanup is not None:
                cleanup(job)
    return stats
