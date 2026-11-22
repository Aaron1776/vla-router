"""The simulator worker loop (docs/SPEC.md §12): claim a job, run it, upload its files, then mark it done.

A runner returns (artifacts, summary): each artifact a mapping with `key` (bucket key), `path` (local file),
`sha256` and `size` (vlar_harness.collect.Runner). Every artifact is uploaded and checked before
`JobQueue.complete` writes the job's result marker, so a result in the bucket always means its snapshots, images
and records are there too (snapshot sync). A preempted worker leaves no marker; the job is re-run on resume and
rewrites the same keys (jobs are idempotent). A failing job is retried up to the queue's attempt limit.
"""

from __future__ import annotations

import hashlib
import traceback
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from .queue import JobQueue
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
               stop: Callable[[], bool] = lambda: False, max_jobs: Optional[int] = None) -> dict[str, int]:
    """Process jobs until none is left, `stop()` is true (spend guard, operator) or `max_jobs` were taken."""
    stats = {"done": 0, "done_local": 0, "retry": 0, "failed": 0}
    taken = 0
    while not stop() and (max_jobs is None or taken < max_jobs):
        job = queue.claim(worker_id)
        if job is None:
            break
        taken += 1
        try:
            artifacts, summary = runner(job)
            upload(storage, artifacts)
            record = {"summary": dict(summary),
                      "artifacts": [{k: a[k] for k in ("key", "sha256", "size")} for a in artifacts]}
            stats["done" if queue.complete(job, record) else "done_local"] += 1
        except Exception:
            status = queue.fail(job, traceback.format_exc())
            stats["failed" if status == "failed" else "retry"] += 1
    return stats
