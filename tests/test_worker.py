"""The worker loop (vlar_ops.worker): files reach the bucket, intact, before a job's result marker; failures
retry; a job interrupted before its marker is re-run after resume."""

import hashlib
from pathlib import Path

from vlar_ops.plan import episode_plan
from vlar_ops.queue import JobQueue, result_key
from vlar_ops.storage import LocalStorage
from vlar_ops.worker import run_worker


class LoggingStorage(LocalStorage):
    def __init__(self, root, corrupt=()):
        super().__init__(root)
        self.order, self.corrupt = [], set(corrupt)

    def put_bytes(self, key, data):
        self.order.append(key)
        super().put_bytes(key, b"x" + data if key in self.corrupt else data)


def files_runner(tmp: Path, fail_times=0):
    calls = {}

    def runner(job):
        calls[job.job_id] = calls.get(job.job_id, 0) + 1
        if calls[job.job_id] <= fail_times:
            raise RuntimeError("simulated crash")
        arts = []
        for name in ("snap.npz", "points.parquet"):
            p = tmp / "staging" / job.job_id / name
            p.parent.mkdir(parents=True, exist_ok=True)
            data = f"{job.job_id}:{name}".encode()
            p.write_bytes(data)
            arts.append({"key": f"files/{job.job_id}/{name}", "path": str(p),
                         "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
        return arts, {"ok": True}
    return runner, calls


def setup(tmp_path, **kw):
    storage = LoggingStorage(tmp_path / "bucket", **kw)
    q = JobQueue(tmp_path / "q.db", storage, tmp_path / "local")
    q.add_plan(episode_plan("base", tasks=[0, 1], inits=[0], namespace=10, mode="routed", profile="none",
                            record="full", order_seed=1))
    return storage, q


def test_files_are_uploaded_and_checked_before_the_marker(tmp_path):
    storage, q = setup(tmp_path)
    runner, _ = files_runner(tmp_path)
    assert run_worker(q, storage, runner, "w") == {"done": 2, "done_local": 0, "retry": 0, "failed": 0}
    for jid in q.done_ids():
        marker = result_key("base", jid)
        files = [k for k in storage.order if k.startswith(f"files/{jid}/")]
        assert len(files) == 2 and storage.order.index(marker) > max(storage.order.index(k) for k in files)
        assert storage.get_bytes(files[0]) is not None


def test_a_failing_job_is_retried_then_given_up(tmp_path):
    storage, q = setup(tmp_path)
    runner, calls = files_runner(tmp_path, fail_times=1)
    stats = run_worker(q, storage, runner, "w")
    assert stats["retry"] == 2 and stats["done"] == 2 and set(calls.values()) == {2}
    storage, q = setup(tmp_path / "again")
    runner, _ = files_runner(tmp_path / "again", fail_times=99)
    stats = run_worker(q, storage, runner, "w")
    assert stats["failed"] == 2 and q.counts()["failed"] == 2
    assert not any(k.startswith("results/") for k in storage.order)


def test_a_corrupted_upload_never_gets_a_marker(tmp_path):
    jobs = episode_plan("base", tasks=[0, 1], inits=[0], namespace=10, mode="routed", profile="none",
                        record="full", order_seed=1).jobs
    bad = {f"files/{j.job_id}/snap.npz" for j in jobs}
    storage, q = setup(tmp_path, corrupt=bad)
    runner, _ = files_runner(tmp_path)
    stats = run_worker(q, storage, runner, "w")
    assert stats["done"] == 0 and stats["retry"] + stats["failed"] > 0
    assert not any(k.startswith("results/") for k in storage.order)


def test_a_job_preempted_before_its_marker_is_rerun_after_resume(tmp_path):
    storage, q = setup(tmp_path)
    claimed = q.claim("w-dead")                       # the VM dies with this job in flight
    assert claimed is not None
    q.close()
    q = JobQueue(tmp_path / "q.db", storage, tmp_path / "local")
    assert q.resume()["released_leases"] == 1
    runner, calls = files_runner(tmp_path)
    assert run_worker(q, storage, runner, "w")["done"] == 2 and calls[claimed.job_id] == 1


def test_stop_is_checked_before_each_claim(tmp_path):
    storage, q = setup(tmp_path)
    runner, calls = files_runner(tmp_path)
    assert run_worker(q, storage, runner, "w", stop=lambda: len(calls) >= 1)["done"] == 1
    assert q.counts()["pending"] == 1
