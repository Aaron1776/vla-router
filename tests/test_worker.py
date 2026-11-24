"""The worker loop (vlar_ops.worker): files reach the bucket, intact, before a job's result marker; failures
retry; a job interrupted before its marker is re-run after resume; refused or interrupted jobs go back untouched."""

import hashlib
import json
from pathlib import Path

import pytest

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
    assert run_worker(q, storage, runner, "w") == {"done": 2, "done_local": 0, "retry": 0, "failed": 0,
                                                   "refused": 0, "lost": 0}
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


def attempts(q):
    return dict(q._db.execute("SELECT job_id, attempts FROM jobs").fetchall())


def test_claims_can_be_limited_to_phases(tmp_path):
    storage, q = setup(tmp_path)
    q.add_plan(episode_plan("explore", tasks=[0], inits=[0], namespace=20, mode="routed", profile="none",
                            record="full", order_seed=1))
    runner, calls = files_runner(tmp_path)
    assert run_worker(q, storage, runner, "w", phases=["explore"])["done"] == 1
    assert q.counts("explore")["done"] == 1 and q.counts("base")["pending"] == 2


def test_a_refused_job_goes_back_untouched_and_the_worker_stops(tmp_path):
    storage, q = setup(tmp_path)
    runner, calls = files_runner(tmp_path)
    cleaned = []
    stats = run_worker(q, storage, runner, "w", refuse=lambda job: "needs other options", cleanup=cleaned.append)
    assert stats["refused"] == 1 and stats["done"] == 0 and not calls and not cleaned
    assert q.counts()["pending"] == 2 and set(attempts(q).values()) == {0}


@pytest.mark.parametrize("signal_exc", [KeyboardInterrupt, SystemExit])
def test_an_interrupted_job_is_released_without_a_marker(tmp_path, signal_exc):
    storage, q = setup(tmp_path)
    cleaned = []

    def runner(job):
        raise signal_exc()
    with pytest.raises(signal_exc):
        run_worker(q, storage, runner, "w", cleanup=lambda job: cleaned.append(job.job_id))
    assert q.counts()["pending"] == 2 and q.counts()["leased"] == 0 and set(attempts(q).values()) == {0}
    assert len(cleaned) == 1 and not any(k.startswith("results/") for k in storage.order)


def test_markers_carry_the_meta_and_staging_is_cleaned(tmp_path):
    storage, q = setup(tmp_path)
    runner, _ = files_runner(tmp_path)
    cleaned = []
    run_worker(q, storage, runner, "w", meta={"worker_id": "w", "provenance": {"harness_version": "x"}},
               cleanup=lambda job: cleaned.append(job.job_id))
    for jid in q.done_ids():
        marker = json.loads(storage.get_bytes(result_key("base", jid)))
        assert marker["meta"] == {"worker_id": "w", "provenance": {"harness_version": "x"}}
    assert sorted(cleaned) == sorted(q.done_ids())


# ---------------------------------------------------------------- leases (amendment 40)

def test_overrun_job_cannot_be_reset_or_completed_by_its_old_worker(tmp_path):
    """The review's case: A's job outlives its lease, B claims it, then A fails (or finishes): A changes nothing,
    the job stays B's (a third worker cannot claim it), and B's result is the one recorded."""
    import time

    from vlar_ops.queue import LeaseLost
    storage, q = setup(tmp_path)
    a = q.claim("A", lease_s=0.05)
    time.sleep(0.1)                                            # A overruns its lease
    b = q.claim("B")
    assert b.job_id == a.job_id
    assert q.fail(a, "A crashed late", worker_id="A") == "lost"
    c = q.claim("C")
    assert c is None or c.job_id != a.job_id                   # still B's: not handed to a third worker
    with pytest.raises(LeaseLost):
        q.complete(a, {"from": "A"}, worker_id="A")
    assert not q.release(a, worker_id="A")
    assert q.complete(b, {"from": "B"}, worker_id="B")
    rec = json.loads(storage.get_bytes(result_key(b.phase, b.job_id)))
    assert rec["result"] == {"from": "B"}
    row = q._db.execute("SELECT status, attempts FROM jobs WHERE job_id=?", (a.job_id,)).fetchone()
    assert row == ("done", 0)


def test_a_running_job_renews_its_lease(tmp_path):
    import time
    storage, q = setup(tmp_path)
    job = q.claim("A", lease_s=0.2)
    ticks = iter(range(10 ** 6))
    with q.keep_lease(job, "A", progress=lambda: next(ticks), lease_s=0.2, every_s=0.05) as lease:
        time.sleep(0.6)                                        # three lease lengths, progressing all the time
        other = q.claim("B")
        assert other is None or other.job_id != job.job_id
    assert lease["renewals"] >= 5 and not lease["lost"] and lease["stalls"] == 0
    assert q.complete(job, {"ok": 1}, worker_id="A")


def test_a_stuck_job_is_not_renewed_and_another_worker_takes_it(tmp_path):
    import time
    storage, q = setup(tmp_path)
    job = q.claim("A", lease_s=0.2)
    with q.keep_lease(job, "A", progress=lambda: 7, lease_s=0.2, every_s=0.05) as lease:
        time.sleep(0.35)                                       # no progress: the lease runs out
        assert q.claim("B").job_id == job.job_id
    assert lease["renewals"] == 0 and lease["stalls"] >= 4
    from vlar_ops.queue import LeaseLost
    with pytest.raises(LeaseLost):
        q.complete(job, {"ok": 1}, worker_id="A")


def test_progress_resumes_renewal_and_counts_uploads(tmp_path):
    import time

    from vlar_ops.worker import Progress, upload
    storage, q = setup(tmp_path)
    job = q.claim("A", lease_s=0.3)
    state = {"steps": 0}
    prog = Progress(lambda: state["steps"])
    with q.keep_lease(job, "A", progress=prog.token, lease_s=0.3, every_s=0.05) as lease:
        time.sleep(0.12)
        stalled = lease["stalls"]
        state["steps"] = 1                                     # the episode advanced
        time.sleep(0.12)
        assert lease["renewals"] >= 1 and stalled >= 1
        before = lease["renewals"]
        runner, _ = files_runner(tmp_path)
        arts, _ = runner(job)
        upload(storage, arts, on_put=prog.bump)                # uploads count as progress too
        assert prog.uploads == len(arts)
        time.sleep(0.12)
        assert lease["renewals"] > before
    assert q.complete(job, {"ok": 1}, worker_id="A")


def test_tracked_episodes_token_follows_the_running_episode():
    from vlar_ops.vm_worker import TrackedEpisodes

    class Ep:
        def __init__(self):
            self.events, self.trajectory = [], []
    made = []
    t = TrackedEpisodes(lambda task: made.append(Ep()) or made[-1])
    assert t.token() is None
    ep = t(3)
    assert ep is made[0]                                       # the episode itself, unchanged
    first = t.token()
    ep.trajectory.append({})
    assert t.token() != first
    second = t.token()
    ep.events.append({})
    assert t.token() != second
    t(3)
    assert t.token()[0] == id(made[1])


def test_worker_exit_status_tells_recycling_from_done():
    from vlar_ops.cli import worker_exit
    zero = {"done": 0, "done_local": 0, "retry": 0, "failed": 0, "refused": 0, "lost": 0}
    assert worker_exit({**zero, "done": 5}, 5) == 4            # took its --max-jobs jobs
    assert worker_exit({**zero, "done": 3, "failed": 1, "lost": 1}, 5) == 4
    assert worker_exit({**zero, "done": 2}, 5) == 0            # nothing left to claim (or the stop flag)
    assert worker_exit({**zero, "done": 2}, None) == 0
    assert worker_exit({**zero, "done": 5, "refused": 1}, 5) == 3


def test_renewal_notices_a_lost_lease_and_the_worker_counts_it(tmp_path, monkeypatch):
    import time

    from vlar_ops import queue as Q
    storage, q = setup(tmp_path)
    monkeypatch.setattr(Q, "RENEW_EVERY_S", 0.05)
    runner, _ = files_runner(tmp_path)
    taken = {}

    def slow(job):
        if not taken:                                          # the first job loses its lease to B mid-run
            q._db.execute("UPDATE jobs SET lease_owner='B' WHERE job_id=?", (job.job_id,))
            taken["id"] = job.job_id
            time.sleep(0.2)
        return runner(job)
    stats = run_worker(q, storage, slow, "A")
    assert stats["lost"] == 1 and stats["done"] == 1
    assert q._db.execute("SELECT status, lease_owner FROM jobs WHERE job_id=?",
                         (taken["id"],)).fetchone() == ("leased", "B")
