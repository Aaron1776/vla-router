import json
import zipfile

import pytest

from vlar_ops import packaging, spend
from vlar_ops.plan import episode_plan
from vlar_ops.queue import JobQueue, result_key, run_worker
from vlar_ops.storage import LocalStorage


# ---------------------------------------------------------------- queue

def small_plan(phase="explore", n_inits=3):
    return episode_plan(phase, tasks=range(2), inits=range(n_inits), namespace=2_000_000,
                        mode="routed", profile="none", record="full", order_seed=1)


@pytest.fixture
def q(tmp_path):
    bucket = LocalStorage(tmp_path / "bucket")
    return JobQueue(tmp_path / "vm" / "queue.sqlite", bucket, tmp_path / "vm" / "local")


def test_jobs_run_in_plan_order_and_land_in_the_bucket(q):
    plan = small_plan()
    assert q.add_plan(plan) == 6
    assert q.add_plan(plan) == 0                       # idempotent
    seen = []
    n = run_worker(q, lambda job: seen.append(job.job_id) or {"success": True}, should_stop=lambda: False)
    assert n == 6 and seen == [j.job_id for j in plan.jobs]
    assert q.counts()["done"] == 6
    rec = json.loads(q.storage.get_bytes(result_key("explore", seen[0])))
    assert rec["result"] == {"success": True} and rec["job"]["job_id"] == seen[0]


def test_resume_after_preemption(tmp_path, q):
    plan = small_plan()
    q.add_plan(plan)
    first = q.claim("w1")
    q.complete(first, {"ok": 1}, worker_id="w1")
    q.claim("w1")                                       # in flight when the VM dies
    q.close()
    # Fresh VM: new local disk, same bucket, plan re-added from the bucket's copy.
    q2 = JobQueue(tmp_path / "vm2" / "queue.sqlite", q.storage, tmp_path / "vm2" / "local")
    q2.add_plan(plan)
    stats = q2.resume()
    assert stats["marked_done_from_bucket"] == 1
    assert q2.counts()["done"] == 1 and q2.counts()["pending"] == 5


def test_results_reach_the_bucket_after_an_upload_failure(tmp_path, q, monkeypatch):
    q.add_plan(small_plan())
    job = q.claim("w1")
    real_put = q.storage.put_file
    monkeypatch.setattr(q.storage, "put_file", lambda *a: (_ for _ in ()).throw(OSError("network")))
    assert q.complete(job, {"ok": 1}, worker_id="w1") is False
    assert q.counts()["done_local"] == 1
    monkeypatch.setattr(q.storage, "put_file", real_put)
    assert q.sync() == 1 and q.counts()["done"] == 1


def test_stop_flag_halts_new_claims_and_failures_retry(tmp_path, q):
    q.add_plan(small_plan())
    calls = {"n": 0}

    def flaky(job):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {}

    stop = spend.StopFlag(tmp_path / "STOP")
    done = run_worker(q, flaky, should_stop=lambda: stop.is_set() or calls["n"] >= 3)
    assert done == 2 and q.counts()["pending"] == 4      # the failed job went back to pending
    stop.set("test")
    assert run_worker(q, flaky, should_stop=stop.is_set) == 0


def test_priority_runs_earlier_plans_first(q):
    a, b = small_plan("explore"), small_plan("dagger")
    q.add_plan(b, priority=1)
    q.add_plan(a, priority=0)
    assert q.claim("w").phase == "explore"


# ---------------------------------------------------------------- spend guard

class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class Collect:
    def __init__(self):
        self.events = []

    def notify(self, event, payload):
        self.events.append(event)


NO_BOOT = lambda: (None, None)            # noqa: E731  (no /proc boot information: sessions start at the guard)
CFG = spend.SpendConfig(soft_limit_usd=40, hard_limit_usd=45, vm_hourly_usd=0.5, boot_allowance_s=0,
                        storage_usd_per_gb_month=0.02, tick_s=60, storage_scan_every_s=1e12, drain_grace_s=600)


def test_ledger_accrues_and_survives_restarts(tmp_path):
    st, clock = LocalStorage(tmp_path), Clock()
    g = spend.SpendGuard(st, CFG, Collect(), now=clock, boot=NO_BOOT)
    g.start()
    clock.t += 10 * 3600
    assert g.tick() == "ok"
    assert spend.estimate(g.load(), clock.t)["vm_usd"] == pytest.approx(5.0)
    clock.t += 5 * 24 * 3600                             # stopped VM: no VM time accrues
    g2 = spend.SpendGuard(st, CFG, Collect(), now=clock, boot=NO_BOOT)
    g2.start()
    clock.t += 3600
    g2.tick()
    assert spend.estimate(g2.load(), clock.t)["vm_usd"] == pytest.approx(5.5)


def test_disk_bills_while_stopped(tmp_path):
    st, clock = LocalStorage(tmp_path), Clock()
    spend.record_disk(st, 200, 0.10, clock.t, "boot")
    clock.t += spend.HOURS_PER_MONTH * 3600
    assert spend.estimate(spend.load_ledger(st), clock.t)["disk_usd"] == pytest.approx(20.0)
    spend.close_disk(st, "boot", clock.t)
    clock.t += 1e7
    assert spend.estimate(spend.load_ledger(st), clock.t)["disk_usd"] == pytest.approx(20.0)


def test_soft_notifies_once_then_hard_drains_and_shuts_down(tmp_path):
    st, clock, notes = LocalStorage(tmp_path), Clock(), Collect()
    spend.reconcile(st, 39.9, clock.t, "billing report")
    g = spend.SpendGuard(st, CFG, notes, now=clock, boot=NO_BOOT)
    stop = spend.StopFlag(tmp_path / "STOP")
    in_flight = {"n": 2}
    log = []

    def sleep(s):
        clock.t += 3600                                  # $0.50 per tick
        in_flight["n"] = max(0, in_flight["n"] - 1)

    state = spend.run_daemon(g, stop, in_flight=lambda: in_flight["n"], release_in_flight=lambda: log.append("release"),
                             final_sync=lambda: log.append("sync"), shutdown=lambda: log.append("shutdown"),
                             sleep=sleep)
    assert state == "hard" and stop.is_set()
    assert notes.events == ["spend_soft_limit", "spend_hard_limit"]
    assert log == ["release", "sync", "shutdown"]
    # Restarting over the limit shuts straight down again; it never clears the stop flag.
    log.clear()
    g3 = spend.SpendGuard(st, CFG, notes, now=clock, boot=NO_BOOT)
    assert spend.run_daemon(g3, stop, in_flight=lambda: 0, release_in_flight=lambda: 0,
                            final_sync=lambda: None, shutdown=lambda: log.append("shutdown"), sleep=sleep) == "hard"
    assert log == ["shutdown"] and stop.is_set()
    assert notes.events.count("spend_soft_limit") == 1


def test_reconcile_sets_total_to_actual(tmp_path):
    st, clock = LocalStorage(tmp_path), Clock()
    g = spend.SpendGuard(st, CFG, Collect(), now=clock, boot=NO_BOOT)
    g.start()
    clock.t += 4 * 3600
    g.tick()
    spend.reconcile(st, 3.10, clock.t)
    assert spend.estimate(g.load(), clock.t)["total_usd"] == pytest.approx(3.10)


# ---------------------------------------------------------------- packaging

def make_tree(root):
    (root / "episodes").mkdir(parents=True)
    for i in range(7):
        (root / "episodes" / f"e{i}.parquet").write_bytes(bytes([i]) * (3000 + 100 * i))
    (root / "manifest.json").write_text('{"schema": 1}')


def test_split_store_mode_and_verify(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    make_tree(src)
    s = packaging.build(src, out, "dataset-v1", part_bytes=8000)
    assert len(s.parts) > 1
    for p in s.parts:
        with zipfile.ZipFile(p) as zf:
            assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())
            assert zf.testzip() is None
    assert packaging.verify(out) == []


def test_packaging_is_reproducible(tmp_path):
    src = tmp_path / "src"
    make_tree(src)
    a = packaging.build(src, tmp_path / "a", "d", part_bytes=8000)
    b = packaging.build(src, tmp_path / "b", "d", part_bytes=8000)
    assert a.sums.read_text() == b.sums.read_text()


def test_verify_detects_damage(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    make_tree(src)
    s = packaging.build(src, out, "d", part_bytes=8000)
    data = bytearray(s.parts[0].read_bytes())
    data[100] ^= 0xFF
    s.parts[0].write_bytes(bytes(data))
    assert any("checksum mismatch" in p for p in packaging.verify(out))
    s.parts[1].unlink()
    assert any("missing" in p for p in packaging.verify(out))
