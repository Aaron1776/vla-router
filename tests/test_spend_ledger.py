"""The spend ledger with several writers (amendment 39): conditional writes with retry, concurrent writers in threads
and in processes, a writer slipping in between another's read and write, a session that went missing; sessions from
boot time and failed boots (amendment 42); alerts per limit (amendment 43); the Cloud Storage generation check."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from vlar_ops import spend
from vlar_ops.storage import ABSENT, GCSStorage, LocalStorage

REPO = Path(__file__).resolve().parents[1]
NO_BOOT = lambda: (None, None)            # noqa: E731
CFG = spend.SpendConfig(soft_limit_usd=40, hard_limit_usd=45, vm_hourly_usd=0.5, boot_allowance_s=0,
                        storage_usd_per_gb_month=0.02, tick_s=60, storage_scan_every_s=1e12, drain_grace_s=600)


class Clock:
    def __init__(self, t=1_700_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class Notes:
    def __init__(self):
        self.events = []

    def notify(self, event, payload):
        self.events.append((event, payload["limit_usd"]))


def test_local_conditional_write(tmp_path):
    st = LocalStorage(tmp_path)
    data, v0 = st.get_versioned("ops/x.json")
    assert data is None and v0 == ABSENT
    assert st.put_if("ops/x.json", b"one", v0)
    assert not st.put_if("ops/x.json", b"two", v0)            # someone wrote since v0
    data, v1 = st.get_versioned("ops/x.json")
    assert data == b"one" and st.put_if("ops/x.json", b"two", v1) and st.get_bytes("ops/x.json") == b"two"
    assert list(st.list("")) == ["ops/x.json"]                # lock files are not storage keys


class Interleaving(LocalStorage):
    """Another writer's update lands between this writer's read and its write, once."""

    def __init__(self, root, other):
        super().__init__(root)
        self.other, self.done = other, False

    def get_versioned(self, key):
        out = super().get_versioned(key)
        if not self.done:
            self.done = True
            self.other()
        return out


def test_a_write_in_between_is_kept_and_the_edit_retried(tmp_path):
    st = Interleaving(tmp_path, lambda: spend.record_disk(LocalStorage(tmp_path), 100, 0.04, 1.0, "boot"))
    calls = []

    def edit(ledger):
        calls.append(1)
        ledger["adjustments"].append({"at": 2.0, "delta_usd": 1.5, "actual_usd": 1.5, "estimate_before_usd": 0.0})
    spend.update_ledger(st, edit, sleep=lambda s: None)
    led = spend.load_ledger(st)
    assert len(calls) == 2                                     # the first write was refused, the edit re-applied
    assert [d["name"] for d in led["disks"]] == ["boot"] and len(led["adjustments"]) == 1


def test_concurrent_writers_in_threads_lose_nothing(tmp_path):
    st, clock = LocalStorage(tmp_path), Clock()
    g = spend.SpendGuard(st, CFG, Notes(), now=clock, boot=NO_BOOT)
    g.start()
    n = 25

    def disks(i):
        for j in range(n):
            spend.record_disk(LocalStorage(tmp_path), 1, 0.04, 1.0, f"d{i}-{j}")

    def adjust():
        for j in range(n):
            spend.reconcile(LocalStorage(tmp_path), 0.0, clock.t, f"r{j}")

    def ticks():
        for _ in range(n):
            clock.t += 60
            g.tick()
    threads = [threading.Thread(target=disks, args=(i,)) for i in range(3)] + [threading.Thread(target=adjust),
                                                                               threading.Thread(target=ticks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    led = spend.load_ledger(st)
    assert len(led["disks"]) == 3 * n and len({d["name"] for d in led["disks"]}) == 3 * n
    assert len(led["adjustments"]) == n
    assert len(led["sessions"]) == 1 and led["sessions"][0]["vm_seconds"] == pytest.approx(n * 60)


WRITER = """
import sys
from vlar_ops import spend
from vlar_ops.storage import LocalStorage
root, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
for j in range(n):
    spend.record_disk(LocalStorage(root), 1, 0.04, 1.0, f"{tag}-{j}")
"""


def test_concurrent_writer_processes_lose_nothing(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, str(tmp_path), f"p{i}", "20"], env=env)
             for i in range(4)]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    names = {d["name"] for d in spend.load_ledger(LocalStorage(tmp_path))["disks"]}
    assert names == {f"p{i}-{j}" for i in range(4) for j in range(20)}


def test_a_missing_session_is_restored_not_a_crash(tmp_path):
    st, clock = LocalStorage(tmp_path), Clock()
    g = spend.SpendGuard(st, CFG, Notes(), now=clock, boot=NO_BOOT)
    g.start()
    clock.t += 3600
    g.tick()
    spend.update_ledger(st, lambda led: led["sessions"].clear())        # e.g. an old copy written back by hand
    clock.t += 3600
    assert g.tick() == "ok"
    (sess,) = spend.load_ledger(st)["sessions"]
    assert sess["id"] == g.session_id and sess["vm_seconds"] == pytest.approx(2 * 3600)


def test_sessions_count_from_boot_and_a_restarted_daemon_continues_its_boot(tmp_path):
    st, clock = LocalStorage(tmp_path), Clock()
    boot = lambda: ("boot-1", clock.t - 900)                   # noqa: E731  (15 minutes of startup install)
    g = spend.SpendGuard(st, CFG, Notes(), now=clock, boot=boot)
    g.start()
    assert spend.load_ledger(st)["sessions"][0]["vm_seconds"] == pytest.approx(900)
    clock.t += 3600
    g.tick()
    g2 = spend.SpendGuard(st, CFG, Notes(), now=clock, boot=lambda: ("boot-1", clock.t - 4500))  # same boot
    clock.t += 600
    g2.start()
    (sess,) = spend.load_ledger(st)["sessions"]
    assert sess["id"] == "boot-1" and sess["vm_seconds"] == pytest.approx(900 + 3600 + 600)


def test_failed_boots_are_counted(tmp_path):
    st, clock = LocalStorage(tmp_path), Clock()
    st.put_bytes("ops/failed-boots/b0.json", json.dumps({"boot_id": "b0", "boot_at": clock.t - 7200,
                                                        "failed_at": clock.t - 3600, "reason": "pip"}).encode())
    led = spend.load_ledger(st)
    pending = spend.pending_failed_boots(st, led, CFG)
    assert [p["vm_seconds"] for p in pending] == [3600]
    g = spend.SpendGuard(st, CFG, Notes(), now=clock, boot=lambda: ("b1", clock.t))
    g.start()
    g.start()                                                  # folded once
    sessions = spend.load_ledger(st)["sessions"]
    assert sorted(s["id"] for s in sessions) == ["b0", "b1"]
    assert spend.estimate(spend.load_ledger(st), clock.t)["vm_usd"] == pytest.approx(0.5)
    assert spend.pending_failed_boots(st, spend.load_ledger(st), CFG) == []


def test_alerts_record_their_limit_and_fire_again_above_a_raised_limit(tmp_path):
    st, clock, notes = LocalStorage(tmp_path), Clock(), Notes()
    spend.reconcile(st, 41.0, clock.t)
    g = spend.SpendGuard(st, CFG, notes, now=clock, boot=NO_BOOT)
    assert g.start() == "soft" and g.tick() == "soft"
    assert notes.events == [("spend_soft_limit", 40)]
    raised = spend.SpendConfig(**{**CFG.__dict__, "soft_limit_usd": 48, "hard_limit_usd": 50})
    g2 = spend.SpendGuard(st, raised, notes, now=clock, boot=NO_BOOT)
    assert g2.start() == "ok"                                  # 41 < 48: nothing new
    spend.reconcile(st, 48.5, clock.t)
    assert g2.tick() == "soft" and g2.tick() == "soft"
    spend.reconcile(st, 50.0, clock.t)
    assert g2.tick() == "hard"
    assert notes.events == [("spend_soft_limit", 40), ("spend_soft_limit", 48), ("spend_hard_limit", 50)]
    alerts = spend.load_ledger(st)["alerts"]
    assert [a["limit_usd"] for a in alerts["soft"]] == [40, 48] and [a["limit_usd"] for a in alerts["hard"]] == [50]


def test_old_alert_records_are_migrated(tmp_path):
    st = LocalStorage(tmp_path)
    old = spend.new_ledger()
    old["alerts"] = {"soft_at": 5.0}
    st.put_bytes(spend.LEDGER_KEY, json.dumps(old).encode())
    led = spend.load_ledger(st)
    assert led["alerts"] == {"soft": [{"at": 5.0, "limit_usd": None}], "hard": []}


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    @property
    def generation(self):
        return self.bucket.objects[self.name][1]

    def download_as_bytes(self, if_generation_match=None):
        data, gen = self.bucket.objects[self.name]
        if if_generation_match is not None and gen != if_generation_match:
            raise type("PreconditionFailed", (Exception,), {})()
        return data

    def upload_from_string(self, data, if_generation_match=None):
        cur = self.bucket.objects.get(self.name, (None, 0))[1]
        if if_generation_match is not None and cur != if_generation_match:
            raise type("PreconditionFailed", (Exception,), {})()
        self.bucket.objects[self.name] = (bytes(data), cur + 1)


class FakeBucket:
    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return FakeBlob(self, name)

    def get_blob(self, name):
        return FakeBlob(self, name) if name in self.objects else None


def test_cloud_storage_uses_generation_preconditions():
    st = GCSStorage.__new__(GCSStorage)
    st._bucket, st.prefix = FakeBucket(), "vla-router"
    data, v = st.get_versioned("ops/ledger.json")
    assert data is None and v == ABSENT
    assert st.put_if("ops/ledger.json", b"a", v)
    assert not st.put_if("ops/ledger.json", b"b", v)          # generation 0 means "must not exist"
    data, v = st.get_versioned("ops/ledger.json")
    assert data == b"a" and v == "1"
    st._bucket.blob("vla-router/ops/ledger.json").upload_from_string(b"other")
    assert not st.put_if("ops/ledger.json", b"mine", v)
    assert st.get_versioned("ops/ledger.json") == (b"other", "2")
