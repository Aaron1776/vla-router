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


# ---------------------------------------------------------------- disk records confirmed by reading back (amendment 48)

RACY_STORAGE = r'''
class LosesTheNextWrite(LocalStorage):
    """The pre-amendment-39 race (2026-11-23's lost disk record): right after a ledger write succeeds, a writer that
    read the ledger before it puts its stale copy back with a blind write."""
    armed = True

    def put_if(self, key, data, version):
        before = self.get_bytes(key)
        ok = super().put_if(key, data, version)
        if ok and self.armed and key == spend.LEDGER_KEY:
            type(self).armed = False
            self.put_bytes(key, before if before is not None else json.dumps(spend.new_ledger()).encode())
        return ok
'''
exec(RACY_STORAGE)


def ops_config(tmp_path, store):
    cfg = tmp_path / "ops.yaml"
    cfg.write_text(f"storage_url: {store}\nlocal_root: {tmp_path / 'local'}\n")
    return str(cfg)


def test_a_lost_disk_record_is_not_confirmed(tmp_path):
    LosesTheNextWrite.armed = True
    st = LosesTheNextWrite(tmp_path)
    entry = spend.record_disk(st, 100, 0.10, 1_795_405_905.9, "vm-boot")       # reported written...
    assert spend.load_ledger(st)["disks"] == []                                         # ...and then lost
    with pytest.raises(spend.DiskRecordNotConfirmed, match="not in the ledger read back"):
        spend.confirm_disk(st, entry)
    entry = spend.record_disk(st, 100, 0.10, 1_795_405_905.9, "vm-boot")       # the next write is kept
    spend.confirm_disk(st, entry)


def test_disk_created_exits_nonzero_unless_the_ledger_holds_the_record(tmp_path, monkeypatch, capsys):
    from vlar_ops import cli
    store = tmp_path / "store"
    LosesTheNextWrite.armed = True
    monkeypatch.setattr(cli, "open_storage", lambda url: LosesTheNextWrite(url))
    args = ["spend", "disk-created", "--name", "vm-boot", "--gb", "100", "--usd-per-gb-month", "0.1",
            "--ops-config", ops_config(tmp_path, store)]
    with pytest.raises(SystemExit, match="disk record NOT confirmed"):
        cli.main(args)
    cli.main(args)
    assert "(confirmed in the ledger)" in capsys.readouterr().out
    assert [d["name"] for d in spend.load_ledger(LocalStorage(store))["disks"]] == ["vm-boot"]


def test_a_disk_recorded_late_counts_from_its_creation_time(tmp_path, capsys):
    """`--since` with gcloud's creationTimestamp: the record counts from then; `--until` closes it at its deletion."""
    from vlar_ops import cli
    store = tmp_path / "store"
    cfg = ops_config(tmp_path, store)
    cli.main(["spend", "disk-created", "--name", "vm-boot", "--gb", "100", "--usd-per-gb-month", "0.1",
              "--since", "2026-11-22T19:51:45.921-08:00", "--ops-config", cfg])
    assert "since 2026-11-23T03:51:45Z (confirmed in the ledger)" in capsys.readouterr().out
    led = spend.load_ledger(LocalStorage(store))
    assert led["disks"][0]["since"] == pytest.approx(1_795_405_905.921)
    later = led["disks"][0]["since"] + 12 * 3600
    assert spend.estimate(led, later)["disk_usd"] == pytest.approx(100 * 0.1 * 12 / spend.HOURS_PER_MONTH)
    cli.main(["spend", "disk-deleted", "--name", "vm-boot", "--until", "2026-11-23T15:51:45.921Z",
              "--ops-config", cfg])
    led = spend.load_ledger(LocalStorage(store))
    assert led["disks"][0]["until"] == pytest.approx(later)
    assert spend.estimate(led, later + 1e6)["disk_usd"] == pytest.approx(100 * 0.1 * 12 / spend.HOURS_PER_MONTH)
    with pytest.raises(SystemExit, match="in the future"):
        cli.main(["spend", "disk-created", "--name", "x", "--gb", "1", "--usd-per-gb-month", "0.1",
                  "--since", "2099-01-01T00:00:00Z", "--ops-config", cfg])
    with pytest.raises(SystemExit, match="give the time zone"):
        cli.main(["spend", "disk-created", "--name", "x", "--gb", "1", "--usd-per-gb-month", "0.1",
                  "--since", "2026-11-22T19:51:45", "--ops-config", cfg])


def test_two_open_records_with_one_name_need_since_to_close_one(tmp_path):
    """A VM recreated under the same name before the old disk's record was closed: closing by name alone would close
    the live disk's record too."""
    st = LocalStorage(tmp_path)
    spend.record_disk(st, 100, 0.1, 1000.0, "vm-boot")
    spend.record_disk(st, 100, 0.1, 5000.0, "vm-boot")
    with pytest.raises(ValueError, match="2 open disks named 'vm-boot'.*pass --since"):
        spend.close_disk(st, "vm-boot", 6000.0)
    assert spend.close_disk(st, "vm-boot", 6000.0, since=1000.4)["since"] == 1000.0
    assert [d["until"] for d in spend.load_ledger(st)["disks"]] == [6000.0, None]
    spend.close_disk(st, "vm-boot", 7000.0)                       # now unambiguous
    with pytest.raises(ValueError, match="no open disk"):
        spend.close_disk(st, "vm-boot", 8000.0)
