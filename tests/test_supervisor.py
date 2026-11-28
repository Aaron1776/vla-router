"""vlar_ops.supervisor (PROPOSALS-2026-11-24 §0.6): memory sizing, recycling after --max-jobs, restarts of dead and
frozen workers with their jobs counted as failed attempts, the restart cap, refusals and the stop flag. The units are
fakes that run scripted workers against a real JobQueue."""

import json

import pytest

from vlar_ops import supervisor as S
from vlar_ops.plan import episode_plan
from vlar_ops.queue import JobQueue
from vlar_ops.storage import LocalStorage

GiB = S.GiB


def make_queue(tmp_path, n=2):
    q = JobQueue(tmp_path / "q.db", LocalStorage(tmp_path / "bucket"), tmp_path / "local")
    q.add_plan(episode_plan("base", tasks=list(range(n)), inits=[0], namespace=10, mode="routed", profile="none",
                            record="full", order_seed=1))
    return q


class FakeUnits:
    """Worker i runs its script's next action each time it is started:
    ("die", status): claims a job and dies holding it; ("work", n): claims and completes up to n jobs, then exits 4
    if it took n (--max-jobs) else 0; ("freeze",): claims a job whose lease is already expired and stays active;
    ("refuse",): exits 3."""

    def __init__(self, q, scripts):
        self.q, self.scripts = q, {i: list(s) for i, s in scripts.items()}
        self.state = {i: (False, None) for i in scripts}
        self.starts, self.stops = [], []

    def active(self, i):
        return self.state[i][0]

    def exit_status(self, i):
        return self.state[i][1]

    def start(self, i):
        self.starts.append(i)
        act = self.scripts[i].pop(0)
        wid = f"w{i}"
        if act[0] == "die":
            self.q.claim(wid, phases=["base"])
            self.state[i] = (False, act[1])
        elif act[0] == "work":
            took = 0
            while took < act[1]:
                job = self.q.claim(wid, phases=["base"])
                if job is None:
                    break
                took += 1
                self.q.complete(job, {"ok": True}, worker_id=wid)
            self.state[i] = (False, 4 if took >= act[1] else 0)
        elif act[0] == "freeze":
            self.q.claim(wid, lease_s=-1.0, phases=["base"])
            self.state[i] = (True, None)
        elif act[0] == "stall":                      # alive, lease still valid at the first pass
            self.q.claim(wid, lease_s=1000.0, phases=["base"])
            self.state[i] = (True, None)
            self.steal_after_first_poll = True
        elif act[0] == "refuse":
            self.state[i] = (False, 3)

    def stop(self, i):
        self.stops.append(i)
        self.state[i] = (False, 137)

    def journal_tail(self, i):
        return f"journal of w{i}\nOut of memory: Killed process"

    def memory(self):                                  # called once at the end of every pass
        if getattr(self, "steal_after_first_poll", False):  # the stalled lease expires and w9 claims it
            self.steal_after_first_poll = False
            self.q._db.execute("UPDATE jobs SET lease_owner='w9' WHERE status='leased' AND lease_owner='w1'")
        return {f"vlar-worker-{i}": (1 + len(self.starts)) * GiB for i in self.state} | {"vlar-xvla": 3 * GiB}


def supervisor(q, units, tmp_path, n, **kw):
    lines = []
    sup = S.Supervisor(q, units, phase="base", workers=n, stop_flag=tmp_path / "STOP",
                       history=tmp_path / "history.jsonl", sleep=lambda s: None, out=lines.append, **kw)
    return sup, lines


def attempts(q):
    return dict(q._db.execute("SELECT job_id, attempts FROM jobs").fetchall())


def test_sizing_fits_workers_and_servers_in_the_vm():
    s = S.size_workers(64 * GiB, 12 * GiB, 8)
    assert s.limit % (256 * S.MiB) == 0
    assert 8 * s.limit + 12 * GiB * 1.25 + 4 * GiB <= 64 * GiB * 0.95
    assert 8 * (s.limit + 256 * S.MiB) + 12 * GiB * 1.25 + 4 * GiB > 64 * GiB * 0.95     # the largest such step
    assert s.max_jobs == int((s.limit - 2 * GiB) // (768 * S.MiB)) >= 1
    assert s.docker_args() == f"--memory={s.limit // S.MiB}m --memory-swap={s.limit // S.MiB}m"
    assert S.size_workers(64 * GiB, 12 * GiB, 8, max_jobs=40).max_jobs == 40
    with pytest.raises(S.SizingError):
        S.size_workers(64 * GiB, 12 * GiB, 16)


def test_docker_sizes_parse():
    assert S.parse_bytes("3.2GiB") == int(3.2 * GiB)
    assert S.parse_bytes("512MiB") == 512 * S.MiB
    assert S.parse_bytes("1.5GB") == 1_500_000_000
    assert S.parse_bytes("0B") == 0
    with pytest.raises(ValueError):
        S.parse_bytes("--")


def test_worker_env_carries_max_jobs_and_the_memory_limit(tmp_path):
    s = S.size_workers(64 * GiB, 12 * GiB, 8)
    S.write_worker_env(tmp_path / "worker.env", "--phase branch", s)
    text = (tmp_path / "worker.env").read_text()
    assert f"VLAR_WORKER_ARGS=--phase branch --max-jobs {s.max_jobs}\n" in text
    assert f"VLAR_WORKER_DOCKER_ARGS=--memory={s.limit // S.MiB}m --memory-swap={s.limit // S.MiB}m\n" in text


def test_dead_worker_is_reaped_and_restarted_recycling_is_not_a_restart(tmp_path):
    q = make_queue(tmp_path, n=3)
    units = FakeUnits(q, {1: [("die", 137), ("work", 1), ("work", 5)], 2: [("refuse",)]})
    sup, lines = supervisor(q, units, tmp_path, 2)
    out = sup.run()
    assert q.counts("base")["done"] == 3
    assert out["restarts"] == {1: 1} and out["recycles"] == {1: 1} and out["refused"] == [2]
    assert units.starts.count(2) == 1                       # a refusal is never restarted
    assert sorted(attempts(q).values()) == [0, 0, 1]        # the job w1 died holding counted one attempt
    hist = [json.loads(x) for x in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert [h["outcome"] for h in hist] == ["worker-reaped", "worker-restart", "worker-refused"]
    assert hist[1]["exit"] == 137 and "Out of memory" in hist[1]["detail"]
    assert out["peak_memory_bytes"]["vlar-worker-1"] > 0


def test_frozen_worker_is_stopped_reaped_and_restarted(tmp_path):
    q = make_queue(tmp_path, n=2)
    units = FakeUnits(q, {1: [("freeze",), ("work", 5)]})
    sup, _ = supervisor(q, units, tmp_path, 1)
    out = sup.run()
    assert units.stops == [1] and out["restarts"] == {1: 1}
    assert q.counts("base")["done"] == 2 and sorted(attempts(q).values()) == [0, 1]


def test_restarts_are_capped_per_worker(tmp_path):
    q = make_queue(tmp_path, n=2)
    units = FakeUnits(q, {1: [("die", 1)] * 5})
    sup, _ = supervisor(q, units, tmp_path, 1, max_restarts=3)
    out = sup.run()
    assert out["restarts"] == {1: 3} and out["gave_up"] == [1] and units.starts.count(1) == 4
    hist = [json.loads(x)["outcome"] for x in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert hist[-1] == "worker-not-restarted"


def test_nothing_is_started_under_the_stop_flag(tmp_path):
    q = make_queue(tmp_path, n=2)
    units = FakeUnits(q, {1: [("die", 137), ("work", 5)]})
    sup, _ = supervisor(q, units, tmp_path, 1)
    (tmp_path / "STOP").write_text("hard limit")
    out = sup.run()
    assert units.starts == [1] and out["restarts"] == {}
    assert q.counts("base")["pending"] == 2                  # the dead worker's job was given back, counted


def test_supervise_refuses_without_both_servers_and_writes_worker_env(tmp_path, monkeypatch):
    q = make_queue(tmp_path, n=1)
    units = FakeUnits(q, {1: [("work", 5)]})
    cfg = {"local_root": str(tmp_path), "storage_url": f"file://{tmp_path / 'bucket'}"}
    with pytest.raises(S.SizingError, match="vlar-pi05"):
        S.supervise(cfg, phase="base", workers=1, worker_args="--phase base", worker_env=tmp_path / "w.env",
                    history=None, step="t", units=units, total=64 * GiB)
    units.memory = lambda: {"vlar-xvla": 3 * GiB, "vlar-pi05": 7 * GiB}
    monkeypatch.setattr("vlar_ops.cli._queue", lambda cfg: q)
    out = S.supervise(cfg, phase="base", workers=1, worker_args="--phase base", worker_env=tmp_path / "w.env",
                      history=None, step="t", units=units, total=64 * GiB, sleep=lambda s: None)
    assert out["sizing"]["servers"] == 10 * GiB and "--max-jobs" in (tmp_path / "w.env").read_text()


def test_a_stalled_worker_whose_job_was_claimed_by_another_is_stopped(tmp_path):
    q = make_queue(tmp_path, n=2)
    units = FakeUnits(q, {1: [("stall",), ("work", 5)]})
    sup, _ = supervisor(q, units, tmp_path, 1)
    out = sup.run()
    assert units.stops == [1] and out["restarts"] == {1: 1}
    assert q.owner(next(iter(attempts(q))))[0] in ("done", "leased")
    hist = [json.loads(x) for x in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert hist[0]["outcome"] == "worker-stopped-frozen" and "lost [" in hist[0]["detail"]


def test_fit_sizing_takes_as_many_workers_as_get_the_minimum_limit():
    """The pilot's resume (the user, 2026-11-24): the first boot's 8 workers were killed at 4 GiB (peak sampled
    4.07 GB); at least 5 GiB each, as many as fit beside the servers measured at that boot's start."""
    mem, servers = 67_416_563_712, 19_052_474_925
    old = S.size_workers(mem, servers, 8, max_jobs=2)
    assert (old.workers, old.limit) == (8, 4 * GiB)                     # what the first boot ran
    s = S.size_workers(mem, servers, 8, min_limit=5 * GiB, max_jobs=2, fit=True)
    assert (s.workers, s.limit, s.max_jobs) == (6, 22 * 256 * S.MiB, 2)  # 6 workers at 5.5 GiB
    assert s.limit >= 1.25 * 4.07e9 and s.limit >= 1.25 * 4 * GiB
    assert s.workers * s.limit + servers * 1.25 + 4 * GiB <= mem * 0.95
    assert S.size_workers(mem, servers, 4, min_limit=5 * GiB, fit=True).workers == 4    # never more than asked
    with pytest.raises(S.SizingError, match="not one worker"):
        S.size_workers(mem, servers, 8, min_limit=64 * GiB, fit=True)


def test_supervise_with_a_minimum_limit_starts_only_the_workers_that_fit(tmp_path, monkeypatch):
    q = make_queue(tmp_path, n=2)
    units = FakeUnits(q, {1: [("work", 5)], 2: [("work", 5)], 3: [("work", 5)]})
    units.memory = lambda: {"vlar-xvla": 5 * GiB, "vlar-pi05": 14 * GiB}
    monkeypatch.setattr("vlar_ops.cli._queue", lambda cfg: q)
    cfg = {"local_root": str(tmp_path), "storage_url": f"file://{tmp_path / 'bucket'}"}
    out = S.supervise(cfg, phase="base", workers=3, worker_args="--phase base", worker_env=tmp_path / "w.env",
                      history=None, step="t", units=units, total=36 * GiB, min_limit=5 * GiB, max_jobs=2,
                      sleep=lambda s: None)
    # 0.95 x 36 - 1.25 x 19 - 4 = 6.45 GiB: one worker of at least 5 GiB
    assert out["sizing"]["workers"] == 1 and units.starts and set(units.starts) == {1}
    assert "--memory=6400m" in (tmp_path / "w.env").read_text()


def test_the_ram_rule_runs_7_workers_at_the_pilots_numbers_and_never_plans_past_ram():
    """SPEC §17 amendment 64 (amendment 62 (4)): N x limit + the servers' peak must fit inside MemTotal; at boot the
    peak is max(1.25 x the servers now, the last measured peak). The pilot's resumed boot: 7 at 5.5 GiB, not 8."""
    mem, start, peak = 67_416_563_712, 18_955_838_160, 20_454_781_747
    s = S.size_workers_ram(mem, start, 8, servers_peak=peak, max_jobs=2)
    assert (s.workers, s.limit, s.max_jobs, s.rule) == (7, S.DEFAULT_WORKER_LIMIT, 2, "ram")
    assert s.servers_peak == int(start * 1.25) > peak                       # the boot estimate is the larger
    assert 7 * s.limit + s.servers_peak <= mem < 8 * s.limit + s.servers_peak
    assert 7 * s.limit + peak <= mem < 8 * s.limit + peak                   # also at the measured peak: 8 do not fit
    assert S.size_workers_ram(mem, start, 6, servers_peak=peak).workers == 6      # never more than asked
    assert S.size_workers_ram(mem, start, 8, servers_peak=30 * 10 ** 9).workers == 6   # a larger measured peak
    assert S.size_workers_ram(mem, start, 8, limit=5 * GiB).workers == 8
    with pytest.raises(S.SizingError, match="not one worker"):
        S.size_workers_ram(mem, start, 8, servers_peak=mem)
    with pytest.raises(S.SizingError, match="no workers"):
        S.size_workers_ram(mem, start, 0)


def test_supervise_sizes_by_the_ram_rule_unless_told_the_pilots_rule_and_reports_the_measured_peak(tmp_path,
                                                                                                 monkeypatch):
    q = make_queue(tmp_path, n=2)
    units = FakeUnits(q, {1: [("work", 5)], 2: [("work", 5)], 3: [("work", 5)]})
    units.memory = lambda: {"vlar-xvla": 5 * GiB, "vlar-pi05": 14 * GiB}
    monkeypatch.setattr("vlar_ops.cli._queue", lambda cfg: q)
    cfg = {"local_root": str(tmp_path), "storage_url": f"file://{tmp_path / 'bucket'}"}
    out = S.supervise(cfg, phase="base", workers=3, worker_args="--phase base", worker_env=tmp_path / "w.env",
                      history=None, step="t", units=units, total=36 * GiB, max_jobs=2, sleep=lambda s: None)
    # 36 GiB - 1.25 x 19 GiB = 12.25 GiB: two workers at 5.5 GiB
    assert out["sizing"]["rule"] == "ram" and out["sizing"]["workers"] == 2 and set(units.starts) == {1, 2}
    assert "--memory=5632m" in (tmp_path / "w.env").read_text()
    rr = out["ram_rule"]
    assert rr["workers"] == 2 and rr["servers_peak_planned"] == int(19 * GiB * 1.25)
    assert rr["fits_at_measured_peak"] == (2 * 5.5 * GiB + rr["servers_peak_measured"] <= 36 * GiB)
    with pytest.raises(S.SizingError, match="pilot's rule"):
        S.supervise(cfg, phase="base", workers=3, worker_args="--phase base", worker_env=tmp_path / "w.env",
                    history=None, step="t", units=units, total=36 * GiB, min_limit=5 * GiB, limit=6 * GiB)
