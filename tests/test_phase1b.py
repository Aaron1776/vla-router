"""Tranche B (amendments 56, 57, 62 (2), 65): `vlar_ops/setupcount.py` and `vlar_ops/phase1b.py`."""

from __future__ import annotations

import math
import io
import json
from collections import Counter

import pytest

from vlar_ops import phase1b as B
from vlar_ops import setupcount as S
from vlar_ops.branchplan import DecisionPointRef
from vlar_ops.plan import Job, job_id, rollout_seed


# ---------------------------------------------------------------- setup count

class Sim:
    built = 0

    def __init__(self, task):
        Sim.built += 1
        self.task, self.closed, self.setups = task, False, []

    def setup_episode(self, init_idx, env_seed):
        self.setups.append(init_idx)

    def close(self):
        self.closed = True


class Episodes:
    def __init__(self):
        self.sims = {}
        self._make_sim = Sim

    def sim(self, task):
        if task not in self.sims:
            self.sims[task] = self._make_sim(task)
        return self.sims[task]


def warm(sim, loads):
    for _ in range(loads - 2):
        sim.setup_episode(0, 0)


def ep_job(subject, task=3, setup_count=None, kind="episode"):
    spec = {"task_id": task, "init_idx": 1}
    if setup_count is not None:
        spec["setup_count"] = setup_count
    return Job(job_id(subject, kind, 1), "explore", kind, subject, 1, 0, 0, spec)


def run(eps, counts, job, setups=1):
    counts.prepare(job)
    for _ in range(setups):
        eps.sim(job.spec["task_id"]).setup_episode(job.spec["init_idx"], 0)
    counts.check(job)
    return counts.job_meta(job)


def test_markers_record_the_count_of_the_episodes_setup_on_a_cached_simulator():
    eps = Episodes()
    c = S.SetupCounts(eps, warm_up=warm)
    assert [run(eps, c, ep_job(f"e{n}"))["setup_count"] for n in range(3)] == [2, 3, 4]
    br = ep_job("snap", kind="b1_continue")
    c.prepare(br)
    assert c.job_meta(br) == {"setup_count": None, "setups_in_job": 0, "setup_count_spec": None}


def test_a_spec_count_of_two_gets_a_fresh_simulator_each_time_and_closes_the_old_one():
    eps = Episodes()
    c = S.SetupCounts(eps, warm_up=warm)
    run(eps, c, ep_job("a"))
    old = eps.sims[3]
    m = run(eps, c, ep_job("b", setup_count=2))
    assert old.closed and eps.sims[3] is not old and m["setup_count"] == 2 and m["setup_count_spec"] == 2
    again = eps.sims[3]
    run(eps, c, ep_job("c", setup_count=2))
    assert again.closed and eps.sims[3].setups == [1]


def test_a_replayed_count_warms_a_fresh_simulator_up_to_it_and_keeps_one_already_there():
    eps = Episodes()
    c = S.SetupCounts(eps, warm_up=warm)
    assert run(eps, c, ep_job("a", setup_count=5))["setup_count"] == 5
    sim = eps.sims[3]
    assert run(eps, c, ep_job("b", setup_count=6))["setup_count"] == 6 and eps.sims[3] is sim   # next setup is 6


def test_a_job_that_sets_up_twice_or_at_another_count_fails():
    eps = Episodes()
    c = S.SetupCounts(eps, warm_up=warm)
    with pytest.raises(S.SetupCountError):
        run(eps, c, ep_job("a", setup_count=2), setups=2)
    with pytest.raises(S.SetupCountError):
        c.prepare(ep_job("b", setup_count=1))


def test_the_counting_runner_and_per_job_meta_reach_the_marker(tmp_path):
    from vlar_ops.queue import JobQueue
    from vlar_ops.storage import LocalStorage
    from vlar_ops.worker import run_worker
    eps = Episodes()
    c = S.SetupCounts(eps, warm_up=warm)
    jobs = [ep_job("x", setup_count=2), ep_job("y")]

    class Runner:
        def __call__(self, job):
            eps.sim(job.spec["task_id"]).setup_episode(1, 0)
            return [], {"run_id": job.subject}
    storage = LocalStorage(tmp_path / "bucket")
    q = JobQueue(tmp_path / "q.sqlite", storage, tmp_path / "local")
    from vlar_ops.plan import Plan
    q.add_plan(Plan("explore", {}, jobs))
    stats = run_worker(q, storage, S.CountingRunner(Runner(), c), "w1", job_meta=S.meta_with_counts({"w": 1}, c))
    assert stats["done"] == 2
    metas = {j.subject: json.loads(storage.get_bytes(f"results/explore/{j.job_id}.json"))["meta"] for j in jobs}
    assert metas["x"]["setup_count"] == 2 and metas["x"]["w"] == 1
    assert metas["y"]["setup_count"] == 3 and metas["y"]["setup_count_spec"] is None


# ---------------------------------------------------------------- screen

def test_the_screen_covers_every_state_then_draws_the_remainder_with_distinct_seeds():
    plan = B.screen_plan(115, seed=7, namespace=2_000_000, p_escalate=0.1)
    assert len(plan.jobs) == 1150 and plan.phase == "explore"
    per = Counter((j.spec["task_id"], j.spec["init_idx"]) for j in plan.jobs)
    assert set(per.values()) == {2, 3} and sum(v == 3 for v in per.values()) == 150
    assert all(sum(v == 3 for (t, _), v in per.items() if t == task) == 15 for task in range(10))
    seeds = [j.spec["rollout_seed"] for j in plan.jobs]
    assert len(set(seeds)) == 1150 and max(seeds) < 2_100_000
    j = next(j for j in plan.jobs if j.spec["screen_repeat"] == 1)
    assert j.spec["rollout_seed"] == rollout_seed(2_001_000, j.spec["task_id"], j.spec["init_idx"])
    assert all(j.spec["record"] == "off" and j.spec["setup_count"] == 2 and j.spec["p_escalate"] == 0.1
               and j.spec["profile"] == "none" for j in plan.jobs)
    assert len({j.job_id for j in plan.jobs}) == 1150
    assert B.screen_plan(115, seed=7, namespace=2_000_000, p_escalate=0.1).plan_id == plan.plan_id
    assert B.screen_plan(115, seed=8, namespace=2_000_000, p_escalate=0.1).plan_id != plan.plan_id
    with pytest.raises(B.TrancheBError):
        B.screen_plan(40, seed=7, namespace=2_000_000, p_escalate=0.1)


# ---------------------------------------------------------------- selection

def rows_for(plan, fail_rate_by_task):
    out = []
    for j in plan.jobs:
        t = j.spec["task_id"]
        fails = (j.spec["init_idx"] + j.spec["screen_repeat"]) % 100 < fail_rate_by_task[t] * 100
        out.append({"episode_id": j.subject, "task_id": t, "success": not fails, "steps": 300, "setup_count": 2,
                    "job": j})
    return out


def test_selection_takes_two_thirds_failed_episodes_per_task_and_makes_up_a_shortfall():
    plan = B.screen_plan(115, seed=7, namespace=2_000_000, p_escalate=0.1)
    rates = {t: 0.3 for t in range(10)}
    rates[4] = 0.05
    rows = rows_for(plan, rates)
    sel = B.select_reruns(rows, per_task=64, k=2, seed=3)
    by = Counter((s["stratum"]["task_id"], s["stratum"]["parent_outcome"]) for s in sel.values())
    assert all(by[(t, "failure")] + by[(t, "success")] == 32 for t in range(10))
    assert by[(0, "failure")] == 21                                      # round(2/3 x 32)
    n_fail_4 = sum(1 for r in rows if r["task_id"] == 4 and not r["success"])
    assert by[(4, "failure")] == n_fail_4 < 21 and by[(4, "success")] == 32 - n_fail_4
    e = next(e for e, s in sel.items() if s["stratum"] == {"source": "explore", "task_id": 0, "parent_outcome": "failure"})
    assert sel[e]["p_episode"] == pytest.approx(21 / sel[e]["stratum_size"])
    assert B.select_reruns(list(reversed(rows)), per_task=64, k=2, seed=3) == sel   # order-free
    assert B.select_reruns(rows, per_task=64, k=2, seed=4) != sel


def test_screened_refuses_a_missing_marker_or_a_wrong_setup_count():
    plan = B.screen_plan(50, seed=1, namespace=2_000_000, p_escalate=0.1, tasks=(0,))
    good = lambda jid: {"meta": {"setup_count": 2}, "result": {"summary": {"success": True, "steps": 9}}}  # noqa: E731
    assert len(B.screened(plan, good)) == 50
    with pytest.raises(B.TrancheBError, match="no result marker"):
        B.screened(plan, lambda jid: None)
    with pytest.raises(B.TrancheBError, match="setup_count"):
        B.screened(plan, lambda jid: {"meta": {"setup_count": 3}, "result": {"summary": {"success": True, "steps": 9}}})


def test_reruns_repeat_the_screened_spec_with_record_full_and_a_new_job_id():
    plan = B.screen_plan(50, seed=1, namespace=2_000_000, p_escalate=0.1, tasks=(0,))
    rows = rows_for(plan, {0: 0.5})
    sel = B.select_reruns(rows, per_task=10, k=2, seed=1)
    rp = B.rerun_plan(rows, sel)
    assert len(rp.jobs) == 5 and {j.subject for j in rp.jobs} == set(sel)
    screen = {j.subject: j for j in plan.jobs}
    for j in rp.jobs:
        s = screen[j.subject]
        assert j.job_id != s.job_id and j.spec["screen_job_id"] == s.job_id and j.phase == "explore"
        assert {k: v for k, v in j.spec.items() if k not in ("record", "screen_job_id")} == \
               {k: v for k, v in s.spec.items() if k != "record"}
        assert j.spec["record"] == "full" and j.spec["setup_count"] == 2


# ---------------------------------------------------------------- comparison

def parquet(rows):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf)
    return buf.getvalue()


def bucket_for(screen, rerun, *, steps_b=None, events_b=None, success_b=True, count_b=2, calib_b="c" * 64):
    run = lambda rec, cal: [{"run_id": screen.subject, "success": True, "record": rec, "steps": 40,  # noqa: E731
                             "calibration_hash": cal}]
    steps = [{"run_id": screen.subject, "step": i, "a": float(i)} for i in range(3)]
    ev = lambda rec: [{"run_id": screen.subject, "seq": 0, "t": "start", "step": 0,  # noqa: E731
                       "event": json.dumps({"record": rec, "x": 1})}]
    files = {}
    for jid, rec, cal, st, evs in ((screen.job_id, "off", "c" * 64, steps, ev("off")),
                                   (rerun.job_id, "full", calib_b, steps_b or steps, events_b or ev("full"))):
        files[f"records/runs/explore/{jid}.parquet"] = parquet(run(rec, cal))
        files[f"records/steps/explore/{jid}.parquet"] = parquet(st)
        files[f"records/events/explore/{jid}.parquet"] = parquet(evs)
    markers = {screen.job_id: {"meta": {"setup_count": 2},
                               "result": {"summary": {"run_id": screen.subject, "success": True, "steps": 40}}},
               rerun.job_id: {"meta": {"setup_count": count_b},
                              "result": {"summary": {"run_id": screen.subject, "success": success_b, "steps": 40}}}}
    return files.get, markers.get


def a_pair():
    plan = B.screen_plan(50, seed=1, namespace=2_000_000, p_escalate=0.1, tasks=(0,))
    rows = rows_for(plan, {0: 0.5})
    rp = B.rerun_plan(rows, B.select_reruns(rows, per_task=2, k=2, seed=1))
    rerun = rp.jobs[0]
    return next(j for j in plan.jobs if j.job_id == rerun.spec["screen_job_id"]), rerun, rp


def test_an_identical_rerun_passes_with_only_the_record_mode_different():
    screen, rerun, _ = a_pair()
    read, marker = bucket_for(screen, rerun)
    assert B.compare_rerun(read, rerun, marker) == {"subject": rerun.subject, "passed": True, "problems": []}


@pytest.mark.parametrize("change", ["steps", "events", "outcome", "setup", "calibration"])
def test_any_difference_fails_the_comparison(change):
    screen, rerun, rp = a_pair()
    kw = {"steps": {"steps_b": [{"run_id": screen.subject, "step": i, "a": float(i) + (i == 2) * 1e-9}
                                for i in range(3)]},
          "events": {"events_b": [{"run_id": screen.subject, "seq": 0, "t": "start", "step": 0,
                                   "event": json.dumps({"record": "full", "x": 2})}]},
          "outcome": {"success_b": False}, "setup": {"count_b": 3},
          "calibration": {"calib_b": "d" * 64}}[change]
    read, marker = bucket_for(screen, rerun, **kw)
    r = B.compare_rerun(read, rerun, marker)
    assert r["passed"] is False and r["problems"]
    rep = B.compare_all(rp, read, marker)
    assert rep["passed"] is False and rep["failed"][0]["subject"] == rerun.subject


def test_a_rerun_without_a_marker_is_pending():
    screen, rerun, rp = a_pair()
    read, _ = bucket_for(screen, rerun)
    rep = B.compare_all(rp, read, lambda jid: None)
    assert rep["passed"] and not rep["all_done"] and rep["pending"] == 1


# ---------------------------------------------------------------- points and branches

def test_points_k_per_episode_with_the_product_inclusion_probability():
    chosen = {"explore/t0/i1/r0": {"stratum": {"source": "explore", "task_id": 0, "parent_outcome": "failure"},
                                   "stratum_size": 30, "stratum_drawn": 21, "p_episode": 21 / 30},
              "explore/t0/i2/r0": {"stratum": {"source": "explore", "task_id": 0, "parent_outcome": "success"},
                                   "stratum_size": 80, "stratum_drawn": 11, "p_episode": 11 / 80}}
    pts = [DecisionPointRef(f"explore/t0/i1/r0/dp{i:02d}", "explore/t0/i1/r0", "explore", 0, 1, i, 260)
           for i in range(20)]
    pts += [DecisionPointRef("explore/t0/i2/r0/dp00", "explore/t0/i2/r0", "explore", 0, 2, 0, 260),
            DecisionPointRef("explore/t0/i2/r0/dp51", "explore/t0/i2/r0", "explore", 0, 2, 52, 260)]  # not eligible
    sel = B.select_points(pts, chosen, k=2, seed=5, step_limit=520, interval=10)
    a = [s for sid, s in sel.items() if sid.startswith("explore/t0/i1/")]
    b = [s for sid, s in sel.items() if sid.startswith("explore/t0/i2/")]
    assert len(a) == 2 and a[0]["inclusion_prob"] == pytest.approx(21 / 30 * 2 / 20)
    assert len(b) == 1 and b[0]["inclusion_prob"] == pytest.approx(11 / 80) and b[0]["by"] == "episode"
    plan, labels = B.branch_plan_b(pts, sel, per_task=64, seed=5, step_limit=520, interval=10,
                                   branches={"b1_continue": {"calls": 0}, "b2_one_call": {"calls": 1},
                                             "b3_three_calls": {"calls": 3}, "b4_max": {"calls": "max"}})
    assert plan.phase == "phase1b" and set(labels) == set(sel)
    assert {j.kind for j in plan.jobs} == {"b1_continue", "b2_one_call"} and len(plan.jobs) == 3 * 2 * 3
    assert all(lab["parent_seeds"] == {} and lab["selection"]["by"] == "episode" for lab in labels.values())


def test_rerun_points_keep_only_listed_snapshots():
    pytest.importorskip("numpy")
    _, rerun, _ = a_pair()
    ev = [{"run_id": rerun.subject, "t": "decision_point", "event": json.dumps({"index": i, "budget_remaining": 260})}
          for i in range(3)] + [{"run_id": rerun.subject, "t": "step", "event": "{}"}]
    listed = {f"{rerun.subject}/dp00", f"{rerun.subject}/dp02"}
    pts = B.rerun_points(rerun, ev, listed)
    assert [p.index for p in pts] == [0, 2] and all(p.source == "explore" for p in pts)


def test_a_difference_writes_the_mismatch_marker_and_the_stop_flag(tmp_path):
    screen, rerun, rp = a_pair()
    read, marker = bucket_for(screen, rerun, success_b=False)
    rep = B.compare_all(rp, read, marker)
    B.record_mismatch(rep, mismatch=str(tmp_path / "MISMATCH.json"), stop_flag=str(tmp_path / "STOP"))
    m = json.loads((tmp_path / "MISMATCH.json").read_text())
    assert m["n_failed"] == 1 and m["failed"][0]["subject"] == rerun.subject and m["failed"][0]["problems"]
    assert rerun.subject in (tmp_path / "STOP").read_text()


# ---------------------------------------------------------------- 740: a difference ends the step with 76

FAKE_LIB = r"""
reason() { echo "$*" > "${VLAR_REASON_FILE:-/dev/null}"; echo "$*"; }
fail() { reason "$*"; exit 1; }
need_input() { reason "$*"; exit 75; }
gate_fail() { reason "$*"; exit 76; }
need_images() { :; }; write_batching_env() { :; }; calibration_frozen() { :; }; wait_servers() { :; }
golden_check() { :; }
run_phase() {   # as lib.sh's: a raised stop flag with work left ends it through need_input
  echo "run_phase $*" >> "$FAKE/log"
  [[ -n "${FAKE_DIFFERS:-}" ]] && { echo '{"failed": [{"subject": "explore/t0/i1/r0"}]}' > "$FAKE_MISMATCH"; echo differs > "$VLAR_STOP_FLAG"; }
  [[ -n "${FAKE_BUDGET:-}" ]] && echo "budget" > "$VLAR_STOP_FLAG"
  [[ -e "$VLAR_STOP_FLAG" ]] && need_input "$1 stopped by the spend guard with work left"
  return 0
}
"""


def run_740(tmp_path, *, differs=False, budget=False, earlier=False):
    import os
    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    fake, bin_ = tmp_path / "fake", tmp_path / "bin"
    fake.mkdir(); bin_.mkdir()
    for name, body in {"gcloud": "exit 0", "docker": 'echo "docker $*" >> "$FAKE/log"; exit 0',
                       "vlar-ops": 'echo "vlar-ops $*" >> "$FAKE/log"; exit 0', "systemctl": "exit 0",
                       "sha256sum": "exit 0", "sleep": "exit 1"}.items():
        (bin_ / name).write_text("#!/bin/bash\n" + body + "\n"); (bin_ / name).chmod(0o755)
    (fake / "lib.sh").write_text(FAKE_LIB)
    rd = tmp_path / "work" / "phase1b" / "rerun"
    rd.mkdir(parents=True)
    (rd / "plan.json").write_text("{}")
    if earlier:
        (rd / "MISMATCH.json").write_text('{"failed": [{"subject": "explore/t0/i1/r0"}]}')
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "FAKE": str(fake), "VLAR_LIB": str(fake / "lib.sh"),
           "VLAR_WORK_ROOT": str(tmp_path / "work"), "VLAR_STORAGE_URL": "gs://b/v", "VLAR_STOP_FLAG": str(tmp_path / "STOP"),
           "VLAR_REASON_FILE": str(tmp_path / "reason"), "FAKE_MISMATCH": str(rd / "MISMATCH.json"),
           "FAKE_DIFFERS": "1" if differs else "", "FAKE_BUDGET": "1" if budget else ""}
    out = subprocess.run(["bash", str(repo / "ops" / "vm" / "postfreeze" / "740-phase1b.sh")], env=env,
                         capture_output=True, text=True, timeout=60)
    log = (fake / "log").read_text() if (fake / "log").exists() else ""
    return out.returncode, (tmp_path / "reason").read_text() if (tmp_path / "reason").exists() else "", log


def test_740_ends_a_rerun_difference_with_76_not_as_a_budget_stop(tmp_path):
    rc, why, log = run_740(tmp_path, differs=True)
    assert rc == 76 and "differs from its screened run" in why and "explore/t0/i1/r0" in why
    assert "run_phase explore" in log and "plan-b" not in log and "run_phase phase1b" not in log


def test_740_keeps_a_budget_stop_during_the_reruns_a_budget_stop(tmp_path):
    rc, why, log = run_740(tmp_path, budget=True)
    assert rc == 75 and "spend guard" in why and "plan-b" not in log


def test_740_refuses_to_rerun_after_a_difference_on_an_earlier_boot(tmp_path):
    rc, why, log = run_740(tmp_path, earlier=True)
    assert rc == 76 and "differs" in why and "queue add" not in log and "run_phase" not in log


# ---------------------------------------------------------------- the screen's yield (amendment 66 (4))

def test_the_yield_counts_failed_episodes_per_task_and_proposes_a_top_up_where_short():
    plan = B.screen_plan(115, seed=1, namespace=2_000_000, p_escalate=0.1, tasks=(0, 1, 2))
    rows = rows_for(plan, {0: 0.4, 1: 0.05, 2: 0.0})
    rep = B.screen_yield(rows, per_task=64, k=2)
    t0, t1, t2 = (rep["tasks"][t] for t in (0, 1, 2))
    assert rep["needed_per_task"] == 21 and rep["short_tasks"] == [1, 2]
    assert t0["screened"] == 115 and t0["failed"] >= 21 and t0["short"] == 0 and "top_up" not in t0
    f1 = t1["failed"]
    assert 0 < f1 < 21 and t1["short"] == 21 - f1
    assert t1["top_up"]["episodes"] == math.ceil(1.1 * (21 - f1) / (f1 / 115)) and t1["top_up"]["rate_basis"] == "observed"
    assert t2["failed"] == 0 and t2["top_up"]["episodes"] == math.ceil(1.1 * 21 / (3 / 115))
    assert "lower bound" in t2["top_up"]["rate_basis"]
    assert rep["top_up_episodes"] == t1["top_up"]["episodes"] + t2["top_up"]["episodes"]


@pytest.mark.parametrize("more", [10, 35, 60, 200])
def test_a_larger_screen_extends_the_same_sequence(more):
    """A top-up screens screen_episodes(n + more) less what was screened: the episodes already screened keep their
    (task, init, repeat), so their ids and seeds."""
    a, b = set(B.screen_episodes(115, seed=7)), set(B.screen_episodes(115 + more, seed=7))
    assert a <= b and len(b - a) == 10 * more


# ---------------------------------------------------------------- front-loaded screen sizes (amendment 67)

def test_front_loaded_sizes_from_the_base_runs():
    from pathlib import Path
    rates = B.base_failure_rates(Path(__file__).resolve().parents[1] / "References/l4-phase0/runs/base-runs.csv")
    assert {t: round(p * 50) for t, p in rates.items()} == {0: 20, 1: 16, 2: 5, 3: 4, 4: 15, 5: 6, 6: 16, 7: 7, 8: 11,
                                                            9: 2}
    sizes = B.front_loaded_sizes(rates)
    assert {t: n - 115 for t, n in sizes.items() if n > 115} == {2: 105, 3: 163, 5: 66, 7: 39, 9: 451}
    assert all(sizes[t] == 115 for t in (0, 1, 4, 6, 8)) and sum(sizes.values()) == 1974


def test_per_task_sizes_extend_the_uniform_screen_and_keep_its_ids_and_seeds():
    sizes = {0: 115, 1: 278, 2: 566}
    small = B.screen_plan(115, seed=3, namespace=2_000_000, p_escalate=0.1, tasks=(0, 1, 2))
    big = B.screen_plan(sizes, seed=3, namespace=2_000_000, p_escalate=0.1, tasks=(0, 1, 2))
    assert B.screen_episodes(115, 3, tasks=(0, 1, 2)) == B.screen_episodes({0: 115, 1: 115, 2: 115}, 3, tasks=(0, 1, 2))
    a = {j.job_id: j.spec["rollout_seed"] for j in small.jobs}
    b = {j.job_id: j.spec["rollout_seed"] for j in big.jobs}
    assert a.items() <= b.items() and len(b) == sum(sizes.values()) and big.params["per_task"] == {"0": 115, "1": 278, "2": 566}
    assert {t: sum(j.spec["task_id"] == t for j in big.jobs) for t in sizes} == sizes
