"""The pilot's selection, re-weighting, projections and stop rule (`vlar_ops.pilot`, `branchplan.stratified_selection`;
SPEC §17 amendments 57-58). The checks that need the simulator are in `tests/reference/test_pilot_checks.py`."""
import random

import pytest

from vlar_ops import pilot
from vlar_ops.branchplan import BranchPlanError, DecisionPointRef, branch_plan, eligible, stratified_selection

STEP_LIMIT, INTERVAL = 520, 10
SEL = {"scheme": "stratified", "seed": 5, "per_task": 15, "failed_share": 2 / 3, "min_calls": 3}
BRANCHES = {"b1_continue": {"calls": 0}, "b2_one_call": {"calls": 1}, "b3_three_calls": {"calls": 3},
            "b4_max": {"calls": "max"}}


def episode(task, i, *, budget=260, indices=range(52), source="base"):
    eid = f"{source}/t{task}/i{i}"
    return [DecisionPointRef(f"{eid}/dp{k:02d}", eid, source, task, i, k, budget) for k in indices]


def population(spec):
    """spec: {task: (failed episodes, successful episodes)} -> points and outcomes."""
    pts, success = [], {}
    for t, (nf, ns) in spec.items():
        for i in range(nf + ns):
            pts += episode(t, i)
            success[f"base/t{t}/i{i}"] = i >= nf
    return pts, success


def select(pts, success, **kw):
    return stratified_selection(pts, success, step_limit=STEP_LIMIT, interval=INTERVAL, dp_sampling={**SEL, **kw})


def test_ten_failed_and_five_successful_points_per_task_with_their_inclusion_probabilities():
    pts, success = population({0: (3, 7), 1: (1, 9)})
    sel = select(pts, success)
    for t, nf in ((0, 3), (1, 1)):
        mine = {k: v for k, v in sel.items() if v["stratum"]["task_id"] == t}
        fail = [v for v in mine.values() if v["stratum"]["parent_outcome"] == "failure"]
        succ = [v for v in mine.values() if v["stratum"]["parent_outcome"] == "success"]
        assert (len(fail), len(succ)) == (10, 5)
        # every episode has 50 points where three calls fit (index <= 49)
        assert all(v["stratum_size"] == 50 * nf and v["inclusion_prob"] == pytest.approx(10 / (50 * nf)) for v in fail)
        assert all(v["stratum_size"] == 50 * (10 - nf) and v["stratum_drawn"] == 5 for v in succ)
        assert all(v["by"] == "point" and v["stratum"]["source"] == "base" for v in mine.values())


def test_a_short_stratum_is_made_up_from_the_other():
    pts = episode(0, 0, indices=range(4)) + episode(0, 1) + episode(1, 0) + episode(2, 0) + episode(2, 1)
    success = {"base/t0/i0": False, "base/t0/i1": True, "base/t1/i0": True, "base/t2/i0": False, "base/t2/i1": False}
    sel = select(pts, success)
    count = lambda t, o: sum(1 for v in sel.values()                                    # noqa: E731
                             if v["stratum"]["task_id"] == t and v["stratum"]["parent_outcome"] == o)
    assert (count(0, "failure"), count(0, "success")) == (4, 11)       # only 4 failed points exist
    assert sel["base/t0/i0/dp00"]["inclusion_prob"] == 1.0
    assert (count(1, "failure"), count(1, "success")) == (0, 15)       # no failed episode at all
    assert (count(2, "failure"), count(2, "success")) == (15, 0)       # no successful one


def test_the_draw_does_not_depend_on_input_order_but_on_the_seed():
    pts, success = population({0: (2, 8), 1: (5, 5)})
    a = select(pts, success)
    shuffled = list(pts)
    random.Random(1).shuffle(shuffled)
    assert select(shuffled, success) == a
    assert set(select(pts, success, seed=6)) != set(a)


def test_only_points_where_the_calls_fit_are_eligible():
    p = lambda k, b=260: DecisionPointRef("s", "e", "base", 0, 0, k, b)                # noqa: E731
    assert eligible(p(49), step_limit=STEP_LIMIT, interval=INTERVAL, min_calls=3)
    assert not eligible(p(50), step_limit=STEP_LIMIT, interval=INTERVAL, min_calls=3)  # 2 decision points left
    assert eligible(p(51), step_limit=STEP_LIMIT, interval=INTERVAL, min_calls=1)
    assert not eligible(p(0, 29), step_limit=STEP_LIMIT, interval=INTERVAL, min_calls=3)  # budget for 2 calls
    pts, success = population({0: (1, 1)})
    pts += episode(0, 2, budget=20)                                    # a failed parent with budget for 2 calls
    success["base/t0/i2"] = False
    sel = select(pts, success)
    assert all(int(s[-2:]) <= 49 and not s.startswith("base/t0/i2/") for s in sel)


def test_only_base_parents_are_drawn_by_point_and_every_parent_needs_an_outcome():
    pts, success = population({0: (1, 1)})
    with pytest.raises(BranchPlanError, match="by episode"):
        select(pts + episode(0, 0, source="explore"), success)
    with pytest.raises(BranchPlanError, match="no recorded outcome"):
        select(pts + episode(0, 7), success)


def test_the_plan_branches_exactly_the_selected_points_and_records_their_selection():
    pts, success = population({0: (2, 3), 1: (1, 4)})
    sel = select(pts, success)
    plan, labels = branch_plan("pilot", pts, branches=BRANCHES, step_limit=STEP_LIMIT, interval=INTERVAL, seeds=3,
                               enabled=list(BRANCHES), tier_one_seeds=2, order_seed=7, dp_sampling=SEL, selection=sel)
    assert set(labels) == set(sel)
    assert all(labels[s]["selection"] == sel[s] for s in sel)
    assert {j.subject for j in plan.jobs} == set(sel) and all(j.spec["snapshot_id"] == j.subject for j in plan.jobs)
    # b1 needs seeds 2-3 (seed 1 is the parent), the others 3 each: 11 runs a point unless branches share jobs
    assert len(plan.jobs) == 11 * len(sel)
    with pytest.raises(BranchPlanError):
        branch_plan("pilot", pts, branches=BRANCHES, step_limit=STEP_LIMIT, interval=INTERVAL, seeds=3,
                    enabled=list(BRANCHES), tier_one_seeds=2, order_seed=7, dp_sampling=SEL)


# ---------------------------------------------------------------- re-weighting (amendment 57 (6))

def test_weighted_rates_recover_the_population_rate_from_a_stratified_sample():
    pts, success = population({0: (2, 8)})
    sel = select(pts, success)
    # positive exactly in failed parents: the population rate is the failed share of eligible points, 100/500
    rate = pilot.weighted_rate([v["stratum"]["parent_outcome"] == "failure" for v in sel.values()],
                               [v["inclusion_prob"] for v in sel.values()])
    assert rate["rate"] == pytest.approx(100 / 500) and rate["n"] == 15
    unweighted = sum(v["stratum"]["parent_outcome"] == "failure" for v in sel.values()) / 15
    assert unweighted == pytest.approx(10 / 15)                         # what unweighted counting would claim
    assert 1 < rate["ess"] < 15


def test_weighted_rates_are_unbiased_over_draws_with_labels_varying_inside_strata():
    pts, success = population({0: (3, 7)})
    truth = {p.snapshot_id: (p.index % 3 == 0) if not success[p.episode_id] else (p.index % 10 == 0) for p in pts}
    elig = [p.snapshot_id for p in pts if eligible(p, step_limit=STEP_LIMIT, interval=INTERVAL, min_calls=3)]
    population_rate = sum(truth[s] for s in elig) / len(elig)
    estimates = []
    for seed in range(300):
        sel = select(pts, success, seed=seed)
        estimates.append(pilot.weighted_rate([truth[s] for s in sel], [v["inclusion_prob"] for v in sel.values()])
                         ["rate"])
    assert sum(estimates) / len(estimates) == pytest.approx(population_rate, abs=0.01)


def test_weighted_rate_refuses_bad_input():
    with pytest.raises(pilot.PilotError):
        pilot.weighted_rate([1, 0], [0.5])
    with pytest.raises(pilot.PilotError):
        pilot.weighted_rate([1], [0.0])
    assert pilot.weighted_rate([1, 0], [1, 1]) == {"rate": 0.5, "n": 2, "ess": 2.0}


# ---------------------------------------------------------------- folds, the stop rule and the decision (amendment 58)

def test_each_fold_trains_on_eight_tasks_without_its_test_and_validation_tasks():
    f = pilot.folds()
    assert sorted(f) == list(range(10)) and all(len(v) == 8 for v in f.values())
    assert 0 not in f[0] and 1 not in f[0] and 9 not in f[9] and 0 not in f[9]
    assert f[3] == [0, 1, 2, 5, 6, 7, 8, 9]


def test_the_stop_rule_flags_tasks_and_short_folds():
    ok = {t: 20.0 for t in range(10)}
    assert pilot.stop_rule(ok)["passes"] and pilot.stop_rule(ok)["fold_expected"][0] == 160
    three = {**ok, 0: 19.9, 1: 0.0, 2: 5.0}
    r = pilot.stop_rule(three)
    # 3 flagged pass; with at most 3 flagged, every fold has 5 tasks at >= 20, so the fold test cannot bind alone
    assert r["flagged_tasks"] == [0, 1, 2] and r["passes"] and r["fold_expected"][3] == pytest.approx(124.9)
    three_spread = {**{t: 40.0 for t in range(10)}, 0: 0.0, 4: 0.0, 7: 0.0}
    assert pilot.stop_rule(three_spread)["passes"]                       # 3 flagged, every fold >= 200
    four = {**three_spread, 9: 0.0}
    assert not pilot.stop_rule(four)["passes"] and pilot.stop_rule(four)["flagged_tasks"] == [0, 4, 7, 9]
    low = {t: 12.4 for t in range(10)}                                  # 99.2 per fold, every task flagged
    assert pilot.stop_rule(low)["folds_short"] == list(range(10))
    edge = {t: 12.5 for t in range(10)}                                 # exactly 100 per fold passes the fold test
    assert pilot.stop_rule(edge, per_task=10)["passes"]


def test_the_decision_rule_set_before_the_pilot():
    assert pilot.decision(True, True) == "keep" and pilot.decision(True, False) == "keep"
    assert pilot.decision(False, True) == "drop" and pilot.decision(False, False) == "stop"
    assert "cascader_nospread" in pilot.DECISIONS["drop"] and "re-plan" in pilot.DECISIONS["stop"]


def test_expected_positives_under_both_schemes():
    assert pilot.expected_positives(47, 0.6, 0.1, scheme="uniform", phi=0.2) == pytest.approx(47 * (0.12 + 0.08))
    assert pilot.expected_positives(48, 0.6, 0.1, scheme="stratified", phi=0.2) == pytest.approx(32 * 0.6 + 16 * 0.1)
    capped = pilot.expected_positives(48, 0.6, 0.1, scheme="stratified", phi=0.2, failed_available=20)
    assert capped == pytest.approx(20 * 0.6 + 28 * 0.1)


def labelled(spec):
    """spec: {(task, outcome): [(episode, positive), ...]} -> pilot points."""
    return [{"task_id": t, "parent_outcome": o, "episode_id": e, "positive": x}
            for (t, o), rows in spec.items() for e, x in rows]


def test_projections_apply_the_rule_at_both_budgets():
    tasks = range(10)
    pts = labelled({(t, o): [(f"base/t{t}/i{i}", o == "failure") for i in range(3) for _ in range(3)]
                    for t in tasks for o in ("failure", "success")})
    strata = {t: {"phi": 0.2, "points_failed": 500} for t in tasks}
    out = pilot.project(pts, strata, {"current": 47, "moved": 107}, pools={"current": 3, "moved": 2}, n_boot=50)
    cur, moved = out["current/stratified"], out["moved/stratified"]
    assert cur["expected"][0] == pytest.approx(47 * 2 / 3) and cur["passes"]
    assert out["current/uniform"]["expected"][0] == pytest.approx(47 * 0.2) and not out["current/uniform"]["passes"]
    assert moved["expected"][0] == pytest.approx(107 * 2 / 3)
    assert cur["expected_ci95"][0] == (pytest.approx(47 * 2 / 3), pytest.approx(47 * 2 / 3))  # no spread to resample
    assert cur["bootstrap_pass_share"] == 1.0 and out["rates"]["t0/failure"] == 1.0
    # few failed points: the stratified draw is capped by what the pools offer
    tight = {t: {"phi": 0.2, "points_failed": 5} for t in tasks}
    out = pilot.project(pts, tight, {"current": 47}, pools={"current": 3}, n_boot=10)
    assert out["current/stratified"]["expected"][0] == pytest.approx(15) and not out["current/stratified"]["passes"]
    rule = (out["current/stratified"]["passes"], True)
    assert pilot.decision(*rule) == "drop"


def test_the_bootstrap_resamples_parent_episodes_not_points():
    pts = labelled({(0, "failure"): [("a", 1)] * 4 + [("b", 0)]})
    assert pilot.cluster_rates(pts)[(0, "failure")] == pytest.approx(0.8)
    rng = random.Random(0)
    seen = {round(pilot.cluster_rates(pts, rng)[(0, "failure")], 6) for _ in range(200)}
    assert seen == {0.0, 0.8, 1.0}             # {a, a}, {a, b}, {b, b}: never a point-level value such as 0.6


# ---------------------------------------------------------------- the report's pieces

def row(task, outcome, ep, gain_none, gain_full, pi=0.5, excluded=()):
    g = {"b2_one_call": {"none": gain_none, "cascader_full": gain_full}}
    return {"task_id": task, "parent_outcome": outcome, "episode_id": ep, "inclusion_prob": pi,
            "gain": g, "excluded": list(excluded), "positive": gain_full is not None and gain_full >= 1 / 3,
            "positive_scored": gain_full is not None}


def test_rate_table_re_weights_and_counts_the_rescues_the_cut_loses():
    rows = [row(4, "failure", "a", 2 / 3, 2 / 3, pi=0.5), row(4, "failure", "a", 1 / 3, None, pi=0.5,
                                                                excluded=("cascader_full",)),
            row(4, "success", "b", 0.0, 0.0, pi=0.1), row(4, "success", "c", -1 / 3, -1 / 3, pi=0.1)]
    t = pilot.rate_table(rows, ("none", "cascader_full"))
    none = t["t4"]["b2_one_call/none"]
    # weights 2, 2, 10, 10: rescue (2 + 2) / 24, harm 10 / 24
    assert none["rescue"] == pytest.approx(4 / 24) and none["harm"] == pytest.approx(10 / 24) and none["n"] == 4
    assert t["t4"]["b2_one_call/cascader_full"]["n"] == 3                      # the excluded point is not scored
    assert t["t4"]["positive_failure"] == {"n": 2, "positives": 1, "unscored": 1}
    assert t["t4"]["rescues_lost_to_cascader_full_cut"] == {"n": 1, "excluded_at_the_point": 1}
    assert t["all"]["n"] == 4 and t["all_but_4_8"]["n"] == 0


def marker(kind, worker, claimed, finished):
    return {"job": {"kind": kind}, "finished_at": finished, "meta": {"claimed_at": claimed, "worker_id": worker}}


def test_costs_apportion_each_branch_window_by_job_wall_time():
    ms = [marker("b1_continue", "w1", 100, 110), marker("b2_one_call", "w2", 100, 130),
          marker("b4_max", "w1", 112, 172), marker("b3_three_calls", "w2", 131, 171),
          marker("b1_continue", "w1", 500, 600)]                                # outside every window
    timing = [{"boot_id": "x", "phase_started": 100.0}, {"boot_id": "x", "phase_ended": 180.0}]
    c = pilot.costs(ms, timing, hourly_usd=0.472)
    # 80 window seconds shared by 10 + 30 + 60 + 40 = 140 wall seconds
    assert c["vm_s_per_run"]["b1_continue"] == pytest.approx(80 * 10 / 140)
    assert c["vm_s_per_run"]["b4_max"] == pytest.approx(80 * 60 / 140)
    assert c["jobs_timed"] == 4 and c["windows_s"] == [80.0]
    r = c["vm_s_per_run"]
    assert c["vm_s_per_point"]["base_without_b3"] == pytest.approx(2 * r["b1_continue"] + 3 * (r["b2_one_call"]
                                                                                                + r["b4_max"]))
    assert c["usd_per_point"]["explore_without_b3"] == pytest.approx(c["vm_s_per_point"]["explore_without_b3"]
                                                                     * 0.472 / 3600)


def test_measured_budgets_reproduce_the_plan_at_its_estimated_prices():
    est = {"base_without_b3": 8 * 7 * 0.472 / 3600, "explore_without_b3": 9 * 7 * 0.472 / 3600}
    b = pilot.measured_budgets(est, hourly_usd=0.472)
    assert b["current"] == pytest.approx(47, abs=0.5) and b["moved"] == pytest.approx(107, abs=0.5)
    dear = pilot.measured_budgets({k: 2 * v for k, v in est.items()}, hourly_usd=0.472)
    assert dear["current"] < 30 and dear["moved"] < 70


def test_restart_overhead_separates_recycles_from_claiming():
    ms = [marker("b1_continue", "w1", 0, 10), marker("b1_continue", "w1", 11, 20), marker("b1_continue", "w1", 80, 90),
          marker("b1_continue", "w2", 0, 5), marker("b1_continue", "w2", 7, 9)]
    o = pilot.restart_overhead(ms)
    assert o["gaps"] == 3 and o["large_gaps"] == 1 and o["large_gap_s_total"] == 60 and o["small_gap_s_median"] == 2


# ---------------------------------------------------------------- the resume (the user, 2026-11-24)

def _jobs_and_labels():
    from vlar_ops.plan import Job
    jobs, labels, k = [], {}, 0
    for task, outcome, point, done in [(0, "failure", "a", 3), (0, "failure", "b", 0), (0, "success", "c", 1),
                                       (1, "failure", "d", 4), (1, "failure", "e", 11), (2, "success", "f", 0)]:
        sid = f"base/t{task}/i{point}/dp01"
        labels[sid] = {"selection": {"stratum": {"task_id": task, "parent_outcome": outcome, "source": "base"}}}
        for i in range(11):
            jobs.append((f"{sid}#{i}", "done" if i < done else "pending",
                         Job(f"{sid}#{i}", "pilot", "b1_continue", sid, 1, 1 + i % 3, 100 - k, {})))
            k += 1
    return jobs, labels


def test_resume_order_goes_point_by_point_rotating_cells_most_finished_first():
    jobs, labels = _jobs_and_labels()
    points, order = pilot.resume_order(jobs, labels)
    # e is complete; round 0: each cell's best point (d 4, a 3, c 1, f 0), round 1: b
    assert points == ["base/t1/id/dp01", "base/t0/ia/dp01", "base/t0/ic/dp01", "base/t2/if/dp01", "base/t0/ib/dp01"]
    subj = {jid: jid.split("#")[0] for jid, _, _ in jobs}
    seen = [subj[j] for j in order]
    assert seen == sorted(seen, key=points.index)                       # each point's jobs together, points in order
    assert len(order) == 66 - 3 - 0 - 1 - 4 - 11 - 0 == len(set(order))  # every open job once, no done job
    by = {jid: job for jid, _, job in jobs}
    first = [j for j in order if subj[j] == points[0]]
    assert [by[j].order for j in first] == sorted(by[j].order for j in first)   # planned order within a point


def test_set_order_reorders_claims_and_keeps_the_job_bodies(tmp_path):
    from vlar_ops.plan import Plan
    from vlar_ops.queue import JobQueue
    from vlar_ops.storage import LocalStorage
    jobs, labels = _jobs_and_labels()
    q = JobQueue(tmp_path / "q.db", LocalStorage(tmp_path / "bucket"), tmp_path / "local")
    q.add_plan(Plan("pilot", {}, [j for _, _, j in jobs]))
    with q._tx():
        q._db.executemany("UPDATE jobs SET status='done' WHERE job_id=?", [(jid,) for jid, s, _ in jobs if s == "done"])
    bodies = dict(q._db.execute("SELECT job_id, body FROM jobs").fetchall())
    rep = pilot.reorder_queue(q, _labels_dir(tmp_path, labels))
    assert rep["points_complete"] == 1 and rep["points_open"] == 5 and rep["jobs_reordered"] == 47
    claimed = [q.claim("w1").subject for _ in range(7)]
    assert claimed == ["base/t1/id/dp01"] * 7                           # d's 7 open jobs first, whatever their tier
    assert q.claim("w1").subject == "base/t0/ia/dp01"
    assert dict(q._db.execute("SELECT job_id, body FROM jobs").fetchall()) == bodies
    with pytest.raises(ValueError, match="open jobs once"):
        q.set_order("pilot", [jobs[-1][0]])
    q.close()


def _labels_dir(tmp_path, labels):
    import json
    d = tmp_path / "plan"
    d.mkdir()
    (d / "labels.json").write_text(json.dumps(labels))
    return d


def test_costs_of_a_stopped_step_come_from_the_jobs_claim_and_finish_times():
    """No phase_ended (the step stopped inside run_phase): the window ends at that boot's last finish."""
    def m(kind, claimed, finished, boot):
        return {"job": {"kind": kind}, "finished_at": finished,
                "meta": {"claimed_at": claimed, "worker_id": "w1", "boot_id": boot}}
    ms = [m("b1_continue", 100, 110, "x"), m("b4_max", 100, 160, "x"), m("b2_one_call", 105, 135, "x"),
          m("b1_continue", 1000, 1020, "y")]
    c = pilot.costs(ms, [{"boot_id": "x", "phase_started": 90.0}], hourly_usd=0.472)
    # boot x: 90..160 = 70 s over 10 + 60 + 30 wall s; boot y (no timing): its own claim..finish, 20 s
    assert c["windows_s"] == [70.0, 20.0] and c["window_ends"] == ["last_job_finish", "last_job_finish"]
    assert c["vm_s_per_run"]["b4_max"] == pytest.approx(70 * 60 / 100)
    assert c["vm_s_per_run"]["b1_continue"] == pytest.approx((70 * 10 / 100 + 20) / 2)
    assert c["worker_s_per_run"]["b1_continue"] == pytest.approx(15) and c["jobs_timed"] == 4
    assert c["usd_per_run"]["b4_max"] == pytest.approx(42 * 0.472 / 3600)


def test_long_episode_subjects_take_one_long_episode_per_task():
    rows = [{"episode_id": "base/t3/i1", "task_id": 3, "end_reason": "step_limit", "time_frac": 0.99},
            {"episode_id": "base/t3/i2", "task_id": 3, "end_reason": "success", "time_frac": 0.50},
            {"episode_id": "base/t3/i3", "task_id": 3, "end_reason": "success", "time_frac": 0.90},
            {"episode_id": "base/t4/i1", "task_id": 4, "end_reason": "step_limit", "time_frac": 1.1},
            {"episode_id": "base/t4/i2", "task_id": 4, "end_reason": "success", "time_frac": 0.2}]
    got = pilot.long_episode_subjects(rows, tasks=(3, 4), seed="s")
    assert got[0] in ("base/t3/i1", "base/t3/i3") and got[1] == "base/t4/i1"
    assert pilot.long_episode_subjects(rows, tasks=(3,), seed="s", exclude=["base/t3/i1"]) == ["base/t3/i3"]
    with pytest.raises(pilot.PilotError, match="task 4"):
        pilot.long_episode_subjects(rows, tasks=(4,), seed="s", exclude=["base/t4/i1"])


def test_fit_table_answers_whether_7_or_8_workers_fit_at_the_limit():
    GiB = 1024 ** 3
    sup = {"sizing": {"mem_total": 67_416_563_712, "servers": 19_052_474_925, "limit": 22 * 256 * 1024 ** 2,
                      "workers": 6, "max_jobs": 2},
           "peak_memory_bytes": {"vlar-pi05": 14_645_838_479, "vlar-xvla": 5_553_392_713,
                                 "vlar-worker-1": 4_500_000_000, "vlar-worker-2": 4_100_000_000}}
    f = pilot.fit_table(sup)
    assert f["servers_peak"] == 14_645_838_479 + 5_553_392_713 and f["rule_max_workers"] == 6
    assert f["counts"][6]["rule_fits"] and not f["counts"][7]["rule_fits"] and not f["counts"][8]["rule_fits"]
    assert f["counts"][7]["limits_fit_ram"] and not f["counts"][8]["limits_fit_ram"]
    assert f["counts"][8]["worker_peaks_fit_ram"]            # 8 x 4.5 GB + 20.2 GB + 4 GiB < 67.4 GB
    assert f["counts"][8]["worker_peaks_bytes"] == 8 * 4_500_000_000 + f["servers_peak"] + 4 * GiB


def test_check_timing_per_step_and_warm_up():
    iii = {"record_off": [
        {"subject": "a", "loads": 4, "outcome": {"steps": 400}, "seconds": {"search": 3.0, "warm_up": 2.0, "episode": 40.0}},
        {"subject": "b", "loads": 2, "outcome": {"steps": 100}, "seconds": {"search": 1.0, "warm_up": 0.0, "episode": 10.0}},
        {"subject": "c", "loads": 2, "outcome": {"steps": 100}}]}                  # a bb7112f report: no seconds
    t = pilot.check_timing(iii)
    assert t["timed"] == 2 and t["episode_s_per_step"] == pytest.approx(0.1)
    assert t["warm_up_s_per_load"] == pytest.approx(1.0) and t["search_s_per_episode"] == pytest.approx(2.0)
