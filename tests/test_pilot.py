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
    assert cur["bootstrap_pass_share"] == 1.0 and out["rates"]["t0/failure"]["rate"] == 1.0
    assert out["pooled_rates"]["failure"]["rate"] == 1.0 and out["pooled_rates"]["success"]["n"] == 90
    # few failed points: the stratified draw is capped by what the pools offer
    tight = {t: {"phi": 0.2, "points_failed": 5} for t in tasks}
    out = pilot.project(pts, tight, {"current": 47}, pools={"current": 3}, n_boot=10)
    assert out["current/stratified"]["expected"][0] == pytest.approx(15) and not out["current/stratified"]["passes"]
    rule = (out["current/stratified"]["passes"], True)
    assert pilot.decision(*rule) == "drop"


def test_projections_pool_stratum_rates_across_tasks_and_own_rates_flag_only_from_10_points():
    """Amendment 62 (1): the rule runs on rates pooled across tasks, applied to each task's phi and failed points;
    a task's own rate flags it only if it has at least 10 labelled points."""
    spec = {}
    for t in range(10):
        # task 0: 0 of 6 failed-parent points positive; the others 6 of 6 -> pooled r_f = 54/60 = 0.9
        spec[(t, "failure")] = [(f"base/t{t}/i{i}", t != 0) for i in range(3) for _ in range(2)]
        spec[(t, "success")] = [(f"base/t{t}/i{10 + i}", False) for i in range(1)]
    pts = labelled(spec)
    strata = {t: {"phi": 0.2, "points_failed": 500} for t in range(10)}
    strata[1] = {"phi": 0.1, "points_failed": 4}                 # few failed points: capped at 3 x 4 = 12
    out = pilot.project(pts, strata, {"current": 30}, pools={"current": 3}, n_boot=50, own_basis="all")
    cur = out["current/stratified"]
    assert out["pooled_rates"]["failure"] == {"rate": pytest.approx(0.9), "positives": 54, "n": 60,
                                              "ci95": out["pooled_rates"]["failure"]["ci95"]}
    assert cur["expected"][0] == pytest.approx(20 * 0.9)        # pooled, not task 0's own 0.0
    assert cur["expected"][1] == pytest.approx(12 * 0.9)        # its own failed-points cap
    assert cur["own_expected"][0] == 0.0 and cur["own_expected"][2] == pytest.approx(20)
    # 7 labelled points per task: task 0's own rate does not flag it; the pooled projections flag all ten (< 20)
    assert out["own_rate_judged_tasks"] == [] and cur["flagged_by_own_rate"] == []
    assert cur["flagged_tasks"] == list(range(10))
    lo, hi = out["rates"]["t0/failure"]["ci95"]
    assert lo == hi == 0.0 and out["rates"]["t2/failure"]["rate"] == 1.0
    # at 10 labelled points a task's own rate can flag it, even where the pooled projection passes
    spec[(0, "failure")] += [(f"base/t0/i{5 + i}", False) for i in range(3)]
    out = pilot.project(labelled(spec), strata, {"big": 90}, pools={"big": 3}, n_boot=20, own_basis="all")
    big = out["big/stratified"]
    assert out["labelled_per_task"][0] == 10 and out["own_rate_judged_tasks"] == [0]
    assert big["expected"][0] >= 20 and big["flagged_by_own_rate"] == [0] and 0 in big["flagged_tasks"]


def test_amendment_63_own_rates_flag_only_from_10_failed_parent_points_and_shrinkage_sits_beside():
    """Amendment 63 (5): a task's own rates flag it only from 10 labelled failed-parent points (9 failed + 1 successful
    is not enough); the pooled projection still decides; the shrinkage projection is reported beside it."""
    spec = {}
    for t in range(10):
        spec[(t, "failure")] = [(f"base/t{t}/i{i}", t != 0) for i in range(3) for _ in range(2)]
        spec[(t, "success")] = [(f"base/t{t}/i{10 + i}", False) for i in range(1)]
    spec[(0, "failure")] += [(f"base/t0/i{5 + i}", False) for i in range(3)]      # task 0: 9 failed, 1 successful
    strata = {t: {"phi": 0.2, "points_failed": 500} for t in range(10)}
    out = pilot.project(labelled(spec), strata, {"big": 90}, pools={"big": 3}, n_boot=20)
    big = out["big/stratified"]
    assert out["own_basis"] == "failure" and out["labelled_failed_per_task"][0] == 9
    assert out["labelled_per_task"][0] == 10 and out["own_rate_judged_tasks"] == []
    assert big["flagged_by_own_rate"] == [] and big["flagged_tasks"] == [] and big["passes"]
    spec[(0, "failure")] += [("base/t0/i9", False)]                                # the 10th failed-parent point
    out = pilot.project(labelled(spec), strata, {"big": 90}, pools={"big": 3}, n_boot=20)
    assert out["own_rate_judged_tasks"] == [0] and out["big/stratified"]["flagged_by_own_rate"] == [0]
    assert out["big/stratified"]["flagged_tasks"] == [0] and out["big/stratified"]["passes"]   # 1 flagged <= 3
    with pytest.raises(pilot.PilotError, match="own_basis"):
        pilot.project(labelled(spec), strata, {"big": 90}, pools={"big": 3}, n_boot=5, own_basis="points")
    # the shrinkage projection is reported for every budget and scheme, and does not decide
    sh = out["big/stratified"]
    assert set(sh["shrinkage_expected"]) == set(range(10)) and isinstance(sh["shrinkage_passes"], bool)
    assert sh["shrinkage_expected"][0] < sh["expected"][0]     # task 0's 0/10 pulls its own estimate down


def test_shrinkage_is_pooled_without_spread_and_near_own_rates_with_much():
    same = {(t, "failure"): (2, 10) for t in range(10)}                         # no between-task spread
    fit = pilot.shrunk_rates(same)
    assert fit["fit"]["failure"]["kappa_at_grid_top"]
    assert all(r == pytest.approx(0.2, abs=1e-3) for r in fit["rates"].values())
    split = {(t, "failure"): ((40, 40) if t < 5 else (0, 40)) for t in range(10)}  # half always, half never
    fit = pilot.shrunk_rates(split)
    k = fit["fit"]["failure"]["kappa"]
    assert k < 1 and fit["rates"][(0, "failure")] > 0.97 and fit["rates"][(9, "failure")] < 0.03
    mu = fit["fit"]["failure"]["mu"]
    assert fit["rates"][(0, "failure")] == pytest.approx((40 + mu * k) / (40 + k))   # the posterior mean


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
    old = pilot.measured_budgets(est, hourly_usd=0.472, phase1_usd=3.24, dagger_branch_usd=1.62, dagger_round_usd=6.04)
    assert old["current"] == pytest.approx(47, abs=0.5) and old["moved"] == pytest.approx(107, abs=0.5)
    b = pilot.measured_budgets(est, hourly_usd=0.472)                     # amendment 61's re-declared budgets
    assert b["current"] == pytest.approx(194, abs=0.5) and b["moved"] == pytest.approx(227, abs=0.5)
    assert pilot.declared_budgets(0.472) == pytest.approx(b)
    assert sum(pilot.PLAN_USD.values()) + 1.60 + 0.98 + 1.65 + 0.1 * 25.8273 == pytest.approx(30.06, abs=0.01)
    dear = pilot.measured_budgets({k: 2 * v for k, v in est.items()}, hourly_usd=0.472)
    assert dear["current"] < 120 and dear["moved"] < 140


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


# ---------------------------------------------------------------- a final tier cut short (amendment 76)

def _two_tranche_draw(per_a=3, per_total=12, seed=7):
    """Tranche A and a later tranche as ONE stratified draw (A's 3 points per task, then 9 more), labels as
    `make_plan` writes them, 5 jobs per later point in the plan's tier order."""
    from dataclasses import asdict
    from vlar_ops.plan import Job
    pts, success = [], {}
    for t in range(3):
        for i in range(8):
            ep = f"base/t{t}/i{i}"
            success[ep] = i % 3 == 0
            pts += [DecisionPointRef(f"{ep}/dp{d:02d}", ep, "base", t, i, d, 300) for d in range(5)]
    dp = {"scheme": "stratified", "seed": seed, "failed_share": 2 / 3, "min_calls": 1}
    sel = lambda k: stratified_selection(pts, success, step_limit=520, interval=10,   # noqa: E731
                                         dp_sampling={**dp, "per_task": k})
    a, total = sel(per_a), sel(per_total)
    f = pilot.exclude_earlier(total, a, set(a), earlier={"phase": "phase1a", "plan_id": "pa", "per_task": per_a})
    point = {p.snapshot_id: asdict(p) for p in pts}
    lab = lambda s, rec, k: {"point": point[s], "selection": rec,                     # noqa: E731
                             "settings": {"dp_sampling": {**dp, "per_task": k}}}
    labels_a = {s: lab(s, r, per_a) for s, r in a.items()}
    labels_f = {s: lab(s, r, per_total) for s, r in f.items()}
    jobs = [(f"{s}#{j}", "pending", Job(f"{s}#{j}", "phase1f", "b1_continue", s, 1, 1 + (j >= 3), 5 * k + j, {}))
            for k, s in enumerate(sorted(labels_f)) for j in range(5)]
    return pts, success, labels_a, labels_f, jobs


def _stratum_of(e):
    st = e["selection"]["stratum"]
    return int(st["task_id"]), st["parent_outcome"]


def test_a_final_tier_stopped_after_any_job_leaves_exactly_the_draw_at_a_smaller_count():
    from vlar_ops.plan import unit_uniform
    pts, success, labels_a, labels_f, jobs = _two_tranche_draw()
    assert len(labels_a) == 9 and len(labels_f) == 27 and len(jobs) == 135
    _, order = pilot.resume_order(jobs, labels_f)
    stratum = {}
    for p in pts:
        stratum.setdefault((p.task_id, "success" if success[p.episode_id] else "failure"), []).append(p)
    draw = {k: [q.snapshot_id for q in sorted(v, key=lambda q: (unit_uniform("dp-stratified", 7, q.episode_id,
                                                                             q.index), q.snapshot_id))]
            for k, v in stratum.items()}
    planned = {k: sum(1 for e in labels_f.values() if _stratum_of(e) == k) for k in draw}
    for cut in range(len(order) + 1):
        claimed = set(order[:cut])
        finished = {s for s in labels_f if all(f"{s}#{j}" in claimed for j in range(5))}
        r = pilot.finished_inclusion(labels_a, labels_f, finished)
        assert r["dropped"] == [] and set(r["points"]) == set(labels_a) | finished
        progress = {}
        for k, ids in draw.items():
            kept = [s for s in ids if s in r["points"]]
            assert kept == ids[:len(kept)]                                  # the draw's first k: one draw, smaller k
            assert all(r["points"][s] == pytest.approx(len(kept) / len(ids)) for s in kept)
            progress[k] = sum(1 for s in kept if s in labels_f) / planned[k]
        assert max(progress.values()) - min(progress.values()) <= 1 / 3 + 1e-9   # every cell near its share
    full = pilot.finished_inclusion(labels_a, labels_f, set(labels_f))
    assert all(full["points"][s] == pytest.approx(labels_f[s]["selection"]["inclusion_prob"]) for s in labels_f)
    combined = {_stratum_of(e): e["selection"]["inclusion_prob"] for e in labels_f.values()}
    assert all(full["points"][s] == pytest.approx(combined[_stratum_of(e)]) for s, e in labels_a.items())


def test_finished_inclusion_drops_points_past_a_gap_and_refuses_two_draws():
    from vlar_ops.plan import unit_uniform
    _, _, labels_a, labels_f, _ = _two_tranche_draw()
    cell = sorted((s for s, e in labels_f.items() if _stratum_of(e) == (0, "failure")),
                  key=lambda s: pilot.draw_key(labels_f[s]))
    k_a = sum(1 for e in labels_a.values() if _stratum_of(e) == (0, "failure"))
    r = pilot.finished_inclusion(labels_a, labels_f, {cell[0], cell[2], cell[3]})
    assert r["dropped"] == [cell[2], cell[3]] and r["strata"]["t0/failure"]["later_finished"] == 1
    assert r["points"][cell[0]] == pytest.approx((k_a + 1) / labels_f[cell[0]]["selection"]["stratum_size"])
    first_a = min((s for s, e in labels_a.items() if _stratum_of(e) == (0, "failure")),
                  key=lambda s: pilot.draw_key(labels_a[s]))
    swapped_a = {**{s: e for s, e in labels_a.items() if s != first_a}, cell[-1]: labels_f[cell[-1]]}
    swapped_f = {**{s: e for s, e in labels_f.items() if s != cell[-1]}, first_a: labels_a[first_a]}
    with pytest.raises(pilot.PilotError, match="not one draw"):
        pilot.finished_inclusion(swapped_a, swapped_f, set())
    other = {s: {**e, "settings": {"dp_sampling": {**e["settings"]["dp_sampling"], "seed": 8}}}
             for s, e in labels_f.items()}
    with pytest.raises(pilot.PilotError, match="seeds"):
        pilot.finished_inclusion(labels_a, other, set())
    assert unit_uniform("dp-stratified", 7, "base/t0/i1", 2) == pilot.draw_key(
        {"point": {"episode_id": "base/t0/i1", "index": 2, "snapshot_id": "x"}, "selection": {"by": "point"},
         "settings": {"dp_sampling": {"scheme": "stratified", "seed": 7}}})[1]


def test_the_old_snapshot_id_order_would_not_leave_a_draw():
    """Before amendment 76 a cell ran by snapshot id (initial state first): a stop would keep the low-numbered
    initial states, not the draw's next points. The new order differs from it in some cell."""
    _, _, _, labels_f, jobs = _two_tranche_draw()
    points, _ = pilot.resume_order(jobs, labels_f)
    by_cell = {}
    for p in points:
        by_cell.setdefault(_stratum_of(labels_f[p]), []).append(p)
    assert any(c != sorted(c) for c in by_cell.values())


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


def test_costs_come_from_the_resumed_boot_with_the_first_boot_apart():
    """Amendment 62 (1): the first boot (earliest claim) is costed alone; the later boots give the costs."""
    def m(kind, claimed, finished, boot):
        return {"job": {"kind": kind}, "finished_at": finished,
                "meta": {"claimed_at": claimed, "worker_id": "w1", "boot_id": boot}}
    ms = [m("b1_continue", 100, 130, "first"), m("b1_continue", 1000, 1010, "resume"),
          m("b4_max", 1000, 1040, "resume")]
    timing = [{"boot_id": "first", "phase_started": 90.0}, {"boot_id": "resume", "phase_started": 995.0},
              {"boot_id": "resume", "phase_ended": 1045.0}]
    c = pilot.costs_by_boot(ms, timing, hourly_usd=0.472)
    assert c["first_boot"] == "first" and c["later_boots"] == ["resume"] and c["untagged_jobs"] == 0
    assert c["first"]["vm_s_per_run"] == {"b1_continue": pytest.approx(40.0)}       # 90..130
    assert c["resumed"]["windows_s"] == [50.0] and c["resumed"]["runs"] == {"b1_continue": 1, "b4_max": 1}
    assert c["resumed"]["vm_s_per_run"]["b1_continue"] == pytest.approx(50 * 10 / 50)
    one = pilot.costs_by_boot(ms[:1], timing[:1], hourly_usd=0.472)
    assert one["resumed"] is None and one["first"]["runs"] == {"b1_continue": 1}


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
    # amendment 62 (4): 8 x 5.5 GiB + the servers' 20.2 GB peak is 27 MB over RAM; 7 fit with 5.9 GB to spare
    assert f["counts"][7]["fits"] and not f["counts"][8]["fits"] and f["max_workers_fit"] == 7
    assert f["counts"][8]["headroom_bytes"] == 67_416_563_712 - 8 * 22 * 256 * 1024 ** 2 - f["servers_peak"] < 0
    assert f["counts"][8]["worker_peaks_fit_ram"]            # measured peaks would fit, but do not decide
    assert f["counts"][8]["worker_peaks_bytes"] == 8 * 4_500_000_000 + f["servers_peak"] + 4 * GiB


def test_check_timing_per_step_and_warm_up():
    iii = {"record_off": [
        {"subject": "a", "loads": 4, "outcome": {"steps": 400}, "seconds": {"search": 3.0, "warm_up": 2.0, "episode": 40.0}},
        {"subject": "b", "loads": 2, "outcome": {"steps": 100}, "seconds": {"search": 1.0, "warm_up": 0.0, "episode": 10.0}},
        {"subject": "c", "loads": 2, "outcome": {"steps": 100}}]}                  # a bb7112f report: no seconds
    t = pilot.check_timing(iii)
    assert t["timed"] == 2 and t["episode_s_per_step"] == pytest.approx(0.1)
    assert t["warm_up_s_per_load"] == pytest.approx(1.0) and t["search_s_per_episode"] == pytest.approx(2.0)


def test_a_later_tranche_is_one_draw_with_the_earlier_one():
    """The final tier (amendment 67 (3), the user 2026-11-26): A and F one stratified draw: the same seed at the
    combined count holds the earlier draw; the later tranche is the rest, at the combined inclusion probability."""
    pts, success = population({0: (3, 7), 1: (1, 9)})
    prev, total = select(pts, success, per_task=15), select(pts, success, per_task=21)
    assert set(prev) <= set(total)
    later = pilot.exclude_earlier(total, prev, set(prev), earlier={"phase": "phase1a", "plan_id": "x", "per_task": 15})
    assert set(later) == set(total) - set(prev) and len(later) == 12
    for sid, rec in later.items():
        assert rec["inclusion_prob"] == total[sid]["inclusion_prob"] == rec["stratum_drawn"] / rec["stratum_size"]
        assert rec["combined_draw"] == {"phase": "phase1a", "plan_id": "x", "per_task": 15}
    by = lambda t, o: sum(1 for r in later.values()                                       # noqa: E731
                          if r["stratum"]["task_id"] == t and r["stratum"]["parent_outcome"] == o)
    assert (by(0, "failure"), by(0, "success")) == (4, 2)          # round(2/3 x 21) = 14 failed, 10 of them earlier


def test_a_later_tranche_refuses_an_earlier_draw_that_is_not_the_first_of_the_order():
    pts, success = population({0: (3, 7)})
    prev, total = select(pts, success, per_task=15), select(pts, success, per_task=21)
    other = select(pts, success, per_task=15, seed=6)                # another seed: not one draw
    with pytest.raises(pilot.PilotError, match="not one draw"):
        pilot.exclude_earlier(total, prev, set(other), earlier={"phase": "a", "plan_id": "x", "per_task": 15})
