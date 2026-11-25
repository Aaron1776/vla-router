import pytest

from vlar_harness.config import load_config
from vlar_ops import branchplan as B
from vlar_ops import plan as P
from vlar_ops.tunables import load_tunables

CFG = load_config()
BR = CFG["branches"]
ALL = list(B.BRANCH_ORDER)


def dp(index, budget, source="explore", snap=None, task=0, init=0):
    return B.DecisionPointRef(snap or f"{source}-t{task}-i{init}-d{index}", f"{source}/t{task}/i{init}",
                              source, task, init, index, budget)


@pytest.mark.parametrize("index,budget,expected", [
    (0, 260, {"b1_continue": 0, "b2_one_call": 1, "b3_three_calls": 3, "b4_max": 26}),
    (40, 260, {"b1_continue": 0, "b2_one_call": 1, "b3_three_calls": 3, "b4_max": 12}),   # episode end caps b4
    (0, 20, {"b1_continue": 0, "b2_one_call": 1, "b3_three_calls": 2, "b4_max": 2}),      # budget caps b3, b4
    (51, 260, {"b1_continue": 0, "b2_one_call": 1, "b3_three_calls": 1, "b4_max": 1}),    # last point
])
def test_branch_calls(index, budget, expected):
    assert B.branch_calls(dp(index, budget), BR, 520, 10) == expected


def test_identical_branches_are_shared_under_lowest_name():
    calls = {"b1_continue": 0, "b2_one_call": 1, "b3_three_calls": 2, "b4_max": 2}
    assert B.shared_branches(calls) == {"b1_continue": "b1_continue", "b2_one_call": "b2_one_call",
                                        "b3_three_calls": "b3_three_calls", "b4_max": "b3_three_calls"}
    calls = {"b1_continue": 0, "b2_one_call": 1, "b3_three_calls": 1, "b4_max": 1}
    assert set(B.shared_branches(calls).values()) == {"b1_continue", "b2_one_call"}


def plan_for(points, **kw):
    args = dict(branches=BR, step_limit=520, interval=10, seeds=3, enabled=ALL, tier_one_seeds=2, order_seed=1,
                dp_sampling={"fraction": 1.0, "seed": 0})
    args.update(kw)
    return B.branch_plan("branches", points, **args)


def test_tiers_and_counts_for_a_full_budget_point():
    plan, labels = plan_for([dp(5, 200)])
    by_tier = {t: [(j.kind, j.seed_index) for j in plan.jobs if j.tier == t] for t in (1, 2, 3)}
    assert sorted(by_tier[1]) == sorted([(b, s) for b in ("b1_continue", "b2_one_call", "b4_max") for s in (1, 2)])
    assert sorted(by_tier[2]) == [("b1_continue", 3), ("b2_one_call", 3), ("b4_max", 3)]
    assert sorted(by_tier[3]) == [("b3_three_calls", s) for s in (1, 2, 3)]
    assert [j.order for j in plan.jobs] == list(range(len(plan.jobs)))
    assert [j.tier for j in plan.jobs] == sorted(j.tier for j in plan.jobs)   # tiers run in order


def test_base_rollout_is_seed_one_of_continue():
    plan, labels = plan_for([dp(5, 260, source="base")])
    kinds = [(j.kind, j.seed_index) for j in plan.jobs]
    assert ("b1_continue", 1) not in kinds
    assert ("b1_continue", 2) in kinds and ("b1_continue", 3) in kinds
    assert len(plan.jobs) == 11      # 2 + 3 + 3 + 3, the spec's "11 branch rollouts"


def test_shared_job_takes_the_earliest_tier():
    # Budget 20: b3 and b4 both make 2 calls; b3's job also serves b4, so it must run in tiers 1-2.
    plan, labels = plan_for([dp(0, 20)])
    shared = [j for j in plan.jobs if j.kind == "b3_three_calls"]
    assert [(j.seed_index, j.tier) for j in sorted(shared, key=lambda j: j.seed_index)] == [(1, 1), (2, 1), (3, 2)]
    assert not any(j.kind == "b4_max" for j in plan.jobs)
    assert labels[plan.jobs[0].subject]["shares"]["b4_max"] == "b3_three_calls"


def test_points_without_budget_are_skipped_and_order_is_seeded():
    pts = [dp(i, 260 if i % 2 else 0, task=i % 10, init=i) for i in range(40)]
    a, la = plan_for(pts)
    b, lb = plan_for(pts)
    c, _ = plan_for(pts, order_seed=2)
    assert len(la) == 20
    assert [j.job_id for j in a.jobs] == [j.job_id for j in b.jobs]
    assert [j.job_id for j in a.jobs] != [j.job_id for j in c.jobs]
    assert a.plan_id == b.plan_id


def test_jobs_of_one_point_stay_together_within_a_tier():
    pts = [dp(3, 260, task=t, init=t) for t in range(10)]
    plan, _ = plan_for(pts)
    tier1 = [j.subject for j in plan.jobs if j.tier == 1]
    seen, last = set(), None
    for s in tier1:
        if s != last:
            assert s not in seen, "a point's tier-1 jobs were split"
            seen.add(s)
            last = s


def test_dropping_branch_three():
    plan, labels = plan_for([dp(5, 200)], enabled=["b1_continue", "b2_one_call", "b4_max"])
    assert not any(j.tier == 3 for j in plan.jobs)
    assert "b3_three_calls" not in next(iter(labels.values()))["shares"]


def test_complete_points_use_each_entrys_recorded_seeds_and_branches():
    pts = [dp(5, 200, snap="A"), dp(5, 200, snap="B", source="base")]
    two, labels2 = plan_for(pts, seeds=2)
    plan, labels = plan_for(pts)
    tier1 = {j.job_id for j in plan.jobs if j.tier == 1}
    assert P.complete_points(labels2, tier1, required=["b1_continue", "b2_one_call"]) == ["A", "B"]
    assert P.complete_points(labels, tier1, required=["b1_continue", "b2_one_call"]) == []      # seed 3 missing
    done = tier1 | {j.job_id for j in plan.jobs if j.tier == 2}
    assert sorted(P.complete_points(labels, done, required=["b1_continue", "b2_one_call", "b4_max"])) == ["A", "B"]
    assert P.complete_points(labels, done) == []                    # by default every enabled branch: b3 missing
    _, no_b3 = plan_for(pts, enabled=["b1_continue", "b2_one_call", "b4_max"])
    assert sorted(P.complete_points(no_b3, done)) == ["A", "B"]
    with pytest.raises(ValueError, match="planner settings"):
        P.complete_points({"A": {k: v for k, v in labels["A"].items() if k != "settings"}}, done)


def test_the_planner_records_its_settings_and_applies_the_sampling():
    pts = [dp(d, 260, task=t, init=i) for t in range(3) for i in range(5) for d in range(0, 50, 5)]
    plan, labels = plan_for(pts, dp_sampling={"fraction": 0.3, "seed": 11})
    kept = {p.snapshot_id for p in pts if B.dp_is_sampled(11, p.episode_id, p.index, 0.3)}
    assert set(labels) == kept and 0 < len(kept) < len(pts)
    want = {"seeds": 3, "enabled": ALL, "tier_one_seeds": 2, "order_seed": 1, "dp_sampling": {"fraction": 0.3, "seed": 11}}
    assert all(e["settings"] == want for e in labels.values())
    assert all(j.spec["settings"] == want for j in plan.jobs) and plan.params["dp_sampling"] == want["dp_sampling"]
    with pytest.raises(B.BranchPlanError, match="fraction"):
        plan_for(pts, dp_sampling={"fraction": None, "seed": 11})
    with pytest.raises(B.BranchPlanError, match="subset"):
        plan_for(pts, enabled=["b5"])
    settings = B.plan_settings(load_tunables())
    # the shipped tunables: b3 and b4 dropped from Phase 1 and DAgger (SPEC §17 amendment 63 (4))
    assert settings["seeds"] == 3 and settings["enabled"] == ["b1_continue", "b2_one_call"]
    assert settings["dp_sampling"]["fraction"] is None


def test_dp_sampling_is_deterministic_and_near_the_fraction():
    draws = [B.dp_is_sampled(7, f"explore/t{t}/i{i}", d, 0.12) for t in range(10) for i in range(50) for d in range(52)]
    assert draws == [B.dp_is_sampled(7, f"explore/t{t}/i{i}", d, 0.12) for t in range(10) for i in range(50) for d in range(52)]
    assert abs(sum(draws) / len(draws) - 0.12) < 0.01


def test_episode_plan_seeds_and_roundtrip(tmp_path):
    coll = P.load_collection_config()
    plan = P.episode_plan("explore", tasks=range(10), inits=range(50),
                          namespace=coll["rollout_seed_namespaces"]["explore"],
                          mode="routed", profile="none", record="full", order_seed=load_tunables()["order_seed"])
    seeds = [j.spec["rollout_seed"] for j in plan.jobs]
    assert {j.spec["scene"] for j in plan.jobs} == {"project"}
    assert len(set(seeds)) == 500
    assert {j.spec["task_id"] for j in plan.jobs[:30]} != {plan.jobs[0].spec["task_id"]}   # shuffled across tasks
    plan.write(tmp_path / "p.jsonl")
    again = P.Plan.read(tmp_path / "p.jsonl")
    assert again.plan_id == plan.plan_id and again.jobs == plan.jobs


def test_rollout_seed_namespaces_are_disjoint():
    ns = P.load_collection_config()["rollout_seed_namespaces"]
    ranges = sorted((v, v + 100 * 9 + 49) for v in ns.values())
    assert all(a_end < b_start for (_, a_end), (b_start, _) in zip(ranges, ranges[1:]))
