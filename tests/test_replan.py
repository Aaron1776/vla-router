"""The re-plan after the pilot (SPEC §16 item 7; §17 amendments 61, 63, 64): `vlar_ops/replan.py`."""

from __future__ import annotations

import math
import random

import pytest

from vlar_ops import replan as R

PILOT_COSTS = {"vm_s_per_run": {"b1_continue": 8.284038451970996, "b2_one_call": 8.126565194637708},
               "vm_s_per_step": {"b1_continue": 0.02188949952134544}}


def prices(**kw):
    return R.Prices.from_pilot(PILOT_COSTS, **kw)


def test_points_cost_branches_1_and_2_only_with_the_rerun_shared_by_k_points():
    p = prices()
    assert R.point_vm_s(p, "base") == pytest.approx(2 * 8.284038451970996 + 3 * 8.126565194637708)
    assert R.point_vm_s(p, "rerun", k=2) == pytest.approx(3 * (8.284038451970996 + 8.126565194637708)
                                                          + 73.85 * 8 / 7 / 2)
    assert R.point_vm_s(p, "rerun", k=4) < R.point_vm_s(p, "rerun", k=2)
    fast = prices(scale_branch=6 / 7, scale_full=1.0)
    assert R.point_vm_s(fast, "base") == pytest.approx(R.point_vm_s(p, "base") * 6 / 7)
    with pytest.raises(ValueError):
        R.point_vm_s(p, "dagger")


def test_the_coupon_budget_spends_exactly_what_is_available_and_points_follow_the_prices():
    p, plan = prices(), R.Plan(available_usd=27.50)
    b = R.coupon_budget(p, plan)
    vm = b["planned_vm_usd"]
    assert vm * (1 + plan.reserve_share) + plan.disk_usd + plan.storage_usd == pytest.approx(27.50)
    assert b["labels_usd"] == pytest.approx(vm - sum(b["fixed_usd"].values()))
    assert b["phase1_labels_usd"] == pytest.approx(2 * b["dagger_labels_usd"])                 # 2:1
    cb, ce = b["usd_per_point"]["base"], b["usd_per_point"]["rerun"]
    n, m = b["phase1_points"] / 2, b["dagger_points"] / 2
    assert n * (cb + ce) == pytest.approx(b["phase1_labels_usd"])          # n base + n exploration points
    assert m * 2 * ce == pytest.approx(b["dagger_labels_usd"])             # m points per half, both halves re-run
    assert b["points_per_task"]["current"] == pytest.approx((2 * n + 2 * m) / 10)
    moved = (b["phase1_labels_usd"] + b["dagger_labels_usd"] + b["fixed_usd"]["dagger_screening"]) / (cb + ce)
    assert b["points_per_task"]["moved"] == pytest.approx(2 * moved / 10)
    # exploration screening: X-VLA steps and p = 0.1's π0.5 calls, no /features (profile none)
    assert b["fixed_usd"]["explore_screening"] == pytest.approx(R.usd(500 * (312 * p.step + 3.1 * 0.5)))
    # cheaper prices buy more points; less money buys fewer
    assert R.coupon_budget(prices(scale_branch=6 / 7, scale_full=1.0), plan)["points_per_task"]["current"] > \
        b["points_per_task"]["current"]
    assert R.coupon_budget(p, R.Plan(available_usd=20.0))["points_per_task"]["current"] < \
        b["points_per_task"]["current"]
    with pytest.raises(R.pilot.PilotError, match="no label money"):
        R.coupon_budget(p, R.Plan(available_usd=3.0))


def test_phase_lines_stop_branch_phases_at_their_allocation_and_let_rollouts_overrun_8_percent():
    b = R.coupon_budget(prices(), R.Plan(available_usd=27.50))
    lines = R.phase_lines(b)
    for r in lines:
        branch = "branches" in r["phase"]
        assert r["line_usd"] == pytest.approx(r["allocation_usd"] * (1.0 if branch else 1.08))
        assert r["stop_usd"] == pytest.approx(max(0.0, r["line_usd"] - 1200 / 3600 * R.HOURLY_USD))   # billed 0.6865
    assert len(lines) == 5 and sum(r["allocation_usd"] for r in lines) == pytest.approx(
        sum(b["fixed_usd"].values()) + b["labels_usd"])


def test_the_request_wait_is_the_network_models_own_expectation():
    """`request_wait` (exact) against the harness's simulated network: the same draws the clock charges."""
    from vlar_harness import clock as clk
    from vlar_harness.network import NetworkParams, SimulatedNetwork
    net = NetworkParams(uplink_mbps=20, downlink_mbps=20, rtt_median_s=0.06, rtt_tail_s=0.3, rtt_tail_prob=0.02,
                        p_fail=0.05, timeout_s=1.0, backoff_s=0.2, max_retries=2, max_attempts_pi05_only=20,
                        upload_bytes=20425 + 1024, download_bytes=2048, pi05_infer_s=0.7161023695000495)
    w = R.request_wait(net)
    link = SimulatedNetwork(net)
    rng = random.Random(3)
    n, total, abandoned = 40000, 0.0, 0
    for _ in range(n):
        out = link.request(lambda: None, step=rng.randrange(520), rollout_seed=rng.randrange(2 ** 31),
                           max_attempts=net.max_attempts_routed)
        total += clk.request_s(out.attempts, net)
        abandoned += not out.ok
    assert total / n == pytest.approx(w["mean_request_s"], rel=0.01)
    assert abandoned / n == pytest.approx(w["abandon_prob"], abs=0.0015)
    assert w["mean_request_s"] > net.pi05_infer_s + net.rtt_median_s       # retries and the round trip's tail


def test_the_evaluation_sheet_adds_up_and_prices_each_decider_by_its_profile():
    p = prices()
    s = R.evaluation_sheet(p, R.EvalCase())
    assert s["total_usd"] == pytest.approx(s["vm_total_usd"] * 1.1 + s["disk_usd"] + s["storage_usd"])
    v = s["vm_usd"]
    assert v["trunk (record full; also the X-VLA-alone baseline)"] == pytest.approx(R.usd(500 * p.rerun))
    full = v["post-DAgger cascader_full suffixes"]
    assert full > 5 * v["post-DAgger cascader_nospread suffixes"]             # probes everywhere vs /features
    assert v["pre-DAgger router suffixes"] == v["post-DAgger router suffixes"] == \
        v["pre-DAgger cascader_nospread suffixes"]                              # same profile, same price
    up = R.evaluation_sheet(p, R.EvalCase(q=1.0, first_at=0.2, escalations=8.0, baseline_escalations=8.0,
                                          disk_days=3.0))
    assert up["total_usd"] > s["total_usd"]
    assert set(s["optional_usd"]) == {"cascader_chunkspread (post-DAgger) with its audit",
                                      "λ sweep: router and cascader_nospread at two more λ"}
    no_setup = R.evaluation_sheet(p, R.EvalCase(), setup=False)
    assert s["vm_total_usd"] - no_setup["vm_total_usd"] == pytest.approx(R.usd(p.setup_vm_s))


def test_decision_at_reads_both_budgets_and_the_shrinkage_beside():
    def proj(cur, mov, shc=True, shm=True):
        return {"current/stratified": {"passes": cur, "shrinkage_passes": shc, "bootstrap_pass_share": 0.9},
                "moved/stratified": {"passes": mov, "shrinkage_passes": shm, "bootstrap_pass_share": 0.95}}
    assert R.decision_at(proj(True, True))["decision"] == "keep"
    assert R.decision_at(proj(False, True))["decision"] == "drop"
    assert R.decision_at(proj(False, False))["decision"] == "stop"
    d = R.decision_at(proj(True, True, shc=False, shm=True))
    assert d["decision"] == "keep" and d["shrinkage_decision"] == "drop"
    assert math.isclose(d["bootstrap_pass_share"]["moved"], 0.95)
