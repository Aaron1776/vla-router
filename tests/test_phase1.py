"""Phase 1 after the handoff (SPEC §17 amendment 65): `vlar_ops/phase1.py`."""

from __future__ import annotations

import pytest

from vlar_ops import phase1 as P
from vlar_ops import replan as R

PILOT_COSTS = {"vm_s_per_run": {"b1_continue": 8.284038451970996, "b2_one_call": 8.126565194637708},
               "vm_s_per_step": {"b1_continue": 0.02188949952134544}}


def prices(**kw):
    return R.Prices.from_pilot(PILOT_COSTS, **kw)


def test_the_budget_spends_the_credit_less_the_end_reserve_and_splits_labels_into_a_b_and_the_final_tier():
    p, plan = prices(), P.Phase1Plan(credit_usd=33.12, untracked_usd=0.4)
    b = P.budget(p, plan)
    other = plan.end_reserve_usd + plan.disk_usd + plan.storage_usd + plan.untracked_usd
    assert b["planned_vm_usd"] * 1.1 + other == pytest.approx(33.12)
    assert b["labels_usd"] == pytest.approx(b["planned_vm_usd"] - sum(b["fixed_usd"].values()))
    a, bb, f = b["tranche_a"], b["tranche_b"], b["final"]
    assert a["points"] == pytest.approx(bb["points"])                                    # n base + n exploration
    assert a["labels_usd"] + bb["labels_usd"] + f["labels_usd"] == pytest.approx(b["labels_usd"])
    assert a["labels_usd"] == pytest.approx(a["points"] * b["usd_per_point"]["base"])
    assert bb["rerun_episodes"] == pytest.approx(bb["points"] / 2)
    # the default final tier buys exactly one b4 seed at every A and B point, about a fifth of the label money
    assert f["b4_covers"] == pytest.approx(1.0) and 0.18 < b["final_share"] < 0.23
    assert f["or_base_points"] == pytest.approx(f["labels_usd"] / b["usd_per_point"]["base"])
    # the screen is sized to supply the failed stratum: share x n / k failed re-run episodes, at the failure rate,
    # with a 10% margin; each screened episode carries a fresh simulator (option (A))
    tb = b["tranche_b"]
    assert tb["failed_episodes_needed"] == pytest.approx(2 / 3 * tb["points"] / 2)
    assert tb["screened_episodes"] == pytest.approx(max(500, 1.1 * tb["failed_episodes_needed"] / (102 / 500)))
    assert tb["screened_episodes"] * 102 / 500 > tb["failed_episodes_needed"]
    assert b["fixed_usd"]["explore_screening"] == pytest.approx(
        R.usd(tb["screened_episodes"] * (p.screen(3.1, False) + plan.fresh_sim_vm_s)))
    small = P.budget(p, P.Phase1Plan(credit_usd=12.0))                    # few points: the 500-episode floor binds
    assert small["tranche_b"]["screened_episodes"] == 500
    fixed_share = P.budget(p, P.Phase1Plan(credit_usd=33.12, final_share=0.2))
    assert fixed_share["final"]["labels_usd"] == pytest.approx(0.2 * fixed_share["labels_usd"])
    assert P.budget(p, P.Phase1Plan(credit_usd=25.0))["tranche_a"]["points"] < a["points"]
    with pytest.raises(P.Phase1Error, match="no label money"):
        P.budget(p, P.Phase1Plan(credit_usd=7.5))


def test_lines_put_the_setup_in_tranche_as_phase_and_stop_branch_money_at_its_allocation():
    b = P.budget(prices(), P.Phase1Plan(credit_usd=33.12))
    rows = {r["phase"]: r for r in P.lines(b)}
    assert list(rows) == ["phase1a", "phase1b-screen", "phase1b", "phase1f"]
    fx = b["fixed_usd"]
    assert rows["phase1a"]["allocation_usd"] == pytest.approx(fx["setup"] + b["tranche_a"]["labels_usd"])
    assert rows["phase1a"]["line_usd"] == pytest.approx(fx["setup"] * 1.08 + b["tranche_a"]["labels_usd"])
    assert rows["phase1b-screen"]["line_usd"] == pytest.approx(fx["explore_screening"] * 1.08)
    assert rows["phase1b"]["line_usd"] == pytest.approx(rows["phase1b"]["allocation_usd"])
    assert sum(r["allocation_usd"] for r in rows.values()) == pytest.approx(b["labels_usd"] + sum(fx.values()))
    for r in rows.values():
        assert r["stop_usd"] == pytest.approx(r["line_usd"] - P.DRAIN_MARGIN_USD)
    # at the real price the drain margin grows; Tranche A's headroom lifts only its own line
    hi = {r["phase"]: r for r in P.lines(b, hourly_usd=0.6865, a_headroom=0.15)}
    assert hi["phase1a"]["line_usd"] == pytest.approx(rows["phase1a"]["line_usd"] + 0.15 * rows["phase1a"]["allocation_usd"])
    assert hi["phase1b"]["line_usd"] == pytest.approx(rows["phase1b"]["line_usd"])
    assert hi["phase1b"]["stop_usd"] == pytest.approx(hi["phase1b"]["line_usd"] - 1200 / 3600 * 0.6865)


def test_guard_limits_leave_the_end_reserve_and_warn_when_the_reserve_begins():
    b = P.budget(prices(), P.Phase1Plan(credit_usd=33.12))
    g = P.guard_limits(16.88, b)
    assert g["hard_limit_usd"] == pytest.approx(16.88 + 33.12 - 5.0)
    assert g["soft_limit_usd"] == pytest.approx(g["hard_limit_usd"] - b["reserve_usd"], abs=0.01)


def test_effective_positives_is_kish_over_expected_positives():
    folds = {0: [1, 2]}
    equal = {(1, "a"): (10, 100, 0.5), (2, "a"): (10, 100, 0.5)}         # equal weights: ESS = the expected count
    assert P.effective_positives(equal, folds)[0] == pytest.approx({"effective": 10.0, "expected": 10.0})
    # weights 10 and 1 on 5 expected positives each: (50 + 5)^2 / (500 + 5)
    mixed = {(1, "a"): (10, 100, 0.5), (2, "a"): (10, 10, 0.5)}
    assert P.effective_positives(mixed, folds)[0]["effective"] == pytest.approx(55 ** 2 / 505)
    assert P.effective_positives({(3, "a"): (10, 10, 0.5)}, folds)[0]["effective"] == 0.0   # not a training task


def test_stratum_alloc_caps_by_the_frame_and_makes_up_the_shortfall():
    assert P.stratum_alloc(90, 1000, 1000, 2 / 3) == (60, 30)
    assert P.stratum_alloc(90, 40, 1000, 2 / 3) == (40, 50)          # failed short: the rest from success
    assert P.stratum_alloc(90, 1000, 10, 2 / 3) == (80, 10)          # success short: the rest from failed


def _frame(nf=800, ns=900):
    return {(t, o): (nf if o == "failure" else ns) for t in range(10) for o in ("failure", "success")}


def test_better_share_picks_the_larger_worst_fold_and_defaults_to_two_thirds_on_a_tie():
    r = P.better_share(_frame(), 100, 0.22, 0.02)
    by = r["by_share"]
    assert r["pick"] == (2 / 3 if by["0.6667"]["min"] > by["0.7500"]["min"] else 3 / 4)
    assert by["0.7500"]["min_expected"] > by["0.6667"]["min_expected"]      # more raw positives at 3/4
    # a frame like the base rollouts' (few failed points in some tasks): 3/4's heavier success weights cost more
    real = {(t, o): v for t, (f, s) in enumerate([(1040, 886), (832, 956), (260, 1109), (208, 1120), (780, 780),
                                                  (312, 806), (832, 824), (364, 1106), (572, 1565), (104, 1308)])
            for o, v in (("failure", f), ("success", s))}
    assert P.better_share(real, 100, 13 / 59, 1 / 50)["pick"] == 2 / 3
    tie = P.better_share(_frame(), 100, 0.22, 0.22, shares=(2 / 3, 2 / 3))
    assert tie["pick"] == 2 / 3


def _rows(early_pos, early_n, late_pos, late_n, mid_pos=0, mid_n=0):
    rows = [{"index": 5, "positive": i < early_pos} for i in range(early_n)]
    rows += [{"index": 25, "positive": i < mid_pos} for i in range(mid_n)]
    return rows + [{"index": 40, "positive": i < late_pos} for i in range(late_n)]


def _stretch_frame(sizes=(300, 300, 300)):
    return {(t, k): s for t in range(10) for k, s in zip(P.STRETCHES, sizes)}


def test_stretch_rule_needs_both_a_bonferroni_difference_and_a_ten_percent_worst_fold_gain():
    frame, by = _frame(900, 900), _stretch_frame()
    flat = P.stretch_rule(_rows(5, 20, 5, 20, 5, 20), by, frame, 100, 0.02)
    assert flat["splits"] == {} and flat["apply"] is None                   # no difference: nothing tested further
    strong = P.stretch_rule(_rows(18, 20, 0, 20, 0, 20), by, frame, 100, 0.02)
    assert strong["tests"]["early"]["p"] < 0.05 / 3 and "early" in strong["splits"]
    sp = strong["splits"]["early"]
    assert sp["high_side"] == ["early"] and sp["rates"]["high"] == pytest.approx(0.9)
    assert sp["gain_min"] >= 0.10 and strong["apply"] in strong["splits"]
    assert min(sp["fold_gains"]) == pytest.approx(sp["gain_min"], abs=1e-9) or sp["gain_min"] <= max(sp["fold_gains"])
    # p = 0.031 for early (7/20 against 4/40): under 0.05 but not under 0.05/3, so no split is even considered
    border = P.stretch_rule(_rows(7, 20, 2, 20, 2, 20), by, frame, 100, 0.02)
    assert 0.05 / 3 < border["tests"]["early"]["p"] < 0.05 and border["splits"] == {} and border["apply"] is None
    # the same difference with the gain required out of reach: not applied
    assert P.stretch_rule(_rows(18, 20, 0, 20, 0, 20), by, frame, 100, 0.02, min_gain=10.0)["apply"] is None
    with pytest.raises(P.Phase1Error):
        P.stretch_of(52)


def test_tranche_a_settings_are_amendment_57s_draw_with_branches_1_and_2():
    s = P.tranche_a_settings(103, 20261125)
    assert s["phase"] == "phase1a" and s["enabled"] == ["b1_continue", "b2_one_call"]
    assert s["seeds"] == 3 and s["tier_one_seeds"] == 2
    assert s["selection"] == {"scheme": "stratified", "seed": 20261125, "per_task": 103, "failed_share": 2 / 3,
                              "min_calls": 1}


def test_recording_every_exploration_episode_in_full_is_priced_without_screening_or_re_runs():
    p = prices()
    scr = P.budget(p, P.Phase1Plan(credit_usd=33.12))
    full = P.budget(p, P.Phase1Plan(credit_usd=33.12, tranche_b="full"))
    assert "explore_screening" not in full["fixed_usd"]
    assert full["fixed_usd"]["explore_full_episodes"] == pytest.approx(R.usd(500 * (p.rerun + 3.0 / 7)))
    assert full["vm_s_per_point"]["rerun"] == pytest.approx(3 * p.run_b1 + 3 * p.run_b2)   # branches only
    assert full["tranche_b"]["rerun_episodes"] == 0 and full["tranche_b"]["full_episodes"] == 500
    for b in (scr, full):
        assert b["labels_usd"] == pytest.approx(b["tranche_a"]["labels_usd"] + b["tranche_b"]["labels_usd"]
                                                + b["final"]["labels_usd"])
    with pytest.raises(P.Phase1Error):
        P.budget(p, P.Phase1Plan(credit_usd=33.12, tranche_b="other"))


def test_fisher_two_sided_matches_scipy():
    assert P.fisher_two_sided(9, 9, 4, 37) == pytest.approx(0.0013384159468717964, rel=1e-9)   # the pilot's early
    assert P.fisher_two_sided(0, 17, 13, 29) == pytest.approx(0.01214457600226028, rel=1e-9)   # and late stretches
    assert P.fisher_two_sided(5, 5, 5, 5) == pytest.approx(1.0)
    scipy = pytest.importorskip("scipy.stats")
    for t in ([[3, 7], [8, 2]], [[0, 5], [6, 1]], [[12, 30], [4, 41]], [[1, 0], [0, 1]]):
        assert P.fisher_two_sided(*t[0], *t[1]) == pytest.approx(scipy.fisher_exact(t)[1], rel=1e-9)


def test_the_final_tier_check_under_cascader_nospread():
    """Amendment 66 (1): A 43/21 and the extra base points 36/18 (the plan doc's 9.9 and 8.3 per task); B by episode,
    21 failed episodes of 2 points (42/22, not 43/21), capped by the screen's failed episodes at the base rate."""
    rates = {"failure": 13 / 59, "success": 1 / 50}
    out = P.final_tier_check(rates, {0: 1.0, 1: 0.0, 2: 0.04, **{t: 1.0 for t in range(3, 10)}})
    rf, rs = rates["failure"], rates["success"]
    assert out["per_point"]["A"] == pytest.approx(43 * rf + 21 * rs) and round(out["per_point"]["A"], 1) == 9.9
    assert out["per_point"]["F_base"] == pytest.approx(36 * rf + 18 * rs) and round(out["per_point"]["F_base"], 1) == 8.3
    unc, cap = out["uncapped"], out["capped"]
    assert unc["b4_every_point"]["B"][1] == pytest.approx(42 * rf + 22 * rs)
    assert unc["b4_every_point"]["expected_per_task"][0] == pytest.approx(85 * rf + 43 * rs)        # 19.59 < 20
    assert unc["b4_every_point"]["flagged_tasks"] == list(range(10)) and not unc["b4_every_point"]["passes"]
    assert not unc["b4_every_point"]["folds_short"]                                                  # 8 x 19.6
    assert unc["more_base_points"]["passes"] and unc["more_base_points"]["expected_per_task"][0] == pytest.approx(
        121 * rf + 61 * rs)
    assert cap["b4_every_point"]["B"][0] == pytest.approx(42 * rf + 22 * rs)                         # p = 1: no cap
    assert cap["b4_every_point"]["B"][1] == pytest.approx(64 * rs)                                   # p = 0: none failed
    assert 64 * rs < cap["b4_every_point"]["B"][2] < unc["b4_every_point"]["B"][2]


def test_the_final_tier_is_one_draw_with_tranche_a_at_the_combined_count():
    s = P.final_tier_settings(64, 18, 20261125)
    a = P.tranche_a_settings(64, 20261125)
    assert s["phase"] == "phase1f" and s["order_seed"] == 13 != a["order_seed"]
    assert s["selection"] == {**a["selection"], "per_task": 82}
    assert (s["enabled"], s["seeds"], s["tier_one_seeds"]) == (a["enabled"], a["seeds"], a["tier_one_seeds"])
    with pytest.raises(P.Phase1Error):
        P.final_tier_settings(64, 0, 20261125)


def test_the_final_tier_is_sized_to_its_stop_point():
    r = P.final_tier_points(1.50)
    assert r["point_usd"] == pytest.approx(5 * 7.18 / 3600 * 0.6865) and r["overhead_usd"] == pytest.approx(0.33 * 0.6865)
    assert (r["per_task"], r["points"], r["jobs"]) == (18, 180, 900) and r["planned_usd"] <= 1.50
    assert P.final_tier_points(r["overhead_usd"] + 10 * r["point_usd"] * 19)["per_task"] == 19
    assert P.final_tier_points(0.20)["per_task"] == 0 and P.final_tier_points(0.20)["planned_usd"] == 0.0
