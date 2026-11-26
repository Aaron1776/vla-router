"""Phase 1 after the handoff (SPEC §17 amendment 65; docs/PHASE1-PLAN-2026-11-25.md).

Oscar takes over after Phase 1 (training, the DAgger round and both evaluations, on his DGX Spark); the coupon funds
only Phase 1, storage and the final download. Phase 1 runs in three tranches:
  A  base-parent points (snapshots exist: no screening, no re-runs), b1 and b2 with 3 seeds;
  B  exploration: screening, selection by episode, record-`full` re-runs, b1 and b2 at k points per re-run episode;
  F  the final tier, about a fifth of the label money: one b4 seed at every A and B point if Oscar asks for b4, else
     more base-parent points (b1, b2).
A and B draw the same number of points (amendment 57's n base + n exploration). The DAgger round's label and rollout
money moves into Phase 1; the 10% reserve stays.

Also here: the pre-stated analyses (i) and (ii) of the re-plan (rules fixed in commit 686588b before they ran):
expected effective positives per fold, (Σw)²/Σw² over a fold's expected positives (amendment 57 (6)'s effective
sample size, applied to the positives), for a failed-parent share and for oversampling a stretch of decision points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from . import pilot
from .replan import HOURLY_USD, Prices, point_vm_s, usd

STRETCHES = {"early": (0, 16), "middle": (17, 33), "late": (34, 51)}   # decision-point index, inclusive (pre-stated)
DRAIN_MARGIN_USD = (900 + 300) / 3600 * HOURLY_USD


class Phase1Error(ValueError):
    pass


@dataclass(frozen=True)
class Phase1Plan:
    credit_usd: float                    # the coupon credit left (billing console) less charges not yet posted
    end_reserve_usd: float = 5.0         # storage after Phase 1 and the final download (the user, 2026-11-25)
    disk_usd: float = 1.65               # the Phase 1 VM's boot disk, ~5 days (A, the gap to B, B, F)
    storage_usd: float = 0.30            # the bucket during Phase 1 (~25-40 GB for ~2 weeks at $0.020/GB-month)
    untracked_usd: float = 0.0           # recurring charges outside the ledger during Phase 1 (the billing check)
    reserve_share: float = 0.10          # of planned VM spend (amendment 61; kept by the user)
    final_share: Optional[float] = None  # None: sized for one b4 seed at every A and B point
    explore_min_episodes: int = 500      # every initial state screened at least once
    explore_escalations: float = 3.1     # p = 0.1 at ~31 decision points
    explore_failure_rate: float = 102 / 500   # est.: X-VLA's base rollouts at 520 steps (exploration's is unknown)
    screen_margin: float = 1.1           # screen enough to expect 10% more failed episodes than the share needs
    failed_share: float = 2 / 3
    tranche_b: str = "screen"            # "screen": screen then re-run (amendment 56); "full": record every episode
    full_episodes: int = 500             # "full": exploration episodes recorded in full (each initial state once)
    boots: int = 4                       # B, F, ~2 preemption restarts (A's boot is the setup's)
    k: int = 2
    b4_run_vm_s: float = 17.2            # the pilot's resumed boot, 6 workers
    fresh_sim_vm_s: float = 3.0 / 7      # option (A): a fresh simulator per fresh-start episode (~3 s, 1 of 7 workers)
    tasks: int = 10


def budget(p: Prices, plan: Phase1Plan, hourly_usd: float = HOURLY_USD) -> dict[str, Any]:
    """P + reserve x P + disk + storage + untracked + end reserve = credit, P = fixed + screening + L (planned VM). L
    splits into the final tier F (f x L) and A + B with n points each. f defaults to the share that buys one b4 seed at
    every A and B point: f = b4 / (b4 + (c_b + c_e) / 2).

    Tranche B, "screen" (amendment 56): n / k re-run episodes, `failed_share` of them from failed parents, so the
    screen must expect that many failed episodes (with a margin) at the exploration failure rate: S = max(500,
    margin x share x n / (k x rate)) screened episodes, a cost that grows with n (solved exactly: it is linear).
    "full": `full_episodes` exploration episodes recorded in full (no screening, no re-runs), points then drawn by
    point like base points (3 b1 seeds: the parent is no b1 seed)."""
    other = plan.end_reserve_usd + plan.disk_usd + plan.storage_usd + plan.untracked_usd
    vm = (plan.credit_usd - other) / (1 + plan.reserve_share)
    fresh = plan.fresh_sim_vm_s
    fixed = {"setup": usd(p.setup_vm_s, hourly_usd), "boots": usd(plan.boots * p.boot_vm_s, hourly_usd)}
    cb_s = point_vm_s(p, "base")
    b4_s = plan.b4_run_vm_s * p.scale_branch
    screen_s = p.screen(plan.explore_escalations, False) + fresh            # one screened episode
    full_s = p.rerun + fresh                                                # one record-full episode
    if plan.tranche_b == "screen":
        ce_s = point_vm_s(p, "rerun", k=plan.k) + fresh / plan.k
    elif plan.tranche_b == "full":
        ce_s = 3 * p.run_b1 + 3 * p.run_b2
    else:
        raise Phase1Error(f"tranche_b {plan.tranche_b!r}")
    f = plan.final_share if plan.final_share is not None else b4_s / (b4_s + (cb_s + ce_s) / 2)
    if not 0 <= f < 1:
        raise Phase1Error(f"final share {f}")
    cb, ce, b4 = usd(cb_s, hourly_usd), usd(ce_s, hourly_usd), usd(b4_s, hourly_usd)
    per_n = (cb + ce) / (1 - f)                                             # label money per n (A + B + F)
    room = vm - sum(fixed.values())
    a = plan.screen_margin * plan.failed_share / (plan.k * plan.explore_failure_rate)   # screened episodes per n
    if plan.tranche_b == "screen":
        n = room / (per_n + a * usd(screen_s, hourly_usd))
        screened = a * n
        if screened < plan.explore_min_episodes:                           # the floor binds: a fixed screen
            screened = plan.explore_min_episodes
            n = (room - screened * usd(screen_s, hourly_usd)) / per_n
        fixed["explore_screening"] = usd(screened * screen_s, hourly_usd)
    else:
        screened = 0.0
        fixed["explore_full_episodes"] = usd(plan.full_episodes * full_s, hourly_usd)
        n = (room - fixed["explore_full_episodes"]) / per_n
    labels = vm - sum(fixed.values())
    if n <= 0 or labels <= 0:
        raise Phase1Error(f"no label money: fixed lines {sum(fixed.values()):.2f} > planned VM {vm:.2f}")
    lf = f * labels
    a_labels, b_labels = n * cb, n * ce
    return {"credit_usd": plan.credit_usd, "end_reserve_usd": plan.end_reserve_usd, "disk_usd": plan.disk_usd,
            "storage_usd": plan.storage_usd, "untracked_usd": plan.untracked_usd, "tranche_b_mode": plan.tranche_b,
            "planned_vm_usd": vm, "reserve_usd": vm * plan.reserve_share, "fixed_usd": fixed, "labels_usd": labels,
            "final_share": f, "usd_per_point": {"base": cb, "rerun": ce, "b4_run": b4},
            "vm_s_per_point": {"base": cb_s, "rerun": ce_s, "b4_run": b4_s},
            "tranche_a": {"points": n, "points_per_task": n / plan.tasks, "labels_usd": a_labels},
            "tranche_b": {"points": n, "points_per_task": n / plan.tasks, "labels_usd": b_labels,
                          "rerun_episodes": n / plan.k if plan.tranche_b == "screen" else 0.0,
                          "screened_episodes": screened,
                          "failed_episodes_needed": plan.failed_share * n / plan.k if plan.tranche_b == "screen" else None,
                          "full_episodes": plan.full_episodes if plan.tranche_b == "full" else 0},
            "final": {"labels_usd": lf, "b4_points": lf / b4, "b4_covers": lf / b4 / (2 * n),
                      "or_base_points": lf / cb, "or_base_points_per_task": lf / cb / plan.tasks},
            "vm_hours": vm / hourly_usd}


def lines(b: Mapping[str, Any], *, overrun: float = 0.08, margin_usd: Optional[float] = None,
          hourly_usd: float = HOURLY_USD, a_headroom: float = 0.0) -> list[dict[str, Any]]:
    """Amendment 59's spend phases for the three tranches. Tranche A's phase holds the new disk's setup and the A
    branches (one boot, one phase: a step runs only while its phase is open); the setup part may overrun 8%, branch
    money stops at its allocation (graceful tiers). Tranche B: screening (may overrun 8%), then re-runs and branches
    with the remaining boots' money; F: the final tier. The stop point is the line less the drain margin (the drain
    grace and 5 minutes at the hourly price). `a_headroom` (a share of Tranche A's allocation) lifts its line so the
    whole draw completes: a tranche cut short leaves a labelled subset without exact inclusion probabilities."""
    margin_usd = (900 + 300) / 3600 * hourly_usd if margin_usd is None else margin_usd
    f = b["fixed_usd"]
    rows = [("phase1a", "setup (new disk: images, weights, VM re-run rows, golden check) and Tranche A branches",
             f["setup"], b["tranche_a"]["labels_usd"]),
            ("phase1b-screen", "Tranche B: exploration screening (record off, fresh simulators)"
             if "explore_screening" in f else "Tranche B: exploration episodes recorded in full",
             f.get("explore_screening", f.get("explore_full_episodes", 0.0)), 0.0),
            ("phase1b", "Tranche B: selection, record-full re-runs and branches", 0.0,
             b["tranche_b"]["labels_usd"] + f["boots"] / 2),
            ("phase1f", "final tier: b4 (one seed per A and B point) or more base points", 0.0,
             b["final"]["labels_usd"] + f["boots"] / 2)]
    out = []
    for name, what, rollout, branch in rows:
        alloc = rollout + branch
        line = rollout * (1 + overrun) + branch
        if name == "phase1a":
            line += a_headroom * alloc
        out.append({"phase": name, "what": what, "allocation_usd": alloc, "line_usd": line,
                    "stop_usd": max(0.0, line - margin_usd)})
    return out


def guard_limits(ledger_total_usd: float, b: Mapping[str, Any]) -> dict[str, float]:
    """The guard's thresholds on the ledger total, re-based on the real credit: the hard limit leaves the end reserve
    (and Phase 1's disk, storage and untracked lines, which the guard does not stop), the soft limit fires when the
    planned VM spend is used up and the reserve begins."""
    hard = ledger_total_usd + b["credit_usd"] - b["end_reserve_usd"]
    soft = hard - b["reserve_usd"]
    return {"soft_limit_usd": round(soft, 2), "hard_limit_usd": round(hard, 2)}


# ---------------------------------------------------------------- analyses (i) and (ii) (pre-stated rules)

def stratum_alloc(n: float, n_failed_frame: int, n_success_frame: int, share: float) -> tuple[float, float]:
    """Amendment 57 (2): round(share x n) from failed parents, capped by the failed points available; a shortfall in
    either stratum made up from the other."""
    nf = min(round(share * n), n_failed_frame)
    ns = min(n - nf, n_success_frame)
    nf = min(n - ns, n_failed_frame)
    return nf, ns


def effective_positives(alloc: Mapping[tuple[int, str], tuple[float, int, float]],
                        folds: Optional[Mapping[int, Sequence[int]]] = None) -> dict[int, dict[str, float]]:
    """alloc: {(task, cell): (points drawn, points in the frame, positive rate)}. Per fold, over its training tasks:
    expected positives and (Σw)²/Σw² over them, w = frame / drawn (constant within a cell)."""
    out = {}
    for t, train in (folds or pilot.folds()).items():
        sw = sw2 = e = 0.0
        for (u, _), (nd, nn, r) in alloc.items():
            if u in train and nd > 0:
                w, ep = nn / nd, nd * r
                sw, sw2, e = sw + ep * w, sw2 + ep * w * w, e + ep
        out[t] = {"effective": sw * sw / sw2 if sw2 else 0.0, "expected": e}
    return out


def summary(per_fold: Mapping[int, Mapping[str, float]]) -> dict[str, float]:
    eff = [v["effective"] for v in per_fold.values()]
    return {"min": min(eff), "mean": sum(eff) / len(eff),
            "min_expected": min(v["expected"] for v in per_fold.values())}


def share_design(frame: Mapping[tuple[int, str], int], n: float, share: float, r_f: float, r_s: float, *,
                 tasks: Sequence[int] = tuple(range(10))) -> dict[tuple[int, str], tuple[float, int, float]]:
    alloc = {}
    for t in tasks:
        nf, ns = stratum_alloc(n, frame[(t, "failure")], frame[(t, "success")], share)
        alloc[(t, "failure")] = (nf, frame[(t, "failure")], r_f)
        alloc[(t, "success")] = (ns, frame[(t, "success")], r_s)
    return alloc


def better_share(frame: Mapping[tuple[int, str], int], n: float, r_f: float, r_s: float,
                 shares: Sequence[float] = (2 / 3, 3 / 4)) -> dict[str, Any]:
    """(ii): the larger minimum over folds; within 1%, the larger mean; still within 1%, the first (2/3, approved)."""
    s = {sh: summary(effective_positives(share_design(frame, n, sh, r_f, r_s))) for sh in shares}
    a, b = shares[0], shares[1]

    def close(x, y):
        return abs(x - y) <= 0.01 * max(x, y)
    if not close(s[a]["min"], s[b]["min"]):
        pick = a if s[a]["min"] > s[b]["min"] else b
    elif not close(s[a]["mean"], s[b]["mean"]):
        pick = a if s[a]["mean"] > s[b]["mean"] else b
    else:
        pick = a
    return {"by_share": {f"{k:.4f}": v for k, v in s.items()}, "pick": pick}


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the table [[a, b], [c, d]]: the summed probability of every table with the same
    margins that is no more likely than the observed one (scipy.stats.fisher_exact's definition, its tolerance)."""
    from math import comb
    r1, c1, n = a + b, a + c, a + b + c + d
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    total = comb(n, c1)
    prob = {x: comb(r1, x) * comb(n - r1, c1 - x) / total for x in range(lo, hi + 1)}
    p_obs = prob[a] * (1 + 1e-7)
    return min(1.0, sum(v for v in prob.values() if v <= p_obs))


def stretch_of(index: int) -> str:
    for k, (lo, hi) in STRETCHES.items():
        if lo <= index <= hi:
            return k
    raise Phase1Error(f"index {index} outside 0-51")


def stretch_rule(fail_rows: Sequence[Mapping[str, Any]], frame_by_stretch: Mapping[tuple[int, str], int],
                 frame: Mapping[tuple[int, str], int], n: float, r_s: float, *, share: float = 2 / 3,
                 alpha: float = 0.05, min_gain: float = 0.10, ratio: float = 2.0) -> dict[str, Any]:
    """(i): fail_rows are the pilot's failed-parent points ({index, positive}). A stretch counts if its rate differs
    from the rest's at p < alpha/3 (two-sided Fisher exact); its split (the higher-rate side drawn at `ratio` times the
    lower side's per-point rate, within each task's failed stratum, each side at its pooled pilot rate) applies only
    if the minimum over folds of the expected effective positives rises by at least `min_gain` against uniform
    drawing at the same budget. The split with the larger gain wins."""
    tab = {k: [0, 0] for k in STRETCHES}
    for r in fail_rows:
        c = tab[stretch_of(int(r["index"]))]
        c[0] += bool(r["positive"])
        c[1] += 1
    tests, splits = {}, {}
    for k in STRETCHES:
        a, na = tab[k]
        b, nb = sum(v[0] for kk, v in tab.items() if kk != k), sum(v[1] for kk, v in tab.items() if kk != k)
        tests[k] = {"positives": a, "points": na, "rest_positives": b, "rest_points": nb,
                    "p": fisher_two_sided(a, na - a, b, nb - b)}
        if tests[k]["p"] >= alpha / len(STRETCHES) or not na or not nb:
            continue
        hi = [k] if a / na > b / nb else [kk for kk in STRETCHES if kk != k]
        lo = [kk for kk in STRETCHES if kk not in hi]

        def rate(ks):
            return sum(tab[kk][0] for kk in ks) / sum(tab[kk][1] for kk in ks)

        def design(boost):
            alloc = {}
            for t in range(10):
                nf, ns = stratum_alloc(n, frame[(t, "failure")], frame[(t, "success")], share)
                wts = {kk: frame_by_stretch[(t, kk)] * (boost if kk in hi else 1.0) for kk in STRETCHES}
                tot = sum(wts.values())
                for kk in STRETCHES:
                    size = frame_by_stretch[(t, kk)]
                    alloc[(t, f"failure/{kk}")] = (min(nf * wts[kk] / tot, size) if tot else 0.0, size,
                                                   rate(hi) if kk in hi else rate(lo))
                alloc[(t, "success")] = (ns, frame[(t, "success")], r_s)
            return effective_positives(alloc)
        u, bo = design(1.0), design(ratio)
        su, sb = summary(u), summary(bo)
        splits[k] = {"high_side": hi, "rates": {"high": rate(hi), "low": rate(lo)},
                     "gain_min": sb["min"] / su["min"] - 1, "gain_mean": sb["mean"] / su["mean"] - 1,
                     "fold_gains": sorted(bo[t]["effective"] / u[t]["effective"] - 1 for t in u)}
    passing = {k: v for k, v in splits.items() if v["gain_min"] >= min_gain}
    apply = max(passing, key=lambda k: passing[k]["gain_min"]) if passing else None
    return {"tests": tests, "splits": splits, "apply": apply,
            "rule": f"p < {alpha}/3 and a worst-fold gain >= {min_gain:.0%} at ratio {ratio:g}"}



# ---------------------------------------------------------------- Tranche A's plan and the analyses' inputs

PHASE_A = "phase1a"
FAILED_SHARE = 2 / 3                     # (ii) kept the approved share (docs/PHASE1-PLAN-2026-11-25.md)
ORDER_SEED_A = 11


def tranche_a_settings(per_task: int, seed: int) -> dict[str, Any]:
    """`pilot.make_plan`'s arguments for Tranche A: amendment 57's stratified draw of base points over the full frame
    (min_calls 1: index <= 51), uniform within each stratum ((i) did not pass its rule), independent of the pilot's
    draw; b1 and b2 only (amendment 63 (4)), 3 seeds, graceful-stop tier 1 = seeds 1-2."""
    from .branchplan import STRATIFIED
    return {"phase": PHASE_A, "enabled": ["b1_continue", "b2_one_call"], "seeds": 3, "tier_one_seeds": 2,
            "order_seed": ORDER_SEED_A,
            "selection": {"scheme": STRATIFIED, "seed": int(seed), "per_task": int(per_task),
                          "failed_share": FAILED_SHARE, "min_calls": 1}}


# ---------------------------------------------------------------- the final tier, version (a) (the user, 2026-11-26)

PHASE_F = "phase1f"
ORDER_SEED_F = 13
BASE_POINT_JOBS = 5                      # b1 x 3 seeds + b2 x 3 seeds, less the base rollout (b1's seed 1)
BRANCH_JOB_VM_S = 7.18                   # Tranche A's measured VM-seconds per branch job
STEP_OVERHEAD_H = 0.33                   # one step's boot, gate, plan and wind-down (amendment 70)
BILLED_HOURLY_USD = 0.6865               # amendment 65 (2): the Spot VM with its L4 as billed (replan.HOURLY_USD,
                                         # 0.472, predates the billing check)


def final_tier_settings(per_task_a: int, per_task_f: int, seed: int) -> dict[str, Any]:
    """`pilot.make_plan`'s arguments for the final tier's version (a): more base points, b1 and b2 with 3 seeds, A
    and F ONE stratified draw (amendment 67 (3)): Tranche A's seed and settings at per_task_a + per_task_f per task,
    A's points left out (`make_plan(earlier=...)`), so every A and F point's inclusion probability is the combined
    draw's. Failed-parent share 2/3 of the combined count, as A's."""
    if per_task_f < 1:
        raise Phase1Error(f"per_task_f is {per_task_f}: at least 1 point per task")
    s = tranche_a_settings(per_task_a + per_task_f, seed)
    return {**s, "phase": PHASE_F, "order_seed": ORDER_SEED_F}


def final_tier_points(stop_usd: float, *, tasks: int = 10, hourly_usd: float = BILLED_HOURLY_USD,
                      job_vm_s: float = BRANCH_JOB_VM_S, overhead_h: float = STEP_OVERHEAD_H) -> dict[str, Any]:
    """Base points per task the final tier buys at its phase's stop point (the user, 2026-11-26: version (a), sized
    to the money left when 740 ends): one step's overhead, then equal points per task at 5 branch jobs each."""
    point_usd = BASE_POINT_JOBS * job_vm_s / 3600 * hourly_usd
    overhead = overhead_h * hourly_usd
    n = max(0, math.floor((stop_usd - overhead) / (tasks * point_usd)))
    return {"per_task": n, "points": n * tasks, "jobs": n * tasks * BASE_POINT_JOBS, "point_usd": point_usd,
            "overhead_usd": overhead, "planned_usd": overhead + n * tasks * point_usd if n else 0.0,
            "stop_usd": stop_usd}


def run_analyses(root, plan_dir, *, n: float, cfg=None) -> dict[str, Any]:
    """(i) and (ii) from local copies of the base and pilot records and the pilot's labels (positives under
    cascader_nospread; unscored points count as not positive, as in the re-plan's pooled rates)."""
    import json
    from pathlib import Path

    from vlar_harness import clock as clk
    from vlar_harness.config import load_config
    from vlar_harness.dataset import read_table

    from .branchplan import eligible
    cfg = cfg or load_config()
    if not getattr(cfg, "frozen", False):
        raise Phase1Error("labels are scored with the frozen calibration file (set VLAR_CALIBRATION to v1.0's)")
    limits = {int(k): float(v) for k, v in cfg["time_limit"]["per_task_s"].items()}
    label_map = json.loads((Path(plan_dir) / "labels.json").read_text())
    runs, events = read_table(root, "runs").to_pylist(), read_table(root, "events").to_pylist()
    rows, problems = pilot.point_rows(runs, events, label_map, limits_s=limits, charges=clk.Charges.from_config(cfg),
                                      stop_profile="cascader_nospread")
    pts, success, _ = pilot.base_points(runs, events)
    frame: dict[tuple[int, str], int] = {}
    by_stretch: dict[tuple[int, str], int] = {}
    for p in pts:
        if eligible(p, step_limit=520, interval=10, min_calls=1):
            o = "success" if success[p.episode_id] else "failure"
            frame[(p.task_id, o)] = frame.get((p.task_id, o), 0) + 1
            if o == "failure":
                k = (p.task_id, stretch_of(p.index))
                by_stretch[k] = by_stretch.get(k, 0) + 1
    cnt = {o: [sum(r["positive"] for r in rows if r["parent_outcome"] == o),
               sum(r["parent_outcome"] == o for r in rows)] for o in ("failure", "success")}
    r_f, r_s = cnt["failure"][0] / cnt["failure"][1], cnt["success"][0] / cnt["success"][1]
    fail = [r for r in rows if r["parent_outcome"] == "failure"]
    return {"points_labelled": len(rows), "unlabelled": len(problems), "rates": {"r_f": r_f, "r_s": r_s, "counts": cnt},
            "n_per_task": n, "frame": {f"{t}/{o}": v for (t, o), v in sorted(frame.items())},
            "i": stretch_rule(fail, by_stretch, frame, n, r_s),
            "ii": better_share(frame, n, r_f, r_s)}

def final_tier_check(rates: Mapping[str, float], base_failure: Mapping[int, float], *, per_task_a: int = 64,
                     per_task_b: int = 64, k: int = 2, screen_per_task: int = 115, more_base: int = 54,
                     share: float = FAILED_SHARE, own_flagged: Sequence[int] = ()) -> dict[str, Any]:
    """Amendment 58's stop rule for the final tier's two options (amendment 65): b4 at every A and B point (b4 adds
    no points and does not enter the positive) or `more_base` more base points per task. `rates`: the pooled
    positive rates by parent outcome under cascader_nospread (amendment 63 (2)); `own_flagged`: tasks flagged by
    their own rates (amendment 63 (5), from 10 labelled failed-parent points). A and the extra base points:
    stratified, round(share x n) from failed parents. B: by episode, round(share x n/k) failed episodes of k points
    each (phase1b.select_reruns), capped by the failed episodes the screen yields at the task's base failure rate,
    E[min(need, Binomial(screen, p))]; exploration parents at base-parent rates (amendment 58). `uncapped`: B with
    every failed episode it asks for. Amendment 66 (1)."""
    r_f, r_s = float(rates["failure"]), float(rates["success"])

    def stratified(n: int) -> float:
        n_f = int(round(share * n))
        return n_f * r_f + (n - n_f) * r_s
    n_ep = -(-per_task_b // k)
    need = int(round(share * n_ep))

    def b_points(p: Optional[float]) -> float:
        if p is None:
            e_f = float(need)
        else:
            n = screen_per_task
            e_f = sum(min(need, x) * math.comb(n, x) * p ** x * (1 - p) ** (n - x) for x in range(n + 1))
        return k * e_f * r_f + (per_task_b - k * e_f) * r_s
    a, f = stratified(per_task_a), stratified(more_base)
    out: dict[str, Any] = {"rates": {"failure": r_f, "success": r_s}, "points_per_task": {
        "A": per_task_a, "B": per_task_b, "F_base": more_base}, "B_failed_episodes_needed": need,
        "screen_per_task": screen_per_task, "per_point": {"A": a, "F_base": f}}
    for basis, cap in (("capped", True), ("uncapped", False)):
        b = {t: b_points(p if cap else None) for t, p in sorted(base_failure.items())}
        for name, extra in (("b4_every_point", 0.0), ("more_base_points", f)):
            exp = {t: a + b[t] + extra for t in b}
            out.setdefault(basis, {})[name] = {"B": b, "expected_per_task": exp,
                                               **pilot.stop_rule(exp, also_flagged=own_flagged)}
    return out
