"""Branch plans (docs/SPEC.md §7, §12; amendment 49). The branch kind's job-spec generator: all of this module is in
the branch data digest (`vlar_ops.datadigest`) and nothing of the base or exploration generators imports it, so the
branch-plan command belongs here too (a `cli.py` command only forwards to it).

The planner records its settings (`settings`: seeds, enabled branches, tier-one seeds, order seed and the
decision-point sampling's fraction and seed: the collection tunables, `configs/collection-v1-tunables.yaml`) in every
job spec and every label entry, and applies the sampling itself, so what is recorded is what was used. Label and
completeness code read the seeds and enabled branches from the entries, never from the config.

Outcome-stratified sampling (amendment 57; PROPOSALS-2026-11-24 §0.5): `stratified_selection` draws a task's points
uniformly within each stratum (parent kind, task, parent outcome), a share of them from failed parents, and
`branch_plan(..., selection=...)` plans exactly those points, recording in every label entry its stratum, the
stratum's size and draw, and its inclusion probability (training weight 1/π). The settings then name the scheme
instead of a fraction. Base parents are drawn by point; parents that are re-run (exploration, DAgger router half)
are drawn by episode and then k points, not implemented yet.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from .plan import Job, Plan, job_id, unit_uniform

BRANCH_ORDER = ("b1_continue", "b2_one_call", "b3_three_calls", "b4_max")


class BranchPlanError(ValueError):
    pass


@dataclass(frozen=True)
class DecisionPointRef:
    snapshot_id: str
    episode_id: str
    source: str                    # base | explore | dagger
    task_id: int
    init_idx: int
    index: int                     # decision point index within the episode, 0..51
    budget_remaining_steps: int


def dp_is_sampled(sampling_seed: int, episode_id: str, index: int, fraction: float) -> bool:
    """Per-point draw, so the choice can be made before or after the rollout with the same result.

    Each point is kept with probability `fraction`, so the kept share of an episode varies
    around the fraction (binomially) rather than being exact.
    """
    return unit_uniform("dp-sample", sampling_seed, episode_id, index) < fraction


def branch_calls(dp: DecisionPointRef, branches: Mapping[str, Mapping[str, Any]],
                 step_limit: int, interval: int) -> dict[str, int]:
    """pi0.5 calls each branch makes from this point, truncated by budget and episode end."""
    remaining_points = step_limit // interval - dp.index
    cap = min(dp.budget_remaining_steps // interval, remaining_points)
    out = {}
    for name in BRANCH_ORDER:
        c = branches[name]["calls"]
        out[name] = cap if c == "max" else min(int(c), cap)
    return out


def shared_branches(calls: Mapping[str, int]) -> dict[str, str]:
    """Map each branch to the branch whose job it shares: branches with equal call counts are
    run once (§7), under the lowest-numbered branch's name."""
    first: dict[int, str] = {}
    out = {}
    for name in BRANCH_ORDER:
        if name in calls:
            out[name] = first.setdefault(calls[name], name)
    return out


def _label_tier(branch: str, seed_index: int, tier_one_seeds: int) -> int:
    if branch == "b3_three_calls":
        return 3
    return 1 if seed_index <= tier_one_seeds else 2


def plan_settings(tunables: Mapping[str, Any]) -> dict[str, Any]:
    """The planner's settings from the collection tunables (`vlar_ops.tunables.load_tunables()`)."""
    b, dp = tunables["branches"], tunables["dp_sampling"]
    return {"seeds": b["seeds"], "enabled": b["enabled"], "tier_one_seeds": b["tier_one_seeds"],
            "order_seed": tunables["order_seed"], "dp_sampling": {"fraction": dp["fraction"], "seed": dp["seed"]}}


STRATIFIED = "stratified"


def _stratified_settings(dp_sampling: Mapping[str, Any]) -> dict[str, Any]:
    out = {"scheme": STRATIFIED, "seed": int(dp_sampling["seed"]), "per_task": int(dp_sampling["per_task"]),
           "failed_share": float(dp_sampling["failed_share"]), "min_calls": int(dp_sampling.get("min_calls", 1))}
    if out["per_task"] < 1 or not 0 <= out["failed_share"] <= 1 or out["min_calls"] < 1:
        raise BranchPlanError(f"stratified sampling settings {dict(dp_sampling)}: per_task >= 1, failed_share in "
                              f"[0, 1], min_calls >= 1")
    return out


def eligible(p: DecisionPointRef, *, step_limit: int, interval: int, min_calls: int = 1) -> bool:
    """At least `min_calls` π0.5 calls fit (budget and remaining decision points): 1 for any branch point (§7), 3
    for the pilot, so that b3 runs its three calls."""
    return min(p.budget_remaining_steps // interval, step_limit // interval - p.index) >= min_calls


def stratified_selection(points: Iterable[DecisionPointRef], parent_success: Mapping[str, bool], *,
                         step_limit: int, interval: int, dp_sampling: Mapping[str, Any]) -> dict[str, dict]:
    """Per task, `per_task` points: round(failed_share x per_task) from failed parents and the rest from successful
    ones, each stratum short of points made up from the other. Within a stratum the draw is uniform without
    replacement (points ordered by a seeded hash of (episode, index), so the draw does not depend on input order),
    hence inclusion probability n/N for every point of the stratum. Base parents only (drawn by point).

    Returns {snapshot_id: selection record}, the record going into the point's label entry."""
    st = _stratified_settings(dp_sampling)
    strata: dict[tuple[str, int, str], list[DecisionPointRef]] = {}
    for p in points:
        if p.source != "base":
            raise BranchPlanError(f"{p.snapshot_id}: only base parents are drawn by point (re-run parents: by episode)")
        if p.episode_id not in parent_success:
            raise BranchPlanError(f"{p.snapshot_id}: parent {p.episode_id} has no recorded outcome")
        if not eligible(p, step_limit=step_limit, interval=interval, min_calls=st["min_calls"]):
            continue
        outcome = "success" if parent_success[p.episode_id] else "failure"
        strata.setdefault((p.source, p.task_id, outcome), []).append(p)
    out: dict[str, dict] = {}
    for task in sorted({k[1] for k in strata}):
        fail = strata.get(("base", task, "failure"), [])
        succ = strata.get(("base", task, "success"), [])
        n = st["per_task"]
        n_f = min(int(round(st["failed_share"] * n)), len(fail))
        n_s = min(n - n_f, len(succ))
        n_f = min(n - n_s, len(fail))
        for outcome, pool, k in (("failure", fail, n_f), ("success", succ, n_s)):
            order = sorted(pool, key=lambda q: (unit_uniform("dp-stratified", st["seed"], q.episode_id, q.index),
                                                q.snapshot_id))
            for q in order[:k]:
                out[q.snapshot_id] = {"stratum": {"source": "base", "task_id": task, "parent_outcome": outcome},
                                      "stratum_size": len(pool), "stratum_drawn": k,
                                      "inclusion_prob": k / len(pool), "by": "point"}
    return out


def _settings(seeds: int, enabled: Sequence[str], tier_one_seeds: int, order_seed: int,
              dp_sampling: Mapping[str, Any]) -> dict[str, Any]:
    if not set(enabled) <= set(BRANCH_ORDER) or not enabled:
        raise BranchPlanError(f"enabled branches {list(enabled)} must be a non-empty subset of {BRANCH_ORDER}")
    if int(seeds) < 1:
        raise BranchPlanError(f"seeds is {seeds}: at least 1")
    if dp_sampling.get("scheme") == STRATIFIED:
        return {"seeds": int(seeds), "enabled": [b for b in BRANCH_ORDER if b in enabled],
                "tier_one_seeds": int(tier_one_seeds), "order_seed": int(order_seed),
                "dp_sampling": _stratified_settings(dp_sampling)}
    fraction = dp_sampling.get("fraction")
    if fraction is None or not 0 < float(fraction) <= 1:
        raise BranchPlanError(f"dp_sampling.fraction is {fraction!r}: set it in (0, 1] (§12, from the pilot)")
    return {"seeds": int(seeds), "enabled": [b for b in BRANCH_ORDER if b in enabled],
            "tier_one_seeds": int(tier_one_seeds), "order_seed": int(order_seed),
            "dp_sampling": {"fraction": float(fraction), "seed": int(dp_sampling["seed"])}}


def branch_plan(phase: str, points: Iterable[DecisionPointRef], *, branches: Mapping[str, Mapping[str, Any]],
                step_limit: int, interval: int, seeds: int, enabled: Sequence[str],
                tier_one_seeds: int, order_seed: int, dp_sampling: Mapping[str, Any],
                selection: Optional[Mapping[str, Mapping[str, Any]]] = None) -> tuple[Plan, dict[str, dict]]:
    """Branch jobs for the candidate decision points that the sampling keeps (`dp_is_sampled`).

    Returns the plan and a label map: for every point, each enabled branch's call count, the
    canonical branch whose jobs it shares, which seeds the parent rollout provides
    (a base rollout is seed 1 of b1_continue at each of its points, §7) and the planner's settings,
    which every job spec records too.
    """
    settings = _settings(seeds, enabled, tier_one_seeds, order_seed, dp_sampling)
    seeds, enabled = settings["seeds"], settings["enabled"]
    points = list(points)
    if settings["dp_sampling"].get("scheme") == STRATIFIED:
        if selection is None:
            raise BranchPlanError("stratified sampling plans the points of a `stratified_selection`; none given")
        pts = [p for p in points if p.snapshot_id in selection]
        missing = set(selection) - {p.snapshot_id for p in pts}
        if missing:
            raise BranchPlanError(f"{len(missing)} selected points are not among the candidates, e.g. {min(missing)}")
    else:
        if selection is not None:
            raise BranchPlanError("a selection needs dp_sampling scheme 'stratified'")
        frac, dp_seed = settings["dp_sampling"]["fraction"], settings["dp_sampling"]["seed"]
        pts = [p for p in points if p.budget_remaining_steps >= interval
               and dp_is_sampled(dp_seed, p.episode_id, p.index, frac)]
    rng = random.Random(f"{phase}|{order_seed}")
    rng.shuffle(pts)
    per_tier: dict[int, list[Job]] = {1: [], 2: [], 3: []}
    labels: dict[str, dict] = {}
    for p in pts:
        calls = {b: n for b, n in branch_calls(p, branches, step_limit, interval).items() if b in enabled}
        share = shared_branches(calls)
        parent_seeds = {"b1_continue": [1]} if p.source == "base" else {}
        labels[p.snapshot_id] = {"point": asdict(p), "calls": calls, "shares": share, "parent_seeds": parent_seeds,
                                 "settings": settings}
        if selection is not None:
            labels[p.snapshot_id]["selection"] = dict(selection[p.snapshot_id])
        # Tier of each canonical job = earliest tier of any branch label it serves.
        tiers: dict[tuple[str, int], int] = {}
        for b, canon in share.items():
            for s in range(1, seeds + 1):
                if s in parent_seeds.get(canon, []):
                    continue
                t = _label_tier(b, s, tier_one_seeds)
                key = (canon, s)
                tiers[key] = min(t, tiers.get(key, t))
        for (canon, s), t in sorted(tiers.items(), key=lambda kv: (BRANCH_ORDER.index(kv[0][0]), kv[0][1])):
            spec = {"snapshot_id": p.snapshot_id, "branch": canon, "pi05_calls": calls[canon], "seed_index": s,
                    "task_id": p.task_id, "init_idx": p.init_idx, "source": p.source, "settings": settings}
            per_tier[t].append(Job(job_id(p.snapshot_id, canon, s), phase, canon, p.snapshot_id, s, t, 0, spec))
    jobs = []
    for t in (1, 2, 3):
        for j in per_tier[t]:
            jobs.append(Job(j.job_id, j.phase, j.kind, j.subject, j.seed_index, j.tier, len(jobs), j.spec))
    params = {**settings, "step_limit": step_limit, "interval": interval, "n_points": len(pts)}
    return Plan(phase, params, jobs), labels
