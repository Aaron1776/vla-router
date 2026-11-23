"""Branch plans (docs/SPEC.md §7, §12; amendment 49). The branch kind's job-spec generator: all of this module is in
the branch data digest (`vlar_ops.datadigest`) and nothing of the base or exploration generators imports it, so the
branch-plan command belongs here too (a `cli.py` command only forwards to it).

The planner records its settings (`settings`: seeds, enabled branches, tier-one seeds, order seed and the
decision-point sampling's fraction and seed: the collection tunables, `configs/collection-v1-tunables.yaml`) in every
job spec and every label entry, and applies the sampling itself, so what is recorded is what was used. Label and
completeness code read the seeds and enabled branches from the entries, never from the config.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

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


def _settings(seeds: int, enabled: Sequence[str], tier_one_seeds: int, order_seed: int,
              dp_sampling: Mapping[str, Any]) -> dict[str, Any]:
    if not set(enabled) <= set(BRANCH_ORDER) or not enabled:
        raise BranchPlanError(f"enabled branches {list(enabled)} must be a non-empty subset of {BRANCH_ORDER}")
    if int(seeds) < 1:
        raise BranchPlanError(f"seeds is {seeds}: at least 1")
    fraction = dp_sampling.get("fraction")
    if fraction is None or not 0 < float(fraction) <= 1:
        raise BranchPlanError(f"dp_sampling.fraction is {fraction!r}: set it in (0, 1] (§12, from the pilot)")
    return {"seeds": int(seeds), "enabled": [b for b in BRANCH_ORDER if b in enabled],
            "tier_one_seeds": int(tier_one_seeds), "order_seed": int(order_seed),
            "dp_sampling": {"fraction": float(fraction), "seed": int(dp_sampling["seed"])}}


def branch_plan(phase: str, points: Iterable[DecisionPointRef], *, branches: Mapping[str, Mapping[str, Any]],
                step_limit: int, interval: int, seeds: int, enabled: Sequence[str],
                tier_one_seeds: int, order_seed: int, dp_sampling: Mapping[str, Any]) -> tuple[Plan, dict[str, dict]]:
    """Branch jobs for the candidate decision points that the sampling keeps (`dp_is_sampled`).

    Returns the plan and a label map: for every point, each enabled branch's call count, the
    canonical branch whose jobs it shares, which seeds the parent rollout provides
    (a base rollout is seed 1 of b1_continue at each of its points, §7) and the planner's settings,
    which every job spec records too.
    """
    settings = _settings(seeds, enabled, tier_one_seeds, order_seed, dp_sampling)
    seeds, enabled = settings["seeds"], settings["enabled"]
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
