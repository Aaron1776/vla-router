"""Job plans for the resumable queue (docs/SPEC.md §7, §12).

A plan is an ordered list of idempotent jobs. Each job is keyed by (subject, kind, seed
index), where the subject is an episode id or a snapshot id and the kind is `episode` or
a canonical branch type. Order is chosen so that stopping at any point leaves a balanced
dataset: episode jobs are shuffled across tasks and initial states, and branch jobs run
in three tiers with the jobs of one decision point kept together inside a tier.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import yaml

BRANCH_ORDER = ("b1_continue", "b2_one_call", "b3_three_calls", "b4_max")
CONFIG_DIR = Path(__file__).parent / "configs"


def load_collection_config(name: str = "collection-v1") -> dict:
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def job_id(subject: str, kind: str, seed_index: int) -> str:
    return hashlib.sha256(f"{subject}|{kind}|{seed_index}".encode()).hexdigest()[:24]


def unit_uniform(*parts: Any) -> float:
    """A deterministic draw in [0, 1) from its arguments, independent of call order."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def rollout_seed(namespace: int, task_id: int, init_idx: int) -> int:
    """Rollout seed of an episode (SPEC §3). The scene seed comes from the initial state, in the harness."""
    return namespace + 100 * task_id + init_idx


@dataclass(frozen=True)
class Job:
    job_id: str
    phase: str
    kind: str                      # "episode" or a canonical branch type
    subject: str                   # episode id or snapshot id
    seed_index: int
    tier: int
    order: int
    spec: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(line: str) -> "Job":
        return Job(**json.loads(line))


@dataclass
class Plan:
    phase: str
    params: dict
    jobs: list[Job]

    @property
    def plan_id(self) -> str:
        h = hashlib.sha256(self.phase.encode())
        for j in self.jobs:
            h.update(j.to_json().encode())
        return h.hexdigest()[:16]

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps({"plan_id": self.plan_id, "phase": self.phase, "params": self.params,
                                "n_jobs": len(self.jobs)}, sort_keys=True) + "\n")
            for j in self.jobs:
                f.write(j.to_json() + "\n")

    @staticmethod
    def read(path: Path) -> "Plan":
        with open(path) as f:
            header = json.loads(f.readline())
            jobs = [Job.from_json(line) for line in f if line.strip()]
        plan = Plan(phase=header["phase"], params=header["params"], jobs=jobs)
        if plan.plan_id != header["plan_id"]:
            raise ValueError(f"{path}: plan id mismatch (file edited?)")
        return plan


# ---------------------------------------------------------------- episode plans

def episode_plan(phase: str, *, tasks: Sequence[int], inits: Sequence[int], namespace: int,
                 mode: str, profile: str, record: str, order_seed: int, scene: str = "project",
                 extra: Optional[Callable[[int, int], Mapping[str, Any]]] = None) -> Plan:
    """One job per (task, initial state), shuffled across all of them (§12).

    `scene` names the harness's `scene_seed` option: `project` everywhere except the π0.5 parity run.
    `extra(task, init)` adds per-episode fields, e.g. which decider drives it in DAgger.
    """
    pairs = [(t, i) for t in tasks for i in inits]
    random.Random(f"{phase}|{order_seed}").shuffle(pairs)
    jobs = []
    for order, (t, i) in enumerate(pairs):
        subject = f"{phase}/t{t}/i{i}"
        spec = {"task_id": t, "init_idx": i, "rollout_seed": rollout_seed(namespace, t, i), "scene": scene,
                "mode": mode, "profile": profile, "record": record, **(extra(t, i) if extra else {})}
        jobs.append(Job(job_id(subject, "episode", 1), phase, "episode", subject, 1, 0, order, spec))
    params = {"tasks": list(tasks), "inits": list(inits), "namespace": namespace, "mode": mode,
              "profile": profile, "record": record, "order_seed": order_seed, "scene": scene}
    return Plan(phase, params, jobs)


# ---------------------------------------------------------------- branch plans

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


def branch_plan(phase: str, points: Iterable[DecisionPointRef], *, branches: Mapping[str, Mapping[str, Any]],
                step_limit: int, interval: int, seeds: int, enabled: Sequence[str],
                tier_one_seeds: int, order_seed: int) -> tuple[Plan, dict[str, dict]]:
    """Branch jobs for the given (already sampled) decision points.

    Returns the plan and a label map: for every point, each enabled branch's call count, the
    canonical branch whose jobs it shares, and which seeds the parent rollout provides
    (a base rollout is seed 1 of b1_continue at each of its points, §7).
    """
    pts = [p for p in points if p.budget_remaining_steps >= interval]
    rng = random.Random(f"{phase}|{order_seed}")
    rng.shuffle(pts)
    per_tier: dict[int, list[Job]] = {1: [], 2: [], 3: []}
    labels: dict[str, dict] = {}
    for p in pts:
        calls = {b: n for b, n in branch_calls(p, branches, step_limit, interval).items() if b in enabled}
        share = shared_branches(calls)
        parent_seeds = {"b1_continue": [1]} if p.source == "base" else {}
        labels[p.snapshot_id] = {"point": asdict(p), "calls": calls, "shares": share, "parent_seeds": parent_seeds}
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
                    "task_id": p.task_id, "init_idx": p.init_idx, "source": p.source}
            per_tier[t].append(Job(job_id(p.snapshot_id, canon, s), phase, canon, p.snapshot_id, s, t, 0, spec))
    jobs = []
    for t in (1, 2, 3):
        for j in per_tier[t]:
            jobs.append(Job(j.job_id, j.phase, j.kind, j.subject, j.seed_index, j.tier, len(jobs), j.spec))
    params = {"seeds": seeds, "enabled": list(enabled), "tier_one_seeds": tier_one_seeds, "order_seed": order_seed,
              "step_limit": step_limit, "interval": interval, "n_points": len(pts)}
    return Plan(phase, params, jobs), labels


def complete_points(labels: Mapping[str, dict], done_job_ids: set[str], *,
                    required: Sequence[str], seeds: int) -> list[str]:
    """Snapshot ids whose required branches have all their seeds finished (§12: training and
    evaluation use only complete points)."""
    out = []
    for snap, lab in labels.items():
        ok = True
        for b in required:
            canon = lab["shares"].get(b)
            if canon is None:
                ok = False
                break
            parent = set(lab["parent_seeds"].get(canon, []))
            if any(s not in parent and job_id(snap, canon, s) not in done_job_ids for s in range(1, seeds + 1)):
                ok = False
                break
        if ok:
            out.append(snap)
    return out
