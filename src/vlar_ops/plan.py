"""Job plans for the resumable queue (docs/SPEC.md §7, §12).

A plan is an ordered list of idempotent jobs. Each job is keyed by (subject, kind, seed
index), where the subject is an episode id or a snapshot id and the kind is `episode` or
a canonical branch type. Order is chosen so that stopping at any point leaves a balanced
dataset: episode jobs are shuffled across tasks and initial states, and branch jobs run
in three tiers with the jobs of one decision point kept together inside a tier (`vlar_ops.branchplan`,
the branch kind's generator, amendment 49). `complete_points` is consumption code, in no data digest.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import yaml

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


# ---------------------------------------------------------------- consumption

def complete_points(labels: Mapping[str, dict], done_job_ids: set[str], *,
                    required: Optional[Sequence[str]] = None) -> list[str]:
    """Snapshot ids whose required branches have all their seeds finished (§12: training and
    evaluation use only complete points). The seed count and, unless `required` is given, the
    branches come from each entry's recorded planner settings (`vlar_ops.branchplan`, amendment 49),
    never from the collection config."""
    out = []
    for snap, lab in labels.items():
        settings = lab.get("settings")
        if not settings:
            raise ValueError(f"label entry {snap} records no planner settings (amendment 49)")
        seeds = int(settings["seeds"])
        ok = True
        for b in (settings["enabled"] if required is None else required):
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
