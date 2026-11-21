"""Branch labels (docs/SPEC.md §7). For a decider with profile P, the label of branch b at decision point d is the
fraction of b's seeds that succeeded before the step limit and within the task's time limit, with time computed
under P from the parent rollout's event log up to d plus the branch's own log. One set of branch rollouts thus
gives labels for every profile and any time limits; labels are computed here, offline, never stored as facts.

Inputs for one decision point:
- `prefix`: the snapshot's event log, the parent's log up to and including d's calls (it ends with d's
  `decision_point` event, whose overhead it already contains);
- the planner's label entry for the point (`vlar_ops.plan.branch_plan`): each branch's canonical job (branches
  with equal π0.5 call counts share one), the call counts, and which seeds the parent rollout itself provides (a
  base rollout is seed 1 of `b1_continue`);
- the branch jobs' own logs, keyed by (canonical branch, seed index), each starting with its `restore` and
  `branch` events; and, where the parent provides a seed, the parent's full log.
Everything is checked to fit together (same point, same branch, seed and call count; the parent's log starts with
the prefix) before anything is scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from . import clock as clk
from . import eventlog
from .config import PROFILES


class LabelError(ValueError):
    pass


@dataclass(frozen=True)
class Label:
    value: float          # successes / seeds
    successes: int
    seeds: int


def check_prefix(prefix: Sequence[Mapping]) -> dict:
    """The snapshot's log: unfinished, ending with its decision point. Returns that decision_point event."""
    eventlog.check(prefix, complete=False)
    last = prefix[-1]
    if last["t"] != "decision_point":
        raise LabelError(f"a snapshot's log ends with its decision point, not {last['t']!r}")
    return dict(last)


def check_branch_log(own: Sequence[Mapping], point: Mapping, *, branch: str, seed_index: int, pi05_calls: int) -> None:
    if len(own) < 2 or own[0]["t"] != "restore" or own[1]["t"] != "branch":
        raise LabelError("a branch's own log starts with its restore and branch events")
    r, b = own[0], own[1]
    if (r["index"], r["step"]) != (point["index"], point["step"]):
        raise LabelError(f"branch restored at point {r['index']} (step {r['step']}), not {point['index']}")
    if (b["branch"], b["seed_index"], b["pi05_calls"]) != (branch, seed_index, pi05_calls):
        raise LabelError(f"expected {branch} seed {seed_index} with {pi05_calls} calls, got "
                         f"{b['branch']} seed {b['seed_index']} with {b['pi05_calls']}")
    ends = [i for i, e in enumerate(own) if e["t"] == "end"]
    if ends != [len(own) - 1]:
        raise LabelError("a branch's log ends with exactly one end event")


def check_parent_log(parent: Sequence[Mapping], prefix: Sequence[Mapping]) -> None:
    if list(parent[:len(prefix)]) != list(prefix):
        raise LabelError("the parent's log does not start with the snapshot's log")
    eventlog.check(parent)


def within(events: Sequence[Mapping], profile: str, charges: clk.Charges, limit_s: float) -> bool:
    """Success before the step limit with the clock under `profile` at most `limit_s` at the success step."""
    return clk.score(events, profile, charges).within(limit_s)


def label(full_logs: Sequence[Sequence[Mapping]], profile: str, charges: clk.Charges, limit_s: float) -> Label:
    if not full_logs:
        raise LabelError("a label needs at least one seed")
    k = sum(within(log, profile, charges, limit_s) for log in full_logs)
    return Label(k / len(full_logs), k, len(full_logs))


def point_labels(prefix: Sequence[Mapping], entry: Mapping, own_logs: Mapping[tuple[str, int], Sequence[Mapping]],
                 *, seeds: int, charges: clk.Charges, limit_s: float,
                 parent_log: Optional[Sequence[Mapping]] = None,
                 profiles: Sequence[str] = PROFILES) -> dict[str, dict[str, Label]]:
    """{branch: {profile: Label}} for every branch in the planner's entry for one decision point."""
    point = check_prefix(prefix)
    p = entry["point"]
    if (p["index"], p["budget_remaining_steps"]) != (point["index"], point["budget_remaining"]):
        raise LabelError(f"planner entry is for point {p['index']} with {p['budget_remaining_steps']} budget steps; "
                         f"the snapshot is point {point['index']} with {point['budget_remaining']}")
    if parent_log is not None:
        check_parent_log(parent_log, prefix)
    cache: dict[tuple[str, int], list] = {}
    out: dict[str, dict[str, Label]] = {}
    for b, canon in entry["shares"].items():
        logs = []
        for s in range(1, seeds + 1):
            if s in entry["parent_seeds"].get(canon, []):
                if parent_log is None:
                    raise LabelError(f"{b} seed {s} is the parent rollout's own continuation; pass parent_log")
                logs.append(parent_log)
                continue
            key = (canon, s)
            if key not in cache:
                own = own_logs.get(key)
                if own is None:
                    raise LabelError(f"missing branch log {canon} seed {s}")
                check_branch_log(own, point, branch=canon, seed_index=s, pi05_calls=int(entry["calls"][canon]))
                cache[key] = list(prefix) + list(own)
            logs.append(cache[key])
        out[b] = {prof: label(logs, prof, charges, limit_s) for prof in profiles}
    return out
