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

Profiles and time (amendments 21-23, 2026-11-21):
- The prefix is re-scored under every profile, also when the parent ran with a live time limit under another
  profile (a DAgger rollout): it ends at a decision point the parent reached before any cut, and every branch runs
  uncut. A parent's full log used as a seed is a run's own outcome, so a cut parent's is refused under any
  profile but its own (clock.score's rule).
- A point whose clock under P, replayed from the prefix, already exceeds the limit is one no live decider of
  type P can reach: it is left out of P's training and validation data (`PointLabels.excluded`), and
  `exclusion_report` counts the exclusions per profile. "Exceeds" is strict, as for the live cut.
- The decider's remaining-time input at a training point is recomputed from the prefix under that decider's own
  profile (`remaining_time_s`, `PointLabels.remaining_s`), never copied from what the collecting decider saw
  (its own profile's time, or infinity in a scripted run).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

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


@dataclass(frozen=True)
class PointLabels:
    """One decision point's labels and per-profile time inputs (§6, §7)."""
    labels: dict[str, dict[str, Label]]      # {branch: {profile: Label}}, for the profiles not excluded
    excluded: tuple[str, ...]                # profiles whose clock at the point already exceeds the limit
    time_s: dict[str, float]                 # clock at the point under each profile, from the parent's log
    remaining_s: dict[str, float]            # limit - time_s: each decider type's remaining-time input


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
    if b.get("time_limit") is not False:
        raise LabelError("a branch runs uncut: its branch event records time_limit false (amendment 20)")
    ends = [i for i, e in enumerate(own) if e["t"] == "end"]
    if ends != [len(own) - 1]:
        raise LabelError("a branch's log ends with exactly one end event")


def check_parent_log(parent: Sequence[Mapping], prefix: Sequence[Mapping]) -> None:
    if list(parent[:len(prefix)]) != list(prefix):
        raise LabelError("the parent's log does not start with the snapshot's log")
    eventlog.check(parent)


def time_at_point(prefix: Sequence[Mapping], profile: str, charges: clk.Charges) -> float:
    """The clock at the snapshot's decision point under `profile`, replayed from the snapshot's log. Valid under
    any profile, also when the parent ran with a live limit under another one (amendment 21)."""
    check_prefix(prefix)
    return clk.replay(prefix, profile, charges)


def remaining_time_s(prefix: Sequence[Mapping], profile: str, charges: clk.Charges, limit_s: float) -> float:
    """The remaining-time input of a decider with profile `profile` at this training point: the task's limit minus
    the clock under that profile, recomputed from the event log; never the value the collecting decider saw
    (amendment 23)."""
    return limit_s - time_at_point(prefix, profile, charges)


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
                 profiles: Sequence[str] = PROFILES) -> PointLabels:
    """Labels for every branch in the planner's entry for one decision point, under every profile whose clock at
    the point is within the limit; the others are listed in `excluded` (amendment 22)."""
    point = check_prefix(prefix)
    p = entry["point"]
    if (p["index"], p["budget_remaining_steps"]) != (point["index"], point["budget_remaining"]):
        raise LabelError(f"planner entry is for point {p['index']} with {p['budget_remaining_steps']} budget steps; "
                         f"the snapshot is point {point['index']} with {point['budget_remaining']}")
    time_s = {prof: clk.replay(prefix, prof, charges) for prof in profiles}
    excluded = tuple(prof for prof in profiles if time_s[prof] > limit_s)
    kept = [prof for prof in profiles if prof not in excluded]
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
                cut = clk.cut_run_profile(parent_log)
                wrong = [prof for prof in kept if cut is not None and prof != cut]
                if wrong:
                    raise LabelError(f"{b} seed {s} is the full log of a run cut live under {cut!r}: as a seed it can "
                                     f"be scored only under {cut!r}, not {wrong} (amendment 21)")
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
        out[b] = {prof: label(logs, prof, charges, limit_s) for prof in kept}
    return PointLabels(out, excluded, time_s, {prof: limit_s - t for prof, t in time_s.items()})


def exclusion_report(points: Iterable[PointLabels]) -> dict[str, dict[str, int]]:
    """Per profile: decision points labelled, how many were excluded because the clock under that profile had
    already passed the time limit at the point (amendment 22), and how many were kept."""
    out: dict[str, dict[str, int]] = {}
    for pl in points:
        for prof in pl.time_s:
            row = out.setdefault(prof, {"points": 0, "excluded": 0, "kept": 0})
            row["points"] += 1
            row["excluded" if prof in pl.excluded else "kept"] += 1
    return out
