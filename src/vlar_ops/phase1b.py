"""Phase 1, Tranche B (SPEC §17 amendments 56, 57, 62 (2), 65): exploration screened, re-run, then branched.

1. **Screen** (`screen_plan`, on the Mac; step 730, spend phase `phase1b-screen`). ~1,150 exploration episodes
   (`phase1.budget`'s `explore_screening` count), record `off`, profile `none`, the scripted escalations at
   p_escalate (recorded in every spec, amendment 49). Every initial state is screened `per_task // 50` times and the
   rest of a task's count is a seeded draw of its initial states for one more repeat. Repeat r of (task, init) has
   rollout seed `plan.rollout_seed(explore namespace + 1000 r, task, init)` (unchanged seed derivation; the offsets
   stay far inside the explore namespace, which spans 1,000,000) and episode id `explore/t<task>/i<init>/r<r>`. Rule
   (A): every screened episode starts on a fresh simulator, `setup_count` 2 (`vlar_ops.setupcount`).
2. **Select** (`select_reruns`, on the VM; step 740, spend phase `phase1b`). By episode (amendment 57 (2)): per task,
   n/k episodes, round(s x n/k) from failed screened episodes (s = 2/3), capped by those available, a shortfall in
   either stratum made up from the other; uniform within a stratum in the order of a hash of (seed, episode). The
   outcome is the screened run's (success at 520 steps, no time cutoff). Each chosen episode is re-run: the same
   spec with record `full`, seed index 2 (so its job and its record files differ from the screen's; its run id, the
   episode id, is the same), at the screen's measured setup count.
3. **Compare** (`compare_rerun`, on the VM while re-runs run and again after them). Every re-run against its screened
   run bit for bit: runs, steps and events, only the columns and event fields that name the record mode aside, and
   the marker summaries' outcome. Any difference stops all re-runs (the spend guard's stop flag; max-effort
   checkpoint), and no re-run is used for branches unless every comparison passed.
4. **Points and branches** (`rerun_points`, `select_points`, `branch_plan_b`). In each re-run episode k eligible
   decision points (`branchplan.eligible`, min_calls 1), uniform in the order of a hash of (seed, episode, index),
   from those whose snapshot the re-run's marker lists; inclusion probability P(episode) x P(point | episode) =
   (drawn / stratum size) x (points drawn / eligible points). b1 and b2, 3 seeds each (none from the parent: an
   exploration rollout is not b1's continuation), graceful-stop tier 1 = seeds 1-2, phase `phase1b`. The branch plan
   is `branchplan.branch_plan` unchanged (its file is the branch kind's data digest; the episode selection is here).

Records: a screened run and its re-run share a run id under phase `explore`; consumers take the record-`full` one
(the run row's `record` column) wherever both exist.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .branchplan import STRATIFIED, DecisionPointRef, branch_plan, eligible
from .plan import Job, Plan, job_id, rollout_seed, unit_uniform

SCREEN_PHASE = "explore"           # Runner._choice: the scripted exploration policy runs only under phase "explore"
BRANCH_PHASE = "phase1b"
INITS = 50
REPEAT_OFFSET = 1000
FRESH = 2
SCREEN_SEED_INDEX, RERUN_SEED_INDEX = 1, 2
ENABLED = ("b1_continue", "b2_one_call")
SEEDS, TIER_ONE_SEEDS = 3, 2
ORDER_SEED_B = 65_002
FAILED_SHARE = 2 / 3
RECORD_SKIP = ("record",)    # the record mode only: the screen and its re-run both run under the frozen calibration,
                             # so calibration_hash must match (unlike pilot.EPISODE_SKIP["record_off"])


class TrancheBError(RuntimeError):
    pass


# ---------------------------------------------------------------- 1. the screen

def screen_episodes(per_task: "int | Mapping[int, int]", seed: int, *, tasks: Sequence[int] = tuple(range(10)),
                    inits: int = INITS) -> list[tuple[int, int, int]]:
    """(task, init, repeat) for `per_task` screened episodes per task (one number, or one per task: amendment 67):
    every initial state `n // inits` times, then the remainder drawn without replacement by a hash of (seed, task,
    init). A larger n extends the sequence: the episodes of a smaller n keep their (task, init, repeat)."""
    size = {t: int(per_task[t] if isinstance(per_task, Mapping) else per_task) for t in tasks}
    for t, n in size.items():
        if n < inits:
            raise TrancheBError(f"task {t}: {n} screened episodes, fewer than one per initial state ({inits})")
    out = [(t, i, r) for t in tasks for r in range(size[t] // inits) for i in range(inits)]
    for t in tasks:
        full, rest = divmod(size[t], inits)
        extra = sorted(range(inits), key=lambda i: (unit_uniform("screen-extra", seed, t, i), i))[:rest]
        out += [(t, i, full) for i in sorted(extra)]
    return out


def front_loaded_sizes(base_failure: Mapping[int, float], *, per_task: int = 115, need: int = 21,
                       margin: float = 1.1) -> dict[int, int]:
    """Amendment 67: each task's screen with amendment 66 (4)'s top-up front-loaded at the task's base failure rate
    p: per_task + ceil(margin x (need - per_task x p) / p) where per_task x p falls short of need, else per_task."""
    out = {}
    for t, p in sorted(base_failure.items()):
        if p <= 0:
            raise TrancheBError(f"task {t}: base failure rate {p}: no top-up size")
        short = need - per_task * p
        out[int(t)] = per_task + (math.ceil(margin * short / p - 1e-9) if short > 0 else 0)
    return out


def base_failure_rates(csv_path) -> dict[int, float]:
    """Per task, the share of failed base rollouts in a base-runs CSV (task_id, success)."""
    import csv
    n: dict[int, list[int]] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            c = n.setdefault(int(row["task_id"]), [0, 0])
            c[0] += 1
            c[1] += row["success"].strip().lower() not in ("true", "1")
    return {t: f / k for t, (k, f) in sorted(n.items())}


def screen_plan(per_task: "int | Mapping[int, int]", seed: int, *, namespace: int, p_escalate: float, profile: str = "none",
                order_seed: int = ORDER_SEED_B, tasks: Sequence[int] = tuple(range(10))) -> Plan:
    jobs = []
    eps = screen_episodes(per_task, seed, tasks=tasks)
    random.Random(f"{SCREEN_PHASE}|phase1b|{order_seed}").shuffle(eps)
    for order, (t, i, r) in enumerate(eps):
        subject = f"{SCREEN_PHASE}/t{t}/i{i}/r{r}"
        spec = {"task_id": t, "init_idx": i, "rollout_seed": rollout_seed(namespace + REPEAT_OFFSET * r, t, i),
                "scene": "project", "mode": "routed", "profile": profile, "record": "off",
                "p_escalate": float(p_escalate), "episodes_per_state": 1, "screen_repeat": r,
                "setup_count": FRESH, "tranche": BRANCH_PHASE}
        jobs.append(Job(job_id(subject, "episode", SCREEN_SEED_INDEX), SCREEN_PHASE, "episode", subject,
                        SCREEN_SEED_INDEX, 0, order, spec))
    sizes = ({str(t): int(per_task[t]) for t in tasks} if isinstance(per_task, Mapping) else per_task)
    params = {"tranche": BRANCH_PHASE, "per_task": sizes, "seed": seed, "namespace": namespace,
              "repeat_offset": REPEAT_OFFSET, "p_escalate": float(p_escalate), "profile": profile,
              "order_seed": order_seed, "tasks": list(tasks), "setup_count": FRESH}
    return Plan(SCREEN_PHASE, params, jobs)


# ---------------------------------------------------------------- 2. selection by episode

def screened(plan: Plan, marker: Callable[[str], Optional[Mapping]]) -> list[dict]:
    """Every screened job with its result marker's outcome and setup count; raises if any is missing or was not set
    up at the count its spec records."""
    out, missing = [], []
    for j in plan.jobs:
        m = marker(j.job_id)
        if m is None:
            missing.append(j.subject)
            continue
        meta, summary = m.get("meta", {}), m["result"]["summary"]
        if meta.get("setup_count") != j.spec["setup_count"]:
            raise TrancheBError(f"{j.subject}: marker setup_count {meta.get('setup_count')!r}, spec "
                                f"{j.spec['setup_count']}")
        out.append({"episode_id": j.subject, "task_id": int(j.spec["task_id"]), "success": bool(summary["success"]),
                    "steps": int(summary["steps"]), "setup_count": int(meta["setup_count"]), "job": j})
    if missing:
        raise TrancheBError(f"{len(missing)} screened episodes have no result marker, e.g. {missing[0]}")
    return out


def screen_yield(rows: Iterable[Mapping], *, per_task: int, k: int, failed_share: float = FAILED_SHARE,
                 margin: float = 1.1) -> dict[str, Any]:
    """After the screen (730, amendment 66 (4)): failed screened episodes per task against the round(share x n/k)
    failed episodes 740's selection takes. Exploration's random π0.5 calls may fail less often than the base rollouts
    the screen was sized on. A task that falls short gets a proposed top-up, for the user to approve before 740's
    selection runs: more episodes of the same deterministic sequence (`screen_episodes` at a larger per_task extends
    it: same ids and seeds for those already screened), sized to expect margin x the shortfall at the task's observed
    failure rate; where none failed, at the rule-of-three bound 3/n, so that size is a lower bound."""
    need = int(round(failed_share * -(-per_task // k)))
    counts: dict[int, list[int]] = {}
    for r in rows:
        c = counts.setdefault(int(r["task_id"]), [0, 0])
        c[0] += 1
        c[1] += not r["success"]
    tasks: dict[int, dict[str, Any]] = {}
    for t, (n, f) in sorted(counts.items()):
        row: dict[str, Any] = {"screened": n, "failed": f, "needed": need, "short": max(0, need - f),
                               "failure_rate": f / n}
        if row["short"]:
            rate = f / n if f else 3 / n
            extra = math.ceil(margin * row["short"] / rate)
            row["top_up"] = {"episodes": extra, "screen_to": n + extra, "rate_used": rate,
                             "rate_basis": "observed" if f else "rule of three (none failed): a lower bound"}
        tasks[t] = row
    short = [t for t, r in tasks.items() if r["short"]]
    return {"needed_per_task": need, "tasks": tasks, "short_tasks": short,
            "top_up_episodes": sum(tasks[t]["top_up"]["episodes"] for t in short)}


def top_up_90(n: int, f: int, *, need: int = 21, target: float = 0.9, max_extra: int = 20_000) -> Optional[int]:
    """Amendment 68: the fewest more screened episodes m for which f + Binomial(m, f/n) reaches `need` with
    probability >= target, at the task's observed screen failure rate f/n; 0 when f >= need; None when f == 0."""
    if f >= need:
        return 0
    if f == 0:
        return None
    p, k = f / n, need - f

    def reach(m: int) -> float:
        pmf, below = (1 - p) ** m, 0.0
        for x in range(k):
            below += pmf
            pmf *= (m - x) / (x + 1) * p / (1 - p)
        return 1 - below
    m = k
    while reach(m) < target:
        m += 1
        if m > max_extra:
            raise TrancheBError(f"no top-up of {max_extra} episodes reaches {need} failed at rate {p:.4f}")
    return m


def top_up_decision(rep: Mapping[str, Any], *, usd_per_episode: float, overhead_usd: float, cap_usd: float = 1.00,
                    min_rate: float = 0.02) -> dict[str, Any]:
    """Amendment 68: the one pre-approved top-up after the screen. Short tasks get `top_up_90` more episodes (the same
    sequence); it runs without the user only if it costs <= cap_usd (episodes x usd_per_episode + one boot's
    overhead) and no task's observed screen failure rate is below min_rate. `sizes`: every task's new screen size."""
    tasks = {int(t): r for t, r in rep["tasks"].items()}
    extra = {t: top_up_90(r["screened"], r["failed"], need=rep["needed_per_task"]) for t, r in tasks.items()}
    stops = [f"task {t}: observed screen failure rate {r['failure_rate']:.3f} < {min_rate}"
             for t, r in sorted(tasks.items()) if r["failure_rate"] < min_rate]
    n_extra = sum(m or 0 for m in extra.values())
    cost = n_extra * usd_per_episode + (overhead_usd if n_extra else 0.0)
    if cost > cap_usd:
        stops.append(f"the top-up costs ${cost:.2f} > ${cap_usd:.2f}")
    return {"extra": {t: m for t, m in sorted(extra.items()) if m}, "episodes": n_extra, "cost_usd": cost,
            "sizes": {t: r["screened"] + (extra[t] or 0) for t, r in sorted(tasks.items())},
            "go": n_extra > 0 and not stops, "stops": stops}


def select_reruns(rows: Iterable[Mapping], *, per_task: int, k: int, seed: int,
                  failed_share: float = FAILED_SHARE) -> dict[str, dict]:
    """Per task n/k episodes (rounded up), round(s x n/k) failed, each stratum's shortfall made up from the other;
    uniform within a stratum. {episode_id: selection record}."""
    n_ep = -(-per_task // k)
    strata: dict[tuple[int, str], list[str]] = {}
    for r in rows:
        strata.setdefault((int(r["task_id"]), "success" if r["success"] else "failure"), []).append(r["episode_id"])
    out: dict[str, dict] = {}
    for task in sorted({t for t, _ in strata}):
        fail, succ = strata.get((task, "failure"), []), strata.get((task, "success"), [])
        n_f = min(int(round(failed_share * n_ep)), len(fail))
        n_s = min(n_ep - n_f, len(succ))
        n_f = min(n_ep - n_s, len(fail))
        for outcome, pool, m in (("failure", fail, n_f), ("success", succ, n_s)):
            order = sorted(pool, key=lambda e: (unit_uniform("ep-stratified", seed, e), e))
            for e in order[:m]:
                out[e] = {"stratum": {"source": "explore", "task_id": task, "parent_outcome": outcome},
                          "stratum_size": len(pool), "stratum_drawn": m, "p_episode": m / len(pool)}
    return out


def rerun_plan(rows: Sequence[Mapping], chosen: Mapping[str, Mapping], *, order_seed: int = ORDER_SEED_B) -> Plan:
    """The re-runs of the chosen screened episodes: the screen's spec with record `full`, seed index 2, at the screen's
    measured setup count."""
    by_id = {r["episode_id"]: r for r in rows}
    ids = sorted(chosen)
    random.Random(f"{SCREEN_PHASE}|rerun|{order_seed}").shuffle(ids)
    jobs = []
    for order, e in enumerate(ids):
        j = by_id[e]["job"]
        spec = {**j.spec, "record": "full", "setup_count": by_id[e]["setup_count"], "screen_job_id": j.job_id}
        jobs.append(Job(job_id(e, "episode", RERUN_SEED_INDEX), SCREEN_PHASE, "episode", e, RERUN_SEED_INDEX, 0,
                        order, spec))
    return Plan(SCREEN_PHASE, {"tranche": BRANCH_PHASE, "reruns": len(jobs), "order_seed": order_seed}, jobs)


# ---------------------------------------------------------------- 3. the comparison

def _events(data: bytes) -> list[tuple]:
    from .pilot import _events_without
    return _events_without(data, ("record",))


def compare_rerun(read: Callable[[str], Optional[bytes]], rerun: Job,
                  marker: Callable[[str], Optional[Mapping]]) -> dict[str, Any]:
    """A finished re-run against its screened run: runs, steps and events bit for bit (record-mode fields aside) and
    the markers' outcome. `passed` None while the re-run has no marker yet."""
    from .smoke import table_difference
    screen_id = rerun.spec["screen_job_id"]
    mr, ms = marker(rerun.job_id), marker(screen_id)
    if mr is None:
        return {"subject": rerun.subject, "passed": None, "problems": []}
    problems = []
    if ms is None:
        problems.append("the screened run has no marker")
    else:
        a = {k: ms["result"]["summary"].get(k) for k in ("run_id", "success", "steps")}
        b = {k: mr["result"]["summary"].get(k) for k in ("run_id", "success", "steps")}
        if a != b:
            problems.append(f"outcome {a} vs {b}")
    if mr.get("meta", {}).get("setup_count") != rerun.spec["setup_count"]:
        problems.append(f"re-run setup_count {mr.get('meta', {}).get('setup_count')!r}, spec {rerun.spec['setup_count']}")
    for table in ("runs", "steps", "events"):
        ka, kb = (f"records/{table}/{SCREEN_PHASE}/{screen_id}.parquet",
                  f"records/{table}/{SCREEN_PHASE}/{rerun.job_id}.parquet")
        x, y = read(ka), read(kb)
        if x is None or y is None:
            problems.append(f"{table}: missing ({'screen' if x is None else 're-run'})")
            continue
        if table == "events":
            ea, eb = _events(x), _events(y)
            d = None if ea == eb else next((f"event {i} differs" for i, (p, q) in enumerate(zip(ea, eb)) if p != q),
                                           f"{len(ea)} vs {len(eb)} events")
        else:
            d = table_difference(x, y, skip=RECORD_SKIP)
        if d:
            problems.append(f"{table}: {d}")
    return {"subject": rerun.subject, "passed": not problems, "problems": problems}


def record_mismatch(rep: Mapping[str, Any], *, mismatch: Optional[str] = None, stop_flag: Optional[str] = None) -> None:
    """A difference: the mismatch marker first (740 checks it before it reads a stop as a budget stop), then the spend
    guard's stop flag, so every re-run stops (amendment 56) and the step ends with 76 (amendment 66 (2))."""
    from pathlib import Path
    failed = [{"subject": r["subject"], "problems": r["problems"]} for r in rep["failed"]]
    if mismatch:
        Path(mismatch).write_text(json.dumps({"n_failed": len(failed), "failed": failed}, sort_keys=True))
    if stop_flag:
        Path(stop_flag).write_text(f"Tranche B: a re-run differs from its screened run ({failed[0]['subject']}): all "
                                   f"re-runs stop (amendment 56)\n")


def compare_all(plan: Plan, read: Callable[[str], Optional[bytes]], marker: Callable[[str], Optional[Mapping]],
                done: Optional[Mapping[str, Mapping]] = None) -> dict[str, Any]:
    """Every re-run with a marker (those in `done`, already compared and passed, are skipped)."""
    done = dict(done or {})
    rows = [done[j.job_id] if j.job_id in done else compare_rerun(read, j, marker) for j in plan.jobs]
    by_id = {j.job_id: r for j, r in zip(plan.jobs, rows)}
    failed = [r for r in rows if r["passed"] is False]
    return {"compared": sum(r["passed"] is not None for r in rows), "pending": sum(r["passed"] is None for r in rows),
            "failed": failed, "passed": not failed, "all_done": all(r["passed"] is not None for r in rows),
            "rows": by_id}


# ---------------------------------------------------------------- 4. points and the branch plan

def rerun_points(rerun: Job, events: Sequence[Mapping], listed: set[str]) -> list[DecisionPointRef]:
    """A re-run's decision points whose snapshot its marker lists."""
    from vlar_harness.snapshot import SnapshotStore
    out = []
    for e in events:
        if e["t"] != "decision_point" or e["run_id"] != rerun.subject:
            continue
        ev = json.loads(e["event"]) if isinstance(e["event"], str) else e["event"]
        sid = SnapshotStore.make_id(rerun.subject, int(ev["index"]))
        if sid in listed:
            out.append(DecisionPointRef(sid, rerun.subject, "explore", int(rerun.spec["task_id"]),
                                        int(rerun.spec["init_idx"]), int(ev["index"]), int(ev["budget_remaining"])))
    return out


def select_points(points: Iterable[DecisionPointRef], chosen: Mapping[str, Mapping], *, k: int, seed: int,
                  step_limit: int, interval: int, min_calls: int = 1) -> dict[str, dict]:
    """k eligible points per re-run episode (all if fewer), uniform by a hash of (seed, episode, index)."""
    by_ep: dict[str, list[DecisionPointRef]] = {}
    for p in points:
        if p.episode_id not in chosen:
            raise TrancheBError(f"{p.snapshot_id}: its episode was not selected")
        if eligible(p, step_limit=step_limit, interval=interval, min_calls=min_calls):
            by_ep.setdefault(p.episode_id, []).append(p)
    out: dict[str, dict] = {}
    for e, pts in sorted(by_ep.items()):
        order = sorted(pts, key=lambda q: (unit_uniform("dp-in-episode", seed, e, q.index), q.snapshot_id))
        m = min(k, len(order))
        c = chosen[e]
        for q in order[:m]:
            out[q.snapshot_id] = {"stratum": dict(c["stratum"]), "stratum_size": c["stratum_size"],
                                  "stratum_drawn": c["stratum_drawn"], "p_episode": c["p_episode"],
                                  "episode_points": len(order), "points_drawn": m,
                                  "inclusion_prob": c["p_episode"] * m / len(order), "by": "episode"}
    return out


def branch_settings(per_task: int, seed: int) -> dict[str, Any]:
    return {"enabled": list(ENABLED), "seeds": SEEDS, "tier_one_seeds": TIER_ONE_SEEDS, "order_seed": ORDER_SEED_B,
            "dp_sampling": {"scheme": STRATIFIED, "seed": int(seed), "per_task": int(per_task),
                            "failed_share": FAILED_SHARE, "min_calls": 1}}


def branch_plan_b(points: Sequence[DecisionPointRef], selection: Mapping[str, Mapping], *, per_task: int, seed: int,
                  branches: Mapping[str, Mapping[str, Any]], step_limit: int, interval: int):
    return branch_plan(BRANCH_PHASE, points, branches=branches, step_limit=step_limit, interval=interval,
                       selection=selection, **branch_settings(per_task, seed))
