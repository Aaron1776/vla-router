"""The pilot (docs/PROPOSALS-2026-11-24.md §4 and §0.4; SPEC §17 amendments 56-58): `vlar-ops pilot ...`.

- `plan` (on the Mac, from local copies of the base rollouts' runs, events and result markers): 15 decision points per
  task from the base snapshots, 10 from failed episodes and 5 from successful ones where available, drawn uniformly
  within each stratum with a recorded seed (`branchplan.stratified_selection`, amendment 57), at points where branch 3
  can make its three calls (§4's "budget >= 30 steps"; the π0.5 budget is full at every base point, so the binding
  limit is the episode's remaining decision points). All four branches, 3 seeds; b1's first seed is the parent.
  Writes `plan.json`, `labels.json` (the label map, with each point's stratum and inclusion probability),
  `selection.json` (strata per task) and `checks.json` (the subjects of the cheap checks below).
- `check` (on the VM, after the golden check, through the production servers): §0.4's cheap checks, bit for bit.
  (i) features: at restored mid-point snapshots, `/features` against the probe `/sample`'s features recorded in the
  points table, with a fresh `/sample` (same seeds) to show the restore and payload reproduce the record;
  (ii) `xvla_only` against routed always-local: 3 base episodes run in `xvla_only` mode with their rollout seeds,
  executed steps and outcome against the base records; (iii) record `off`: 3 base episodes re-run with record `off`,
  runs, steps and events against their record-`full` originals. Nothing is uploaded to the dataset.
- `project` and the stop rule (amendment 58): projected positives per task and fold under uniform and stratified
  sampling at the plan's budgets, and the decision the user set before the pilot ran.
"""

from __future__ import annotations

import io
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .branchplan import BRANCH_ORDER, STRATIFIED, DecisionPointRef, branch_plan, eligible, stratified_selection
from .plan import Job, job_id, unit_uniform
from .queue import result_key

PHASE = "pilot"
SELECTION = {"scheme": STRATIFIED, "seed": 20261124, "per_task": 15, "failed_share": 2 / 3, "min_calls": 3}
SEEDS, TIER_ONE_SEEDS, ORDER_SEED = 3, 2, 7
CHECK_SEED = "pilot-checks-2026-11-24"
N_EPISODE_CHECKS = 3
POSITIVE = 1 / 3                 # p_2 - p_1 >= 1/3 under the stop rule's profile (§4)
STOP_PROFILE = "cascader_nospread"     # SPEC §17 amendment 63 (2): the headline cascader's profile (the router's)
PILOT_RULE_PROFILE = "cascader_full"   # amendment 58 as the pilot report applied it (the headline before 63)
FULL_PROFILE = "cascader_full"         # the profile whose probe charges cut rescues (counted descriptively)


class PilotError(RuntimeError):
    pass


# ---------------------------------------------------------------- plan (Mac)

def base_points(runs: Sequence[Mapping], events: Sequence[Mapping]) -> tuple[list[DecisionPointRef],
                                                                             dict[str, bool], dict[str, str]]:
    """Every base decision point from the runs and events tables (a base rollout records a snapshot at each one):
    the points, each parent's outcome (success at 520 steps, no time cutoff) and each point's kind."""
    from vlar_harness.snapshot import SnapshotStore
    base = {r["run_id"]: r for r in runs if r["run_kind"] == "episode" and str(r["run_id"]).startswith("base/")}
    if not base:
        raise PilotError("no base runs in the runs table")
    pts, kinds = [], {}
    for e in events:
        if e["t"] != "decision_point" or e["run_id"] not in base:
            continue
        ev, r = json.loads(e["event"]), base[e["run_id"]]
        sid = SnapshotStore.make_id(r["run_id"], int(ev["index"]))
        pts.append(DecisionPointRef(sid, r["run_id"], "base", int(r["task_id"]), int(r["init_idx"]), int(ev["index"]),
                                    int(ev["budget_remaining"])))
        kinds[sid] = ev["kind"]
    return pts, {k: bool(r["success"]) for k, r in base.items()}, kinds


def snapshots_listed(markers_dir: Path) -> set[str]:
    """Snapshot ids every base result marker lists (the snapshot objects were checked against them, 2026-11-24)."""
    out = set()
    for f in Path(markers_dir).glob("*.json"):
        for a in json.loads(f.read_text())["result"]["artifacts"]:
            if a["key"].startswith("snapshots/") and a["key"].endswith(".npz"):
                out.add(a["key"][len("snapshots/"):-len(".npz")])
    return out


def strata_table(points: Sequence[DecisionPointRef], success: Mapping[str, bool], *, step_limit: int, interval: int,
                 min_calls: int) -> dict[int, dict[str, Any]]:
    """Per task: eligible points and episodes in each stratum, and φ, the share of eligible points in failed
    parents (what uniform per-point sampling draws from failed parents)."""
    out: dict[int, dict[str, Any]] = {}
    for p in points:
        if not eligible(p, step_limit=step_limit, interval=interval, min_calls=min_calls):
            continue
        row = out.setdefault(p.task_id, {"points_failed": 0, "points_success": 0, "episodes_failed": set(),
                                         "episodes_success": set()})
        k = "failed" if not success[p.episode_id] else "success"
        row[f"points_{k}"] += 1
        row[f"episodes_{k}"].add(p.episode_id)
    for row in out.values():
        for k in ("failed", "success"):
            row[f"episodes_{k}"] = len(row[f"episodes_{k}"])
        row["phi"] = row["points_failed"] / (row["points_failed"] + row["points_success"])
    return dict(sorted(out.items()))


def exclude_earlier(sel_total: Mapping[str, dict], sel_prev: Mapping[str, dict], earlier_points,
                    *, earlier: Mapping[str, Any]) -> dict[str, dict]:
    """A later tranche drawn as ONE stratified draw with an earlier one (SPEC §17 amendment 67 (3): A and F one
    draw): the same seed, strata and settings at the combined per-task count. Within a stratum the draw takes the
    first k points of a seeded order, so the earlier tranche's points must be exactly the draw at its own count
    (`sel_prev`) and a subset of the combined draw; the later tranche is the rest. Each kept record keeps the
    combined draw's stratum_drawn and inclusion probability (k_total / N) and names the earlier tranche."""
    prev, pts, total = set(sel_prev), set(earlier_points), set(sel_total)
    if prev != pts:
        raise PilotError(f"the earlier tranche's {len(pts)} points are not the draw at its own count ({len(prev)} "
                         f"points; {len(pts - prev)} not in it, {len(prev - pts)} missing): not one draw")
    if not prev <= total:
        raise PilotError(f"{len(prev - total)} of the earlier tranche's points are not in the combined draw")
    return {sid: {**rec, "combined_draw": dict(earlier)} for sid, rec in sorted(sel_total.items()) if sid not in prev}


def make_plan(root: Path, markers_dir: Path, out_dir: Path, *, cfg=None, phase: str = PHASE,
              selection: Optional[Mapping[str, Any]] = None, enabled: Sequence[str] = BRANCH_ORDER,
              seeds: int = SEEDS, tier_one_seeds: int = TIER_ONE_SEEDS, order_seed: int = ORDER_SEED,
              earlier: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """The branch plan over a stratified draw of base points: the pilot's by default; Phase 1's Tranche A (and the
    final tier's extra base points) with its own phase, selection settings and branches (`vlar_ops.phase1`).
    `earlier` ({"phase", "plan_id", "per_task", "points"}): this tranche and the earlier one are one stratified draw
    at `selection`'s per_task (the combined count); the earlier tranche's points are left out (`exclude_earlier`)."""
    from vlar_harness.config import load_config
    from vlar_harness.dataset import read_table
    cfg = cfg or load_config()
    sel_settings = dict(selection or SELECTION)
    step_limit, interval = int(cfg["env"]["step_limit"]), int(cfg["decision"]["interval_steps"])
    points, success, kinds = base_points(read_table(root, "runs").to_pylist(), read_table(root, "events").to_pylist())
    if len(success) != 500:
        raise PilotError(f"{len(success)} base runs, not 500: download all of records/runs/base")
    sel = stratified_selection(points, success, step_limit=step_limit, interval=interval, dp_sampling=sel_settings)
    if earlier is not None:
        prev = stratified_selection(points, success, step_limit=step_limit, interval=interval,
                                    dp_sampling={**sel_settings, "per_task": int(earlier["per_task"])})
        sel = exclude_earlier(sel, prev, earlier["points"],
                              earlier={k: earlier[k] for k in ("phase", "plan_id", "per_task")})
    listed = snapshots_listed(markers_dir)
    missing = sorted(set(sel) - listed)
    if missing:
        raise PilotError(f"{len(missing)} selected snapshots are not listed by any base result marker: {missing[:3]}")
    plan, labels = branch_plan(phase, points, branches=cfg["branches"], step_limit=step_limit, interval=interval,
                               seeds=seeds, enabled=list(enabled), tier_one_seeds=tier_one_seeds,
                               order_seed=order_seed, dp_sampling=sel_settings, selection=sel)
    by_task: dict[int, list[str]] = defaultdict(list)
    for sid, s in sorted(sel.items()):
        by_task[s["stratum"]["task_id"]].append(sid)
    rank = lambda *k: unit_uniform(CHECK_SEED, *k)                                  # noqa: E731
    features = [min((sid for sid in sids if kinds[sid] == "mid"), key=lambda s: rank("features", s))
                for _, sids in sorted(by_task.items()) if any(kinds[s] == "mid" for s in sids)]
    episodes = sorted(success, key=lambda e: rank("episode", e))
    checks = {"features": [{"snapshot_id": s, "task_id": labels[s]["point"]["task_id"],
                            "episode_id": labels[s]["point"]["episode_id"]} for s in features],
              "xvla_only": episodes[:N_EPISODE_CHECKS], "record_off": episodes[N_EPISODE_CHECKS:2 * N_EPISODE_CHECKS],
              "seed": CHECK_SEED}
    strata = {t: {**row, "selected_failed": sum(1 for s in by_task[t] if sel[s]["stratum"]["parent_outcome"] == "failure"),
                  "selected_success": sum(1 for s in by_task[t] if sel[s]["stratum"]["parent_outcome"] == "success")}
              for t, row in strata_table(points, success, step_limit=step_limit, interval=interval,
                                         min_calls=sel_settings["min_calls"]).items()}
    phase1 = strata_table(points, success, step_limit=step_limit, interval=interval, min_calls=1)
    selection = {"settings": sel_settings, "phase": phase, "n_points": len(sel), "n_jobs": len(plan.jobs), "plan_id": plan.plan_id,
                 "strata": strata, "phase1_strata": phase1,
                 "base_episode_success": {t: sum(success[e] for e in success if int(e.split("/t")[1].split("/")[0]) == t)
                                          for t in sorted(strata)}}
    if earlier is not None:       # one draw with the earlier tranche: settings.per_task is the combined count
        selection["combined_with"] = {k: earlier[k] for k in ("phase", "plan_id", "per_task")}
    out_dir.mkdir(parents=True, exist_ok=True)
    plan.write(out_dir / "plan.json")
    for name, obj in (("labels", labels), ("checks", checks), ("selection", selection)):
        (out_dir / f"{name}.json").write_text(json.dumps(obj, indent=1, sort_keys=True, default=int))
    return selection


# ---------------------------------------------------------------- checks (VM)

def _f32(a):
    import numpy as np
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32).ravel())


def _same(a, b) -> dict[str, Any]:
    import numpy as np
    x, y = _f32(a), _f32(b)
    if x.shape != y.shape:
        return {"equal": False, "detail": f"shapes {x.shape} vs {y.shape}"}
    diff = np.flatnonzero(x.view(np.uint32) != y.view(np.uint32))
    return {"equal": diff.size == 0, "values_differing": int(diff.size),
            "max_abs_diff": float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if diff.size else 0.0}


def features_check(make_episode: Callable[[int], Any], read_snapshot: Callable[[str], Any],
                   recorded: Callable[[str], Mapping[str, Any]], client, subjects: Sequence[Mapping]) -> dict:
    """(i): at each restored mid point, /features against the recorded probe features (amendment 55 (5), §0.3's router
    re-runs). A fresh /sample with the probe's seeds must reproduce the record first, or the comparison shows
    nothing about the endpoints."""
    from vlar_harness import seeding
    rows, problems = [], []
    for s in subjects:
        sid = s["snapshot_id"]
        ep = make_episode(int(s["task_id"]))
        if ep.restore(read_snapshot(sid), snapshot_id=sid) is None or ep.state.point_kind != "mid":
            problems.append(f"{sid}: not a mid decision point after restore")
            continue
        payload = ep.xvla.measurement_payload(ep.sim, ep.goal)
        seeds = [seeding.call_seed(ep.state.rollout_seed, ep.sim.steps, "xvla", "probe", k) for k in range(ep.K)]
        fs, ff = client.sample(payload, seeds)["features"], client.features(payload)["features"]
        rec = recorded(sid)
        row = {"snapshot_id": sid}
        for name, key in (("main", "enc_mean"), ("wrist", "wrist_mean")):
            row[f"sample_vs_record_{name}"] = _same(fs[key], rec[f"features_{name}"])
            row[f"features_vs_record_{name}"] = _same(ff[key], rec[f"features_{name}"])
        rows.append(row)
    reproduced = bool(rows) and all(r[f"sample_vs_record_{n}"]["equal"] for r in rows for n in ("main", "wrist"))
    if not reproduced:
        problems.append("a fresh /sample did not reproduce the recorded probe features: the restore or the payload "
                        "differs from the recorded run (unexplained: max-effort checkpoint)")
    equal = reproduced and all(r[f"features_vs_record_{n}"]["equal"] for r in rows for n in ("main", "wrist"))
    return {"passed": not problems, "features_equal_probe": equal, "sample_reproduces_record": reproduced,
            "points": rows, "problems": problems,
            "consequence": ("trunk features usable offline; router half of DAgger screened (§0.3)" if equal else
                            "router half of DAgger records full from the start (+$2.1); /features recomputed at trunk "
                            "snapshots (amendment 55 (5)); the user's decision 1")}


EPISODE_SKIP = {"xvla_only": ("mode", "record", "calibration_hash"), "record_off": ("record", "calibration_hash")}


def _table(data: bytes):
    import pyarrow.parquet as pq
    return pq.read_table(io.BytesIO(data))


def _events_without(data: bytes, drop: Sequence[str]) -> list[tuple]:
    rows = _table(data).to_pylist()
    return [(r["run_id"], r["seq"], r["t"], r["step"],
             json.dumps({k: v for k, v in json.loads(r["event"]).items() if k not in drop}, sort_keys=True))
            for r in rows]


def episode_check(run_job: Callable[[Job], list[dict]], original: Callable[[str], Optional[bytes]], job: Job,
                  variant: str) -> dict:
    """(ii) `xvla_only` (steps and outcome) or (iii) record `off` (runs, steps and events) against the original base
    job's record-`full` tables, bit for bit; the columns that name the run's settings (and the calibration version,
    provenance) aside."""
    from .smoke import table_difference
    spec = dict(job.spec)
    spec.update({"mode": "xvla_only", "record": "off"} if variant == "xvla_only" else {"record": "off"})
    arts = {a["key"]: a for a in run_job(Job(job.job_id, job.phase, job.kind, job.subject, job.seed_index, job.tier,
                                             job.order, spec))}
    problems, compared = [], []
    tables = ("runs", "steps") if variant == "xvla_only" else ("runs", "steps", "events")
    for table in tables:
        key = f"records/{table}/{job.phase}/{job.job_id}.parquet"
        orig = original(key)
        if orig is None or key not in arts:
            problems.append(f"{key}: missing ({'original' if orig is None else 're-run'})")
            continue
        mine = Path(arts[key]["path"]).read_bytes()
        if table == "events":
            a, b = _events_without(orig, ("record",)), _events_without(mine, ("record",))
            d = None if a == b else next((f"event {i} differs" for i, (x, y) in enumerate(zip(a, b)) if x != y),
                                         f"{len(a)} vs {len(b)} events")
        else:
            d = table_difference(orig, mine, skip=EPISODE_SKIP[variant])
        compared.append(table)
        if d:
            problems.append(f"{table}: {d}")
    extra = sorted(k for k in arts if k.startswith(("snapshots/", "images/", "records/points/")))
    if extra:
        problems.append(f"record off wrote {len(extra)} snapshot, image or point files")
    run = _table(Path(arts[f"records/runs/{job.phase}/{job.job_id}.parquet"]["path"]).read_bytes()).to_pylist()[0] \
        if f"records/runs/{job.phase}/{job.job_id}.parquet" in arts else {}
    return {"subject": job.subject, "variant": variant, "passed": not problems, "problems": problems,
            "tables": compared, "outcome": {k: run.get(k) for k in ("success", "end_reason", "steps", "success_step")},
            "ran_as": {k: run.get(k) for k in ("mode", "record")}}


# ---------------------------------------------------------------- the base run's setup (finding of 2026-11-24)

MAX_LOADS = 64           # a base worker ran ~6 episodes of a task per process; far more would be a different problem
FIXTURE_FIELDS = ("body_pos", "body_quat")   # what a fixture's placement writes; site_rgba may toggle in an episode


def model_fields(sim, fields: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """The model fields LIBERO writes at run time (`vlar_harness.sim.state.MODEL_FIELDS`: fixture poses, site
    colours), copied."""
    import numpy as np
    from vlar_harness.sim import state as sim_state
    model = sim.env.env.sim.model._model
    return {f: np.array(getattr(model, f), copy=True) for f in (fields or sim_state.MODEL_FIELDS)}


def fields_equal(a: Mapping[str, Any], b: Mapping[str, Any], fields: Optional[Sequence[str]] = None) -> bool:
    import numpy as np
    keys = list(fields) if fields else sorted(set(a) | set(b))
    return all(k in a and k in b and np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in keys)


def fields_max_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    import numpy as np
    return max((float(np.max(np.abs(np.asarray(a[k], dtype=np.float64) - np.asarray(b[k], dtype=np.float64))))
                if np.shape(a[k]) == np.shape(b[k]) else math.inf for k in set(a) & set(b)), default=math.inf)


def base_loads(make_sim: Callable[[int], Any], task_id: int, init_idx: int, scene_seed: int,
               target: Mapping[str, Any], *, max_loads: int = MAX_LOADS,
               nearest: Optional[dict] = None) -> Optional[int]:
    """The model-load count of the base run's simulator when it set the episode up, that setup included.

    LIBERO appends its open/close and on/off property samplers to a list kept across hard resets
    (`object_property_initializers`), and at every reset they draw from np.random after `env.seed` and before the
    fixtures are placed; a fixture's pose lives in the model (`body_pos`, `body_quat`), which `set_init_state` does
    not restore. So the fixtures of tasks 3, 8 and 9 sit where the load count puts them, and a simulator that has
    set up a different number of episodes sees another scene (docs/HARNESS.md, "Fixture placement depends on the
    simulator's load count"). A fresh simulator is load 1 and each `setup_episode` one more; the count is the first
    whose setup gives the model fields of the base run's decision-point-0 snapshot, bit for bit. None if none up to
    `max_loads`; `nearest` (a dict, if given) gets the closest count and its largest difference either way (on another
    CPU architecture the placement can differ in the last bit: 5e-19 m for task 3's cabinet on the Mac)."""
    best = (None, math.inf)
    sim = make_sim(task_id)
    try:
        for n in range(2, max_loads + 1):
            sim.setup_episode(init_idx, scene_seed)
            here = model_fields(sim)
            if fields_equal(here, target):
                best = (n, 0.0)
                return n
            d = fields_max_diff(here, target)
            if d < best[1]:
                best = (n, d)
        return None
    finally:
        sim.close()
        if nearest is not None:
            nearest.update({"loads": best[0], "max_abs_diff": best[1]})


def warm_up(sim, loads: int) -> None:
    """Bring a fresh simulator (load 1) to `loads - 1` loads, so that the next `setup_episode` is load `loads`. Which
    episodes the extra loads set up does not matter: only their number reaches the next setup (checked on the Mac
    against the VM's base t8/i30, 2026-11-24)."""
    for _ in range(loads - 2):
        sim.setup_episode(0, 0)


def replicated_episode_check(cfg, clients: Mapping[str, Any], read_bytes: Callable[[str], Optional[bytes]],
                             read_snapshot: Callable[[str], Any], staging: Path, header: Mapping[str, Any], job: Job,
                             variant: str, *, make_sim: Callable[[int], Any]) -> dict:
    """(ii) or (iii) on a fresh simulator brought to the base run's load count (`base_loads`), so the re-run's setup is
    the base run's; afterwards its fixtures must still be the base run's. `seconds`: wall time of the load-count search
    (a fresh simulator and n setups), the warm-up (a fresh simulator and n - 2 setups) and the episode itself (the
    user, 2026-11-24: the first direct measure of a screened run's cost, alone on the servers)."""
    import time

    from vlar_harness.collect import Runner
    from vlar_harness.snapshot import SnapshotStore

    from . import vm_worker
    task, init = int(job.spec["task_id"]), int(job.spec["init_idx"])
    first = read_snapshot(SnapshotStore.make_id(job.subject, 0))
    target, scene = first.sim["model_fields"], int(first.episode["scene_seed"])
    near: dict = {}
    t0 = time.monotonic()
    n = base_loads(make_sim, task, init, scene, target, nearest=near)
    seconds = {"search": time.monotonic() - t0}
    if n is None:
        return {"subject": job.subject, "variant": variant, "passed": False, "tables": [], "outcome": {},
                "ran_as": {}, "loads": None, "nearest": near, "fixtures_equal_base": None, "seconds": seconds,
                "problems": [f"no load count up to {MAX_LOADS} reproduces the base run's model fields at step 0 "
                             f"(nearest: {near})"]}
    episodes = vm_worker.ProductionEpisodes(cfg, clients["xvla"], clients["pi05"], make_sim=make_sim)
    try:
        t0 = time.monotonic()
        warm_up(episodes.sim(task), n)
        seconds["warm_up"] = time.monotonic() - t0
        runner = Runner(cfg, episodes, read_bytes, staging, p_escalate=0.0, header=header)
        t0 = time.monotonic()
        rep = episode_check(lambda j: runner(j)[0], read_bytes, job, variant)
        seconds["episode"] = time.monotonic() - t0
        same = fields_equal(model_fields(episodes.sim(task), FIXTURE_FIELDS), target, FIXTURE_FIELDS)
    finally:
        episodes.close()
    if not same:
        rep = {**rep, "passed": False, "problems": rep["problems"] + ["the re-run's fixtures are not the base run's"]}
    return {**rep, "loads": n, "fixtures_equal_base": same, "seconds": seconds}


def long_episode_subjects(rows: Sequence[Mapping[str, Any]], *, tasks: Sequence[int], seed: str,
                          exclude: Sequence[str] = (), near: float = 0.85) -> list[str]:
    """One long base episode per task, for the corrected check (iii): ended at the step limit, or succeeded with its
    profile-none time at least `near` of the task's calibrated limit; the first of each task by a seeded rank.
    `rows`: {"episode_id", "task_id", "end_reason", "time_frac"} per base run."""
    out = []
    for t in tasks:
        ok = [r for r in rows if int(r["task_id"]) == t and r["episode_id"] not in exclude and (
            r["end_reason"] == "step_limit" or (r["end_reason"] == "success" and r["time_frac"] >= near))]
        if not ok:
            raise PilotError(f"task {t}: no long base episode")
        out.append(min(ok, key=lambda r: unit_uniform(seed, r["episode_id"]))["episode_id"])
    return out


def base_time_rows(runs: Sequence[Mapping], events: Sequence[Mapping], cfg) -> list[dict[str, Any]]:
    """Per base run: task, end, and its profile-none simulated time as a fraction of the task's calibrated limit."""
    from vlar_harness import clock as clk
    charges = clk.Charges.from_config(cfg)
    by: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by[e["run_id"]].append(json.loads(e["event"]))
    out = []
    for r in runs:
        if r["run_kind"] != "episode" or not str(r["run_id"]).startswith("base/"):
            continue
        t = clk.replay(by[r["run_id"]], "none", charges)
        out.append({"episode_id": r["run_id"], "task_id": int(r["task_id"]), "end_reason": r["end_reason"],
                    "steps": int(r["steps"]), "time_frac": t / float(cfg.time_limit_s(int(r["task_id"])))})
    return out


def run_checks(ops_cfg: Mapping[str, Any], plan_dir: Path, staging: Path, *, clients=None, make_sim=None,
               server_wait_s: float = 600.0, subjects_file: str = "checks.json") -> dict:
    import os
    import sys
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "cgl")
    from vlar_harness import provenance
    from vlar_harness import snapshot as snap
    from vlar_harness.config import load_config
    from vlar_harness.dataset import snapshot_key

    from . import golden, vm_worker
    from .storage import open_storage
    cfg = load_config()
    if not cfg.frozen:
        raise PilotError("the pilot's checks are for the frozen harness")
    image = vm_worker.image_check(cfg)
    if clients is None:
        from vlar_harness.servers.clients import PI05Client, XVLAClient
        srv = cfg["servers"]
        clients = {"xvla": XVLAClient(srv["xvla"]["host"], int(srv["xvla"]["port"])),
                   "pi05": PI05Client(srv["pi05"]["host"], int(srv["pi05"]["port"]))}
    vm_worker.check_servers(cfg, clients, wait_s=server_wait_s)
    storage = open_storage(ops_cfg["storage_url"])
    subjects = json.loads((Path(plan_dir) / subjects_file).read_text())
    if make_sim is None:
        from vlar_harness.sim.libero_sim import LiberoSim, SimSpec
        spec = SimSpec.from_config(cfg)
        make_sim = lambda task_id: LiberoSim(task_id, spec)                       # noqa: E731
    episodes = vm_worker.ProductionEpisodes(cfg, clients["xvla"], clients["pi05"], make_sim=make_sim)
    header = provenance.record_header(cfg)

    def read(sid):
        data = storage.get_bytes(snapshot_key(sid))
        if data is None:
            raise PilotError(f"snapshot {sid} is not in the bucket")
        return snap.from_bytes(data)

    def recorded(sid):
        ep_id = sid.rsplit("/", 1)[0]
        rows = _table(storage.get_bytes(f"records/points/base/{job_id(ep_id, 'episode', 1)}.parquet")).to_pylist()
        (row,) = [r for r in rows if r["snapshot_id"] == sid]
        return row

    def original_job(subject):
        data = storage.get_bytes(result_key("base", job_id(subject, "episode", 1)))
        if data is None:
            raise PilotError(f"{subject}: no result marker")
        return Job(**json.loads(data)["job"])

    rep: dict[str, Any] = {}
    try:
        if subjects.get("features"):
            rep["features"] = features_check(episodes, read, recorded, clients["xvla"], subjects["features"])
        for variant in ("xvla_only", "record_off"):
            if subjects.get(variant):
                rep[variant] = [replicated_episode_check(cfg, clients, storage.get_bytes, read, staging, header,
                                                         original_job(s), variant, make_sim=make_sim)
                                for s in subjects[variant]]
    finally:
        episodes.close()
    rep["passed"] = rep.get("features", {}).get("sample_reproduces_record", True) and all(
        r["passed"] for v in ("xvla_only", "record_off") for r in rep.get(v, []))
    return {**rep, "subjects_file": subjects_file, "boot_id": golden.boot_id(), "calibration_hash": cfg.calibration_hash,
            "image_check": image}


def summary_line(rep: Mapping[str, Any]) -> str:
    """'passed', or what failed (for the step log and the runner's reason)."""
    if rep.get("passed"):
        return "passed"
    out = {}
    if "features" in rep:
        out["features"] = rep["features"]["problems"]
        out["features_equal_probe"] = rep["features"]["features_equal_probe"]
    for v in ("xvla_only", "record_off"):
        if v in rep:
            out[v] = [{"subject": r["subject"], "loads": r.get("loads"), "problems": r["problems"]} for r in rep[v]
                      if not r["passed"]]
    return "MISMATCH: " + json.dumps(out)[:1500]


RESUME_CHECK_SEED = "pilot-check-iii-2026-11-24"
RESUME_CHECK_TASKS = (3, 9, 8, 2, 4, 5)    # the three whose fixtures move, the other stove scene, the two time-bound


def write_long_episode_checks(root: Path, plan_dir: Path, out: Path, *, cfg=None) -> dict[str, Any]:
    """The corrected check (iii) (the user, 2026-11-24): base/t8/i30 again, plus one long base episode near its limit
    from each of RESUME_CHECK_TASKS, none already checked; record off, each on the base run's load count."""
    from vlar_harness.config import load_config
    from vlar_harness.dataset import read_table
    cfg = cfg or load_config()
    done = json.loads((Path(plan_dir) / "checks.json").read_text())
    before = set(done["xvla_only"]) | set(done["record_off"])
    rows = base_time_rows(read_table(root, "runs").to_pylist(), read_table(root, "events").to_pylist(), cfg)
    subjects = ["base/t8/i30"] + long_episode_subjects(rows, tasks=RESUME_CHECK_TASKS, seed=RESUME_CHECK_SEED,
                                                        exclude=sorted(before))
    by = {r["episode_id"]: r for r in rows}
    obj = {"record_off": subjects, "seed": RESUME_CHECK_SEED, "tasks": list(RESUME_CHECK_TASKS),
           "rule": "base/t8/i30, then per task the long base episode (step limit, or success at >= 0.85 of the "
                   "task's limit under profile none) of lowest seeded rank, not checked before",
           "episodes": {s: by[s] for s in subjects}}
    Path(out).write_text(json.dumps(obj, indent=1, sort_keys=True))
    return obj


# ---------------------------------------------------------------- the resume's claim order (the user, 2026-11-24)

def resume_order(jobs: Sequence[tuple[str, str, Job]], labels: Mapping[str, Mapping]) -> tuple[list[str], list[str]]:
    """Point by point, so a stop leaves whole labelled points. Points with a job left are grouped by stratum cell (task,
    parent outcome) and sorted within each cell by finished jobs (most first); round r takes each cell's r-th point,
    the round's points again by finished jobs (then task, outcome). Each point's open jobs keep their planned order.
    `jobs`: (job_id, status, Job) of the phase. Returns (point order, job order)."""
    by_point: dict[str, list[tuple[str, Job]]] = defaultdict(list)
    done: dict[str, int] = defaultdict(int)
    for _, status, job in jobs:
        by_point[job.subject].append((status, job))
        done[job.subject] += status == "done"
    cells: dict[tuple, list[str]] = defaultdict(list)
    for p, js in by_point.items():
        if any(s != "done" for s, _ in js):
            st = labels[p]["selection"]["stratum"]
            cells[(int(st["task_id"]), st["parent_outcome"])].append(p)
    for c in cells.values():
        c.sort(key=lambda p: (-done[p], p))
    points: list[str] = []
    for r in range(max((len(c) for c in cells.values()), default=0)):
        level = sorted(((c[r], key) for key, c in cells.items() if r < len(c)), key=lambda x: (-done[x[0]], x[1]))
        points += [p for p, _ in level]
    order = [j.job_id for p in points for _, j in sorted((x for x in by_point[p] if x[0] != "done"),
                                                         key=lambda x: x[1].order)]
    return points, order


def reorder_queue(q, plan_dir: Path) -> dict[str, Any]:
    """Set the claim order of the plan's open jobs (`resume_order`); only the queue's claim columns change. The phase
    is the plan's (selection.json's `phase`: the pilot's plans have none, so `pilot`)."""
    labels = json.loads((Path(plan_dir) / "labels.json").read_text())
    sel = Path(plan_dir) / "selection.json"
    phase = json.loads(sel.read_text()).get("phase", PHASE) if sel.exists() else PHASE
    jobs = q.phase_jobs(phase)
    points, order = resume_order(jobs, labels)
    q.set_order(phase, order)
    subjects = {j.subject for _, _, j in jobs}
    done_by = defaultdict(int)
    for _, s, j in jobs:
        done_by[j.subject] += s == "done"
    return {"points": len(subjects), "points_complete": len(subjects) - len(points), "points_open": len(points),
            "jobs_reordered": len(order), "first_points": [(p, done_by[p]) for p in points[:5]],
            "order_points": points}


# ---------------------------------------------------------------- projections and the stop rule (amendment 58)

def folds(tasks: Sequence[int] = tuple(range(10))) -> dict[int, list[int]]:
    """Fold i: test task i, validation task i+1, the other 8 train (§7)."""
    n = len(tasks)
    return {t: [u for u in tasks if u not in (t, tasks[(i + 1) % n])] for i, t in enumerate(tasks)}


def expected_positives(n: float, rate_f: float, rate_s: float, *, scheme: str, phi: float, share: float = 2 / 3,
                       failed_available: float = math.inf) -> float:
    """E[positives] among n points of a task: uniform per point draws a share φ from failed parents; stratified draws
    round(share x n) from them, capped by the failed points available (§0.4)."""
    if scheme == "uniform":
        return n * (phi * rate_f + (1 - phi) * rate_s)
    n_f = min(share * n, failed_available)
    return n_f * rate_f + (n - n_f) * rate_s


def stop_rule(expected: Mapping[int, float], *, per_task: float = 20, per_fold: float = 100,
              max_flagged: int = 3, also_flagged: Sequence[int] = ()) -> dict[str, Any]:
    """§4: a task expecting fewer than 20 positives is flagged; more than 3 flagged tasks, or a training fold (8 tasks)
    expecting fewer than 100, stops before Phase 1. `also_flagged`: tasks flagged by another test (amendment 62 (1):
    a task's own rate, at 10 or more labelled points)."""
    flagged = sorted({t for t, e in expected.items() if e < per_task} | set(also_flagged))
    fold_sum = {f: sum(expected[t] for t in train) for f, train in folds(sorted(expected)).items()}
    short = sorted(f for f, s in fold_sum.items() if s < per_fold)
    return {"passes": len(flagged) <= max_flagged and not short, "flagged_tasks": flagged,
            "folds_short": short, "fold_expected": fold_sum}


DECISIONS = {"keep": "keep the DAgger round (router and cascader_nospread collectors) and both evaluation rounds "
                     "(amendment 63 (2))",
             "drop": "drop the DAgger round and move its money into Phase 1 points; one evaluation round (router, "
                     "cascader_nospread, cascader_full, the baselines) (amendment 63 (2))",
             "stop": "stop before Phase 1 and re-plan with the user"}


def decision(passes_current: bool, passes_moved: bool) -> str:
    """Amendment 58, set by the user before the pilot ran: stratified projections at the current plan's points per
    task, then at the points per task with the DAgger round's money moved into Phase 1."""
    return "keep" if passes_current else "drop" if passes_moved else "stop"


def weighted_rate(values: Sequence[float], inclusion_probs: Sequence[float]) -> dict[str, float]:
    """Σ w·x / Σ w with w = 1/π, π each point's recorded inclusion probability: a rate over the sampled points
    re-weighted to the population they were drawn from (amendment 57 (6)), with the effective sample size
    (Σw)²/Σw²."""
    if len(values) != len(inclusion_probs) or not values:
        raise PilotError(f"{len(values)} values and {len(inclusion_probs)} inclusion probabilities")
    if not all(0 < p <= 1 for p in inclusion_probs):
        raise PilotError("inclusion probabilities must be in (0, 1]")
    w = [1 / p for p in inclusion_probs]
    sw = sum(w)
    return {"rate": sum(wi * x for wi, x in zip(w, values)) / sw, "n": len(values),
            "ess": sw * sw / sum(wi * wi for wi in w)}


def cell_counts(points: Sequence[Mapping], rng: Optional[random.Random] = None) -> dict[tuple[int, str], tuple[int, int]]:
    """(positives, points) per (task, parent outcome); with `rng`, one bootstrap replicate that resamples parent
    episodes with replacement within each cell (points of one episode are not independent)."""
    groups: dict[tuple[int, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for p in points:
        groups[(p["task_id"], p["parent_outcome"])][p["episode_id"]].append(int(p["positive"]))
    out = {}
    for k in sorted(groups):
        eps = groups[k]
        keys = sorted(eps)
        pick = keys if rng is None else [rng.choice(keys) for _ in keys]
        out[k] = (sum(sum(eps[e]) for e in pick), sum(len(eps[e]) for e in pick))
    return out


def cluster_rates(points: Sequence[Mapping], rng: Optional[random.Random] = None) -> dict[tuple[int, str], float]:
    """Positive rate per (task, parent outcome) (a bootstrap replicate with `rng`, as `cell_counts`)."""
    return {k: pos / n if n else 0.0 for k, (pos, n) in cell_counts(points, rng).items()}


OUTCOMES = ("failure", "success")


def pooled_rates(cells: Mapping[tuple[int, str], tuple[int, int]]) -> dict[str, float]:
    """Amendment 62 (1): each parent-outcome stratum's positive rate pooled across tasks, positives over labelled
    points of that outcome, every task's points together."""
    out = {}
    for o in OUTCOMES:
        pos = sum(c[0] for (t, oo), c in cells.items() if oo == o)
        n = sum(c[1] for (t, oo), c in cells.items() if oo == o)
        out[o] = pos / n if n else 0.0
    return out


MIN_OWN_POINTS = 10     # a task's own rates flag it only from 10 labelled points of the basis below
OWN_BASIS = "failure"   # amendment 63 (5): its labelled failed-parent points ("all": amendment 62 (1), every point)


def shrunk_rates(cells: Mapping[tuple[int, str], tuple[int, int]], *, mus: Optional[Sequence[float]] = None,
                 kappas: Optional[Sequence[float]] = None) -> dict[str, Any]:
    """Amendment 63 (5): an empirical-Bayes estimate of each task's positive rate in each parent-outcome stratum. A
    beta-binomial across tasks (mean mu, concentration kappa = alpha + beta) is fitted to the (positives, points) of
    every task by maximum marginal likelihood on a grid, and each task gets its posterior mean
    (k_t + mu kappa) / (n_t + kappa). Points count as independent here (the pooled projection's bootstrap carries
    the episode clustering). A fit at the grid's largest kappa means no between-task spread is detected: the
    estimates are then the pooled rate."""
    mus = list(mus or [i / 400 for i in range(1, 400)])
    kappas = list(kappas or [10 ** (e / 20) for e in range(-20, 101)])          # 0.1 .. 1e5

    def lbeta(a: float, b: float) -> float:
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    fit: dict[str, Any] = {}
    rates: dict[tuple[int, str], float] = {}
    for o in OUTCOMES:
        data = [(t, k, n) for (t, oo), (k, n) in sorted(cells.items()) if oo == o and n > 0]
        if not data:
            continue
        best = (-math.inf, mus[0], kappas[0])
        for kap in kappas:
            for mu in mus:
                a, b = mu * kap, (1 - mu) * kap
                ll = sum(lbeta(k + a, n - k + b) for _, k, n in data) - len(data) * lbeta(a, b)
                if ll > best[0]:
                    best = (ll, mu, kap)
        _, mu, kap = best
        fit[o] = {"mu": mu, "kappa": kap, "kappa_at_grid_top": kap == kappas[-1],
                  "mu_at_grid_edge": mu in (mus[0], mus[-1])}
        for t, k, n in data:
            rates[(t, o)] = (k + mu * kap) / (n + kap)
    return {"fit": fit, "rates": rates}


def project(points: Sequence[Mapping], strata: Mapping[int, Mapping[str, Any]], budgets: Mapping[str, float], *,
            pools: Mapping[str, int], share: float = 2 / 3, n_boot: int = 2000, seed: int = 0,
            min_own: int = MIN_OWN_POINTS, own_basis: str = OWN_BASIS) -> dict[str, Any]:
    """Expected positives per task and fold under both schemes at each budget (points per task), with cluster-
    bootstrap 95% intervals and the share of replicates that pass the stop rule. `points`: one dict per labelled
    pilot point (task_id, parent_outcome, episode_id, positive). `strata[t]`: phi and points_failed (Phase 1's
    eligibility). `pools[budget]`: parent pools the budget draws from (base, exploration, DAgger), each assumed to
    offer as many failed-parent points as the base rollouts (exploration and DAgger parents are projected with base
    rates: none exist yet).

    Amendment 62 (1), set before the results: the projections the stop rule runs on apply the stratum rates pooled
    across tasks to each task's own failure rate (phi, failed points available) and budget. Each task's projection
    from its own rates is shown beside them (`own_expected`), and flags the task only if it has at least `min_own`
    labelled points of `own_basis`: its failed-parent points (amendment 63 (5)) or all its points (amendment 62 (1)).
    Every replicate resamples episodes within each (task, outcome) cell, then pools. Beside them, not deciding
    (amendment 63 (5): the user decides if it would change the decision), the projection from `shrunk_rates`."""
    if own_basis not in ("failure", "all"):
        raise PilotError(f"own_basis {own_basis!r}: 'failure' (amendment 63) or 'all' (amendment 62)")
    tasks = sorted(strata)
    n_task = {t: sum(1 for p in points if p["task_id"] == t) for t in tasks}
    n_failed = {t: sum(1 for p in points if p["task_id"] == t and p["parent_outcome"] == "failure") for t in tasks}
    judged = [t for t in tasks if (n_failed if own_basis == "failure" else n_task)[t] >= min_own]

    def exp_for(rate_of, b, scheme):
        n = budgets[b]
        return {t: expected_positives(n, rate_of(t, "failure"), rate_of(t, "success"), scheme=scheme,
                                      phi=strata[t]["phi"], share=share,
                                      failed_available=pools[b] * strata[t]["points_failed"]) for t in tasks}

    def one(cells):
        own = {k: pos / n if n else 0.0 for k, (pos, n) in cells.items()}
        pooled = pooled_rates(cells)
        res = {}
        for b in budgets:
            for scheme in ("uniform", "stratified"):
                res[(b, scheme)] = (exp_for(lambda t, o: pooled[o], b, scheme),
                                    exp_for(lambda t, o: own.get((t, o), 0.0), b, scheme))
        return res, own, pooled

    def rule(pair):
        exp, own_exp = pair
        return stop_rule(exp, also_flagged=[t for t in judged if own_exp[t] < 20])

    cells = cell_counts(points)
    point, own_rates, pooled = one(cells)
    shrink = shrunk_rates(cells)
    rng = random.Random(seed)
    boots = [one(cell_counts(points, rng)) for _ in range(n_boot)]

    def q(vals):
        vals = sorted(vals)
        return (float(vals[int(0.025 * (len(vals) - 1))]), float(vals[int(math.ceil(0.975 * (len(vals) - 1)))]))
    out: dict[str, Any] = {}
    for key, pair in point.items():
        b, scheme = key
        exp, own_exp = pair
        r = rule(pair)
        out[f"{b}/{scheme}"] = {
            "points_per_task": budgets[b], "expected": exp,
            "expected_ci95": {t: q([bt[0][key][0][t] for bt in boots]) for t in tasks},
            "fold_ci95": {f: q([sum(bt[0][key][0][u] for u in train) for bt in boots])
                          for f, train in folds(tasks).items()},
            **r, "flagged_by_own_rate": sorted(t for t in judged if own_exp[t] < 20),
            "own_expected": own_exp,
            "own_expected_ci95": {t: q([bt[0][key][1][t] for bt in boots]) for t in tasks},
            "bootstrap_pass_share": sum(rule(bt[0][key])["passes"] for bt in boots) / len(boots)}
        sh = exp_for(lambda t, o: shrink["rates"].get((t, o), pooled[o]), b, scheme)
        sr = stop_rule(sh)
        out[f"{b}/{scheme}"].update({"shrinkage_expected": sh, "shrinkage_flagged": sr["flagged_tasks"],
                                     "shrinkage_folds_short": sr["folds_short"], "shrinkage_passes": sr["passes"]})
    out["pooled_rates"] = {o: {"rate": pooled[o], "positives": sum(c[0] for (t, oo), c in cells.items() if oo == o),
                               "n": sum(c[1] for (t, oo), c in cells.items() if oo == o),
                               "ci95": q([bt[2][o] for bt in boots])} for o in OUTCOMES}
    out["rates"] = {f"t{t}/{o}": {"rate": own_rates[(t, o)], "positives": cells[(t, o)][0], "n": cells[(t, o)][1],
                                  "ci95": q([bt[1].get((t, o), 0.0) for bt in boots])}
                    for (t, o) in sorted(cells)}
    out["shrinkage"] = {"fit": shrink["fit"], "rates": {f"t{t}/{o}": r for (t, o), r in sorted(shrink["rates"].items())}}
    out["labelled_per_task"] = n_task
    out["labelled_failed_per_task"] = n_failed
    out["own_basis"] = own_basis
    out["own_rate_judged_tasks"] = judged
    return out


# ---------------------------------------------------------------- the report (Mac, after the pilot)

COMPARED = ("b2_one_call", "b3_three_calls", "b4_max")
TIME_TASKS = (4, 8)          # π0.5's successful parity episodes overran the limit only there (§4)


def _gain(labels: Mapping[str, Mapping[str, Any]], b: str, prof: str):
    """p_b - p_1 as an exact fraction under `prof`, or None if either label is missing (profile excluded)."""
    from fractions import Fraction
    x, y = labels.get(b, {}).get(prof), labels.get("b1_continue", {}).get(prof)
    return None if x is None or y is None else Fraction(x.successes, x.seeds) - Fraction(y.successes, y.seeds)


def point_rows(runs: Sequence[Mapping], events: Sequence[Mapping], label_map: Mapping[str, Mapping], *,
               limits_s: Mapping[int, float], charges, profiles: Optional[Sequence[str]] = None,
               stop_profile: str = STOP_PROFILE) -> tuple[list[dict], list[str]]:
    """One row per pilot point: its stratum and inclusion probability, p_b per branch and profile (as
    `vlar_harness.labels.point_labels` scores them from the base parent's log and the pilot's branch logs), the
    excluded profiles, and `positive` (p_2 - p_1 >= 1/3 under `stop_profile`, §4: cascader_full in amendment 58 as the
    pilot report applied it, cascader_nospread since amendment 63; False where that profile is excluded, which
    `positive_scored` records), and `positive_by` (the same test under every profile; None where excluded). Points
    that cannot be labelled are listed, not guessed."""
    from vlar_harness import labels as L
    from vlar_harness.config import PROFILES
    from vlar_train.data import group_events, prefix_of
    profiles = tuple(profiles or PROFILES)
    logs = group_events(events)
    own: dict[str, dict[tuple[str, int], list]] = defaultdict(dict)
    for r in runs:
        if r.get("snapshot_id") in label_map and r.get("branch"):
            own[r["snapshot_id"]][(r["branch"], int(r["seed_index"]))] = logs.get(r["run_id"], [])
    rows, problems = [], []
    for sid, entry in sorted(label_map.items()):
        p, sel = entry["point"], entry["selection"]
        parent = logs.get(p["episode_id"])
        if parent is None:
            problems.append(f"{sid}: no event log for the parent {p['episode_id']}")
            continue
        try:
            pl = L.point_labels(prefix_of(parent, p["index"]), entry, own.get(sid, {}),
                                seeds=int(entry["settings"]["seeds"]), charges=charges,
                                limit_s=float(limits_s[int(p["task_id"])]),
                                parent_log=parent if entry.get("parent_seeds") else None, profiles=profiles)
        except (L.LabelError, KeyError) as e:
            problems.append(f"{sid}: {e}")
            continue
        g = _gain(pl.labels, "b2_one_call", stop_profile)
        pos_by = {prof: (None if (gp := _gain(pl.labels, "b2_one_call", prof)) is None else bool(gp >= POSITIVE_EXACT))
                  for prof in profiles}
        rows.append({"snapshot_id": sid, "task_id": int(p["task_id"]), "episode_id": p["episode_id"],
                     "index": int(p["index"]), "parent_outcome": sel["stratum"]["parent_outcome"],
                     "inclusion_prob": float(sel["inclusion_prob"]), "excluded": list(pl.excluded),
                     "gain": {b: {prof: (None if (d := _gain(pl.labels, b, prof)) is None else float(d))
                                  for prof in profiles} for b in COMPARED if b in pl.labels},
                     "p": {b: {prof: lab.value for prof, lab in d.items()} for b, d in pl.labels.items()},
                     "positive": bool(g is not None and g >= POSITIVE_EXACT), "positive_scored": g is not None,
                     "positive_by": pos_by})
    return rows, problems


def _frac_third():
    from fractions import Fraction
    return Fraction(1, 3)


POSITIVE_EXACT = _frac_third()   # POSITIVE as an exact fraction: labels are k/3, so the test must not round


def rate_table(rows: Sequence[Mapping], profiles: Sequence[str]) -> dict[str, Any]:
    """Per task (and all tasks): rescue (p_b > p_1), harm (p_b < p_1) and mean p_b - p_1 for b2, b3, b4 against b1
    under each profile, re-weighted to the eligible population (w = 1/π, amendment 57 (6)), with n and the
    effective sample size; positives per stratum unweighted (the projection's inputs); and, per task, the rescues
    under `none` that cascader_full's cut loses (§4's tasks 4 and 8 apart)."""
    out: dict[str, Any] = {}
    groups = {f"t{t}": [r for r in rows if r["task_id"] == t] for t in sorted({r["task_id"] for r in rows})}
    groups["all"] = list(rows)
    groups["all_but_4_8"] = [r for r in rows if r["task_id"] not in TIME_TASKS]
    for name, rs in groups.items():
        g: dict[str, Any] = {"n": len(rs)}
        for b in COMPARED:
            for prof in profiles:
                sc = [r for r in rs if r["gain"].get(b, {}).get(prof) is not None]
                if not sc:
                    continue
                pi = [r["inclusion_prob"] for r in sc]
                d = [r["gain"][b][prof] for r in sc]
                g[f"{b}/{prof}"] = {"rescue": weighted_rate([x > 0 for x in d], pi)["rate"],
                                    "harm": weighted_rate([x < 0 for x in d], pi)["rate"],
                                    "mean_gain": weighted_rate(d, pi)["rate"], **{k: v for k, v in
                                    weighted_rate(d, pi).items() if k != "rate"}}
        for o in ("failure", "success"):
            st = [r for r in rs if r["parent_outcome"] == o]
            g[f"positive_{o}"] = {"n": len(st), "positives": sum(r["positive"] for r in st),
                                  "unscored": sum(not r["positive_scored"] for r in st)}
            g[f"positive_{o}_by_profile"] = {
                prof: {"positives": sum(bool(r.get("positive_by", {}).get(prof)) for r in st),
                       "unscored": sum(r.get("positive_by", {}).get(prof) is None for r in st)}
                for prof in profiles}
        pos = [r for r in rs if r["positive_scored"]]
        if pos:
            g["positive_reweighted"] = weighted_rate([r["positive"] for r in pos], [r["inclusion_prob"] for r in pos])
        lost = [r for r in rs if (r["gain"].get("b2_one_call", {}).get("none") or 0) > 0
                and not ((r["gain"]["b2_one_call"].get(FULL_PROFILE) or 0) > 0)]
        g["rescues_lost_to_cascader_full_cut"] = {"n": len(lost), "excluded_at_the_point": sum(
            FULL_PROFILE in r["excluded"] for r in lost)}
        out[name] = g
    return out


def costs(markers: Sequence[Mapping], timing: Sequence[Mapping], *, hourly_usd: float) -> dict[str, Any]:
    """VM-s per run by branch type: each branch window's VM seconds (timing.jsonl, phase start to end, per boot)
    apportioned by job wall time (claimed_at to finished_at) among the jobs that finished in it; and per point,
    with and without b3, base parent (b1 needs 2 seeds) and exploration parent (3), and b4's share. The boot, the
    golden check and the pilot's checks are outside the windows and reported apart by the caller."""
    jobs = [{"kind": m["job"]["kind"], "wall": float(m["finished_at"]) - float(m["meta"]["claimed_at"]),
             "end": float(m["finished_at"]), "start": float(m["meta"]["claimed_at"]), "boot": m["meta"].get("boot_id"),
             "steps": (m.get("result") or {}).get("summary", {}).get("steps")}
            for m in markers if m.get("meta", {}).get("claimed_at") is not None]
    windows = branch_windows(timing, jobs)
    vm_s: dict[str, float] = defaultdict(float)
    worker_s: dict[str, float] = defaultdict(float)
    steps: dict[str, int] = defaultdict(int)
    n: dict[str, int] = defaultdict(int)
    for a, b, boot, _ in windows:
        inside = [j for j in jobs if (j["boot"] == boot if boot is not None and j["boot"] is not None
                                      else a <= j["end"] <= b)]
        total = sum(j["wall"] for j in inside)
        for j in inside:
            vm_s[j["kind"]] += (b - a) * j["wall"] / total if total else 0.0
            worker_s[j["kind"]] += j["wall"]
            steps[j["kind"]] += int(j["steps"] or 0)
            n[j["kind"]] += 1
    per_run = {k: vm_s[k] / n[k] for k in n}
    r = lambda k: per_run.get(k, float("nan"))                                             # noqa: E731
    point = {"base_with_b3": 2 * r("b1_continue") + 3 * (r("b2_one_call") + r("b3_three_calls") + r("b4_max")),
             "base_without_b3": 2 * r("b1_continue") + 3 * (r("b2_one_call") + r("b4_max")),
             "explore_without_b3": 3 * (r("b1_continue") + r("b2_one_call") + r("b4_max"))}
    usd = {k: v * hourly_usd / 3600 for k, v in point.items()}
    return {"windows_s": [b - a for a, b, _, _ in windows], "window_ends": [src for *_, src in windows],
            "jobs_timed": sum(n.values()), "jobs_untimed": len(markers) - len(jobs), "vm_s_per_run": per_run,
            "worker_s_per_run": {k: worker_s[k] / n[k] for k in n},
            "steps_per_run": {k: steps[k] / n[k] for k in n},
            "vm_s_per_step": {k: vm_s[k] / steps[k] for k in n if steps[k]}, "usd_per_run": {k: v * hourly_usd / 3600
                                                                                    for k, v in per_run.items()},
            "runs": dict(n), "vm_s_per_point": point, "usd_per_point": usd,
            "b4_share_of_base_point": 3 * r("b4_max") / point["base_without_b3"] if point["base_without_b3"] else None}


def costs_by_boot(markers: Sequence[Mapping], timing: Sequence[Mapping], *, hourly_usd: float) -> dict[str, Any]:
    """Amendment 62 (1): costs from the resumed boot's jobs only. The first boot (the earliest first claim) is costed
    on its own and reported apart; every later boot's jobs, each in its own window, give the costs the measured
    budgets use (the pilot's resume is one boot; a preemption would add one, listed in `later_boots`)."""
    first_claim: dict[Optional[str], float] = {}
    for m in markers:
        c = m.get("meta", {}).get("claimed_at")
        if c is not None:
            b = m["meta"].get("boot_id")
            first_claim[b] = min(first_claim.get(b, float(c)), float(c))
    boots = sorted((b for b in first_claim if b is not None), key=lambda b: first_claim[b])
    if not boots:
        return {"first_boot": None, "later_boots": [], "first": None, "resumed": costs(markers, timing,
                                                                                       hourly_usd=hourly_usd)}
    first = boots[0]
    later = boots[1:]

    def pick(bs):
        return ([m for m in markers if m.get("meta", {}).get("boot_id") in bs],
                [t for t in timing if t.get("boot_id") in bs])
    return {"first_boot": first, "later_boots": later,
            "untagged_jobs": sum(1 for m in markers if m.get("meta", {}).get("boot_id") is None),
            "first": costs(*pick({first}), hourly_usd=hourly_usd),
            "resumed": costs(*pick(set(later)), hourly_usd=hourly_usd) if later else None}


def check_timing(iii: Mapping[str, Any]) -> dict[str, Any]:
    """The corrected check (iii)'s record-off episodes as screened runs, alone on the servers: per episode its load
    count, steps and seconds (search, warm-up, episode; reports from bb7112f have no per-episode seconds)."""
    rows = [{"subject": r["subject"], "loads": r.get("loads"), "steps": (r.get("outcome") or {}).get("steps"),
             "seconds": r.get("seconds")} for r in iii.get("record_off", [])]
    timed = [r for r in rows if r["seconds"] and "episode" in r["seconds"] and r["steps"]]
    out: dict[str, Any] = {"episodes": rows, "timed": len(timed)}
    if timed:
        out["episode_s_per_step"] = sum(r["seconds"]["episode"] for r in timed) / sum(r["steps"] for r in timed)
        out["warm_up_s_per_load"] = (sum(r["seconds"]["warm_up"] for r in timed)
                                     / max(1, sum(r["loads"] - 2 for r in timed)))
        out["search_s_per_episode"] = sum(r["seconds"]["search"] for r in timed) / len(timed)
    return out


def fit_table(summary: Mapping[str, Any], counts: Sequence[int] = (6, 7, 8), *, usable: float = 0.95,
              server_margin: float = 1.25, host_reserve: int = 4 * 1024 ** 3) -> dict[str, Any]:
    """Would `counts` workers at the run's per-worker limit fit (the user, 2026-11-24: the exploration, DAgger and
    evaluation estimates assumed 8-worker throughput)? Amendment 62 (4): a count fits only if its per-worker limits
    plus the servers' measured peaks fit inside RAM (`fits`, with the headroom left). Shown beside it, not deciding:
    the boot-time sizing rule (servers' resident memory at start x 1.25, a 4 GiB host reserve, 95% of RAM), and every
    worker at the largest measured worker peak."""
    sz, peaks = summary["sizing"], summary.get("peak_memory_bytes") or {}
    mem, start, limit = int(sz["mem_total"]), int(sz["servers"]), int(sz["limit"])
    servers = {k: v for k, v in peaks.items() if k in ("vlar-xvla", "vlar-pi05")}
    srv = sum(servers.values()) or start
    wk = max((v for k, v in peaks.items() if k.startswith("vlar-worker")), default=None)
    rows = {}
    for n in counts:
        rule, full = n * limit + start * server_margin + host_reserve, n * limit + srv
        row = {"fits": full <= mem, "limits_at_server_peak_bytes": full, "headroom_bytes": mem - full,
               "rule_bytes": rule, "rule_fits": rule <= mem * usable}
        if wk is not None:
            row.update({"worker_peaks_bytes": n * wk + srv + host_reserve,
                        "worker_peaks_fit_ram": n * wk + srv + host_reserve <= mem})
        rows[n] = row
    return {"mem_total": mem, "limit": limit, "servers_at_start": start, "servers_peak": srv, "server_peaks": servers,
            "worker_peak_max": wk, "max_workers_fit": int((mem - srv) // limit),
            "rule_max_workers": int((mem * usable - start * server_margin - host_reserve) // limit),
            "counts": rows}


def branch_windows(timing: Sequence[Mapping], jobs: Sequence[Mapping]) -> list[tuple[float, float, Optional[str], str]]:
    """Per boot: (start, end, boot_id, how the end is known) of the branch phase. The start is timing.jsonl's
    phase_started (else the boot's first claim); the end its phase_ended, else the boot's last job finish: a step
    stopped inside `run_phase` writes no phase_ended (the user, 2026-11-24). Jobs without a boot id (older markers)
    count in windows from timing.jsonl only."""
    start, end = {}, {}
    for t in timing:
        if "phase_started" in t:
            start.setdefault(t["boot_id"], float(t["phase_started"]))
        elif "phase_ended" in t:
            end[t["boot_id"]] = float(t["phase_ended"])
    by_boot: dict[str, list[Mapping]] = defaultdict(list)
    for j in jobs:
        if j.get("boot") is not None:
            by_boot[j["boot"]].append(j)
    out = []
    for boot in sorted(set(start) | set(by_boot)):
        js = by_boot.get(boot, [])
        a = start.get(boot, min((j["start"] for j in js), default=None))
        b, src = (end[boot], "phase_ended") if boot in end else (max((j["end"] for j in js), default=None),
                                                                 "last_job_finish")
        if a is not None and b is not None and b > a:
            out.append((a, b, boot if js else None, src))
    return out


def declared_budgets(hourly_usd: float) -> dict[str, float]:
    """Amendment 58's two budgets in points per task at the plan's estimated prices (amendment 61: ~194 and ~227)."""
    est = {"base_without_b3": 8 * EST_VM_S_PER_RUN * hourly_usd / 3600,
           "explore_without_b3": 9 * EST_VM_S_PER_RUN * hourly_usd / 3600}
    return dict(measured_budgets(est, hourly_usd=hourly_usd))


# SPEC §17 amendment 61 (the user, 2026-11-24): evaluation after the DAgger round on another account; its $15.23
# goes to labels (2:1), reserve recomputed. Amendment 58's budgets: Phase 1 and DAgger labels with the DAgger round
# ("current"); the DAgger round's rollouts and labels moved into Phase 1 ("moved"). Was 3.24 / 1.62 / 6.04.
PLAN_USD = {"phase1": 13.4048, "dagger_branch": 6.7024, "dagger_round": 3.14}
EST_VM_S_PER_RUN = 7.0


def measured_budgets(c: Mapping[str, float], *, rerun_vm_s: float = 74.0, hourly_usd: float,
                     phase1_usd: float = PLAN_USD["phase1"], dagger_branch_usd: float = PLAN_USD["dagger_branch"],
                     dagger_round_usd: float = PLAN_USD["dagger_round"], tasks: int = 10) -> dict[str, float]:
    """Points per task the prices buy (PROPOSALS §0.1 (e), amendment 61): Phase 1 n base and n exploration points,
    n = Phase 1 $ / (c_b + c_e + r/2) (an exploration re-run serves k = 2 points); DAgger m points in each half,
    m = DAgger label $ / (2 c_e + r/2) (the router half's re-runs); moved: the DAgger round's coupon money (its
    rollouts) and its label money into Phase 1 points at Phase 1's price. r: the re-run's cost (74 VM-s, est.)."""
    r = rerun_vm_s * hourly_usd / 3600
    cb, ce = c["base_without_b3"], c["explore_without_b3"]
    n = phase1_usd / (cb + ce + r / 2)
    m = dagger_branch_usd / (2 * ce + r / 2)
    moved = (phase1_usd + dagger_branch_usd + dagger_round_usd) / (cb + ce + r / 2)
    return {"current": (2 * n + 2 * m) / tasks, "moved": 2 * moved / tasks}


def restart_overhead(markers: Sequence[Mapping], *, gap_s: float = 20.0) -> dict[str, Any]:
    """Idle time between one job's end and the same worker's next claim: the small gaps are claiming, the large
    ones (> gap_s) recycles and restarts (container start, imports), the overhead task-affine claiming would cut."""
    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for m in markers:
        meta = m.get("meta", {})
        if meta.get("claimed_at") is not None and meta.get("worker_id"):
            by[meta["worker_id"]].append((float(meta["claimed_at"]), float(m["finished_at"])))
    gaps = []
    for js in by.values():
        js.sort()
        gaps += [b[0] - a[1] for a, b in zip(js, js[1:])]
    big = [g for g in gaps if g > gap_s]
    small = sorted(g for g in gaps if g <= gap_s)
    return {"workers": len(by), "gaps": len(gaps), "large_gaps": len(big), "large_gap_s_total": sum(big),
            "large_gap_s_median": sorted(big)[len(big) // 2] if big else None,
            "small_gap_s_median": small[len(small) // 2] if small else None}


def report(root: Path, plan_dir: Path, reports_dir: Path, *, hourly_usd: float, phase_vm_usd: Optional[float] = None,
           cfg=None, n_boot: int = 2000, stop_profile: str = STOP_PROFILE, own_basis: str = OWN_BASIS) -> dict[str, Any]:
    """The pilot report (§4, §0.4, amendment 58) from local copies: <root>/records/{runs,events}/{base,pilot}/,
    <root>/results/pilot/ (markers), the plan directory and <storage>/pilot/reports/. `docs/PILOT-REPORT-2026-11-25.md`
    is `stop_profile=PILOT_RULE_PROFILE, own_basis="all"` (amendments 58 and 62 as set before the results); the
    defaults are amendment 63's."""
    from vlar_harness import clock as clk
    from vlar_harness.config import PROFILES, load_config
    from vlar_harness.dataset import read_table
    cfg = cfg or load_config()
    if not getattr(cfg, "frozen", False):
        raise PilotError("labels are scored with the frozen calibration file (set VLAR_CALIBRATION to v1.0's)")
    limits = {int(k): float(v) for k, v in cfg["time_limit"]["per_task_s"].items()}
    label_map = json.loads((Path(plan_dir) / "labels.json").read_text())
    selection = json.loads((Path(plan_dir) / "selection.json").read_text())
    rows, problems = point_rows(read_table(root, "runs").to_pylist(), read_table(root, "events").to_pylist(),
                                label_map, limits_s=limits, charges=clk.Charges.from_config(cfg),
                                stop_profile=stop_profile)
    markers = [json.loads(f.read_text()) for f in sorted((Path(root) / "results" / PHASE).glob("*.json"))]
    rd = Path(reports_dir)
    read_jsonl = lambda f: [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []  # noqa
    by_boot = costs_by_boot(markers, read_jsonl(rd / "timing.jsonl"), hourly_usd=hourly_usd)
    cost = by_boot["resumed"] or by_boot["first"]          # one boot only: the first is all there is
    budgets = declared_budgets(hourly_usd)
    if by_boot["resumed"] is not None and all(math.isfinite(v) for v in cost["usd_per_point"].values()):
        mb = measured_budgets(cost["usd_per_point"], hourly_usd=hourly_usd)
        budgets.update({"measured_current": mb["current"], "measured_moved": mb["moved"]})
    pools = {k: (3 if k.endswith("current") else 2) for k in budgets}
    strata = {int(t): v for t, v in selection["phase1_strata"].items()}
    proj = project(rows, strata, budgets, pools=pools, n_boot=n_boot, own_basis=own_basis) if rows else {}
    dec = {"rule": decision(proj["current/stratified"]["passes"], proj["moved/stratified"]["passes"])} if rows else {}
    if "measured_current/stratified" in proj:
        dec["at_measured_prices"] = decision(proj["measured_current/stratified"]["passes"],
                                             proj["measured_moved/stratified"]["passes"])
        dec["differs"] = dec["at_measured_prices"] != dec["rule"]
    sup = json.loads((rd / "pilot-supervisor.json").read_text()) if (rd / "pilot-supervisor.json").exists() else {}
    checks = json.loads((rd / "checks.json").read_text()) if (rd / "checks.json").exists() else None
    iii = json.loads((rd / "checks-iii.json").read_text()) if (rd / "checks-iii.json").exists() else None
    return {"plan_id": selection["plan_id"], "points": len(label_map), "labelled": len(rows), "problems": problems,
            "jobs_done": len(markers), "jobs_planned": selection["n_jobs"], "rates": rate_table(rows, PROFILES),
            "costs": cost, "costs_basis": "resumed boot" if by_boot["resumed"] is not None else "first boot only",
            "costs_first_boot": by_boot["first"], "boots": {"first": by_boot["first_boot"],
                                                             "later": by_boot["later_boots"]},
            "phase_vm_usd": phase_vm_usd, "budgets_points_per_task": budgets, "projections": proj,
            "decision": dec, "decision_text": DECISIONS.get(dec.get("rule", ""), "no labelled points"),
            "checks": None if checks is None else {
                "passed": checks["passed"], "features_equal_probe": checks["features"]["features_equal_probe"],
                "sample_reproduces_record": checks["features"]["sample_reproduces_record"],
                "xvla_only": [r["passed"] for r in checks["xvla_only"]],
                "record_off": [r["passed"] for r in checks["record_off"]],
                "consequence": checks["features"]["consequence"]},
            "check_iii_corrected": None if iii is None else {
                "passed": iii["passed"], "record_off": [{k: r.get(k) for k in ("subject", "loads", "passed", "problems",
                                                                               "fixtures_equal_base", "outcome")}
                                                        for r in iii["record_off"]]},
            "memory_fit": fit_table(sup) if sup.get("sizing") else None,
            "check_iii_timing": check_timing(iii) if iii else None,
            "supervisor": {k: sup.get(k) for k in ("sizing", "peak_memory_bytes", "restarts", "recycles", "refused",
                                                   "gave_up",
                                                   "counts")},
            "supervisor_events": {e: sum(1 for x in sup.get("events", []) if x.get("outcome") == e)
                                  for e in sorted({x.get("outcome") for x in sup.get("events", [])})},
            "restart_overhead": restart_overhead(markers),
            "stop_profile": stop_profile, "own_basis": own_basis,
            "notes": ["Amendment 62 (logged 2026-11-24 22:34Z, before any result of the resumed boot was downloaded "
                      "or read): the stop rule runs on stratum rates pooled across tasks, applied to each task's own "
                      "failure rate and budget; costs and measured-price budgets come from the resumed boot's jobs "
                      "only (the first boot's apart); a worker count fits only if its limits plus the servers' peaks "
                      "fit inside RAM.",
                      f"Positives scored under {stop_profile} (amendment 58: cascader_full; amendment 63: "
                      f"cascader_nospread); a task's own rates flag it only from {MIN_OWN_POINTS} labelled "
                      + ("failed-parent points (amendment 63 (5))." if own_basis == "failure"
                         else "points (amendment 62 (1))."),
                      "Exploration and DAgger parents are projected with base-parent rates (none exist yet).",
                      "The pilot's eligible points end at index 49 (b3's three calls); Phase 1's at 51.",
                      "The re-run cost r = 74 VM-s is an estimate; the pilot does not measure it."]}
