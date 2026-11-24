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
POSITIVE = 1 / 3                 # p_2 - p_1 >= 1/3 under cascader_full's profile (§4)
STOP_PROFILE = "cascader_full"


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


def make_plan(root: Path, markers_dir: Path, out_dir: Path, *, cfg=None) -> dict[str, Any]:
    from vlar_harness.config import load_config
    from vlar_harness.dataset import read_table
    cfg = cfg or load_config()
    step_limit, interval = int(cfg["env"]["step_limit"]), int(cfg["decision"]["interval_steps"])
    points, success, kinds = base_points(read_table(root, "runs").to_pylist(), read_table(root, "events").to_pylist())
    if len(success) != 500:
        raise PilotError(f"{len(success)} base runs, not 500: download all of records/runs/base")
    sel = stratified_selection(points, success, step_limit=step_limit, interval=interval, dp_sampling=SELECTION)
    listed = snapshots_listed(markers_dir)
    missing = sorted(set(sel) - listed)
    if missing:
        raise PilotError(f"{len(missing)} selected snapshots are not listed by any base result marker: {missing[:3]}")
    plan, labels = branch_plan(PHASE, points, branches=cfg["branches"], step_limit=step_limit, interval=interval,
                               seeds=SEEDS, enabled=BRANCH_ORDER, tier_one_seeds=TIER_ONE_SEEDS,
                               order_seed=ORDER_SEED, dp_sampling=SELECTION, selection=sel)
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
                                         min_calls=SELECTION["min_calls"]).items()}
    phase1 = strata_table(points, success, step_limit=step_limit, interval=interval, min_calls=1)
    selection = {"settings": SELECTION, "n_points": len(sel), "n_jobs": len(plan.jobs), "plan_id": plan.plan_id,
                 "strata": strata, "phase1_strata": phase1,
                 "base_episode_success": {t: sum(success[e] for e in success if int(e.split("/t")[1].split("/")[0]) == t)
                                          for t in sorted(strata)}}
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
            "tables": compared, "outcome": {k: run.get(k) for k in ("success", "end_reason", "steps", "success_step")}}


def run_checks(ops_cfg: Mapping[str, Any], plan_dir: Path, staging: Path, *, clients=None, make_sim=None,
               server_wait_s: float = 600.0) -> dict:
    import os
    import sys
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "cgl")
    from vlar_harness import provenance
    from vlar_harness import snapshot as snap
    from vlar_harness.collect import Runner
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
    subjects = json.loads((Path(plan_dir) / "checks.json").read_text())
    episodes = vm_worker.ProductionEpisodes(cfg, clients["xvla"], clients["pi05"], make_sim=make_sim)
    runner = Runner(cfg, episodes, storage.get_bytes, staging, p_escalate=0.0, header=provenance.record_header(cfg))

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

    try:
        rep = {"features": features_check(episodes, read, recorded, clients["xvla"], subjects["features"])}
        for variant in ("xvla_only", "record_off"):
            rep[variant] = [episode_check(lambda j: runner(j)[0], storage.get_bytes, original_job(s), variant)
                            for s in subjects[variant]]
    finally:
        episodes.close()
    rep["passed"] = rep["features"]["sample_reproduces_record"] and all(
        r["passed"] for v in ("xvla_only", "record_off") for r in rep[v])
    return {**rep, "boot_id": golden.boot_id(), "calibration_hash": cfg.calibration_hash, "image_check": image}


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
              max_flagged: int = 3) -> dict[str, Any]:
    """§4: a task expecting fewer than 20 positives is flagged; more than 3 flagged tasks, or a training fold (8 tasks)
    expecting fewer than 100, stops before Phase 1."""
    flagged = sorted(t for t, e in expected.items() if e < per_task)
    fold_sum = {f: sum(expected[t] for t in train) for f, train in folds(sorted(expected)).items()}
    short = sorted(f for f, s in fold_sum.items() if s < per_fold)
    return {"passes": len(flagged) <= max_flagged and not short, "flagged_tasks": flagged,
            "folds_short": short, "fold_expected": fold_sum}


DECISIONS = {"keep": "keep the DAgger round (the current plan)",
             "drop": "drop the DAgger round and its post-DAgger evaluation; move its money into Phase 1 points; "
                     "evaluate cascader_nospread in the single round",
             "stop": "stop before Phase 1 and re-plan with the user"}


def decision(passes_current: bool, passes_moved: bool) -> str:
    """Amendment 58, set by the user before the pilot ran: stratified projections at the current plan's points per
    task, then at the points per task with the DAgger round's money moved into Phase 1."""
    return "keep" if passes_current else "drop" if passes_moved else "stop"


def cluster_rates(points: Sequence[Mapping], rng: Optional[random.Random] = None) -> dict[tuple[int, str], float]:
    """Positive rate per (task, parent outcome); with `rng`, one bootstrap replicate that resamples parent episodes
    with replacement within each stratum (points of one episode are not independent)."""
    groups: dict[tuple[int, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for p in points:
        groups[(p["task_id"], p["parent_outcome"])][p["episode_id"]].append(int(p["positive"]))
    out = {}
    for k, eps in groups.items():
        keys = sorted(eps)
        pick = keys if rng is None else [rng.choice(keys) for _ in keys]
        pos = sum(sum(eps[e]) for e in pick)
        n = sum(len(eps[e]) for e in pick)
        out[k] = pos / n if n else 0.0
    return out


def project(points: Sequence[Mapping], strata: Mapping[int, Mapping[str, Any]], budgets: Mapping[str, float], *,
            pools: Mapping[str, int], share: float = 2 / 3, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Expected positives per task and fold under both schemes at each budget (points per task), with cluster-
    bootstrap 95% intervals and the share of replicates that pass the stop rule. `points`: one dict per labelled
    pilot point (task_id, parent_outcome, episode_id, positive). `strata[t]`: phi and points_failed (Phase 1's
    eligibility). `pools[budget]`: parent pools the budget draws from (base, exploration, DAgger), each assumed to
    offer as many failed-parent points as the base rollouts (exploration and DAgger parents are projected with base
    rates: none exist yet)."""
    tasks = sorted(strata)

    def one(rates):
        res = {}
        for b, n in budgets.items():
            for scheme in ("uniform", "stratified"):
                exp = {t: expected_positives(n, rates.get((t, "failure"), 0.0), rates.get((t, "success"), 0.0),
                                             scheme=scheme, phi=strata[t]["phi"], share=share,
                                             failed_available=pools[b] * strata[t]["points_failed"])
                       for t in tasks}
                res[(b, scheme)] = exp
        return res
    point = one(cluster_rates(points))
    rng = random.Random(seed)
    boots = [one(cluster_rates(points, rng)) for _ in range(n_boot)]
    out: dict[str, Any] = {}
    for key, exp in point.items():
        b, scheme = key
        rule = stop_rule(exp)
        q = lambda vals: (float(sorted(vals)[int(0.025 * (len(vals) - 1))]),            # noqa: E731
                          float(sorted(vals)[int(math.ceil(0.975 * (len(vals) - 1)))]))
        task_ci = {t: q([bt[key][t] for bt in boots]) for t in tasks}
        fold_ci = {f: q([sum(bt[key][u] for u in train) for bt in boots]) for f, train in folds(tasks).items()}
        out[f"{b}/{scheme}"] = {"points_per_task": budgets[b], "expected": exp, "expected_ci95": task_ci,
                                "fold_ci95": fold_ci, **rule,
                                "bootstrap_pass_share": sum(stop_rule(bt[key])["passes"] for bt in boots) / len(boots)}
    out["rates"] = {f"t{t}/{o}": r for (t, o), r in sorted(cluster_rates(points).items())}
    return out
