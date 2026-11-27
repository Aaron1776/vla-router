"""Training points from the records (SPEC §6, §7; HANDOFF "Decider training on the Mac").

Inputs: a local copy of `records/{runs,events,points}` (`gcloud storage cp -r .../records/<table> ...`), the branch
planner's label map (`vlar_ops.branchplan.branch_plan`: snapshot id -> the point, each branch's call count, the
canonical branch it shares, the seeds the parent provides, and the planner's settings, whose seed count is used here:
never the collection config's, amendment 49) and the frozen calibration (per-task time limits, gripper
thresholds, clock charges). Snapshots and images are not needed: a point's prefix is its parent run's event log up
to and including the point's `decision_point` event, and `vlar_harness.labels.point_labels` checks that it fits.

Per point and per profile: the four branch labels (NaN where the planner ran no such branch), whether the point is
excluded for that profile (its clock already beyond the limit, amendment 22) and the remaining-time input recomputed
under that profile (amendment 23); the measured inputs from the points table, gripper fractions computed offline
(amendment 35).

Per point, its sampling weight (amendment 57 (3)): w_i = 1/π_i, π_i the inclusion probability the planner recorded in
the point's label entry (`selection.inclusion_prob`; for re-run parents P(episode) × P(point | episode)). An entry
planned with the uniform per-point draw has no `selection`; its π_i is the planner's recorded `dp_sampling.fraction`.
The weights are kept un-normalised here; each fold normalises them to mean 1 within its training rows and within its
validation rows (`train.unit_mean`).

Shared run ids (amendment 66 (5)): a screened exploration run (record `off`) and its re-run (record `full`) share a
run id under phase `explore`, and consumers take the record-`full` row. `record_full_runs` keeps that row, with the
amendment's check (no run id twice afterwards, every kept row of a shared id record `full`), and the run's event log
is the kept row's job's part of the events table: `read_events` reads the table part by part and adds each part's
job id (`records/events/<phase>/<job_id>.parquet`), since an events row itself does not say which job wrote it.

Complete points only (§12, as `vlar_ops.plan.complete_points`: training uses only complete points): a labelled point
one of whose branch seeds has no finished run in the records (no log, or a log without its `end` event) is left out
and listed in `PointSet.report`, with the label entries that have no row in the points table; the build goes on with
the others. Every other inconsistency still stops the build (`vlar_harness.labels` checks that the logs fit). Leaving
points out makes the remaining points' recorded inclusion probabilities inexact (amendment 65 (6): "a tranche cut
short leaves a labelled subset without exact inclusion probabilities"), so the report is printed, never silent.

Per point, its parent episode (`PointSet.episode`, the label entry's `point.episode_id`): points from one episode
share its outcome, so reports count and resample by episode (amendments 57 (5) and 68 (3); `stats`). Folds are by
whole task (§7), so an episode never straddles a split."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .deciders import BRANCHES, DECIDERS, DRIFT, SPREAD, STEP_LIMIT


class DataError(ValueError):
    pass


@dataclass
class PointSet:
    cols: dict[str, np.ndarray]              # input columns, (n, k)
    task: np.ndarray                         # (n,) task of the point's episode
    snapshot_id: list[str]
    can_escalate: np.ndarray                 # (n,) budget remains for one call
    remaining_s: dict[str, np.ndarray]       # profile -> (n,)
    y: dict[str, np.ndarray]                 # profile -> (n, 4) labels, NaN where a branch was not run
    excluded: dict[str, np.ndarray]          # profile -> (n,)
    weight: Optional[np.ndarray] = None      # (n,) 1/π_i (amendment 57 (3)), un-normalised; all 1 when not given
    episode: Optional[np.ndarray] = None     # (n,) parent episode id (amendment 68 (3)); from the snapshot id if absent
    report: dict = field(default_factory=dict)   # what the build left out and why (`build`); empty for synthetic sets

    def __post_init__(self) -> None:
        if self.episode is None:               # snapshot id = <episode_id>/dpNN (datacard, "Identifiers and seeds")
            self.episode = [s.rsplit("/", 1)[0] for s in self.snapshot_id]
        self.episode = np.asarray(self.episode, dtype=str)
        if self.episode.shape != (len(self.task),):
            raise DataError("one parent episode id per point (amendment 68 (3))")
        if self.weight is None:
            self.weight = np.ones(len(self.task), dtype=np.float64)
        self.weight = np.asarray(self.weight, dtype=np.float64)
        if self.weight.shape != (len(self.task),) or not (np.isfinite(self.weight) & (self.weight > 0)).all():
            raise DataError("sampling weights must be one finite positive number per point (amendment 57)")

    def __len__(self) -> int:
        return len(self.task)

    def targets(self, decider: str) -> tuple[np.ndarray, np.ndarray]:
        y = self.y[decider]
        m = ~np.isnan(y)
        return np.where(m, y, 0.0), m.astype(np.float32)

    def subset(self, rows: np.ndarray) -> "PointSet":
        idx = np.flatnonzero(rows) if rows.dtype == bool else rows
        return PointSet({k: v[idx] for k, v in self.cols.items()}, self.task[idx],
                        [self.snapshot_id[i] for i in idx], self.can_escalate[idx],
                        {p: v[idx] for p, v in self.remaining_s.items()}, {p: v[idx] for p, v in self.y.items()},
                        {p: v[idx] for p, v in self.excluded.items()}, self.weight[idx], self.episode[idx],
                        dict(self.report))


def prefix_of(parent_events: Sequence[Mapping], index: int) -> list[Mapping]:
    """The parent's log up to and including decision point `index`'s event."""
    for k, e in enumerate(parent_events):
        if e.get("t") == "decision_point" and int(e.get("index", -1)) == int(index):
            return list(parent_events[:k + 1])
    raise DataError(f"no decision_point event with index {index} in the parent's log")


def group_events(events: Sequence[Mapping]) -> dict[str, list[dict]]:
    rows: dict[str, list] = defaultdict(list)
    for e in events:
        rows[e["run_id"]].append(e)
    out = {}
    for rid, rs in rows.items():
        rs.sort(key=lambda e: int(e["seq"]))
        if [int(e["seq"]) for e in rs] != list(range(len(rs))):
            raise DataError(f"run {rid}: incomplete event log")
        out[rid] = [json.loads(e["event"]) for e in rs]
    return out


def inclusion_prob(snapshot_id: str, entry: Mapping[str, Any]) -> float:
    """π_i of a labelled point (amendment 57 (3)): the planner's `selection.inclusion_prob` (stratified draws: n/N
    in the point's stratum for base parents, P(episode) × P(point | episode) for re-run parents); for the uniform
    per-point draw, which records no selection, the recorded `dp_sampling.fraction`."""
    sel = entry.get("selection")
    if sel is not None:
        pi = sel.get("inclusion_prob")
    else:
        dp = (entry.get("settings") or {}).get("dp_sampling") or {}
        pi = None if dp.get("scheme") else dp.get("fraction")
    if pi is None or not 0.0 < float(pi) <= 1.0:
        raise DataError(f"label entry {snapshot_id} records no inclusion probability in (0, 1] (amendment 57 (3)): "
                        f"{pi!r}")
    return float(pi)


def read_inclusion(path: Path | str) -> tuple[dict[str, float], dict]:
    """The inclusion probabilities that replace the label maps' own for the points a later tier re-drew: the final
    tier extended batch A's draw within the same strata, so every point of batch A and of the final tier has the
    probability of the combined draw, not the one its label map recorded. The file holds `points` (snapshot id ->
    π), `strata` (per task and parent outcome: sizes and the stratum's π) and `dropped`. Checked here: every π is in
    (0, 1], and a point of task T has the π of one of task T's strata. Returns the points and what was read."""
    path = Path(path)
    doc = json.loads(path.read_text())
    points = doc.get("points") if isinstance(doc, dict) else None
    if not isinstance(points, dict) or not points:
        raise DataError(f"{path.name}: no `points` (snapshot id -> inclusion probability)")
    out = {}
    for sid, pi in points.items():
        if not isinstance(pi, (int, float)) or isinstance(pi, bool) or not 0.0 < float(pi) <= 1.0:
            raise DataError(f"{path.name}: point {sid} has no inclusion probability in (0, 1]: {pi!r}")
        out[str(sid)] = float(pi)
    strata = doc.get("strata") or {}
    by_task: dict[str, list[float]] = defaultdict(list)
    for name, st in strata.items():
        by_task[name.split("/")[0]].append(float(st["inclusion_prob"]))
    checked = 0
    for sid, pi in out.items():
        parts = sid.split("/")
        if len(parts) > 1 and parts[1] in by_task:
            checked += 1
            if not any(abs(pi - q) <= 1e-12 for q in by_task[parts[1]]):
                raise DataError(f"{path.name}: point {sid} has π {pi}, not that of a stratum of {parts[1]} "
                                f"({sorted(by_task[parts[1]])})")
    return out, {"file": path.name, "points": len(out), "strata": len(strata), "checked_against_strata": checked,
                 "dropped": len(doc.get("dropped") or [])}


def record_full_runs(runs: Sequence[Mapping]) -> tuple[list[Mapping], dict[str, str]]:
    """One row per run id (amendment 66 (5)). A run id may appear twice only as a screened exploration run and its
    re-run: both rows of phase `explore`, one record `off` and one record `full`; the record-`full` row is kept.
    Anything else that repeats a run id is an error. Returns the kept rows and, for each shared run id, the kept
    row's job id (its events are that job's)."""
    by: dict[str, list] = defaultdict(list)
    for r in runs:
        by[r["run_id"]].append(r)
    kept, shared = [], {}
    for rid, rs in by.items():
        if len(rs) > 1:
            full = [r for r in rs if r.get("record") == "full"]
            off = [r for r in rs if r.get("record") == "off"]
            if len(rs) != 2 or len(full) != 1 or len(off) != 1 or any(r.get("phase") != "explore" for r in rs):
                raise DataError(f"run id {rid} has {len(rs)} rows (phase, record: "
                                f"{[(r.get('phase'), r.get('record')) for r in rs]}): only a screened exploration "
                                "run and its record-full re-run may share one (amendment 66 (5))")
            if not full[0].get("job_id") or full[0].get("job_id") == off[0].get("job_id"):
                raise DataError(f"run id {rid}: the screen's and the re-run's rows need different job ids")
            rs, shared[rid] = full, str(full[0]["job_id"])
        kept.append(rs[0])
    return kept, shared


def read_events(root: Path | str) -> list[dict]:
    """The events table of a local copy of the records, part by part, each row with the job id of its part
    (`records/events/<phase>/<job_id>.parquet`, `vlar_harness.dataset.record_key`), which tells a screened run's
    events from its re-run's (amendment 66 (5))."""
    import pyarrow.parquet as pq
    out: list[dict] = []
    for part in sorted((Path(root) / "records" / "events").rglob("*.parquet")):
        for row in pq.read_table(part, columns=["run_id", "seq", "event"]).to_pylist():
            row["job_id"] = part.stem
            out.append(row)
    return out


def missing_seeds(entry: Mapping[str, Any], own: Mapping[tuple[str, int], Sequence[Mapping]], seeds: int) -> list[str]:
    """The branch seeds of a label entry with no finished run in the records: no log at all, or a log that does not
    end with its `end` event. Seeds the parent rollout provides are not branch runs (§7). A point with any such
    seed is incomplete (§12: training uses only complete points)."""
    out = []
    for canon in dict.fromkeys(entry["shares"].values()):         # each canonical branch once, in the entry's order
        for s in range(1, seeds + 1):
            if s in entry["parent_seeds"].get(canon, []):
                continue
            log = own.get((canon, s))
            if not log or log[-1].get("t") != "end":
                out.append(f"{canon}/s{s}")
    return out


def _vec(v, n: int) -> np.ndarray:
    a = np.asarray(v if v is not None else [np.nan] * n, dtype=np.float64).ravel()
    if a.size != n:
        raise DataError(f"expected {n} values, got {a.size}")
    return a


def point_columns(row: Mapping[str, Any], *, thresholds: tuple[float, float]) -> dict[str, np.ndarray]:
    """One point's input columns (unmeasured values as 0, so the decider sees 0 and, for tracking, the flag)."""
    from vlar_harness.measure import gripper_fractions_of_point
    shut, partial = gripper_fractions_of_point(dict(row), *thresholds)
    valid = bool(row.get("tracking_valid"))
    c = {"features_main": _vec(row["features_main"], 1024), "features_wrist": _vec(row["features_wrist"], 1024),
         "robot_state": _vec(row["robot_state"], 11), "proposal": _vec(row["proposal"], 70),
         "budget_frac": np.array([row["budget_remaining_steps"] / 100.0]), "steps_frac": np.array([row["step"] / STEP_LIMIT]),
         "chunk_pos_frac": np.array([row["chunk_pos"] / 20.0]), "prev_segment_pi05": np.array([float(bool(row["prev_segment_pi05"]))]),
         "track_pos_err": np.array([row["track_pos_err"] if valid else 0.0]),
         "track_rot_err": np.array([row["track_rot_err"] if valid else 0.0]), "tracking_valid": np.array([float(valid)]),
         "grip_shut_frac": np.array([shut]), "grip_partial_frac": np.array([partial])}
    for k in SPREAD:
        c[f"chunk_spread_{k}"] = np.array([row.get(f"chunk_spread_{k}")], dtype=np.float64)
        c[f"fresh_spread_{k}"] = np.array([row.get(f"fresh_spread_{k}")], dtype=np.float64)
    for k in DRIFT:
        c[f"drift_{k}"] = np.array([row.get(f"drift_{k}")], dtype=np.float64)
    return {k: np.nan_to_num(v.astype(np.float64), nan=0.0) for k, v in c.items()}


def history_columns(points: Sequence[Mapping], wanted: set, names: Sequence[str], *,
                    thresholds: tuple[float, float]) -> dict[str, dict[str, np.ndarray]]:
    """{snapshot id: {`hist_<name>`: values}} for the labelled points (`wanted`): each of their parent runs' measured
    points is walked in order through an `EpisodeHistory`, as a live decider would see them (`history`)."""
    from . import history as H
    by_run: dict[str, list] = defaultdict(list)
    for p in points:
        by_run[p["run_id"]].append(p)
    out = {}
    for rows in by_run.values():
        if not any(p.get("snapshot_id") in wanted for p in rows):
            continue
        h = H.EpisodeHistory(names)
        for row, view in H.episode_views(rows, thresholds=thresholds):
            cols = h.push(view)
            if row.get("snapshot_id") in wanted:
                out[row["snapshot_id"]] = cols
    return out


def build(runs: Sequence[Mapping], events: Sequence[Mapping], points: Sequence[Mapping],
          label_map: Mapping[str, Mapping], *, limits_s: Mapping[int, float], thresholds: tuple[float, float],
          charges, profiles: Sequence[str] = DECIDERS, history: Sequence[str] = (),
          inclusion: Optional[Mapping[str, float]] = None) -> PointSet:
    """`history`: names of registered history features (`vlar_train.history`) to add as columns `hist_<name>`;
    none by default, and no decider's `COLUMNS` uses one.

    `inclusion`: snapshot id -> π for the points a later tier re-drew (`read_inclusion`). Where given, a point in
    it takes that π in place of its label entry's own. Check: every labelled point whose parent is a base rollout
    must be in it (batch A and the final tier are base points, and all of them were re-drawn); one that is not
    stops the build, since its label map's π would be the stale one. Points of other parents keep their own."""
    from vlar_harness import labels as L
    runs, shared = record_full_runs(runs)                       # amendment 66 (5): the record-full row of a shared id
    if shared:
        if any(e.get("job_id") is None for e in events if e["run_id"] in shared):
            raise DataError("a screened run and its re-run share a run id: their events can only be told apart by "
                            "job id; read the events table with `read_events` (amendment 66 (5))")
        events = [e for e in events if e["run_id"] not in shared or str(e["job_id"]) == shared[e["run_id"]]]
    run = {r["run_id"]: r for r in runs}
    logs = group_events(events)
    own: dict[str, dict[tuple[str, int], list]] = defaultdict(dict)
    for r in runs:
        if r.get("snapshot_id") and r.get("branch"):
            own[r["snapshot_id"]][(r["branch"], int(r["seed_index"]))] = logs.get(r["run_id"], [])
    rows = [p for p in points if p.get("snapshot_id") in label_map]
    if not rows:
        raise DataError("no point of the label map is in the points table")
    hist = history_columns(points, {p["snapshot_id"] for p in rows}, history, thresholds=thresholds) if history else {}
    report = {"label_entries": len(label_map), "points": 0, "incomplete": [], "shared_run_ids": len(shared),
              "no_points_row": sorted(set(label_map) - {p["snapshot_id"] for p in rows})}
    cols: dict[str, list] = defaultdict(list)
    task, sid, can, rem, ys, exc = [], [], [], defaultdict(list), defaultdict(list), defaultdict(list)
    weight, episode = [], []
    stale, inc = [], {"applied": 0, "changed_from_label_map": 0, "base_points": 0}
    if inclusion is not None:
        for p in rows:
            if run[p["run_id"]].get("phase") == "base":
                inc["base_points"] += 1
                if p["snapshot_id"] not in inclusion:
                    stale.append(p["snapshot_id"])
        if stale:
            raise DataError(f"{len(stale)} labelled base points have no inclusion probability in the inclusion file "
                            f"(first: {sorted(stale)[0]}): their label map's own would be the stale one")
        inc["in_file_not_labelled"] = len(set(inclusion) - {p["snapshot_id"] for p in rows})
        report["inclusion"] = inc
    for p in rows:
        parent = run[p["run_id"]]
        if parent.get("record") not in (None, "full"):            # points are measured only in record-full runs
            raise DataError(f"point {p['snapshot_id']}: its parent's run row is record {parent['record']!r}, not full")
        t = int(parent["task_id"])
        entry = label_map[p["snapshot_id"]]
        if not (entry.get("settings") or {}).get("seeds"):
            raise DataError(f"label entry {p['snapshot_id']} records no planner settings (amendment 49)")
        if inclusion is not None and p["snapshot_id"] in inclusion:
            pi = float(inclusion[p["snapshot_id"]])
            if not 0.0 < pi <= 1.0:
                raise DataError(f"inclusion probability of {p['snapshot_id']} is not in (0, 1]: {pi!r}")
            mapped = (entry.get("selection") or {}).get("inclusion_prob")
            inc["applied"] += 1
            inc["changed_from_label_map"] += mapped is None or float(mapped) != pi
        else:
            pi = inclusion_prob(p["snapshot_id"], entry)
        missing = missing_seeds(entry, own.get(p["snapshot_id"], {}), int(entry["settings"]["seeds"]))
        if missing:                                               # incomplete: left out and reported (§12)
            report["incomplete"].append({"snapshot_id": p["snapshot_id"], "missing": missing})
            continue
        parent_log = logs[p["run_id"]]
        pl = L.point_labels(prefix_of(parent_log, p["index"]), entry, own.get(p["snapshot_id"], {}),
                            seeds=int(entry["settings"]["seeds"]),
                            charges=charges, limit_s=float(limits_s[t]),
                            parent_log=parent_log if entry.get("parent_seeds") else None, profiles=tuple(profiles))
        for k, v in point_columns(p, thresholds=thresholds).items():
            cols[k].append(v)
        if history:
            if p["snapshot_id"] not in hist:
                raise DataError(f"point {p['snapshot_id']} has no history view (π0.5 budget spent at the point?)")
            for k, v in hist[p["snapshot_id"]].items():
                cols[k].append(v)
        task.append(t)
        sid.append(p["snapshot_id"])
        can.append(int(p["budget_remaining_steps"]) >= 10)
        weight.append(1.0 / pi)
        episode.append(str((entry.get("point") or {}).get("episode_id") or parent.get("episode_id") or p["run_id"]))
        for prof in profiles:
            rem[prof].append(pl.remaining_s[prof])
            exc[prof].append(prof in pl.excluded)
            ys[prof].append([pl.labels[b][prof].value if b in pl.labels and prof in pl.labels[b] else np.nan
                             for b in BRANCHES])
    report["points"] = len(sid)
    if not sid:
        raise DataError(f"none of the {len(rows)} labelled points is complete: every one misses a branch seed "
                        f"(first: {report['incomplete'][0]})")
    return PointSet({k: np.stack(v) for k, v in cols.items()}, np.array(task), sid, np.array(can),
                    {k: np.array(v, dtype=np.float64) for k, v in rem.items()},
                    {k: np.array(v, dtype=np.float64) for k, v in ys.items()},
                    {k: np.array(v, dtype=bool) for k, v in exc.items()}, np.array(weight, dtype=np.float64),
                    np.array(episode, dtype=str), report)


def read_label_maps(paths) -> tuple[dict, list[dict]]:
    """One label map from one file or several (each tranche of the collection has its own). The maps are keyed by
    snapshot id; a point named by two files must have the same entry in both, else the files disagree about its
    branches, seeds or sampling and nothing is loaded. Returns the merged map and, per file, its name, its entries
    and how many of them an earlier file already held."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(x) for x in paths]
    if not paths:
        raise DataError("no label map given")
    merged, files = {}, []
    for path in paths:
        part = json.loads(path.read_text())
        if not isinstance(part, dict):
            raise DataError(f"label map {path.name} is not a JSON object keyed by snapshot id")
        clash = sorted(k for k, v in part.items() if k in merged and merged[k] != v)
        if clash:
            raise DataError(f"label map {path.name} gives {len(clash)} points a different entry than an earlier map "
                            f"(first: {clash[0]}): the maps disagree, nothing is loaded")
        files.append({"file": path.name, "entries": len(part), "already_held": sum(k in merged for k in part)})
        merged.update(part)
    return merged, files


def from_records(root: Path | str, label_map_path, *, cfg=None, history: Sequence[str] = (),
                 inclusion: Optional[Path | str] = None) -> PointSet:
    """`label_map_path`: one label map file or a sequence of them (`read_label_maps`). `inclusion`: the file of
    re-drawn inclusion probabilities (`read_inclusion`), applied and checked by `build`."""
    from vlar_harness import clock as clk
    from vlar_harness.config import load_config
    from vlar_harness.dataset import read_table
    cfg = cfg or load_config()
    if not getattr(cfg, "frozen", False):
        # The lineage versions are calibrated too: step 2's hold the empty-gripper shut threshold 9.004 mm, the
        # frozen file 3.0 mm (amendment 50). Gripper fractions are computed here from raw widths, so only the frozen
        # file's thresholds may be used.
        raise DataError("training data needs the frozen calibration file (VLAR_CALIBRATION; v1.0 b80124b2), not "
                        "an earlier version: their gripper thresholds differ")
    g = cfg["measurement"]["gripper_thresholds"]
    tl = cfg["time_limit"]
    if tl.get("status") != "calibrated" or g.get("status") != "calibrated":
        raise DataError("the calibration file must hold calibrated time limits and gripper thresholds (Phase 0)")
    label_map, files = read_label_maps(label_map_path)
    redrawn, redrawn_info = read_inclusion(inclusion) if inclusion is not None else (None, None)
    ps = build(read_table(root, "runs").to_pylist(), read_events(root),
               read_table(root, "points").to_pylist(), label_map,
               limits_s={int(k): float(v) for k, v in tl["per_task_s"].items()},
               thresholds=(float(g["shut_width_m"]), float(g["open_width_m"])),
               charges=clk.Charges.from_config(cfg), history=history, inclusion=redrawn)
    ps.report["label_maps"] = files
    if redrawn_info is not None:
        ps.report["inclusion"] = {**redrawn_info, **ps.report["inclusion"]}
    return ps


def synthetic(n_per_task: int = 60, n_tasks: int = 10, seed: int = 0, signal: float = 3.0,
              feature_dim: int = 1024) -> PointSet:
    """A synthetic point set with the records' shapes, for tests and dry runs: escalation helps (branch 2 beats
    branch 1) exactly where a hidden direction of the features and the spread inputs is high. Sampling weights as
    under amendment 57: two strata per task with inclusion probabilities 0.8 and 0.1 (drawn from a generator of
    their own, so every other array is what it was before the weights existed), and each task's points spread over
    12 parent episodes (likewise)."""
    rng = np.random.default_rng(seed)
    n = n_per_task * n_tasks
    task = np.repeat(np.arange(n_tasks), n_per_task)
    hidden = rng.normal(size=n)
    cols = {"features_main": rng.normal(size=(n, feature_dim)), "features_wrist": rng.normal(size=(n, feature_dim)),
            "robot_state": rng.normal(size=(n, 11)), "proposal": rng.normal(size=(n, 70)),
            "budget_frac": rng.integers(0, 27, size=(n, 1)) / 10.0, "steps_frac": rng.uniform(0, 1, (n, 1)),
            "chunk_pos_frac": rng.integers(0, 3, (n, 1)) / 2.0, "prev_segment_pi05": rng.integers(0, 2, (n, 1)) * 1.0,
            "track_pos_err": rng.uniform(0, 0.02, (n, 1)), "track_rot_err": rng.uniform(0, 0.1, (n, 1)),
            "tracking_valid": np.ones((n, 1)), "grip_shut_frac": rng.uniform(0, 1, (n, 1)),
            "grip_partial_frac": rng.uniform(0, 1, (n, 1))}
    cols["features_main"][:, 0] += signal * hidden
    for k in SPREAD:
        cols[f"chunk_spread_{k}"] = (hidden + rng.normal(scale=0.5, size=n)).reshape(-1, 1)
        cols[f"fresh_spread_{k}"] = (hidden + rng.normal(scale=0.3, size=n)).reshape(-1, 1)
    for k in DRIFT:
        cols[f"drift_{k}"] = (hidden + rng.normal(scale=0.3, size=n)).reshape(-1, 1)
    p_cont = 1 / (1 + np.exp(-(-0.5 - hidden)))
    p_one = 1 / (1 + np.exp(-(0.3 + 0.5 * hidden)))
    y_base = np.stack([p_cont, p_one, np.minimum(1, p_one + 0.05), np.minimum(1, p_one + 0.1)], axis=1)
    y = {p: np.round(np.clip(y_base + rng.normal(scale=0.05, size=y_base.shape), 0, 1) * 3) / 3 for p in DECIDERS}
    remaining = {p: rng.uniform(5, 60, n) for p in DECIDERS}
    excluded = {p: np.zeros(n, dtype=bool) for p in DECIDERS}
    excluded["router"][rng.choice(n, size=n // 50, replace=False)] = True
    pi = np.where(np.random.default_rng([seed, 57]).random(n) < 2 / 3, 0.8, 0.1)
    episode = [f"synthetic/t{t}/i{k}" for t, k in zip(task, np.random.default_rng([seed, 68]).integers(0, 12, n))]
    return PointSet(cols, task, [f"s{i}" for i in range(n)], cols["budget_frac"][:, 0] >= 0.1, remaining, y, excluded,
                    1.0 / pi, episode)
