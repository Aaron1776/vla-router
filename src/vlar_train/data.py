"""Training points from the records (SPEC §6, §7; HANDOFF "Decider training on the Mac").

Inputs: a local copy of `records/{runs,events,points}` (`gcloud storage cp -r .../records/<table> ...`), the branch
planner's label map (`vlar_ops.plan.branch_plan`: snapshot id -> the point, each branch's call count, the canonical
branch it shares, the seeds the parent provides) and the frozen calibration (per-task time limits, gripper
thresholds, clock charges). Snapshots and images are not needed: a point's prefix is its parent run's event log up
to and including the point's `decision_point` event, and `vlar_harness.labels.point_labels` checks that it fits.

Per point and per profile: the four branch labels (NaN where the planner ran no such branch), whether the point is
excluded for that profile (its clock already beyond the limit, amendment 22) and the remaining-time input recomputed
under that profile (amendment 23); the measured inputs from the points table, gripper fractions computed offline
(amendment 35)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
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
                        {p: v[idx] for p, v in self.excluded.items()})


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


def build(runs: Sequence[Mapping], events: Sequence[Mapping], points: Sequence[Mapping],
          label_map: Mapping[str, Mapping], *, limits_s: Mapping[int, float], thresholds: tuple[float, float],
          charges, seeds: int, profiles: Sequence[str] = DECIDERS) -> PointSet:
    from vlar_harness import labels as L
    run = {r["run_id"]: r for r in runs}
    logs = group_events(events)
    own: dict[str, dict[tuple[str, int], list]] = defaultdict(dict)
    for r in runs:
        if r.get("snapshot_id") and r.get("branch"):
            own[r["snapshot_id"]][(r["branch"], int(r["seed_index"]))] = logs.get(r["run_id"], [])
    rows = [p for p in points if p.get("snapshot_id") in label_map]
    if not rows:
        raise DataError("no point of the label map is in the points table")
    cols: dict[str, list] = defaultdict(list)
    task, sid, can, rem, ys, exc = [], [], [], defaultdict(list), defaultdict(list), defaultdict(list)
    for p in rows:
        parent = run[p["run_id"]]
        t = int(parent["task_id"])
        entry = label_map[p["snapshot_id"]]
        parent_log = logs[p["run_id"]]
        pl = L.point_labels(prefix_of(parent_log, p["index"]), entry, own.get(p["snapshot_id"], {}), seeds=seeds,
                            charges=charges, limit_s=float(limits_s[t]),
                            parent_log=parent_log if entry.get("parent_seeds") else None, profiles=tuple(profiles))
        for k, v in point_columns(p, thresholds=thresholds).items():
            cols[k].append(v)
        task.append(t)
        sid.append(p["snapshot_id"])
        can.append(int(p["budget_remaining_steps"]) >= 10)
        for prof in profiles:
            rem[prof].append(pl.remaining_s[prof])
            exc[prof].append(prof in pl.excluded)
            ys[prof].append([pl.labels[b][prof].value if b in pl.labels and prof in pl.labels[b] else np.nan
                             for b in BRANCHES])
    return PointSet({k: np.stack(v) for k, v in cols.items()}, np.array(task), sid, np.array(can),
                    {k: np.array(v, dtype=np.float64) for k, v in rem.items()},
                    {k: np.array(v, dtype=np.float64) for k, v in ys.items()},
                    {k: np.array(v, dtype=bool) for k, v in exc.items()})


def from_records(root: Path | str, label_map_path: Path | str, *, seeds: int, cfg=None) -> PointSet:
    from vlar_harness import clock as clk
    from vlar_harness.config import load_config
    from vlar_harness.dataset import read_table
    cfg = cfg or load_config()
    g = cfg["measurement"]["gripper_thresholds"]
    tl = cfg["time_limit"]
    if tl.get("status") != "calibrated" or g.get("status") != "calibrated":
        raise DataError("the calibration file must hold calibrated time limits and gripper thresholds (Phase 0)")
    label_map = json.loads(Path(label_map_path).read_text())
    return build(read_table(root, "runs").to_pylist(), read_table(root, "events").to_pylist(),
                 read_table(root, "points").to_pylist(), label_map,
                 limits_s={int(k): float(v) for k, v in tl["per_task_s"].items()},
                 thresholds=(float(g["shut_width_m"]), float(g["open_width_m"])),
                 charges=clk.Charges.from_config(cfg), seeds=seeds)


def synthetic(n_per_task: int = 60, n_tasks: int = 10, seed: int = 0, signal: float = 3.0,
              feature_dim: int = 1024) -> PointSet:
    """A synthetic point set with the records' shapes, for tests and dry runs: escalation helps (branch 2 beats
    branch 1) exactly where a hidden direction of the features and the spread inputs is high."""
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
    return PointSet(cols, task, [f"s{i}" for i in range(n)], cols["budget_frac"][:, 0] >= 0.1, remaining, y, excluded)
