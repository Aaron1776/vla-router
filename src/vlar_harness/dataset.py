"""Dataset v1 records and files (docs/SPEC.md §13): what each queue job writes, and how to read it back.

Layout (keys relative to the bucket, or to a local copy of it):

    records/<table>/<phase>/<job_id>.parquet   one Parquet part per job and table (tables below)
    snapshots/<episode_id>/dpNN.npz            one snapshot per measured decision point (snapshot.py format)
    images/<episode_id>.npz                    lossless PNGs of every camera at every measured decision point,
                                               member `dpNN/<camera key>` (uint8 bytes; allow_pickle=False)
    results/<phase>/<job_id>.json              the queue's marker, written last; lists every file above
                                               with its SHA-256 and size (vlar_ops.queue, vlar_ops.worker)

Tables (schema version 1; every array column is flattened row-major, shapes given here):
- runs: one row per episode or branch run: identity, seeds, mode, profile, record, the live-limit flag, outcome,
  pi0.5 usage, simulated time under the run's own profile, config hash and harness version.
- steps: one row per executed step: driver, action (7, float32, as executed), the observation's end-effector
  position (3), quaternion (4) and finger positions (2).
- events: one row per event-log entry, the event as canonical JSON (sorted keys); `seq` is its position in the
  run's own log (a branch's log starts with its `restore` and `branch` events; the parent's prefix is the
  parent's).
- points: one row per measured decision point (record="full"): index, step, kind, chunk position, budget left,
  snapshot id, robot state (11), proposal (10 x 7), both feature vectors (1024 each, float32), the raw chunk-time
  samples (K x 30 x 20, float32; at the point after they were drawn) or the raw probe (K x 30 x 20), the chunk
  spread, fresh spread and drift summaries, the previous segment (n steps of action 7, controller position 3,
  rotation matrix 9, finger positions 2) with its tracking errors. The gripper fractions are not stored: they need
  the calibrated thresholds and are computed offline from the stored segment (amendment 35,
  measure.gripper_fractions_of_point).
Nothing here is lossy: every value is the one the run produced, in its own precision.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from . import measure as meas
from .snapshot import Snapshot, SnapshotStore, to_bytes

SCHEMA_VERSION = 1
TABLES = ("runs", "steps", "events", "points")
SPREAD = ("pos_std_mean", "pos_std_max", "rot_dev_mean", "grip_disagree_frac")
DRIFT = ("pos_dist_mean", "rot_dist_mean", "grip_disagree_frac")


def _schemas():
    import pyarrow as pa
    f32, f64, i32, i64, s, b = pa.float32(), pa.float64(), pa.int32(), pa.int64(), pa.string(), pa.bool_()
    fixed = lambda t, n: pa.list_(t, n)
    runs = pa.schema([
        ("run_id", s), ("run_kind", s), ("phase", s), ("job_id", s), ("episode_id", s), ("snapshot_id", s),
        ("branch", s), ("seed_index", i32), ("pi05_calls", i32), ("task_id", i32), ("init_idx", i32),
        ("scene_seed", i64), ("rollout_seed", i64), ("parent_rollout_seed", i64), ("mode", s), ("profile", s),
        ("record", s), ("time_limit", b), ("time_limit_s", f64), ("success", b), ("end_reason", s), ("steps", i32),
        ("success_step", i32), ("pi05_steps", i32), ("pi05_requests", i32), ("pi05_failed_requests", i32),
        ("config_hash", s), ("calibration_hash", s), ("harness_version", s), ("schema_version", i32)])
    steps = pa.schema([("run_id", s), ("step", i32), ("driver", s), ("action", fixed(f32, 7)),
                       ("eef_pos", fixed(f64, 3)), ("eef_quat", fixed(f64, 4)), ("gripper_qpos", fixed(f64, 2))])
    events = pa.schema([("run_id", s), ("seq", i32), ("t", s), ("step", i32), ("event", s)])
    points = pa.schema(
        [("run_id", s), ("index", i32), ("step", i32), ("kind", s), ("chunk_pos", i32), ("prev_segment_pi05", b),
         ("budget_remaining_steps", i32), ("snapshot_id", s), ("robot_state", fixed(f64, 11)),
         ("proposal", fixed(f32, 70)), ("features_main", fixed(f32, 1024)), ("features_wrist", fixed(f32, 1024)),
         ("n_samples", i32), ("chunk_samples_raw", pa.list_(f32)), ("probe_raw", pa.list_(f32))]
        + [(f"chunk_spread_{k}", f64) for k in SPREAD] + [(f"fresh_spread_{k}", f64) for k in SPREAD]
        + [(f"drift_{k}", f64) for k in DRIFT]
        + [("prev_len", i32), ("prev_driver", s), ("prev_actions", pa.list_(f32)), ("prev_ee_pos", pa.list_(f64)),
           ("prev_ee_ori_mat", pa.list_(f64)), ("prev_gripper_qpos", pa.list_(f64)), ("track_pos_err", f64),
           ("track_rot_err", f64), ("tracking_valid", b)])
    return {"runs": runs, "steps": steps, "events": events, "points": points}


@dataclasses.dataclass(frozen=True)
class Artifact:
    key: str                       # where it goes in the bucket
    path: Path                     # the local file
    sha256: str
    size: int

    def as_record(self) -> dict[str, Any]:
        return {"key": self.key, "path": str(self.path), "sha256": self.sha256, "size": self.size}


def record_key(table: str, phase: str, job_id: str) -> str:
    return f"records/{table}/{phase}/{job_id}.parquet"


def snapshot_key(snapshot_id: str) -> str:
    return f"snapshots/{snapshot_id}.npz"


def images_key(episode_id: str) -> str:
    return f"images/{episode_id}.npz"


def _flat(a, dtype) -> Optional[list]:
    return None if a is None else np.asarray(a, dtype=dtype).ravel().tolist()


def png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buf, format="PNG")
    return buf.getvalue()


def png_decode(data: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data)))


class JobOutput:
    """Collects one job's rows and files in a local staging directory; `finish()` writes them and lists them."""

    def __init__(self, staging: Path | str, phase: str, job_id: str, *, header: Mapping[str, Any]):
        self.dir = Path(staging) / phase / job_id
        self.phase, self.job_id = phase, job_id
        self.header = dict(header)                     # harness_version, config_hash, calibration_hash
        self.rows: dict[str, list[dict]] = {t: [] for t in TABLES}
        self.files: list[tuple[str, Path]] = []
        self.images: dict[str, dict[str, bytes]] = {}  # episode id -> member -> PNG bytes
        self.snapshots = SnapshotStore(self.dir / "snapshots")

    # ------------------------------------------------------------ rows

    def add_run(self, row: Mapping[str, Any]) -> None:
        self.rows["runs"].append({**row, "phase": self.phase, "job_id": self.job_id,
                                  "config_hash": self.header.get("config_hash"),
                                  "calibration_hash": self.header.get("calibration_hash"),
                                  "harness_version": self.header.get("harness_version"),
                                  "schema_version": SCHEMA_VERSION})

    def add_steps(self, run_id: str, trajectory: Sequence[Mapping]) -> None:
        for r in trajectory:
            self.rows["steps"].append({
                "run_id": run_id, "step": int(r["step"]), "driver": r["driver"],
                "action": _flat(r["action"], np.float32), "eef_pos": _flat(r["robot0_eef_pos"], np.float64),
                "eef_quat": _flat(r["robot0_eef_quat"], np.float64),
                "gripper_qpos": _flat(r["robot0_gripper_qpos"], np.float64)})

    def add_events(self, run_id: str, events: Sequence[Mapping]) -> None:
        for i, e in enumerate(events):
            self.rows["events"].append({"run_id": run_id, "seq": i, "t": e["t"], "step": e.get("step"),
                                        "event": json.dumps(e, sort_keys=True, allow_nan=True)})

    def add_point(self, run_id: str, m: Mapping[str, Any], *, budget_remaining_steps: int, snapshot_id: str,
                  n_samples: int) -> None:
        """The gripper fractions are not stored (amendment 35): they need the calibrated thresholds and are
        computed offline from `prev_actions` and `prev_gripper_qpos` (measure.gripper_fractions_of_point)."""
        seg = m["prev_segment"]
        track = meas.tracking(seg)
        row = {"run_id": run_id, "index": int(m["index"]), "step": int(m["step"]), "kind": m["kind"],
               "chunk_pos": int(m["chunk_pos"]), "prev_segment_pi05": bool(m["prev_segment_pi05"]),
               "budget_remaining_steps": int(budget_remaining_steps), "snapshot_id": snapshot_id,
               "robot_state": _flat(m["robot_state"], np.float64), "proposal": _flat(m["proposal"], np.float32),
               "features_main": _flat(m["features_main"], np.float32),
               "features_wrist": _flat(m["features_wrist"], np.float32), "n_samples": int(n_samples),
               "chunk_samples_raw": _flat(m["chunk_samples_raw"], np.float32),
               "probe_raw": _flat(m["probe_raw"], np.float32),
               "prev_len": len(seg), "prev_driver": seg[0]["driver"] if seg else None,
               "prev_actions": _flat([r["action"] for r in seg], np.float32) if seg else [],
               "prev_ee_pos": _flat([r["ee_pos"] for r in seg], np.float64) if seg else [],
               "prev_ee_ori_mat": _flat([r["ee_ori_mat"] for r in seg], np.float64) if seg else [],
               "prev_gripper_qpos": _flat([r["gripper_qpos"] for r in seg], np.float64) if seg else [],
               "track_pos_err": track[0], "track_rot_err": track[1], "tracking_valid": track[2]}
        for group, names in (("chunk_spread", SPREAD), ("fresh_spread", SPREAD), ("drift", DRIFT)):
            summary = m[group]
            for k in names:
                row[f"{group}_{k}"] = None if summary is None else float(summary[k])
        self.rows["points"].append(row)

    # ------------------------------------------------------------ files

    def put_snapshot(self, snapshot_id: str, snapshot: Snapshot) -> None:
        self.snapshots.put(snapshot_id, snapshot)
        self.files.append((snapshot_key(snapshot_id), self.snapshots.path(snapshot_id)))

    def put_images(self, episode_id: str, index: int, images: Mapping[str, np.ndarray]) -> None:
        members = self.images.setdefault(episode_id, {})
        for cam, img in images.items():
            members[f"dp{index:02d}/{cam}"] = png_bytes(img)

    def finish(self) -> list[Artifact]:
        import pyarrow as pa
        import pyarrow.parquet as pq
        schemas = _schemas()
        out = list(self.files)
        for table, rows in self.rows.items():
            if not rows:
                continue
            path = self.dir / "records" / f"{table}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=schemas[table]), path)
            out.append((record_key(table, self.phase, self.job_id), path))
        for episode_id, members in self.images.items():
            path = self.dir / "images" / f"{episode_id.replace('/', '__')}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                np.savez(f, **{k: np.frombuffer(v, dtype=np.uint8) for k, v in sorted(members.items())})
            out.append((images_key(episode_id), path))
        arts = []
        for key, path in out:
            data = Path(path).read_bytes()
            arts.append(Artifact(key, Path(path), hashlib.sha256(data).hexdigest(), len(data)))
        return arts


# ---------------------------------------------------------------- reading a local copy

def read_table(root: Path | str, table: str):
    """Every job's part of `table` under a local copy of the bucket, as one pyarrow Table."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    parts = sorted((Path(root) / "records" / table).rglob("*.parquet"))
    schema = _schemas()[table]
    if not parts:
        return schema.empty_table()
    return pa.concat_tables([pq.read_table(p, schema=schema) for p in parts])


def read_images(root: Path | str, episode_id: str) -> dict[str, np.ndarray]:
    with np.load(Path(root) / images_key(episode_id), allow_pickle=False) as z:
        return {k: png_decode(z[k].tobytes()) for k in z.files}
