"""Snapshots of decision points (docs/SPEC.md §7 as amended) and their on-disk format.

A snapshot is a tree of plain data: the simulator state from `sim.state.capture`, the X-VLA
client's state, the episode fields (task, initial state, seeds, step, budget, previous driver,
...), the event log so far and the decision point's measurements. The run loop fills the last
three; this module only stores them.

File format: one `.npz` per snapshot (loaded with allow_pickle=False). Arrays are stored as npz
members, deduplicated by content (the observation cache repeats the camera images); everything
else is JSON with tags that keep exact types (tuples, NumPy scalars, non-string dict keys), so a
loaded snapshot restores the same Python types and bit-identical numbers.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

FORMAT = "vlar-snapshot/1"


@dataclass
class Snapshot:
    sim: dict[str, Any]
    xvla: Optional[dict[str, Any]]
    episode: dict[str, Any]
    event_log: list = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)
    header: dict[str, Any] = field(default_factory=dict)       # harness version, config hash, ...

    def as_tree(self) -> dict[str, Any]:
        return {"format": FORMAT, "sim": self.sim, "xvla": self.xvla, "episode": self.episode,
                "event_log": self.event_log, "measurements": self.measurements, "header": self.header}

    @staticmethod
    def from_tree(tree: dict[str, Any]) -> "Snapshot":
        if tree.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} snapshot: {tree.get('format')!r}")
        return Snapshot(tree["sim"], tree["xvla"], tree["episode"], tree["event_log"], tree["measurements"],
                        tree["header"])


# ---------------------------------------------------------------- capture / restore

def take(sim, xvla_adapter, episode: dict[str, Any], *, event_log=None, measurements=None,
         header=None) -> Snapshot:
    from .sim import state as sim_state
    return Snapshot(sim=sim_state.capture(sim), xvla=None if xvla_adapter is None else xvla_adapter.get_state(),
                    episode=dict(episode), event_log=list(event_log or []), measurements=dict(measurements or {}),
                    header=dict(header or {}))


def restore(snapshot: Snapshot, sim, xvla_adapter) -> dict[str, Any]:
    """Put a snapshot back into `sim` and the X-VLA adapter. Returns the episode fields."""
    from .sim import state as sim_state
    sim_state.restore(sim, snapshot.sim)
    if xvla_adapter is not None and snapshot.xvla is not None:
        xvla_adapter.set_state(snapshot.xvla)
    return dict(snapshot.episode)


# ---------------------------------------------------------------- encoding

class _Encoder:
    def __init__(self):
        self.arrays: dict[str, np.ndarray] = {}
        self._by_digest: dict[str, str] = {}

    def array(self, a: np.ndarray) -> str:
        if a.dtype == object:
            raise TypeError("object arrays cannot be stored")
        a = np.ascontiguousarray(a)
        digest = hashlib.sha256(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes()).hexdigest()
        key = self._by_digest.get(digest)
        if key is None:
            key = f"a{len(self.arrays)}"
            self.arrays[key] = a
            self._by_digest[digest] = key
        return key

    def encode(self, v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return {"$nd": self.array(v)}
        if isinstance(v, np.floating):
            return {"$np": v.dtype.str, "f": float.hex(float(v))}   # exact for float16/32/64
        if isinstance(v, np.generic):
            return {"$np": v.dtype.str, "v": v.item()}
        if v is None or isinstance(v, (bool, str)):
            return v
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return {"$f": float.hex(v)}                    # exact, and covers inf/nan
        if isinstance(v, tuple):
            return {"$tuple": [self.encode(x) for x in v]}
        if isinstance(v, list):
            return [self.encode(x) for x in v]
        if isinstance(v, dict):
            return {"$dict": [[self.encode(k), self.encode(x)] for k, x in v.items()]}
        raise TypeError(f"cannot store {type(v).__name__} in a snapshot")


def _decode(v: Any, arrays) -> Any:
    if isinstance(v, list):
        return [_decode(x, arrays) for x in v]
    if isinstance(v, dict):
        if "$nd" in v:
            return np.array(arrays[v["$nd"]])
        if "$np" in v:
            raw = float.fromhex(v["f"]) if "f" in v else v["v"]
            return np.dtype(v["$np"]).type(raw)
        if "$f" in v:
            return float.fromhex(v["$f"])
        if "$tuple" in v:
            return tuple(_decode(x, arrays) for x in v["$tuple"])
        if "$dict" in v:
            return {_decode(k, arrays): _decode(x, arrays) for k, x in v["$dict"]}
        raise ValueError(f"unknown tagged value {sorted(v)}")
    return v


def to_bytes(snapshot: Snapshot, compress: bool = True) -> bytes:
    enc = _Encoder()
    meta = json.dumps(enc.encode(snapshot.as_tree()), separators=(",", ":"), allow_nan=False)
    buf = io.BytesIO()
    (np.savez_compressed if compress else np.savez)(buf, __meta__=np.frombuffer(meta.encode(), dtype=np.uint8),
                                                     **enc.arrays)
    return buf.getvalue()


def from_bytes(data: bytes) -> Snapshot:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "__meta__"}
        meta = json.loads(z["__meta__"].tobytes().decode())
    return Snapshot.from_tree(_decode(meta, arrays))


# ---------------------------------------------------------------- store

class SnapshotStore:
    """Snapshots on local disk, one file per id. Ids are `<episode id>/dp<index>`, so writes are idempotent."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    @staticmethod
    def make_id(episode_id: str, decision_index: int) -> str:
        return f"{episode_id}/dp{decision_index:02d}"

    def path(self, snapshot_id: str) -> Path:
        if snapshot_id.startswith("/") or ".." in snapshot_id.split("/"):
            raise ValueError(f"bad snapshot id {snapshot_id!r}")
        return self.root / f"{snapshot_id}.npz"

    def put(self, snapshot_id: str, snapshot: Snapshot) -> str:
        """Write the snapshot; returns the SHA-256 of the file."""
        data = to_bytes(snapshot)
        p = self.path(snapshot_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        tmp.write_bytes(data)
        tmp.replace(p)
        return hashlib.sha256(data).hexdigest()

    def get(self, snapshot_id: str) -> Snapshot:
        return from_bytes(self.path(snapshot_id).read_bytes())

    def exists(self, snapshot_id: str) -> bool:
        return self.path(snapshot_id).exists()
