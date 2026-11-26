"""Live deciders for the Spark's evaluation (SPEC §8, §10), ops-side: a `vlar_train` checkpoint run live from
`observe()`. No harness or `vlar_train` code is changed; `vlar_train` (with the history hook, bde8c4a) is imported
from the source mounted next to the images' copy.

- `TrainedDecider.load(fold_dir)`: `result.json` (decider, profile, best epoch, chosen λ) and the best epoch's
  checkpoint (`checkpoints/epochNNN.pt`: weights, the training points' mean and std, n_in, hidden sizes).
- `RandomDecider(decider, seed)`: the same model shape with random weights, mean 0, std 1 and λ 0: a dummy for the
  smoke test that still exercises every input column.
- Inputs: `columns(obs)` builds the decider's point columns from `observe()` with the keys and scales of
  `vlar_train.data.point_columns` (unmeasured values as 0, tracking errors 0 unless valid), history columns through
  `vlar_train.history.EpisodeHistory` and `live_view`, then `vlar_train.deciders.inputs` with the run's own
  remaining time. Rule (`vlar_train.lam`): escalate iff p(one call) - p(continue) > λ and the point can escalate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _vec(v, n: int) -> np.ndarray:
    a = np.asarray(v if v is not None else [np.nan] * n, dtype=np.float64).ravel()
    if a.size != n:
        raise ValueError(f"expected {n} values, got {a.size}")
    return a


def point_columns_live(obs) -> dict[str, np.ndarray]:
    """`vlar_train.data.point_columns`, from `observe()` instead of a points-table row (same keys and scales)."""
    from vlar_train.deciders import DRIFT, SPREAD, STEP_LIMIT
    c, p, pr, fs = obs.common, obs.point, obs.proposal, obs.free_signals
    valid = bool(fs.tracking_valid) if fs is not None else False
    one = lambda x: np.array([np.nan if x is None else float(x)])  # noqa: E731
    cols = {"features_main": _vec(c.features_main, 1024), "features_wrist": _vec(c.features_wrist, 1024),
            "robot_state": _vec(c.robot_state, 11),
            "proposal": _vec(None if pr is None else pr.actions, 70),
            "budget_frac": np.array([c.budget_remaining_steps / 100.0]), "steps_frac": np.array([p.step / STEP_LIMIT]),
            "chunk_pos_frac": one(None if pr is None else pr.chunk_pos / 20.0),
            "prev_segment_pi05": one(None if pr is None else float(bool(pr.prev_segment_pi05))),
            "track_pos_err": np.array([fs.track_pos_err if valid else 0.0]),
            "track_rot_err": np.array([fs.track_rot_err if valid else 0.0]), "tracking_valid": np.array([float(valid)]),
            "grip_shut_frac": one(None if fs is None else fs.grip_closed_shut_frac),
            "grip_partial_frac": one(None if fs is None else fs.grip_closed_partial_frac)}
    for k in SPREAD:
        cols[f"chunk_spread_{k}"] = one(getattr(obs.chunk_spread, k, None) if obs.chunk_spread else None)
        cols[f"fresh_spread_{k}"] = one(getattr(obs.fresh_spread, k, None) if obs.fresh_spread else None)
    for k in DRIFT:
        cols[f"drift_{k}"] = one(getattr(obs.drift, k, None) if obs.drift else None)
    return {k: np.nan_to_num(v.astype(np.float64), nan=0.0) for k, v in cols.items()}


class _Decider:
    name: str
    profile: str
    lam: float
    meta: dict

    def __init__(self, model, mean, std, decider: str, lam: float, meta: dict):
        from vlar_train import deciders as D
        self.model, self.mean, self.std, self.name, self.lam = model, mean, std, decider, float(lam)
        self.profile = D.profile_of(decider)
        self.columns = D.COLUMNS[decider]
        self.history_names = [c[len("hist_"):] for c in self.columns if c.startswith("hist_")]
        self.meta = {**meta, "decider": decider, "profile": self.profile, "lambda": self.lam,
                     "history": self.history_names}
        self.hist = None

    def new_episode(self) -> None:
        from vlar_train.history import EpisodeHistory
        self.hist = EpisodeHistory(self.history_names) if self.history_names else None

    def decide(self, obs) -> tuple[str, dict[str, Any]]:
        import torch
        from vlar_train import deciders as D
        cols = point_columns_live(obs)
        if self.hist is not None:
            from vlar_train.history import live_view
            cols.update(self.hist.push(live_view(obs)))
        x = D.inputs({k: v.reshape(1, -1) for k, v in cols.items()}, np.array([obs.common.time_remaining_s]), self.name)
        with torch.no_grad():
            z = torch.from_numpy(((x - self.mean) / self.std).astype(np.float32))
            p = torch.sigmoid(self.model(z)).numpy().astype(np.float64)[0]
        esc = bool(obs.point.can_escalate) and (p[1] - p[0]) > self.lam
        return ("escalate" if esc else "local"), {"p_continue": float(p[0]), "p_one_call": float(p[1])}


class TrainedDecider(_Decider):
    @classmethod
    def load(cls, fold_dir: str | Path) -> "TrainedDecider":
        import torch
        from vlar_train import deciders as D
        d = Path(fold_dir)
        res = json.loads((d / "result.json").read_text())
        ck = torch.load(d / "checkpoints" / f"epoch{int(res['best_epoch']):03d}.pt", map_location="cpu",
                        weights_only=False)                        # a local file this project wrote
        model = D.make_model(int(ck["n_in"]), tuple(ck["hidden"]))
        model.load_state_dict(ck["state"]); model.eval()
        lam = res["lambda"]["lambda"]
        return cls(model, np.asarray(ck["mean"]), np.asarray(ck["std"]), res["decider"],
                   math.inf if lam is None else lam,
                   {"source": str(d), "fold": res.get("fold"), "best_epoch": res["best_epoch"]})


class RandomDecider(_Decider):
    def __init__(self, decider: str, seed: int = 0, n_in: Optional[int] = None):
        import torch
        from vlar_train import deciders as D
        torch.manual_seed(seed)
        width = n_in or sum({"features_main": 1024, "features_wrist": 1024, "robot_state": 11, "proposal": 70}.get(c, 1)
                            for c in D.COLUMNS[decider]) + 1
        model = D.make_model(width); model.eval()
        super().__init__(model, np.zeros(width), np.ones(width), decider, 0.0,
                         {"source": f"dummy random weights, seed {seed}"})
