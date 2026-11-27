"""Live deciders for the Spark's evaluation (SPEC §8, §10), ops-side: a `vlar_train` checkpoint run live from
`observe()`. No harness or `vlar_train` code is changed; `vlar_train` (the history hook and the 2 x 2 variants;
seed members from cascader-2x2 3094f4a) is imported from the source mounted over the images' copy.

- `TrainedDecider.load(fold_dir, min_members=...)`: `result.json` (decider, profile, chosen λ, `training_seeds`) and
  every member's checkpoint at its own best epoch (`checkpoints/seed<k>/epochNNN.pt`, or `checkpoints/epochNNN.pt`
  for a one-member fold: weights, the training points' mean and std, n_in, hidden sizes). The decider's
  probabilities are the members' mean, computed as `vlar_train.train.Averaged.predict` does.
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


def ensure_decider(decider: str) -> None:
    """Make `decider` known to `vlar_train.deciders`. The four deciders of §8 always are; the 2 x 2's history
    variants (`router_hist`, `cascader_hist`) and their history features exist only after
    `vlar_train.variants.install()`, which the trainer calls for `--variants 2x2`."""
    from vlar_train import deciders as D
    if decider not in D.COLUMNS:
        try:
            from vlar_train import variants as V
        except ImportError as e:
            raise KeyError(f"decider {decider!r} is not one of {sorted(D.COLUMNS)} and this vlar_train has no "
                           "variants module") from e
        V.install()
    if decider not in D.COLUMNS:
        raise KeyError(f"decider {decider!r} is not known to vlar_train (known: {sorted(D.COLUMNS)})")


def column_width(column: str) -> int:
    """Width of one input column: the measured columns' fixed sizes, a history column's registered width."""
    from vlar_train import history as H
    if column.startswith(H.PREFIX):
        return int(H.FEATURES[column[len(H.PREFIX):]].width)
    return {"features_main": 1024, "features_wrist": 1024, "robot_state": 11, "proposal": 70}.get(column, 1)


class _Decider:
    name: str
    profile: str
    lam: float
    meta: dict

    def __init__(self, members: list, decider: str, lam: float, meta: dict):
        """`members`: [(model, mean, std)], one per training seed; the decider's probabilities are their mean."""
        from vlar_train import deciders as D
        ensure_decider(decider)
        if not members:
            raise ValueError("a decider needs at least one member")
        self.members, self.name, self.lam = list(members), decider, float(lam)
        self.profile = D.profile_of(decider)
        self.columns = D.COLUMNS[decider]
        self.history_names = [c[len("hist_"):] for c in self.columns if c.startswith("hist_")]
        self.meta = {**meta, "decider": decider, "profile": self.profile, "lambda": self.lam,
                     "history": self.history_names}
        self.hist = None
        self.last_inputs = None                 # (columns, x, remaining time) of the latest `build`

    def new_episode(self) -> None:
        from vlar_train.history import EpisodeHistory
        self.hist = EpisodeHistory(self.history_names) if self.history_names else None

    def build(self, obs) -> np.ndarray:
        """The decider's input row at this point, (1, width). Call once per decision point, in order: it adds the
        point to the episode's history. Keeps what it built in `last_inputs`."""
        from vlar_train import deciders as D
        cols = point_columns_live(obs)
        if self.hist is not None:
            from vlar_train.history import live_view
            cols.update(self.hist.push(live_view(obs)))
        x = D.inputs({k: v.reshape(1, -1) for k, v in cols.items()}, np.array([obs.common.time_remaining_s]), self.name)
        self.last_inputs = ({c: cols[c] for c in self.columns}, x, float(obs.common.time_remaining_s))
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """(n, 4) probabilities: each member as `vlar_train.train.Trained.predict` (its own mean and std, float32
        model, sigmoid, float64), and with several members their mean as `vlar_train.train.Averaged.predict`."""
        import torch
        ps = []
        for model, mean, std in self.members:
            with torch.no_grad():
                z = torch.from_numpy(((x - mean) / std).astype(np.float32))
                ps.append(torch.sigmoid(model(z)).numpy().astype(np.float64))
        return ps[0] if len(ps) == 1 else np.mean(ps, axis=0)

    def decide(self, obs) -> tuple[str, dict[str, Any]]:
        x = self.build(obs)
        p = self.predict(x)[0]
        esc = bool(obs.point.can_escalate) and (p[1] - p[0]) > self.lam
        return ("escalate" if esc else "local"), {"p_continue": float(p[0]), "p_one_call": float(p[1])}


class TrainedDecider(_Decider):
    @classmethod
    def load(cls, fold_dir: str | Path, *, min_members: int = 1) -> "TrainedDecider":
        """Every training-seed member of a `vlar_train` fold (`result.json` `training_seeds`; checkpoints in
        `checkpoints/seed<k>/` with several members, in `checkpoints/` with one), each at its own best epoch.
        Refuses a fold with fewer members than `min_members`, and any member whose checkpoint is missing or whose
        input count is not the variant's columns + 1: a decider is never loaded with fewer members than it was
        trained with."""
        import torch
        from vlar_train import deciders as D
        d = Path(fold_dir)
        res = json.loads((d / "result.json").read_text())
        ensure_decider(res["decider"])
        ts = res.get("training_seeds") or {}
        seeds = list(ts.get("seeds") or [None])
        n = len(seeds)
        if n < min_members:
            raise ValueError(f"{d}: trained with {n} member(s), {min_members} required")
        if n == 1:
            epochs = [int(ts["best_epochs"][0]) if ts.get("best_epochs") else int(res["best_epoch"])]
            paths = [d / "checkpoints" / f"epoch{epochs[0]:03d}.pt"]
        else:
            epochs = [int(e) for e in ts["best_epochs"]]
            if len(epochs) != n:
                raise ValueError(f"{d}: {n} training seeds but {len(epochs)} best epochs")
            paths = [d / "checkpoints" / f"seed{k}" / f"epoch{epochs[k]:03d}.pt" for k in range(n)]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise ValueError(f"{d}: {len(missing)} of {n} member checkpoints missing, e.g. {missing[0]}")
        want = sum(column_width(c) for c in D.COLUMNS[res["decider"]]) + 1
        members = []
        for p in paths:
            ck = torch.load(p, map_location="cpu", weights_only=False)      # a local file this project wrote
            if int(ck["n_in"]) != want:
                raise ValueError(f"{p}: the checkpoint takes {ck['n_in']} inputs, {res['decider']}'s columns give {want}")
            model = D.make_model(int(ck["n_in"]), tuple(ck["hidden"]))
            model.load_state_dict(ck["state"]); model.eval()
            members.append((model, np.asarray(ck["mean"]), np.asarray(ck["std"])))
        lam = res["lambda"]["lambda"]
        return cls(members, res["decider"], math.inf if lam is None else lam,
                   {"source": str(d), "fold": res.get("fold"), "best_epoch": res["best_epoch"],
                    "members": n, "training_seeds": seeds, "member_best_epochs": epochs})


class RandomDecider(_Decider):
    def __init__(self, decider: str, seed: int = 0, n_in: Optional[int] = None):
        import torch
        from vlar_train import deciders as D
        ensure_decider(decider)
        torch.manual_seed(seed)
        width = n_in or sum(column_width(c) for c in D.COLUMNS[decider]) + 1
        model = D.make_model(width); model.eval()
        super().__init__([(model, np.zeros(width), np.ones(width))], decider, 0.0,
                         {"source": f"dummy random weights, seed {seed}", "members": 1})
