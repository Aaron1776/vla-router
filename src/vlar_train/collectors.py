"""The deciders that drive the DAgger round (SPEC §9): trained from scratch on all 10 tasks, each with the median
stopping epoch and the median λ of its 10 cross-validation folds.

    python -m vlar_train --variants 2x2 --collectors CV_OUT --out OUT   (+ the data options of the cross-validation)

`CV_OUT` is the folder the cross-validation of the same deciders wrote. With `--variants 2x2` the collectors are
`router_hist` and `cascader_hist`, the two history deciders; otherwise those named by `--deciders`. No fold is
trained and no λ is chosen in this mode: both numbers are read from the folds' `result.json`.

What a collector is:
- **Rows:** every labelled point of every task that its profile does not exclude, with the weights w = 1/π
  normalised to mean 1 over those rows. There is no validation task and no test task.
- **Stopping epoch:** the median over the folds of each fold's stopping epoch (itself the median of that fold's
  training seeds). A median ending in .5 (an even number of folds, or of seeds) is rounded half up, to the later
  epoch: a collector has ten tasks to learn from where a fold had eight, so when two epochs are equally supported
  it takes the longer training. Epochs count from 0, so stopping epoch E means E + 1 passes over the rows. There
  is no early stopping: the collector is the model as it is after that epoch.
- **Training seeds:** as in the folds, the mean of the same number of members, each trained from scratch with its
  own seed (the fold number in the seed's derivation is -1: no fold). The collector's probabilities are the mean of
  its members' probabilities.
- **Settings:** the training settings the cross-validation recorded in its `summary.json`, not the command line's.
- **λ:** the median of the folds' λ (`lam.median`: with an even count the two middle values are averaged when both
  are finite, otherwise the larger is taken). A median of +inf means the collector never escalates; that is
  recorded and printed, not changed.

Every fold of the decider must be there: a missing fold stops the run. Output, per collector, in
`OUT/collectors/<decider>/`: `collector.json` (the folds' numbers, the two medians, the rounding, the seeds, the
rows, each member's training loss; strict JSON) and `checkpoints/seed<k>/` (one checkpoint per epoch; the last one
is the member). A live collector loads every member, averages their probabilities and escalates when
p_one_call - p_continue exceeds λ and budget remains (§8's rule).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import folds as F
from . import lam as LAM
from .benefit import strict
from .data import PointSet
from .deciders import inputs, profile_of
from .train import TrainConfig, member_seeds, train_seeds, unit_mean

NO_FOLD = -1                 # the fold number in a collector's seeds
ROUNDING = "half up: a median ending in .5 goes to the later epoch"


class CollectorError(ValueError):
    pass


def round_epoch(median: float) -> int:
    """The stopping epoch from the folds' median: half up."""
    if not math.isfinite(median) or median < 0:
        raise CollectorError(f"no stopping epoch from a median of {median!r}")
    return int(math.floor(median + 0.5))


def config_of(cv_out: Path | str) -> TrainConfig:
    """The training settings the cross-validation ran with (its `summary.json`)."""
    f = Path(cv_out) / "summary.json"
    if not f.exists():
        raise CollectorError(f"no summary.json under {cv_out}: the cross-validation's settings are not known")
    cfg = dict(json.loads(f.read_text())["config"])
    cfg["hidden"] = tuple(cfg["hidden"])
    return TrainConfig(**cfg)


def plan(cv_out: Path | str, decider: str, folds: Sequence[int] = range(F.N_TASKS)) -> dict:
    """The two medians of a decider's folds, and what they come from."""
    rows = []
    for i in folds:
        f = Path(cv_out) / f"fold{i}" / decider / "result.json"
        if not f.exists():
            raise CollectorError(f"{decider}: fold {i} has no result under {cv_out}; a collector needs every fold")
        r = json.loads(f.read_text())
        lam = r["lambda"]["lambda"]
        rows.append({"fold": int(i), "best_epoch": float(r["best_epoch"]),
                     "lambda": float(lam) if not isinstance(lam, str) else float(lam.replace("Infinity", "inf")),
                     "lambda_source": r["lambda"].get("source")})
    med = float(np.median([x["best_epoch"] for x in rows]))
    lam = LAM.median([x["lambda"] for x in rows])
    return {"decider": decider, "folds": rows, "median_best_epoch": med, "stop_epoch": round_epoch(med),
            "rounding": ROUNDING, "epochs_trained": round_epoch(med) + 1, "lambda": lam,
            "never_escalates": lam == math.inf}


def train_collector(ps: PointSet, decider: str, pl: dict, cfg: TrainConfig, out: Optional[Path]) -> dict:
    prof = profile_of(decider)
    keep = ~ps.excluded[prof]
    x = inputs(ps.cols, ps.remaining_s[prof], decider)
    y, ym = ps.targets(prof)
    w = unit_mean(ps.weight[keep], int(keep.sum()))
    seeds = member_seeds(NO_FOLD, f"{decider}|collector", cfg.seeds)
    d = None if out is None else Path(out) / "collectors" / decider
    model = train_seeds(x[keep], y[keep], ym[keep], seeds=seeds, cfg=cfg, w_tr=w, stop_epoch=pl["stop_epoch"],
                        ckpt_dir=None if d is None else d / "checkpoints")
    res = {**pl, "profile": prof, "width": int(x.shape[1]), "seeds": seeds, "averaged": "predicted probabilities",
           "points": {"trained_on": int(keep.sum()), "excluded": int((~keep).sum()),
                      "tasks": sorted(int(t) for t in set(np.asarray(ps.task)[keep].tolist()))},
           "train_loss": [[float(v) for v in m.train_loss] for m in model.members],
           "config": {**cfg.__dict__, "hidden": list(cfg.hidden)}, "data": ps.report}
    if d is not None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "collector.json").write_text(json.dumps(strict(res), indent=1, allow_nan=False))
    res["model"] = model
    return res


def run(ps: PointSet, cv_out: Path | str, deciders: Sequence[str], out: Optional[Path | str], *,
        folds: Sequence[int] = range(F.N_TASKS)) -> dict:
    cfg = config_of(cv_out)
    plans = {d: plan(cv_out, d, folds) for d in deciders}                # every fold of every collector, first
    return {d: train_collector(ps, d, plans[d], cfg, None if out is None else Path(out)) for d in deciders}
