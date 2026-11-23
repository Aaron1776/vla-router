"""Phase 2 cross-validation (SPEC §7-§8): every fold, every decider, from scratch.

    python -m vlar_train --records DIR --label-map labels.json --out runs/phase2 [--folds 0-9] [--deciders ...]
    python -m vlar_train --synthetic --out /tmp/p2        (dry run on synthetic points)

Per fold and decider: the profile's excluded points are dropped (amendment 22); train on the 8 training tasks,
early-stop on the validation task, choose λ on the validation task from the branch labels (`lam.choose`), predict
the test task (kept for Phase 4 evaluation, not used here). Everything lands in <out>/fold<i>/<decider>/ (a
checkpoint per epoch, best.json, result.json) and <out>/summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import folds as F
from . import lam as LAM
from .data import PointSet
from .deciders import BRANCHES, DECIDERS, inputs
from .train import TrainConfig, seed_for, train


def run_fold_decider(ps: PointSet, i: int, decider: str, out: Optional[Path], cfg: TrainConfig) -> dict:
    keep = ~ps.excluded[decider]
    m = F.masks(ps.task, i)
    x = inputs(ps.cols, ps.remaining_s[decider], decider)
    y, ym = ps.targets(decider)
    tr, va, te = (m[k] & keep for k in ("train", "val", "test"))
    d = None if out is None else out / f"fold{i}" / decider
    model = train(x[tr], y[tr], ym[tr], x[va], y[va], ym[va], seed=seed_for(i, decider), cfg=cfg,
                  ckpt_dir=None if d is None else d / "checkpoints")
    pv = model.predict(x[va])
    both = (ym[va][:, 0] > 0) & (ym[va][:, 1] > 0)          # λ needs labels of branches 1 and 2 at the point
    lam = LAM.choose(pv[both, 0], pv[both, 1], y[va][both, 0], y[va][both, 1], ps.can_escalate[va][both])
    res = {"fold": i, "decider": decider, "width": int(x.shape[1]), "best_epoch": model.best_epoch,
           "val_loss": model.val_loss[model.best_epoch], "epochs_run": len(model.val_loss),
           "points": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
                      "excluded": int((~keep).sum())},
           "lambda": lam}
    if d is not None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(res, indent=1, default=float))
        if te.any():
            np.savez(d / "test_predictions.npz", p=model.predict(x[te]), y=y[te], mask=ym[te],
                     snapshot_id=np.array([s for s, k in zip(ps.snapshot_id, te) if k]))
    return res


def run_cv(ps: PointSet, *, folds: Sequence[int] = range(F.N_TASKS), deciders: Sequence[str] = DECIDERS,
           out: Optional[Path] = None, cfg: TrainConfig = TrainConfig()) -> dict:
    results = [run_fold_decider(ps, i, dname, out, cfg) for i in folds for dname in deciders]
    summary = {"results": results, "branches": list(BRANCHES), "config": cfg.__dict__}
    for dname in deciders:
        rs = [r for r in results if r["decider"] == dname]
        summary[dname] = {"median_best_epoch": float(np.median([r["best_epoch"] for r in rs])),
                          "median_lambda": float(np.median([r["lambda"]["lambda"] for r in rs]))}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def _range(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m vlar_train")
    ap.add_argument("--records", help="local copy of the bucket (records/runs, events, points)")
    ap.add_argument("--label-map", help="the branch planner's label map (JSON)")
    ap.add_argument("--seeds", type=int, default=3, help="branch seeds per label (collection config branches.seeds)")
    ap.add_argument("--synthetic", action="store_true", help="dry run on synthetic points")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="0-9")
    ap.add_argument("--deciders", default=",".join(DECIDERS))
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    a = ap.parse_args(argv)
    if a.synthetic:
        from .data import synthetic
        ps = synthetic()
    else:
        if not (a.records and a.label_map):
            ap.error("give --records and --label-map (or --synthetic)")
        from .data import from_records
        ps = from_records(a.records, a.label_map, seeds=a.seeds)
    s = run_cv(ps, folds=_range(a.folds), deciders=a.deciders.split(","), out=Path(a.out),
               cfg=TrainConfig(epochs=a.epochs))
    for d in a.deciders.split(","):
        print(d, s[d])


if __name__ == "__main__":
    main()
