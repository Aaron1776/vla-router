"""Phase 2 cross-validation (SPEC §7-§8): every fold, every decider, from scratch.

    python -m vlar_train --records DIR --label-map labels.json --out runs/phase2 [--folds 0-9] [--deciders ...]
    python -m vlar_train --synthetic --out /tmp/p2        (dry run on synthetic points)

Per fold and decider: the profile's excluded points are dropped (amendment 22); train on the 8 training tasks,
early-stop on the validation task, choose λ on the validation task from the branch labels (`lam.choose`), predict
the test task (kept for Phase 4 evaluation, not used here). Sampling weights (amendment 57 (3)-(4)): w_i = 1/π_i,
normalised to mean 1 separately within the fold's training rows and its validation rows (the rows the decider
uses, after its profile's exclusions), in the loss, in early stopping and in λ's gain and escalation rate.

λ fallback (amendment 63 (5)): when the fold's validation task has fewer than 10 positive labelled points under the
decider's own profile (a positive is p_2 − p_1 ≥ 1/3, amendment 63 (2)), λ is chosen by §8's rule on cross-fitted
predictions instead: two auxiliary deciders, trained as the fold's own decider is but each on one half of the fold's
8 training tasks (halves drawn with a recorded seed), each predict the other half; λ is chosen on those out-of-fold
predictions together with the validation task's points and applied to the fold's decider. No test-task label enters.
Each result records which rule chose its λ, and the summary lists the folds that used the fallback. Everything lands in <out>/fold<i>/<decider>/ (a
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
from .train import TrainConfig, seed_for, train, unit_mean


POSITIVE = 1 / 3            # a positive labelled point: p_2 − p_1 ≥ 1/3 under the decider's profile (amendment 63 (2))
MIN_POSITIVES = 10          # fewer on the validation task: λ from cross-fitted predictions (amendment 63 (5))
HALVES_SEED = 63            # base of the recorded seed that draws each fold's two halves


def positives(y: np.ndarray, ym: np.ndarray) -> np.ndarray:
    """Rows with labels for branches 1 and 2 and p_2 − p_1 ≥ 1/3 (labels are thirds: tolerance for 1 − 2/3)."""
    return (ym[:, 0] > 0) & (ym[:, 1] > 0) & (y[:, 1] - y[:, 0] >= POSITIVE - 1e-9)


def halves(i: int, n_tasks: int = F.N_TASKS, base: int = HALVES_SEED) -> dict:
    """Fold i's training tasks in two halves, drawn with a recorded seed (amendment 63 (5)). The draw depends on the
    fold only, so every decider of a fold uses the same halves."""
    tasks = list(F.fold(i, n_tasks)["train"])
    seed = seed_for(i, "lambda-fallback-halves", base)
    order = [tasks[k] for k in np.random.default_rng(seed).permutation(len(tasks))]
    cut = len(order) // 2
    return {"seed": seed, "a": sorted(order[:cut]), "b": sorted(order[cut:])}


def cross_fitted_lambda(ps: PointSet, i: int, decider: str, cfg: TrainConfig, *, x, y, ym, tr, va, w_tr, w_va,
                        pv) -> dict:
    """λ by §8's rule on cross-fitted predictions (amendment 63 (5)). Two auxiliary deciders, trained as the fold's
    own decider is (from scratch, the same settings, weighted loss, early stopping on the fold's validation task),
    each on one half of the fold's training tasks, each predict the other half. λ is chosen on those out-of-fold
    predictions together with the validation task's points (scored by the fold's own decider, `pv`). Weights: each
    training row keeps the weight it has in the fold's training rows (mean 1 there) and each validation row its
    validation weight (mean 1 there). Only training and validation rows are read: no test-task label enters."""
    h = halves(i)
    task = np.asarray(ps.task)
    p_tr = np.full((int(tr.sum()), y.shape[1]), np.nan)
    tr_task = task[tr]
    for k, (fit, held) in enumerate(((h["a"], h["b"]), (h["b"], h["a"]))):
        fit_rows, held_in_tr = tr & np.isin(task, fit), np.isin(tr_task, held)
        aux = train(x[fit_rows], y[fit_rows], ym[fit_rows], x[va], y[va], ym[va],
                    seed=seed_for(i, f"{decider}|lambda-fallback-aux{k}"), cfg=cfg,
                    w_tr=ps.weight[fit_rows], w_va=w_va)
        p_tr[held_in_tr] = aux.predict(x[tr][held_in_tr])
    p = np.concatenate([p_tr, pv])
    yy, mm = np.concatenate([y[tr], y[va]]), np.concatenate([ym[tr], ym[va]])
    w = np.concatenate([w_tr, w_va])
    can = np.concatenate([ps.can_escalate[tr], ps.can_escalate[va]])
    both = (mm[:, 0] > 0) & (mm[:, 1] > 0)
    lam = LAM.choose(p[both, 0], p[both, 1], yy[both, 0], yy[both, 1], can[both], weight=w[both])
    return {**lam, "source": "cross_fitted", "halves": h,
            "points": {"out_of_fold": int(both[:len(p_tr)].sum()), "validation": int(both[len(p_tr):].sum())}}


def run_fold_decider(ps: PointSet, i: int, decider: str, out: Optional[Path], cfg: TrainConfig) -> dict:
    keep = ~ps.excluded[decider]
    m = F.masks(ps.task, i)
    x = inputs(ps.cols, ps.remaining_s[decider], decider)
    y, ym = ps.targets(decider)
    tr, va, te = (m[k] & keep for k in ("train", "val", "test"))
    d = None if out is None else out / f"fold{i}" / decider
    w_tr, w_va = unit_mean(ps.weight[tr], int(tr.sum())), unit_mean(ps.weight[va], int(va.sum()))   # amendment 57 (3)
    model = train(x[tr], y[tr], ym[tr], x[va], y[va], ym[va], seed=seed_for(i, decider), cfg=cfg,
                  ckpt_dir=None if d is None else d / "checkpoints", w_tr=w_tr, w_va=w_va)
    pv = model.predict(x[va])
    both = (ym[va][:, 0] > 0) & (ym[va][:, 1] > 0)          # λ needs labels of branches 1 and 2 at the point
    n_pos = int(positives(y[va], ym[va]).sum())
    if n_pos < MIN_POSITIVES:                               # amendment 63 (5)
        lam = cross_fitted_lambda(ps, i, decider, cfg, x=x, y=y, ym=ym, tr=tr, va=va, w_tr=w_tr, w_va=w_va, pv=pv)
    else:
        lam = {**LAM.choose(pv[both, 0], pv[both, 1], y[va][both, 0], y[va][both, 1], ps.can_escalate[va][both],
                            weight=w_va[both]), "source": "validation"}
    lam["validation_positives"] = n_pos
    res = {"fold": i, "decider": decider, "width": int(x.shape[1]), "best_epoch": model.best_epoch,
           "val_loss": model.val_loss[model.best_epoch], "epochs_run": len(model.val_loss),
           "points": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
                      "excluded": int((~keep).sum())},
           "lambda": lam}
    if d is not None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(res, indent=1, default=float))
        if te.any():
            np.savez(d / "test_predictions.npz", p=model.predict(x[te]), y=y[te], mask=ym[te], weight=ps.weight[te],
                     snapshot_id=np.array([s for s, k in zip(ps.snapshot_id, te) if k]))
    return res


def run_cv(ps: PointSet, *, folds: Sequence[int] = range(F.N_TASKS), deciders: Sequence[str] = DECIDERS,
           out: Optional[Path] = None, cfg: TrainConfig = TrainConfig()) -> dict:
    results = [run_fold_decider(ps, i, dname, out, cfg) for i in folds for dname in deciders]
    summary = {"results": results, "branches": list(BRANCHES), "config": cfg.__dict__}
    for dname in deciders:
        rs = [r for r in results if r["decider"] == dname]
        summary[dname] = {"median_best_epoch": float(np.median([r["best_epoch"] for r in rs])),
                          "median_lambda": float(np.median([r["lambda"]["lambda"] for r in rs])),
                          "lambda_fallback_folds": [r["fold"] for r in rs if r["lambda"]["source"] == "cross_fitted"]}
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
        ps = from_records(a.records, a.label_map)   # seeds from each label entry (amendment 49)
    s = run_cv(ps, folds=_range(a.folds), deciders=a.deciders.split(","), out=Path(a.out),
               cfg=TrainConfig(epochs=a.epochs))
    for d in a.deciders.split(","):
        print(d, s[d])


if __name__ == "__main__":
    main()
