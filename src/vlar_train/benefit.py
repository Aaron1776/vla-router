"""The planned analysis of the 2 x 2 (memory by X-VLA information): the benefit curve and its contrasts
(the 2 x 2 pre-registration amendment of SPEC §17, its items (5) and (6); the amendment's number is assigned
when it is logged, so nothing here depends on it).

    python -m vlar_train.benefit --out TRAINING_OUT [--report DIR] [--resamples 1000]

Reads what `python -m vlar_train --variants 2x2 --out TRAINING_OUT` wrote: per fold and variant, the held-out test
task's predictions (`test_predictions.npz`: p, y, mask, weight, episode, snapshot_id, can_escalate) and the
validation-chosen λ (`result.json`). Writes `benefit.json` (strict JSON: an undefined value is null, an infinite λ
the string "Infinity") and `benefit.txt`. Everything below is fixed before any
Phase 1 label is opened; nothing here chooses an analysis from the data.

Per test task (fold i tests task i), on the points with labels for branches 1 and 2, with w = 1/π:
- score = p_one_call - p_continue; a decider escalates the points whose score exceeds a threshold;
- rate = Σ w × escalated / Σ w; gain = Σ w × escalated × (b2 - b1) / Σ w (the gain per offered point of §8's λ
  rule, `lam.curve`; not the mean over the escalated points);
- the curve: gain against rate over every threshold that changes a decision (`lam.candidates`), up to rate 0.5;
- gain at the fixed rates 10% and 20%: read from the curve by linear interpolation where tied scores make the rate
  jump over the target (the same as escalating that fraction of the tied points), 0 at rate 0. These thresholds
  are quantiles of the test task's own scores: test inputs, no test label;
- gain and rate at the validation-chosen λ, as the training code chose it; "never escalate" is gain 0 at rate 0;
- beside each: random escalation at the same rate (rate × the weighted mean of b2 - b1), and the gain's excess over it;
- secondary: step-stratified AUROC of 1 - p_continue for "branch 1 succeeded in at most one of its three seeds"
  (pairs of points at the same step only, each pair weighted by the product of its two weights).

Pooling: the unweighted mean of the per-task values, each task once. A task with no usable test point is left out
and named; no task is dropped for having few or no positive points. Intervals: whole episodes resampled within
each task (the draws of `stats.cluster_bootstrap`), 1,000 resamples from a recorded seed per task, the same
resampled episodes for all four variants, each task's thresholds held at their full-sample values. A task whose
points come from one episode has nothing to resample and enters every resample unchanged.

Contrasts, paired on the same points and resamples: memory (`router_hist` - `router`); X-VLA information given
memory (`cascader_hist` - `router_hist`); X-VLA information without memory (`cascader_nospread` - `router`); their
interaction (given minus without). Primary: X-VLA information given memory, gain at the 10% rate. Everything else
is secondary, reported without a correction for multiple comparisons, and the report counts the intervals.

Every labelled point is one where escalation is offered (the branch plan's eligibility); this is asserted, from the
saved `can_escalate` where the predictions carry it.

`check_same_profiles` is run by the training entry point before anything is trained: the router's and
`cascader_nospread`'s profiles must give identical labels, exclusions and remaining time, or the four variants
would not be on the same rows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from . import folds as F
from . import lam as LAM
from . import stats as ST
from .train import seed_for

VARIANTS = ("router", "router_hist", "cascader_nospread", "cascader_hist")
CONTRASTS = {"memory": ("router_hist", "router"),
             "xvla_given_memory": ("cascader_hist", "router_hist"),
             "xvla_without_memory": ("cascader_nospread", "router")}
INTERACTION = ("xvla_given_memory", "xvla_without_memory")
PRIMARY = ("xvla_given_memory", "gain_at_10")
RATES = (0.10, 0.20)
CAP = LAM.MAX_RATE                      # the curve is shown up to the budget's share
FAIL_AT_MOST = 1 / 3                    # branch 1 succeeded in at most one of three seeds
SEED_BASE = 0                           # base of the recorded bootstrap seeds
SEED_LABEL = "2x2-preregistration-bootstrap"   # with the task and the base, what each task's seed is derived from
METRICS = ("gain_at_10", "gain_at_20", "gain_at_lambda", "rate_at_lambda", "excess_at_10", "excess_at_20",
           "excess_at_lambda", "failure_auroc")
GRID = np.round(np.linspace(0.0, CAP, 51), 4)


class AnalysisError(ValueError):
    pass


# ---------------------------------------------------------------- the check run before training

def check_same_profiles(ps, a: str = "router", b: str = "cascader_nospread") -> dict:
    """The two profiles must give identical labels, exclusions and remaining time at every point. Returns the
    counts compared; raises `AnalysisError` naming what differs."""
    diffs = {}
    for name, x, y in (("labels", ps.y[a], ps.y[b]), ("exclusions", ps.excluded[a], ps.excluded[b]),
                       ("remaining time", ps.remaining_s[a], ps.remaining_s[b])):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            diffs[name] = f"shapes {x.shape} and {y.shape}"
            continue
        same = (x == y) | ((x != x) & (y != y)) if x.dtype.kind == "f" else (x == y)
        rows = np.flatnonzero(~same.reshape(len(x), -1).all(axis=1))
        if len(rows):
            diffs[name] = f"{len(rows)} of {len(x)} points differ (first: {ps.snapshot_id[int(rows[0])]})"
    if diffs:
        raise AnalysisError(f"profiles {a} and {b} do not give the same " + "; ".join(f"{k}: {v}" for k, v in diffs.items())
                            + ". The four variants would not be on the same rows: stop and report this first")
    return {"profiles": [a, b], "points": int(len(ps.task)), "identical": ["labels", "exclusions", "remaining time"]}


# ---------------------------------------------------------------- reading a training run

def load_task(out: Path, task: int, variants: Sequence[str] = VARIANTS) -> Optional[dict]:
    """The test task's shared rows and each variant's predictions and λ; None when the fold wrote no test
    predictions. The four variants must be on identical rows."""
    per = {}
    for v in variants:
        d = Path(out) / f"fold{task}" / v
        if not (d / "test_predictions.npz").exists():
            per[v] = None
            continue
        z = np.load(d / "test_predictions.npz")
        per[v] = {k: z[k] for k in z.files}
        per[v]["lambda"] = float(json.loads((d / "result.json").read_text())["lambda"]["lambda"])
    have = [v for v in variants if per[v] is not None]
    if not have:
        return None
    if len(have) != len(variants):
        raise AnalysisError(f"task {task}: test predictions for {have} but not for {[v for v in variants if v not in have]}")
    ref = per[variants[0]]
    for v in variants[1:]:
        for k in ("snapshot_id", "y", "mask", "weight", "episode"):
            if not np.array_equal(per[v][k], ref[k], equal_nan=(ref[k].dtype.kind == "f")):
                raise AnalysisError(f"task {task}: {v} and {variants[0]} are not on the same test rows ({k} differs)")
    usable = (ref["mask"][:, 0] > 0) & (ref["mask"][:, 1] > 0)
    can = np.asarray(ref["can_escalate"], bool) if "can_escalate" in ref else np.ones(len(usable), bool)
    if not can[usable].all():
        raise AnalysisError(f"task {task}: {int((~can[usable]).sum())} labelled points where escalation is not offered")
    if not usable.any():
        return None
    sid = [str(s) for s in ref["snapshot_id"][usable]]
    step = np.array([10 * int(s.rsplit("/dp", 1)[1]) if "/dp" in s else -1 for s in sid])
    return {"task": task, "n": int(usable.sum()), "snapshot_id": sid, "step": step,
            "w": np.asarray(ref["weight"], float)[usable], "episode": np.asarray(ref["episode"])[usable],
            "y0": np.asarray(ref["y"], float)[usable, 0], "y1": np.asarray(ref["y"], float)[usable, 1],
            "can_escalate_saved": "can_escalate" in ref,
            "p": {v: np.asarray(per[v]["p"], float)[usable] for v in variants},
            "lambda": {v: per[v]["lambda"] for v in variants}}


# ---------------------------------------------------------------- the metric

def escalation_at_rate(score: np.ndarray, w: np.ndarray, rate: float) -> np.ndarray:
    """Per point, the share of it escalated when the highest scores are escalated up to weighted rate `rate`: 1
    above the boundary score, 0 below, and at the boundary score the fraction that makes the rate exact. The gain
    this gives equals linear interpolation of the curve between the two neighbouring thresholds."""
    score, w = np.asarray(score, float), np.asarray(w, float)
    total, above = float(w.sum()), 0.0
    for b in np.unique(score)[::-1]:
        tie = score == b
        at = float(w[tie].sum()) / total
        if above + at >= rate - 1e-12:
            f = min(max((rate - above) / at, 0.0), 1.0)
            return (score > b).astype(float) + f * tie
        above += at
    return np.ones(len(score))


def curve_points(t: Mapping, v: str) -> tuple[np.ndarray, np.ndarray]:
    """(rate, gain) of `lam.curve` at every threshold that changes a decision, by increasing rate."""
    p = t["p"][v]
    pts = LAM.curve(p[:, 0], p[:, 1], t["y0"], t["y1"], np.ones(t["n"], bool), LAM.candidates(p[:, 0], p[:, 1]),
                    weight=t["w"], episode=t["episode"])
    rate, gain = np.array([c["rate"] for c in pts]), np.array([c["gain"] for c in pts])
    order = np.argsort(rate, kind="stable")
    rate, gain = rate[order], gain[order]
    keep = np.r_[True, np.diff(rate) > 0]
    return rate[keep], gain[keep]


def _pairs(t: Mapping, v: str) -> tuple[np.ndarray, np.ndarray]:
    """Pair matrices of the step-stratified AUROC: [i, j] counts a failing point i against a non-failing point j at
    the same step (1 if i scores higher, 0.5 on a tie), and the matrix of the pairs that exist."""
    fail = t["y0"] <= FAIL_AT_MOST + 1e-9
    s = 1.0 - t["p"][v][:, 0]
    exists = (fail[:, None] & ~fail[None, :] & (t["step"][:, None] == t["step"][None, :])).astype(float)
    wins = exists * ((s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :]))
    return wins, exists


def prepare(t: Mapping, variants: Sequence[str] = VARIANTS) -> dict:
    """Per variant, the fixed per-point escalation shares (thresholds held at their full-sample values) and the
    AUROC's pair matrices."""
    d = t["y1"] - t["y0"]
    out = {"d": d, "w": t["w"], "esc": {}, "auc": {}}
    for v in variants:
        score = t["p"][v][:, 1] - t["p"][v][:, 0]
        out["esc"][v] = {"10": escalation_at_rate(score, t["w"], RATES[0]),
                         "20": escalation_at_rate(score, t["w"], RATES[1]),
                         "lambda": (score > t["lambda"][v]).astype(float)}
        out["auc"][v] = _pairs(t, v)
    return out


def task_metrics(pre: Mapping, v: str, counts: Optional[np.ndarray] = None) -> list[float]:
    """The METRICS of one variant on one task, for the rows as they are (`counts` None) or resampled (`counts`:
    how many times each row is drawn)."""
    cw = pre["w"] if counts is None else pre["w"] * counts
    total = float(cw.sum())
    mean_d = float((cw * pre["d"]).sum()) / total
    g, r = {}, {}
    for k, e in pre["esc"][v].items():
        g[k] = float((cw * e * pre["d"]).sum()) / total
        r[k] = float((cw * e).sum()) / total
    wins, exists = pre["auc"][v]
    den = float(cw @ exists @ cw)
    auc = float(cw @ wins @ cw) / den if den > 0 else float("nan")
    return [g["10"], g["20"], g["lambda"], r["lambda"], g["10"] - r["10"] * mean_d, g["20"] - r["20"] * mean_d,
            g["lambda"] - r["lambda"] * mean_d, auc]


def resample_counts(episode: Sequence, seed: int, resamples: int) -> np.ndarray:
    """(resamples, n) row counts of whole-episode resamples: the draws `stats.cluster_bootstrap` makes with this
    seed. With fewer than two episodes there is nothing to resample: every row once."""
    ids, inv = np.unique(np.asarray(episode), return_inverse=True)
    if len(ids) < 2:
        return np.ones((resamples, len(inv)))
    rng = np.random.default_rng(seed)
    out = np.empty((resamples, len(inv)))
    for b in range(resamples):
        out[b] = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))[inv]
    return out


def _interval(draws: np.ndarray, level: float = ST.LEVEL) -> Optional[list[float]]:
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return None
    lo, hi = np.percentile(draws, [50 * (1 - level), 50 * (1 + level)])
    return [float(lo), float(hi)]


def _mean(a: np.ndarray, axis: int) -> np.ndarray:
    """Mean over the tasks where the value is defined (NaN where it is defined for none)."""
    ok = np.isfinite(a)
    n = ok.sum(axis=axis)
    return np.where(n > 0, np.where(ok, a, 0.0).sum(axis=axis) / np.maximum(n, 1), np.nan)


# ---------------------------------------------------------------- the analysis

def analyse(out: Path | str, *, tasks: Sequence[int] = range(F.N_TASKS), variants: Sequence[str] = VARIANTS,
            resamples: int = ST.RESAMPLES, seed_base: int = SEED_BASE) -> dict:
    loaded = {t: load_task(Path(out), t, variants) for t in tasks}
    used = [t for t in tasks if loaded[t] is not None]
    if not used:
        raise AnalysisError(f"no test predictions of {list(variants)} under {out}")
    V, M = len(variants), len(METRICS)
    point = np.full((len(used), V, M), np.nan)
    boot = np.full((resamples, len(used), V, M), np.nan)
    per_task, grid = [], np.full((len(used), V, len(GRID)), np.nan)
    for a, t in enumerate(used):
        data = loaded[t]
        pre = prepare(data, variants)
        seed = seed_for(t, SEED_LABEL, seed_base)
        counts = resample_counts(data["episode"], seed, resamples)
        d = data["y1"] - data["y0"]
        for b, v in enumerate(variants):
            point[a, b] = task_metrics(pre, v)
            for k in range(resamples):
                boot[k, a, b] = task_metrics(pre, v, counts[k])
            rate, gain = curve_points(data, v)
            grid[a, b] = np.interp(GRID, rate, gain)
        positive = d >= ST.POSITIVE - 1e-9
        per_task.append({"task": int(t), "points": data["n"], "episodes": int(len(set(data["episode"].tolist()))),
                         "positives": int(positive.sum()),
                         "positive_episodes": int(len(set(data["episode"][positive].tolist()))),
                         "ess": ST.ess(data["w"]), "mean_b2_minus_b1": ST.weighted_mean(d, data["w"]),
                         "bootstrap_seed": int(seed), "can_escalate_saved": bool(data["can_escalate_saved"]),
                         "lambda": {v: data["lambda"][v] for v in variants},
                         "random_at_10": RATES[0] * ST.weighted_mean(d, data["w"]),
                         "random_at_20": RATES[1] * ST.weighted_mean(d, data["w"]),
                         "values": {v: dict(zip(METRICS, map(float, point[a, b]))) for b, v in enumerate(variants)}})
    pooled, pooled_boot = _mean(point, 0), _mean(boot, 1)                  # (V, M) and (resamples, V, M)
    idx = {v: b for b, v in enumerate(variants)}
    res = {"analysis": "the 2 x 2 pre-registration amendment: memory by X-VLA information, benefit curve and contrasts",
           "variants": list(variants), "metrics": list(METRICS), "rates": list(RATES), "curve_cap": CAP,
           "resamples": int(resamples), "seed_base": int(seed_base), "seed_label": SEED_LABEL, "level": ST.LEVEL,
           "tasks_used": [int(t) for t in used], "tasks_left_out": [int(t) for t in tasks if loaded[t] is None],
           "failure_auroc_undefined_tasks": {v: [int(t) for a, t in enumerate(used) if not np.isfinite(point[a, idx[v], -1])]
                                             for v in variants},
           "per_task": per_task, "pooled": {}, "contrasts": {},
           "curve": {"rate": [float(x) for x in GRID],
                     "gain": {v: [float(x) for x in _mean(grid[:, idx[v]], 0)] for v in variants}}}
    for v in variants:
        res["pooled"][v] = {m: {"value": float(pooled[idx[v], j]), "ci95": _interval(pooled_boot[:, idx[v], j])}
                            for j, m in enumerate(METRICS)}

    def contrast(pos: str, neg: str) -> tuple[np.ndarray, np.ndarray]:
        return pooled[idx[pos]] - pooled[idx[neg]], pooled_boot[:, idx[pos]] - pooled_boot[:, idx[neg]]
    have = {k: contrast(*pair) for k, pair in CONTRASTS.items() if all(x in idx for x in pair)}
    if all(k in have for k in INTERACTION):
        have["interaction"] = (have[INTERACTION[0]][0] - have[INTERACTION[1]][0],
                               have[INTERACTION[0]][1] - have[INTERACTION[1]][1])
    for k, (val, draws) in have.items():
        res["contrasts"][k] = {m: {"value": float(val[j]), "ci95": _interval(draws[:, j]),
                                   "primary": (k, m) == PRIMARY} for j, m in enumerate(METRICS)}
    res["primary"] = {"contrast": PRIMARY[0], "metric": PRIMARY[1],
                      **({k: v for k, v in res["contrasts"][PRIMARY[0]][PRIMARY[1]].items() if k != "primary"}
                         if PRIMARY[0] in res["contrasts"] else {"value": None, "ci95": None})}
    res["intervals_reported"] = int(len(variants) * M + len(res["contrasts"]) * M)
    return res


def _fmt(cell: Mapping) -> str:
    ci = cell.get("ci95")
    v = cell["value"]
    if v is None or not math.isfinite(v):
        return "undefined"
    return f"{v:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else f"{v:+.4f} [no interval]"


def table(res: Mapping) -> str:
    cols = ("gain_at_10", "gain_at_20", "gain_at_lambda", "rate_at_lambda", "failure_auroc")
    lines = ["Benefit analysis of the 2 x 2 pre-registration amendment (gain per offered point, inclusion-weighted; unweighted mean over tasks;",
             f"95% intervals from {res['resamples']} whole-episode resamples within task)", "",
             f"tasks used: {res['tasks_used']}; left out (no usable test point): {res['tasks_left_out'] or 'none'}",
             "", "variant | " + " | ".join(cols)]
    for v in res["variants"]:
        lines.append(f"{v} | " + " | ".join(_fmt(res["pooled"][v][m]) for m in cols))
    lines += ["", "excess over random escalation at the same rate | at 10% | at 20% | at lambda"]
    for v in res["variants"]:
        lines.append(f"{v} | " + " | ".join(_fmt(res["pooled"][v][m]) for m in ("excess_at_10", "excess_at_20", "excess_at_lambda")))
    lines += ["", "contrast | " + " | ".join(cols)]
    for k, row in res["contrasts"].items():
        lines.append(f"{k} | " + " | ".join(_fmt(row[m]) for m in cols))
    p = res["primary"]
    lines += ["", f"PRIMARY: {p['contrast']}, {p['metric']}: {_fmt(p) if p['value'] is not None else 'not available'}",
              f"every other number is secondary, with no correction for multiple comparisons; "
              f"intervals reported: {res['intervals_reported']}", "",
              "task | points | episodes | positives (episodes) | mean b2-b1 | " + " | ".join(f"gain at 10%, {v}" for v in res["variants"])]
    for t in res["per_task"]:
        lines.append(f"{t['task']} | {t['points']} | {t['episodes']} | {t['positives']} ({t['positive_episodes']}) | "
                     f"{t['mean_b2_minus_b1']:+.4f} | " + " | ".join(f"{t['values'][v]['gain_at_10']:+.4f}" for v in res["variants"]))
    undefined = {v: ts for v, ts in res["failure_auroc_undefined_tasks"].items() if ts}
    if undefined:
        lines += ["", f"failure AUROC undefined (no step with both outcomes) in tasks: {undefined}"]
    return "\n".join(lines) + "\n"


def strict(obj):
    """`obj` as strict JSON can hold it: an undefined value (NaN) becomes null, and an infinite one (a "never
    escalate" λ is +inf) the string "Infinity" or "-Infinity". Finite numbers are unchanged."""
    if isinstance(obj, Mapping):
        return {str(k): strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return strict(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if math.isfinite(x) else None if math.isnan(x) else "Infinity" if x > 0 else "-Infinity"
    return obj


def write_report(res: Mapping, dst: Path | str) -> str:
    """Write `benefit.json` (strict JSON, `strict`) and `benefit.txt` into `dst`; returns the text table."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "benefit.json").write_text(json.dumps(strict(res), indent=1, allow_nan=False))
    text = table(res)
    (dst / "benefit.txt").write_text(text)
    return text


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m vlar_train.benefit", description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="the folder a `python -m vlar_train --variants 2x2` run wrote")
    ap.add_argument("--report", default=None, help="where to write benefit.json and benefit.txt (default: --out)")
    ap.add_argument("--resamples", type=int, default=ST.RESAMPLES)
    a = ap.parse_args(argv)
    res = analyse(a.out, resamples=a.resamples)
    print(write_report(res, a.report or a.out), end="")


if __name__ == "__main__":
    main()
