"""Reporting by episode (SPEC §17 amendments 57 (5)-(6) and 68 (3)).

Points from one episode share its outcome, so they are not independent: in the low-failure tasks the failed-parent
points come from a handful of episodes (amendment 68 (3)). Nothing here changes training; it reports

- the effective sample size (Σw)²/Σw² of a set of weighted rows (amendment 57 (6));
- label statistics re-weighted to the population (amendment 57 (6)): the rescue rate (p_2 > p_1), the harm rate
  (p_2 < p_1) and the mean p_2 − p_1, as the pilot report defines them, and the positives (p_2 − p_1 ≥ 1/3,
  amendment 63 (2)) both as points and as distinct episodes with a positive (amendment 68 (3));
- intervals by cluster bootstrap: points from one episode are resampled together (amendment 57 (5)).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

POSITIVE = 1 / 3            # a positive labelled point: p_2 − p_1 ≥ 1/3 (amendment 63 (2))
RESAMPLES = 1000
LEVEL = 0.95


def ess(w: Sequence[float]) -> float:
    """Effective sample size (Σw)²/Σw² (amendment 57 (6)); n for equal weights, 0 for no rows."""
    w = np.asarray(w, dtype=np.float64)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def weighted_mean(v: np.ndarray, w: np.ndarray) -> float:
    return float((w * v).sum() / w.sum()) if len(v) else float("nan")


def cluster_bootstrap(stat: Callable[[np.ndarray], float], episode: Sequence, *, seed: int,
                      resamples: int = RESAMPLES, level: float = LEVEL) -> Optional[list[float]]:
    """A percentile interval for `stat(rows)` (rows: indices into the caller's arrays), resampling whole episodes
    with replacement so that points from one episode stay together (amendment 57 (5)). None when there are fewer
    than two episodes: one episode has no between-episode spread to resample."""
    ids, inv = np.unique(np.asarray(episode), return_inverse=True)
    if len(ids) < 2:
        return None
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(ids) + 1))
    rows_of = [order[bounds[k]:bounds[k + 1]] for k in range(len(ids))]
    rng = np.random.default_rng(seed)
    out = np.empty(resamples)
    for b in range(resamples):
        out[b] = stat(np.concatenate([rows_of[k] for k in rng.integers(len(ids), size=len(ids))]))
    out = out[np.isfinite(out)]
    if not len(out):
        return None
    lo, hi = np.percentile(out, [50 * (1 - level), 50 * (1 + level)])
    return [float(lo), float(hi)]


def sample_report(w: np.ndarray, episode: Sequence) -> dict:
    """How much data a set of rows is: points, distinct episodes and the effective sample size of its weights."""
    return {"points": int(len(w)), "episodes": int(len(set(np.asarray(episode).tolist()))), "ess": ess(w)}


def label_stats(y: np.ndarray, ym: np.ndarray, w: np.ndarray, episode: Sequence, *, seed: int,
                resamples: int = RESAMPLES) -> dict:
    """Over the rows with labels for branches 1 and 2 (amendment 57 (6), 68 (3)): rescue rate, harm rate and mean
    p_2 − p_1, re-weighted to the population (Σ w × value / Σ w), each with its episode-bootstrap interval; the
    positives as points and as distinct episodes with a positive (unweighted counts); the rows' effective sample
    size."""
    both = (ym[:, 0] > 0) & (ym[:, 1] > 0)
    d, w, ep = (y[:, 1] - y[:, 0])[both], np.asarray(w, dtype=np.float64)[both], np.asarray(episode)[both]
    pos = d >= POSITIVE - 1e-9
    out = {**sample_report(w, ep), "positives": int(pos.sum()), "positive_episodes": int(len(set(ep[pos].tolist())))}
    for name, v in (("rescue_rate", (d > 1e-9).astype(float)), ("harm_rate", (d < -1e-9).astype(float)),
                    ("mean_p2_minus_p1", d)):
        out[name] = weighted_mean(v, w)
        out[f"{name}_ci95"] = cluster_bootstrap(lambda r, v=v: weighted_mean(v[r], w[r]), ep, seed=seed,
                                                resamples=resamples)
    return out


def decision_stats(p0, p1, y0, y1, can, w, lam: float) -> Callable[[np.ndarray], tuple[float, float]]:
    """For a fixed λ: rows -> (gain, escalation rate) as §8 estimates them (weighted, amendment 57 (4))."""
    esc = np.asarray(can, bool) & ((np.asarray(p1, float) - np.asarray(p0, float)) > lam)
    d = np.where(esc, np.asarray(y1, float) - np.asarray(y0, float), 0.0)
    w = np.asarray(w, dtype=np.float64)
    return lambda r: (weighted_mean(d[r], w[r]), weighted_mean(esc[r].astype(float), w[r]))
