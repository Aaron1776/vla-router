"""λ on the validation task (SPEC §8): the decision rule is "escalate if p_one_call - p_continue > λ and budget
remains". For each candidate λ, the estimated success gain over always-local is the mean over the validation points
of (branch 2 label - branch 1 label) where the rule escalates (0 elsewhere), and the escalation rate is the share of
points where it escalates. Chosen: the λ with the lowest escalation rate whose gain is at least `share` (95%
[ASSUMPTION]) of the best gain, among those whose rate stays within `max_rate`; if no λ gains anything, never
escalate (λ = +inf).

Sampling weights (amendment 57 (4)): with `weight` (w_i = 1/π_i, any scale), the gain and the escalation rate are
weighted means over the validation points, Σ w_i × value_i / Σ w_i. Without it every weight is 1."""

from __future__ import annotations

import math

import numpy as np

SHARE = 0.95


def _weights(weight, n: int) -> np.ndarray:
    if weight is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weight, float)
    if w.shape != (n,) or not (np.isfinite(w) & (w > 0)).all():
        raise ValueError("weights must be one finite positive number per point")
    return w


def curve(p_continue, p_one, y_continue, y_one, can_escalate, lambdas, weight=None) -> list[dict]:
    p0, p1 = np.asarray(p_continue, float), np.asarray(p_one, float)
    d = np.asarray(y_one, float) - np.asarray(y_continue, float)
    ok = np.asarray(can_escalate, bool)
    w = _weights(weight, len(d))
    total = float(w.sum())
    out = []
    for lam in lambdas:
        esc = ok & ((p1 - p0) > lam)
        out.append({"lambda": float(lam), "gain": float((w * np.where(esc, d, 0.0)).sum() / total) if len(d) else 0.0,
                    "rate": float((w * esc).sum() / total) if len(d) else 0.0})
    return out


def candidates(p_continue, p_one) -> np.ndarray:
    """Every threshold that changes a decision (the score values), plus the never-escalate end."""
    s = np.unique(np.asarray(p_one, float) - np.asarray(p_continue, float))
    return np.concatenate([[-math.inf], s, [math.inf]])


def choose(p_continue, p_one, y_continue, y_one, can_escalate, *, share: float = SHARE,
           max_rate: float = 1.0, weight=None) -> dict:
    pts = curve(p_continue, p_one, y_continue, y_one, can_escalate, candidates(p_continue, p_one), weight)
    allowed = [c for c in pts if c["rate"] <= max_rate + 1e-12]
    best = max((c["gain"] for c in allowed), default=0.0)
    if best <= 0.0:
        never = next(c for c in pts if c["lambda"] == math.inf)
        return {**never, "best_gain": best, "reason": "no λ gains over always-local"}
    ok = [c for c in allowed if c["gain"] >= share * best - 1e-12]
    pick = min(ok, key=lambda c: (c["rate"], -c["gain"], -c["lambda"]))
    return {**pick, "best_gain": best, "reason": f"lowest rate with >= {share:.0%} of the best gain"}
