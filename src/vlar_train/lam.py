"""λ on the validation task (SPEC §8): the decision rule is "escalate if p_one_call - p_continue > λ and budget
remains". For each candidate λ, the estimated success gain over always-local is the mean over the validation points
of (branch 2 label - branch 1 label) where the rule escalates (0 elsewhere), and the escalation rate is the share of
points where it escalates. Chosen: the λ with the lowest escalation rate whose gain is at least `share` (95%
[ASSUMPTION]) of the best gain, among those whose rate stays within `max_rate`; if no λ gains anything, never
escalate (λ = +inf).

Sampling weights (amendment 57 (4)): with `weight` (w_i = 1/π_i, any scale), the gain and the escalation rate are
weighted means over the validation points, Σ w_i × value_i / Σ w_i. Without it every weight is 1.

Three guards around the rule (none changes it where it is well supported):

- "Subject to the escalation rate staying within budget" (§8): `MAX_RATE`, the π0.5 budget as a share of the
  episode, 260 of 520 steps = 26 of the 52 decision points (§4) [ASSUMPTION: §8 names no number]. A λ whose estimated
  escalation rate is above it is not a candidate, so "always escalate" cannot be chosen.
- Support by episode [ASSUMPTION, proposed in the training-code review, for Miguel's decision]: points from one
  episode share its outcome (amendment 68 (3)), so with `min_episodes` = `MIN_EPISODES` a λ is a candidate only if
  the points it escalates include a point where branch 2 beat branch 1 from at least that many distinct episodes; a
  gain resting on a single episode chooses nothing. `min_episodes=1` is §8's rule exactly.
- The median over folds (§9: the DAgger collectors use the median λ of their 10 folds) never comes out NaN
  (`median`)."""

from __future__ import annotations

import math

import numpy as np

SHARE = 0.95
MAX_RATE = 260 / 520        # §4: at most 260 π0.5 steps of 520 [ASSUMPTION: the rate cap §8 leaves unnamed]
MIN_EPISODES = 2            # distinct episodes that must support a chosen λ [ASSUMPTION, see above]


def _weights(weight, n: int) -> np.ndarray:
    if weight is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weight, float)
    if w.shape != (n,) or not (np.isfinite(w) & (w > 0)).all():
        raise ValueError("weights must be one finite positive number per point")
    return w


def curve(p_continue, p_one, y_continue, y_one, can_escalate, lambdas, weight=None, episode=None) -> list[dict]:
    """Per candidate λ: the estimated gain and escalation rate, and its `support`: the distinct episodes (every
    point its own episode when none are given) with an escalated point where branch 2 beat branch 1."""
    p0, p1 = np.asarray(p_continue, float), np.asarray(p_one, float)
    d = np.asarray(y_one, float) - np.asarray(y_continue, float)
    ok = np.asarray(can_escalate, bool)
    w = _weights(weight, len(d))
    total = float(w.sum())
    ep = np.arange(len(d)) if episode is None else np.unique(np.asarray(episode), return_inverse=True)[1]
    if len(ep) != len(d):
        raise ValueError("one episode per point")
    out = []
    for lam in lambdas:
        esc = ok & ((p1 - p0) > lam)
        out.append({"lambda": float(lam), "gain": float((w * np.where(esc, d, 0.0)).sum() / total) if len(d) else 0.0,
                    "rate": float((w * esc).sum() / total) if len(d) else 0.0,
                    "support": int(len(np.unique(ep[esc & (d > 0)])))})
    return out


def candidates(p_continue, p_one) -> np.ndarray:
    """Every threshold that changes a decision (the score values), plus the never-escalate end."""
    s = np.unique(np.asarray(p_one, float) - np.asarray(p_continue, float))
    return np.concatenate([[-math.inf], s, [math.inf]])


def choose(p_continue, p_one, y_continue, y_one, can_escalate, *, share: float = SHARE,
           max_rate: float = 1.0, weight=None, episode=None, min_episodes: int = 1) -> dict:
    """§8's rule. The defaults are the bare rule (no rate cap, no support guard); `run` passes `MAX_RATE`, the
    points' episodes and `MIN_EPISODES`."""
    pts = curve(p_continue, p_one, y_continue, y_one, can_escalate, candidates(p_continue, p_one), weight, episode)
    never = next(c for c in pts if c["lambda"] == math.inf)
    allowed = [c for c in pts if c["rate"] <= max_rate + 1e-12]
    if min_episodes > 1:
        supported = [c for c in allowed if c["support"] >= min_episodes or c is never]
        if max((c["gain"] for c in supported), default=0.0) <= 0.0 < max((c["gain"] for c in allowed), default=0.0):
            return {**never, "best_gain": 0.0,
                    "reason": f"no gaining λ is supported by {min_episodes} episodes: never escalate"}
        allowed = supported
    best = max((c["gain"] for c in allowed), default=0.0)
    if best <= 0.0:
        return {**never, "best_gain": best, "reason": "no λ gains over always-local"}
    ok = [c for c in allowed if c["gain"] >= share * best - 1e-12]
    pick = min(ok, key=lambda c: (c["rate"], -c["gain"], -c["lambda"]))
    return {**pick, "best_gain": best, "reason": f"lowest rate with >= {share:.0%} of the best gain"}


def median(lambdas) -> float:
    """The median of the folds' λ (§9), never NaN. With an even count the two middle values are averaged when both
    are finite; when one of them is infinite (a fold that never escalates, or one that always does) their mean is
    undefined or infinite, and the larger of the two is taken: fewer escalations, which §4 prefers."""
    v = sorted(float(x) for x in lambdas)
    if not v or any(math.isnan(x) for x in v):
        raise ValueError(f"the median needs at least one λ and no NaN: {v}")
    lo, hi = v[(len(v) - 1) // 2], v[len(v) // 2]
    return (lo + hi) / 2 if math.isfinite(lo) and math.isfinite(hi) else hi
