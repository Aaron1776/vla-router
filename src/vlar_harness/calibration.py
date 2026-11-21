"""Per-task time limits and the shared margin (docs/SPEC.md §5, §11 step 6; rule approved 2026-11-21, §17 item 9).

limit(task) = p90(task) x m, where p90(task) is the 90th percentile of X-VLA's successful-episode completion times
under profile `none` in the base rollouts (run uncut; times from their event logs), and m is the smallest candidate
in `time_limit.margin_candidates` (1.25, 1.5, 1.75, 2.0) such that at least `margin_rule_min_pi05_share` (80%) of
π0.5's successful parity episodes, pooled over tasks and timed under profile `none` with the network model, finish
within their task's limit. No candidate reaching the share is an error. The percentile is NumPy's default (linear)
method, computed here in pure Python with NumPy's arithmetic. A completion time is the clock right after the
success step.

    python -m vlar_harness.calibration --base base.jsonl ... --parity parity.jsonl ... [--write]

prints, per task, the limit and the share of π0.5 successes that fit, and with --write stores
`time_limit.per_task_s`, `margin` and `status` in the calibration file (config.write_calibration).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from . import clock as clk
from . import config as C
from . import eventlog

PERCENTILE = 90
BASE_MODES = ("routed", "xvla_only")         # base rollouts: X-VLA always-local in routed mode (identical runs)
PARITY_MODES = ("pi05_only",)


class CalibrationError(ValueError):
    pass


def percentile_linear(values: Iterable[float], q_percent: float) -> float:
    """`numpy.percentile(values, q_percent)` with the default 'linear' method, same floating-point steps."""
    x = sorted(float(v) for v in values)
    n = len(x)
    if n == 0:
        raise ValueError("percentile of no values")
    h = (n - 1) * (q_percent / 100)
    lo = min(max(math.floor(h), 0), n - 1)
    g = h - lo
    a, b = x[lo], x[min(lo + 1, n - 1)]
    d = b - a
    return b - d * (1 - g) if g >= 0.5 else a + d * g


@dataclass
class Times:
    """Per task: completion times of the successful episodes, and the number of episodes."""
    success_s: dict[int, list[float]] = field(default_factory=dict)
    episodes: dict[int, int] = field(default_factory=dict)


def completion_times(logs: Iterable[Sequence[Mapping]], charges: clk.Charges, *, modes: Sequence[str],
                     allow_pi05: bool, profile: str = "none") -> Times:
    out = Times()
    for events in logs:
        start = eventlog.check(events)
        if start["mode"] not in modes:
            raise CalibrationError(f"expected modes {tuple(modes)}, got a {start['mode']} episode")
        if start["time_limit_s"] is not None:
            raise CalibrationError("calibration uses uncut runs; this episode had a live time limit")
        kinds = {e["t"] for e in events}
        if kinds & {"restore", "branch"}:
            raise CalibrationError("calibration uses whole episodes, not branches or restored runs")
        if not allow_pi05 and "pi05_request" in kinds:
            raise CalibrationError("a base rollout must be X-VLA alone; this one called pi0.5")
        task = int(start["task_id"])
        out.episodes[task] = out.episodes.get(task, 0) + 1
        out.success_s.setdefault(task, [])
        s = clk.score(events, profile, charges)
        if s.success:
            out.success_s[task].append(s.time_s)
    return out


def choose_margin(p90_s: Mapping[int, float], pi05_success_s: Mapping[int, Sequence[float]],
                  candidates: Sequence[float], min_share: float) -> tuple[float, dict[float, float]]:
    """The smallest candidate m with at least `min_share` of π0.5's successes (pooled) within p90 x m."""
    total = sum(len(v) for v in pi05_success_s.values())
    if total == 0:
        raise CalibrationError("no successful pi0.5 parity episodes to calibrate the margin with")
    unknown = sorted(set(pi05_success_s) - set(p90_s))
    if unknown:
        raise CalibrationError(f"pi0.5 parity episodes for tasks without an X-VLA percentile: {unknown}")
    shares = {}
    for m in sorted(candidates):
        fit = sum(1 for t, times in pi05_success_s.items() for x in times if x <= p90_s[t] * m)
        shares[m] = fit / total
    for m in sorted(candidates):
        if shares[m] >= min_share:
            return m, shares
    raise CalibrationError(f"no margin candidate reaches {min_share:.0%} of pi0.5's successes: {shares}")


@dataclass
class TimeLimits:
    p90_s: dict[int, float]
    margin: float
    per_task_s: dict[int, float]
    share_by_margin: dict[float, float]
    rows: list[dict]


def calibrate(base_logs: Iterable[Sequence[Mapping]], parity_logs: Iterable[Sequence[Mapping]], cfg) -> TimeLimits:
    charges = clk.Charges.from_config(cfg)
    n_tasks = int(cfg["env"]["n_tasks"])
    xv = completion_times(base_logs, charges, modes=BASE_MODES, allow_pi05=False)
    missing = [t for t in range(n_tasks) if not xv.success_s.get(t)]
    if missing:
        raise CalibrationError(f"no successful X-VLA base rollouts for tasks {missing}")
    extra = sorted(set(xv.episodes) - set(range(n_tasks)))
    if extra:
        raise CalibrationError(f"base rollouts for unknown tasks {extra}")
    p90 = {t: percentile_linear(xv.success_s[t], PERCENTILE) for t in range(n_tasks)}
    pi = completion_times(parity_logs, charges, modes=PARITY_MODES, allow_pi05=True)
    tl = cfg["time_limit"]
    margin, shares = choose_margin(p90, pi.success_s, tl["margin_candidates"], float(tl["margin_rule_min_pi05_share"]))
    per_task = {t: p90[t] * margin for t in range(n_tasks)}
    rows = []
    for t in range(n_tasks):
        fits = sum(1 for x in pi.success_s.get(t, []) if x <= per_task[t])
        n_pi = len(pi.success_s.get(t, []))
        rows.append({"task": t, "xvla_successes": len(xv.success_s[t]), "xvla_episodes": xv.episodes.get(t, 0),
                     "p90_s": p90[t], "limit_s": per_task[t], "pi05_successes": n_pi,
                     "pi05_episodes": pi.episodes.get(t, 0), "pi05_fit": fits,
                     "pi05_fit_share": fits / n_pi if n_pi else float("nan")})
    return TimeLimits(p90, margin, per_task, shares, rows)


def write(tl: TimeLimits, name: str = C.DEFAULT_CONFIG, config_dir: Path = C.CONFIG_DIR, note: str = "") -> Path:
    return C.write_calibration({"time_limit": {"per_task_s": {int(t): float(v) for t, v in tl.per_task_s.items()},
                                               "margin": tl.margin, "status": "calibrated"}},
                               name, config_dir, note=note or "Phase 0 step 6: time limits from base and parity logs")


def report(tl: TimeLimits) -> str:
    lines = [f"margin m = {tl.margin} (smallest candidate with enough pi0.5 successes within the limit)",
             "share of pi0.5 successes within p90 x m, by candidate: "
             + ", ".join(f"{m}: {s:.1%}" for m, s in sorted(tl.share_by_margin.items())),
             "task  X-VLA succ/eps   p90 (s)   limit (s)   pi0.5 succ   fit   share"]
    for r in tl.rows:
        share = "  n/a" if r["pi05_successes"] == 0 else f"{r['pi05_fit_share']:.1%}"
        lines.append(f"{r['task']:>4}  {r['xvla_successes']:>5}/{r['xvla_episodes']:<5}  {r['p90_s']:>9.3f}  "
                     f"{r['limit_s']:>10.3f}  {r['pi05_successes']:>10}  {r['pi05_fit']:>4}  {share}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m vlar_harness.calibration")
    ap.add_argument("--base", nargs="+", required=True, help="JSON-lines event logs of the base rollouts")
    ap.add_argument("--parity", nargs="+", required=True, help="JSON-lines event logs of the pi0.5 parity run")
    ap.add_argument("--config", default=C.DEFAULT_CONFIG)
    ap.add_argument("--write", action="store_true", help="store the limits and margin in the calibration file")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    tl = calibrate(eventlog.read_logs(a.base), eventlog.read_logs(a.parity), cfg)
    print(report(tl))
    if a.write:
        print(f"written to {write(tl, a.config)}")


if __name__ == "__main__":
    main()
