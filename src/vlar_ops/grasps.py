"""The grasp check after the base rollouts, before the freeze (approved by the user 2026-11-23 with the gripper
procedure, `vlar_ops.step2.gripper`): `vlar-ops grasps --root <local copy of the bucket> --out grasps.json`.

Phase 0 step 2 calibrates the finger-width thresholds on an empty gripper (open, then closed on nothing). Real grasps
close on objects, so their widths are what the gripper-mismatch signals (§6: a step commanded closed whose fingers are
fully shut, or partly open) actually classify. This check reports where the base rollouts' commanded-closed steps
fall relative to the two thresholds (steps table: `action[6] > 0`; width = qpos[0] - qpos[1]) and flags the
calibration for the user when they do not separate cleanly. It decides nothing: the flag asks for a look before the
freeze.

- Settled steps: commanded closed for at least `settle_steps` consecutive steps up to and including this one, so the
  fingers have finished moving (the closing transient crosses both thresholds on every grasp and says nothing about
  them). Reported for all commanded-closed steps too.
- Clean separation: few settled widths lie close to a threshold, where a millimetre decides between "shut" and
  "holding something" (or between "holding" and "open"). Close means within `band` of the open-shut range
  (`open_width_m - shut_width_m`) on either side of a threshold; the flag goes up when more than `limit` of the
  settled steps are that close to either threshold. Defaults: band 5% (half the calibration margin), limit 2%.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

SETTLE_STEPS = 10           # half a second at LIBERO's 20 Hz
BAND = 0.05                 # of the open-shut range, on either side of each threshold
LIMIT = 0.02                # share of settled commanded-closed steps allowed that close to a threshold
HIST_BIN_M = 0.001


def _settled(closed: np.ndarray, settle: int) -> np.ndarray:
    """Per step: commanded closed for at least `settle` consecutive steps up to and including it."""
    run = np.zeros(len(closed), dtype=np.int64)
    n = 0
    for i, c in enumerate(closed):
        n = n + 1 if c else 0
        run[i] = n
    return run >= settle


def _summary(width: np.ndarray, shut: float, open_: float, band_m: float) -> dict[str, Any]:
    n = int(width.size)
    if not n:
        return {"n": 0}
    q = {f"p{p}": float(np.percentile(width, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
    near_shut = np.abs(width - shut) <= band_m
    near_open = np.abs(width - open_) <= band_m
    return {"n": n, "shut_frac": float(np.mean(width <= shut)),
            "partial_frac": float(np.mean((width > shut) & (width < open_))),
            "open_frac": float(np.mean(width >= open_)), "near_shut_frac": float(near_shut.mean()),
            "near_open_frac": float(near_open.mean()), "width_m": q, "min_m": float(width.min()),
            "max_m": float(width.max())}


def grasp_report(runs: Iterable[Mapping], steps: Iterable[Mapping], *, shut_width_m: float, open_width_m: float,
                 phase: str = "base", settle_steps: int = SETTLE_STEPS, band: float = BAND,
                 limit: float = LIMIT) -> dict[str, Any]:
    """The report from runs-table and steps-table rows (dicts: `run_id`, `phase`, `task_id`; `run_id`, `step`,
    `action` (7), `gripper_qpos` (2))."""
    if not shut_width_m < open_width_m:
        raise ValueError("gripper thresholds need shut_width_m < open_width_m")
    task_of = {r["run_id"]: int(r["task_id"]) for r in runs if r["phase"] == phase}
    if not task_of:
        raise ValueError(f"no {phase!r} runs in the records")
    by_run: dict[str, list] = {}
    for s in steps:
        if s["run_id"] in task_of:
            by_run.setdefault(s["run_id"], []).append(s)
    band_m = band * (open_width_m - shut_width_m)
    widths, settled_widths = [], []
    per_task: dict[int, dict[str, list]] = {}
    for run_id, rows in by_run.items():
        rows.sort(key=lambda r: int(r["step"]))
        act = np.asarray([r["action"] for r in rows], dtype=np.float64).reshape(-1, 7)
        qpos = np.asarray([r["gripper_qpos"] for r in rows], dtype=np.float64).reshape(-1, 2)
        closed = act[:, 6] > 0
        width = qpos[:, 0] - qpos[:, 1]
        settled = _settled(closed, settle_steps)
        widths.append(width[closed])
        settled_widths.append(width[settled])
        t = per_task.setdefault(task_of[run_id], {"closed": [], "settled": []})
        t["closed"].append(width[closed])
        t["settled"].append(width[settled])
    cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0)   # noqa: E731
    w_all, w_set = cat(widths), cat(settled_widths)
    settled = _summary(w_set, shut_width_m, open_width_m, band_m)
    reasons = []
    if not settled["n"]:
        reasons.append(f"no step was commanded closed for {settle_steps} consecutive steps")
    else:
        for side in ("shut", "open"):
            frac = settled[f"near_{side}_frac"]
            if frac > limit:
                reasons.append(f"{frac:.1%} of settled commanded-closed steps lie within {band_m * 1000:.2f} mm of "
                               f"the {side} threshold (limit {limit:.0%})")
    top = max(float(w_all.max()) if w_all.size else 0.0, open_width_m) + 2 * HIST_BIN_M
    edges = np.arange(0.0, top + HIST_BIN_M, HIST_BIN_M)
    hist, _ = np.histogram(np.clip(w_set, 0.0, edges[-1]), bins=edges)
    return {
        "phase": phase, "runs": len(by_run), "thresholds": {"shut_width_m": shut_width_m, "open_width_m": open_width_m},
        "rule": {"settle_steps": settle_steps, "band": band, "band_m": band_m, "limit": limit},
        "flag": bool(reasons), "reasons": reasons,
        "commanded_closed": _summary(w_all, shut_width_m, open_width_m, band_m), "settled": settled,
        "per_task": {str(k): {"closed": _summary(cat(v["closed"]), shut_width_m, open_width_m, band_m),
                              "settled": _summary(cat(v["settled"]), shut_width_m, open_width_m, band_m)}
                     for k, v in sorted(per_task.items())},
        "settled_histogram": {"bin_m": HIST_BIN_M, "counts": hist.tolist()},
    }


def thresholds_from_calibration() -> tuple[float, float]:
    from vlar_harness.config import load_config
    g = load_config()["measurement"]["gripper_thresholds"]
    if g.get("status") != "calibrated":
        raise ValueError(f"gripper thresholds are {g.get('status')!r} in the calibration file (Phase 0 step 2)")
    return float(g["shut_width_m"]), float(g["open_width_m"])


def run(root: Path, out: Path, *, phase: str = "base", shut: Optional[float] = None, open_: Optional[float] = None,
        **rule) -> dict:
    from vlar_harness.dataset import read_table
    if shut is None or open_ is None:
        shut, open_ = thresholds_from_calibration()
    runs = read_table(root, "runs").select(["run_id", "phase", "task_id"]).to_pylist()
    steps = read_table(root, "steps").select(["run_id", "step", "action", "gripper_qpos"]).to_pylist()
    report = grasp_report(runs, steps, shut_width_m=shut, open_width_m=open_, phase=phase, **rule)
    out.write_text(json.dumps(report, indent=1))
    return report
