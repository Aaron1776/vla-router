"""The golden-set self-check (SPEC §15; Phase 0 step 7): `vlar-ops golden select|check`.

Five snapshots from the base rollouts, each with the trajectory and outcome its base run actually recorded after it.
Replaying one restores the snapshot and lets X-VLA continue locally to the end, as the base run did (same rollout
seed, so the same model noise): every executed step (action, end-effector pose, finger positions) and the outcome
(success, success step, step count) must equal the record bit for bit. That is the property data collection rests
on: a base rollout counts as one seed of branch 1 at every decision point (§7).

- `select --root DIR --out golden.json`: chooses the snapshots (one per task for five tasks drawn with a recorded
  seed, one decision point each), and stores their expected trajectories from the `steps` table and outcomes from the
  `runs` table (`golden.npz` beside the JSON, exact values).
- `check --golden golden.json --snapshots DIR [--marker F]`: replays them through the production servers; exit 0 and
  a pass marker for this boot only if all match, else exit 1 with the first differing step. Run at every start of a
  frozen harness before any worker (`lib.sh` `run_phase`); `vlar-ops worker` refuses a frozen harness without this
  boot's pass marker (`vm_worker.require_golden_pass`).

Ops code: it uses the image's harness (`Episode.restore`, `decide`) and changes nothing in it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

GOLDEN_N = 5
GOLDEN_SEED = 20261123
FIELDS = (("action", "float32", 7), ("eef_pos", "float64", 3), ("eef_quat", "float64", 4),
          ("gripper_qpos", "float64", 2))
TRAJ_KEYS = {"eef_pos": "robot0_eef_pos", "eef_quat": "robot0_eef_quat", "gripper_qpos": "robot0_gripper_qpos"}
MARKER = Path("/var/lib/vlar/golden-pass.json")


class GoldenMismatch(RuntimeError):
    pass


def boot_id() -> Optional[str]:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def select(runs: Sequence[Mapping], points: Sequence[Mapping], steps: Sequence[Mapping], *, n: int = GOLDEN_N,
           seed: int = GOLDEN_SEED) -> tuple[dict, dict]:
    """(golden record, expected arrays). Base runs only; points with a snapshot and at least one decision point
    behind them; tasks drawn without replacement (more than one point per task only when fewer tasks exist)."""
    import numpy as np
    base = {r["run_id"]: r for r in runs if r["phase"] == "base"}
    cands: dict[int, list] = {}
    for p in sorted(points, key=lambda p: (p["run_id"], int(p["index"]))):
        if p["run_id"] in base and p.get("snapshot_id") and int(p["index"]) >= 1:
            cands.setdefault(int(base[p["run_id"]]["task_id"]), []).append(p)
    if not cands:
        raise GoldenMismatch("no base snapshots to choose a golden set from")
    rng = random.Random(seed)
    tasks = sorted(cands)
    rng.shuffle(tasks)
    chosen: list = []
    while len(chosen) < n and any(cands.values()):
        for t in tasks:
            if len(chosen) < n and cands[t]:
                chosen.append(cands[t].pop(rng.randrange(len(cands[t]))))
    by_run: dict[str, list] = {}
    for s in steps:
        if any(s["run_id"] == c["run_id"] for c in chosen):
            by_run.setdefault(s["run_id"], []).append(s)
    golden, arrays = [], {}
    for i, c in enumerate(chosen):
        run = base[c["run_id"]]
        after = sorted((s for s in by_run.get(c["run_id"], []) if int(s["step"]) > int(c["step"])),
                       key=lambda s: int(s["step"]))
        if not after:
            raise GoldenMismatch(f"{c['snapshot_id']}: no recorded steps after it")
        arrays[f"{i}_step"] = np.array([int(s["step"]) for s in after], dtype=np.int64)
        for name, dtype, width in FIELDS:
            arrays[f"{i}_{name}"] = np.array([s[name] for s in after], dtype=dtype).reshape(-1, width)
        golden.append({"snapshot_id": c["snapshot_id"], "run_id": c["run_id"], "task_id": int(run["task_id"]),
                       "index": int(c["index"]), "step": int(c["step"]), "expected_steps": len(after),
                       "outcome": {"success": bool(run["success"]), "success_step": run["success_step"],
                                   "steps": int(run["steps"])}})
    return {"n": len(golden), "seed": seed, "golden": golden, "checked": False}, arrays


def replay(make_episode: Callable[[int], Any], snapshot, snapshot_id: str, task_id: int) -> tuple[dict, dict]:
    """Restore and continue locally to the end, measurements off (they never change what executes, §11 step 3)."""
    import numpy as np
    ep = make_episode(task_id)
    dp = ep.restore(snapshot, snapshot_id=snapshot_id)
    if dp is None:
        raise GoldenMismatch(f"{snapshot_id}: not at a decision point of a running episode")
    ep.state.record = "off"
    while dp is not None:
        dp = ep.decide("local").next_point
    traj = ep.trajectory
    got = {"step": np.array([int(r["step"]) for r in traj], dtype=np.int64)}
    for name, dtype, width in FIELDS:
        key = TRAJ_KEYS.get(name, name)
        got[name] = np.array([np.asarray(r[key], dtype=dtype).ravel() for r in traj], dtype=dtype).reshape(-1, width)
    res = ep.result()
    return got, {"success": bool(res.success), "success_step": res.success_step, "steps": int(res.steps)}


def compare(i: int, g: Mapping, arrays: Mapping, got: Mapping, outcome: Mapping) -> Optional[str]:
    """None if the replay equals the record bit for bit, else where it first differs."""
    import numpy as np
    want_steps, got_steps = arrays[f"{i}_step"], got["step"]
    n = min(len(want_steps), len(got_steps))
    for name in ("step",) + tuple(f for f, _, _ in FIELDS):
        want = arrays[f"{i}_{name}"][:n]
        have = got[name][:n]
        bad = np.flatnonzero(~np.all((want == have).reshape(n, -1), axis=1)) if n else []
        if len(bad):
            k = int(bad[0])
            return f"{g['snapshot_id']}: {name} differs first at step {int(want_steps[k])}"
    if len(want_steps) != len(got_steps):
        return f"{g['snapshot_id']}: {len(got_steps)} steps replayed, {len(want_steps)} recorded"
    if dict(outcome) != dict(g["outcome"]):
        return f"{g['snapshot_id']}: outcome {dict(outcome)}, recorded {dict(g['outcome'])}"
    return None


def check(record: Mapping, arrays: Mapping, make_episode: Callable[[int], Any],
          read_snapshot: Callable[[str], Any]) -> dict:
    results = []
    for i, g in enumerate(record["golden"]):
        got, outcome = replay(make_episode, read_snapshot(g["snapshot_id"]), g["snapshot_id"], g["task_id"])
        results.append({"snapshot_id": g["snapshot_id"], "problem": compare(i, g, arrays, got, outcome)})
    return {"passed": all(r["problem"] is None for r in results), "results": results}


def write_marker(path: Path, report: Mapping, calibration_hash: Optional[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"boot_id": boot_id(), "calibration_hash": calibration_hash,
                                "passed": bool(report["passed"]), "results": report["results"]}, indent=1))


def passed_this_boot(path: Path, calibration_hash: Optional[str]) -> bool:
    try:
        m = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    return bool(m.get("passed")) and m.get("boot_id") == boot_id() and m.get("calibration_hash") == calibration_hash


# ---------------------------------------------------------------- files

def save(record: Mapping, arrays: Mapping, out: Path) -> None:
    import numpy as np
    out.write_text(json.dumps(record, indent=1))
    with open(out.with_suffix(".npz"), "wb") as f:
        np.savez(f, **arrays)


def load(path: Path) -> tuple[dict, dict]:
    import numpy as np
    with np.load(Path(path).with_suffix(".npz"), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return json.loads(Path(path).read_text()), arrays


def select_from_records(root: Path | str, out: Path) -> dict:
    from vlar_harness.dataset import read_table
    runs = read_table(root, "runs").select(["run_id", "phase", "task_id", "success", "success_step", "steps"])
    points = read_table(root, "points").select(["run_id", "index", "step", "snapshot_id"])
    steps = read_table(root, "steps").select(["run_id", "step", "action", "eef_pos", "eef_quat", "gripper_qpos"])
    record, arrays = select(runs.to_pylist(), points.to_pylist(), steps.to_pylist())
    save(record, arrays, out)
    return record
