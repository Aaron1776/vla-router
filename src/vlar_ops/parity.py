"""The π0.5 parity gate (SPEC §11 step 4; the user's rule of 2026-11-22): `vlar-ops parity gate`.

Reads the runs table of the `pi05_parity` phase from a local copy of the bucket's records, checks it is exactly the
planned run (tasks 0-9 x initial states 0-9, `pi05_only`, scene seed 1000 + k, one calibration file whose
write-once group is the current one), and compares it episode by episode with the reference run in `References/`
(`reference_runs.compare`). Pass: the success rate is within sampling error (exact McNemar p >= 0.05,
`within_sampling_error`); the Phase 0 step then queues the base rollouts. Fail: exit 76 (stop, no debugging).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from . import reference_runs

PHASE = "pi05_parity"
GATE_FAILED_EXIT = 76
TASKS, INITS = range(10), range(10)
SCENE_OFFSET = 1000


class ParityRunInvalid(RuntimeError):
    """The records are not the planned parity run: no verdict (the step fails; nothing is queued)."""


def gate(records_root: Path | str, references: Path | str, *, calibration_hash: Optional[str]) -> dict[str, Any]:
    from vlar_harness.dataset import read_table
    rows = [r for r in read_table(records_root, "runs").to_pylist() if r["phase"] == PHASE]
    keys = [(r["task_id"], r["init_idx"]) for r in rows]
    want = {(t, i) for t in TASKS for i in INITS}
    problems = []
    if sorted(keys) != sorted(want):
        problems.append(f"{len(rows)} runs for {len(set(keys))} distinct (task, init) pairs; need exactly {len(want)}")
    for r in rows:
        if r["run_kind"] != "episode" or r["mode"] != "pi05_only" or r["time_limit"]:
            problems.append(f"{r['run_id']}: {r['run_kind']} {r['mode']} time_limit={r['time_limit']}")
        if r["scene_seed"] != SCENE_OFFSET + r["init_idx"]:
            problems.append(f"{r['run_id']}: scene seed {r['scene_seed']}, need {SCENE_OFFSET + r['init_idx']}")
        if calibration_hash is not None and r["calibration_hash"] != calibration_hash:
            problems.append(f"{r['run_id']}: calibration {str(r['calibration_hash'])[:12]}, current {calibration_hash[:12]}")
    if problems:
        raise ParityRunInvalid("; ".join(problems[:10]))
    ours = {(r["task_id"], r["init_idx"]): bool(r["success"]) for r in rows}
    reference_runs.verify_files(references)
    res = reference_runs.compare(ours, reference_runs.load_pi05(references), limit=520)
    res["end_reasons"] = {e: sum(r["end_reason"] == e for r in rows) for e in sorted({r["end_reason"] for r in rows})}
    res["passed"] = bool(res["within_sampling_error"])
    return res
