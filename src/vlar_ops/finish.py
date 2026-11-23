"""Phase 0 steps 6 and 7 from the dataset records (SPEC §5, §11 steps 6-7, §15): `vlar-ops finish <command>`.

Ops code, outside the frozen harness: it runs in the simulator image with this package mounted from the host source
over the image's copy, and calls the image's own harness functions (`vlar_harness.calibration`, `eventlog`,
`provenance`), so nothing here changes the harness's source digest or needs another image build.

- `logs --root DIR --phase P`: the event logs of a phase's runs, rebuilt from the `runs` and `events` tables (each
  run's events in `seq` order, exactly as the engine logged them).
- `calibrate --root DIR [--write]`: step 6, `calibration.calibrate` over the base rollouts (`base`) and the π0.5
  parity episodes (`pi05_parity`): per-task limits = p90 of X-VLA's successful base completion times x the margin m,
  the smallest candidate with at least 80% of π0.5's successful parity episodes within their task's limit (§5,
  amendment 9). `--write` stores them in the calibration file (`VLAR_CALIBRATION`).
- `first-targets --root DIR --out F`: X-VLA's first-target distances at ordinary chunk boundaries of the base
  rollouts, the input of the relative handoff check (amendment 10).
- `freeze --golden G.json`: step 7, `provenance.write_freeze` with the golden set's snapshot ids (after
  `vlar-ops golden check` passed on it).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

BASE_PHASE, PARITY_PHASE = "base", "pi05_parity"


class RecordsError(RuntimeError):
    pass


def logs_from_rows(runs: Sequence[Mapping], events: Sequence[Mapping], phase: str) -> dict[str, list[dict]]:
    """run_id -> its event log, for the runs of `phase`. Every run of the phase must have a complete log: `seq`
    0, 1, 2, ... with no gap or duplicate (a partial upload would bias the limits)."""
    ids = {r["run_id"] for r in runs if r["phase"] == phase}
    if not ids:
        raise RecordsError(f"no {phase!r} runs in the records")
    rows: dict[str, list] = defaultdict(list)
    for e in events:
        if e["run_id"] in ids:
            rows[e["run_id"]].append(e)
    out = {}
    for rid in sorted(ids):
        seq = sorted(rows.get(rid, []), key=lambda e: int(e["seq"]))
        if [int(e["seq"]) for e in seq] != list(range(len(seq))) or not seq:
            raise RecordsError(f"run {rid}: incomplete event log ({len(seq)} rows)")
        out[rid] = [json.loads(e["event"]) for e in seq]
    return out


def logs_from_records(root: Path | str, phase: str) -> dict[str, list[dict]]:
    from vlar_harness.dataset import read_table
    runs = read_table(root, "runs").select(["run_id", "phase"]).to_pylist()
    events = read_table(root, "events").select(["run_id", "seq", "event"]).to_pylist()
    return logs_from_rows(runs, events, phase)


def calibrate(root: Path | str, *, write: bool = False, cfg=None) -> dict[str, Any]:
    from vlar_harness import calibration, eventlog
    from vlar_harness.config import load_config
    cfg = cfg or load_config()
    base = list(logs_from_records(root, BASE_PHASE).values())
    parity = list(logs_from_records(root, PARITY_PHASE).values())
    for events in base + parity:
        eventlog.check(events)
    tl = calibration.calibrate(base, parity, cfg)
    out = {"report": calibration.report(tl), "margin": tl.margin, "per_task_s": tl.per_task_s, "rows": tl.rows,
           "share_by_margin": tl.share_by_margin, "base_runs": len(base), "parity_runs": len(parity)}
    if write:
        out["written"] = str(calibration.write(tl))
    return out


def first_targets(root: Path | str, out: Path) -> int:
    from vlar_harness import eventlog
    logs = list(logs_from_records(root, BASE_PHASE).values())
    for events in logs:
        eventlog.check(events)
    d = eventlog.first_target_distances(logs, "stretch")
    out.write_text(json.dumps(d))
    return len(d)


def freeze(golden: Path) -> Path:
    from vlar_harness import provenance
    rec = json.loads(Path(golden).read_text())
    if not rec.get("checked"):
        raise RecordsError("the golden set has not passed `vlar-ops golden check`: no freeze")
    return provenance.write_freeze([g["snapshot_id"] for g in rec["golden"]])
