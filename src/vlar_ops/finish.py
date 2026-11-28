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
- `summary --root DIR --time-limits F --grasps F --out S.json`: the end of step 6, for the user: per-task limits and
  m as the calibration file now holds them, π0.5's share within the limits, the grasp check; S.json and S.txt.
  Step 6 then stops (exit 75): the freeze is queued only with the user's explicit approval after reading it.
- `unchanged --root DIR --previous S.json`: exit 0 if S.json was made from these records and the calibration file
  is still the one it describes (a re-run of step 6 then has nothing to do).
- `freeze --golden G.json`: step 7, `provenance.write_freeze` with the golden set's snapshot ids (after
  `vlar-ops golden check` passed on it).
"""

from __future__ import annotations

import hashlib
import json
import math
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


def records_digest(root: Path | str) -> str:
    """SHA-256 over the downloaded records under `root`/records (relative paths and contents)."""
    base = Path(root) / "records"
    h = hashlib.sha256()
    for p in sorted(q for q in base.rglob("*") if q.is_file()):
        h.update(p.relative_to(base).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def summary(root: Path | str, time_limits: Mapping[str, Any], grasps: Mapping[str, Any], cfg=None) -> dict[str, Any]:
    """Step 6's result for the user (per-task limits, m, π0.5's share within the limits, the grasp check). The limits
    and m are read back from the calibration file in effect and must be the ones this run computed."""
    from vlar_harness.config import load_config
    cfg = cfg or load_config()
    tl = cfg["time_limit"]
    if tl.get("status") != "calibrated":
        raise RecordsError(f"the calibration file's time limits are {tl.get('status')!r}: run `finish calibrate --write`")
    margin, limits = float(tl["margin"]), {int(t): float(v) for t, v in tl["per_task_s"].items()}
    rows = {int(r["task"]): r for r in time_limits["rows"]}
    if margin != float(time_limits["margin"]) or set(limits) != set(rows) or any(
            not math.isclose(limits[t], float(rows[t]["limit_s"]), rel_tol=1e-12) for t in limits):
        raise RecordsError("the calibration file's time limits are not the ones this step computed")
    shares = {float(m): float(v) for m, v in time_limits["share_by_margin"].items()}
    return {
        "calibration_hash": cfg.calibration_hash, "records_digest": records_digest(root),
        "base_runs": int(time_limits["base_runs"]), "parity_runs": int(time_limits["parity_runs"]),
        "margin": margin, "pi05_share_within_limits": shares[margin],
        "min_pi05_share": float(tl["margin_rule_min_pi05_share"]), "share_by_margin": shares,
        "per_task": [{"task": t, "limit_s": limits[t], **{k: rows[t][k] for k in (
            "p90_s", "xvla_successes", "xvla_episodes", "pi05_successes", "pi05_fit", "pi05_fit_share")}}
            for t in sorted(limits)],
        "grasps": {k: grasps[k] for k in ("flag", "reasons", "thresholds", "runs", "settled")},
    }


def summary_text(s: Mapping[str, Any]) -> str:
    g, st = s["grasps"], s["grasps"]["settled"]
    lines = [
        "Phase 0 step 6: calibration summary. The freeze (step 7, 600) is queued only with the user's approval.",
        f"calibration file {s['calibration_hash'][:12]}; records {s['records_digest'][:12]} "
        f"({s['base_runs']} base runs, {s['parity_runs']} π0.5 parity runs)",
        f"margin m = {s['margin']}: {s['pi05_share_within_limits']:.1%} of π0.5's successful parity episodes finish "
        f"within their task's limit (rule: the smallest candidate with at least {s['min_pi05_share']:.0%})",
        "by candidate: " + ", ".join(f"{m}: {v:.1%}" for m, v in sorted(s["share_by_margin"].items())),
        "task   limit (s)   X-VLA p90 (s)   X-VLA succ/eps   π0.5 within/succ",
    ]
    for r in s["per_task"]:
        lines.append(f"{r['task']:>4}  {r['limit_s']:>10.3f}  {r['p90_s']:>14.3f}  "
                     f"{r['xvla_successes']:>8}/{r['xvla_episodes']:<6}  {r['pi05_fit']:>9}/{r['pi05_successes']}")
    th = g["thresholds"]
    lines.append(f"grasp check: {'FLAGGED' if g['flag'] else 'clean'} ({g['runs']} base runs; thresholds shut "
                 f"{th['shut_width_m'] * 1000:.2f} mm, open {th['open_width_m'] * 1000:.2f} mm; settled "
                 f"commanded-closed steps {st.get('n', 0)}: shut {st.get('shut_frac', 0):.1%}, partly open "
                 f"{st.get('partial_frac', 0):.1%}, open {st.get('open_frac', 0):.1%}; near the shut threshold "
                 f"{st.get('near_shut_frac', 0):.1%}, near the open threshold {st.get('near_open_frac', 0):.1%})")
    lines += [f"  {reason}" for reason in g["reasons"]]
    lines.append("To approve the freeze: ops/gcp/31_phase0_queue.sh ops/vm/phase0/steps/600-freeze.sh, then start "
                 "the VM.")
    return "\n".join(lines)


def write_summary(s: Mapping[str, Any], out: Path) -> Path:
    out = Path(out)
    out.write_text(json.dumps(s, indent=1) + "\n")
    text = out.with_suffix(".txt")
    text.write_text(summary_text(s) + "\n")
    return text


def unchanged(previous: Mapping[str, Any], root: Path | str, cfg=None) -> tuple[bool, str]:
    """Whether `previous` (an earlier summary) was made from these records with the calibration file in effect."""
    from vlar_harness.config import load_config
    cfg = cfg or load_config()
    if previous.get("records_digest") != records_digest(root):
        return False, "the records differ from the summary's"
    if previous.get("calibration_hash") != cfg.calibration_hash:
        return False, "the calibration file differs from the summary's"
    return True, "records and calibration file are the summary's"


def freeze(golden: Path) -> Path:
    from vlar_harness import provenance
    rec = json.loads(Path(golden).read_text())
    if not rec.get("checked"):
        raise RecordsError("the golden set has not passed `vlar-ops golden check`: no freeze")
    return provenance.write_freeze([g["snapshot_id"] for g in rec["golden"]])
