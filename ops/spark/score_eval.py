"""Score an evaluation run's episodes from their event logs (SPEC §10): outcome without the time limit, and success
within each task's calibrated limit, under a profile, with `vlar_harness.clock.score` and the frozen calibration's
limits (`cfg.time_limit_s`). Ops-side; the harness's own scoring code, unchanged.

    python score_eval.py --run DIR [--profile none] [--out FILE]
DIR: an evaluate/repro output folder (`*-results-*.jsonl` and `records/<phase>/<job>/records/events.parquet`).
"""
import argparse, collections, glob, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True); ap.add_argument("--profile", default=None); ap.add_argument("--out")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from vlar_harness import clock as clk
    from vlar_harness.config import load_config
    cfg = load_config()
    charges = clk.Charges.from_config(cfg)
    lines = [json.loads(l) for f in sorted(glob.glob(f"{a.run}/*-results-*.jsonl")) for l in open(f)]
    rows, ends = [], collections.Counter()
    for x in lines:
        r = x["run"]; t = int(r["task_id"])
        prof = a.profile or r["profile"]
        ev = pq.read_table(glob.glob(f"{a.run}/records/*/{x['job_id']}/records/events.parquet")[0]).to_pylist()
        events = [json.loads(e["event"]) if isinstance(e["event"], str) else e["event"] for e in sorted(ev, key=lambda e: e["seq"])]
        s = clk.score(events, prof, charges)
        lim = float(cfg.time_limit_s(t))
        rows.append({"run_id": x["run_id"], "task_id": t, "success": s.success, "end_reason": s.end_reason,
                     "success_step": s.success_step, "time_s": round(s.time_s, 3), "limit_s": lim,
                     "within_limit": s.within(lim), "pi05_steps": int(r["pi05_steps"] or 0),
                     "steps": int(r["steps"]), "agrees_with_run_row": s.success == bool(r["success"])})
        ends[s.end_reason] += 1
    per = {t: {"episodes": 0, "success": 0, "within_limit": 0} for t in range(10)}
    for r in rows:
        p = per[r["task_id"]]; p["episodes"] += 1; p["success"] += r["success"]; p["within_limit"] += r["within_limit"]
    out = {"episodes": len(rows), "profile": a.profile or "each run's own", "calibration_hash": cfg.calibration_hash,
           "success": sum(r["success"] for r in rows), "within_limit": sum(r["within_limit"] for r in rows),
           "end_reasons": dict(ends), "per_task": per,
           "pi05_step_share": round(sum(r["pi05_steps"] for r in rows) / max(1, sum(r["steps"] for r in rows)), 4),
           "score_disagrees_with_run_rows": sum(not r["agrees_with_run_row"] for r in rows)}
    if a.out:
        Path(a.out).write_text(json.dumps({**out, "episodes_detail": rows}, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
