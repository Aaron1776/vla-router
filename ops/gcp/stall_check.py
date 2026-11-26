"""Stall alarm from what reaches the bucket (the user, 2026-11-26): no access to the VM, no VM code. While a step
runs, its mismatch marker, the spend guard's stop flag and the job queue stay on the VM's disk, so a check from the
Mac never sees them: it says so ("VM-internal state not visible mid-step") and never calls a step clear. What it does
see: each job's result marker, uploaded as the job finishes (`queue.complete`: written, uploaded, then marked done;
0.1 s after `finished_at` on 60 sampled markers), and each stage's plan when the step uploads it. If the VM is RUNNING
and nothing has landed for longer than the measured gaps allow, it reports a possible stop.

Allowances (measured 2026-11-26; a gap between results cannot outlast the longest job a busy worker is running):
  - a stage's gap: the longest job of that kind plus slack. 740 re-runs (record full, 7 workers): longest job 17.0
    min of 145, largest gap 8.5 min -> 20 min. Branch jobs (Tranche A, 7 workers): longest 1.7 min of 400 sampled,
    largest gap 1.1 min of 3,200 (screens 1.3) -> 5 min.
  - boot window, from the VM's start to the first result: first claim 6-16 min after a start (16 with Tranche A's
    gate), plus the longest re-run -> 35 min.
  - a stage's opening, from its plan's upload to its first result: the time to a first claim (a gate, the servers'
    start: <= 16 min measured) plus the stage's gap -> 15 min + the gap (re-runs 35, branches 20).
  - between stages (the last result of one, before the next one's plan lands): the finished stage's gap.
  - finishing, every job landed until the VM stops (measured tails <= 3.9 min) -> 10 min.

Read-only, application-default credentials; run from the repo root after `source ops/gcp/env.sh && export PROJECT_ID
BUCKET VM_NAME`:
  .venv-ops/bin/python ops/gcp/stall_check.py --stage reruns:explore:vla-router/phase1b/rerun/plan.json:20 \\
      --stage branches:phase1b:vla-router/phase1b/plan/plan.json:5
Prints one JSON line; exit 0 no stall, 1 possible stop (stalled), 2 unreadable."""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

NOT_VISIBLE = "VM-internal state not visible mid-step (mismatch marker, stop flag, queue counts)"
BOOT_WINDOW_MIN = 35.0
OPEN_WINDOW_MIN = 15.0
FINISH_WINDOW_MIN = 10.0


def stall_verdict(now: float, vm_status: str, last_start: float, stages: list, *,
                  boot_window_s: float = BOOT_WINDOW_MIN * 60, open_window_s: float = OPEN_WINDOW_MIN * 60,
                  finish_window_s: float = FINISH_WINDOW_MIN * 60) -> dict:
    """`stages`, in the step's order: {"name", "total" (jobs in its plan, None while the plan is not in the bucket),
    "landed" (finish times of its results, any boot), "opened_at" (its plan's upload time, or None), "gap_s"}."""
    out = {"vm": vm_status, "not_visible": NOT_VISIBLE,
           "landed": {s["name"]: f"{len(s['landed'])}/{s['total'] if s['total'] is not None else '?'}" for s in stages}}
    if vm_status != "RUNNING":
        return {**out, "stage": None, "stalled": False, "why": "VM not running: the watcher reads its stop"}
    cur = next((i for i, s in enumerate(stages) if s["total"] is None or len(s["landed"]) < s["total"]), None)
    events = [(t, f"{s['name']} result") for s in stages for t in s["landed"] if t >= last_start]
    events += [(s["opened_at"], f"{s['name']} plan") for s in stages
               if s["opened_at"] is not None and s["opened_at"] >= last_start]
    if not events:
        stage, ref, what, allow = "boot", last_start, "VM start", boot_window_s
    else:
        ref, what = max(events)
        if cur is None:
            stage, allow = "finishing", finish_window_s
        elif stages[cur]["total"] is None:          # between stages: the next plan is not in the bucket yet
            stage, allow = f"before {stages[cur]['name']}", stages[cur - 1]["gap_s"] if cur else boot_window_s
        elif what == f"{stages[cur]['name']} plan":
            stage, allow = f"{stages[cur]['name']} opening", open_window_s + stages[cur]["gap_s"]
        else:
            stage, allow = stages[cur]["name"], stages[cur]["gap_s"]
    quiet = now - ref
    return {**out, "stage": stage, "last_landing": what,
            "last_landing_at": datetime.fromtimestamp(ref, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "quiet_min": round(quiet / 60, 1), "allowance_min": round(allow / 60, 1), "stalled": quiet > allow}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="append", required=True,
                    help="NAME:PHASE:PLAN_OBJECT:GAP_MIN, in the step's order")
    a = ap.parse_args()
    try:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        from google.cloud import storage
        P, V = os.environ["PROJECT_ID"], os.environ["VM_NAME"]
        Z = os.environ.get("ZONE_OVERRIDE", "us-central1-a")
        cred, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        r = AuthorizedSession(cred).get(f"https://compute.googleapis.com/compute/v1/projects/{P}/zones/{Z}/instances/{V}",
                                        timeout=60)
        r.raise_for_status()
        inst = r.json()
        last_start = datetime.fromisoformat(inst.get("lastStartTimestamp", "1970-01-01T00:00:00+00:00")).timestamp()
        bkt = storage.Client(project=P).bucket(os.environ["BUCKET"])
        stages = []
        for spec in a.stage:
            name, phase, plan_obj, gap = spec.split(":")
            blob = bkt.get_blob(plan_obj)
            if blob is None:
                stages.append({"name": name, "total": None, "landed": [], "opened_at": None, "gap_s": float(gap) * 60})
                continue
            ids = {json.loads(line)["job_id"] for line in blob.download_as_text().splitlines()[1:] if line.strip()}
            prefix = f"vla-router/results/{phase}/"
            landed = [b.time_created.timestamp() for b in bkt.list_blobs(prefix=prefix)
                      if b.name.endswith(".json") and b.name[len(prefix):-len(".json")] in ids]
            stages.append({"name": name, "total": len(ids), "landed": landed,
                           "opened_at": blob.time_created.timestamp(), "gap_s": float(gap) * 60})
    except Exception as e:  # noqa: BLE001 - any read failure is "unreadable", never "no stall"
        print(json.dumps({"unreadable": f"{type(e).__name__}: {str(e)[:200]}", "not_visible": NOT_VISIBLE}))
        sys.exit(2)
    v = stall_verdict(time.time(), inst["status"], last_start, stages)
    print(json.dumps(v))
    sys.exit(1 if v["stalled"] else 0)


if __name__ == "__main__":
    main()
