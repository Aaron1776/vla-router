"""Whether a queued step finished its own work, judged from the step's own records, never from the runner's state:
the runner's STATE.json in the bucket is the last boot's, so after a boot preempted before it read the queue it
still says "idle ... every queued step is done" from the step before (2026-11-26 07:51Z: 730's state while 735 was
queued). Read-only; through application-default credentials.
  .venv-ops/bin/python ops/gcp/step_done.py --plan <the step's plan.json> --phase explore --step 730 --since <ISO>
Done = every job in the plan has its result marker (results/<phase>/<job id>.json) AND the step's script recorded
exit 0 in <reports>/step-exits.jsonl at or after --since (the step's first start; 735 runs 730's script, so its exits
say "730"). Prints one JSON line; exit 0 when done, 1 when not."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def step_done(job_ids, marker_ids, exits, *, step: str, since: float) -> dict:
    """`job_ids`: the plan's jobs; `marker_ids`: job ids with a result marker; `exits`: step-exits.jsonl records."""
    jobs, have = set(job_ids), set(marker_ids)
    missing = len(jobs - have)
    after = [e for e in exits if str(e.get("step")) == step and float(e.get("at", 0)) >= since]
    ok = any(int(e.get("step_exit", -1)) == 0 for e in after)
    return {"done": missing == 0 and ok, "jobs": len(jobs), "with_marker": len(jobs & have), "missing": missing,
            "exits_since": [int(e.get("step_exit", -1)) for e in after], "exit_0_since": ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True); ap.add_argument("--phase", default="explore")
    ap.add_argument("--step", required=True); ap.add_argument("--since", required=True)
    ap.add_argument("--exits", default="vla-router/phase1b/reports/step-exits.jsonl")
    a = ap.parse_args()
    from google.cloud import storage
    bkt = storage.Client(project=os.environ["PROJECT_ID"]).bucket(os.environ["BUCKET"])
    head, *jobs = [json.loads(line) for line in Path(a.plan).read_text().splitlines() if line.strip()]
    prefix = f"vla-router/results/{a.phase}/"
    have = [b.name[len(prefix):-len(".json")] for b in bkt.list_blobs(prefix=prefix) if b.name.endswith(".json")]
    blob = bkt.blob(a.exits)
    exits = [json.loads(line) for line in blob.download_as_text().splitlines() if line.strip()] if blob.exists() else []
    since = datetime.strptime(a.since, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    r = step_done([j["job_id"] for j in jobs], have, exits, step=a.step, since=since)
    print(json.dumps({"plan_id": head.get("plan_id"), **r}))
    sys.exit(0 if r["done"] else 1)


if __name__ == "__main__":
    main()
