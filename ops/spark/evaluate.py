"""The Spark's evaluation runs (SPEC §10), ops-side: plans with `vlar_ops.plan` in the evaluation rollout-seed namespace
(`eval`, the same seed for every method at a task and initial state), every episode at setup count 2 (a fresh
simulator per episode: the re-plan's option (A) for fresh-start runs, amendment 62 (2)); episodes run by
`repro.py run` (fresh simulator per episode, `vlar_harness.collect.Runner`, records kept).

    python evaluate.py plan --method xvla_only --out DIR [--tasks 0-9] [--inits 0-49]
        writes DIR/eval_<method>-jobs.jsonl
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

FRESH_SETUP_COUNT = 2          # option (A): a fresh simulator per episode
METHODS = {"xvla_only": dict(mode="xvla_only", profile="none", record="off"),
           "pi05_only": dict(mode="pi05_only", profile="none", record="off")}


def span(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def cmd_plan(a):
    from vlar_ops.plan import episode_plan, load_collection_config
    m = METHODS[a.method]
    ns = int(load_collection_config()["rollout_seed_namespaces"]["eval"])
    phase = f"eval_{a.method}"
    plan = episode_plan(phase, tasks=span(a.tasks), inits=span(a.inits), namespace=ns, mode=m["mode"],
                        profile=m["profile"], record=m["record"], order_seed=0, scene="project")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dst = Path(a.out) / f"{phase}-jobs.jsonl"
    dst.write_text("".join(json.dumps({"job": asdict(j), "setup_count": FRESH_SETUP_COUNT}) + "\n" for j in plan.jobs))
    print(json.dumps({"phase": phase, "jobs": len(plan.jobs), "namespace": ns, "setup_count": FRESH_SETUP_COUNT,
                      "out": str(dst)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--method", choices=METHODS, required=True); p.add_argument("--out", required=True)
    p.add_argument("--tasks", default="0-9"); p.add_argument("--inits", default="0-49")
    a = ap.parse_args()
    {"plan": cmd_plan}[a.cmd](a)


if __name__ == "__main__":
    main()
