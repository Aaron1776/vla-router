"""`vlar-ops`: queue, spend and packaging commands. Settings come from an ops config file
(`--ops-config`, or the VLAR_OPS_CONFIG environment variable); see ops/ops.example.yaml."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import packaging, spend
from .plan import Plan
from .queue import JobQueue
from .storage import open_storage


def _ops_config(path: str | None) -> dict:
    path = path or os.environ.get("VLAR_OPS_CONFIG")
    if not path:
        sys.exit("give --ops-config or set VLAR_OPS_CONFIG (see ops/ops.example.yaml)")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "REPLACE" in str(cfg.get("storage_url", "")):
        sys.exit(f"{path}: set storage_url first")
    return cfg


def _queue(cfg: dict) -> JobQueue:
    root = Path(cfg["local_root"])
    return JobQueue(root / "queue.sqlite", open_storage(cfg["storage_url"]), root / "local")


def cmd_queue(a: argparse.Namespace) -> None:
    cfg = _ops_config(a.ops_config)
    q = _queue(cfg)
    if a.action == "add":
        for p in a.plans:
            plan = Plan.read(Path(p))
            print(f"{p}: plan {plan.plan_id} ({plan.phase}), {q.add_plan(plan)} new of {len(plan.jobs)} jobs")
    elif a.action == "resume":
        print(json.dumps(q.resume()))
    elif a.action == "sync":
        print(f"uploaded {q.sync()} results")
    print(json.dumps(q.counts(a.phase)))


def cmd_spend(a: argparse.Namespace) -> None:
    cfg = _ops_config(a.ops_config)
    storage = open_storage(cfg["storage_url"])
    scfg = spend.SpendConfig.from_mapping(cfg.get("spend", {}))
    now = time.time()
    if a.action == "status":
        g = spend.SpendGuard(storage, scfg, spend.PrintNotifier())
        led = g.load()
        est = spend.estimate(led, now)
        print(json.dumps({**{k: round(v, 2) for k, v in est.items()},
                          "soft_limit_usd": scfg.soft_limit_usd, "hard_limit_usd": scfg.hard_limit_usd,
                          "vm_hours": round(sum(s["vm_seconds"] for s in led["sessions"]) / 3600, 2),
                          "sessions": len(led["sessions"]), "alerts": led["alerts"]}, indent=1))
    elif a.action == "reconcile":
        if a.actual_usd is None:
            sys.exit("--actual-usd is required")
        print(json.dumps(spend.reconcile(storage, a.actual_usd, now, a.note or "")))
    elif a.action == "disk-created":
        if a.gb is None or a.usd_per_gb_month is None:
            sys.exit("--gb and --usd-per-gb-month are required")
        spend.record_disk(storage, a.gb, a.usd_per_gb_month, now, a.name)
        print(f"recorded disk {a.name}")
    elif a.action == "disk-deleted":
        spend.close_disk(storage, a.name, now)
        print(f"closed disk {a.name}")
    elif a.action == "daemon":
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # preemption: final tick, then exit
        root = Path(cfg["local_root"])
        q = _queue(cfg)
        guard = spend.SpendGuard(storage, scfg, spend.GcloudLogNotifier())
        shutdown_cmd = cfg.get("shutdown_command", ["sudo", "shutdown", "-h", "now"])
        state = spend.run_daemon(
            guard, spend.StopFlag(root / "STOP"),
            in_flight=q.in_flight, release_in_flight=q.release_all_leases, final_sync=q.sync,
            shutdown=lambda: subprocess.run(shutdown_cmd, check=False))
        print(f"spend daemon exited in state {state}")


def cmd_package(a: argparse.Namespace) -> None:
    if a.action == "build":
        s = packaging.build(a.src, a.out, a.name, int(a.part_gb * 1024**3))
        print(f"{len(s.parts)} parts, index {s.index.name}, checksums {s.sums}")
    else:
        problems = packaging.verify(a.out, deep=not a.quick)
        print("\n".join(problems) if problems else "verified")
        sys.exit(1 if problems else 0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vlar-ops")
    sub = ap.add_subparsers(dest="group", required=True)

    q = sub.add_parser("queue")
    q.add_argument("action", choices=["add", "resume", "sync", "status"])
    q.add_argument("plans", nargs="*")
    q.add_argument("--phase")
    q.add_argument("--ops-config")
    q.set_defaults(fn=cmd_queue)

    s = sub.add_parser("spend")
    s.add_argument("action", choices=["status", "reconcile", "disk-created", "disk-deleted", "daemon"])
    s.add_argument("--actual-usd", type=float)
    s.add_argument("--note")
    s.add_argument("--name", default="boot")
    s.add_argument("--gb", type=float)
    s.add_argument("--usd-per-gb-month", type=float)
    s.add_argument("--ops-config")
    s.set_defaults(fn=cmd_spend)

    p = sub.add_parser("package")
    p.add_argument("action", choices=["build", "verify"])
    p.add_argument("--src")
    p.add_argument("--out", required=True)
    p.add_argument("--name")
    p.add_argument("--part-gb", type=float, default=10.0)
    p.add_argument("--quick", action="store_true", help="verify: part checksums only")
    p.set_defaults(fn=cmd_package)

    a = ap.parse_args(argv)
    if a.group == "package" and a.action == "build" and not (a.src and a.name):
        ap.error("package build needs --src and --name")
    a.fn(a)


if __name__ == "__main__":
    main()
