"""`vlar-ops`: queue, spend, worker, dataset and packaging commands. Settings come from an ops config file
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


def cmd_worker(a: argparse.Namespace) -> None:
    from . import vm_worker
    cfg = _ops_config(a.ops_config)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # abandon the current job cleanly (lease released)
    wid = a.worker_id or f"worker-{os.getpid()}"
    try:
        stats = vm_worker.serve(cfg, worker_id=wid, options=vm_worker.parse_options(a.option), phases=a.phase,
                                max_jobs=a.max_jobs, server_wait_s=a.server_wait_s)
    except vm_worker.StartupError as e:
        sys.exit(f"worker {wid} not started: {e}")
    print(json.dumps({"worker": wid, **stats}))
    sys.exit(vm_worker.REFUSED_EXIT if stats["refused"] else 0)


def cmd_dataset(a: argparse.Namespace) -> None:
    from . import manifest
    if a.action == "manifest":
        try:
            m = manifest.build(a.root, quick=a.quick, move_unlisted=a.move_unlisted, freeze=a.freeze)
        except manifest.ManifestError as e:
            sys.exit(f"no manifest written: {e}")
        print(json.dumps({k: m[k] for k in ("dataset", "harness_version", "configs", "phases", "tables", "totals",
                                            "warnings")}, indent=1))
    else:
        problems = manifest.verify(a.root, quick=a.quick)
        print("\n".join(problems) if problems else "verified")
        sys.exit(1 if problems else 0)


def cmd_reference(a: argparse.Namespace) -> None:
    from . import reference_runs
    report = reference_runs.check(a.dir)
    t = report["totals"]
    print(f"reference run OK: {t['episodes']} episodes per model | pi0.5 {t['pi05_successes']}/500 | "
          f"X-VLA {t['xvla_successes_900']}/500 at 900 steps, {t['xvla_successes_520']}/500 within 520 | "
          f"pi0.5 parity subset (initial states 0-9): {report['pi05_parity_subset']}/100")
    for task, row in report["per_task"].items():
        print(f"  task {task}: pi0.5 {row['pi05']}/50, X-VLA {row['xvla_520']}/50 within 520")


def cmd_package(a: argparse.Namespace) -> None:
    if a.action == "build":
        s = packaging.build(a.src, a.out, a.name, int(a.part_gb * 1024**3))
        print(f"{len(s.parts)} parts, index {s.index.name}, checksums {s.sums}")
    else:
        problems = packaging.verify(a.out, deep=not a.quick)
        print("\n".join(problems) if problems else "verified")
        sys.exit(1 if problems else 0)


def cmd_plan(a: argparse.Namespace) -> None:
    """Write an episode plan (one job per task and initial state) with the phase's rollout-seed namespace."""
    from .plan import episode_plan, load_collection_config

    def span(text: str) -> list[int]:
        lo, _, hi = text.partition("-")
        return list(range(int(lo), int(hi or lo) + 1))
    ns = int(load_collection_config()["rollout_seed_namespaces"][a.namespace or a.phase])
    plan = episode_plan(a.phase, tasks=span(a.tasks), inits=span(a.inits), namespace=ns, mode=a.mode,
                        profile=a.profile, record=a.record, order_seed=a.order_seed, scene=a.scene)
    plan.write(Path(a.out))
    print(f"{a.out}: plan {plan.plan_id} ({plan.phase}), {len(plan.jobs)} jobs, namespace {ns}")


def cmd_parity(a: argparse.Namespace) -> None:
    from vlar_harness.config import load_config
    from . import parity
    cal = load_config().calibration_hash
    try:
        res = parity.gate(a.records, a.references, calibration_hash=cal)
    except parity.ParityRunInvalid as e:
        sys.exit(f"parity run invalid, no verdict: {e}")
    Path(a.out).write_text(json.dumps(res, indent=1, default=list))
    print(json.dumps(res, indent=1, default=list))
    if not res["passed"]:
        print(f"GATE FAILED: McNemar p = {res['mcnemar_p']:.4f} < 0.05 (ours {res['ours']}/{res['n']}, "
              f"reference {res['reference']}/{res['n']})")
        sys.exit(parity.GATE_FAILED_EXIT)


def cmd_step2(a: argparse.Namespace) -> None:
    from . import step2
    sys.exit(step2.run(a.cmd, a))


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

    w = sub.add_parser("worker", help="run one simulator worker on the VM (start one per process)")
    w.add_argument("--worker-id", help="name in leases and result markers (default worker-<pid>)")
    w.add_argument("--phase", action="append", help="claim only jobs of this phase (repeatable)")
    w.add_argument("--option", action="append", default=[], metavar="NAME=CHOICE",
                   help="named harness option, e.g. scene_seed=pi05_parity (repeatable)")
    w.add_argument("--max-jobs", type=int)
    w.add_argument("--server-wait-s", type=float, default=600.0, help="how long to wait for the model servers")
    w.add_argument("--ops-config")
    w.set_defaults(fn=cmd_worker)

    d = sub.add_parser("dataset", help="write or verify the dataset's MANIFEST.json and DATACARD.md (§13)")
    d.add_argument("action", choices=["manifest", "verify"])
    d.add_argument("--root", required=True, help="local copy of the bucket's dataset files")
    d.add_argument("--quick", action="store_true", help="check sizes only, not SHA-256")
    d.add_argument("--move-unlisted", help="manifest: move data files no result marker lists into this directory")
    d.add_argument("--freeze", help="manifest: the v1.0 freeze record (freeze.json); every record's calibration hash "
                                    "must be the frozen file's or in its lineage")
    d.set_defaults(fn=cmd_dataset)

    r = sub.add_parser("reference")
    r.add_argument("action", choices=["check"])
    r.add_argument("--dir", default="References")
    r.set_defaults(fn=cmd_reference)

    pl = sub.add_parser("plan", help="write an episode plan for `queue add`")
    pl.add_argument("kind", choices=["episodes"])
    pl.add_argument("--phase", required=True)
    pl.add_argument("--namespace", help="rollout-seed namespace name (default: the phase)")
    pl.add_argument("--tasks", default="0-9")
    pl.add_argument("--inits", required=True, help="e.g. 0-9 or 0-49")
    pl.add_argument("--mode", required=True, choices=["routed", "xvla_only", "pi05_only"])
    pl.add_argument("--profile", default="none")
    pl.add_argument("--record", default="off", choices=["off", "full"])
    pl.add_argument("--scene", default="project", help="the harness's scene_seed choice")
    pl.add_argument("--order-seed", type=int, default=0)
    pl.add_argument("--out", required=True)
    pl.set_defaults(fn=cmd_plan)

    pa = sub.add_parser("parity", help="the π0.5 parity gate (SPEC §11 step 4): exit 76 if it fails")
    pa.add_argument("action", choices=["gate"])
    pa.add_argument("--records", required=True, help="local copy of the bucket (records/runs/pi05_parity/)")
    pa.add_argument("--references", default="References")
    pa.add_argument("--out", required=True)
    pa.set_defaults(fn=cmd_parity)

    s2 = sub.add_parser("step2", help="Phase 0 step 2 measurements on the L4 (vlar_ops/step2.py)")
    s2.add_argument("cmd", choices=["capture", "choose", "latency", "upload", "gripper", "updates"])
    s2.add_argument("--out", required=True)
    s2.add_argument("--obs", help="latency: the capture directory")
    s2.add_argument("--xvla", help="choose: X-VLA bench JSON")
    s2.add_argument("--pi05", help="choose: π0.5 bench JSON")
    s2.add_argument("--dir", help="updates: the directory holding geometry/latency/upload/gripper.json")
    s2.set_defaults(fn=cmd_step2)

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
