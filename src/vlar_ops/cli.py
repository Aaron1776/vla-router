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


def _when(text: str, now: float) -> float:
    """A time for the ledger: Unix seconds, or ISO 8601 with a time zone (e.g. gcloud's creationTimestamp); never
    in the future."""
    import datetime
    try:
        t = float(text)
    except ValueError:
        d = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
        if d.tzinfo is None:
            sys.exit(f"{text}: give the time zone")
        t = d.timestamp()
    if t > now + 60:
        sys.exit(f"{text} is in the future")
    return t


def _iso(t: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_spend(a: argparse.Namespace) -> None:
    cfg = _ops_config(a.ops_config)
    storage = open_storage(cfg["storage_url"])
    scfg = spend.SpendConfig.from_mapping(cfg.get("spend", {}))
    now = time.time()
    if a.action == "status":
        led = spend.load_ledger(storage)
        failed = spend.pending_failed_boots(storage, led, scfg)     # failed boots not yet folded in by the daemon
        led["sessions"] += failed
        est = spend.estimate(led, now)
        print(json.dumps({**{k: round(v, 2) for k, v in est.items()},
                          "soft_limit_usd": scfg.soft_limit_usd, "hard_limit_usd": scfg.hard_limit_usd,
                          "vm_hours": round(sum(s["vm_seconds"] for s in led["sessions"]) / 3600, 2),
                          "sessions": len(led["sessions"]), "failed_boots_pending": len(failed),
                          "alerts": led["alerts"]}, indent=1))
    elif a.action == "reconcile":
        if a.actual_usd is None:
            sys.exit("--actual-usd is required")
        print(json.dumps(spend.reconcile(storage, a.actual_usd, now, a.note or "")))
    elif a.action == "disk-created":
        if a.gb is None or a.usd_per_gb_month is None:
            sys.exit("--gb and --usd-per-gb-month are required")
        since = _when(a.since, now) if a.since else now
        entry = spend.record_disk(storage, a.gb, a.usd_per_gb_month, since, a.name)
        try:
            spend.confirm_disk(storage, entry)       # read back: exit 0 only for a record the ledger holds
        except spend.DiskRecordNotConfirmed as e:
            sys.exit(f"disk record NOT confirmed: {e}")
        print(f"recorded disk {a.name}: {a.gb:g} GB at ${a.usd_per_gb_month:g}/GB-month since {_iso(since)} "
              f"(confirmed in the ledger)")
    elif a.action == "disk-deleted":
        until = _when(a.until, now) if a.until else now
        try:
            closed = spend.close_disk(storage, a.name, until, _when(a.since, now) if a.since else None)
        except ValueError as e:
            sys.exit(str(e))
        print(f"closed disk {a.name}: since {_iso(closed['since'])}, until {_iso(until)}")
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
            m = manifest.build(a.root, quick=a.quick, move_unlisted=a.move_unlisted, freeze=a.freeze,
                               ops_archives=a.ops_archives)
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
    """Write an episode plan (one job per task and initial state) with the phase's rollout-seed namespace. An
    exploration plan records the exploration tunables in every job spec (amendment 49)."""
    from .plan import episode_plan, load_collection_config
    from .tunables import explore_fields

    def span(text: str) -> list[int]:
        lo, _, hi = text.partition("-")
        return list(range(int(lo), int(hi or lo) + 1))
    ns = int(load_collection_config()["rollout_seed_namespaces"][a.namespace or a.phase])
    fields = explore_fields() if a.phase == "explore" else None
    plan = episode_plan(a.phase, tasks=span(a.tasks), inits=span(a.inits), namespace=ns, mode=a.mode,
                        profile=a.profile, record=a.record, order_seed=a.order_seed, scene=a.scene,
                        extra=(lambda t, i: dict(fields)) if fields else None)
    plan.write(Path(a.out))
    print(f"{a.out}: plan {plan.plan_id} ({plan.phase}), {len(plan.jobs)} jobs, namespace {ns}")


def cmd_finish(a: argparse.Namespace) -> None:
    from . import finish
    if a.cmd == "calibrate":
        res = finish.calibrate(a.root, write=a.write)
        print(res["report"])
        if a.out:
            Path(a.out).write_text(json.dumps({k: v for k, v in res.items() if k != "report"}, indent=1, default=str))
        print(f"{res['base_runs']} base and {res['parity_runs']} parity runs; "
              + (f"written to {res['written']}" if a.write else "not written (--write)"))
    elif a.cmd == "first-targets":
        print(f"{finish.first_targets(a.root, Path(a.out))} first-target distances -> {a.out}")
    elif a.cmd == "summary":
        summary = finish.summary(a.root, json.loads(Path(a.time_limits).read_text()),
                                 json.loads(Path(a.grasps).read_text()))
        text = finish.write_summary(summary, Path(a.out))
        print(text.read_text(), end="")
    elif a.cmd == "unchanged":
        same, why = finish.unchanged(json.loads(Path(a.previous).read_text()), a.root)
        print(why)
        if not same:
            sys.exit(1)
    elif a.cmd == "freeze":
        print(f"frozen: {finish.freeze(Path(a.golden))}")


def cmd_golden(a: argparse.Namespace) -> None:
    from . import golden
    if a.cmd == "select":
        rec = golden.select_from_records(a.root, Path(a.out))
        print(json.dumps([{k: g[k] for k in ("snapshot_id", "task_id", "expected_steps")} for g in rec["golden"]],
                         indent=1))
        return
    from . import vm_worker
    try:
        report = vm_worker.golden_check(_ops_config(a.ops_config), Path(a.golden),
                                        marker=Path(a.marker) if a.marker else None)
    except vm_worker.StartupError as e:
        sys.exit(f"golden check not run: {e}")
    print(json.dumps(report, indent=1))
    if not report["passed"]:
        sys.exit("GOLDEN SET MISMATCH: the harness does not replay its own records; no job may run (§15)")
    if a.mark_checked:
        rec, arrays = golden.load(Path(a.golden))
        rec["checked"] = True
        golden.save(rec, arrays, Path(a.golden))


def cmd_grasps(a: argparse.Namespace) -> None:
    from . import grasps
    rep = grasps.run(Path(a.root), Path(a.out), phase=a.phase, shut=a.shut, open_=a.open)
    s = rep["settled"]
    print(f"{rep['runs']} {rep['phase']} runs; settled commanded-closed steps: {s.get('n', 0)}, shut "
          f"{s.get('shut_frac', 0):.1%}, partly open {s.get('partial_frac', 0):.1%}, open {s.get('open_frac', 0):.1%}; "
          f"near the shut threshold {s.get('near_shut_frac', 0):.1%}, near the open threshold "
          f"{s.get('near_open_frac', 0):.1%} (report: {a.out})")
    if rep["flag"]:
        print("FLAG (the thresholds do not separate the real grasps cleanly): " + "; ".join(rep["reasons"]))
        sys.exit(75)


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
    s.add_argument("--since", help="disk-created: when the disk was created (default now); disk-deleted: which open "
                                   "record of --name to close. Unix seconds or ISO 8601 with a time zone")
    s.add_argument("--until", help="disk-deleted: when the disk was deleted (default now)")
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
    d.add_argument("--ops-archives", help="manifest: a copy of phase0/src/ (<commit>.tar.gz); needed when a record's "
                                          "ops digest is not the freeze's (amendment 49)")
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

    fi = sub.add_parser("finish", help="Phase 0 steps 6-7 from the records (vlar_ops/finish.py)")
    fi.add_argument("cmd", choices=["calibrate", "first-targets", "summary", "unchanged", "freeze"])
    fi.add_argument("--root", help="local copy of the bucket's records (calibrate, first-targets, summary, unchanged)")
    fi.add_argument("--out", help="calibrate: JSON summary; first-targets: the distances file; summary: "
                                  "calibration-summary.json (the text beside it, .txt)")
    fi.add_argument("--time-limits", help="summary: calibrate's --out file")
    fi.add_argument("--grasps", help="summary: the grasp check's report (vlar-ops grasps --out)")
    fi.add_argument("--previous", help="unchanged: an earlier calibration-summary.json")
    fi.add_argument("--write", action="store_true", help="calibrate: store the limits in the calibration file")
    fi.add_argument("--golden", help="freeze: the checked golden set (golden.json)")
    fi.set_defaults(fn=cmd_finish)

    go = sub.add_parser("golden", help="the golden-set self-check (§15; vlar_ops/golden.py)")
    go.add_argument("cmd", choices=["select", "check"])
    go.add_argument("--root", help="select: local copy of the bucket's records")
    go.add_argument("--out", help="select: golden.json (golden.npz beside it)")
    go.add_argument("--golden", help="check: golden.json")
    go.add_argument("--marker", help="check: the pass marker (default /var/lib/vlar/golden-pass.json)")
    go.add_argument("--mark-checked", action="store_true", help="check: record the pass in golden.json (step 7)")
    go.add_argument("--ops-config")
    go.set_defaults(fn=cmd_golden)

    gr = sub.add_parser("grasps", help="after the base rollouts: their commanded-closed finger widths against the "
                                       "gripper thresholds; exit 75 if they do not separate cleanly")
    gr.add_argument("--root", required=True, help="local copy of the bucket (records/runs, records/steps)")
    gr.add_argument("--out", required=True)
    gr.add_argument("--phase", default="base")
    gr.add_argument("--shut", type=float, help="shut_width_m (default: the calibration file's)")
    gr.add_argument("--open", type=float, help="open_width_m (default: the calibration file's)")
    gr.set_defaults(fn=cmd_grasps)

    s2 = sub.add_parser("step2", help="Phase 0 step 2 measurements on the L4 (vlar_ops/step2.py)")
    s2.add_argument("cmd", choices=["capture", "choose", "latency", "upload", "gripper", "updates", "render", "load",
                                    "memory"])
    s2.add_argument("--out", required=True)
    s2.add_argument("--obs", help="latency, load: the capture directory")
    s2.add_argument("--xvla", help="choose: X-VLA bench JSON")
    s2.add_argument("--pi05", help="choose: π0.5 bench JSON")
    s2.add_argument("--memory-dir", help="choose: the directory holding memory-<size>.json (joint measurements)")
    s2.add_argument("--task", type=int, default=0, help="render: LIBERO-10 task")
    s2.add_argument("--seconds", type=float, default=60.0, help="render, load: duration")
    s2.add_argument("--size", type=int, help="load, memory: the candidate batch size")
    s2.add_argument("--smi", help="memory: the nvidia-smi samples")
    s2.add_argument("--workers", type=int, help="memory: renderers running during the measurement")
    s2.add_argument("--error", help="memory: why the measurement could not complete")
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
