"""`vlar-ops`: queue, spend, worker, dataset and packaging commands. Settings come from an ops config file
(`--ops-config`, or the VLAR_OPS_CONFIG environment variable); see ops/ops.example.yaml."""

from __future__ import annotations

import argparse
import io
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
                          "alerts": led["alerts"], "phase": spend.phase_state(led, now)}, indent=1))
    elif a.action == "check":
        led = spend.load_ledger(storage)
        led["sessions"] += spend.pending_failed_boots(storage, led, scfg)
        code, why = spend.check(led, scfg, now, phase=a.phase)
        print(why)
        sys.exit(code)
    elif a.action == "phase-open":
        if a.phase is None or a.allocation_usd is None or a.line_usd is None:
            sys.exit("--phase, --allocation-usd and --line-usd are required")
        margin = spend.default_phase_margin_usd(scfg) if a.margin_usd is None else a.margin_usd
        try:
            p = spend.start_phase(storage, a.phase, allocation_usd=a.allocation_usd, line_usd=a.line_usd,
                                  margin_usd=margin, at=now)
        except spend.PhaseError as e:
            sys.exit(str(e))
        print(json.dumps(p))
    elif a.action == "phase-raise":
        if a.phase is None or a.raise_usd is None or not a.reason:
            sys.exit("--phase, --raise-usd and --reason are required")
        try:
            print(json.dumps(spend.raise_phase(storage, a.phase, raise_usd=a.raise_usd, reason=a.reason, at=now)))
        except spend.PhaseError as e:
            sys.exit(str(e))
    elif a.action == "phase-close":
        if a.phase is None:
            sys.exit("--phase is required")
        try:
            print(json.dumps(spend.end_phase(storage, a.phase, now)))
        except spend.PhaseError as e:
            sys.exit(str(e))
    elif a.action == "reconcile":
        if a.actual_usd is None:
            sys.exit("--actual-usd is required")
        print(json.dumps(spend.reconcile(storage, a.actual_usd, now, a.note or "")))
    elif a.action == "charge":
        if a.usd is None or not a.note:
            sys.exit("--usd and --note are required")
        print(json.dumps(spend.add_charge(storage, a.usd, now, a.note)))
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
    sys.exit(worker_exit(stats, a.max_jobs))



def cmd_supervise(a: argparse.Namespace) -> None:
    from . import supervisor as S
    cfg = _ops_config(a.ops_config)
    try:
        summary = S.supervise(cfg, phase=a.phase, workers=a.workers, worker_args=a.worker_args,
                              worker_env=Path(a.worker_env), history=Path(a.history) if a.history else None,
                              step=a.step, max_jobs=a.max_jobs, max_restarts=a.max_restarts, poll_s=a.poll_s,
                              min_limit=int(a.min_limit_gib * S.GiB) if a.min_limit_gib else None,
                              limit=int(a.limit_gib * S.GiB) if a.limit_gib else None,
                              servers_peak=int(a.servers_peak_gb * 1e9) if a.servers_peak_gb else None)
    except S.SizingError as e:
        sys.exit(f"supervise: {e}")
    if a.summary:                       # peak memory, restarts and recycles for the step's report (pilot §0.4 (iv))
        Path(a.summary).write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"supervised": a.phase, "gave_up": summary["gave_up"]}))   # run_phase checks the counts


def worker_exit(stats: dict, max_jobs: int | None) -> int:
    """3: refused a job (never restarted); 4: took its --max-jobs jobs (the supervisor starts it again while work is
    left, PROPOSALS §0.6); 0: nothing left to claim, or the stop flag."""
    from . import vm_worker
    if stats["refused"]:
        return vm_worker.REFUSED_EXIT
    taken = sum(stats[k] for k in ("done", "done_local", "retry", "failed", "lost"))
    return vm_worker.RECYCLE_EXIT if max_jobs is not None and taken >= max_jobs else 0


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


def cmd_smoke(a: argparse.Namespace) -> None:
    from . import smoke
    cfg = _ops_config(a.ops_config)
    out = Path(a.out)
    subjects = [s for s in (a.subjects or "").split(",") if s]
    if a.cmd == "plan":
        smoke.plan_from_markers(open_storage(cfg["storage_url"]), subjects).write(out)
        print(out.read_text().splitlines()[0])
        return
    if a.cmd == "compare":
        from vlar_harness import provenance
        from vlar_harness.config import load_config
        from . import golden
        hcfg = load_config()
        lineage = provenance.read_freeze(provenance.freeze_path(hcfg)).get("calibration_lineage", [])
        rep = smoke.compare(open_storage(cfg["storage_url"]), open_storage(a.smoke_url), subjects,
                            boot=a.boot or golden.boot_id(), calibration_hash=hcfg.calibration_hash, lineage=lineage)
    else:
        rep = smoke.run_on_servers(cfg, Path(a.golden), a.cmd)
    out.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "jobs"}, default=str)[:4000])
    if not rep["passed"]:
        sys.exit(f"SMOKE {a.cmd.upper()} MISMATCH: {rep.get('problems') or [j['problems'] for j in rep.get('jobs', [])]}")


def pilot_defaults(name: str):
    from . import pilot
    return getattr(pilot, name)


def cmd_pilot(a: argparse.Namespace) -> None:
    from . import pilot
    out = Path(a.out)
    if a.cmd == "plan":                      # on the Mac, from local copies of the base runs, events and markers
        sel = pilot.make_plan(Path(a.root), Path(a.markers), out)
        print(json.dumps({k: sel[k] for k in ("plan_id", "n_points", "n_jobs")}))
        return
    if a.cmd == "report":                    # on the Mac, from local copies of the records and reports
        rep = pilot.report(Path(a.root), Path(a.plan_dir), Path(a.reports), hourly_usd=a.hourly_usd,
                           phase_vm_usd=a.phase_vm_usd, stop_profile=a.stop_profile, own_basis=a.own_basis)
        out.write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps({"labelled": rep["labelled"], "problems": len(rep["problems"]), "decision": rep["decision"],
                          "budgets": rep["budgets_points_per_task"]}, default=str))
        return
    if a.cmd == "replan":                    # on the Mac: SPEC §16 item 7 (amendments 61, 63, 64; vlar_ops/replan.py)
        from . import replan
        if a.available_usd is None or not a.pilot_report:
            sys.exit("replan: --available-usd and --pilot-report are required")
        rep = replan.report(Path(a.root), Path(a.plan_dir), json.loads(Path(a.pilot_report).read_text()),
                            available_usd=a.available_usd, hourly_usd=a.hourly_usd)
        out.write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps({"decision": rep["decision"], "budgets": rep["budgets_points_per_task"]}, default=str))
        return
    if a.cmd == "long-episodes":             # on the Mac: the corrected check (iii)'s subjects file in the plan dir
        print(json.dumps(pilot.write_long_episode_checks(Path(a.root), Path(a.plan_dir), out)))
        return
    if a.cmd == "reorder":                   # on the VM, after `queue add` and `queue resume`, before the workers
        q = _queue(_ops_config(a.ops_config))
        try:
            rep = pilot.reorder_queue(q, Path(a.plan_dir))
        finally:
            q.close()
        out.write_text(json.dumps(rep, indent=1))
        print(json.dumps({k: rep[k] for k in ("points", "points_complete", "jobs_reordered", "first_points")}))
        return
    cfg = _ops_config(a.ops_config)          # check: on the VM, after this boot's golden check
    rep = pilot.run_checks(cfg, Path(a.plan_dir), Path(a.staging), subjects_file=a.subjects)
    out.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({"passed": rep["passed"],
                      "features_equal_probe": rep.get("features", {}).get("features_equal_probe"),
                      **{v: [(r["subject"], r.get("loads"), r["passed"]) for r in rep[v]]
                         for v in ("xvla_only", "record_off") if v in rep}}))
    if not rep["passed"]:
        sys.exit(f"PILOT CHECK {pilot.summary_line(rep)}")


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


B_CMDS = ("plan-b-screen", "select-b", "yield-b", "compare-b", "plan-b")


def cmd_phase1b(a) -> None:
    """Tranche B (vlar_ops/phase1b.py): the screen plan (Mac); selection, re-run comparison and branch plan (VM, from
    the bucket's markers and records)."""
    from . import phase1b as B
    from .plan import Plan, load_collection_config
    from .queue import result_key
    out = Path(a.out)
    if a.cmd == "plan-b-screen":
        from .tunables import explore_fields
        if not (a.per_task_screen or a.sizes_file) or a.seed is None:
            sys.exit("plan-b-screen: --per-task-screen (or --sizes-file) and --seed are required")
        ns = int(load_collection_config()["rollout_seed_namespaces"]["explore"])
        sizes = a.per_task_screen
        if a.sizes_file:                    # amendment 68: the top-up's sizes (top_up_decision's `sizes`)
            sizes = {int(t): int(n) for t, n in json.loads(Path(a.sizes_file).read_text()).items()}
        elif a.front_load_base_runs:          # amendment 67: each task's top-up front-loaded at its base failure rate
            sizes = B.front_loaded_sizes(B.base_failure_rates(a.front_load_base_runs), per_task=a.per_task_screen)
        plan = B.screen_plan(sizes, a.seed, namespace=ns, p_escalate=explore_fields()["p_escalate"])
        out.mkdir(parents=True, exist_ok=True)
        plan.write(out / "plan.json")
        print(json.dumps({"phase": plan.phase, "plan_id": plan.plan_id, "n_jobs": len(plan.jobs), "per_task": sizes}))
        return
    storage = open_storage(_ops_config(a.ops_config)["storage_url"])

    def marker(jid):
        data = storage.get_bytes(result_key(B.SCREEN_PHASE, jid))
        return None if data is None else json.loads(data)
    if not a.plan_dir:
        sys.exit(f"{a.cmd}: --plan-dir is required")
    pd = Path(a.plan_dir)
    if a.cmd == "yield-b":
        if not a.per_task:
            sys.exit("yield-b: --per-task is required")
        rep = B.screen_yield(B.screened(Plan.read(pd / "plan.json"), marker), per_task=a.per_task, k=a.k)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1, sort_keys=True))
        print(json.dumps({"needed_per_task": rep["needed_per_task"], "short_tasks": rep["short_tasks"],
                          "failed": {t: r["failed"] for t, r in rep["tasks"].items()},
                          "top_up_episodes": rep["top_up_episodes"]}))
        return
    if a.cmd == "select-b":
        if not a.per_task or a.seed is None:
            sys.exit("select-b: --per-task and --seed are required")
        rows = B.screened(Plan.read(pd / "plan.json"), marker)
        chosen = B.select_reruns(rows, per_task=a.per_task, k=a.k, seed=a.seed)
        rp = B.rerun_plan(rows, chosen)
        out.mkdir(parents=True, exist_ok=True)
        rp.write(out / "plan.json")
        (out / "episodes.json").write_text(json.dumps({"settings": {"per_task": a.per_task, "k": a.k, "seed": a.seed,
                                                                    "failed_share": B.FAILED_SHARE},
                                                       "plan_id": rp.plan_id, "chosen": chosen}, indent=1, sort_keys=True))
        print(json.dumps({"screened": len(rows), "failed": sum(not r["success"] for r in rows), "reruns": len(rp.jobs),
                          "plan_id": rp.plan_id}))
        return
    rp = Plan.read(pd / "plan.json")
    if a.cmd == "compare-b":
        prev = json.loads(out.read_text()) if out.exists() else {}
        done = {k: v for k, v in prev.get("rows", {}).items() if v.get("passed")}
        rep = B.compare_all(rp, storage.get_bytes, marker, done)
        out.write_text(json.dumps(rep, indent=1, sort_keys=True))
        print(json.dumps({k: rep[k] for k in ("compared", "pending", "passed", "all_done")}))
        if not rep["passed"]:
            B.record_mismatch(rep, mismatch=a.mismatch, stop_flag=a.stop_flag)
            sys.exit(76)
        if a.require_all and not rep["all_done"]:
            sys.exit(f"compare-b: {rep['pending']} re-runs without a marker")
        return
    # plan-b: the branch plan over the re-runs' points
    import pyarrow.parquet as pq
    from vlar_harness.config import load_config
    cfg = load_config()
    if not a.per_task or a.seed is None:
        sys.exit("plan-b: --per-task and --seed are required")
    if not json.loads(Path(a.compare).read_text())["all_done"] or not json.loads(Path(a.compare).read_text())["passed"]:
        sys.exit("plan-b: the comparison report must show every re-run compared and passed")
    chosen = json.loads((pd / "episodes.json").read_text())["chosen"]
    points = []
    for j in rp.jobs:
        m = marker(j.job_id)
        listed = {x["key"][len("snapshots/"):-len(".npz")] for x in m["result"]["artifacts"]
                  if x["key"].startswith("snapshots/") and x["key"].endswith(".npz")}
        data = storage.get_bytes(f"records/events/{B.SCREEN_PHASE}/{j.job_id}.parquet")
        points += B.rerun_points(j, pq.read_table(io.BytesIO(data)).to_pylist(), listed)
    step_limit, interval = int(cfg["env"]["step_limit"]), int(cfg["decision"]["interval_steps"])
    sel = B.select_points(points, chosen, k=a.k, seed=a.seed, step_limit=step_limit, interval=interval)
    plan, labels = B.branch_plan_b(points, sel, per_task=a.per_task, seed=a.seed, branches=cfg["branches"],
                                   step_limit=step_limit, interval=interval)
    out.mkdir(parents=True, exist_ok=True)
    plan.write(out / "plan.json")
    (out / "labels.json").write_text(json.dumps(labels, indent=1, sort_keys=True, default=int))
    selection = {"phase": plan.phase, "plan_id": plan.plan_id, "n_points": len(sel), "n_jobs": len(plan.jobs),
                 "settings": B.branch_settings(a.per_task, a.seed), "k": a.k, "rerun_plan_id": rp.plan_id}
    (out / "selection.json").write_text(json.dumps(selection, indent=1, sort_keys=True))
    print(json.dumps({k: selection[k] for k in ("phase", "n_points", "n_jobs", "plan_id")}))


def cmd_phase1(a) -> None:
    """Phase 1 after the handoff (amendment 65; vlar_ops/phase1.py): the budget, the pre-stated analyses (i) and (ii),
    and the Tranche A plan (on the Mac, from local copies of the base runs, events and markers)."""
    from . import phase1, pilot, replan
    out = Path(a.out)
    if a.cmd == "budget":
        if a.credit_usd is None or not a.pilot_report:
            sys.exit("budget: --credit-usd and --pilot-report are required")
        costs = json.loads(Path(a.pilot_report).read_text())["costs"]
        p = replan.Prices.from_pilot(costs)
        plan = phase1.Phase1Plan(credit_usd=a.credit_usd, untracked_usd=a.untracked_usd)
        b = phase1.budget(p, plan, hourly_usd=a.hourly_usd)
        rep = {"budget": b, "lines": phase1.lines(b, hourly_usd=a.hourly_usd), "prices": p.__dict__, "plan": plan.__dict__}
        if a.ledger_total_usd is not None:
            rep["guard"] = phase1.guard_limits(a.ledger_total_usd, b)
        out.write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps({"tranche_a": b["tranche_a"], "final": b["final"], "labels_usd": b["labels_usd"],
                          "guard": rep.get("guard")}, default=lambda x: round(x, 3)))
        return
    if a.cmd == "analyses":
        rep = phase1.run_analyses(Path(a.root), Path(a.plan_dir), n=a.per_task)
        out.write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps({"i_apply": rep["i"]["apply"], "ii_pick": rep["ii"]["pick"],
                          "i_splits": {k: round(v["gain_min"], 4) for k, v in rep["i"]["splits"].items()}}))
        return
    if a.cmd in B_CMDS:
        return cmd_phase1b(a)
    if not a.per_task or a.seed is None:      # plan-a, plan-f
        sys.exit(f"{a.cmd}: --per-task and --seed are required")
    if a.cmd == "plan-f":                     # the final tier, version (a): A and F one stratified draw
        if not a.a_plan_dir:
            sys.exit("plan-f: --a-plan-dir (Tranche A's plan directory) is required")
        ad = Path(a.a_plan_dir)
        asel = json.loads((ad / "selection.json").read_text())
        want = phase1.tranche_a_settings(int(asel["settings"]["per_task"]), a.seed)["selection"]
        if asel["phase"] != phase1.PHASE_A or asel["settings"] != want:
            sys.exit(f"plan-f: Tranche A's settings {asel['settings']} are not {want} (the seed must be A's)")
        earlier = {"phase": asel["phase"], "plan_id": asel["plan_id"], "per_task": int(asel["settings"]["per_task"]),
                   "points": set(json.loads((ad / "labels.json").read_text()))}
        sel = pilot.make_plan(Path(a.root), Path(a.markers), out, earlier=earlier,
                              **phase1.final_tier_settings(earlier["per_task"], a.per_task, a.seed))
    else:
        sel = pilot.make_plan(Path(a.root), Path(a.markers), out, **phase1.tranche_a_settings(a.per_task, a.seed))
    print(json.dumps({k: sel[k] for k in ("phase", "n_points", "n_jobs", "plan_id")}))


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
    s.add_argument("action", choices=["status", "reconcile", "charge", "disk-created", "disk-deleted", "daemon",
                                      "check", "phase-open", "phase-close", "phase-raise"])
    s.add_argument("--phase", help="check: the step's spend phase (must be the open one); phase-open/-close: its name")
    s.add_argument("--allocation-usd", type=float, help="phase-open: the phase's planned VM spend")
    s.add_argument("--raise-usd", type=float, help="phase-raise: add this to the open phase's line and stop point")
    s.add_argument("--reason", help="phase-raise: why (recorded in the ledger)")
    s.add_argument("--line-usd", type=float, help="phase-open: the most the phase's VM time may cost")
    s.add_argument("--margin-usd", type=float, help="phase-open: stop this far below the line (default: the drain "
                                                    "grace plus 5 minutes at the hourly price)")
    s.add_argument("--actual-usd", type=float)
    s.add_argument("--usd", type=float, help="charge: a cost outside the ledger (e.g. image builds)")
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

    su = sub.add_parser("supervise", help="size, start and supervise the workers of one phase (run_phase; "
                                          "vlar_ops/supervisor.py)")
    su.add_argument("--phase", required=True)
    su.add_argument("--workers", type=int, required=True)
    su.add_argument("--worker-args", required=True, help="VLAR_WORKER_ARGS without --max-jobs")
    su.add_argument("--max-jobs", type=int, help="default: from the memory limit")
    su.add_argument("--min-limit-gib", type=float,
                    help="the pilot's rule: as many workers (at most --workers) as get this memory limit each, or more")
    su.add_argument("--limit-gib", type=float,
                    help="amendment 64's RAM rule (the default): each worker's memory limit (default 5.5)")
    su.add_argument("--servers-peak-gb", type=float,
                    help="RAM rule: the servers' last measured peak resident memory, GB (else 1.25 x at start)")
    su.add_argument("--worker-env", default="/etc/vlar/worker.env")
    su.add_argument("--history")
    su.add_argument("--step", default="")
    su.add_argument("--max-restarts", type=int, default=3)
    su.add_argument("--poll-s", type=float, default=15.0)
    su.add_argument("--summary", help="write the supervisor's summary (sizing, restarts, recycles, peak "
                                              "memory, events) to this file")
    su.add_argument("--ops-config")
    su.set_defaults(fn=cmd_supervise)

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

    sm = sub.add_parser("smoke", help="post-freeze smoke checks on the L4 (vlar_ops/smoke.py)")
    sm.add_argument("cmd", choices=["plan", "probe", "branch", "compare"])
    sm.add_argument("--subjects", help="plan, compare: comma-separated episode ids (base/t2/i34,...)")
    sm.add_argument("--golden", help="probe, branch: golden.json")
    sm.add_argument("--smoke-url", help="compare: the re-runs' storage root")
    sm.add_argument("--boot", help="compare: the boot the re-runs ran on (default this boot)")
    sm.add_argument("--out", required=True, help="plan: the plan file; else the JSON report")
    sm.add_argument("--ops-config")
    sm.set_defaults(fn=cmd_smoke)
    pi = sub.add_parser("pilot", help="the pilot (PROPOSALS-2026-11-24 §4, §0.4; vlar_ops/pilot.py)")
    pi.add_argument("cmd", choices=["plan", "check", "report", "reorder", "long-episodes", "replan"])
    pi.add_argument("--subjects", default="checks.json",
                    help="check: the subjects file in --plan-dir (checks-iii.json: the corrected check (iii))")
    pi.add_argument("--root", help="plan: local copy of records/{runs,events}/base; report: also records/*/pilot "
                                   "and results/pilot")
    pi.add_argument("--markers", help="plan: local copy of results/base")
    pi.add_argument("--plan-dir", help="check, report: plan.json, labels.json, selection.json, checks.json")
    pi.add_argument("--reports", help="report: local copy of <storage>/pilot/reports/")
    pi.add_argument("--hourly-usd", type=float, default=0.472, help="report: VM price per hour")
    pi.add_argument("--phase-vm-usd", type=float, help="report: the ledger's VM spend in the pilot phase")
    pi.add_argument("--available-usd", type=float, help="replan: $40 less the ledger's total")
    pi.add_argument("--pilot-report", help="replan: the pilot report's JSON (its measured costs)")
    pi.add_argument("--stop-profile", default=pilot_defaults("STOP_PROFILE"),
                    help="report: the profile positives are scored under (amendment 63: cascader_nospread; the "
                         "pilot report's amendment 58 rule: cascader_full)")
    pi.add_argument("--own-basis", choices=["failure", "all"], default=pilot_defaults("OWN_BASIS"),
                    help="report: count a task's labelled failed-parent points (amendment 63) or all (amendment 62)")
    pi.add_argument("--staging", default="/var/lib/vlar/pilot/staging", help="check: where the re-runs are written")
    pi.add_argument("--out", required=True, help="plan: the output directory; check: the JSON report")
    pi.add_argument("--ops-config")
    pi.set_defaults(fn=cmd_pilot)
    p1 = sub.add_parser("phase1", help="Phase 1 after the handoff (amendment 65; vlar_ops/phase1.py)")
    p1.add_argument("cmd", choices=["budget", "analyses", "plan-a", "plan-f", *B_CMDS])
    p1.add_argument("--per-task-screen", type=int, help="plan-b-screen: screened episodes per task (~115)")
    p1.add_argument("--sizes-file", help="plan-b-screen: JSON {task: screened episodes} (amendment 68's top-up)")
    p1.add_argument("--front-load-base-runs", help="plan-b-screen: a base-runs CSV; each task's screen grows by "
                    "amendment 66 (4)'s top-up at its base failure rate (amendment 67)")
    p1.add_argument("--k", type=int, default=2, help="select-b, yield-b, plan-b: points per re-run episode")
    p1.add_argument("--compare", help="plan-b: the comparison report (compare-b's --out)")
    p1.add_argument("--stop-flag", help="compare-b: the spend guard's stop flag, written on any difference")
    p1.add_argument("--mismatch", help="compare-b: the mismatch marker, written before the stop flag on any "
                    "difference (740 ends with 76 on it, not as a budget stop)")
    p1.add_argument("--require-all", action="store_true", help="compare-b: fail while any re-run has no marker")
    p1.add_argument("--ops-config")
    p1.add_argument("--credit-usd", type=float, help="budget: the coupon credit left, less charges not yet posted")
    p1.add_argument("--untracked-usd", type=float, default=0.0, help="budget: charges outside the ledger in Phase 1")
    p1.add_argument("--ledger-total-usd", type=float, help="budget: the ledger's total now (for the guard's limits)")
    p1.add_argument("--pilot-report", help="budget: the pilot report's JSON (its measured costs)")
    p1.add_argument("--hourly-usd", type=float, default=0.6865, help="the Spot price with its L4 (catalog 2026-11-25)")
    p1.add_argument("--root", help="analyses: local records/{runs,events}/{base,pilot}; plan-a: records/*/base")
    p1.add_argument("--plan-dir", help="analyses: the pilot's plan directory (labels.json); select-b: the screen "
                    "plan; compare-b, plan-b: the re-run plan")
    p1.add_argument("--markers", help="plan-a: local copy of results/base")
    p1.add_argument("--per-task", type=int, help="analyses, plan-a: base points per task; plan-f: the final tier's "
                    "new base points per task (A's count is added for the one draw)")
    p1.add_argument("--a-plan-dir", help="plan-f: Tranche A's plan directory (selection.json, labels.json)")
    p1.add_argument("--seed", type=int, help="plan-a: the selection seed")
    p1.add_argument("--out", required=True, help="budget, analyses: the JSON report; plan-a: the output directory")
    p1.set_defaults(fn=cmd_phase1)
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
