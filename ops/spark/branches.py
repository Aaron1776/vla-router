"""Branch jobs on the Spark (SPEC §7, §12): the planner's b1 and b2 jobs at base decision points, run from the Spark's
own base snapshots, and the dataset folder a trainer reads. Ops-side only: the plan is `vlar_ops.branchplan`'s, every
job runs through `vlar_harness.collect.Runner` unchanged, and π0.5 is reached only through the gate (`pi05_gate.py`).

    python branches.py plan --base DIR --out PLAN_DIR --phase NAME [--per-task 4] [--failed-share 0.6667] [--seed S]
        base decision points from the staged base records (`vlar_ops.pilot.base_points`), a stratified draw per
        task (`stratified_selection`, amendment 57) and the branch plan with the collection tunables (seeds, enabled
        branches, tiers, order seed). Writes plan.json, labels.json (the label map) and selection.json, as the
        L4's plan commands do, and parent-check.json: one b1 job with seed index 1 per drawn point, which the
        planner never plans for base parents (the parent is that seed, §7), kept out of the plan and the dataset.
    python branches.py run --plan PLAN_DIR/plan.json --base DIR --out DIR [--shard I/N] [--limit K] [--pi05-port 8779]
        run the shard's jobs (resumable). Refuses unless π0.5's /info shows the gate with one request in flight.
    python branches.py assemble --base DIR --runs DIR --plan-dir PLAN_DIR --out ROOT [--note TEXT]
        the dataset layout (`vlar_harness.dataset`: records/<table>/<phase>/<job_id>.parquet, snapshots/<id>.npz)
        from the staged job folders, by hard link (copy if the link is refused), with the plan files in ROOT/plan
        and a README naming what the folder is. Standard library only.

Staged layout (what `Runner` leaves): <dir>/records/<phase>/<job_id>/{records/<table>.parquet, snapshots/<id>.npz}.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

TABLES = ("runs", "events", "points", "steps")


# ---------------------------------------------------------------- staged files

def staged_parts(staged: Path | str, table: str, phase: str = "*") -> list[Path]:
    """Every job's part of `table` in a staging folder, sorted."""
    return sorted(Path(p) for p in glob.glob(str(Path(staged) / "records" / phase / "*" / "records" / f"{table}.parquet")))


def staged_snapshots(base: Path | str) -> dict[str, Path]:
    """{snapshot id: file} for every snapshot staged under `base`."""
    out: dict[str, Path] = {}
    for job in glob.glob(str(Path(base) / "records" / "*" / "*" / "snapshots")):
        for f in Path(job).rglob("*.npz"):
            sid = f.relative_to(job).with_suffix("").as_posix()
            if sid in out:
                raise RuntimeError(f"snapshot {sid} staged twice: {out[sid]} and {f}")
            out[sid] = f
    return out


def snapshot_reader(base: Path | str):
    """`read_bytes(key)` for `Runner`: `snapshots/<id>.npz` from the staged base job that recorded it, else None."""
    def read(key: str):
        if not key.startswith("snapshots/") or ".." in key.split("/"):
            return None
        hits = glob.glob(str(Path(base) / "records" / "*" / "*" / key))
        if len(hits) > 1:
            raise RuntimeError(f"{key}: staged {len(hits)} times under {base}")
        return Path(hits[0]).read_bytes() if hits else None
    return read


def _place(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def assemble(staged: list[Path], root: Path, *, snapshots: dict[str, Path], plan_dir: Path, note: str) -> dict:
    """Dataset layout under `root` from staging folders. A (table, phase, job) staged twice is an error."""
    seen, how = {}, {"link": 0, "copy": 0}
    for s in staged:
        for table in TABLES:
            for part in staged_parts(s, table):
                job, phase = part.parent.parent.name, part.parent.parent.parent.name
                key = f"records/{table}/{phase}/{job}.parquet"
                if key in seen:
                    raise RuntimeError(f"{key} staged twice: {seen[key]} and {part}")
                seen[key] = part
                how[_place(part, root / key)] += 1
    for sid, src in sorted(snapshots.items()):
        how[_place(src, root / "snapshots" / f"{sid}.npz")] += 1
    for f in sorted(plan_dir.glob("*.json")):
        how[_place(f, root / "plan" / f.name)] += 1
    counts: dict[str, int] = {}
    for key in seen:
        t, ph = key.split("/")[1:3]
        counts[f"{t}/{ph}"] = counts.get(f"{t}/{ph}", 0) + 1
    (root / "README.md").write_text(
        f"# {note}\n\nBuilt by `ops/spark/branches.py assemble`. Layout: `records/<table>/<phase>/<job_id>.parquet`, "
        "`snapshots/<snapshot_id>.npz`, `plan/` (plan.json, labels.json = the label map, selection.json).\n\n"
        "Parts per table and phase:\n\n" + "".join(f"- {k}: {v}\n" for k, v in sorted(counts.items())))
    return {"parts": counts, "snapshots": len(snapshots), "files": how}


# ---------------------------------------------------------------- plan

def cmd_plan(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vlar_harness.config import load_config
    from vlar_ops.branchplan import branch_plan, stratified_selection
    from vlar_ops.pilot import base_points
    from vlar_ops.plan import Job, job_id
    from vlar_ops.tunables import load_tunables

    def table(name, columns=None):
        parts = staged_parts(a.base, name, "base")
        if not parts:
            raise SystemExit(f"no staged {name} parts under {a.base}/records/base")
        return pa.concat_tables([pq.read_table(p, columns=columns) for p in parts]).to_pylist()

    cfg, tun = load_config(), load_tunables()
    step_limit, interval = int(cfg["env"]["step_limit"]), int(cfg["decision"]["interval_steps"])
    points, success, kinds = base_points(table("runs"), table("events", ["run_id", "t", "event"]))
    have = staged_snapshots(a.base)
    staged = [p for p in points if p.snapshot_id in have]
    dp = {"scheme": "stratified", "seed": int(a.seed if a.seed is not None else tun["dp_sampling"]["seed"]),
          "per_task": int(a.per_task), "failed_share": float(a.failed_share), "min_calls": 1}
    sel = stratified_selection(staged, success, step_limit=step_limit, interval=interval, dp_sampling=dp)
    b = tun["branches"]
    plan, labels = branch_plan(a.phase, staged, branches=cfg["branches"], step_limit=step_limit, interval=interval,
                               seeds=b["seeds"], enabled=b["enabled"], tier_one_seeds=b["tier_one_seeds"],
                               order_seed=tun["order_seed"], dp_sampling=dp, selection=sel)
    check = [asdict(Job(job_id(sid, "b1_continue", 1), f"{a.phase}_parentcheck", "b1_continue", sid, 1, 1, k,
                        {"snapshot_id": sid, "branch": "b1_continue", "pi05_calls": 0, "seed_index": 1,
                         "task_id": e["point"]["task_id"], "init_idx": e["point"]["init_idx"], "source": "base",
                         "settings": e["settings"]}))
             for k, (sid, e) in enumerate(sorted(labels.items()))]
    by_task: dict[int, dict[str, int]] = {}
    for s in sel.values():
        row = by_task.setdefault(s["stratum"]["task_id"], {"failure": 0, "success": 0})
        row[s["stratum"]["parent_outcome"]] += 1
    selection = {"phase": a.phase, "settings": dp, "n_points": len(sel), "n_jobs": len(plan.jobs),
                 "plan_id": plan.plan_id, "base_points": len(points), "base_points_staged": len(staged),
                 "base_episodes": len(success), "drawn": {str(t): v for t, v in sorted(by_task.items())},
                 "kinds": {k: sum(1 for s in sel if kinds[s] == k) for k in sorted(set(kinds.values()))},
                 "calibration_hash": cfg.calibration_hash}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plan.write(out / "plan.json")
    for name, obj in (("labels", labels), ("selection", selection), ("parent-check", check)):
        (out / f"{name}.json").write_text(json.dumps(obj, indent=1, sort_keys=True, default=int))
    print(json.dumps({k: selection[k] for k in ("phase", "n_points", "n_jobs", "plan_id", "base_points",
                                                "base_points_staged", "drawn")}))


# ---------------------------------------------------------------- run

def load_jobs(path: Path | str) -> list[dict]:
    """Jobs of a plan.json (`Plan.write`: a header line, then one job per line) or of a JSON list of jobs."""
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        return list(json.loads(text))
    lines = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not lines or "plan_id" not in lines[0]:
        raise ValueError(f"{path}: not a plan file")
    if lines[0]["n_jobs"] != len(lines) - 1:
        raise ValueError(f"{path}: header says {lines[0]['n_jobs']} jobs, file has {len(lines) - 1}")
    return lines[1:]


def cmd_run(a):
    import pyarrow.parquet as pq
    from vlar_harness import provenance
    from vlar_harness.collect import Runner
    from vlar_harness.config import load_config
    from vlar_harness.servers.clients import PI05Client, XVLAClient
    from vlar_ops.plan import Job
    from vlar_ops.vm_worker import ProductionEpisodes, check_servers

    cfg = load_config()
    srv = cfg["servers"]
    xc = XVLAClient(srv["xvla"]["host"], int(srv["xvla"]["port"]))
    pc = PI05Client(srv["pi05"]["host"], int(a.pi05_port))
    infos = check_servers(cfg, {"xvla": xc, "pi05": pc})
    gate = infos["pi05"].get("gate") or {}
    if gate.get("mode") != "gate" or gate.get("max_in_flight") != 1:
        raise SystemExit(f"π0.5 is not behind the gate (port {a.pi05_port}: {gate or 'no gate block in /info'})")
    header = provenance.record_header(cfg)
    i, n = map(int, a.shard.split("/"))
    jobs = load_jobs(a.plan)[i::n][:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res_path = out / f"results-{i}.jsonl"
    done = {json.loads(line)["job_id"] for line in open(res_path)} if res_path.exists() else set()
    platform = json.loads(Path(os.environ["VLAR_SPARK_PLATFORM"]).read_text()) if os.environ.get("VLAR_SPARK_PLATFORM") else {}
    read = snapshot_reader(a.base)
    for x in jobs:
        job = Job(**x)
        if job.job_id in done:
            continue
        if job.kind == "episode":
            raise SystemExit(f"{job.job_id}: an episode job; this driver runs branch jobs only")
        episodes = ProductionEpisodes(cfg, xc, pc)
        try:
            t0 = time.monotonic()
            runner = Runner(cfg, episodes, read, out / "records", p_escalate=0.0, header=header)
            _, summary = runner(job)
            t1 = time.monotonic()
        finally:
            episodes.close()
        run_row = pq.read_table(runner.job_dir(job) / "records" / "runs.parquet").to_pylist()[0]
        line = {"job_id": job.job_id, "run_id": summary["run_id"], "phase": job.phase, "run": run_row,
                "seconds": round(t1 - t0, 2), "calibration_hash": cfg.calibration_hash,
                "image_digests": header.get("image_digests"), "servers": {k: v.get("batching") for k, v in infos.items()},
                "pi05_gate": gate, "platform": platform}
        with open(res_path, "a") as f:
            f.write(json.dumps(line, default=str) + "\n")
        print(f"{summary['run_id']}: success={run_row.get('success')} steps={run_row.get('steps')} "
              f"pi05_requests={run_row.get('pi05_requests')} {line['seconds']} s", flush=True)


# ---------------------------------------------------------------- assemble

def cmd_assemble(a):
    plan_dir = Path(a.plan_dir)
    labels = json.loads((plan_dir / "labels.json").read_text())
    have = staged_snapshots(a.base)
    missing = sorted(set(labels) - set(have))
    if missing:
        raise SystemExit(f"{len(missing)} labelled snapshots are not staged under {a.base}, e.g. {missing[0]}")
    snaps = {sid: have[sid] for sid in labels}
    rep = assemble([Path(a.base), Path(a.runs)], Path(a.out), snapshots=snaps, plan_dir=plan_dir, note=a.note)
    print(json.dumps(rep, sort_keys=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--base", required=True); p.add_argument("--out", required=True); p.add_argument("--phase", required=True)
    p.add_argument("--per-task", type=int, default=4); p.add_argument("--failed-share", type=float, default=2 / 3)
    p.add_argument("--seed", type=int, default=None)
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True); r.add_argument("--base", required=True); r.add_argument("--out", required=True)
    r.add_argument("--shard", default="0/1"); r.add_argument("--limit", type=int, default=None)
    r.add_argument("--pi05-port", type=int, default=8779)
    s = sub.add_parser("assemble")
    s.add_argument("--base", required=True); s.add_argument("--runs", required=True); s.add_argument("--plan-dir", required=True)
    s.add_argument("--out", required=True); s.add_argument("--note", default="Dataset folder")
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    main()
