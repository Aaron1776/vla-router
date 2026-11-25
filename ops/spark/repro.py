"""The Spark's reproduction of the L4's Phase 0 (π0.5 parity, X-VLA base rollouts), episode by episode, paired.

Runs in the sim image (`docker run ... vlar-sim:<commit> python ops/spark/repro.py ...`), against the production
servers. Nothing here changes the harness or the ops code: it rebuilds the L4's jobs with `vlar_ops.plan` exactly as
steps 300 and 400 planned them, brings a fresh simulator to each episode's reconstructed setup count with
`vlar_ops.pilot.warm_up` (References/l4-phase0/README.md), and runs it with `vlar_harness.collect.Runner`, as the
L4's workers did. Each episode's records stay in the output folder; one JSON line per episode carries the run row,
the setup count and the platform.

    python repro.py plan --phase base|pi05_parity --runs RUNS.csv --out DIR
        rebuild the plan; every CSV row must match a planned job (job id, task, init, rollout seed, mode, profile,
        record) and carry a setup count; writes DIR/<phase>-jobs.jsonl
    python repro.py run --jobs DIR/<phase>-jobs.jsonl --shard I/N --out DIR [--option scene_seed=pi05_parity] [--limit K]
        run the shard's episodes (resumable: episodes already in DIR/<phase>-results-I.jsonl are skipped)
    python repro.py compare --phase base|pi05_parity --runs RUNS.csv --results DIR
        the pass criteria (ops/spark/README.md, "Phase 0 reproduction"): paired exact McNemar against the L4
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

PLANS = {"base": dict(inits=range(50), mode="routed", profile="none", record="full", scene="project"),
         "pi05_parity": dict(inits=range(10), mode="pi05_only", profile="none", record="off", scene="pi05_parity")}


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def cmd_plan(a):
    from vlar_ops.plan import episode_plan, load_collection_config
    p = PLANS[a.phase]
    ns = int(load_collection_config()["rollout_seed_namespaces"][a.phase])
    plan = episode_plan(a.phase, tasks=range(10), inits=p["inits"], namespace=ns, mode=p["mode"],
                        profile=p["profile"], record=p["record"], order_seed=0, scene=p["scene"])
    by_subject = {j.subject: j for j in plan.jobs}
    problems, out = [], []
    for r in rows(a.runs):
        j = by_subject.get(r["run_id"])
        if j is None:
            problems.append(f"{r['run_id']}: not in the rebuilt plan"); continue
        s = j.spec
        want = (r["job_id"], int(r["task_id"]), int(r["init_idx"]), int(r["rollout_seed"]), r["mode"], r["profile"], r["record"])
        got = (j.job_id, s["task_id"], s["init_idx"], s["rollout_seed"], s["mode"], s["profile"], s["record"])
        if want != got:
            problems.append(f"{r['run_id']}: L4 {want} != rebuilt {got}")
        n = r.get("setup_count_reconstructed")
        if not n:
            problems.append(f"{r['run_id']}: no setup count")
        out.append({"job": asdict(j), "setup_count": int(n or 0), "l4_scene_seed": int(r["scene_seed"])})
    if len(out) != len(plan.jobs):
        problems.append(f"{len(out)} CSV rows for {len(plan.jobs)} planned jobs")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dst = Path(a.out) / f"{a.phase}-jobs.jsonl"
    dst.write_text("".join(json.dumps(x) + "\n" for x in sorted(out, key=lambda x: x["job"]["order"])))
    counts = sorted({x["setup_count"] for x in out})
    print(json.dumps({"phase": a.phase, "jobs": len(out), "matched": len(out) - len(problems), "problems": problems[:10],
                      "setup_counts": [counts[0], counts[-1]] if counts else None, "out": str(dst)}))
    raise SystemExit(1 if problems else 0)


def cmd_run(a):
    from vlar_harness import provenance
    from vlar_harness.collect import Runner
    from vlar_harness.config import load_config
    from vlar_harness.servers.clients import PI05Client, XVLAClient
    from vlar_ops.pilot import warm_up
    from vlar_ops.plan import Job
    from vlar_ops.vm_worker import ProductionEpisodes, check_servers
    import pyarrow.parquet as pq

    opts = dict(o.split("=", 1) for o in a.option)
    cfg = load_config(**opts)
    srv = cfg["servers"]
    clients = {"xvla": XVLAClient(srv["xvla"]["host"], int(srv["xvla"]["port"])),
               "pi05": PI05Client(srv["pi05"]["host"], int(srv["pi05"]["port"]))}
    infos = check_servers(cfg, clients)
    header = provenance.record_header(cfg)
    i, n = map(int, a.shard.split("/"))
    jobs = [json.loads(l) for l in open(a.jobs)][i::n][:a.limit]
    phase = jobs[0]["job"]["phase"] if jobs else "none"
    out = Path(a.out); res_path = out / f"{phase}-results-{i}.jsonl"
    done = {json.loads(l)["job_id"] for l in open(res_path)} if res_path.exists() else set()
    staging = out / "records"
    platform = json.loads(Path(os.environ.get("VLAR_SPARK_PLATFORM", "/dev/null")).read_text() or "{}") \
        if os.environ.get("VLAR_SPARK_PLATFORM") else {}
    for x in jobs:
        job = Job(**x["job"])
        if job.job_id in done:
            continue
        task = int(job.spec["task_id"])
        episodes = ProductionEpisodes(cfg, clients["xvla"], clients["pi05"])
        try:
            t0 = time.monotonic()
            warm_up(episodes.sim(task), x["setup_count"])
            t1 = time.monotonic()
            runner = Runner(cfg, episodes, lambda key: None, staging, p_escalate=0.0, header=header)
            artifacts, summary = runner(job)
            t2 = time.monotonic()
        finally:
            episodes.close()
        runs_file = runner.job_dir(job) / "records" / "runs.parquet"
        if not runs_file.exists():
            raise RuntimeError(f"{job.subject}: no staged runs table at {runs_file}")
        run_row = pq.read_table(runs_file).to_pylist()[0]
        line = {"job_id": job.job_id, "run_id": job.subject, "setup_count": x["setup_count"], "run": run_row,
                "seconds": {"warm_up": round(t1 - t0, 2), "episode": round(t2 - t1, 2)},
                "calibration_hash": cfg.calibration_hash, "image_digests": header.get("image_digests"),
                "servers": {k: v.get("batching") for k, v in infos.items()}, "platform": platform}
        with open(res_path, "a") as f:
            f.write(json.dumps(line, default=str) + "\n")
        print(f"{job.subject}: success={run_row.get('success')} steps={run_row.get('steps')} "
              f"loads={x['setup_count']} {line['seconds']}", flush=True)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: binomial(b + c, 1/2) on the discordant pairs."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    p = sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def cmd_compare(a):
    l4 = {r["run_id"]: r["success"] == "True" for r in rows(a.runs)}
    spark = {}
    for f in glob.glob(str(Path(a.results) / f"{a.phase}-results-*.jsonl")):
        for l in open(f):
            x = json.loads(l)
            spark[x["run_id"]] = bool(x["run"]["success"])
    both = sorted(set(l4) & set(spark))
    b = sum(1 for k in both if l4[k] and not spark[k])        # L4 success, Spark failure
    c = sum(1 for k in both if spark[k] and not l4[k])        # Spark success, L4 failure
    p = mcnemar_exact(b, c)
    res = {"phase": a.phase, "paired": len(both), "missing": len(set(l4) - set(spark)),
           "l4_success": sum(l4[k] for k in both), "spark_success": sum(spark[k] for k in both),
           "l4_only": b, "spark_only": c, "mcnemar_exact_p": p,
           "per_task": {t: {"l4": sum(l4[k] for k in both if k.split("/")[1] == f"t{t}"),
                            "spark": sum(spark[k] for k in both if k.split("/")[1] == f"t{t}")} for t in range(10)},
           }
    # Second criterion: the L4's own gate against the reference run (vlar_ops.parity for π0.5, the X-VLA parity of
    # SPEC §11 step 5 for base): exact McNemar p >= 0.05 against References/, paired by (task, init), 520 steps.
    from vlar_ops import reference_runs as RR
    ours = {(int(k.split("/")[1][1:]), int(k.split("/")[2][1:])): spark[k] for k in both}
    ref = RR.load_pi05(a.references) if a.phase == "pi05_parity" else RR.load_xvla(a.references)
    vs_ref = RR.compare(ours, ref, limit=520)
    res["vs_reference"] = {k: vs_ref[k] for k in vs_ref if not isinstance(vs_ref[k], (list, dict))}
    res["pass_vs_l4"] = len(both) == len(l4) and p >= 0.05
    res["pass_vs_reference"] = bool(vs_ref["within_sampling_error"])
    res["pass"] = res["pass_vs_l4"] and res["pass_vs_reference"]
    print(json.dumps(res, indent=1, default=str))
    raise SystemExit(0 if res["pass"] else 76)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--phase", choices=PLANS, required=True)
    p.add_argument("--runs", required=True); p.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--jobs", required=True); r.add_argument("--out", required=True)
    r.add_argument("--shard", default="0/1"); r.add_argument("--limit", type=int, default=None); r.add_argument("--option", action="append", default=[])
    c = sub.add_parser("compare"); c.add_argument("--phase", choices=PLANS, required=True)
    c.add_argument("--runs", required=True); c.add_argument("--results", required=True)
    c.add_argument("--references", default="/opt/vla-router/References")
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
