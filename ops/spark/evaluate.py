"""The Spark's evaluation runs (SPEC §10), ops-side; no harness or `vlar_ops` change.

Every method's episodes use the evaluation rollout-seed namespace (`eval`: the same seed for every method at a task
and initial state) and scene seed 142 + init, each on a fresh simulator at setup count 2 (the re-plan's option (A)
for fresh-start runs, amendment 62 (2)). π0.5 is reached only through the gate (`pi05_gate.py`, one `/act` in flight).

Methods (`--method`):
- `xvla_only`: X-VLA alone (profile `none`). `pi05_only`: π0.5 alone (profile `none`, through the gate).
- `fixed`: routed, `router` profile, escalates at every `--spacing`-th decision point (index % spacing == 0).
- `random`: routed, `router` profile, escalates with probability `--rate` (`vlar_ops.plan.unit_uniform(rollout seed, "eval_random", index)`
  per decision point: reproducible and independent of call order).
  Scripted methods run to 520 steps with no live time cut and are scored offline from their event logs.
- `decider`: routed under the decider's own profile with the live time limit, deciding from `observe()`
  (`live.py`): `--decider-dir <vlar_train fold dir>` or `--decider dummy:<decider name>` (random weights).

    python evaluate.py plan --method M --out DIR [--tag T] [--tasks 0-9] [--inits 0-49] [method options]
    python evaluate.py run --jobs DIR/<phase>-jobs.jsonl --shard I/N --out DIR [--pi05-port 8779] [--limit K]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

FRESH_SETUP_COUNT = 2          # option (A): a fresh simulator per episode
METHODS = {"xvla_only": dict(mode="xvla_only", profile="none", record="off", time_limit=False),
           "pi05_only": dict(mode="pi05_only", profile="none", record="off", time_limit=False),
           "fixed": dict(mode="routed", profile="router", record="off", time_limit=False),
           "random": dict(mode="routed", profile="router", record="off", time_limit=False),
           "decider": dict(mode="routed", profile=None, record="off", time_limit=True)}


def span(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def cmd_plan(a):
    from vlar_ops.plan import episode_plan, load_collection_config
    m = dict(METHODS[a.method])
    policy = {"method": a.method}
    if a.method == "fixed":
        policy["spacing"] = int(a.spacing)
    if a.method == "random":
        policy["rate"] = float(a.rate)
    if a.method == "decider":
        from live import RandomDecider, TrainedDecider
        dec = RandomDecider(a.decider.split(":", 1)[1]) if a.decider and a.decider.startswith("dummy:") \
            else TrainedDecider.load(a.decider_dir)
        m["profile"] = dec.profile
        policy.update({**dec.meta, "decider_arg": a.decider or None, "decider_dir": a.decider_dir})
    ns = int(load_collection_config()["rollout_seed_namespaces"]["eval"])
    phase = f"eval_{a.method}" + (f"_{a.tag}" if a.tag else "")
    plan = episode_plan(phase, tasks=span(a.tasks), inits=span(a.inits), namespace=ns, mode=m["mode"],
                        profile=m["profile"], record=m["record"], order_seed=0, scene="project",
                        extra=lambda t, i: {"time_limit": m["time_limit"], "policy": policy})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dst = Path(a.out) / f"{phase}-jobs.jsonl"
    dst.write_text("".join(json.dumps({"job": asdict(j), "setup_count": FRESH_SETUP_COUNT}, default=float) + "\n"
                           for j in plan.jobs))
    print(json.dumps({"phase": phase, "jobs": len(plan.jobs), "namespace": ns, "profile": m["profile"],
                      "time_limit": m["time_limit"], "policy": policy, "out": str(dst)}, default=str))


def make_runner_class():
    from vlar_harness.collect import Runner, _run_row
    from vlar_ops.plan import unit_uniform

    class EvalRunner(Runner):
        """`Runner` for evaluation jobs: the same episode code, with the job's live time limit and its policy."""

        def __init__(self, *a, policy, **k):
            super().__init__(*a, **k)
            self.policy, self.decisions = policy, []

        def refuse(self, job):
            spec = job.spec
            scene = self.cfg.selected.get("scene_seed")
            if spec.get("scene", "project") != scene:
                return f"job wants scene_seed={spec.get('scene')}, this config has {scene}"
            return None

        def _choice(self, job, ep, dp):
            p = job.spec["policy"]
            if not dp.can_escalate:
                return "local", None
            if p["method"] == "fixed":
                return ("escalate" if dp.index % int(p["spacing"]) == 0 else "local"), None
            if p["method"] == "random":
                u = unit_uniform(ep.state.rollout_seed, "eval_random", dp.index)
                return ("escalate" if u < float(p["rate"]) else "local"), {"u": u}
            if p["method"] == "decider":
                return self.policy.decide(ep.observe())
            raise ValueError(f"no policy for {p['method']}")

        def _episode(self, job, out):
            spec = job.spec
            ep = self.make_episode(int(spec["task_id"]))
            episode_id = job.subject
            dp = ep.start(int(spec["init_idx"]), int(spec["rollout_seed"]), spec["mode"], episode_id,
                          profile=spec["profile"], record=spec["record"], time_limit=bool(spec.get("time_limit")))
            self.decisions = []
            if spec["mode"] == "routed":
                if self.policy is not None:
                    self.policy.new_episode()
                while dp is not None:
                    choice, info = self._choice(job, ep, dp)
                    self.decisions.append({"index": dp.index, "step": dp.step, "choice": choice, **(info or {})})
                    dp = ep.decide(choice).next_point
            else:
                ep.run_episode()
            res, st = ep.result(), ep.state
            out.add_run(_run_row(episode_id, "episode", st, res, episode_id=episode_id))
            out.add_steps(episode_id, ep.trajectory)
            out.add_events(episode_id, ep.own_events)
            return {"run_id": episode_id, "success": res.success, "steps": res.steps}

    return EvalRunner


def cmd_run(a):
    import pyarrow.parquet as pq
    from vlar_harness import provenance
    from vlar_harness.config import load_config
    from vlar_harness.servers.clients import PI05Client, XVLAClient
    from vlar_ops.pilot import warm_up
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
    jobs = [json.loads(line) for line in open(a.jobs)][i::n][:a.limit]
    phase = jobs[0]["job"]["phase"] if jobs else "none"
    pol = jobs[0]["job"]["spec"]["policy"] if jobs else {}
    policy = None
    if pol.get("method") == "decider":
        from live import RandomDecider, TrainedDecider
        arg = pol.get("decider_arg") or ""
        policy = RandomDecider(arg.split(":", 1)[1]) if arg.startswith("dummy:") else TrainedDecider.load(pol["decider_dir"])
    EvalRunner = make_runner_class()
    out = Path(a.out); res_path = out / f"{phase}-results-{i}.jsonl"
    done = {json.loads(line)["job_id"] for line in open(res_path)} if res_path.exists() else set()
    platform = json.loads(Path(os.environ["VLAR_SPARK_PLATFORM"]).read_text()) if os.environ.get("VLAR_SPARK_PLATFORM") else {}
    for x in jobs:
        job = Job(**x["job"])
        if job.job_id in done:
            continue
        episodes = ProductionEpisodes(cfg, xc, pc)
        try:
            t0 = time.monotonic()
            warm_up(episodes.sim(int(job.spec["task_id"])), x["setup_count"])
            t1 = time.monotonic()
            runner = EvalRunner(cfg, episodes, lambda key: None, out / "records", p_escalate=0.0, header=header,
                                policy=policy)
            runner(job)
            t2 = time.monotonic()
        finally:
            episodes.close()
        run_row = pq.read_table(runner.job_dir(job) / "records" / "runs.parquet").to_pylist()[0]
        line = {"job_id": job.job_id, "run_id": job.subject, "setup_count": x["setup_count"], "run": run_row,
                "policy": job.spec["policy"], "decisions": runner.decisions,
                "seconds": {"warm_up": round(t1 - t0, 2), "episode": round(t2 - t1, 2)},
                "calibration_hash": cfg.calibration_hash, "image_digests": header.get("image_digests"),
                "servers": {k: v.get("batching") for k, v in infos.items()}, "pi05_gate": gate, "platform": platform}
        with open(res_path, "a") as f:
            f.write(json.dumps(line, default=str) + "\n")
        print(f"{job.subject}: success={run_row.get('success')} steps={run_row.get('steps')} "
              f"end={run_row.get('end_reason')} pi05_steps={run_row.get('pi05_steps')} {line['seconds']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--method", choices=METHODS, required=True); p.add_argument("--out", required=True)
    p.add_argument("--tag", default=""); p.add_argument("--tasks", default="0-9"); p.add_argument("--inits", default="0-49")
    p.add_argument("--spacing", type=int, default=5); p.add_argument("--rate", type=float, default=0.1)
    p.add_argument("--decider", default=None); p.add_argument("--decider-dir", default=None)
    r = sub.add_parser("run")
    r.add_argument("--jobs", required=True); r.add_argument("--out", required=True); r.add_argument("--shard", default="0/1")
    r.add_argument("--limit", type=int, default=None); r.add_argument("--pi05-port", type=int, default=8779)
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
