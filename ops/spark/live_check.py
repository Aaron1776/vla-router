"""Check that a live decider's inputs equal what training builds from the records (SPEC §6), on real episodes.

A live decider (`live.py`) builds its input row from `observe()`; training (`vlar_train.data`) builds it from the
points table, the event log and the frozen calibration. This runs evaluation episodes with a trained decider driving,
records each decision point's row exactly as `collect.Runner` does under record `full` (`JobOutput.add_point`, the
same arguments), keeps the rows the decider built, then rebuilds the same rows offline and compares them column by
column. A second decider of the same profile can be built on the side from the same `observe()` (it does not act), so
one episode set covers a variant and its plain decider.

    python live_check.py run --jobs JOBS.jsonl --out DIR [--side FOLD_DIR] [--shard I/N] [--limit K] [--pi05-port 8779]
        JOBS.jsonl: `evaluate.py plan --method decider --decider-dir ...`. Episodes run as `evaluate.py run` runs
        them (fresh simulator, setup count, live time limit, π0.5 through the gate), with record `full`. No
        snapshots and no images are written: only the points rows, which carry the snapshot id as a name.
    python live_check.py compare --run DIR [--out FILE] [--eval DIR]
        per decider and column: points compared, the largest absolute difference, the points table's storage type.
        `--eval`: an `evaluate.py run` output of the same jobs (record `off`), to compare the decisions made.
        Exit 1 if any column differs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

SEP = "|"       # capture keys: <decider>|<column>, <decider>|x, <decider>|remaining_s, index


def make_check_runner():
    from evaluate import make_runner_class
    from vlar_harness.collect import _run_row
    from vlar_harness.snapshot import SnapshotStore
    EvalRunner = make_runner_class()

    class CheckRunner(EvalRunner):
        """`EvalRunner`'s decider episode, with the points rows of a record-`full` run and the built inputs kept."""

        def __init__(self, *a, side=None, **k):
            super().__init__(*a, **k)
            self.side, self.captured = side, []

        def _episode(self, job, out):
            spec = job.spec
            ep = self.make_episode(int(spec["task_id"]))
            episode_id = job.subject
            dp = ep.start(int(spec["init_idx"]), int(spec["rollout_seed"]), spec["mode"], episode_id,
                          profile=spec["profile"], record="full", time_limit=bool(spec.get("time_limit")))
            self.decisions, self.captured = [], []
            deciders = [d for d in (self.policy, self.side) if d is not None]
            for d in deciders:
                d.new_episode()
            while dp is not None:
                m = ep.point_meas
                if m is not None and m["index"] == dp.index:           # as collect.Runner._episode, record full
                    out.add_point(episode_id, m, budget_remaining_steps=dp.budget_remaining_steps,
                                  snapshot_id=SnapshotStore.make_id(episode_id, dp.index), n_samples=ep.K)
                choice, info = "local", None
                if dp.can_escalate:
                    obs = ep.observe()
                    choice, info = self.policy.decide(obs)
                    if self.side is not None:
                        self.side.build(obs)
                    self.captured.append({"index": dp.index, **{d.name: d.last_inputs for d in deciders}})
                self.decisions.append({"index": dp.index, "step": dp.step, "choice": choice, **(info or {})})
                dp = ep.decide(choice).next_point
            res, st = ep.result(), ep.state
            out.add_run(_run_row(episode_id, "episode", st, res, episode_id=episode_id))
            out.add_steps(episode_id, ep.trajectory)
            out.add_events(episode_id, ep.own_events)
            return {"run_id": episode_id, "success": res.success, "steps": res.steps}

    return CheckRunner


def pack(captured: list[dict]) -> dict[str, np.ndarray]:
    """One episode's captures as arrays (rows = the points `observe()` served, in order)."""
    out = {"index": np.array([c["index"] for c in captured], dtype=np.int64)}
    for name in (k for k in (captured[0] if captured else {}) if k != "index"):
        cols = {c: np.stack([cap[name][0][c] for cap in captured]) for c in captured[0][name][0]}
        for c, a in cols.items():
            out[f"{name}{SEP}{c}"] = a
        out[f"{name}{SEP}x"] = np.concatenate([cap[name][1] for cap in captured], axis=0)
        out[f"{name}{SEP}remaining_s"] = np.array([cap[name][2] for cap in captured], dtype=np.float64)
    return out


def cmd_run(a):
    import pyarrow.parquet as pq
    from live import TrainedDecider
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
    if not jobs:
        return
    pol = jobs[0]["job"]["spec"]["policy"]
    if pol.get("method") != "decider" or not pol.get("decider_dir"):
        raise SystemExit("the jobs must be a trained decider's (evaluate.py plan --method decider --decider-dir)")
    policy = TrainedDecider.load(pol["decider_dir"])
    side = TrainedDecider.load(a.side) if a.side else None
    if side is not None and (side.profile != policy.profile or side.name == policy.name):
        raise SystemExit(f"the side decider must be another decider of profile {policy.profile} "
                         f"(it is {side.name}, profile {side.profile}): observe() serves the run's own profile")
    phase = jobs[0]["job"]["phase"]
    out = Path(a.out); (out / "captures").mkdir(parents=True, exist_ok=True)
    res_path = out / f"{phase}-results-{i}.jsonl"
    done = {json.loads(line)["job_id"] for line in open(res_path)} if res_path.exists() else set()
    platform = json.loads(Path(os.environ["VLAR_SPARK_PLATFORM"]).read_text()) if os.environ.get("VLAR_SPARK_PLATFORM") else {}
    CheckRunner = make_check_runner()
    for x in jobs:
        job = Job(**x["job"])
        if job.job_id in done:
            continue
        episodes = ProductionEpisodes(cfg, xc, pc)
        try:
            warm_up(episodes.sim(int(job.spec["task_id"])), x["setup_count"])
            t0 = time.monotonic()
            runner = CheckRunner(cfg, episodes, lambda key: None, out / "records", p_escalate=0.0, header=header,
                                 policy=policy, side=side)
            runner(job)
            t1 = time.monotonic()
        finally:
            episodes.close()
        np.savez(out / "captures" / f"{job.job_id}.npz", **pack(runner.captured))
        run_row = pq.read_table(runner.job_dir(job) / "records" / "runs.parquet").to_pylist()[0]
        line = {"job_id": job.job_id, "run_id": job.subject, "phase": phase, "setup_count": x["setup_count"],
                "run": run_row, "policy": job.spec["policy"], "side": side.meta if side else None,
                "decisions": runner.decisions, "observed_points": len(runner.captured), "seconds": round(t1 - t0, 2),
                "calibration_hash": cfg.calibration_hash, "image_digests": header.get("image_digests"),
                "servers": {k: v.get("batching") for k, v in infos.items()}, "pi05_gate": gate, "platform": platform}
        with open(res_path, "a") as f:
            f.write(json.dumps(line, default=str) + "\n")
        print(f"{job.subject}: success={run_row.get('success')} steps={run_row.get('steps')} "
              f"end={run_row.get('end_reason')} pi05_steps={run_row.get('pi05_steps')} "
              f"observed={len(runner.captured)} {line['seconds']} s", flush=True)


# ---------------------------------------------------------------- compare

def offline_rows(points: list[dict], events: list[dict], indices, decider: str, *, limit_s: float, thresholds,
                 charges) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """What training builds for `decider` at the episode's points `indices`: its columns (`data.point_columns`,
    `data.history_columns`), the remaining time under its profile replayed from the event log (as
    `vlar_harness.labels`: limit less `clock.replay` of the prefix) and the input rows (`deciders.inputs`)."""
    from vlar_harness import clock as clk
    from vlar_train import data as DT
    from vlar_train import deciders as D
    from vlar_train import history as H
    by_index = {int(p["index"]): p for p in points}
    rows = [by_index[int(k)] for k in indices]
    names = [c[len(H.PREFIX):] for c in D.COLUMNS[decider] if c.startswith(H.PREFIX)]
    hist = DT.history_columns(points, {r["snapshot_id"] for r in rows}, names, thresholds=thresholds) if names else {}
    cols: dict[str, list] = {c: [] for c in D.COLUMNS[decider]}
    for r in rows:
        pc = {**DT.point_columns(r, thresholds=thresholds), **hist.get(r["snapshot_id"], {})}
        for c in cols:
            cols[c].append(pc[c])
    profile = D.profile_of(decider)
    rem = np.array([limit_s - clk.replay(DT.prefix_of(events, int(k)), profile, charges) for k in indices])
    arrays = {c: np.stack(v) if v else np.zeros((0, 1)) for c, v in cols.items()}
    return arrays, rem, D.inputs(arrays, rem, decider)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


STORED_AS = {"budget_frac": "budget_remaining_steps", "steps_frac": "step", "chunk_pos_frac": "chunk_pos",
             "grip_shut_frac": "prev_actions, prev_gripper_qpos (computed offline)",
             "grip_partial_frac": "prev_actions, prev_gripper_qpos (computed offline)"}


def cmd_compare(a):
    import pyarrow.parquet as pq
    from live import ensure_decider
    from vlar_harness import clock as clk
    from vlar_harness.config import load_config
    cfg = load_config()
    g = cfg["measurement"]["gripper_thresholds"]
    thresholds, charges = (float(g["shut_width_m"]), float(g["open_width_m"])), clk.Charges.from_config(cfg)
    lines = [json.loads(line) for f in sorted(glob.glob(f"{a.run}/*-results-*.jsonl")) for line in open(f)]
    rep: dict[str, dict] = {}
    types: dict[str, str] = {}
    episodes = []
    for x in lines:
        d = glob.glob(f"{a.run}/records/*/{x['job_id']}/records")[0]
        pts = pq.read_table(f"{d}/points.parquet")
        types = {f.name: str(f.type) for f in pts.schema}
        points = pts.to_pylist()
        ev = sorted(pq.read_table(f"{d}/events.parquet").to_pylist(), key=lambda e: int(e["seq"]))
        events = [json.loads(e["event"]) for e in ev]
        cap = np.load(f"{a.run}/captures/{x['job_id']}.npz")
        idx = cap["index"]
        after_pi05 = sum(bool(p["prev_segment_pi05"]) for p in points if int(p["index"]) in set(idx.tolist()))
        episodes.append({"run_id": x["run_id"], "points_rows": len(points), "observed": int(len(idx)),
                         "observed_after_pi05_segment": int(after_pi05),
                         "escalations": sum(dc["choice"] == "escalate" for dc in x["decisions"])})
        if not len(idx):
            continue
        for name in sorted({k.split(SEP)[0] for k in cap.files if SEP in k}):
            ensure_decider(name)
            cols, rem, xo = offline_rows(points, events, idx, name, limit_s=float(cfg.time_limit_s(int(x["run"]["task_id"]))),
                                         thresholds=thresholds, charges=charges)
            r = rep.setdefault(name, {"points": 0, "episodes": 0, "columns": {}, "remaining_s": 0.0, "x": 0.0})
            r["points"] += int(len(idx)); r["episodes"] += 1
            for c, off in cols.items():
                e = r["columns"].setdefault(c, {"width": int(off.shape[1]), "max_abs_diff": 0.0,
                                                "stored_as": types.get(STORED_AS.get(c, c), STORED_AS.get(c, "history: computed from the rows"))})
                e["max_abs_diff"] = max(e["max_abs_diff"], max_abs(cap[f"{name}{SEP}{c}"], off))
            r["remaining_s"] = max(r["remaining_s"], max_abs(cap[f"{name}{SEP}remaining_s"], rem))
            r["x"] = max(r["x"], max_abs(cap[f"{name}{SEP}x"], xo))
    out = {"run": str(a.run), "calibration_hash": cfg.calibration_hash, "episodes": episodes, "deciders": rep}
    for name, r in rep.items():
        r["differing_columns"] = sorted(c for c, e in r["columns"].items() if e["max_abs_diff"] != 0.0)
        r["equal"] = not r["differing_columns"] and r["remaining_s"] == 0.0 and r["x"] == 0.0
    if a.eval:
        ev_lines = {json.loads(line)["run_id"]: json.loads(line) for f in glob.glob(f"{a.eval}/*-results-*.jsonl") for line in open(f)}
        same = []
        for x in lines:
            y = ev_lines.get(x["run_id"])
            if y is None:
                continue
            n = min(len(x["decisions"]), len(y["decisions"]))
            first = next((k for k in range(n) if x["decisions"][k] != y["decisions"][k]), None)
            same.append({"run_id": x["run_id"], "decisions": [len(x["decisions"]), len(y["decisions"])],
                         "identical": first is None and len(x["decisions"]) == len(y["decisions"]),
                         "first_difference": None if first is None else x["decisions"][first]["index"],
                         "success": [bool(x["run"]["success"]), bool(y["run"]["success"])]})
        out["against_record_off_run"] = same
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    brief = {n: {"points": r["points"], "episodes": r["episodes"], "equal": r["equal"], "differing_columns":
                 {c: r["columns"][c] for c in r["differing_columns"]}, "remaining_s_max_abs_diff": r["remaining_s"],
                 "x_max_abs_diff": r["x"]} for n, r in rep.items()}
    print(json.dumps({"episodes": len(episodes), "observed_points": sum(e["observed"] for e in episodes),
                      "observed_after_pi05_segment": sum(e["observed_after_pi05_segment"] for e in episodes),
                      "deciders": brief, "against_record_off_run": out.get("against_record_off_run")}, default=float))
    raise SystemExit(0 if rep and all(r["equal"] for r in rep.values()) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--jobs", required=True); r.add_argument("--out", required=True); r.add_argument("--side", default=None)
    r.add_argument("--shard", default="0/1"); r.add_argument("--limit", type=int, default=None)
    r.add_argument("--pi05-port", type=int, default=8779)
    c = sub.add_parser("compare")
    c.add_argument("--run", required=True); c.add_argument("--out", default=None); c.add_argument("--eval", default=None)
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
