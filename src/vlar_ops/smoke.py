"""Post-freeze smoke checks on the L4 with the real models (the user's item 5, 2026-11-24): `vlar-ops smoke ...`.

Run by `ops/vm/postfreeze/700-smoke.sh` after the golden-set self-check has passed on the same boot. Nothing here
writes to the dataset: re-runs go to a separate storage root, and the other checks only compare in memory.

- `plan --subjects base/t2/i34,...`: a plan of the exact jobs the dataset ran, read from their result markers (the
  base generator digest changed after `0a0cd4c`, so re-planning could give other specs; these are the originals).
- `probe --golden golden.json`: probe neutrality with the real models. One golden snapshot, X-VLA continuing to the
  end twice: once as a live `cascader_full` decider (its measurements at every point, `observe()` wherever a
  decision is offered: chunk-time samples and mid-point probes), once as profile `none` with no measurement client
  at all. Executed steps, events (without the run's `record`/`profile` settings) and outcome must be equal bit for
  bit, and both equal to the golden record. The Mac version (`tests/reference/test_measurements.py`) uses stand-ins.
- `branch --golden golden.json`: π0.5 determinism. The golden set never calls π0.5 (base runs never escalate), so a
  one-call branch (`b2_one_call`) from one golden snapshot runs twice; trajectories, events and outcome must be equal.
- `compare --smoke-url URL --subjects ...`: each re-run against the dataset's original: the same artifact keys, the
  four record tables bit for bit (SHA-256, else the first differing column and row), the images' PNGs, the snapshots
  (all but their provenance header); the marker's job and summary; and that the re-run's marker carries the ops
  waiver against the freeze's ops digest, this boot's id, and this boot's golden pass with its image check.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .plan import Job, Plan, job_id
from .queue import result_key

PROBE_PROFILE = "cascader_full"
BRANCH = "b2_one_call"
STEP_KEYS = ("action", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
RUN_SETTINGS = ("record", "profile")       # what the two probe runs set differently; not executed behaviour
OUTCOME = ("success", "end_reason", "steps", "success_step", "pi05_steps", "pi05_requests", "pi05_failed_requests")


class SmokeError(RuntimeError):
    pass


# ---------------------------------------------------------------- plan

def plan_from_markers(storage, subjects: Sequence[str], phase: str = "base") -> Plan:
    jobs = []
    for s in subjects:
        data = storage.get_bytes(result_key(phase, job_id(s, "episode", 1)))
        if data is None:
            raise SmokeError(f"{s}: no result marker in the dataset")
        jobs.append(Job(**json.loads(data)["job"]))
    return Plan(phase, {"smoke": True, "from": "result markers", "subjects": list(subjects)}, jobs)


# ---------------------------------------------------------------- executed behaviour, bit for bit

def trace(trajectory: Sequence[Mapping]) -> list[tuple]:
    import numpy as np
    out = []
    for r in trajectory:
        row = [int(r["step"]), r["driver"]]
        for k in STEP_KEYS:
            a = np.ascontiguousarray(r[k])
            row += [a.dtype.str, a.shape, a.tobytes()]
        out.append(tuple(row))
    return out


def events_text(events: Sequence[Mapping], drop: Sequence[str] = ()) -> list[str]:
    return [json.dumps({k: v for k, v in e.items() if k not in drop}, sort_keys=True, allow_nan=True)
            for e in events]


def outcome(result) -> dict:
    return {k: getattr(result, k) for k in OUTCOME}


def first_difference(a: Sequence, b: Sequence, what: str) -> Optional[str]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"{what} differ first at item {i}"
    if len(a) != len(b):
        return f"{what}: {len(a)} vs {len(b)}"
    return None


class CountingClient:
    """The measurement client with a count of its calls, so a run shows it really measured."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = {"sample": 0, "sample_seeds": 0, "features": 0}

    def sample(self, payload, seeds):
        self.calls["sample"] += 1
        self.calls["sample_seeds"] += len(seeds)
        return self.inner.sample(payload, seeds)

    def features(self, payload):
        self.calls["features"] += 1
        return self.inner.features(payload)


def continue_locally(ep, snapshot, snapshot_id: str, *, profile: str, measure: bool) -> dict:
    """Restore and let X-VLA continue to the end as a live decider of `profile` (measure=True: through the counting
    measurement client, `observe()` wherever a decision is offered), or with no measurement client (measure=False)."""
    from . import golden
    client = CountingClient(ep.measure_client) if measure else None
    ep.measure_client = client
    dp = ep.restore(snapshot, snapshot_id=snapshot_id)
    if dp is None:
        raise SmokeError(f"{snapshot_id}: not at a decision point of a running episode")
    ep.state.record, ep.state.profile = "off", profile
    observed = probe_points = 0
    while dp is not None:
        if measure and dp.can_escalate:
            ep.observe()
            observed += 1
            probe_points += ep.point_meas is not None and ep.point_meas.get("probe_raw") is not None
        dp = ep.decide("local").next_point
    import numpy as np
    got = {"step": np.array([int(r["step"]) for r in ep.trajectory], dtype=np.int64)}
    for name, dtype, width in golden.FIELDS:
        key = golden.TRAJ_KEYS.get(name, name)
        got[name] = np.array([np.asarray(r[key], dtype=dtype).ravel() for r in ep.trajectory],
                             dtype=dtype).reshape(-1, width)
    res = ep.result()
    return {"trace": trace(ep.trajectory), "events": events_text(ep.own_events, RUN_SETTINGS),
            "outcome": outcome(res), "sim_time_s": res.sim_time_s, "observed": observed,
            "probe_points": int(probe_points), "calls": client.calls if client else None, "golden_got": got,
            "golden_outcome": {"success": bool(res.success), "success_step": res.success_step,
                               "steps": int(res.steps)}}


def probe_neutrality(make_episode: Callable[[int], Any], read_snapshot: Callable[[str], Any], record: Mapping,
                     arrays: Mapping, *, index: Optional[int] = None, profile: str = PROBE_PROFILE) -> dict:
    from . import golden
    gs = record["golden"]
    i = max(range(len(gs)), key=lambda k: gs[k]["expected_steps"]) if index is None else index
    g = gs[i]
    live = continue_locally(make_episode(g["task_id"]), read_snapshot(g["snapshot_id"]), g["snapshot_id"],
                            profile=profile, measure=True)
    plain = continue_locally(make_episode(g["task_id"]), read_snapshot(g["snapshot_id"]), g["snapshot_id"],
                             profile="none", measure=False)
    problems = [p for p in (
        first_difference(live["trace"], plain["trace"], "executed steps"),
        first_difference(live["events"], plain["events"], "events"),
        None if live["outcome"] == plain["outcome"] else f"outcomes {live['outcome']} vs {plain['outcome']}",
        golden.compare(i, g, arrays, live["golden_got"], live["golden_outcome"]),
        golden.compare(i, g, arrays, plain["golden_got"], plain["golden_outcome"])) if p]
    calls = live["calls"]
    if not live["observed"] or not live["probe_points"] or not calls["sample"]:
        problems.append(f"the live run measured too little to show anything: observed {live['observed']}, "
                        f"probe points {live['probe_points']}, calls {calls}")
    return {"passed": not problems, "problems": problems, "snapshot_id": g["snapshot_id"], "profile": profile,
            "steps": len(plain["trace"]), "events": len(plain["events"]), "observed": live["observed"],
            "probe_points": live["probe_points"], "measure_calls": calls, "outcome": plain["outcome"],
            "sim_time_s": {"live": live["sim_time_s"], "none": plain["sim_time_s"]}}


def branch_repeat(make_episode: Callable[[int], Any], read_snapshot: Callable[[str], Any], record: Mapping, *,
                  index: Optional[int] = None, branch: str = BRANCH, pi05_calls: int = 1,
                  seed_indices: Sequence[int] = (1, 2, 3), repeats: int = 2) -> dict:
    """The first seed index whose branch executes π0.5 steps (a simulated network failure can abandon the call),
    run `repeats` times from fresh restores."""
    gs = record["golden"]
    i = min(range(len(gs)), key=lambda k: gs[k]["expected_steps"]) if index is None else index
    g = gs[i]

    def one(seed_index):
        br = make_episode(g["task_id"]).run_branch(read_snapshot(g["snapshot_id"]), branch=branch,
                                                   pi05_calls=pi05_calls, seed_index=seed_index,
                                                   snapshot_id=g["snapshot_id"])
        return {"trace": trace(br.trajectory), "events": events_text(br.events),
                "result": json.dumps({**outcome(br.result), "sim_time_s": br.result.sim_time_s}, allow_nan=True),
                "pi05_steps": sum(r["driver"] == "pi05" for r in br.trajectory), "outcome": outcome(br.result)}
    tried, runs = [], []
    for s in seed_indices:
        first = one(s)
        tried.append({"seed_index": s, "pi05_steps": first["pi05_steps"], "outcome": first["outcome"]})
        if first["pi05_steps"]:
            runs = [first] + [one(s) for _ in range(repeats - 1)]
            break
    if not runs:
        return {"passed": False, "problems": ["no seed index executed a π0.5 step: nothing shown"],
                "snapshot_id": g["snapshot_id"], "tried": tried}
    problems = [p for r in runs[1:] for p in (
        first_difference(runs[0]["trace"], r["trace"], "executed steps"),
        first_difference(runs[0]["events"], r["events"], "events"),
        None if runs[0]["result"] == r["result"] else f"results {runs[0]['result']} vs {r['result']}") if p]
    return {"passed": not problems, "problems": problems, "snapshot_id": g["snapshot_id"], "branch": branch,
            "seed_index": tried[-1]["seed_index"], "repeats": len(runs), "pi05_steps": runs[0]["pi05_steps"],
            "steps": len(runs[0]["trace"]), "events": len(runs[0]["events"]), "outcome": runs[0]["outcome"],
            "tried": tried}


# ---------------------------------------------------------------- re-runs against the dataset

def _bits(v):
    if isinstance(v, float):
        return struct.pack("<d", v)
    if isinstance(v, (list, tuple)):
        return tuple(_bits(x) for x in v)
    return v


def table_difference(a: bytes, b: bytes) -> Optional[str]:
    """None if two Parquet files hold the same values bit for bit, else the first differing column and row."""
    import pyarrow.parquet as pq
    ta, tb = pq.read_table(io.BytesIO(a)), pq.read_table(io.BytesIO(b))
    if ta.schema != tb.schema:
        return "schemas differ"
    if ta.num_rows != tb.num_rows:
        return f"{ta.num_rows} rows vs {tb.num_rows}"
    for name in ta.column_names:
        for r, (x, y) in enumerate(zip(ta[name].to_pylist(), tb[name].to_pylist())):
            if _bits(x) != _bits(y):
                return f"column {name} differs first at row {r}"
    return None


def _npz(data: bytes) -> dict:
    import numpy as np
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def npz_difference(a: bytes, b: bytes, *, snapshot: bool) -> tuple[Optional[str], list[str]]:
    """(first difference, header keys that differ). A snapshot's `__meta__` is compared without its provenance
    header, whose ops digest legitimately differs under the waiver."""
    za, zb = _npz(a), _npz(b)
    if sorted(za) != sorted(zb):
        return f"members {sorted(set(za) ^ set(zb))} differ", []
    header_keys: list[str] = []
    for k in sorted(za):
        x, y = za[k], zb[k]
        if snapshot and k == "__meta__":
            ma, mb = json.loads(x.tobytes().decode()), json.loads(y.tobytes().decode())
            ha, hb = ma.pop("header", {}), mb.pop("header", {})
            header_keys = sorted(h for h in set(ha) | set(hb) if ha.get(h) != hb.get(h))
            if ma != mb:
                return "snapshot meta differs outside its header", header_keys
        elif x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            return f"member {k} differs", header_keys
    return None, header_keys


def compare_job(orig, smoke, phase: str, jid: str, *, boot: Optional[str], calibration_hash: Optional[str]) -> dict:
    problems: list[str] = []
    mo_b, ms_b = orig.get_bytes(result_key(phase, jid)), smoke.get_bytes(result_key(phase, jid))
    if mo_b is None or ms_b is None:
        return {"job_id": jid, "passed": False,
                "problems": [f"no {'original' if mo_b is None else 're-run'} result marker"]}
    mo, ms = json.loads(mo_b), json.loads(ms_b)
    meta, ometa = ms.get("meta", {}), mo.get("meta", {})
    if ms.get("job") != mo.get("job"):
        problems.append("the job differs")
    if ms["result"].get("summary") != mo["result"].get("summary"):
        problems.append(f"summary {ms['result'].get('summary')} vs {mo['result'].get('summary')}")
    waiver = meta.get("ops_waiver")
    if not isinstance(waiver, dict) or waiver.get("freeze_ops_digest") != ometa.get("provenance", {}).get("ops_digest"):
        problems.append(f"marker's ops_waiver is {waiver!r}, not against the original's ops digest")
    if not boot or meta.get("boot_id") != boot:
        problems.append(f"marker's boot_id {meta.get('boot_id')!r} is not this boot's {boot!r}")
    gp = meta.get("golden_pass") or {}
    if not (gp.get("passed") and gp.get("boot_id") == boot and gp.get("calibration_hash") == calibration_hash):
        problems.append("marker's golden_pass is not this boot's pass on the frozen calibration")
    if not gp.get("image_check"):
        problems.append("marker's golden_pass records no image check")
    prov_o, prov_s = ometa.get("provenance", {}), meta.get("provenance", {})
    prov_diff = sorted(k for k in set(prov_o) | set(prov_s) if prov_o.get(k) != prov_s.get(k))
    arts_o = {a["key"]: a for a in mo["result"]["artifacts"]}
    arts_s = {a["key"]: a for a in ms["result"]["artifacts"]}
    if set(arts_o) != set(arts_s):
        problems.append(f"artifact keys differ: {sorted(set(arts_o) ^ set(arts_s))[:5]}")
    same_bytes, header_keys = 0, set()
    for key in sorted(set(arts_o) & set(arts_s)):
        a, b = orig.get_bytes(key), smoke.get_bytes(key)       # the objects themselves, not the markers' word
        if a is None or b is None:
            problems.append(f"{key}: missing from the bucket")
            continue
        ha, hb = hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest()
        if ha != arts_o[key]["sha256"] or hb != arts_s[key]["sha256"]:
            problems.append(f"{key}: the bucket's object is not the one its marker lists")
        if ha == hb:
            same_bytes += 1
            continue
        if key.startswith("records/"):
            d = table_difference(a, b)
        else:
            d, hk = npz_difference(a, b, snapshot=key.startswith("snapshots/"))
            header_keys.update(hk)
        if d:
            problems.append(f"{key}: {d}")
    return {"job_id": jid, "subject": mo["job"]["subject"], "passed": not problems, "problems": problems,
            "artifacts": len(arts_o), "identical_bytes": same_bytes, "provenance_differs_in": prov_diff,
            "snapshot_header_differs_in": sorted(header_keys), "summary": ms["result"].get("summary"),
            "worker": {"original": ometa.get("worker_id"), "rerun": meta.get("worker_id")},
            "ops_waiver": waiver, "boot_id": meta.get("boot_id")}


def compare(orig, smoke, subjects: Sequence[str], *, boot: Optional[str], calibration_hash: Optional[str],
            phase: str = "base") -> dict:
    jobs = [compare_job(orig, smoke, phase, job_id(s, "episode", 1), boot=boot, calibration_hash=calibration_hash)
            for s in subjects]
    return {"passed": all(j["passed"] for j in jobs), "jobs": jobs}


# ---------------------------------------------------------------- on the VM

def run_on_servers(ops_cfg: Mapping[str, Any], golden_json: Path, what: str, *, clients=None, make_sim=None,
                   server_wait_s: float = 600.0) -> dict:
    """`probe` or `branch` through the production servers (checked as for a worker), snapshots from the bucket."""
    import os
    import sys
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "cgl")
    from vlar_harness import snapshot as snap
    from vlar_harness.config import load_config
    from vlar_harness.dataset import snapshot_key

    from . import golden, vm_worker
    from .storage import open_storage
    cfg = load_config()
    if not cfg.frozen:
        raise SmokeError("the smoke checks are for the frozen harness")
    image = vm_worker.image_check(cfg)
    if clients is None:
        from vlar_harness.servers.clients import PI05Client, XVLAClient
        srv = cfg["servers"]
        clients = {"xvla": XVLAClient(srv["xvla"]["host"], int(srv["xvla"]["port"])),
                   "pi05": PI05Client(srv["pi05"]["host"], int(srv["pi05"]["port"]))}
    vm_worker.check_servers(cfg, clients, wait_s=server_wait_s)
    storage = open_storage(ops_cfg["storage_url"])
    record, arrays = golden.load(golden_json)
    episodes = vm_worker.ProductionEpisodes(cfg, clients["xvla"], clients["pi05"], make_sim=make_sim)

    def read(sid):
        data = storage.get_bytes(snapshot_key(sid))
        if data is None:
            raise SmokeError(f"snapshot {sid} is not in the bucket")
        return snap.from_bytes(data)
    try:
        rep = probe_neutrality(episodes, read, record, arrays) if what == "probe" else branch_repeat(episodes, read,
                                                                                                    record)
    finally:
        episodes.close()
    return {**rep, "boot_id": golden.boot_id(), "calibration_hash": cfg.calibration_hash, "image_check": image}
