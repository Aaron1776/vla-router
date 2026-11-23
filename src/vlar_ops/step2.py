"""Phase 0 step 2 on the L4 (SPEC §11 step 2; amendments 14, 24, 25, 34, 35): `vlar-ops step2 <command>`.

Each command writes one JSON file; `updates` turns them into the calibration-file update that
`python -m vlar_harness.calfile write` applies (the write-once group: π0.5's latency, the upload size, the batch
geometry; and the rest: X-VLA's latencies, the worker count, the gripper thresholds).

- `capture --out DIR`: real requests for the bench, from the simulator (sim image): X-VLA payloads and π0.5 raw
  observations at the first decision point of tasks 0-9.
- `choose --xvla B1.json --pi05 B2.json --out geometry.json`: the geometry rule (`choose_geometry`) over the two
  benches (`python -m vlar_harness.servers.bench`). Exit 75 (the runner keeps the step queued and stops the VM)
  if fixed, deterministic batching is too slow, with the numbers in the file: amendment 24 says stop and ask.
- `latency --obs DIR --out latency.json`: the clock charges, unpadded (amendment 25): servers started at
  `--max-batch 1` (X-VLA `--sample-rows K`); median client wall time over localhost of X-VLA `/act`, `/features`,
  `/sample` with K seeds, and π0.5 `/act`, all seeded as in collection.
- `upload --out upload.json`: amendment 14: π0.5-alone decision points (tasks 0-9, first 16 calls each, no network
  model), each request's two images sent to `/upload_bytes`; the median over requests of their summed JPEG sizes.
- `gripper --out gripper.json`: the finger-width thresholds (`gripper_thresholds`).
- `updates --dir DIR --out updates.json`: the calibration update from the four files.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

NEEDS_USER_EXIT = 75
MEMORY_BUDGET_GB = 20.0        # both models' peak at the chosen size; the rest of the L4's 24 GB for EGL and CUDA
NEAR_BEST = 1.10               # the smallest size within 10% of the best GPU cost per request
DETERMINISM_LIMIT = 1.25       # deterministic mode may cost at most 25% at the chosen size, else stop and ask
MAX_WORKERS = 12               # 16 vCPUs, minus the servers' request processing
GRIPPER_MARGIN = 0.10          # thresholds 10% of the open-shut range inside each end
UPLOAD_CALLS_PER_TASK = 16


class TooSlow(RuntimeError):
    pass


# ---------------------------------------------------------------- the geometry rule (pure, unit-tested)

def _cost(xvla: Mapping, pi05: Mapping, mode: str) -> float:
    """GPU seconds per request of each kind at full batches: (t_act + t_sample + t_features + t_pi05) / B."""
    x, p = xvla[mode], pi05[mode]
    return sum(x[k]["median_s"] for k in ("act", "sample", "features")) + p["act"]["median_s"]


def choose_geometry(xvla_bench: Mapping, pi05_bench: Mapping, *, k: int) -> dict[str, Any]:
    """The batch geometry and worker count from the two benches (sizes -> {det, nondet} timings).

    Feasible: both models ran at that size and their deterministic peaks sum to at most MEMORY_BUDGET_GB. Chosen:
    the smallest feasible size whose deterministic cost per request is within NEAR_BEST of the best (smaller
    batches waste less on partial batches). Stop (TooSlow) if deterministic mode costs more than DETERMINISM_LIMIT
    times the non-deterministic run at that size. Padding cost is reported, not gated: at the chosen size a batch
    of n < B requests costs t(B) instead of t(n); the worker count (2 x B, at most MAX_WORKERS) keeps batches full.
    """
    rows = {}
    for s in sorted({int(b) for b in xvla_bench} & {int(b) for b in pi05_bench}):
        xb, pb = xvla_bench[str(s)], pi05_bench[str(s)]
        if "error" in xb or "error" in pb:
            rows[s] = {"feasible": False, "why": xb.get("error") or pb.get("error")}
            continue
        mem = xb["det"]["peak_memory_gb"] + pb["det"]["peak_memory_gb"]
        det, nondet = _cost(xb, pb, "det") / s, _cost(xb, pb, "nondet") / s
        rows[s] = {"feasible": mem <= MEMORY_BUDGET_GB, "peak_memory_gb": mem, "det_s_per_request": det,
                   "nondet_s_per_request": nondet, "determinism_cost": det / nondet,
                   "why": None if mem <= MEMORY_BUDGET_GB else f"{mem:.1f} GB > {MEMORY_BUDGET_GB} GB"}
    feasible = {s: r for s, r in rows.items() if r["feasible"]}
    if not feasible:
        raise TooSlow(f"no feasible batch size: {rows}")
    best = min(r["det_s_per_request"] for r in feasible.values())
    chosen = min(s for s, r in feasible.items() if r["det_s_per_request"] <= NEAR_BEST * best)
    r = rows[chosen]
    full = {s: _cost(xvla_bench[str(s)], pi05_bench[str(s)], "det") for s in feasible}
    padding = {n: full[chosen] / full[n] for n in sorted(full) if n < chosen}   # cost of a batch of n padded to B
    out = {"max_batch": chosen, "xvla_sample_rows": chosen * k, "workers": min(MAX_WORKERS, max(2, 2 * chosen)),
           "rule": {"memory_budget_gb": MEMORY_BUDGET_GB, "near_best": NEAR_BEST,
                    "determinism_limit": DETERMINISM_LIMIT, "max_workers": MAX_WORKERS},
           "sizes": {str(s): v for s, v in rows.items()}, "padding_cost_by_fill": {str(n): v for n, v in padding.items()}}
    if r["determinism_cost"] > DETERMINISM_LIMIT:
        out["too_slow"] = (f"deterministic mode costs {r['determinism_cost']:.2f}x at batch size {chosen} "
                           f"(limit {DETERMINISM_LIMIT}x)")
    return out


def gripper_thresholds(open_widths: Sequence[float], shut_widths: Sequence[float]) -> dict[str, float]:
    """Shut: at most the widest fully-closed width plus GRIPPER_MARGIN of the range; open: at least the narrowest
    fully-open width minus GRIPPER_MARGIN of the range."""
    lo, hi = max(shut_widths), min(open_widths)
    if not lo < hi:
        raise ValueError(f"closed widths {shut_widths} overlap open widths {open_widths}")
    span = hi - lo
    return {"shut_width_m": lo + GRIPPER_MARGIN * span, "open_width_m": hi - GRIPPER_MARGIN * span}


def calibration_updates(geometry: Mapping, latency: Mapping, upload: Mapping, gripper: Mapping,
                        choices: Sequence[str]) -> dict:
    lat = latency["median_s"]
    return {
        "latency": {"status": "measured", "source": latency["source"], "xvla_act_s": lat["xvla_act"],
                    "xvla_features_s": lat["xvla_features"], "xvla_sample_k_s": lat["xvla_sample_k"],
                    "pi05_infer_s": lat["pi05_act"]},
        "servers": {"workers": int(geometry["workers"]),
                    "batching": {"status": "measured", "max_batch": int(geometry["max_batch"]),
                                 "xvla_sample_rows": int(geometry["xvla_sample_rows"])}},
        "measurement": {"gripper_thresholds": {"status": "calibrated", **{k: float(gripper[k]) for k in
                                                                         ("shut_width_m", "open_width_m")}}},
        "options": {"network": {"choices": {c: {"upload_image_bytes": int(upload["upload_image_bytes"]),
                                                "upload_image_bytes_status": "measured"} for c in choices}}},
    }


# ---------------------------------------------------------------- simulator-side commands (sim image)

def _sim(cfg, task_id: int):
    from vlar_harness.sim.libero_sim import LiberoSim, SimSpec
    return LiberoSim(task_id, SimSpec.from_config(cfg))


def capture(out: Path, tasks: Sequence[int] = range(10)) -> dict:
    import json_numpy
    from vlar_harness import seeding
    from vlar_harness.adapters.pi05 import PI05Adapter
    from vlar_harness.adapters.xvla import XVLAAdapter
    from vlar_harness.config import load_config
    cfg = load_config()
    xvla, pi05 = [], []
    for t in tasks:
        sim = _sim(cfg, t)
        try:
            sim.setup_episode(0, env_seed=seeding.scene_seed(0, int(cfg.option("scene_seed"))))
            sim.require_images()
            xvla.append(XVLAAdapter(lambda *a, **k: None).measurement_payload(sim, sim.task_description))
            pi05.append({"obs": PI05Adapter(sim).raw_observation(), "task": sim.task_description})
        finally:
            sim.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "xvla_payloads.json").write_text(json.dumps(xvla))
    (out / "pi05_requests.json").write_text(json_numpy.dumps(pi05))
    return {"tasks": list(tasks), "xvla": len(xvla), "pi05": len(pi05)}


def latency(obs: Path, *, k: int, calls: int = 30, warmup: int = 5, host: str = "127.0.0.1") -> dict:
    """Median client wall time per call against servers at --max-batch 1 (X-VLA --sample-rows K)."""
    import json_numpy
    from vlar_harness.servers.clients import PI05Client, XVLAClient
    xc, pc = XVLAClient(host, 8777), PI05Client(host, 8778)
    xi, pi = xc.info(), pc.info()
    if xi["batching"]["max_batch"] != 1 or xi["batching"]["sample_rows"] != k or pi["batching"]["max_batch"] != 1:
        raise RuntimeError(f"servers must run unpadded (max_batch 1, X-VLA sample_rows {k}): {xi['batching']}, "
                           f"{pi['batching']}")
    xp = json.loads((obs / "xvla_payloads.json").read_text())
    pp = json_numpy.loads((obs / "pi05_requests.json").read_text())
    fns = {"xvla_act": lambda i: xc.act(xp[i % len(xp)], seed=i),
           "xvla_features": lambda i: xc.features(xp[i % len(xp)]),
           "xvla_sample_k": lambda i: xc.sample(xp[i % len(xp)], [k * i + j for j in range(k)]),
           "pi05_act": lambda i: pc.act(pp[i % len(pp)]["obs"], pp[i % len(pp)]["task"], seed=i)}
    res = {}
    for name, fn in fns.items():
        for i in range(warmup):
            fn(i)
        ts = []
        for i in range(calls):
            t0 = time.perf_counter()
            fn(warmup + i)
            ts.append(time.perf_counter() - t0)
        res[name] = ts
    med = {n: statistics.median(ts) for n, ts in res.items()}
    if med["xvla_sample_k"] < med["xvla_act"]:
        raise RuntimeError(f"/sample K faster than /act ({med}): the clock's sample_k_less_act charge would be negative")
    return {"median_s": med, "calls": calls, "warmup": warmup, "samples_s": res, "servers": {"xvla": xi, "pi05": pi},
            "source": f"Phase 0 step 2 on the L4: servers at max_batch 1 (X-VLA sample_rows {k}), median of {calls} "
                      f"seeded calls after {warmup} warm-ups, client wall time over localhost HTTP"}


class _Enough(Exception):
    pass


def upload(*, tasks: Sequence[int] = range(10), calls_per_task: int = UPLOAD_CALLS_PER_TASK,
           host: str = "127.0.0.1", quality: int = 90) -> dict:
    from vlar_harness import network
    from vlar_harness.adapters.pi05 import PI05Adapter
    from vlar_harness.adapters.xvla import XVLAAdapter
    from vlar_harness.config import load_config
    from vlar_harness.episode import Episode
    from vlar_harness.servers.clients import PI05Client
    cfg = load_config()
    pc = PI05Client(host, 8778)
    per_request: list[list[int]] = []
    for t in tasks:
        sim = _sim(cfg, t)
        n = [0]

        def call(raw_obs, task, seed):
            sizes = pc.upload_bytes(raw_obs, task, quality)
            per_request.append([int(x) for x in sizes["jpeg_bytes"]])
            n[0] += 1
            if n[0] >= calls_per_task:
                raise _Enough
            return pc.act(raw_obs, task, seed)["actions"]
        try:
            ep = Episode(sim, XVLAAdapter(lambda *a, **k: None), PI05Adapter(sim), call, cfg,
                         remote=network.DirectLink())
            ep.start(0, rollout_seed=9_000_000 + t, mode="pi05_only")
            ep.run_episode()
        except _Enough:
            pass
        finally:
            sim.close()
    return {"upload_image_bytes": network.median_upload_image_bytes(per_request), "requests": len(per_request),
            "quality": quality, "per_request": per_request}


def gripper(*, tasks: Sequence[int] = (0, 3, 6), steps: int = 30) -> dict:
    """In π0.5's delta mode with no arm motion: hold the gripper open, then closed on nothing, `steps` each, at the
    first initial state of a few tasks; the settled finger widths (qpos[0] - qpos[1]) give the thresholds."""
    import numpy as np
    from vlar_harness import seeding
    from vlar_harness.config import load_config
    cfg = load_config(render="used")
    opens, shuts = [], []
    for t in tasks:
        sim = _sim(cfg, t)
        try:
            sim.setup_episode(0, env_seed=seeding.scene_seed(0, int(cfg.option("scene_seed"))))
            sim.set_control_mode("delta")
            for g, sink in ((-1.0, opens), (1.0, shuts)):
                for _ in range(steps):
                    sim.step(np.array([0, 0, 0, 0, 0, 0, g], dtype=np.float64))
                q = np.asarray(sim.obs["robot0_gripper_qpos"], dtype=np.float64)
                sink.append(float(q[0] - q[1]))
        finally:
            sim.close()
    return {**gripper_thresholds(opens, shuts), "open_widths_m": opens, "shut_widths_m": shuts,
            "tasks": list(tasks), "steps": steps, "margin": GRIPPER_MARGIN}


def run(cmd: str, a) -> int:
    from vlar_harness.config import load_config
    out = Path(a.out)
    k = int(load_config()["measurement"]["K"])
    if cmd == "capture":
        res = capture(out)
    elif cmd == "choose":
        res = choose_geometry(json.loads(Path(a.xvla).read_text())["sizes"],
                              json.loads(Path(a.pi05).read_text())["sizes"], k=k)
    elif cmd == "latency":
        res = latency(Path(a.obs), k=k)
    elif cmd == "upload":
        res = upload()
    elif cmd == "gripper":
        res = gripper()
    elif cmd == "updates":
        d = Path(a.dir)
        load = lambda n: json.loads((d / f"{n}.json").read_text())   # noqa: E731
        res = calibration_updates(load("geometry"), load("latency"), load("upload"), load("gripper"),
                                  _network_choices())
    else:
        raise ValueError(cmd)
    if cmd != "capture":
        out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k_: v for k_, v in res.items() if k_ not in ("samples_s", "per_request")}, indent=1))
    if cmd == "choose" and res.get("too_slow"):
        print(f"STOP: {res['too_slow']} (amendment 24: bring the numbers to the user)")
        return NEEDS_USER_EXIT
    return 0


def _network_choices() -> list[str]:
    from vlar_harness import config as C
    return list(C._read_yaml(C.CONFIG_DIR / f"{C.DEFAULT_CONFIG}.yaml")["options"]["network"]["choices"])
