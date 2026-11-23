"""Phase 0 step 2 on the L4 (SPEC §11 step 2; amendments 14, 24, 25, 34, 35): `vlar-ops step2 <command>`.

Each command writes one JSON file; `updates` turns them into the calibration-file update that
`python -m vlar_harness.calfile write` applies (the write-once group: π0.5's latency, the upload size, the batch
geometry; and the rest: X-VLA's latencies, the worker count, the gripper thresholds).

- `capture --out DIR`: real requests for the bench, from the simulator (sim image): X-VLA payloads and π0.5 raw
  observations at the first decision point of tasks 0-9.
- `render --task T --seconds S --out F`, `load --obs DIR --size B --seconds S --out F`, `memory --smi F --size B
  --workers W [--error E] --out F`: the joint memory measurement at a candidate size B (step 200 starts both
  production servers at that geometry and 2 x B renderers, runs `load` while nvidia-smi samples the card, and
  `memory` turns the samples into memory-B.json; approved by the user 2026-11-23).
- `choose --xvla B1.json --pi05 B2.json --memory-dir DIR --out geometry.json`: the geometry rule (`choose_geometry`)
  over the two benches (`python -m vlar_harness.servers.bench`) and the joint measurements. Exit 75 (the runner
  keeps the step queued and stops the VM) if fixed, deterministic batching is too slow, with the numbers in the
  file: amendment 24 says stop and ask.
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
HEADROOM_MIB = 2048            # left free on the card at the joint peak (at least 2 GB; approved 2026-11-23)
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


def workers_for(size: int) -> int:
    """The planned number of simulator workers at batch size `size`: 2 x B, at least 2, at most MAX_WORKERS."""
    return min(MAX_WORKERS, max(2, 2 * int(size)))


def choose_geometry(xvla_bench: Mapping, pi05_bench: Mapping, *, k: int, memory: Mapping) -> dict[str, Any]:
    """The batch geometry and worker count from the two benches (sizes -> {det, nondet} timings) and the joint memory
    measurements (sizes -> `memory_result`).

    Feasible: both models ran at that size in the bench, and the joint measurement at that size (both production
    servers at that geometry, the planned `workers_for(size)` workers rendering, full batches to both servers) left at
    least HEADROOM_MIB of the card free at its peak (nvidia-smi: every process on the GPU, rendering included). Chosen:
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
        m = memory.get(str(s))
        if m is None or m.get("error") or int(m.get("workers", -1)) != workers_for(s):
            why = ("no joint memory measurement" if m is None else m.get("error")
                   or f"joint measurement with {m.get('workers')} workers rendering, not {workers_for(s)}")
            rows[s] = {"feasible": False, "why": f"size {s}: {why}"}
            continue
        free = int(m["total_mib"]) - int(m["peak_used_mib"])
        det, nondet = _cost(xb, pb, "det") / s, _cost(xb, pb, "nondet") / s
        rows[s] = {"feasible": free >= HEADROOM_MIB, "joint_peak_mib": int(m["peak_used_mib"]),
                   "total_mib": int(m["total_mib"]), "free_mib": free, "workers_rendering": int(m["workers"]),
                   "bench_peak_memory_gb": xb["det"]["peak_memory_gb"] + pb["det"]["peak_memory_gb"],
                   "det_s_per_request": det, "nondet_s_per_request": nondet, "determinism_cost": det / nondet,
                   "why": None if free >= HEADROOM_MIB else f"{free} MiB free at the joint peak < {HEADROOM_MIB} MiB"}
    feasible = {s: r for s, r in rows.items() if r["feasible"]}
    if not feasible:
        raise TooSlow(f"no feasible batch size: {rows}")
    best = min(r["det_s_per_request"] for r in feasible.values())
    chosen = min(s for s, r in feasible.items() if r["det_s_per_request"] <= NEAR_BEST * best)
    r = rows[chosen]
    full = {s: _cost(xvla_bench[str(s)], pi05_bench[str(s)], "det") for s in feasible}
    padding = {n: full[chosen] / full[n] for n in sorted(full) if n < chosen}   # cost of a batch of n padded to B
    out = {"max_batch": chosen, "xvla_sample_rows": chosen * k, "workers": workers_for(chosen),
           "rule": {"headroom_mib": HEADROOM_MIB, "memory": "joint peak from nvidia-smi: both servers at the size, "
                    "2 x B workers rendering (at most MAX_WORKERS), full batches to both", "near_best": NEAR_BEST,
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


def memory_peak(smi_text: str) -> dict[str, int]:
    """Samples of `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits` (MiB, one GPU):
    the peak used and the card's total."""
    used, total = [], set()
    for line in smi_text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and all(x.isdigit() for x in parts):
            used.append(int(parts[0]))
            total.add(int(parts[1]))
    if not used:
        raise ValueError("no nvidia-smi memory samples")
    if len(total) != 1:
        raise ValueError(f"more than one GPU or a changing total in the samples: {sorted(total)}")
    return {"peak_used_mib": max(used), "total_mib": total.pop(), "samples": len(used)}


def memory_result(smi_text: Optional[str], *, size: int, workers: int, error: Optional[str] = None) -> dict:
    """memory-B.json: the joint peak at size B with `workers` renderers, or the error that prevented it (a size whose
    servers or renderers did not all run through the measurement is infeasible)."""
    out: dict[str, Any] = {"size": int(size), "workers": int(workers), "headroom_mib": HEADROOM_MIB}
    if error:
        out["error"] = error
    try:
        out.update(memory_peak(smi_text or ""))
    except ValueError as e:
        out.setdefault("error", str(e))
    if "error" not in out:
        out["free_mib"] = out["total_mib"] - out["peak_used_mib"]
    return out


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


def render(*, task: int, seconds: float) -> dict:
    """One planned worker's rendering for the joint memory measurement: the harness's simulator for `task`, every
    camera rendered at every step (`render="all"`, more than a worker's `used`), a still arm stepped for `seconds`.
    Prints "rendering" once the first frame is out (step 200 waits for it)."""
    import numpy as np
    from vlar_harness import seeding
    from vlar_harness.config import load_config
    cfg = load_config(render="all")
    sim = _sim(cfg, task)
    n = 0
    try:
        sim.setup_episode(0, env_seed=seeding.scene_seed(0, int(cfg.option("scene_seed"))))
        sim.set_control_mode("delta")
        t0 = time.time()
        while n == 0 or time.time() - t0 < seconds:
            sim.step(np.zeros(7, dtype=np.float64))
            sim.require_images()
            n += 1
            if n == 1:
                print("rendering", flush=True)
    finally:
        sim.close()
    return {"task": task, "steps": n, "seconds": round(time.time() - t0, 1)}


def load(obs: Path, *, size: int, k: int, seconds: float, host: str = "127.0.0.1") -> dict:
    """Full batches to both production servers for `seconds`: `size` client threads per server, X-VLA's looping
    /act, /sample (K seeds) and /features, π0.5's looping /act, so every batch fills (each one runs at the fixed
    geometry anyway). Raises on the first failed call: a server that ran out of memory fails the measurement."""
    import threading

    import json_numpy
    from vlar_harness.servers.clients import PI05Client, XVLAClient
    xi, pi = XVLAClient(host, 8777).info(), PI05Client(host, 8778).info()
    if xi["batching"]["max_batch"] != size or xi["batching"]["sample_rows"] != size * k \
            or pi["batching"]["max_batch"] != size:
        raise RuntimeError(f"servers not at size {size}: {xi['batching']}, {pi['batching']}")
    xp = json.loads((obs / "xvla_payloads.json").read_text())
    pp = json_numpy.loads((obs / "pi05_requests.json").read_text())
    stop_at = time.time() + seconds
    calls = {"xvla": 0, "pi05": 0}
    errors: list[str] = []
    lock = threading.Lock()

    def xvla(j):
        c, i = XVLAClient(host, 8777), j
        while time.time() < stop_at and not errors:
            p = xp[i % len(xp)]
            c.act(p, seed=i)
            c.sample(p, [k * i + m for m in range(k)])
            c.features(p)
            i += size
            with lock:
                calls["xvla"] += 3

    def pi05(j):
        c, i = PI05Client(host, 8778), j
        while time.time() < stop_at and not errors:
            r = pp[i % len(pp)]
            c.act(r["obs"], r["task"], seed=i)
            i += size
            with lock:
                calls["pi05"] += 1

    def guard(fn, j):
        try:
            fn(j)
        except Exception as e:      # noqa: BLE001 - recorded, then raised below
            errors.append(f"{fn.__name__}: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=guard, args=(fn, j)) for fn in (xvla, pi05) for j in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(f"load at size {size} failed: {errors[0]}")
    return {"size": size, "seconds": seconds, "calls": calls, "threads_per_server": size,
            "servers": {"xvla": xi["batching"], "pi05": pi["batching"]}}


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
        memory = {str(json.loads(f.read_text())["size"]): json.loads(f.read_text())
                  for f in sorted(Path(a.memory_dir).glob("memory-*.json"))}
        res = choose_geometry(json.loads(Path(a.xvla).read_text())["sizes"],
                              json.loads(Path(a.pi05).read_text())["sizes"], k=k, memory=memory)
    elif cmd == "render":
        res = render(task=int(a.task), seconds=float(a.seconds))
    elif cmd == "load":
        res = load(Path(a.obs), size=int(a.size), k=k, seconds=float(a.seconds))
    elif cmd == "memory":
        smi = Path(a.smi).read_text() if a.smi and Path(a.smi).exists() else None
        res = memory_result(smi, size=int(a.size), workers=int(a.workers), error=a.error)
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
