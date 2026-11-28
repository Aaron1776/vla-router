"""The VM worker entry point (docs/SPEC.md §12, §15): `vlar-ops worker`, one simulator worker per process.

Start several on the VM (ops/vm/vlar-worker@.service); the model servers batch across them. Each process:
1. loads the harness config with the named options given (`--option name=choice`), and refuses a frozen harness
   whose code or config differ from its freeze record (provenance.verify_freeze), except ops code that differs
   only where it cannot change a record (`vlar_ops.waiver`, amendment 49). A frozen harness must also have
   passed the golden-set self-check (§15, `vlar_ops.golden`) on this boot: `run_phase` runs it before any worker
   and writes the pass marker this checks. It also refuses to run without the calibration file
   (VLAR_CALIBRATION) and its complete write-once group (upload size, π0.5 latency, batch geometry: Phase 0 step 2,
   amendment 35), so no record is ever made under placeholder values that would change outcomes. A frozen harness
   runs only on the frozen images (`image_check`, amendment 54; the golden-set self-check runs it too);
2. checks the pinned reference files it runs: the authors' X-VLA client and LeRobot's LIBERO and π0.5 files;
3. waits for both model servers' /info and checks they run the configured fixed batch geometry with the
   deterministic settings (amendment 24): a server started otherwise (Phase 0 step 2's unpadded latency runs)
   would give results that depend on it;
4. builds the production episode factory: `XVLAClient` (the X-VLA transport and its measurement endpoints),
   `PI05Client`, the configured network model, one `LiberoSim` per task (built on first use, then reused);
5. runs `worker.run_worker` with the spend guard's stop flag (`<local_root>/STOP`): once the flag is up no job is
   claimed; a SIGTERM or Ctrl-C abandons the current job cleanly (lease released, no marker). A job the worker's
   options cannot run is given back and the worker stops (exit status 3).
Every result marker carries the provenance header (harness version, config hash and options, calibration hash,
source digest, ops digest, git commit, image digests), the worker id, p_escalate, the servers' settings and the boot id;
a frozen harness's markers also carry this boot's golden-set pass, and those made under the ops waiver its record
(`ops_waiver`), so the dataset manifest can check that nothing from different harness versions is mixed (§15). An
exploration job must record this worker's p_escalate in its spec (amendment 49); others are given back. Staged files
are deleted after each job.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .queue import JobQueue
from .spend import StopFlag
from .storage import open_storage
from .worker import Progress, run_worker

REFUSED_EXIT = 3
RECYCLE_EXIT = 4      # took its --max-jobs jobs: the supervisor starts it again while work is left (§0.6)
XVLA_CLIENT_FILE = "evaluation/libero/libero_client.py"


class StartupError(RuntimeError):
    """The worker must not start: a server, the harness or the queue is not as it must be."""


class ProductionEpisodes:
    """`make_episode(task_id)` for `collect.Runner`: a fresh `Episode` per job on a cached simulator of the task.

    `xvla` has `act(payload, seed)`, `sample(payload, seeds)` and `features(payload)` (servers.clients.XVLAClient);
    `pi05` has `act(raw_obs, task, seed)` returning {"actions": 50 x 7} (servers.clients.PI05Client).
    """

    def __init__(self, cfg, xvla, pi05, *, make_sim: Optional[Callable[[int], Any]] = None):
        self.cfg, self.xvla, self.pi05 = cfg, xvla, pi05
        self.state_mode = cfg.option("xvla_state_mode")
        self.client_sha256 = cfg["reference"]["xvla_repo_files"][XVLA_CLIENT_FILE]
        self._make_sim = make_sim or self._libero
        self.sims: dict[int, Any] = {}

    def _libero(self, task_id: int):
        from vlar_harness.sim.libero_sim import LiberoSim, SimSpec
        return LiberoSim(task_id, SimSpec.from_config(self.cfg))

    def sim(self, task_id: int):
        if task_id not in self.sims:
            self.sims[task_id] = self._make_sim(task_id)
        return self.sims[task_id]

    def pi05_call(self, raw_obs: dict, task: str, seed: Optional[int]):
        return self.pi05.act(raw_obs, task, seed)["actions"]

    def __call__(self, task_id: int):
        from vlar_harness.adapters.pi05 import PI05Adapter
        from vlar_harness.adapters.xvla import XVLAAdapter
        from vlar_harness.episode import Episode
        sim = self.sim(task_id)
        xvla = XVLAAdapter(self.xvla.act, state_mode=self.state_mode, client_sha256=self.client_sha256)
        return Episode(sim, xvla, PI05Adapter(sim), self.pi05_call, self.cfg, measure_client=self.xvla)

    def close(self) -> None:
        for s in self.sims.values():
            s.close()
        self.sims.clear()


def expected_servers(cfg) -> dict[str, dict[str, Any]]:
    """What each server's /info must report for data collection (amendment 24)."""
    b = cfg["servers"]["batching"]
    det = {"deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False}
    return {"xvla": {"model": "xvla", "batching.max_batch": int(b["max_batch"]),
                     "batching.sample_rows": int(b["xvla_sample_rows"]),
                     "batching.steps": int(cfg["models"]["xvla"]["flow_steps"]), **det},
            "pi05": {"model": "pi05", "batching.max_batch": int(b["max_batch"]), **det}}


def _lookup(info: Mapping, dotted: str) -> Any:
    cur: Any = info
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return "<missing>"
        cur = cur[part]
    return cur


def check_servers(cfg, clients: Mapping[str, Any], *, wait_s: float = 600.0, poll_s: float = 5.0,
                  sleep: Callable[[float], None] = time.sleep,
                  now: Callable[[], float] = time.monotonic) -> dict[str, dict]:
    """Wait up to `wait_s` for each server's /info, then check it against `expected_servers`. Returns the infos."""
    infos: dict[str, dict] = {}
    deadline = now() + wait_s
    for name, client in clients.items():
        while True:
            try:
                infos[name] = client.info()
                break
            except Exception as e:   # not up yet (connection refused, loading weights)
                if now() >= deadline:
                    raise StartupError(f"{name} server not answering /info after {wait_s:.0f} s: {e}") from e
                sleep(poll_s)
    problems = [f"{name} {key} is {_lookup(infos[name], key)!r}, must be {want!r}"
                for name, keys in expected_servers(cfg).items() for key, want in keys.items()
                if _lookup(infos[name], key) != want]
    if problems:
        raise StartupError("model servers are not set up for data collection: " + "; ".join(problems))
    return infos


def parse_options(pairs: Sequence[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        name, sep, choice = p.partition("=")
        if not sep or not name or not choice:
            raise StartupError(f"--option takes NAME=CHOICE, not {p!r}")
        out[name] = choice
    return out


def image_check(cfg) -> Optional[dict]:
    """A frozen harness runs only on the frozen images (amendment 54): the running images' digests
    (`/etc/vlar/image-digests.json`) must equal the freeze record's, else its dated supplement's. None if not frozen."""
    if not cfg.frozen:
        return None
    from vlar_harness import provenance

    from . import freezesupp as FS
    try:
        return FS.check_running(provenance.freeze_path(cfg), provenance.image_digests())
    except (FS.SupplementError, provenance.FreezeError, OSError, ValueError) as e:
        raise StartupError(f"frozen harness, images not the frozen ones: {e}") from e


def golden_check(ops_cfg: Mapping[str, Any], golden_json: Path, *, marker: Optional[Path] = None,
                 clients: Optional[Mapping[str, Any]] = None, make_sim: Optional[Callable[[int], Any]] = None,
                 server_wait_s: float = 600.0) -> dict:
    """`vlar-ops golden check`: replay the golden set through the production servers (checked as for a worker),
    snapshots read from the bucket; writes the pass (or fail) marker for this boot."""
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "cgl")
    from vlar_harness import snapshot as snap
    from vlar_harness.config import load_config
    from vlar_harness.dataset import snapshot_key

    from . import golden
    cfg = load_config()
    images = image_check(cfg)
    if clients is None:
        from vlar_harness.servers.clients import PI05Client, XVLAClient
        srv = cfg["servers"]
        clients = {"xvla": XVLAClient(srv["xvla"]["host"], int(srv["xvla"]["port"])),
                   "pi05": PI05Client(srv["pi05"]["host"], int(srv["pi05"]["port"]))}
    check_servers(cfg, clients, wait_s=server_wait_s)
    storage = open_storage(ops_cfg["storage_url"])
    record, arrays = golden.load(golden_json)
    episodes = ProductionEpisodes(cfg, clients["xvla"], clients["pi05"], make_sim=make_sim)

    def read(sid):
        data = storage.get_bytes(snapshot_key(sid))
        if data is None:
            raise golden.GoldenMismatch(f"snapshot {sid} is not in the bucket")
        return snap.from_bytes(data)
    try:
        report = golden.check(record, arrays, episodes, read)
    finally:
        episodes.close()
    golden.write_marker(Path(marker or ops_cfg.get("golden_marker", golden.MARKER)), report, cfg.calibration_hash,
                        image_check=images)
    return report


def _safe(worker_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in worker_id)


def tunables_refusal(job, p_escalate: float) -> Optional[str]:
    """Why this worker must not run `job` because of a tunable its spec records (amendment 49), or None."""
    if job.kind == "episode" and job.phase == "explore":
        want = job.spec.get("p_escalate")
        if want is None:
            return "exploration job without p_escalate in its spec (amendment 49): re-plan it"
        if float(want) != float(p_escalate):
            return f"job's p_escalate is {want}, this worker's is {p_escalate}"
    return None


class TrackedEpisodes:
    """`make_episode` for the Runner that remembers the episode it made last, so the lease thread can see it
    advance. Ops code outside the data digest's roots: it returns ProductionEpisodes' episode unchanged."""

    def __init__(self, make_episode: Callable[[int], Any]):
        self.make_episode, self.episode = make_episode, None

    def __call__(self, task_id: int):
        self.episode = self.make_episode(task_id)
        return self.episode

    def token(self) -> Optional[tuple]:
        ep = self.episode
        return None if ep is None else (id(ep), len(ep.events), len(ep.trajectory))


def serve(ops_cfg: Mapping[str, Any], *, worker_id: str, options: Mapping[str, str] = {},
          phases: Optional[Sequence[str]] = None, max_jobs: Optional[int] = None,
          clients: Optional[Mapping[str, Any]] = None, make_sim: Optional[Callable[[int], Any]] = None,
          p_escalate: Optional[float] = None, server_wait_s: float = 600.0) -> dict[str, int]:
    """Run one worker until the queue has nothing it may claim, the stop flag is up, or it refuses a job.

    `clients` ({"xvla": ..., "pi05": ...}) and `make_sim` replace the production ones in tests; `p_escalate`
    defaults to the collection tunables' (exploration runs), and an exploration job whose spec records another
    value is given back."""
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "cgl")
    from vlar_harness import provenance, reference
    from vlar_harness.collect import Runner
    from vlar_harness.config import load_config

    cfg = load_config(**dict(options))
    if not cfg.write_once_complete:
        raise StartupError("no calibration file with the complete write-once group (VLAR_CALIBRATION; Phase 0 "
                           f"step 2, amendment 35): missing {list(cfg.write_once_missing)}")
    waiver = None
    try:
        provenance.verify_freeze(cfg)
    except provenance.FreezeError as e:
        from . import waiver as W
        try:
            waiver = W.check(cfg, e, open_storage(ops_cfg["storage_url"]), ops_source=os.environ.get("VLAR_OPS_SOURCE"))
        except W.WaiverRefused as r:
            raise provenance.FreezeError(f"{e} The ops-only waiver (amendment 49) does not apply: {r}") from e
    image_check(cfg)
    from . import golden
    golden_pass = None
    if cfg.frozen:
        marker = Path(ops_cfg.get("golden_marker", golden.MARKER))
        if not golden.passed_this_boot(marker, cfg.calibration_hash):
            raise StartupError(f"frozen harness without this boot's golden-set pass ({marker}): run "
                               "`vlar-ops golden check` first (§15)")
        golden_pass = json.loads(marker.read_text())
    reference.verify_lerobot(cfg["reference"]["lerobot_files"])

    root = Path(ops_cfg["local_root"])
    stop = StopFlag(root / "STOP")
    if stop.is_set():
        raise StartupError(f"stop flag is up ({stop.path}: {stop.path.read_text().strip()}); no jobs are taken")
    if clients is None:
        from vlar_harness.servers.clients import PI05Client, XVLAClient
        srv = cfg["servers"]
        clients = {"xvla": XVLAClient(srv["xvla"]["host"], int(srv["xvla"]["port"])),
                   "pi05": PI05Client(srv["pi05"]["host"], int(srv["pi05"]["port"]))}
    infos = check_servers(cfg, clients, wait_s=server_wait_s)
    if p_escalate is None:
        from .tunables import explore_fields
        p_escalate = explore_fields()["p_escalate"]

    header = provenance.record_header(cfg)
    storage = open_storage(ops_cfg["storage_url"])
    queue = JobQueue(root / "queue.sqlite", storage, root / "local")
    episodes = ProductionEpisodes(cfg, clients["xvla"], clients["pi05"], make_sim=make_sim)
    from .setupcount import CountingRunner, SetupCounts, meta_with_counts
    counts = SetupCounts(episodes)        # every marker records its setup count; specs with one replay it (amd. 62)
    tracked = TrackedEpisodes(episodes)
    runner = Runner(cfg, tracked, storage.get_bytes, root / "staging" / _safe(worker_id), p_escalate=p_escalate,
                    header=header)
    meta = {"worker_id": worker_id, "provenance": header, "p_escalate": p_escalate,
            "ops_source": os.environ.get("VLAR_OPS_SOURCE"), "boot_id": golden.boot_id(),
            "servers": {n: {k: v for k, v in i.items() if k not in ("batch_sizes", "peak_memory_gb")}
                        for n, i in infos.items()}}
    if golden_pass is not None:
        meta["golden_pass"] = golden_pass
    if waiver is not None:
        meta["ops_waiver"] = waiver
    try:
        counted = CountingRunner(runner, counts)
        return run_worker(queue, storage, counted, worker_id, stop=stop.is_set, max_jobs=max_jobs, phases=phases,
                          refuse=lambda job: runner.refuse(job) or tunables_refusal(job, p_escalate), meta=meta,
                          cleanup=lambda job: shutil.rmtree(runner.job_dir(job), ignore_errors=True),
                          progress=Progress(tracked.token), job_meta=meta_with_counts(meta, counts))
    finally:
        episodes.close()
        queue.close()
