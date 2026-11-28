"""Stand-ins for the L4's hardware and containers, so Phase 0 step 200 (ops/vm/phase0/steps/200-step2-measure.sh) runs
end to end on the Mac (tests/reference/test_step200_e2e.py). Run as `python step200_fakes.py <fake> <args>`:

- `docker`: `run` (foreground: the container's command as a host process; `-d --name`: a background process in its
  own process group), `inspect -f {{.State.Running}}`, `ps -q|-aq --filter name=`, `rm -f`, `logs`, `image inspect`.
  Containers see the host's files (mounts are not needed), the `-e` variables, the host environment without any
  `VLAR_*` variable, their own /tmp, and MuJoCo's Mac renderer (the images' `MUJOCO_GL=egl` is dropped). Commands
  run for real (`vlar-ops ...`, `python ...`), except the model images' commands, which have no GPU here:
  the servers (`python -m vlar_harness.servers.{xvla,pi05}_server`) are the production FastAPI apps
  (`create_app`) over fake model services, and the GPU benches (`python -m vlar_harness.servers.bench`) write
  bench files from the plan. The renderers (`vlar-ops step2 render`) are the real `render` command over a stand-in
  simulator (the fake workers).
- `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -lms 250`: samples whose "used" is
  the memory model below over the running containers (each server's share grows with its --max-batch, each
  renderer adds its own), so the joint peak depends on what step 200 actually started.

The plan ($FAKE/plan.json, re-read at every use) sets the benches and the failures:
  {"bench": {"sizes": [1, 2, 4, 8, 16], "costs": {"1": 0.4, ...}, "slow": 1.0},
   "oom_at": 16,                   servers at --max-batch >= this exit at start (out of memory)
   "per_size": {"2": "renderer-no-start" | "renderer-dies" | "server-dies-under-load"}}
The renderers that misbehave are the ones given task 0; a dying server is the X-VLA one.
Every call is logged to $FAKE/calls.jsonl.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

FAKE = Path(os.environ.get("FAKE", "/nonexistent"))
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

TOTAL_MIB = 23034                  # an L4 as nvidia-smi reports it
DRIVER_MIB = 300
XVLA_MIB = (4000, 600)             # base, per row of --max-batch
PI05_MIB = (7000, 500)
RENDER_MIB = 150


def joint_mib(size: int, renderers: int) -> int:
    """What the fake nvidia-smi reads with both servers at `size` and `renderers` renderers running."""
    return DRIVER_MIB + XVLA_MIB[0] + XVLA_MIB[1] * size + PI05_MIB[0] + PI05_MIB[1] * size + RENDER_MIB * renderers


def plan() -> dict:
    p = FAKE / "plan.json"
    return json.loads(p.read_text()) if p.exists() else {}


def log_call(kind: str, argv: list[str]) -> None:
    with open(FAKE / "calls.jsonl", "a") as f:
        f.write(json.dumps({"fake": kind, "argv": argv, "t": time.time()}) + "\n")


# ------------------------------------------------------------------------------------------------ containers

def _records() -> dict[str, dict]:
    """The containers that exist: a stopped `--rm` container is removed, as docker does."""
    out = {}
    for p in sorted((FAKE / "containers").glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if rec["rm"] and not _alive(rec):
            p.unlink(missing_ok=True)
            continue
        out[p.stem] = rec
    return out


def _alive(rec: dict) -> bool:
    try:
        os.kill(rec["pid"], 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a process that has exited but not been reaped yet counts as gone
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(rec["pid"])], capture_output=True, text=True).stdout
    return bool(out.strip()) and not out.strip().startswith("Z")


def _visible(rec: dict, everything: bool) -> bool:
    """`docker ps` lists running containers; `-a` also stopped ones."""
    return everything or _alive(rec)


def _remove(name: str) -> bool:
    p = FAKE / "containers" / f"{name}.json"
    if not p.exists():
        return False
    rec = json.loads(p.read_text())
    try:
        os.killpg(rec["pid"], signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    p.unlink(missing_ok=True)
    return True


def kill_all() -> None:
    """Test cleanup: every background container this fake started."""
    for name in list(_records()):
        _remove(name)


def current_size() -> int | None:
    """--max-batch of the joint measurement's running X-VLA server (what the renderers and servers' plan keys on)."""
    rec = _records().get("vlar-xvla-mem")
    if rec and _alive(rec):
        a = rec["cmd"]
        return int(a[a.index("--max-batch") + 1])
    return None


VALUE_OPTS = {"--name", "-v", "--volume", "-e", "--env", "--network", "--gpus", "-w", "--workdir"}
FLAG_OPTS = {"--rm", "-d", "--detach", "-i", "-t", "-it"}
DROPPED_ENV = {"MUJOCO_GL", "PYOPENGL_PLATFORM"}       # the images render with EGL; the Mac with CGL


def _container_command(image: str, cmd: list[str], tmp: Path) -> list[str]:
    py, fakes = sys.executable, str(Path(__file__).resolve())
    kind = image.split(":")[0]
    if kind not in ("vlar-sim", "vlar-xvla", "vlar-pi05"):
        raise SystemExit(f"fake docker: unknown image {image}")
    cmd = [str(tmp) + a[len("/tmp"):] if a.startswith("/tmp/") else a for a in cmd]
    if cmd[:3] == ["python", "-m", "vlar_harness.servers.bench"]:
        if kind != f"vlar-{cmd[3]}":
            raise SystemExit(f"fake docker: bench {cmd[3]} in {image}")
        return [py, fakes, "bench", *cmd[3:]]
    for model in ("xvla", "pi05"):
        if cmd[:3] == ["python", "-m", f"vlar_harness.servers.{model}_server"]:
            if kind != f"vlar-{model}":
                raise SystemExit(f"fake docker: {model} server in {image}")
            return [py, fakes, f"{model}-server", *cmd[3:]]
    if kind != "vlar-sim":
        raise SystemExit(f"fake docker: unexpected command in {image}: {cmd}")
    if cmd[:3] == ["vlar-ops", "step2", "render"]:
        return [py, fakes, "render", *cmd[1:]]
    if cmd[0] == "vlar-ops":
        return [py, "-m", "vlar_ops.cli", *cmd[1:]]
    if cmd[0] == "python":
        return [py, *cmd[1:]]
    raise SystemExit(f"fake docker: unexpected command {cmd}")


def docker_run(argv: list[str]) -> int:
    i, name, env_opts, flags = 0, None, [], set()
    while i < len(argv) and argv[i].startswith("-"):
        a = argv[i]
        if a in FLAG_OPTS:
            flags.add(a)
            i += 1
        elif a in VALUE_OPTS:
            if a == "--name":
                name = argv[i + 1]
            elif a in ("-e", "--env"):
                env_opts.append(argv[i + 1])
            i += 2
        else:
            raise SystemExit(f"fake docker run: unknown option {a}")
    image, cmd = argv[i], argv[i + 1:]
    detached = bool(flags & {"-d", "--detach"})
    cid = name or f"anon-{time.time_ns()}"
    tmp = FAKE / "tmp" / cid
    tmp.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLAR_")}
    for e in env_opts:
        k, _, v = e.partition("=")
        if k not in DROPPED_ENV:
            env[k] = v if "=" in e else os.environ.get(k, "")
    env.update(TMPDIR=str(tmp), PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(HERE)]),
               PATH=os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin", "/usr/sbin", "/sbin"]))
    full = _container_command(image, cmd, tmp)
    if not detached:
        return subprocess.run(full, env=env).returncode
    if name is None:
        raise SystemExit("fake docker run -d needs --name")
    rec_path = FAKE / "containers" / f"{name}.json"
    if rec_path.exists():
        print(f"docker: Error response from daemon: Conflict. The container name \"/{name}\" is already in use",
              file=sys.stderr)
        return 125
    logf = FAKE / "logs" / f"{name}.{time.time_ns()}.log"
    logf.parent.mkdir(exist_ok=True)
    with open(logf, "wb") as out:
        proc = subprocess.Popen(full, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    rec_path.write_text(json.dumps({"pid": proc.pid, "image": image, "cmd": cmd, "log": str(logf),
                                    "rm": "--rm" in flags}))
    print(name)
    return 0


def docker(argv: list[str]) -> int:
    log_call("docker", argv)
    sub = argv[0]
    if argv[:2] == ["image", "inspect"]:
        return 0
    if sub == "run":
        return docker_run(argv[1:])
    if sub == "inspect":
        if argv[1:3] != ["-f", "{{.State.Running}}"]:
            raise SystemExit(f"fake docker inspect: {argv}")
        rec = _records().get(argv[3])
        if rec is None:
            print(f"Error: No such object: {argv[3]}", file=sys.stderr)
            return 1
        print("true" if _alive(rec) else "false")
        return 0
    if sub == "ps":
        everything = "-aq" in argv or "-a" in argv
        pattern = argv[argv.index("--filter") + 1].split("=", 1)[1]
        for name, rec in _records().items():
            if pattern in name and _visible(rec, everything):
                print(name)
        return 0
    if sub == "rm":
        names = [a for a in argv[1:] if not a.startswith("-")]
        missing = [n for n in names if not _remove(n)]
        for n in missing:
            print(f"Error response from daemon: No such container: {n}", file=sys.stderr)
        return 1 if missing else 0
    if sub == "logs":
        rec = _records().get(argv[1])
        if rec is None:
            return 1
        sys.stdout.write(Path(rec["log"]).read_text(errors="replace"))
        return 0
    raise SystemExit(f"fake docker: unhandled {argv}")


# ------------------------------------------------------------------------------------------------ nvidia-smi

def nvidia_smi(argv: list[str]) -> int:
    log_call("nvidia-smi", argv)
    if argv != ["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-lms", "250"]:
        raise SystemExit(f"fake nvidia-smi: unexpected arguments {argv}")
    (FAKE / "smi-started").write_text(str(current_size()))     # the size whose sampling (and load) has begun
    while True:
        used = DRIVER_MIB
        for name, rec in _records().items():
            if not _alive(rec):
                continue
            c = rec["cmd"]
            if "vlar_harness.servers.xvla_server" in c:
                used += XVLA_MIB[0] + XVLA_MIB[1] * int(c[c.index("--max-batch") + 1])
            elif "vlar_harness.servers.pi05_server" in c:
                used += PI05_MIB[0] + PI05_MIB[1] * int(c[c.index("--max-batch") + 1])
            elif c[:3] == ["vlar-ops", "step2", "render"]:
                used += RENDER_MIB
        print(f"{used}, {TOTAL_MIB}", flush=True)
        time.sleep(0.25)


# ------------------------------------------------------------------------------------------------ model images

def bench(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["xvla", "pi05"])
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args(argv)
    obs = Path(a.obs)
    if not ((obs / "xvla_payloads.json").stat().st_size and (obs / "pi05_requests.json").stat().st_size):
        raise SystemExit("bench: no captured requests")
    b = plan()["bench"]
    slow = float(b.get("slow", 1.0))
    sizes = {}
    for s in b["sizes"]:
        t = float(b["costs"][str(s)])
        if a.model == "xvla":
            row = {mode: {**{k: {"median_s": t / (1 if mode == "det" else slow)} for k in ("act", "sample", "features")},
                          "peak_memory_gb": 3.0} for mode in ("det", "nondet")}
        else:
            row = {mode: {"act": {"median_s": t / (1 if mode == "det" else slow)}, "peak_memory_gb": 4.0}
                   for mode in ("det", "nondet")}
        sizes[str(s)] = row
    Path(a.out).write_text(json.dumps({"model": a.model, "k": a.k, "sizes": sizes, "fake": True}, indent=1))
    return 0


class _Counter:
    def __init__(self, die_after: int | None):
        self.n, self.die_after, self.lock = 0, die_after, threading.Lock()

    def tick(self):
        with self.lock:
            self.n += 1
            if self.die_after is not None and self.n > self.die_after:
                print(f"fake server: exiting after {self.die_after} requests", flush=True)
                os._exit(1)


class FakeXVLAService:
    """X-VLA's endpoints at the server's fixed geometry, answering in the production shapes (30 x 20 actions,
    1024-wide pooled features)."""

    def __init__(self, batching, die_after=None):
        self.batching, self.count = batching, _Counter(die_after)

    def _features(self):
        return {"enc_mean": [0.0] * 1024, "wrist_mean": [0.0] * 1024}

    def _check(self, payload):
        if not isinstance(payload, dict) or not payload:
            raise ValueError("empty payload")
        self.count.tick()

    def act(self, payload):
        self._check(payload)
        if payload.get("seed") is None:
            raise ValueError("unseeded /act")
        time.sleep(0.002)
        return {"action": [[0.0] * 20] * 30, "features": self._features(), "timing_s": {}}

    def sample(self, payload):
        self._check(payload)
        k = self.batching.sample_rows // self.batching.max_batch
        if len(payload.get("seeds") or []) != k:
            raise ValueError(f"/sample needs {k} seeds at this geometry, got {payload.get('seeds')}")
        time.sleep(0.006)
        return {"actions": [[[0.0] * 20] * 30] * k, "features": self._features(), "timing_s": {}}

    def features(self, payload):
        self._check(payload)
        time.sleep(0.001)
        return {"features": self._features(), "timing_s": {}}

    def info(self):
        from dataclasses import asdict
        return {"model": "xvla", "fake": True, "batching": asdict(self.batching)}


def _images(obs, path=""):
    import numpy as np
    if isinstance(obs, dict):
        for k, v in obs.items():
            yield from _images(v, f"{path}.{k}" if path else k)
    elif isinstance(obs, np.ndarray) and obs.dtype == np.uint8 and obs.ndim in (3, 4) and obs.shape[-1] == 3:
        yield path, obs[0] if obs.ndim == 4 else obs


class FakePI05Service:
    """π0.5's endpoints: a 50 x 7 chunk of zeros (the arm holds still) and the JPEG sizes of the observation's
    camera images (`network.jpeg_size`, as the real service encodes its preprocessed images)."""

    def __init__(self, batching, die_after=None):
        self.batching, self.count = batching, _Counter(die_after)

    def act(self, request):
        self.count.tick()
        if not request.get("obs") or not request.get("task") or request.get("seed") is None:
            raise ValueError("π0.5 /act needs obs, task and a seed")
        time.sleep(0.003)
        import numpy as np
        return {"actions": np.zeros((50, 7), dtype=np.float32), "timing_s": {}}

    def upload_jpeg_bytes(self, request):
        from vlar_harness.network import jpeg_size
        self.count.tick()
        quality = int(request.get("quality", 90))
        cams = list(_images(request["obs"]))
        if not cams:
            raise ValueError("no camera images in the observation")
        return {"cameras": [c for c, _ in cams], "jpeg_bytes": [jpeg_size(im, quality) for _, im in cams],
                "shapes": [list(im.shape) for _, im in cams], "quality": quality}

    def info(self):
        from dataclasses import asdict
        return {"model": "pi05", "fake": True, "batching": asdict(self.batching)}


def server(model: str, argv: list[str]) -> int:
    """The production server's command line, the production app, a fake service."""
    import argparse

    import uvicorn
    ap = argparse.ArgumentParser()
    if model == "xvla":
        ap.add_argument("--model-path", required=True)
        ap.add_argument("--lora-path")
        ap.add_argument("--sample-rows", type=int, required=True)
    else:
        ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--max-batch", type=int, required=True)
    a = ap.parse_args(argv)
    p = plan()
    if p.get("oom_at") is not None and a.max_batch >= int(p["oom_at"]):
        time.sleep(0.3)
        print(f"torch.OutOfMemoryError: CUDA out of memory (fake, --max-batch {a.max_batch})", flush=True)
        return 1
    die = 20 if model == "xvla" and p.get("per_size", {}).get(str(a.max_batch)) == "server-dies-under-load" else None
    if model == "xvla":
        from vlar_harness.servers.xvla_server import XVLABatching, create_app
        app = create_app(FakeXVLAService(XVLABatching(max_batch=a.max_batch, sample_rows=a.sample_rows), die))
    else:
        from vlar_harness.servers.pi05_server import PI05Batching, create_app
        app = create_app(FakePI05Service(PI05Batching(max_batch=a.max_batch), die))
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


# ------------------------------------------------------------------------------------------------ renderers

class FakeSim:
    """The simulator interface `step2.render` uses; with `dies_at_size`, it fails once nvidia-smi sampling (and so
    the full-batch load) has begun at that size: after every renderer has started, before the end of the load."""

    def __init__(self, task: int, dies_at_size: int | None):
        self.task, self.dies_at_size = task, dies_at_size

    def setup_episode(self, init_index, env_seed):
        pass

    def set_control_mode(self, mode):
        assert mode == "delta"

    def step(self, action):
        assert len(action) == 7
        time.sleep(0.02)
        started = FAKE / "smi-started"
        if self.dies_at_size is not None and started.exists() and started.read_text() == str(self.dies_at_size):
            raise RuntimeError("fake renderer died")

    def require_images(self):
        return None

    def close(self):
        pass


def render(argv: list[str]) -> int:
    """`vlar-ops step2 render ...` with the stand-in simulator (argv: step2 render --task T --seconds S --out F)."""
    from vlar_ops import cli, step2
    task = int(argv[argv.index("--task") + 1])
    size = current_size()
    behaviour = plan().get("per_size", {}).get(str(size))
    if task == 0 and behaviour == "renderer-no-start":
        print("fake renderer: could not start", flush=True)
        return 1
    dies_at = size if task == 0 and behaviour == "renderer-dies" else None

    def fake_sim(cfg, task_id):
        assert cfg.option("render") == "every_step", cfg.option("render")      # load_config(render="all")
        return FakeSim(task_id, dies_at)

    step2._sim = fake_sim
    cli.main(argv)
    return 0


if __name__ == "__main__":
    fake, args = sys.argv[1], sys.argv[2:]
    if fake not in ("docker", "nvidia-smi"):
        log_call(fake, args)
    fn = {"docker": docker, "nvidia-smi": nvidia_smi, "bench": bench, "render": render,
          "xvla-server": lambda a: server("xvla", a), "pi05-server": lambda a: server("pi05", a)}[fake]
    sys.exit(fn(args))
