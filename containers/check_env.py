"""Environment check for the three container images (docs/SPEC.md §15, amendments 28-29).

Run at the end of every image build (no GPU) and again on the L4 VM with `--gpu` before anything else:

    python /opt/vla-router/containers/check_env.py --role sim|pi05|xvla --lock /opt/vla-router/containers/<env>.lock [--gpu]

Checks: CPython 3.13.16; every installed distribution equals the lockfile's pin and nothing is missing; torch is
2.11.0 built for CUDA 13.0 (the notebook's 2.11.0+cu130); the harness's pinned reference files (LeRobot's files for
sim and pi05, the X-VLA checkout's for sim and xvla); for sim and pi05, that LIBERO itself resolves its assets to
the expected directory, whose tree digest is the pinned one, and that all ten LIBERO-Long scenes build from it
(amendment 32). With `--gpu`: the NVIDIA driver is new enough for CUDA 13.0, torch sees the GPU, and (sim) MuJoCo
opens an EGL context. Exit status 0 only if everything holds; every problem is printed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

ROLES = ("sim", "pi05", "xvla")
PYTHON = (3, 13, 16)               # the reference notebook's Python (amendment 30)
TORCH = "2.11.0"
TORCH_CUDA = "13.0"
MIN_DRIVER_TEXT = "580.65.06"       # CUDA 13.0's minimum Linux driver (CUDA 13.0 release notes)
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_pins(text: str) -> dict[str, str]:
    """{canonical name: version} from a pip/uv lockfile (hash continuation lines and comments ignored)."""
    pins = {}
    for line in text.splitlines():
        m = _PIN.match(line.strip())
        if m:
            pins[canon(m.group(1))] = m.group(2)
    return pins


def same_version(locked: str, installed: str) -> bool:
    """PEP 440 equality (packaging's Version), so only different spellings of one version match: thop's wheel
    metadata says 0.1.1-2209072238 for the lock's 0.1.1.post2209072238. Strict otherwise: a different release,
    post-release or local version (2.11.0 vs 2.11.0+cu130) differs; an unparsable version must match exactly. The
    hash-checked install is the real guarantee; this is the second line of defense."""
    from packaging.version import InvalidVersion, Version
    try:
        return Version(locked) == Version(installed)
    except InvalidVersion:
        return locked == installed


def compare(pins: dict[str, str], installed: dict[str, str], *, ignore: Iterable[str] = ()) -> list[str]:
    """Problems between the lock and what is installed: missing, different, or installed but not locked."""
    ignore = {canon(n) for n in ignore}
    out = [f"{n}: locked {v}, not installed" for n, v in sorted(pins.items()) if n not in installed]
    out += [f"{n}: locked {v}, installed {installed[n]}" for n, v in sorted(pins.items())
            if n in installed and not same_version(v, installed[n])]
    out += [f"{n}: installed {v}, not in the lock" for n, v in sorted(installed.items())
            if n not in pins and n not in ignore]
    return out


def installed_distributions() -> dict[str, str]:
    from importlib import metadata
    return {canon(d.metadata["Name"]): d.version for d in metadata.distributions()}


def tree_digest(root: Path, exclude: Iterable[str] = (".cache",)) -> str:
    """SHA-256 over every file's relative path and content hash, in sorted order (as provenance.source_digest)."""
    exclude = set(exclude)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or exclude & set(rel.parts):
            continue
        h.update(rel.as_posix().encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


# Where the images' LIBERO reads its assets: hf-libero 0.1.4's get_assets_path() falls back to its download cache,
# ~/.cache/libero/assets, as in the reference run; the containers run as root (containers/Dockerfile, amendment 32).
LIBERO_ASSETS = "/root/.cache/libero/assets"
LIBERO_LONG_TASKS = 10


def libero_asset_problems(want_sha256: str, expected: Path) -> list[str]:
    """The directory LIBERO itself resolves for its scene and object files (get_assets_path(), not config.yaml, which
    its loaders ignore) must be `expected` and hold the pinned snapshot (tree digest)."""
    try:
        from libero.libero import get_assets_path
        used = Path(get_assets_path())
    except Exception as e:
        return [f"LIBERO assets: cannot ask LIBERO where they are: {type(e).__name__}: {e}"]
    if not used.is_dir():
        return [f"LIBERO assets: LIBERO reads them from {used}, which does not exist (expected {expected})"]
    out = []
    if used.resolve() != expected.resolve():
        out.append(f"LIBERO assets: LIBERO reads them from {used}, expected {expected}")
    got = tree_digest(used)
    if got != want_sha256:
        out.append(f"LIBERO assets at {used}: tree digest {got}, pinned {want_sha256}")
    return out


def libero_scene_problems(expected: Path, suite: str = "libero_10") -> list[str]:
    """Builds every LIBERO-Long task's environment through LIBERO's own loader (bddl file, scene XML, objects, the
    compiled MuJoCo model; no renderer, no camera observations), as the harness's OffScreenRenderEnv does, and checks
    each scene XML came from `expected`."""
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs.env_wrapper import ControlEnv
        tasks = benchmark.get_benchmark_dict()[suite]()
    except Exception as e:
        return [f"LIBERO scenes: cannot load the {suite} benchmark: {type(e).__name__}: {e}"]
    out = [] if tasks.n_tasks == LIBERO_LONG_TASKS else [f"LIBERO scenes: {suite} has {tasks.n_tasks} tasks, not 10"]
    for i in range(tasks.n_tasks):
        task = tasks.get_task(i)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        try:
            env = ControlEnv(bddl_file_name=bddl, use_camera_obs=False, has_renderer=False,
                             has_offscreen_renderer=False)
            try:
                arena = Path(env.env._arena_xml).resolve()
                if expected.resolve() not in arena.parents:
                    out.append(f"LIBERO scene of {suite} task {i} ({task.name}) came from {arena}, not {expected}")
            finally:
                env.close()
        except Exception as e:
            out.append(f"LIBERO scene of {suite} task {i} ({task.name}): {type(e).__name__}: {e}")
    return out


def parse_driver(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.strip().split("."))


MIN_DRIVER = parse_driver(MIN_DRIVER_TEXT)


def driver_problem(version: Optional[str]) -> Optional[str]:
    if not version:
        return "no NVIDIA driver found (nvidia-smi)"
    if parse_driver(version) < MIN_DRIVER:
        return f"NVIDIA driver {version} is older than {MIN_DRIVER_TEXT}, the minimum for CUDA {TORCH_CUDA}"
    return None


def nvidia_driver() -> Optional[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None


def check(role: str, lock: Path, gpu: bool) -> list[str]:
    problems: list[str] = []
    if sys.version_info[:3] != PYTHON:
        problems.append(f"Python {sys.version.split()[0]}, the images use CPython {'.'.join(map(str, PYTHON))}")
    # The project itself is not installed (PYTHONPATH); pip is the venv's own.
    problems += compare(lock_pins(lock.read_text()), installed_distributions(), ignore=("pip",))

    import torch
    if not torch.__version__.startswith(TORCH):
        problems.append(f"torch {torch.__version__}, must be {TORCH}")
    if torch.version.cuda != TORCH_CUDA:
        problems.append(f"torch built for CUDA {torch.version.cuda}, must be {TORCH_CUDA}")

    from vlar_harness import reference
    from vlar_harness.config import load_config
    cfg = load_config()
    try:
        if role in ("sim", "pi05"):
            reference.verify_lerobot(cfg["reference"]["lerobot_files"])
        if role in ("sim", "xvla"):
            reference.verify_xvla(cfg["reference"]["xvla_repo_files"])
    except Exception as e:
        problems.append(f"reference files: {e}")
    if role in ("sim", "pi05"):
        assets = Path(os.environ.get("VLAR_LIBERO_ASSETS", LIBERO_ASSETS))
        problems += libero_asset_problems(cfg["sim"]["assets_sha256"], assets)
        problems += libero_scene_problems(assets)

    if gpu:
        p = driver_problem(nvidia_driver())
        if p:
            problems.append(p)
        if not torch.cuda.is_available():
            problems.append("torch sees no GPU")
        if role == "sim":
            try:
                import mujoco
                ctx = mujoco.GLContext(64, 64)
                ctx.make_current()
                ctx.free()
            except Exception as e:
                problems.append(f"MuJoCo EGL context: {type(e).__name__}: {e}")
    return problems


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--role", choices=ROLES, required=True)
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--gpu", action="store_true", help="also check the driver, CUDA and (sim) EGL: on the VM only")
    a = ap.parse_args(argv)
    problems = check(a.role, a.lock, a.gpu)
    print("\n".join(problems) if problems else f"{a.role} environment OK" + (" (GPU checked)" if a.gpu else ""))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
