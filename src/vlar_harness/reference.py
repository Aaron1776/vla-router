"""Access to the two reference harnesses' own code (docs/SPEC.md §2: "must be reused, not rewritten").

- X-VLA: the authors' repository at the pinned commit (a git checkout; `VLAR_XVLA_REPO`, default
  `<repo>/third_party/X-VLA`). The simulator side imports `evaluation/libero/libero_client.py`
  from it; the X-VLA server imports `models/` from it.
- LeRobot: the installed `lerobot==0.6.1` package (sim side and π0.5 server).

Every file the harness takes code from is checked against the SHA-256 recorded in the harness
config, so a different checkout or package version fails loudly instead of silently changing
preprocessing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
XVLA_REPO_ENV = "VLAR_XVLA_REPO"


class ReferenceMismatch(RuntimeError):
    pass


def xvla_repo() -> Path:
    path = Path(os.environ.get(XVLA_REPO_ENV, REPO_ROOT / "third_party" / "X-VLA"))
    if not (path / "evaluation" / "libero" / "libero_client.py").exists():
        raise FileNotFoundError(f"X-VLA checkout not found at {path}; set {XVLA_REPO_ENV}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(root: Path, expected: Mapping[str, str], what: str) -> None:
    bad = []
    for rel, digest in expected.items():
        p = root / rel
        if not p.exists():
            bad.append(f"{rel} (missing)")
        elif _sha256(p) != digest:
            bad.append(f"{rel} (sha256 differs)")
    if bad:
        raise ReferenceMismatch(f"{what} at {root} does not match the pinned reference: {', '.join(bad)}")


def lerobot_root() -> Path:
    spec = importlib.util.find_spec("lerobot")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("lerobot is not installed in this environment")
    return Path(spec.origin).parent


def verify_lerobot(expected: Mapping[str, str]) -> None:
    verify_files(lerobot_root(), expected, "lerobot")


def verify_xvla(expected: Mapping[str, str]) -> None:
    verify_files(xvla_repo(), expected, "X-VLA checkout")


@lru_cache(maxsize=1)
def xvla_libero_client(expected_sha256: str | None = None) -> ModuleType:
    """The authors' `libero_client.py`, imported as a module (its `__main__` block does not run).

    Importing it needs LIBERO, robosuite, torch, torchvision, imageio, json_numpy, requests and
    tqdm, exactly as when the reference run executed it.
    """
    path = xvla_repo() / "evaluation" / "libero" / "libero_client.py"
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ReferenceMismatch(f"{path} does not match the pinned sha256")
    name = "vlar_reference_libero_client"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def add_xvla_repo_to_path() -> Path:
    """For the X-VLA server: make the authors' `models` package importable."""
    repo = xvla_repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo
