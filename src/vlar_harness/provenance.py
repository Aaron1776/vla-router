"""Provenance header for every record, and the freeze check (docs/SPEC.md §15).

The source digest covers every file of this package that can change behaviour (Python
sources and configs). At the Phase 0 freeze it is written to
`configs/<name>.frozen.json`; from then on `verify_freeze` refuses to run a harness
whose digest differs, because any change to harness code is a new harness version.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CONFIG_DIR, DEFAULT_CONFIG, HarnessConfig

PACKAGE_DIR = Path(__file__).parent
IMAGE_DIGESTS_ENV = "VLAR_IMAGE_DIGESTS"          # JSON: {"sim": "sha256:...", "xvla": ..., "pi05": ...}
IMAGE_DIGESTS_FILE = Path("/etc/vlar/image-digests.json")


class FreezeError(RuntimeError):
    pass


def _digest_files(root: Path) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        if p.suffix not in (".py", ".yaml", ".yml", ".json", ".txt", ".lock", ".toml"):
            continue
        if p.name.endswith(".frozen.json"):
            continue
        files.append(p)
    return files


def source_digest(root: Path = PACKAGE_DIR) -> str:
    h = hashlib.sha256()
    for p in _digest_files(root):
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


GIT_COMMIT_ENV = "VLAR_GIT_COMMIT"                # set in the container images, which are built from a clean commit


def git_commit(repo: Path = PACKAGE_DIR) -> dict[str, Any]:
    """The commit the harness came from. The images carry no .git: they are built from `git archive` of a clean
    commit (ops/gcp/35_build_images.sh) and record it in VLAR_GIT_COMMIT."""
    if os.environ.get(GIT_COMMIT_ENV):
        return {"commit": os.environ[GIT_COMMIT_ENV], "harness_dirty": False}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                              timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", str(PACKAGE_DIR)], cwd=repo,
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "harness_dirty": None}
    return {"commit": head or None, "harness_dirty": bool(dirty)}


def image_digests() -> dict[str, str]:
    raw = os.environ.get(IMAGE_DIGESTS_ENV)
    if raw:
        return json.loads(raw)
    if IMAGE_DIGESTS_FILE.exists():
        return json.loads(IMAGE_DIGESTS_FILE.read_text())
    return {}


def record_header(cfg: HarnessConfig) -> dict[str, Any]:
    return {
        **cfg.record_fields(),
        "source_digest": source_digest(),
        **git_commit(),
        "image_digests": image_digests(),
    }


def freeze_path(name: str = DEFAULT_CONFIG, config_dir: Path = CONFIG_DIR) -> Path:
    return config_dir / f"{name}.frozen.json"


def write_freeze(cfg: HarnessConfig, golden_set: list[str], config_dir: Path = CONFIG_DIR) -> Path:
    """Record the frozen harness identity (Phase 0 step 7). The config must say `frozen: true`."""
    if not cfg.frozen:
        raise FreezeError("set `frozen: true` and the release harness_version in the config first")
    from .config import load_config
    path = freeze_path(cfg.name, config_dir)
    path.write_text(json.dumps({
        "harness_version": cfg["harness_version"],
        "config_hash": load_config(cfg.name, config_dir).hash,   # default options
        "source_digest": source_digest(),
        "golden_set": list(golden_set),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    return path


def verify_freeze(cfg: HarnessConfig, config_dir: Path = CONFIG_DIR) -> dict[str, Any] | None:
    """For a frozen config, check the code and config are exactly what was frozen."""
    if not cfg.frozen:
        return None
    path = freeze_path(cfg.name, config_dir)
    if not path.exists():
        raise FreezeError(f"config is frozen but {path.name} is missing")
    rec = json.loads(path.read_text())
    problems = []
    if rec["harness_version"] != cfg["harness_version"]:
        problems.append("harness_version")
    if rec["source_digest"] != source_digest():
        problems.append("source code or config files")
    # The option choices change the hash, so compare against the default-options config.
    from .config import load_config
    if rec["config_hash"] != load_config(cfg.name, config_dir).hash:
        problems.append("config values")
    if problems:
        raise FreezeError(f"harness differs from frozen {rec['harness_version']} in: {', '.join(problems)}. "
                          "Any harness change needs a new version and the Phase 0 tests (§15).")
    return rec
