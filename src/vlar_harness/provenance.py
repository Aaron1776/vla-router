"""Provenance header for every record, and the freeze check (docs/SPEC.md §15, amendments 34, 35).

The source digest covers every file of this package that can change behaviour (Python sources and configs); the
calibration file lives outside the package and is identified by its own hash (`calibration_hash`, in every record).
The Phase 0 freeze (step 7) marks the calibration file frozen and writes `freeze.json` beside it: harness version,
config hash, source digest, image digests, the final calibration hash and its lineage (earlier calibration versions
whose write-once values are unchanged in the final file). From then on `verify_freeze` refuses to run a harness that
differs from the record, and `in_lineage` tells which records count as v1.0.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import config as C
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


FREEZE_FILE = "freeze.json"


def freeze_path(cfg: HarnessConfig) -> Path:
    if not cfg.calibration_path:
        raise FreezeError("no calibration file: the freeze record lives beside it (amendment 35)")
    return Path(cfg.calibration_path).with_name(FREEZE_FILE)


def write_freeze(golden_set: list[str], calibration: Any = C.FROM_ENV, name: str = DEFAULT_CONFIG,
                 config_dir: Path = CONFIG_DIR) -> Path:
    """Phase 0 step 7: freeze the calibration file and record the frozen harness identity beside it. The write-once
    group must be complete; the frozen config must validate (no placeholders)."""
    target = C.calibration_target(calibration)
    cfg = C.load_config(name, config_dir, calibration=target)
    if not cfg.write_once_complete:
        raise FreezeError(f"write-once calibration values missing: {list(cfg.write_once_missing)}")
    C.write_calibration({}, target, name=name, config_dir=config_dir, note="v1.0 freeze", freeze=True)
    cfg = C.load_config(name, config_dir, calibration=target)           # default options
    rec = {
        "harness_version": cfg["harness_version"],
        "config_hash": cfg.hash,
        "source_digest": source_digest(),
        "image_digests": image_digests(),
        "calibration_hash": cfg.calibration_hash,
        "calibration_lineage": C.calibration_lineage(target, name, config_dir),
        "golden_set": list(golden_set),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = freeze_path(cfg)
    path.write_text(json.dumps(rec, indent=2) + "\n")
    return path


def read_freeze(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def in_lineage(rec: dict[str, Any], calibration_hash: Optional[str]) -> bool:
    """Whether a record made under `calibration_hash` counts as the frozen version (amendment 35)."""
    return calibration_hash is not None and (calibration_hash == rec["calibration_hash"]
                                             or calibration_hash in rec["calibration_lineage"])


def verify_freeze(cfg: HarnessConfig, config_dir: Path = CONFIG_DIR) -> dict[str, Any] | None:
    """For a frozen calibration file, check code, config and calibration are exactly what was frozen, and re-check
    the lineage: every listed version is present, hashes to its name and has the final write-once values."""
    if not cfg.frozen:
        return None
    path = freeze_path(cfg)
    if not path.exists():
        raise FreezeError(f"the calibration file is frozen but {path} is missing")
    rec = read_freeze(path)
    problems = []
    if rec["harness_version"] != cfg["harness_version"]:
        problems.append("harness_version")
    if rec["source_digest"] != source_digest():
        problems.append("source code or config files")
    # The option choices change the hash, so compare against the default-options config.
    if rec["config_hash"] != C.load_config(cfg.name, config_dir, calibration=cfg.calibration_path).hash:
        problems.append("config values")
    if rec["calibration_hash"] != cfg.calibration_hash:
        problems.append("calibration file")
    else:
        try:
            lineage = C.calibration_lineage(Path(cfg.calibration_path), cfg.name, config_dir)
        except C.ConfigError as e:
            problems.append(f"calibration lineage ({e})")
        else:
            if sorted(lineage) != sorted(rec["calibration_lineage"]):
                problems.append("calibration lineage (versions missing, added or changed)")
    if problems:
        raise FreezeError(f"harness differs from frozen {rec['harness_version']} in: {', '.join(problems)}. "
                          "Any harness change needs a new version and the Phase 0 tests (§15).")
    return rec
