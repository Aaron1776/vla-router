"""LIBERO reads exactly the pinned assets from the expected directory and builds all ten LIBERO-Long scenes from them
(containers/check_env.py, SPEC §17 amendment 32).

hf-libero 0.1.4's scene and object loaders call `get_assets_path()`, which ignores config.yaml: it takes
`<package>/libero/assets` if present, else `~/.cache/libero/assets` (its download cache, filled from lerobot/libero-assets
on first use, as in the reference run). In the images the expected directory is VLAR_LIBERO_ASSETS
(/root/.cache/libero/assets); on the Mac it is third_party/libero-assets, which .venv-sim's LIBERO reaches through a
package-relative symlink."""

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

from conftest import REPO, requires_libero
from vlar_harness.config import load_config

_spec = importlib.util.spec_from_file_location("check_env", REPO / "containers" / "check_env.py")
check_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_env)

EXPECTED = Path(os.environ.get("VLAR_LIBERO_ASSETS") or REPO / "third_party" / "libero-assets")
PINNED = load_config()["sim"]["assets_sha256"]


@requires_libero
def test_libero_reads_the_pinned_assets_and_builds_all_ten_scenes():
    assert check_env.libero_asset_problems(PINNED, EXPECTED) == []
    assert check_env.libero_scene_problems(EXPECTED) == []


def _point_libero_at(monkeypatch, path):
    import libero.libero as lib
    import libero.libero.envs.bddl_base_domain as domain
    monkeypatch.setattr(lib, "get_assets_path", lambda: str(path))
    monkeypatch.setattr(domain, "get_assets_path", lambda: str(path))


@requires_libero
def test_missing_assets_fail_both_checks(monkeypatch, tmp_path):
    _point_libero_at(monkeypatch, tmp_path / "nowhere")
    assert "does not exist" in " ".join(check_env.libero_asset_problems(PINNED, EXPECTED))
    problems = check_env.libero_scene_problems(EXPECTED)
    assert len(problems) == 10 and all("FileNotFoundError" in p or "Error" in p for p in problems)


@requires_libero
def test_the_same_files_elsewhere_are_refused(monkeypatch, tmp_path):
    copy = tmp_path / "assets"
    shutil.copytree(EXPECTED, copy, symlinks=False, ignore=shutil.ignore_patterns(".cache"))
    _point_libero_at(monkeypatch, copy)
    assert check_env.libero_asset_problems(PINNED, EXPECTED) == [
        f"LIBERO assets: LIBERO reads them from {copy}, expected {EXPECTED}"]
    problems = check_env.libero_scene_problems(EXPECTED)
    assert len(problems) == 10 and all(f"not {EXPECTED}" in p for p in problems)


@requires_libero
def test_a_changed_asset_file_fails_the_digest(monkeypatch, tmp_path):
    copy = tmp_path / "assets"
    shutil.copytree(EXPECTED, copy, symlinks=False, ignore=shutil.ignore_patterns(".cache"))
    scene = next((copy / "scenes").glob("*.xml"))
    scene.write_bytes(scene.read_bytes() + b"\n")
    _point_libero_at(monkeypatch, copy)
    problems = check_env.libero_asset_problems(PINNED, copy)
    assert len(problems) == 1 and "tree digest" in problems[0]
