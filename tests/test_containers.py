"""The container definitions, checked without a container runtime (SPEC §15, §18 item 5; amendments 28-29).

The images are built and tested at the start of Phase 0 (ops/gcp/35_build_images.sh). Here: the base image is pinned
by digest; every COPY source exists and is in the build context; every pip install is hash-checked; every lock entry
carries hashes; the locks pin exactly what the harness config, pyproject and the reference notebook say (amendment
29); the Dockerfile's X-VLA commit and LIBERO assets revision are the config's; the sdist lists are consistent; the
lock split, the environment check's helpers and the driver threshold behave; the shell scripts parse.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from vlar_harness.config import load_config

REPO = Path(__file__).resolve().parents[1]
C = REPO / "containers"
DOCKERFILE = (C / "Dockerfile").read_text()
CFG = load_config()
CONTEXT = {"pyproject.toml", "src", "tests", "References", "containers"}     # .dockerignore's allow list
IN_IMAGE = os.environ.get("VLAR_IN_IMAGE") == "1"                            # set by containers/Dockerfile


def repo_only(*paths):
    """Skip inside an image (VLAR_IN_IMAGE=1) when these repo files are absent there; outside one, a missing file
    still fails the test."""
    missing = [p for p in paths if not (REPO / p).exists()]
    return pytest.mark.skipif(IN_IMAGE and bool(missing), reason=f"repo-only files, not in the image: {missing}")


def load(name):
    spec = importlib.util.spec_from_file_location(name, C / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


locktool, check_env = load("locktool"), load("check_env")


def pins(lock):
    return check_env.lock_pins((C / lock).read_text())


def arg(name):
    m = re.search(rf"^ARG {name}=(\S+)$", DOCKERFILE, re.M)
    assert m, name
    return m.group(1)


def test_the_base_image_is_pinned_by_digest_and_every_stage_builds_on_it():
    assert re.fullmatch(r"ubuntu:24\.04@sha256:[0-9a-f]{64}", arg("UBUNTU"))
    stages = set()
    for src, name in re.findall(r"^FROM (?:--platform=\S+ )?(\S+) AS (\S+)$", DOCKERFILE, re.M):
        assert src == "${UBUNTU}" or src in stages, f"stage {name} builds on {src}"
        stages.add(name)
    assert {"sim", "pi05", "xvla"} <= stages
    assert len(re.findall(r"^FROM ", DOCKERFILE, re.M)) == len(stages)


@repo_only(".dockerignore")
def test_copy_sources_exist_and_are_in_the_build_context():
    ignore = (REPO / ".dockerignore").read_text().split()
    assert "*" in ignore and {f"!{c}" if "." in c else f"!{c}/" for c in CONTEXT} <= set(ignore)
    joined = DOCKERFILE.replace("\\\n", " ")
    for line in re.findall(r"^COPY (?!--from)(.+)$", joined, re.M):
        for src in line.split()[:-1]:
            assert src.split("/")[0] in CONTEXT, src
            assert (REPO / src).exists(), src


def test_every_pip_install_is_hash_checked():
    joined = DOCKERFILE.replace("\\\n", " ")
    installs = re.findall(r"pip (?:install|wheel) [^&\n]+", joined)
    assert len(installs) == 5
    for cmd in installs:
        if "--no-index /tmp/wheels/" in cmd:     # wheels built in the image from hash-checked sdists
            continue
        assert "--require-hashes" in cmd and "--no-deps" in cmd, cmd


@pytest.mark.parametrize("lock", ["lerobot.lock", "xvla.lock", "build.lock"])
def test_every_lock_entry_carries_hashes_and_targets_the_vm(lock):
    text = (C / lock).read_text()
    assert "linux/x86_64" in text and "CPython 3.13" in text
    options, entries = locktool.entries(text)
    assert options == ["--index-url https://pypi.org/simple"] and entries
    for name, entry in entries:
        assert re.search(r"--hash=sha256:[0-9a-f]{64}", entry), name


@pytest.mark.parametrize("env", ["lerobot", "xvla", "build"])
def test_the_lock_has_every_pin_of_its_input(env):
    locked = pins(f"{env}.lock")
    for line in (C / f"{env}.in").read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._\-\[\],]*?)(?:\[[^\]]*\])?==(\S+)$", line.strip())
        if m:
            assert locked[check_env.canon(m.group(1))] == m.group(2), line


def test_lerobot_lock_matches_the_config_and_the_notebook():
    p, sim, pi05 = pins("lerobot.lock"), CFG["sim"], CFG["models"]["pi05"]
    want = {"lerobot": pi05["lerobot_version"], "hf-libero": sim["hf_libero"], "robosuite": sim["robosuite"],
            "mujoco": sim["mujoco"], "numpy": sim["numpy"],
            # amendment 29: the notebook's install output
            "torch": "2.11.0", "torchvision": "0.26.0", "transformers": "5.5.4"}
    assert {k: p.get(k) for k in want} == want
    dataset_pin = re.search(r'dataset = \["pyarrow==([^"]+)"\]', (REPO / "pyproject.toml").read_text()).group(1)
    assert p["pyarrow"] == dataset_pin
    assert p["nvidia-cuda-runtime"].startswith("13.0."), "PyPI's torch 2.11.0 must be the CUDA 13.0 build"


def test_xvla_lock_matches_the_authors_server_pins():
    p = pins("xvla.lock")
    for name, version in CFG["models"]["xvla"]["server_pins"].items():
        assert p[check_env.canon(name)] == version, name
    assert p["torch"] == "2.11.0" and p["nvidia-cuda-runtime"].startswith("13.0.")


def test_dockerfile_sources_match_the_config():
    x = CFG["models"]["xvla"]
    assert arg("XVLA_COMMIT") == x["repo_commit"] and arg("XVLA_REPO") == x["repo"]
    assert arg("ASSETS_REVISION") == CFG["sim"]["assets_revision"] and arg("ASSETS_REPO") == CFG["sim"]["assets_repo"]
    assert re.fullmatch(r"[0-9a-f]{64}", CFG["sim"]["assets_sha256"])


def test_sdist_lists_are_consistent():
    lerobot = (C / "lerobot.sdists").read_text().split()
    assert lerobot and set(lerobot) <= set(pins("lerobot.lock"))
    assert not (C / "xvla.sdists").read_text().split()           # the xvla stage installs --only-binary
    assert not (C / "build.sdists").read_text().split()
    wheels, sdists = locktool.split((C / "lerobot.lock").read_text(), set(lerobot))
    assert set(check_env.lock_pins(sdists)) == set(lerobot)
    assert set(check_env.lock_pins(wheels)) | set(lerobot) == set(pins("lerobot.lock"))
    assert not set(check_env.lock_pins(wheels)) & set(lerobot)


def test_wheel_tags():
    assert locktool.PY_MINOR == 13
    assert locktool.has_wheel(["x-1.0-py3-none-any.whl"])
    assert locktool.has_wheel(["x-1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"])
    assert locktool.has_wheel(["x-1.0-py313-none-any.whl"])                 # e.g. multiprocess
    assert locktool.has_wheel(["x-1.0-cp39-abi3-manylinux_2_28_x86_64.whl"])
    assert not locktool.has_wheel(["x-1.0-cp314-abi3-manylinux_2_28_x86_64.whl"])
    assert not locktool.has_wheel(["x-1.0-cp312-cp312-manylinux_2_28_x86_64.whl", "x-1.0-py312-none-any.whl",
                                   "x-1.0.tar.gz"])
    assert not locktool.has_wheel(["x-1.0-cp313-cp313-macosx_11_0_arm64.whl"])


def test_the_images_run_the_reference_python():
    assert arg("PYTHON_VERSION") == "3.13.16" == ".".join(map(str, check_env.PYTHON))
    assert re.fullmatch(r"[0-9a-f]{64}", arg("PYTHON_SHA256")) and re.fullmatch(r"\d{8}", arg("PYTHON_RELEASE"))
    assert 'sha256sum -c' in DOCKERFILE and "python3.12" not in DOCKERFILE
    assert "--python-version 3.13" in (C / "lock.sh").read_text()


def test_environment_check_helpers(tmp_path):
    lock = "a-b==1.0 \\\n    --hash=sha256:00\n# c\nC_d==2.0 ; sys_platform == 'linux' \\\n    --hash=sha256:11\n"
    assert check_env.lock_pins(lock) == {"a-b": "1.0", "c-d": "2.0"}
    assert check_env.compare({"a-b": "1.0", "c-d": "2.0"}, {"a-b": "1.0", "c-d": "2.0", "pip": "24.0"},
                             ignore=["pip"]) == []
    assert check_env.compare({"a-b": "1.0", "c-d": "2.0"}, {"a-b": "1.1", "e": "3"}) == [
        "c-d: locked 2.0, not installed", "a-b: locked 1.0, installed 1.1", "e: installed 3, not in the lock"]
    assert check_env.driver_problem("580.65.06") is None and check_env.driver_problem("590.1") is None
    assert "older" in check_env.driver_problem("575.57.08") and check_env.driver_problem(None)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f").write_text("x")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "g").write_text("y")
    d = check_env.tree_digest(tmp_path)
    (tmp_path / ".cache" / "g").write_text("z")
    assert check_env.tree_digest(tmp_path) == d
    (tmp_path / "a" / "f").write_text("w")
    assert check_env.tree_digest(tmp_path) != d


@repo_only("ops/vm/load_images.sh", "ops/gcp/30_create_vm.sh")
def test_the_driver_threshold_is_the_same_everywhere():
    want = check_env.MIN_DRIVER_TEXT
    assert want == "580.65.06" and check_env.MIN_DRIVER == (580, 65, 6)
    assert f'MIN_DRIVER="{want}"' in (REPO / "ops" / "vm" / "load_images.sh").read_text()
    assert ">= 580" in (REPO / "ops" / "gcp" / "30_create_vm.sh").read_text()


@repo_only("ops/gcp/env.example.sh", "ops/gcp/35_build_images.sh", "ops/gcp/build_vm_startup.sh",
           "ops/vm/load_images.sh", "ops/gcp/30_create_vm.sh")
def test_build_settings_exist_and_scripts_parse():
    env = (REPO / "ops" / "gcp" / "env.example.sh").read_text()
    used = set(re.findall(r"\$\{(BUILD_[A-Z_]+)", (REPO / "ops" / "gcp" / "35_build_images.sh").read_text()))
    assert used and all(f"export {v}=" in env for v in used)
    for script in ["containers/lock.sh", "ops/gcp/35_build_images.sh", "ops/gcp/build_vm_startup.sh",
                   "ops/vm/load_images.sh", "ops/gcp/30_create_vm.sh", "ops/gcp/env.example.sh"]:
        assert subprocess.run(["bash", "-n", str(REPO / script)]).returncode == 0, script


@repo_only("ops/vm")
def test_service_units_run_the_right_images_on_the_configured_ports():
    vm = REPO / "ops" / "vm"
    for model, image in (("xvla", "vlar-xvla"), ("pi05", "vlar-pi05")):
        unit = (vm / f"vlar-{model}.service").read_text()
        assert f"{image}:${{VLAR_IMAGE_COMMIT}}" in unit and "--host 127.0.0.1" in unit
        assert f"--port {CFG['servers'][model]['port']}" in unit and "--max-batch ${VLAR_MAX_BATCH}" in unit
        assert "/weights:ro" in unit and "[Install]" not in unit          # started by hand, never at boot
    worker = (vm / "vlar-worker@.service").read_text()
    assert "vlar-sim:${VLAR_IMAGE_COMMIT} vlar-ops worker" in worker and "[Install]" not in worker
    assert "--sample-rows ${VLAR_XVLA_SAMPLE_ROWS}" in (vm / "vlar-xvla.service").read_text()
    loader = (vm / "load_images.sh").read_text()
    assert "/etc/vlar/images.env" in loader and "/etc/vlar/batching.env" in loader


def test_every_image_carries_the_in_image_marker():
    # Set in the base stage, which all three targets build on; the repo-only skips depend on it.
    base = DOCKERFILE.split("FROM base AS", 1)[0]
    assert "VLAR_IN_IMAGE=1" in base and DOCKERFILE.count("VLAR_IN_IMAGE") == 1


def test_sdists_build_as_the_reference_notebook_did():
    # The notebook's install cell: cmake>=3.29,<4.2, CMAKE_POLICY_VERSION_MINIMUM=3.5, --no-build-isolation (SPEC §2).
    m = re.search(r"^cmake==(\d+)\.(\d+)\.(\d+) ", (C / "build.lock").read_text(), re.M)
    assert m and (3, 29) <= tuple(map(int, m.groups()[:2])) < (4, 2)
    stage = DOCKERFILE.split("FROM base AS sdist-wheels", 1)[1].split("\nFROM ", 1)[0]
    assert "CMAKE_POLICY_VERSION_MINIMUM=3.5" in stage and "--no-build-isolation" in stage


# Every locked version whose wheel's core metadata spells it differently from the lock (scan of all three locks
# against the locked wheels' metadata on PyPI and the locally built sdist wheels, 2026-11-22): only thop.
METADATA_SPELLINGS = {"lerobot.lock": {"thop": ("0.1.1.post2209072238", "0.1.1-2209072238")}}


@pytest.mark.parametrize("lock, name, locked, metadata",
                         [(lk, n, a, b) for lk, d in METADATA_SPELLINGS.items() for n, (a, b) in d.items()])
def test_metadata_spellings_of_the_locked_version_match(lock, name, locked, metadata):
    assert check_env.lock_pins((C / lock).read_text())[name] == locked
    assert check_env.compare({name: locked}, {name: metadata}) == []


@pytest.mark.parametrize("locked, installed", [
    ("1.0", "1.1"), ("0.1.1.post2209072238", "0.1.1"), ("0.1.1.post2209072238", "0.1.1-2209072239"),
    ("2.11.0", "2.11.0+cu130"), ("2.11.0+cu130", "2.11.0+cu129"), ("1.0rc1", "1.0"), ("not-a-version", "not_a_version"),
])
def test_the_version_check_is_strict_otherwise(locked, installed):
    assert check_env.compare({"p": locked}, {"p": installed}) == [f"p: locked {locked}, installed {installed}"]
