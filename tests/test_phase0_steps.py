"""Phase 0 steps 6-7 on the VM (ops/vm/phase0/steps/500, 600; lib.sh): the golden-set check's options reach docker and
vlar-ops in the right places, the ops code is mounted over the image's copy (and only vlar_ops), a frozen harness runs
the self-check before any worker, every container that runs or checks ops code mounts it (amendment 46), and the step
chain 400 -> 500 (by hand) -> stop for the user -> 600 (by hand, with the user's approval)."""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "ops" / "vm" / "phase0" / "lib.sh"
STEPS = REPO / "ops" / "vm" / "phase0" / "steps"
# ops/ is not in the images (VLAR_IN_IMAGE=1, containers/Dockerfile); outside one, a missing script still fails.
pytestmark = pytest.mark.skipif(os.environ.get("VLAR_IN_IMAGE") == "1" and not LIB.exists(),
                                reason="repo-only files (ops/), not in the image")


def run_lib(tmp_path, script):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "docker").write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$OUT"\n')
    (bin_ / "docker").chmod(0o755)
    out = tmp_path / "args"
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "OUT": str(out),
           "VLAR_PHASE0_DIR": str(tmp_path / "p0"), "VLAR_IMAGE_COMMIT": "abc"}
    r = subprocess.run(["bash", "-c", f'source "{LIB}"; C=abc; {script}'], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out.read_text().splitlines()


def test_golden_check_options_and_the_ops_mount(tmp_path):
    args = run_lib(tmp_path, 'golden_check -v /etc/vlar/calibration:/etc/vlar/calibration -- --mark-checked')
    image = args.index("vlar-sim:abc")
    assert args[image + 1:] == ["vlar-ops", "golden", "check", "--golden", "/etc/vlar/calibration/golden/golden.json",
                                "--mark-checked"]
    before = args[:image]
    assert "/etc/vlar/calibration:/etc/vlar/calibration" in before and "/var/lib/vlar:/var/lib/vlar" in before
    mounts = [before[i + 1] for i, a in enumerate(before) if a == "-v"]
    assert "/opt/vla-router/src/vlar_ops:/opt/vla-router/src/vlar_ops:ro" in mounts
    assert not any("vlar_harness" in m for m in mounts)        # the harness stays the image's (v1.0)


def test_the_golden_check_fetches_the_freeze_supplements_first(tmp_path):
    # Amendment 54: the image digests the freeze pins live in a dated supplement in the bucket's calibration folder;
    # a disk restored before it was written lacks it, so every golden check (each boot's, before any worker) fetches it.
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "calls"
    for tool in ("docker", "gcloud"):
        (bin_ / tool).write_text(f'#!/bin/bash\necho "{tool} $*" >> "{log}"\n')
        (bin_ / tool).chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "VLAR_PHASE0_DIR": str(tmp_path / "p0"),
           "VLAR_STORAGE_URL": "gs://b/vla-router"}
    r = subprocess.run(["bash", "-c", f'source "{LIB}"; C=abc; CAL_DIR="{tmp_path}/cal"; golden_check -v /a:/a -- --x'], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    calls = log.read_text().splitlines()
    assert calls[0] == f"gcloud storage cp gs://b/vla-router/calibration/freeze-supplement-*.json {tmp_path}/cal/"
    assert calls[1].startswith("docker run") and len(calls) == 2


def test_a_frozen_harness_checks_its_golden_set_before_any_worker():
    lib = LIB.read_text()
    body = lib[lib.index("run_phase() {"):]
    assert body.index("wait_servers") < body.index("calibration_frozen") < body.index("golden_check") \
        < body.index("vlar-ops supervise")
    unit = (REPO / "ops" / "vm" / "vlar-worker@.service").read_text()
    assert "-v /opt/vla-router/src/vlar_ops:/opt/vla-router/src/vlar_ops:ro" in unit


def test_the_step_chain():
    s500, s600 = (STEPS / "500-calibrate.sh").read_text(), (STEPS / "600-freeze.sh").read_text()
    # Step 6 ends by stopping for the user (exit 75) once the calibration and its summary are written and uploaded;
    # it never queues the freeze (the user's rule, 2026-11-23). A re-run with nothing new ends done (exit 0).
    assert "queue/" not in s500 and "grasps-accepted" not in s500
    code = [ln for ln in s500.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert code[-1].startswith('need_input "step 6 done, calibration written') and "user's explicit approval" in code[-1]
    order = ["finish unchanged", "exit 0", "handoff-relative-l4", "finish calibrate", "--write", "sync_calibration",
             " grasps ", "finish summary", "the user cannot review step 6", 'need_input "step 6 done']
    body = "\n".join(code)
    assert [body.index(k) for k in order] == sorted(body.index(k) for k in order)
    assert re.search(r'\(\( rc == 0 \|\| rc == 75 \)\) \|\| fail "grasp check failed', s500)
    assert s600.index("golden select") < s600.index("--mark-checked") < s600.index("finish freeze") \
        < s600.index("verify_freeze")
    assert "user's explicit approval" in s600.split("source ")[0]
    assert "queue/" not in s600                                   # Phase 0 ends: the VM stops when the queue is empty
    # Amendment 54: v1.0's freeze recorded no image digests; a later freeze's container sees the loaded images'.
    freeze_cmd = s600.replace("\\\n", " ").split("vlar-ops finish freeze --golden")[0].rsplit("docker run", 1)[1]
    assert "-v /etc/vlar/image-digests.json:/etc/vlar/image-digests.json:ro" in freeze_cmd


def test_every_container_running_ops_code_mounts_the_host_copy():
    # Amendment 46: the freeze pins the digest of the vlar_ops that runs; a container with the image's copy would
    # record or check another digest. Harness-only containers (calfile, calibration_frozen, servers) need no mount.
    for path in (LIB, STEPS / "500-calibrate.sh", STEPS / "600-freeze.sh"):
        text = path.read_text().replace("\\\n", " ")
        for cmd in re.findall(r"docker run [^\n]*", text):
            if "vlar-ops" in cmd or "provenance" in cmd:
                assert "OPS_RUN" in cmd or "SIM_RUN" in cmd or cmd.startswith('docker run "${SIM_RUN'), (path.name, cmd)
    assert '"${OPS_RUN[@]}"' in LIB.read_text().split("SIM_RUN=(")[1].split(")")[0]


def _run_550(tmp_path, frozen):
    """Step 550 with lib.sh's functions stubbed: every calfile/sync call is logged."""
    log = tmp_path / "calls"
    stub = tmp_path / "lib.sh"
    stub.write_text(f"""
fail() {{ echo "FAIL $*" >> "{log}"; exit 1; }}
need_images() {{ :; }}
calibration_frozen() {{ return {0 if frozen else 1}; }}
calfile() {{ echo "calfile $*" >> "{log}"
  case "$1" in write) echo 0.003 > "{tmp_path}/shut" ;; get) cat "{tmp_path}/shut" 2>/dev/null || echo 0.009004 ;; esac; }}
sync_calibration() {{ echo sync >> "{log}"; }}
P0D="{tmp_path}"; VLAR_PHASE0_URL=gs://b/p0
""")
    script = (STEPS / "550-gripper.sh").read_text()
    assert script.count("source /opt/vla-router/ops/vm/phase0/lib.sh") == 1
    (tmp_path / "550.sh").write_text(script.replace("source /opt/vla-router/ops/vm/phase0/lib.sh", f'source "{stub}"'))
    r = subprocess.run(["bash", str(tmp_path / "550.sh")], capture_output=True, text=True)
    return r.returncode, log.read_text().splitlines()


def test_550_refuses_a_frozen_calibration_and_writes_nothing(tmp_path):
    (tmp_path / "frozen").mkdir()
    rc, calls = _run_550(tmp_path / "frozen", frozen=True)
    assert rc == 1 and calls[-1] == "FAIL the calibration file is frozen: gripper thresholds not written"
    assert not any(c.startswith(("calfile write", "sync")) for c in calls)
    rc, calls = _run_550(tmp_path, frozen=False)                # the same stubs reach the write when not frozen
    assert rc == 0 and any(c.startswith("calfile write") for c in calls) and "sync" in calls


def test_the_calibration_writer_refuses_550s_update_once_frozen(tmp_path):
    """The second guard, below the step's: `config.write_calibration` refuses any change to a frozen file, the
    gripper thresholds included (they are not write-once, so only the freeze protects them)."""
    pytest.importorskip("yaml")
    import shutil

    import vlar_harness.config as C
    cfg_dir = tmp_path / "configs"
    shutil.copytree(Path(C.CONFIG_DIR), cfg_dir)
    cal = tmp_path / "calibration.yaml"
    cal.write_text("schema: " + str(C.CALIBRATION_SCHEMA) + "\nconfig: harness-v1\nfrozen: true\nvalues: {}\n")
    with pytest.raises(C.ConfigError, match="frozen"):
        C.write_calibration({"measurement": {"gripper_thresholds": {"shut_width_m": 0.003}}}, cal, config_dir=cfg_dir)
    assert "0.003" not in cal.read_text()
