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


def test_a_frozen_harness_checks_its_golden_set_before_any_worker():
    lib = LIB.read_text()
    body = lib[lib.index("run_phase() {"):]
    assert body.index("wait_servers") < body.index("calibration_frozen") < body.index("golden_check") \
        < body.index('systemctl start "vlar-worker@')
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


def test_every_container_running_ops_code_mounts_the_host_copy():
    # Amendment 46: the freeze pins the digest of the vlar_ops that runs; a container with the image's copy would
    # record or check another digest. Harness-only containers (calfile, calibration_frozen, servers) need no mount.
    for path in (LIB, STEPS / "500-calibrate.sh", STEPS / "600-freeze.sh"):
        text = path.read_text().replace("\\\n", " ")
        for cmd in re.findall(r"docker run [^\n]*", text):
            if "vlar-ops" in cmd or "provenance" in cmd:
                assert "OPS_RUN" in cmd or "SIM_RUN" in cmd or cmd.startswith('docker run "${SIM_RUN'), (path.name, cmd)
    assert '"${OPS_RUN[@]}"' in LIB.read_text().split("SIM_RUN=(")[1].split(")")[0]
