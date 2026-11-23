"""Phase 0 steps 6-7 on the VM (ops/vm/phase0/steps/500, 600; lib.sh): the golden-set check's options reach docker and
vlar-ops in the right places, the ops code is mounted over the image's copy (and only vlar_ops), a frozen harness runs
the self-check before any worker, and the step chain 400 -> 500 (by hand) -> 600."""

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "ops" / "vm" / "phase0" / "lib.sh"
STEPS = REPO / "ops" / "vm" / "phase0" / "steps"


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
    assert "queue/600-freeze.sh" in s500 and "finish calibrate" in s500 and "--write" in s500
    assert "grasps-accepted" in s500 and "need_input" in s500 and "handoff-relative-l4" in s500
    assert s600.index("golden select") < s600.index("--mark-checked") < s600.index("finish freeze") \
        < s600.index("verify_freeze")
    assert "queue/" not in s600                                   # Phase 0 ends: the VM stops when the queue is empty
