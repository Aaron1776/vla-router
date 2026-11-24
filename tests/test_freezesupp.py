"""Freeze supplements (vlar_ops.freezesupp, SPEC §17 amendment 54): the image digests `freeze.json` left out, pinned
by a dated file beside it that names the freeze record by its SHA-256, taken from markers written at or before the
freeze; the boot cross-check (`vm_worker.image_check`) refuses a frozen harness on missing or other images."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from vlar_ops import freezesupp as FS

IMG = {"commit": "a" * 40, "sim": "sha256:" + "1" * 64, "pi05": "sha256:" + "2" * 64, "xvla": "sha256:" + "3" * 64}
FROZEN_AT = "2026-11-24T09:46:21+00:00"
T_FREEZE = datetime.fromisoformat(FROZEN_AT).timestamp()


def _freeze(d, images=None):
    p = d / "freeze.json"
    p.write_text(json.dumps({"harness_version": "1.0.0", "calibration_hash": "c" * 64, "frozen_at": FROZEN_AT,
                             "image_digests": images or {}}, indent=2) + "\n")
    return p


def _marker(i, images=IMG, t=T_FREEZE - 3600):
    return {"finished_at": t, "job": {"phase": "base", "job_id": f"j{i}"},
            "meta": {"provenance": {"image_digests": images}}}


def _write(freeze, sup, date="2026-11-24"):
    p = freeze.with_name(FS.name_for(date))
    p.write_text(json.dumps(sup))
    return p


def test_a_supplement_pins_the_markers_images_and_names_the_freeze(tmp_path):
    freeze = _freeze(tmp_path)
    sup = FS.make(freeze, [_marker(i) for i in range(3)], date="2026-11-24", evidence=["build record agrees"])
    assert sup["image_digests"] == IMG and sup["supplements"] == "freeze.json"
    assert sup["freeze_sha256"] == FS._sha256(freeze) and sup["evidence"]["markers"] == 3
    assert sup["evidence"]["latest_marker_at"] == "2026-11-24T08:46:21+00:00"
    assert FS.frozen_image_digests(freeze) == {}                 # nothing beside it yet
    _write(freeze, sup)
    assert FS.frozen_image_digests(freeze) == IMG
    assert FS.check_running(freeze, dict(IMG)) == IMG


@pytest.mark.parametrize("case,match", [
    ("after the freeze", "written after the freeze"),
    ("disagree", "disagree"),
    ("empty digests", "must name"),
    ("no markers", "no markers"),
])
def test_make_refuses(tmp_path, case, match):
    freeze = _freeze(tmp_path)
    markers = {"after the freeze": [_marker(0), _marker(1, t=T_FREEZE + 1)],
               "disagree": [_marker(0), _marker(1, images={**IMG, "xvla": "sha256:" + "4" * 64})],
               "empty digests": [_marker(0, images={})],
               "no markers": []}[case]
    with pytest.raises(FS.SupplementError, match=match):
        FS.make(freeze, markers, date="2026-11-24")


def test_a_supplement_of_another_freeze_or_disagreeing_sources_are_refused(tmp_path):
    freeze = _freeze(tmp_path)
    _write(freeze, FS.make(freeze, [_marker(0)], date="2026-11-24"))
    freeze.write_text(freeze.read_text().replace("1.0.0", "1.0.1"))        # another freeze record
    with pytest.raises(FS.SupplementError, match="another freeze record"):
        FS.frozen_image_digests(freeze)
    (tmp_path / "x").mkdir()
    freeze = _freeze(tmp_path / "x")
    _write(freeze, FS.make(freeze, [_marker(0)], date="2026-11-24"))
    other = {**FS.make(freeze, [_marker(0)], date="2026-11-25"), "image_digests": {**IMG, "sim": "sha256:" + "9" * 64}}
    _write(freeze, other, "2026-11-25")
    with pytest.raises(FS.SupplementError, match="disagrees"):
        FS.frozen_image_digests(freeze)


def test_a_freeze_with_its_own_digests_needs_no_supplement_and_must_agree_with_one(tmp_path):
    freeze = _freeze(tmp_path, IMG)
    assert FS.frozen_image_digests(freeze) == IMG
    sup = {**FS.make(freeze, [_marker(0)], date="2026-11-24"), "image_digests": {**IMG, "pi05": "sha256:" + "8" * 64}}
    _write(freeze, sup)
    with pytest.raises(FS.SupplementError, match="differ from its supplement"):
        FS.frozen_image_digests(freeze)


@pytest.mark.parametrize("running,match", [
    ({**IMG, "xvla": "sha256:" + "5" * 64}, "in: xvla"),
    ({k: v for k, v in IMG.items() if k != "sim"}, "in: sim"),
    ({}, "in: commit, pi05, sim, xvla"),
])
def test_check_running_refuses_other_images(tmp_path, running, match):
    freeze = _freeze(tmp_path)
    _write(freeze, FS.make(freeze, [_marker(0)], date="2026-11-24"))
    with pytest.raises(FS.SupplementError, match=match):
        FS.check_running(freeze, running)


def test_check_running_refuses_a_freeze_that_pins_no_images(tmp_path):
    with pytest.raises(FS.SupplementError, match="pins no image digests"):
        FS.check_running(_freeze(tmp_path), IMG)


def test_the_boot_cross_check(tmp_path, monkeypatch):
    """`vm_worker.image_check`, run by the golden check and by every worker of a frozen harness."""
    from vlar_harness import provenance as P
    from vlar_ops import vm_worker
    freeze = _freeze(tmp_path)
    cfg = SimpleNamespace(frozen=True, calibration_path=str(tmp_path / "calibration.yaml"))
    monkeypatch.setattr(P, "image_digests", lambda: dict(IMG))
    with pytest.raises(vm_worker.StartupError, match="pins no image digests"):
        vm_worker.image_check(cfg)
    _write(freeze, FS.make(freeze, [_marker(0)], date="2026-11-24"))
    assert vm_worker.image_check(cfg) == IMG
    monkeypatch.setattr(P, "image_digests", lambda: {**IMG, "sim": "sha256:" + "7" * 64})
    with pytest.raises(vm_worker.StartupError, match="in: sim"):
        vm_worker.image_check(cfg)
    assert vm_worker.image_check(SimpleNamespace(frozen=False)) is None


def test_serve_and_the_golden_check_run_the_image_check_before_anything_else():
    import inspect

    from vlar_ops import vm_worker
    serve, gc = inspect.getsource(vm_worker.serve), inspect.getsource(vm_worker.golden_check)
    assert serve.index("verify_freeze") < serve.index("image_check(cfg)") < serve.index("passed_this_boot") \
        < serve.index("run_worker(")
    assert gc.index("load_config()") < gc.index("image_check(cfg)") < gc.index("golden.check(")
