import shutil

import pytest
import yaml

from vlar_harness import config as C
from vlar_harness import provenance as P


@pytest.fixture
def cfg_dir(tmp_path):
    d = tmp_path / "configs"
    shutil.copytree(C.CONFIG_DIR, d)
    return d


def test_loads_with_spec_values(cfg_dir):
    cfg = C.load_config(config_dir=cfg_dir)
    assert cfg.step_limit == 520
    assert cfg.n_decision_points == 52
    assert cfg.max_escalations == 26
    assert cfg.option("network")["p_fail"] == 0.05
    assert cfg.option("xvla_state_mode") == "quirk"
    assert cfg.option("render") == "decision_points"                     # §12
    assert C.load_config(config_dir=cfg_dir, render="all").option("render") == "every_step"
    assert cfg.time_limit_s(3) == pytest.approx(37.2)
    assert "proposal" not in cfg["observe_inputs"]["router"]


def test_hash_is_stable_and_tracks_options(cfg_dir):
    a = C.load_config(config_dir=cfg_dir)
    b = C.load_config(config_dir=cfg_dir)
    c = C.load_config(config_dir=cfg_dir, xvla_state_mode="measured")
    assert a.hash == b.hash
    assert a.hash != c.hash
    assert c.record_fields()["config_options"]["xvla_state_mode"] == "measured"


def test_only_named_options(cfg_dir):
    with pytest.raises(C.ConfigError, match="unknown option"):
        C.load_config(config_dir=cfg_dir, step_limit="600")
    with pytest.raises(C.ConfigError, match="not one of"):
        C.load_config(config_dir=cfg_dir, network="lossy")


def test_validation_catches_router_seeing_proposal(cfg_dir):
    path = cfg_dir / "harness-v1.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["observe_inputs"]["router"].append("proposal")
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(C.ConfigError, match="router must never see the proposal"):
        C.load_config(config_dir=cfg_dir)


def _once(cfg_dir, pi05=0.5, max_batch=4, rows=20, upload=19000):
    choices = yaml.safe_load((cfg_dir / "harness-v1.yaml").read_text())["options"]["network"]["choices"]
    return {"latency": {"pi05_infer_s": pi05}, "servers": {"batching": {"max_batch": max_batch, "xvla_sample_rows": rows}},
            "options": {"network": {"choices": {c: {"upload_image_bytes": upload, "upload_image_bytes_status": "measured"}
                                                for c in choices}}}}


def test_calibration_merges_without_changing_the_config_hash(cfg_dir, tmp_path):
    cal = tmp_path / "cal" / "calibration.yaml"
    before = C.load_config(config_dir=cfg_dir, calibration=None)
    assert before.calibration_hash is None and not before.write_once_complete
    C.write_calibration({"latency": {"xvla_features_s": 0.09, "status": "measured"}}, cal, config_dir=cfg_dir)
    cfg = C.load_config(config_dir=cfg_dir, calibration=cal)
    assert cfg["latency"]["xvla_features_s"] == 0.09
    assert cfg["latency"]["xvla_act_s"] == 0.420            # untouched keys survive the merge
    assert cfg.hash == before.hash                          # fixed settings and options only (amendment 35)
    assert cfg.calibration_hash == C.file_sha256(cal) and cfg.record_fields()["calibration_hash"] == cfg.calibration_hash
    assert (cal.parent / "versions" / f"{cfg.calibration_hash}.yaml").read_bytes() == cal.read_bytes()


def test_calibration_file_from_the_environment(cfg_dir, tmp_path, monkeypatch):
    cal = tmp_path / "calibration.yaml"
    monkeypatch.setenv(C.CALIBRATION_ENV, str(cal))
    with pytest.raises(C.ConfigError, match="does not exist"):       # never silently the placeholders
        C.load_config(config_dir=cfg_dir)
    C.write_calibration({"latency": {"xvla_act_s": 0.3}}, config_dir=cfg_dir)
    assert C.load_config(config_dir=cfg_dir)["latency"]["xvla_act_s"] == 0.3
    monkeypatch.delenv(C.CALIBRATION_ENV)
    assert C.load_config(config_dir=cfg_dir)["latency"]["xvla_act_s"] == 0.420


def test_calibration_rejects_other_paths_and_bad_values(cfg_dir, tmp_path):
    cal = tmp_path / "calibration.yaml"
    with pytest.raises(C.ConfigError, match="not calibration paths"):
        C.write_calibration({"budget": {"pi05_max_steps": 520}}, cal, config_dir=cfg_dir)
    with pytest.raises(C.ConfigError):
        C.write_calibration({"latency": {"xvla_sample_k_s": 0.1}}, cal, config_dir=cfg_dir)  # below /act
    with pytest.raises(C.ConfigError, match="xvla_sample_rows must hold K"):
        C.write_calibration({"servers": {"batching": {"xvla_sample_rows": 4}}}, cal, config_dir=cfg_dir)
    assert not cal.exists() and not (tmp_path / "calibration.yaml.tmp").exists()
    cal.write_text(yaml.safe_dump({"schema": 1, "config": "harness-v1", "values": {"env": {"step_limit": 600}}}))
    with pytest.raises(C.ConfigError, match="non-calibration paths"):
        C.load_config(config_dir=cfg_dir, calibration=cal)


def test_write_once_values_never_change(cfg_dir, tmp_path):
    cal = tmp_path / "calibration.yaml"
    C.write_calibration({"latency": {"pi05_infer_s": 0.5}}, cal, config_dir=cfg_dir)
    assert not C.load_config(config_dir=cfg_dir, calibration=cal).write_once_complete
    C.write_calibration(_once(cfg_dir), cal, config_dir=cfg_dir)             # the same π0.5 latency again: fine
    cfg = C.load_config(config_dir=cfg_dir, calibration=cal)
    assert cfg.write_once_complete and cfg["servers"]["batching"]["max_batch"] == 4
    frozen_bytes = cal.read_bytes()
    for update in ({"latency": {"pi05_infer_s": 0.51}}, {"servers": {"batching": {"max_batch": 8}}},
                   {"servers": {"batching": {"xvla_sample_rows": 40}}},
                   {"options": {"network": {"choices": {"default": {"upload_image_bytes": 19001}}}}},
                   {"latency": {"pi05_infer_s": None}}):
        with pytest.raises(C.ConfigError, match="write-once"):
            C.write_calibration(update, cal, config_dir=cfg_dir)
    assert cal.read_bytes() == frozen_bytes
    C.write_calibration({"latency": {"xvla_act_s": 0.3}, "servers": {"workers": 6, "batching": {"wait_window_ms": 3}},
                         "time_limit": {"margin": 1.5}}, cal, config_dir=cfg_dir)   # the rest stays changeable
    assert C.load_config(config_dir=cfg_dir, calibration=cal)["servers"]["workers"] == 6


def test_lineage_is_every_version_with_the_final_write_once_values(cfg_dir, tmp_path):
    cal = tmp_path / "calibration.yaml"
    C.write_calibration({"latency": {"pi05_infer_s": 0.5}}, cal, config_dir=cfg_dir)
    partial = C.file_sha256(cal)                                           # group incomplete: not in the lineage
    C.write_calibration(_once(cfg_dir), cal, config_dir=cfg_dir)
    parity = C.file_sha256(cal)
    C.write_calibration({"time_limit": {"margin": 1.5}}, cal, config_dir=cfg_dir)
    base = C.file_sha256(cal)
    C.write_calibration({"servers": {"workers": 5}}, cal, config_dir=cfg_dir)
    assert sorted(C.calibration_lineage(cal, config_dir=cfg_dir)) == sorted([parity, base])
    assert partial not in C.calibration_lineage(cal, config_dir=cfg_dir)
    (cal.parent / "versions" / f"{parity}.yaml").write_text("tampered")
    with pytest.raises(C.ConfigError, match="does not hash to its name"):
        C.calibration_lineage(cal, config_dir=cfg_dir)


def _complete(cfg_dir, cal):
    C.write_calibration(_once(cfg_dir), cal, config_dir=cfg_dir)
    C.write_calibration({"latency": {"status": "measured"}, "servers": {"workers": 4, "batching": {"status": "measured"}},
                         "measurement": {"gripper_thresholds": {"status": "calibrated", "shut_width_m": 0.01,
                                                                "open_width_m": 0.07}},
                         "time_limit": {"status": "calibrated", "margin": 1.5}}, cal, config_dir=cfg_dir)


def test_frozen_calibration_refuses_changes_and_placeholders(cfg_dir, tmp_path):
    cal = tmp_path / "calibration.yaml"
    C.write_calibration(_once(cfg_dir), cal, config_dir=cfg_dir)
    with pytest.raises(C.ConfigError, match="placeholder"):
        C.write_calibration({}, cal, config_dir=cfg_dir, freeze=True)
    _complete(cfg_dir, cal)
    C.write_calibration({}, cal, config_dir=cfg_dir, freeze=True)
    assert C.load_config(config_dir=cfg_dir, calibration=cal).frozen
    assert not C.load_config(config_dir=cfg_dir, calibration=None).frozen
    with pytest.raises(C.ConfigError, match="frozen"):
        C.write_calibration({"latency": {"xvla_act_s": 0.4}}, cal, config_dir=cfg_dir)
    path = cfg_dir / "harness-v1.yaml"
    path.write_text(path.read_text() + "frozen: true\n")
    with pytest.raises(C.ConfigError, match="must not set `frozen`"):
        C.load_config(config_dir=cfg_dir, calibration=None)


def test_freeze_record_pins_the_calibration_and_its_lineage(cfg_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CONFIG_DIR", cfg_dir)
    cal = tmp_path / "calibration.yaml"
    _complete(cfg_dir, cal)
    with_once = sorted(C.calibration_lineage(cal, config_dir=cfg_dir) + [C.file_sha256(cal)])
    rec_path = P.write_freeze(["g1"], cal, config_dir=cfg_dir)
    cfg = C.load_config(config_dir=cfg_dir, calibration=cal)
    rec = P.verify_freeze(cfg, config_dir=cfg_dir)
    assert rec_path == cal.with_name("freeze.json") and rec["calibration_hash"] == cfg.calibration_hash
    assert sorted(rec["calibration_lineage"]) == with_once                  # every version before the freeze
    assert all(P.in_lineage(rec, h) for h in with_once) and not P.in_lineage(rec, "0" * 64)
    victim = cal.parent / "versions" / f"{with_once[0]}.yaml"
    keep = victim.read_bytes()
    victim.unlink()
    with pytest.raises(P.FreezeError, match="lineage"):
        P.verify_freeze(cfg, config_dir=cfg_dir)
    victim.write_bytes(keep)
    rec_bad = dict(rec, calibration_hash="0" * 64)
    cal.with_name("freeze.json").write_text(__import__("json").dumps(rec_bad))
    with pytest.raises(P.FreezeError, match="calibration file"):
        P.verify_freeze(cfg, config_dir=cfg_dir)


def test_source_digest_changes_with_code(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "configs").mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n")
    (pkg / "configs" / "c.yaml").write_text("k: 1\n")
    d1 = P.source_digest(pkg)
    (pkg / "configs" / "c.frozen.json").write_text("{}")      # the freeze record is excluded
    assert P.source_digest(pkg) == d1
    (pkg / "a.py").write_text("x = 2\n")
    assert P.source_digest(pkg) != d1
