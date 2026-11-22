import shutil

import pytest
import yaml

from vlar_harness import config as C
from vlar_harness import provenance as P


@pytest.fixture
def cfg_dir(tmp_path):
    d = tmp_path / "configs"
    shutil.copytree(C.CONFIG_DIR, d)
    for f in d.glob("*.calibration.yaml"):
        f.unlink()
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


def test_calibration_merges_and_changes_hash(cfg_dir):
    before = C.load_config(config_dir=cfg_dir).hash
    C.write_calibration({"latency": {"xvla_features_s": 0.09, "status": "measured"}}, config_dir=cfg_dir)
    cfg = C.load_config(config_dir=cfg_dir)
    assert cfg["latency"]["xvla_features_s"] == 0.09
    assert cfg["latency"]["xvla_act_s"] == 0.420            # untouched keys survive the merge
    assert cfg.hash != before


def test_calibration_rejects_other_paths_and_bad_values(cfg_dir):
    with pytest.raises(C.ConfigError, match="not calibration paths"):
        C.write_calibration({"budget": {"pi05_max_steps": 520}}, config_dir=cfg_dir)
    with pytest.raises(C.ConfigError):
        C.write_calibration({"latency": {"xvla_sample_k_s": 0.1}}, config_dir=cfg_dir)  # below /act
    assert not (cfg_dir / "harness-v1.calibration.yaml").exists()   # rolled back


def test_frozen_config_refuses_calibration_and_placeholders(cfg_dir):
    path = cfg_dir / "harness-v1.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["frozen"] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(C.ConfigError, match="placeholder"):
        C.load_config(config_dir=cfg_dir)
    with pytest.raises(C.ConfigError, match="frozen"):
        C.write_calibration({"latency": {"xvla_act_s": 0.4}}, config_dir=cfg_dir)


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
