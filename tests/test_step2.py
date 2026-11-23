"""Phase 0 step 2's pure parts (amendments 24, 25, 35): the geometry rule, the gripper thresholds, the calibration
update they produce, and the calibration-file CLI the VM steps call."""

import json

import pytest
import yaml

from vlar_harness import calfile
from vlar_harness import config as C
from vlar_ops import step2 as S


def bench(costs, mem=3.0, slow=1.0):
    """costs: size -> deterministic seconds per full batch for each of the four calls."""
    out = {}
    for b, t in costs.items():
        det = {k: {"median_s": t} for k in ("act", "sample", "features")}
        nondet = {k: {"median_s": t / slow} for k in ("act", "sample", "features")}
        out[str(b)] = {"det": {**det, "peak_memory_gb": mem}, "nondet": {**nondet, "peak_memory_gb": mem}}
    pi = {str(b): {"det": {"act": {"median_s": t}, "peak_memory_gb": mem},
                   "nondet": {"act": {"median_s": t / slow}, "peak_memory_gb": mem}} for b, t in costs.items()}
    return out, pi


def test_smallest_size_near_the_best_cost_per_request():
    x, p = bench({1: 0.40, 2: 0.42, 4: 0.50, 8: 0.85, 16: 1.80})      # per request: .40 .21 .125 .106 .1125
    g = S.choose_geometry(x, p, k=5)
    assert (g["max_batch"], g["xvla_sample_rows"], g["workers"]) == (8, 40, 12)
    assert "too_slow" not in g and g["padding_cost_by_fill"]["1"] == pytest.approx(0.85 / 0.40)
    x, p = bench({1: 0.40, 2: 0.42, 4: 0.46, 8: 0.85})                   # 4: .115, within 10% of 8's .106
    assert S.choose_geometry(x, p, k=5)["max_batch"] == 4


def test_memory_and_errors_rule_sizes_out_and_determinism_cost_stops():
    x, p = bench({1: 0.40, 2: 0.42, 4: 0.50, 8: 0.85}, mem=11.0)        # 22 GB > 20 GB everywhere
    with pytest.raises(S.TooSlow, match="no feasible"):
        S.choose_geometry(x, p, k=5)
    x, p = bench({1: 0.40, 2: 0.42, 4: 0.50, 8: 0.85})
    x["8"] = {"error": "out of memory"}
    assert S.choose_geometry(x, p, k=5)["max_batch"] == 4
    x, p = bench({1: 0.40, 2: 0.42, 4: 0.50, 8: 0.85}, slow=1.3)
    g = S.choose_geometry(x, p, k=5)
    assert "deterministic mode costs 1.30x" in g["too_slow"]


def test_gripper_thresholds_sit_inside_the_measured_widths():
    t = S.gripper_thresholds([0.080, 0.079], [0.001, 0.002])
    assert t["shut_width_m"] == pytest.approx(0.002 + 0.1 * 0.077) and t["open_width_m"] == pytest.approx(0.079 - 0.1 * 0.077)
    with pytest.raises(ValueError, match="overlap"):
        S.gripper_thresholds([0.01], [0.02])


def test_updates_fill_the_write_once_group_and_validate(tmp_path, monkeypatch):
    choices = list(C._read_yaml(C.CONFIG_DIR / "harness-v1.yaml")["options"]["network"]["choices"])
    upd = S.calibration_updates({"max_batch": 4, "xvla_sample_rows": 20, "workers": 8},
                                {"median_s": {"xvla_act": 0.3, "xvla_features": 0.2, "xvla_sample_k": 0.9,
                                              "pi05_act": 0.35}, "source": "test"},
                                {"upload_image_bytes": 19000}, {"shut_width_m": 0.01, "open_width_m": 0.07}, choices)
    cal = tmp_path / "calibration.yaml"
    (tmp_path / "u.json").write_text(json.dumps(upd))
    monkeypatch.setenv(C.CALIBRATION_ENV, str(cal))
    assert calfile.main(["write", str(tmp_path / "u.json"), "--note", "step 2"]) == 0
    cfg = C.load_config()
    assert cfg.write_once_complete and cfg["latency"]["pi05_infer_s"] == 0.35 and cfg["servers"]["workers"] == 8
    assert cfg.option("network")["upload_image_bytes"] == 19000


def test_calfile_refuses_without_the_write_once_group(tmp_path, monkeypatch, capsys):
    cal = tmp_path / "calibration.yaml"
    monkeypatch.setenv(C.CALIBRATION_ENV, str(cal))
    C.write_calibration({"latency": {"xvla_act_s": 0.3}})
    assert calfile.main(["require-write-once"]) == calfile.INCOMPLETE_EXIT
    assert calfile.main(["batching-env"]) == calfile.INCOMPLETE_EXIT
    choices = list(C._read_yaml(C.CONFIG_DIR / "harness-v1.yaml")["options"]["network"]["choices"])
    C.write_calibration({"latency": {"pi05_infer_s": 0.4}, "servers": {"batching": {"max_batch": 2, "xvla_sample_rows": 10}},
                         "options": {"network": {"choices": {c: {"upload_image_bytes": 1} for c in choices}}}})
    capsys.readouterr()
    assert calfile.main(["batching-env"]) == 0
    assert capsys.readouterr().out.split() == ["VLAR_MAX_BATCH=2", "VLAR_XVLA_SAMPLE_ROWS=10", "VLAR_WAIT_WINDOW_MS=5"]
    assert calfile.main(["show"]) == 0
    assert json.loads(capsys.readouterr().out)["write_once"]["servers.batching.max_batch"] == 2


def test_plan_cli_uses_the_phase_namespace(tmp_path):
    from vlar_ops import cli
    from vlar_ops.plan import Plan
    cli.main(["plan", "episodes", "--phase", "pi05_parity", "--inits", "0-9", "--mode", "pi05_only",
              "--scene", "pi05_parity", "--out", str(tmp_path / "p.json")])
    plan = Plan.read(tmp_path / "p.json")
    assert len(plan.jobs) == 100 and {j.spec["scene"] for j in plan.jobs} == {"pi05_parity"}
    assert min(j.spec["rollout_seed"] for j in plan.jobs) == 5_000_000
