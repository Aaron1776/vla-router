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


L4_MIB = 23034                     # an L4's total as nvidia-smi reports it


def joint(sizes, peak=12000, total=L4_MIB, **override):
    """Joint memory measurements (memory-B.json) at each size: `peak` MiB used, with the planned workers rendering;
    `override` maps a size to its own result."""
    out = {str(b): S.memory_result(f"{peak}, {total}\n", size=b, workers=S.workers_for(b)) for b in sizes}
    out.update({str(b): v for b, v in override.items()})
    return out


def test_smallest_size_near_the_best_cost_per_request():
    x, p = bench({1: 0.40, 2: 0.42, 4: 0.50, 8: 0.85, 16: 1.80})      # per request: .40 .21 .125 .106 .1125
    g = S.choose_geometry(x, p, k=5, memory=joint([1, 2, 4, 8, 16]))
    assert (g["max_batch"], g["xvla_sample_rows"], g["workers"]) == (8, 40, 12)
    assert "too_slow" not in g and g["padding_cost_by_fill"]["1"] == pytest.approx(0.85 / 0.40)
    assert g["sizes"]["8"]["free_mib"] == L4_MIB - 12000 and g["sizes"]["8"]["workers_rendering"] == 12
    assert g["rule"]["headroom_mib"] == 2048
    x, p = bench({1: 0.40, 2: 0.42, 4: 0.46, 8: 0.85})                   # 4: .115, within 10% of 8's .106
    assert S.choose_geometry(x, p, k=5, memory=joint([1, 2, 4, 8]))["max_batch"] == 4


def test_joint_memory_needs_2_gb_free_with_the_workers_rendering():
    x, p = bench({1: 0.40, 2: 0.42, 4: 0.46, 8: 0.85})
    # 8 leaves 2047 MiB free: out; 4 leaves exactly 2048 MiB: in (the bench's own per-model peaks no longer gate)
    mem = joint([1, 2], **{"4": S.memory_result(f"{L4_MIB - 2048}, {L4_MIB}", size=4, workers=8),
                           "8": S.memory_result(f"{L4_MIB - 2047}, {L4_MIB}", size=8, workers=12)})
    g = S.choose_geometry(x, p, k=5, memory=mem)
    assert g["max_batch"] == 4 and "2047 MiB free" in g["sizes"]["8"]["why"]
    x, p = bench({1: 0.40, 2: 0.42, 4: 0.50, 8: 0.85}, mem=11.0)          # per-model peaks summing to 22 GB
    assert S.choose_geometry(x, p, k=5, memory=joint([1, 2, 4, 8]))["max_batch"] == 8
    with pytest.raises(S.TooSlow, match="no feasible"):
        S.choose_geometry(x, p, k=5, memory=joint([1, 2, 4, 8], peak=L4_MIB - 1000))


def test_a_size_without_a_complete_joint_measurement_is_infeasible():
    x, p = bench({1: 0.40, 2: 0.42, 4: 0.46, 8: 0.85})
    for bad in ({}, {"8": S.memory_result(None, size=8, workers=12, error="a server exited during the measurement")},
                {"8": S.memory_result("9000, 23034", size=8, workers=8)}):    # rendered with too few workers
        mem = joint([1, 2, 4])
        mem.update(bad)
        g = S.choose_geometry(x, p, k=5, memory=mem)
        assert g["max_batch"] == 4 and not g["sizes"]["8"]["feasible"]
    assert "with 8 workers rendering, not 12" in g["sizes"]["8"]["why"]


def test_errors_rule_sizes_out_and_determinism_cost_stops():
    x, p = bench({1: 0.40, 2: 0.42, 4: 0.50, 8: 0.85})
    x["8"] = {"error": "out of memory"}
    assert S.choose_geometry(x, p, k=5, memory=joint([1, 2, 4, 8]))["max_batch"] == 4
    x, p = bench({1: 0.40, 2: 0.42, 4: 0.50, 8: 0.85}, slow=1.3)
    g = S.choose_geometry(x, p, k=5, memory=joint([1, 2, 4, 8]))
    assert "deterministic mode costs 1.30x" in g["too_slow"]


def test_memory_samples_and_result():
    smi = "11000, 23034\n12500, 23034\n\n[N/A], 23034\n12000, 23034\n"
    assert S.memory_peak(smi) == {"peak_used_mib": 12500, "total_mib": 23034, "samples": 3}
    with pytest.raises(ValueError, match="no nvidia-smi"):
        S.memory_peak("")
    with pytest.raises(ValueError, match="more than one GPU"):
        S.memory_peak("100, 23034\n100, 15360\n")
    r = S.memory_result(smi, size=4, workers=8)
    assert r["free_mib"] == 23034 - 12500 and "error" not in r
    r = S.memory_result(None, size=4, workers=8, error="servers did not come up")
    assert r["error"] == "servers did not come up" and "free_mib" not in r
    assert S.memory_result("", size=4, workers=8)["error"] == "no nvidia-smi memory samples"
    assert [S.workers_for(b) for b in (1, 2, 4, 6, 8, 16)] == [2, 4, 8, 12, 12, 12]


def test_choose_cli_reads_the_memory_files(tmp_path):
    from vlar_ops import cli
    x, p = bench({1: 0.40, 2: 0.42, 4: 0.46, 8: 0.85})
    (tmp_path / "bx.json").write_text(json.dumps({"sizes": x}))
    (tmp_path / "bp.json").write_text(json.dumps({"sizes": p}))
    for b, r in joint([1, 2, 4], **{"8": S.memory_result("22000, 23034", size=8, workers=12)}).items():
        (tmp_path / f"memory-{b}.json").write_text(json.dumps(r))
    with pytest.raises(SystemExit) as e:
        cli.main(["step2", "choose", "--xvla", str(tmp_path / "bx.json"), "--pi05", str(tmp_path / "bp.json"),
                  "--memory-dir", str(tmp_path), "--out", str(tmp_path / "geometry.json")])
    assert e.value.code == 0
    assert json.loads((tmp_path / "geometry.json").read_text())["max_batch"] == 4


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
