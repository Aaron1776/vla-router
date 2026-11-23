"""The grasp check after the base rollouts (`vlar_ops.grasps`, `vlar-ops grasps`): commanded-closed finger widths
against the calibrated gripper thresholds, the settled-step rule, the flag, and the CLI on Parquet records."""

import json

import pytest

np = pytest.importorskip("numpy")

from vlar_ops.grasps import grasp_report  # noqa: E402

SHUT, OPEN = 0.008, 0.072          # thresholds of the size Phase 0 step 2 calibrates (Panda: ~0 shut, ~0.08 open)


def episode(run_id, holds, *, open_w=0.08, lead=5, transient=4, hold_steps=20):
    """Steps of one run: `lead` open steps, then for each held width a closing transient and `hold_steps` steps
    commanded closed at that width, then the gripper opens again for `lead` steps."""
    rows, step, w = [], 0, open_w

    def add(g, width):
        nonlocal step
        rows.append({"run_id": run_id, "step": step, "action": [0, 0, 0, 0, 0, 0, g],
                     "gripper_qpos": [width / 2, -width / 2]})
        step += 1
    for _ in range(lead):
        add(-1.0, w)
    for held in holds:
        for k in range(1, transient + 1):                  # closing: the fingers sweep from open to the object
            add(1.0, open_w + (held - open_w) * k / (transient + 1))
        for _ in range(hold_steps):
            add(1.0, held)
        for _ in range(lead):
            add(-1.0, open_w)
    return rows


def runs_and_steps(spec, phase="base"):
    runs, steps = [], []
    for i, (task, holds) in enumerate(spec):
        rid = f"{phase}-{i}"
        runs.append({"run_id": rid, "phase": phase, "task_id": task})
        steps += episode(rid, holds)
    return runs, steps


def test_clean_grasps_are_not_flagged_and_the_transient_is_left_out():
    runs, steps = runs_and_steps([(0, [0.04]), (3, [0.0005]), (6, [0.03, 0.05])])
    rep = grasp_report(runs, steps, shut_width_m=SHUT, open_width_m=OPEN)
    assert not rep["flag"] and rep["reasons"] == []
    s, c = rep["settled"], rep["commanded_closed"]
    # settled: commanded closed for >= 10 consecutive steps (transient included in the count): 15 of each
    # 4 + 20 closed steps, all inside the hold
    assert s["n"] == 4 * 15
    assert s["shut_frac"] == pytest.approx(15 / 60) and s["partial_frac"] == pytest.approx(45 / 60)
    assert s["near_shut_frac"] == 0 and s["near_open_frac"] == 0
    # every commanded-closed step, transient included
    assert c["n"] == 4 * (4 + 20)
    assert c["near_open_frac"] > 0                      # the transient passes the open threshold
    assert set(rep["per_task"]) == {"0", "3", "6"}
    assert rep["per_task"]["3"]["settled"]["shut_frac"] == 1.0
    assert sum(rep["settled_histogram"]["counts"]) == s["n"]


def test_a_grasp_at_the_shut_threshold_is_flagged():
    runs, steps = runs_and_steps([(0, [0.04]), (5, [SHUT + 0.0004]), (5, [SHUT - 0.0003])])
    rep = grasp_report(runs, steps, shut_width_m=SHUT, open_width_m=OPEN)
    assert rep["flag"]
    assert len(rep["reasons"]) == 1 and "shut threshold" in rep["reasons"][0]
    assert rep["settled"]["near_shut_frac"] == pytest.approx(2 / 3)
    assert rep["rule"]["band_m"] == pytest.approx(0.05 * (OPEN - SHUT))


def test_a_grasp_at_the_open_threshold_is_flagged():
    runs, steps = runs_and_steps([(1, [OPEN - 0.001]), (2, [0.03])])
    rep = grasp_report(runs, steps, shut_width_m=SHUT, open_width_m=OPEN)
    assert rep["flag"] and "open threshold" in rep["reasons"][0]


def test_the_limit_and_band_are_the_rule():
    runs, steps = runs_and_steps([(0, [0.04])] * 49 + [(1, [SHUT + 0.0001])])      # 2% exactly: not above it
    assert not grasp_report(runs, steps, shut_width_m=SHUT, open_width_m=OPEN)["flag"]
    runs, steps = runs_and_steps([(0, [0.04])] * 48 + [(1, [SHUT + 0.0001])] * 2)
    assert grasp_report(runs, steps, shut_width_m=SHUT, open_width_m=OPEN)["flag"]
    runs, steps = runs_and_steps([(0, [SHUT + 0.004])])                               # 4 mm off: outside 3.2 mm
    assert not grasp_report(runs, steps, shut_width_m=SHUT, open_width_m=OPEN)["flag"]
    assert grasp_report(runs, steps, shut_width_m=SHUT, open_width_m=OPEN, band=0.1)["flag"]


def test_only_the_phase_runs_count_and_short_closes_never_settle():
    base_runs, base_steps = runs_and_steps([(0, [0.04])])
    other_runs, other_steps = runs_and_steps([(0, [SHUT])], phase="pi05_parity")
    rep = grasp_report(base_runs + other_runs, base_steps + other_steps, shut_width_m=SHUT, open_width_m=OPEN)
    assert rep["runs"] == 1 and not rep["flag"]
    rows = []
    for k in range(9):                                   # 9 closed steps: never settled at 10
        rows.append({"run_id": "base-0", "step": k, "action": [0] * 6 + [1.0], "gripper_qpos": [0.0, 0.0]})
    rep = grasp_report(base_runs, rows, shut_width_m=SHUT, open_width_m=OPEN)
    assert rep["settled"]["n"] == 0 and rep["commanded_closed"]["n"] == 9
    assert rep["flag"] and "consecutive" in rep["reasons"][0]
    with pytest.raises(ValueError, match="no 'base' runs"):
        grasp_report(other_runs, other_steps, shut_width_m=SHUT, open_width_m=OPEN)
    with pytest.raises(ValueError, match="shut_width_m < open_width_m"):
        grasp_report(base_runs, base_steps, shut_width_m=OPEN, open_width_m=SHUT)


def test_steps_are_ordered_by_step_before_the_settled_rule():
    runs, steps = runs_and_steps([(0, [0.04])])
    a = grasp_report(runs, steps, shut_width_m=SHUT, open_width_m=OPEN)
    b = grasp_report(runs, list(reversed(steps)), shut_width_m=SHUT, open_width_m=OPEN)
    assert a == b


def test_cli_on_parquet_records(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from vlar_harness.dataset import _schemas
    from vlar_ops.cli import main
    schemas = _schemas()
    runs, steps = runs_and_steps([(0, [0.04]), (5, [SHUT + 0.0002])])
    full_runs = [{f.name: r.get(f.name) for f in schemas["runs"]} for r in runs]
    full_steps = [{**s, "driver": "xvla", "eef_pos": [0.0] * 3, "eef_quat": [0.0, 0.0, 0.0, 1.0]} for s in steps]
    for table, rows in (("runs", full_runs), ("steps", full_steps)):
        d = tmp_path / "records" / table / "base"
        d.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=schemas[table]), d / "job-0.parquet")
    out = tmp_path / "grasps.json"
    with pytest.raises(SystemExit) as e:
        main(["grasps", "--root", str(tmp_path), "--out", str(out), "--shut", str(SHUT), "--open", str(OPEN)])
    assert e.value.code == 75
    rep = json.loads(out.read_text())
    assert rep["flag"] and rep["runs"] == 2 and rep["settled"]["n"] == 30
    out2 = tmp_path / "clean.json"
    main(["grasps", "--root", str(tmp_path), "--out", str(out2), "--shut", "0.001", "--open", str(OPEN)])
    assert not json.loads(out2.read_text())["flag"]
