"""Phase 0 steps 6-7 from the records (`vlar_ops.finish`): event logs rebuilt from the runs and events tables exactly,
incomplete logs refused, step 6's calibration over them equal to the calibration over the logs themselves, and no
freeze before the golden set passed."""

import json

import pytest

from test_labels_calibration import CFG, synthetic_runs
from vlar_harness import calibration as cal
from vlar_ops import finish


def rows(phases):
    """runs and events table rows (as `dataset.JobOutput` writes them) for {phase: [log, ...]}."""
    runs, events, logs = [], [], {}
    for phase, phase_logs in phases.items():
        for i, log in enumerate(phase_logs):
            rid = f"{phase}-{i}"
            runs.append({"run_id": rid, "phase": phase})
            logs[rid] = log
            events += [{"run_id": rid, "seq": k, "t": e["t"], "step": e.get("step"),
                        "event": json.dumps(e, sort_keys=True, allow_nan=True)} for k, e in enumerate(log)]
    return runs, events[::-1], logs                        # stored order is not log order


def test_logs_are_rebuilt_exactly_per_phase():
    base, parity = synthetic_runs()
    runs, events, logs = rows({"base": base, "pi05_parity": parity, "explore": base[:3]})
    got = finish.logs_from_rows(runs, events, "base")
    assert sorted(got) == sorted(r for r in logs if r.startswith("base-"))
    assert all(got[r] == logs[r] for r in got)
    assert len(finish.logs_from_rows(runs, events, "pi05_parity")) == len(parity)


def test_incomplete_logs_and_missing_phases_are_refused():
    base, _ = synthetic_runs()
    runs, events, _ = rows({"base": base[:2]})
    with pytest.raises(finish.RecordsError, match="no 'pi05_parity' runs"):
        finish.logs_from_rows(runs, events, "pi05_parity")
    gap = [e for e in events if not (e["run_id"] == "base-1" and e["seq"] == 3)]
    with pytest.raises(finish.RecordsError, match="base-1: incomplete"):
        finish.logs_from_rows(runs, gap, "base")
    with pytest.raises(finish.RecordsError, match="base-0: incomplete"):
        finish.logs_from_rows(runs, [e for e in events if e["run_id"] != "base-0"], "base")


def test_step6_calibrates_from_the_records_as_from_the_logs(monkeypatch):
    base, parity = synthetic_runs()
    runs, events, _ = rows({"base": base, "pi05_parity": parity})
    monkeypatch.setattr(finish, "logs_from_records", lambda root, phase: finish.logs_from_rows(runs, events, phase))
    res = finish.calibrate("unused", cfg=CFG)
    want = cal.calibrate(base, parity, CFG)
    assert res["margin"] == want.margin and res["per_task_s"] == want.per_task_s
    assert res["base_runs"] == len(base) and res["parity_runs"] == len(parity) and "written" not in res


def test_no_freeze_before_the_golden_check(tmp_path):
    g = tmp_path / "golden.json"
    g.write_text(json.dumps({"golden": [{"snapshot_id": "base/t0/i0/dp03"}], "checked": False}))
    with pytest.raises(finish.RecordsError, match="has not passed"):
        finish.freeze(g)


class _Cfg(dict):
    """The two things `finish.summary` reads from a loaded config: the time-limit section and the calibration hash."""
    def __init__(self, time_limit, calibration_hash):
        super().__init__(time_limit=time_limit)
        self.calibration_hash = calibration_hash


def step6_outputs(tmp_path, monkeypatch, flag=False):
    base, parity = synthetic_runs()
    runs, events, _ = rows({"base": base, "pi05_parity": parity})
    monkeypatch.setattr(finish, "logs_from_records", lambda root, phase: finish.logs_from_rows(runs, events, phase))
    res = json.loads(json.dumps(finish.calibrate("unused", cfg=CFG), default=str))     # as `--out` stores it
    (tmp_path / "records" / "runs" / "base").mkdir(parents=True)
    (tmp_path / "records" / "runs" / "base" / "j0.parquet").write_bytes(b"rows")
    cfg = _Cfg({**CFG["time_limit"], "status": "calibrated", "margin": res["margin"],
                "per_task_s": {int(t): v for t, v in res["per_task_s"].items()}}, "h" * 64)
    settled = {"n": 400, "shut_frac": 0.6, "partial_frac": 0.39, "open_frac": 0.01, "near_shut_frac": 0.05,
               "near_open_frac": 0.0}
    grasps = {"flag": flag, "reasons": ["5.0% of settled steps near the shut threshold"] if flag else [],
              "thresholds": {"shut_width_m": 0.002, "open_width_m": 0.07}, "runs": len(base), "settled": settled,
              "per_task": {}}
    return res, grasps, cfg


def test_step6_summary_for_the_user(tmp_path, monkeypatch):
    res, grasps, cfg = step6_outputs(tmp_path, monkeypatch, flag=True)
    s = finish.summary(tmp_path, res, grasps, cfg=cfg)
    m = res["margin"]
    assert s["margin"] == m and s["pi05_share_within_limits"] == res["share_by_margin"][str(m)]
    assert [r["limit_s"] for r in s["per_task"]] == [res["per_task_s"][str(t)] for t in range(len(s["per_task"]))]
    assert s["calibration_hash"] == "h" * 64 and s["records_digest"] == finish.records_digest(tmp_path)
    assert s["grasps"]["flag"] and s["base_runs"] == res["base_runs"]
    text = finish.write_summary(s, tmp_path / "calibration-summary.json").read_text()
    assert f"margin m = {m}" in text and "grasp check: FLAGGED" in text and "600-freeze.sh" in text
    assert len([ln for ln in text.splitlines() if ln[:4].strip().isdigit()]) == len(s["per_task"])
    assert json.loads((tmp_path / "calibration-summary.json").read_text())["margin"] == m


def test_step6_summary_refuses_limits_it_did_not_compute(tmp_path, monkeypatch):
    res, grasps, cfg = step6_outputs(tmp_path, monkeypatch)
    cfg["time_limit"]["per_task_s"][0] += 1.0
    with pytest.raises(finish.RecordsError, match="not the ones this step computed"):
        finish.summary(tmp_path, res, grasps, cfg=cfg)
    cfg["time_limit"]["status"] = "placeholder"
    with pytest.raises(finish.RecordsError, match="calibrate --write"):
        finish.summary(tmp_path, res, grasps, cfg=cfg)


def test_a_rerun_of_step6_has_nothing_to_do_only_for_the_same_records_and_calibration(tmp_path, monkeypatch):
    res, grasps, cfg = step6_outputs(tmp_path, monkeypatch)
    s = json.loads(json.dumps(finish.summary(tmp_path, res, grasps, cfg=cfg)))
    (tmp_path / "time_limits.json").write_text("{}")             # outside records/: not part of the digest
    assert finish.unchanged(s, tmp_path, cfg=cfg)[0]
    assert not finish.unchanged(s, tmp_path, cfg=_Cfg(cfg["time_limit"], "i" * 64))[0]
    (tmp_path / "records" / "runs" / "base" / "j1.parquet").write_bytes(b"more rows")
    assert finish.unchanged(s, tmp_path, cfg=cfg) == (False, "the records differ from the summary's")
