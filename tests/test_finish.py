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
