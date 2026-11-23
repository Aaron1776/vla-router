"""The π0.5 parity gate (SPEC §11 step 4): the planned run only, then McNemar against References/."""

from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from vlar_ops import parity, reference_runs  # noqa: E402

REFS = Path(__file__).resolve().parents[1] / "References"
pytestmark = pytest.mark.skipif(not (REFS / "progress.jsonl").exists(), reason="References/ not present")


def write_runs(root, successes, cal="h" * 64, scene=1000, drop=()):
    from vlar_harness.dataset import _schemas
    rows = []
    for (t, i), ok in sorted(successes.items()):
        if (t, i) in drop:
            continue
        rows.append({"run_id": f"pi05_parity/t{t}/i{i}", "run_kind": "episode", "phase": "pi05_parity",
                     "job_id": f"j{t}{i}", "episode_id": f"pi05_parity/t{t}/i{i}", "task_id": t, "init_idx": i,
                     "scene_seed": scene + i, "rollout_seed": 5_000_000 + 100 * t + i, "mode": "pi05_only",
                     "profile": "none", "record": "off", "time_limit": False, "success": ok,
                     "end_reason": "success" if ok else "step_limit", "steps": 300 if ok else 520,
                     "calibration_hash": cal, "schema_version": 1})
    d = Path(root) / "records" / "runs" / "pi05_parity"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=_schemas()["runs"]), d / "all.parquet")
    return root


@pytest.fixture(scope="module")
def ref():
    return {k: e.success_within(520) for k, e in reference_runs.load_pi05(REFS).items()
            if k[0] in parity.TASKS and k[1] in parity.INITS}


def test_the_reference_itself_passes_and_a_collapse_fails(tmp_path, ref):
    res = parity.gate(write_runs(tmp_path / "a", ref), REFS, calibration_hash="h" * 64)
    assert res["passed"] and res["n"] == 100 and res["only_ours"] == res["only_reference"] == 0
    worse = {k: False for k in ref}
    res = parity.gate(write_runs(tmp_path / "b", worse), REFS, calibration_hash="h" * 64)
    assert not res["passed"] and res["mcnemar_p"] < 0.05


def test_anything_but_the_planned_run_gets_no_verdict(tmp_path, ref):
    with pytest.raises(parity.ParityRunInvalid, match="need exactly 100"):
        parity.gate(write_runs(tmp_path / "a", ref, drop={(3, 4)}), REFS, calibration_hash="h" * 64)
    with pytest.raises(parity.ParityRunInvalid, match="scene seed"):
        parity.gate(write_runs(tmp_path / "b", ref, scene=142), REFS, calibration_hash="h" * 64)
    with pytest.raises(parity.ParityRunInvalid, match="calibration"):
        parity.gate(write_runs(tmp_path / "c", ref), REFS, calibration_hash="g" * 64)
