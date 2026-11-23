"""The golden-set self-check (§15, `vlar_ops.golden`) end to end on the Mac with fake models: base episodes collected
through the queue into a local bucket, five snapshots chosen from the Parquet records, each replayed from its snapshot
to the end bit for bit equal to the base run's recorded steps and outcome; a changed record is caught at the right
step; the pass marker gates a frozen harness's workers."""

import json

import pytest

from conftest import requires_lerobot
from test_collect import CFG, TASK, make_episode

np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")

pytestmark = requires_lerobot


@pytest.fixture(scope="module")
def sim():
    from vlar_harness.sim.libero_sim import LiberoSim, SimSpec
    s = LiberoSim(TASK, SimSpec.from_config(CFG))
    yield s
    s.close()


@pytest.fixture(scope="module")
def golden_set(sim, tmp_path_factory):
    from vlar_harness.collect import Runner
    from vlar_ops import golden
    from vlar_ops.plan import episode_plan
    from vlar_ops.queue import JobQueue
    from vlar_ops.storage import LocalStorage
    from vlar_ops.worker import run_worker
    tmp = tmp_path_factory.mktemp("golden")
    storage = LocalStorage(tmp / "bucket")
    q = JobQueue(tmp / "q.db", storage, tmp / "local")
    q.add_plan(episode_plan("base", tasks=[TASK], inits=[0, 1], namespace=1_000_000, mode="routed", profile="none",
                            record="full", order_seed=3))
    runner = Runner(CFG, make_episode(sim), storage.get_bytes, tmp / "staging", p_escalate=0.1)
    assert run_worker(q, storage, runner, "w1")["done"] == 2
    q.close()
    out = tmp / "golden.json"
    record = golden.select_from_records(tmp / "bucket", out)
    yield {"storage": storage, "out": out, "record": record, "make": make_episode(sim)}


def read_from(storage):
    from vlar_harness import snapshot as snap
    from vlar_harness.dataset import snapshot_key
    return lambda sid: snap.from_bytes(storage.get_bytes(snapshot_key(sid)))


def test_selection_is_recorded_and_reproducible(golden_set, tmp_path):
    from vlar_ops import golden
    rec = golden_set["record"]
    assert rec["n"] == 5 and rec["seed"] == golden.GOLDEN_SEED and not rec["checked"]
    assert len({g["snapshot_id"] for g in rec["golden"]}) == 5
    assert all(g["index"] >= 1 and g["expected_steps"] > 0 for g in rec["golden"])
    again = golden.select_from_records(golden_set["storage"].root, tmp_path / "again.json")
    assert again == rec


def test_the_golden_set_replays_bit_for_bit(golden_set):
    from vlar_ops import golden
    rec, arrays = golden.load(golden_set["out"])
    report = golden.check(rec, arrays, golden_set["make"], read_from(golden_set["storage"]))
    assert report["passed"], report


@pytest.mark.parametrize("what", ["action", "eef_pos", "outcome"])
def test_a_changed_record_is_caught(golden_set, what):
    from vlar_ops import golden
    rec, arrays = golden.load(golden_set["out"])
    rec = json.loads(json.dumps(rec))
    i, k = 2, 4
    if what == "outcome":
        rec["golden"][i]["outcome"]["steps"] += 1
    else:
        arrays = dict(arrays)
        arrays[f"{i}_{what}"] = arrays[f"{i}_{what}"].copy()
        arrays[f"{i}_{what}"][k, 0] = np.nextafter(arrays[f"{i}_{what}"][k, 0], np.inf)
    report = golden.check(rec, arrays, golden_set["make"], read_from(golden_set["storage"]))
    assert not report["passed"]
    problems = [r["problem"] for r in report["results"] if r["problem"]]
    assert len(problems) == 1 and rec["golden"][i]["snapshot_id"] in problems[0]
    if what != "outcome":
        step = int(arrays[f"{i}_step"][k])
        assert f"{what} differs first at step {step}" in problems[0]
    else:
        assert "outcome" in problems[0]


def test_the_pass_marker_gates_a_frozen_harness(tmp_path, monkeypatch):
    """A frozen harness's worker starts only with a pass marker for this boot and this calibration file."""
    import vlar_harness.config as C
    from vlar_harness import provenance, reference
    from vlar_ops import golden, vm_worker

    class Frozen:
        write_once_complete, frozen, calibration_hash = True, True, "cal-1"

        def __getitem__(self, key):
            return {"lerobot_files": {}}

    class Reached(Exception):
        pass
    monkeypatch.setattr(C, "load_config", lambda **kw: Frozen())
    monkeypatch.setattr(provenance, "verify_freeze", lambda cfg: None)
    monkeypatch.setattr(reference, "verify_lerobot", lambda files: (_ for _ in ()).throw(Reached()))
    marker = tmp_path / "golden-pass.json"
    ops = {"local_root": str(tmp_path), "golden_marker": str(marker)}
    with pytest.raises(vm_worker.StartupError, match="golden-set pass"):
        vm_worker.serve(ops, worker_id="w")
    golden.write_marker(marker, {"passed": False, "results": []}, "cal-1")
    with pytest.raises(vm_worker.StartupError, match="golden-set pass"):
        vm_worker.serve(ops, worker_id="w")
    golden.write_marker(marker, {"passed": True, "results": []}, "cal-0")      # another calibration file
    with pytest.raises(vm_worker.StartupError, match="golden-set pass"):
        vm_worker.serve(ops, worker_id="w")
    golden.write_marker(marker, {"passed": True, "results": []}, "cal-1")
    with pytest.raises(Reached):                                                # past the gate
        vm_worker.serve(ops, worker_id="w")
    m = json.loads(marker.read_text())
    m["boot_id"] = "another-boot"
    marker.write_text(json.dumps(m))
    assert not golden.passed_this_boot(marker, "cal-1")
