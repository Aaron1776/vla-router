"""The post-freeze smoke checks (`vlar_ops.smoke`, item 5) on the Mac with stand-in models in the real simulator:
probe neutrality and the π0.5 repeat pass on a sound harness and catch a measurement that leaks into what executes
and a π0.5 that is not deterministic; the re-run comparison passes for the same job collected twice and catches a
changed record and a marker without the waiver, this boot or its golden pass."""
import io
import json

import pytest
from conftest import requires_lerobot
from test_collect import CFG, TASK, THRESHOLDS
from test_golden import golden_set, read_from, sim  # noqa: F401  (fixtures)

np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")
pytestmark = requires_lerobot


class Leaky:
    """An X-VLA service whose measurement calls change what /act returns (the failure probes must not have)."""

    def __init__(self):
        from episode_driver import FakeXVLAService
        self.svc, self.n = FakeXVLAService(), 0

    def act(self, payload, seed):
        return self.svc.act(payload, seed + self.n)

    def sample(self, payload, seeds):
        self.n += 1
        return self.svc.sample(payload, seeds)

    def features(self, payload):
        return self.svc.features(payload)


class Drifting:
    """A π0.5 whose answer depends on how often it was called."""

    def __init__(self):
        from episode_driver import FakePI05
        self.fake, self.n = FakePI05(), 0

    def __call__(self, raw_obs, task, seed):
        self.n += 1
        return self.fake(raw_obs, task, seed + self.n)


def factory(sim, *, leaky=False, drifting=None):
    def make(task_id):
        from episode_driver import FakePI05, FakeXVLAService
        from vlar_harness.adapters.pi05 import PI05Adapter
        from vlar_harness.adapters.xvla import XVLAAdapter
        from vlar_harness.episode import Episode
        assert task_id == TASK
        svc = Leaky() if leaky else FakeXVLAService()
        ep = Episode(sim, XVLAAdapter(svc.act), PI05Adapter(sim), drifting or FakePI05(), CFG, measure_client=svc)
        ep.gripper_thresholds = THRESHOLDS
        ep.calibration_frozen = True        # observe() needs it; the VM's harness is frozen
        return ep
    return make


def test_probe_neutrality_passes_and_really_measures(sim, golden_set):
    from vlar_ops import golden, smoke
    rec, arrays = golden.load(golden_set["out"])
    rep = smoke.probe_neutrality(factory(sim), read_from(golden_set["storage"]), rec, arrays)
    assert rep["passed"], rep["problems"]
    assert rep["observed"] > 0 and rep["probe_points"] > 0 and rep["measure_calls"]["sample"] > 0
    assert rep["sim_time_s"]["live"] > rep["sim_time_s"]["none"]      # cascader_full pays for its probes


def test_probe_neutrality_catches_a_measurement_that_changes_what_executes(sim, golden_set):
    from vlar_ops import golden, smoke
    rec, arrays = golden.load(golden_set["out"])
    rep = smoke.probe_neutrality(factory(sim, leaky=True), read_from(golden_set["storage"]), rec, arrays)
    assert not rep["passed"]
    assert any("executed steps differ" in p for p in rep["problems"])


def test_the_pi05_branch_repeats_and_a_drifting_pi05_is_caught(sim, golden_set):
    from vlar_ops import golden, smoke
    rec, _ = golden.load(golden_set["out"])
    rep = smoke.branch_repeat(factory(sim), read_from(golden_set["storage"]), rec)
    assert rep["passed"], rep
    assert rep["pi05_steps"] > 0 and rep["repeats"] == 2
    bad = smoke.branch_repeat(factory(sim, drifting=Drifting()), read_from(golden_set["storage"]), rec)
    assert not bad["passed"] and any("executed steps differ" in p for p in bad["problems"])


# ---------------------------------------------------------------- re-runs against the dataset

SUBJECT = f"base/t{TASK}/i0"
BOOT, CAL = "boot-1", "cal-1"


def collect(sim, root, meta):
    from vlar_harness.collect import Runner
    from vlar_ops.plan import episode_plan
    from vlar_ops.queue import JobQueue
    from vlar_ops.storage import LocalStorage
    from vlar_ops.worker import run_worker
    storage = LocalStorage(root / "bucket")
    q = JobQueue(root / "q.db", storage, root / "local")
    q.add_plan(episode_plan("base", tasks=[TASK], inits=[0], namespace=1_000_000, mode="routed", profile="none",
                            record="full", order_seed=3))
    runner = Runner(CFG, factory(sim), storage.get_bytes, root / "staging", p_escalate=0.1)
    assert run_worker(q, storage, runner, "w1", meta=meta)["done"] == 1
    q.close()
    return storage


@pytest.fixture(scope="module")
def pair(sim, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    # as on the VM: the originals under a lineage calibration and the freeze's ops code, the re-run under the
    # frozen calibration and newer ops code (provenance only: every executed value must still be equal)
    orig = collect(sim, tmp / "orig", {"provenance": {"ops_digest": "frozen-ops"}})
    relabel(orig, "cal-0", "frozen-ops")
    good = {"provenance": {"ops_digest": "new-ops"}, "ops_waiver": {"freeze_ops_digest": "frozen-ops"},
            "boot_id": BOOT, "golden_pass": {"passed": True, "boot_id": BOOT, "calibration_hash": CAL,
                                             "image_check": {"sim": "sha256:x"}}}
    rerun = collect(sim, tmp / "smoke", good)
    relabel(rerun, CAL, "new-ops")
    return orig, rerun, good


def relabel(storage, cal, ops):
    """Rewrite one collection's provenance (the runs table's calibration hash, every snapshot header) and the
    marker's hashes, as records made under another calibration version and ops code would have it."""
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vlar_ops.plan import job_id
    from vlar_ops.queue import result_key
    mkey = result_key("base", job_id(SUBJECT, "episode", 1))
    marker = json.loads(storage.get_bytes(mkey))
    for art in marker["result"]["artifacts"]:
        data = storage.get_bytes(art["key"])
        if art["key"].startswith("records/runs/"):
            t = pq.read_table(io.BytesIO(data))
            t = t.set_column(t.column_names.index("calibration_hash"), "calibration_hash",
                             pa.array([cal] * t.num_rows, pa.string()))
            buf = io.BytesIO()
            pq.write_table(t, buf)
            data = buf.getvalue()
        elif art["key"].startswith("snapshots/"):
            z = dict(np.load(io.BytesIO(data), allow_pickle=False))
            meta = json.loads(z["__meta__"].tobytes().decode())
            for k, v in meta["$dict"]:
                if k == "header":
                    v["$dict"] = [kv for kv in v.get("$dict", []) if kv[0] not in ("calibration_hash", "ops_digest")]
                    v["$dict"] += [["calibration_hash", cal], ["ops_digest", ops]]
            z["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
            buf = io.BytesIO()
            np.savez_compressed(buf, **z)
            data = buf.getvalue()
        else:
            continue
        storage.put_bytes(art["key"], data)
        art["sha256"], art["size"] = hashlib.sha256(data).hexdigest(), len(data)
    storage.put_bytes(mkey, json.dumps(marker).encode())


def test_plan_from_markers_gives_the_original_jobs(pair, tmp_path):
    from vlar_ops import smoke
    from vlar_ops.plan import Plan
    orig, _, _ = pair
    plan = smoke.plan_from_markers(orig, [SUBJECT])
    plan.write(tmp_path / "p.json")
    (job,) = Plan.read(tmp_path / "p.json").jobs
    assert job.subject == SUBJECT and job.spec["init_idx"] == 0
    with pytest.raises(smoke.SmokeError):
        smoke.plan_from_markers(orig, ["base/t9/i49"])


def test_the_same_job_twice_compares_equal(pair):
    from vlar_ops import smoke
    orig, rerun, _ = pair
    rep = smoke.compare(orig, rerun, [SUBJECT], boot=BOOT, calibration_hash=CAL, lineage=["cal-0"])
    assert rep["passed"], rep
    (j,) = rep["jobs"]
    assert j["provenance_differs_in"] == ["ops_digest"] and j["artifacts"] > 4
    assert j["snapshot_header_differs_in"] == ["calibration_hash", "ops_digest"]
    assert j["calibration"] == {"original": ["cal-0"], "rerun": [CAL]}
    # an original outside the freeze's lineage is not v1.0
    bad = smoke.compare(orig, rerun, [SUBJECT], boot=BOOT, calibration_hash=CAL, lineage=["cal-9"])
    assert not bad["passed"] and any("lineage" in p for p in bad["jobs"][0]["problems"])


def test_a_snapshot_changed_outside_its_header_is_caught(pair):
    from vlar_ops import smoke
    from vlar_harness.dataset import snapshot_key
    orig, rerun, _ = pair
    key = snapshot_key(f"{SUBJECT}/dp03")
    saved = rerun.get_bytes(key)
    z = dict(np.load(io.BytesIO(saved), allow_pickle=False))
    meta = json.loads(z["__meta__"].tobytes().decode())
    pairs = dict((k, i) for i, (k, _) in enumerate(meta["$dict"]))
    meta["$dict"][pairs["event_log"]][1].pop()                     # one event fewer in the stored prefix
    z["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **z)
    try:
        rerun.put_bytes(key, buf.getvalue())
        rep = smoke.compare(orig, rerun, [SUBJECT], boot=BOOT, calibration_hash=CAL, lineage=["cal-0"])
        assert any(f"{key}: snapshot meta differs outside its header" in p for p in rep["jobs"][0]["problems"])
    finally:
        rerun.put_bytes(key, saved)


def test_a_changed_record_is_caught_with_its_column(pair):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vlar_ops import smoke
    from vlar_ops.plan import job_id
    from vlar_harness.dataset import record_key
    orig, rerun, _ = pair
    key = record_key("steps", "base", job_id(SUBJECT, "episode", 1))
    saved = rerun.get_bytes(key)
    try:
        t = pq.read_table(io.BytesIO(saved))
        rows = t.to_pylist()
        rows[7]["eef_pos"][1] = float(np.nextafter(rows[7]["eef_pos"][1], np.inf))
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows, schema=t.schema), buf)
        rerun.put_bytes(key, buf.getvalue())
        rep = smoke.compare(orig, rerun, [SUBJECT], boot=BOOT, calibration_hash=CAL, lineage=["cal-0"])
        assert not rep["passed"]
        assert any("column eef_pos differs first at row 7" in p for p in rep["jobs"][0]["problems"])
    finally:
        rerun.put_bytes(key, saved)


@pytest.mark.parametrize("change", ["no_waiver", "other_freeze", "other_boot", "failed_golden", "no_image_check"])
def test_a_marker_without_the_waiver_this_boot_or_its_golden_pass_is_caught(pair, change):
    from vlar_ops import smoke
    from vlar_ops.plan import job_id
    from vlar_ops.queue import result_key
    orig, rerun, _ = pair
    key = result_key("base", job_id(SUBJECT, "episode", 1))
    saved = rerun.get_bytes(key)
    m = json.loads(saved)
    meta = m["meta"]
    if change == "no_waiver":
        meta.pop("ops_waiver")
    elif change == "other_freeze":
        meta["ops_waiver"]["freeze_ops_digest"] = "another"
    elif change == "other_boot":
        meta["boot_id"] = "boot-0"
    elif change == "failed_golden":
        meta["golden_pass"]["passed"] = False
    else:
        meta["golden_pass"]["image_check"] = None
    try:
        rerun.put_bytes(key, json.dumps(m).encode())
        assert not smoke.compare(orig, rerun, [SUBJECT], boot=BOOT, calibration_hash=CAL, lineage=["cal-0"])["passed"]
    finally:
        rerun.put_bytes(key, saved)
