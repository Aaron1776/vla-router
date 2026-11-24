"""The pilot's cheap checks (`vlar_ops.pilot`, PROPOSALS §0.4 (i)-(iii)) on the Mac with stand-in models in the real
simulator: (i) `/features` against the recorded probe features passes on a sound service, reports a `/features` that
differs (the fallback, not a failure) and fails when a fresh `/sample` does not reproduce the record; (ii) `xvla_only`
and (iii) record `off` re-runs of a recorded base episode pass, and record `off` catches a measurement that leaks into
what executes."""
import io

import pytest
from conftest import requires_lerobot
from test_collect import CFG, TASK
from test_golden import read_from, sim  # noqa: F401  (fixture)
from test_smoke import factory

np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")
pytestmark = requires_lerobot

SUBJECT = f"base/t{TASK}/i0"


def collect(sim, root, *, leaky=False):
    from vlar_harness.collect import Runner
    from vlar_ops.plan import episode_plan
    from vlar_ops.queue import JobQueue
    from vlar_ops.storage import LocalStorage
    from vlar_ops.worker import run_worker
    storage = LocalStorage(root / "bucket")
    q = JobQueue(root / "q.db", storage, root / "local")
    q.add_plan(episode_plan("base", tasks=[TASK], inits=[0], namespace=1_000_000, mode="routed", profile="none",
                            record="full", order_seed=3))
    runner = Runner(CFG, factory(sim, leaky=leaky), storage.get_bytes, root / "staging", p_escalate=0.0)
    assert run_worker(q, storage, runner, "w1", meta={})["done"] == 1
    q.close()
    return storage


@pytest.fixture(scope="module")
def sound(sim, tmp_path_factory):
    return collect(sim, tmp_path_factory.mktemp("sound"))


@pytest.fixture(scope="module")
def leaky(sim, tmp_path_factory):
    return collect(sim, tmp_path_factory.mktemp("leaky"), leaky=True)


def points_of(storage):
    import pyarrow.parquet as pq
    from vlar_ops.plan import job_id
    rows = pq.read_table(io.BytesIO(storage.get_bytes(f"records/points/base/{job_id(SUBJECT, 'episode', 1)}.parquet")))
    return {r["snapshot_id"]: r for r in rows.to_pylist()}


def original_job(storage):
    import json
    from vlar_ops.plan import Job, job_id
    from vlar_ops.queue import result_key
    return Job(**json.loads(storage.get_bytes(result_key("base", job_id(SUBJECT, "episode", 1))))["job"])


class Shifted:
    """An X-VLA service whose /features (or /sample's features) differ from the probe's by one ulp-scale shift."""

    def __init__(self, where):
        from episode_driver import FakeXVLAService
        self.svc, self.where = FakeXVLAService(), where

    def _shift(self, resp):
        f = dict(resp["features"])
        f["enc_mean"] = (np.asarray(f["enc_mean"], dtype=np.float32) + np.float32(1e-3)).tolist()
        return {**resp, "features": f}

    def sample(self, payload, seeds):
        r = self.svc.sample(payload, seeds)
        return self._shift(r) if self.where == "sample" else r

    def features(self, payload):
        r = self.svc.features(payload)
        return self._shift(r) if self.where == "features" else r


def subjects(storage, kind="mid", n=2):
    rows = [r for r in points_of(storage).values() if r["kind"] == kind]
    assert len(rows) >= n
    return [{"snapshot_id": r["snapshot_id"], "task_id": TASK, "episode_id": SUBJECT} for r in rows[1:1 + n]]


def test_features_equal_to_the_probe_pass(sim, sound):
    from episode_driver import FakeXVLAService
    from vlar_ops import pilot
    pts = points_of(sound)
    rep = pilot.features_check(factory(sim), read_from(sound), pts.__getitem__, FakeXVLAService(), subjects(sound))
    assert rep["passed"] and rep["features_equal_probe"] and rep["sample_reproduces_record"], rep["problems"]
    assert len(rep["points"]) == 2 and "screened" in rep["consequence"]


def test_a_features_endpoint_that_differs_is_reported_with_its_fallback(sim, sound):
    from vlar_ops import pilot
    pts = points_of(sound)
    rep = pilot.features_check(factory(sim), read_from(sound), pts.__getitem__, Shifted("features"), subjects(sound))
    assert rep["passed"] and rep["sample_reproduces_record"] and not rep["features_equal_probe"]
    assert rep["points"][0]["features_vs_record_main"]["values_differing"] == 1024
    assert rep["points"][0]["features_vs_record_wrist"]["equal"]
    assert "records full" in rep["consequence"]


def test_a_sample_that_does_not_reproduce_the_record_fails(sim, sound):
    from vlar_ops import pilot
    pts = points_of(sound)
    rep = pilot.features_check(factory(sim), read_from(sound), pts.__getitem__, Shifted("sample"), subjects(sound))
    assert not rep["passed"] and not rep["sample_reproduces_record"] and not rep["features_equal_probe"]
    assert any("did not reproduce" in p for p in rep["problems"])


def test_a_chunk_point_is_not_a_subject(sim, sound):
    from episode_driver import FakeXVLAService
    from vlar_ops import pilot
    pts = points_of(sound)
    rep = pilot.features_check(factory(sim), read_from(sound), pts.__getitem__, FakeXVLAService(),
                               subjects(sound, kind="chunk", n=1))
    assert not rep["passed"] and any("not a mid decision point" in p for p in rep["problems"])


def rerun(sim, storage, tmp_path, *, leaky=False):
    from vlar_harness.collect import Runner
    runner = Runner(CFG, factory(sim, leaky=leaky), storage.get_bytes, tmp_path / "staging", p_escalate=0.0)
    return lambda job: runner(job)[0]


@pytest.mark.parametrize("variant", ["record_off", "xvla_only"])
def test_a_sound_harness_re_runs_the_base_episode_bit_for_bit(sim, sound, tmp_path, variant):
    from vlar_ops import pilot
    rep = pilot.episode_check(rerun(sim, sound, tmp_path), sound.get_bytes, original_job(sound), variant)
    assert rep["passed"], rep["problems"]
    assert rep["tables"] == (["runs", "steps"] if variant == "xvla_only" else ["runs", "steps", "events"])
    assert rep["outcome"]["steps"] > 0 and rep["subject"] == SUBJECT
    assert rep["ran_as"] == {"mode": "xvla_only" if variant == "xvla_only" else "routed", "record": "off"}


def test_record_off_catches_a_measurement_that_changes_what_executes(sim, leaky, tmp_path):
    from vlar_ops import pilot
    rep = pilot.episode_check(rerun(sim, leaky, tmp_path, leaky=True), leaky.get_bytes, original_job(leaky),
                              "record_off")
    assert not rep["passed"] and any(p.startswith(("steps", "runs", "events")) for p in rep["problems"])
