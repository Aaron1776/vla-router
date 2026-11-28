"""The pilot's cheap checks (`vlar_ops.pilot`, PROPOSALS §0.4 (i)-(iii)) on the Mac with stand-in models in the real
simulator: (i) `/features` against the recorded probe features passes on a sound service, reports a `/features` that
differs (the fallback, not a failure) and fails when a fresh `/sample` does not reproduce the record; (ii) `xvla_only`
and (iii) record `off` re-runs of a recorded base episode pass, and record `off` catches a measurement that leaks into
what executes."""
import io
from pathlib import Path

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


def test_point_rows_label_the_pilot_points_from_the_parent_and_branch_logs(sim, sound, tmp_path):
    """Two late points of the recorded base episode, branched with 2 seeds by the production runner: `point_rows`
    scores each branch from the parent's log (b1's seed 1) and the branch logs, as `vlar_harness.labels` does, and
    lists a point whose branch log is missing instead of labelling it."""
    from vlar_harness import clock as clk
    from vlar_harness.collect import Runner
    from vlar_harness.dataset import read_table
    from vlar_ops import pilot
    from vlar_ops.branchplan import branch_plan, stratified_selection
    runs, events = read_table(sound.root, "runs").to_pylist(), read_table(sound.root, "events").to_pylist()
    points, success, _ = pilot.base_points(runs, events)
    late = [p for p in points if p.index >= 47]
    dp = {"scheme": "stratified", "seed": 1, "per_task": 2, "failed_share": 0.5, "min_calls": 1}
    sel = stratified_selection(late, success, step_limit=520, interval=10, dp_sampling=dp)
    plan, labels = branch_plan("pilot", late, branches=CFG["branches"], step_limit=520, interval=10, seeds=2,
                               enabled=["b1_continue", "b2_one_call", "b3_three_calls", "b4_max"], tier_one_seeds=2,
                               order_seed=1, dp_sampling=dp, selection=sel)
    runner = Runner(CFG, factory(sim), sound.get_bytes, tmp_path / "staging", p_escalate=0.0)
    for job in plan.jobs:
        for a in runner(job)[0]:
            sound.put_bytes(a["key"], Path(a["path"]).read_bytes())
    runs, events = read_table(sound.root, "runs").to_pylist(), read_table(sound.root, "events").to_pylist()
    rows, problems = pilot.point_rows(runs, events, labels, limits_s={TASK: 1e9}, charges=clk.Charges.from_config(CFG))
    assert not problems and len(rows) == 2
    won = {r["run_id"]: bool(r["success"]) for r in runs}
    for r in rows:
        sid = r["snapshot_id"]
        b1 = [won[SUBJECT], won[f"{sid}/b1_continue/s2"]]                 # seed 1 is the parent's own continuation
        assert r["p"]["b1_continue"]["none"] == sum(b1) / 2
        canon = labels[sid]["shares"]["b2_one_call"]
        b2 = [won[f"{sid}/{canon}/s{s}"] for s in (1, 2)]
        assert r["gain"]["b2_one_call"]["none"] == pytest.approx(sum(b2) / 2 - sum(b1) / 2)
        assert r["positive"] == (sum(b2) / 2 - sum(b1) / 2 >= 1 / 3 - 1e-12) and r["positive_scored"]
        assert r["inclusion_prob"] == sel[sid]["inclusion_prob"] and r["parent_outcome"] == (
            "success" if won[SUBJECT] else "failure")
    gone = f"{rows[0]['snapshot_id']}/b1_continue/s2"
    rows, problems = pilot.point_rows([r for r in runs if r["run_id"] != gone], events, labels, limits_s={TASK: 1e9},
                                      charges=clk.Charges.from_config(CFG))
    assert len(rows) == 1 and len(problems) == 1 and "missing branch log" in problems[0]


def test_base_loads_finds_the_load_count_that_placed_task_8s_stove():
    """The finding of 2026-11-24: LIBERO places task 8's stove from samplers that accumulate with every model load, so
    base/t8/i30 (the 4th load of w5's task-8 simulator) is another scene on a fresh simulator. `base_loads` recovers
    the count from the model fields at step 0; `warm_up` then reproduces them."""
    from vlar_harness.sim.libero_sim import LiberoSim, SimSpec
    from vlar_ops import pilot
    spec = SimSpec.from_config(CFG)
    make = lambda task: LiberoSim(task, spec)                                   # noqa: E731
    sim = make(8)
    try:
        sim.setup_episode(30, 172)
        fresh = pilot.model_fields(sim)
        sim.close()
        sim = make(8)
        pilot.warm_up(sim, 4)
        sim.setup_episode(30, 172)
        fourth = pilot.model_fields(sim)
    finally:
        sim.close()
    assert not pilot.fields_equal(fresh, fourth, pilot.FIXTURE_FIELDS)        # the stove moved
    assert pilot.base_loads(make, 8, 30, 172, fourth) == 4
    assert pilot.base_loads(make, 8, 30, 172, fresh) == 2
    assert pilot.base_loads(make, 8, 30, 172, {**fresh, "body_pos": fresh["body_pos"] + 1}, max_loads=3) is None
