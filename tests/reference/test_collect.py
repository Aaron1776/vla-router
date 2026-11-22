"""Collection end to end on the Mac (§7, §12, §13): queue -> worker -> runner -> bucket -> dataset, with fake models.

A base and an exploration episode (task 0, record="full") run through `run_worker` into a local bucket, then a
branch job continues one of the base episode's snapshots from the bucket. Everything in the dataset must equal a
direct run of the same episode: the snapshot files byte for byte, the trajectory, the events, every measured point,
the images. The exploration run must escalate exactly where its seed says.
"""

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")

from conftest import requires_lerobot
from vlar_harness.config import load_config

CFG = load_config()
TASK, INIT = 0, 4
THRESHOLDS = (0.01, 0.07)
P_ESCALATE = 0.1
BRANCH_DP = 7


@pytest.fixture(scope="module")
def sim():
    from vlar_harness.sim.libero_sim import LiberoSim, SimSpec
    s = LiberoSim(TASK, SimSpec.from_config(CFG))
    yield s
    s.close()


def make_episode(sim):
    def make(task_id):
        from episode_driver import FakePI05, FakeXVLAService
        from vlar_harness.adapters.pi05 import PI05Adapter
        from vlar_harness.adapters.xvla import XVLAAdapter
        from vlar_harness.episode import Episode
        assert task_id == TASK
        svc = FakeXVLAService()
        ep = Episode(sim, XVLAAdapter(svc.act), PI05Adapter(sim), FakePI05(), CFG, measure_client=svc)
        ep.gripper_thresholds = THRESHOLDS
        return ep
    return make


@pytest.fixture(scope="module")
def collected(sim, tmp_path_factory):
    from vlar_harness import snapshot as snap
    from vlar_harness.collect import Runner
    from vlar_harness.dataset import snapshot_key
    from vlar_ops.plan import Job, Plan, episode_plan, job_id
    from vlar_ops.queue import JobQueue
    from vlar_ops.storage import LocalStorage
    from vlar_ops.worker import run_worker

    tmp = tmp_path_factory.mktemp("collect")
    storage = LocalStorage(tmp / "bucket")
    q = JobQueue(tmp / "q.db", storage, tmp / "local")
    for phase, ns in (("base", 1_000_000), ("explore", 2_000_000)):
        q.add_plan(episode_plan(phase, tasks=[TASK], inits=[INIT], namespace=ns, mode="routed", profile="none",
                                record="full", order_seed=7))
    runner = Runner(CFG, make_episode(sim), storage.get_bytes, tmp / "staging", p_escalate=P_ESCALATE)
    stats = run_worker(q, storage, runner, "w1")

    # The same base episode run directly, snapshots kept in memory.
    ep = make_episode(sim)(TASK)
    direct = {"snaps": {}}
    dp = ep.start(INIT, 1_000_000 + INIT, "routed", "base/t0/i4", record="full")
    while dp is not None:
        if ep.point_meas is not None and ep.point_meas["index"] == dp.index:
            direct["snaps"][dp.index] = snap.to_bytes(ep.snapshot(header=runner.header))
        dp = ep.decide("local").next_point
    direct.update(trajectory=list(ep.trajectory), events=list(ep.events), measurements=list(ep.measurements))

    sid = f"base/t{TASK}/i{INIT}/dp{BRANCH_DP:02d}"
    spec = {"snapshot_id": sid, "branch": "b2_one_call", "pi05_calls": 1, "seed_index": 1, "task_id": TASK,
            "init_idx": INIT, "source": "base"}
    q.add_plan(Plan("branches", {}, [Job(job_id(sid, "b2_one_call", 1), "branches", "b2_one_call", sid, 1, 1, 0,
                                         spec)]))
    bstats = run_worker(q, storage, runner, "w1")
    ep_b = make_episode(sim)(TASK)
    direct_branch = ep_b.run_branch(snap.from_bytes(storage.get_bytes(snapshot_key(sid))), branch="b2_one_call",
                                    pi05_calls=1, seed_index=1, snapshot_id=sid)
    yield {"root": tmp / "bucket", "storage": storage, "stats": stats, "bstats": bstats, "direct": direct,
           "direct_branch": direct_branch, "queue": q, "sid": sid}
    q.close()


def table(c, name):
    from vlar_harness.dataset import read_table
    return read_table(c["root"], name).to_pylist()


@requires_lerobot
def test_every_job_is_done_with_its_files_in_the_bucket(collected):
    import hashlib
    from vlar_ops.queue import RESULTS_PREFIX
    assert collected["stats"]["done"] == 2 and collected["bstats"]["done"] == 1
    st = collected["storage"]
    markers = list(st.list(f"{RESULTS_PREFIX}/"))
    assert len(markers) == 3
    for key in markers:
        rec = json.loads(st.get_bytes(key))
        assert rec["result"]["artifacts"]
        for a in rec["result"]["artifacts"]:
            assert hashlib.sha256(st.get_bytes(a["key"])).hexdigest() == a["sha256"]


@requires_lerobot
def test_snapshots_in_the_bucket_equal_the_direct_run_byte_for_byte(collected):
    from vlar_harness.dataset import snapshot_key
    direct = collected["direct"]["snaps"]
    assert direct
    for d, data in direct.items():
        assert collected["storage"].get_bytes(snapshot_key(f"base/t{TASK}/i{INIT}/dp{d:02d}")) == data, d


@requires_lerobot
def test_runs_steps_and_events_equal_the_direct_run(collected):
    runs = {r["run_id"]: r for r in table(collected, "runs")}
    base = runs[f"base/t{TASK}/i{INIT}"]
    direct = collected["direct"]
    assert (base["run_kind"], base["record"], base["steps"], base["schema_version"]) == ("episode", "full", 520, 1)
    assert base["config_hash"] == CFG.hash
    steps = [s for s in table(collected, "steps") if s["run_id"] == base["run_id"]]
    assert len(steps) == len(direct["trajectory"]) == 520
    for s, r in zip(steps, direct["trajectory"]):
        assert s["step"] == r["step"] and s["driver"] == r["driver"]
        assert np.array_equal(np.asarray(s["action"], np.float32), r["action"])
        assert np.array_equal(np.asarray(s["eef_pos"]), r["robot0_eef_pos"])
    events = [e for e in table(collected, "events") if e["run_id"] == base["run_id"]]
    assert [json.loads(e["event"]) for e in sorted(events, key=lambda e: e["seq"])] == direct["events"]


@requires_lerobot
def test_points_equal_the_direct_runs_measurements(collected):
    pts = sorted((p for p in table(collected, "points") if p["run_id"] == f"base/t{TASK}/i{INIT}"),
                 key=lambda p: p["index"])
    ms = collected["direct"]["measurements"]
    assert [p["index"] for p in pts] == [m["index"] for m in ms] and len(pts) == 52
    for p, m in zip(pts, ms):
        assert np.array_equal(np.asarray(p["features_main"], np.float32), m["features_main"])
        assert np.array_equal(np.asarray(p["proposal"], np.float32).reshape(10, 7), m["proposal"])
        assert np.array_equal(np.asarray(p["robot_state"]), m["robot_state"])
        if m["chunk_samples_raw"] is not None:
            assert np.array_equal(np.asarray(p["chunk_samples_raw"], np.float32).reshape(m["chunk_samples_raw"].shape),
                                  m["chunk_samples_raw"])
        else:
            assert p["chunk_samples_raw"] is None
        assert (p["probe_raw"] is None) == (m["probe_raw"] is None)
        assert p["drift_pos_dist_mean"] == m["drift"]["pos_dist_mean"]
        assert p["chunk_spread_rot_dev_mean"] == m["chunk_spread"]["rot_dev_mean"]
        assert p["prev_len"] == len(m["prev_segment"]) and p["tracking_valid"] == (p["index"] > 0)
        assert p["grip_closed_shut_frac"] is not None             # thresholds set for the test


@requires_lerobot
def test_images_are_the_observations_at_each_point(collected, sim):
    from vlar_harness import snapshot as snap
    from vlar_harness.dataset import read_images
    imgs = read_images(collected["root"], f"base/t{TASK}/i{INIT}")
    assert len(imgs) == 52 * 4
    for d in (0, 13, 51):
        snap.restore(snap.from_bytes(collected["direct"]["snaps"][d]), sim, None)
        for cam in sim.camera_keys:
            assert np.array_equal(imgs[f"dp{d:02d}/{cam}"], sim.obs[cam]), (d, cam)


@requires_lerobot
def test_exploration_escalates_exactly_where_its_seed_says(collected):
    from vlar_harness import seeding
    run = next(r for r in table(collected, "runs") if r["run_id"].startswith("explore/"))
    events = [json.loads(e["event"]) for e in table(collected, "events") if e["run_id"] == run["run_id"]]
    offered = [e["index"] for e in events if e["t"] == "decision_point" and e["can_escalate"]]
    chosen = {e["index"] for e in events if e["t"] == "decision" and e["choice"] == "escalate"}
    seed = seeding.stream_seed(run["rollout_seed"], "explore")
    assert chosen == {d for d in offered if seeding.uniform(seed, f"dp{d}") < P_ESCALATE} and chosen
    assert run["pi05_requests"] >= len(chosen)


@requires_lerobot
def test_a_branch_job_continues_the_bucket_snapshot(collected):
    runs = {r["run_id"]: r for r in table(collected, "runs")}
    row = runs[f"{collected['sid']}/b2_one_call/s1"]
    br = collected["direct_branch"]
    assert row["run_kind"] == "branch" and row["record"] == "off" and row["pi05_calls"] == 1
    assert row["rollout_seed"] == br.rollout_seed and row["parent_rollout_seed"] == 1_000_000 + INIT
    steps = [s for s in table(collected, "steps") if s["run_id"] == row["run_id"]]
    assert len(steps) == len(br.trajectory) == 520 - 10 * BRANCH_DP
    assert all(np.array_equal(np.asarray(s["action"], np.float32), r["action"]) for s, r in zip(steps, br.trajectory))
    assert not [p for p in table(collected, "points") if p["run_id"] == row["run_id"]]
