"""Snapshot restore-and-replay and the determinism tests of SPEC §11 step 3 (effort step 2).

One scripted episode (task 2: its stove moves with the env seed and has a burner light) mixes
X-VLA chunks and π0.5 segments. A snapshot is taken at every decision point, through the on-disk
format. Each snapshot is then restored into three targets and the rest of the episode replayed:
  A  the same env, after the original episode ended;
  B  another env of the same task that ran a different episode with a different env seed
     (so its stove sits elsewhere);
  C  a freshly built env that has never been reset.
Every step of every replay must equal the original exactly: all observation values (both camera
sizes, all proprio and object observations), MuJoCo's full integration state, the executed
action and the success flag. Sensitivity checks show the comparison catches each piece of state
a restore could miss.
"""

import copy

import pytest

np = pytest.importorskip("numpy")

from conftest import requires_lerobot
from vlar_harness.config import load_config

CFG = load_config()
TASK = 2
INIT = 3
EPISODE_SEED = 777
ENV_SEED = 145
UNTIL = 130
REPLAY_FROM = (1, 2, 3, 4, 5, 6, 7, 10, 12)   # mid-chunk, chunk points, before/after π0.5, both directions


def schedule(d: int) -> str:
    return "pi05" if d in (4, 5, 9) else "xvla"


def _spec():
    from vlar_harness.sim.libero_sim import SimSpec
    return SimSpec.from_config(CFG)


def _adapters(sim):
    from episode_driver import FakeXVLA
    from vlar_harness.adapters.pi05 import PI05Adapter
    from vlar_harness.adapters.xvla import XVLAAdapter
    return XVLAAdapter(FakeXVLA()), PI05Adapter(sim)


def _driver(sim, **kw):
    from episode_driver import Driver, FakePI05, FakeXVLA
    xa, pa = _adapters(sim)
    return Driver(sim, xa, pa, schedule, FakeXVLA(), FakePI05(), **kw), xa


def decision_requests(sim, xa, pa) -> dict:
    """Digests of what each model would be sent right now: the X-VLA measurement request (images +
    measured pose from the controller's cache) and π0.5's batched raw observation."""
    from episode_driver import digest
    payload = xa.measurement_payload(sim, sim.task_description)
    raw = pa.raw_observation()
    flat = []

    def feed(x):
        if isinstance(x, dict):
            for k in sorted(x):
                feed(x[k])
        else:
            flat.append(np.ravel(np.asarray(x)).view(np.uint8))

    feed(raw)
    return {"xvla": digest(np.frombuffer(repr(sorted(payload.items())).encode(), dtype=np.uint8)),
            "pi05": digest(np.concatenate(flat))}


def _same(a, b):
    """Records equal; on failure, name the first differing step and field."""
    for ra, rb in zip(a, b):
        if ra != rb:
            fields = [f for f in ("step", "driver", "action", "mj", "done") if getattr(ra, f) != getattr(rb, f)]
            keys = sorted(k for k in set(ra.obs) | set(rb.obs) if ra.obs.get(k) != rb.obs.get(k))
            return f"step {ra.step}: differs in {fields} and observations {keys[:6]}"
    if len(a) != len(b):
        return f"lengths differ: {len(a)} vs {len(b)}"
    return None


@pytest.fixture(scope="module")
def original():
    """The scripted episode with a snapshot (and an object-graph walk) at every decision point."""
    import statewalk
    from episode_driver import EpisodeState
    from vlar_harness import snapshot as snap
    from vlar_harness.sim.libero_sim import LiberoSim
    from vlar_harness.sim.state import model_digest

    sim = LiberoSim(TASK, _spec())
    sim.setup_episode(INIT, env_seed=ENV_SEED)
    snaps, walks, sizes, requests = {}, {}, {}, {}
    holder = {}

    def on_decision(d, ep):
        requests[d] = decision_requests(sim, holder["xa"], holder["pa"])
        s = snap.take(sim, holder["xa"], {"episode_seed": ep.episode_seed, "prev_driver": ep.prev_driver,
                                          "pi05_steps": ep.pi05_steps, "decision_index": d, "step": sim.steps})
        data = snap.to_bytes(s)
        sizes[d] = len(data)
        snaps[d] = data
        walks[d] = (statewalk.walk(sim.env.env), model_digest(sim.env.env.sim.model._model))

    drv, xa = _driver(sim, measure=True, on_decision=on_decision)
    holder["xa"], holder["pa"] = xa, drv.pa
    xa.reset()
    records = drv.run(EpisodeState(EPISODE_SEED), UNTIL)
    yield {"sim": sim, "records": records, "snaps": snaps, "walks": walks, "sizes": sizes, "requests": requests}
    sim.close()


@pytest.fixture(scope="module")
def other_env():
    """Target B: same task, a different episode with a different env seed (stove elsewhere)."""
    from episode_driver import EpisodeState
    from vlar_harness.sim.libero_sim import LiberoSim

    sim = LiberoSim(TASK, _spec())
    sim.setup_episode(7, env_seed=9999)
    drv, xa = _driver(sim)
    xa.reset()
    drv.run(EpisodeState(4242), 55)
    yield sim
    sim.close()


@pytest.fixture(scope="module")
def fresh_env():
    """Target C: built, never reset."""
    from vlar_harness.sim.libero_sim import LiberoSim
    sim = LiberoSim(TASK, _spec())
    yield sim
    sim.close()


def replay(snapshot_bytes, sim, *, tamper=None, after_restore=None):
    """`tamper(snapshot, sim)` edits the snapshot before restore; `after_restore(snapshot, sim)` runs after it."""
    from episode_driver import EpisodeState
    from vlar_harness import snapshot as snap

    s = snap.from_bytes(snapshot_bytes)
    if tamper is not None:
        tamper(s, sim)
    drv, xa = _driver(sim)
    episode = snap.restore(s, sim, xa)
    if after_restore is not None:
        after_restore(s, sim)
    ep = EpisodeState(episode["episode_seed"], episode["prev_driver"], episode["pi05_steps"])
    return drv.run(ep, UNTIL), episode


# ---------------------------------------------------------------- the episode itself

@requires_lerobot
def test_scripted_episode_exercises_what_snapshots_must_carry(original):
    recs = original["records"]
    assert len(recs) == UNTIL and not any(r.done for r in recs)
    assert {r.driver for r in recs} == {"xvla", "pi05"}
    assert len(original["snaps"]) == UNTIL // 10
    # A snapshot holds four camera images; the file stays small enough to keep one per decision point.
    assert max(original["sizes"].values()) < 2_000_000


@requires_lerobot
def test_snapshot_file_roundtrip_is_exact(original):
    from vlar_harness import snapshot as snap

    data = original["snaps"][3]
    s1 = snap.from_bytes(data)
    assert snap.to_bytes(s1) == data                       # same bytes after a decode/encode cycle

    def same(a, b, path="snap"):
        assert type(a) is type(b), f"{path}: {type(a).__name__} != {type(b).__name__}"
        if isinstance(a, dict):
            assert list(a) == list(b), path
            for k in a:
                same(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                same(x, y, f"{path}[{i}]")
        elif isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), path
        else:
            assert a == b or (a != a and b != b), path

    same(s1.as_tree(), snap.from_bytes(snap.to_bytes(s1)).as_tree())


# ---------------------------------------------------------------- replay

@requires_lerobot
@pytest.mark.parametrize("target", ["same_env", "other_env", "fresh_env"])
def test_restore_and_replay_reproduce_the_trajectory_exactly(original, other_env, fresh_env, target):
    import statewalk
    from episode_driver import mj_digest
    from vlar_harness import snapshot as snap
    from vlar_harness.sim.state import model_digest

    sim = {"same_env": original["sim"], "other_env": other_env, "fresh_env": fresh_env}[target]
    for d in REPLAY_FROM:
        # Right after restore, every Python leaf, the compiled model and MuJoCo's state equal capture time.
        s = snap.from_bytes(original["snaps"][d])
        drv, xa = _driver(sim)
        snap.restore(s, sim, xa)
        walk_then, model_then = original["walks"][d]
        missing = statewalk.diff(walk_then, statewalk.walk(sim.env.env))
        assert not missing, f"dp {d}: state not restored: {missing[:8]}"
        assert model_digest(sim.env.env.sim.model._model) == model_then
        assert decision_requests(sim, xa, drv.pa) == original["requests"][d], f"dp {d}: model requests differ"
        # ... and the rest of the episode is identical, step by step.
        records, _ = replay(original["snaps"][d], sim)
        problem = _same(original["records"][d * 10:], records)
        assert problem is None, f"{target}, replay from dp {d}: {problem}"


@requires_lerobot
def test_restore_rejects_another_task(original):
    from vlar_harness import snapshot as snap
    from vlar_harness.sim.libero_sim import LiberoSim
    from vlar_harness.sim.state import StateMismatch

    sim = LiberoSim(0, _spec())
    try:
        with pytest.raises(StateMismatch):
            snap.restore(snap.from_bytes(original["snaps"][3]), sim, None)
    finally:
        sim.close()


# ---------------------------------------------------------------- determinism (§11 step 3)

@requires_lerobot
def test_identical_seeds_reproduce_identical_runs_without_measurements_or_snapshots(original, other_env):
    """Same seeds in a reused env -> the same run. The original took measurements and snapshots at
    every decision point; this run takes neither, so equality also shows they change nothing."""
    from episode_driver import EpisodeState

    sim = other_env
    sim.setup_episode(INIT, env_seed=ENV_SEED)
    drv, xa = _driver(sim, measure=False)
    xa.reset()
    records = drv.run(EpisodeState(EPISODE_SEED), UNTIL)
    assert _same(original["records"], records) is None


# ---------------------------------------------------------------- sensitivity: each missing piece is caught

def _replay_differs(original, sim, d, tamper=None, after_restore=None):
    records, _ = replay(original["snaps"][d], sim, tamper=tamper, after_restore=after_restore)
    return _same(original["records"][d * 10:], records)


@requires_lerobot
@pytest.mark.parametrize("d", [6, 7])
def test_the_old_restore_definition_is_caught(original, other_env, d):
    """SPEC §7 before the amendment: set_init_state from the flattened state, re-rendered observations
    (here applied on top of an otherwise complete restore, so it is the only difference)."""

    def old_way(s, sim):
        sim.obs = sim.env.set_init_state(s.sim["flat_state"])   # forced observation update + re-render

    assert _replay_differs(original, other_env, d, after_restore=old_way) is not None


@requires_lerobot
@pytest.mark.parametrize("piece", ["fixture_placement", "gripper", "observation_timers"])
def test_each_piece_of_state_matters(original, other_env, piece):
    """Restore everything except one piece (keep the target env's own value) -> the replay diverges."""
    from vlar_harness.sim import state as sim_state

    def tamper(s, sim):
        if piece == "fixture_placement":
            sim.setup_episode(7, env_seed=9999)               # the target's own stove position
        current = sim_state.capture(sim)
        st = s.sim
        if piece == "fixture_placement":
            assert not np.array_equal(current["model_fields"]["body_pos"], st["model_fields"]["body_pos"])
            st["model_fields"] = current["model_fields"]
        elif piece == "gripper":
            # A stale integrated gripper command, e.g. a gripper left open in an env whose snapshot
            # gripper is closing (or the reverse).
            saved = st["robots"][0]["gripper"]["current_action"]
            stale = -np.sign(saved) * np.maximum(np.abs(saved), 1.0)
            assert not np.array_equal(stale, saved)
            st["robots"][0]["gripper"]["current_action"] = stale.astype(saved.dtype)
        elif piece == "observation_timers":
            for name, saved in st["observables"].items():
                saved["_time_since_last_sample"] = current["observables"][name]["_time_since_last_sample"] + 0.002

    d = 6 if piece != "gripper" else 4
    assert _replay_differs(original, other_env, d, tamper) is not None, f"missing {piece} went unnoticed"


@requires_lerobot
def test_missing_controller_state_changes_the_model_requests(original, other_env):
    """robosuite recomputes the controller's cached pose at the next step and absolute mode resets its
    goals every step, so a missing controller restore shows up in what is read before stepping: the
    measured pose in X-VLA's /sample and /features requests and π0.5's observation."""
    from vlar_harness import snapshot as snap
    from vlar_harness.sim import state as sim_state

    d = 7
    other_env.setup_episode(7, env_seed=9999)
    stale = sim_state.capture(other_env)["robots"][0]["controller"]
    s = snap.from_bytes(original["snaps"][d])
    s.sim["robots"][0]["controller"] = stale
    drv, xa = _driver(other_env)
    snap.restore(s, other_env, xa)
    got = decision_requests(other_env, xa, drv.pa)
    assert got["xvla"] != original["requests"][d]["xvla"]
    assert got["pi05"] != original["requests"][d]["pi05"]


@requires_lerobot
def test_stove_burner_light_survives_restore(other_env, fresh_env):
    """Task 2's burner glow is a compiled-model colour LIBERO toggles each step; it must come back."""
    from episode_driver import FakePI05, obs_digest
    from vlar_harness import snapshot as snap
    from vlar_harness.adapters.pi05 import PI05Adapter
    from vlar_harness.sim.libero_sim import LiberoSim

    sim = LiberoSim(TASK, _spec())
    try:
        sim.setup_episode(0, env_seed=31)
        sim.set_control_mode("delta")
        inner = sim.env.env
        inner.object_states_dict["flat_stove_1"].set_joint(1.0)     # knob into the "on" range
        noop = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
        for _ in range(3):
            sim.step(noop)
        site = inner.sim.model.site_name2id("flat_stove_1_burner")
        assert inner.sim.model.site_rgba[site][3] > 0, "burner did not light up"
        s = snap.from_bytes(snap.to_bytes(snap.take(sim, None, {})))
        pi05 = PI05Adapter(sim)
        fake = FakePI05()
        actions = fake(pi05.raw_observation(), sim.task_description, 1)[:15]
        ref = []
        for a in actions:
            sim.step(pi05.env_action(a))
            ref.append(obs_digest(sim.obs))
        for target in (other_env, fresh_env):
            snap.restore(s, target, None)
            assert target.env.env.sim.model.site_rgba[site][3] > 0
            got = []
            for a in actions:
                target.step(PI05Adapter.env_action(a))
                got.append(obs_digest(target.obs))
            assert got == ref
    finally:
        sim.close()


@requires_lerobot
@pytest.mark.parametrize("d", [3, 6, 10])
def test_missing_solver_warm_start_is_caught(original, other_env, d):
    """Restore without MuJoCo's constraint-solver warm start (what restoring only qpos/qvel/time gives)."""

    def drop_warm_start(s, sim):
        sim.env.env.sim.data._data.qacc_warmstart[:] = 0.0

    assert _replay_differs(original, other_env, d, after_restore=drop_warm_start) is not None
