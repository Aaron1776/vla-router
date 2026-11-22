"""Rendering only the observations that are used (SPEC §12, amendment 7) against always-on rendering.

The approval condition is that limiting rendering passes the same exact-equality tests as always-on
rendering. The other reference tests run under the config's default (`render: used`); this file compares
the two modes directly. One scripted routed episode (task 2: the stove moves with the env seed and has a
burner light; X-VLA chunks, mid-chunk points and π0.5 segments in both handoff directions) runs to the
step limit once with `render: used` and once with `render: all`, with measurement calls and a snapshot at
every decision point. At every step the executed action, MuJoCo's integration state, the success flag,
every non-camera observation and every observable's sampling timer must be identical; at every decision
point the camera images, both models' requests and the snapshot file must be identical byte for byte.
`used` must render only on the steps whose observation a decision point reads, and nothing else may
expose an image from a step that did not render.
"""

import time

import pytest

np = pytest.importorskip("numpy")

from conftest import requires_lerobot
from episode_driver import decision_requests
from vlar_harness.config import load_config

CFG = load_config()
TASK = 2
INIT = 3
EPISODE_SEED = 777
ENV_SEED = 145
UNTIL = CFG.step_limit
INTERVAL = CFG.interval
TIMER_FIELDS = ("_time_since_last_sample", "_sampled", "_current_delay")


def schedule(d: int) -> str:
    return "pi05" if d % 7 in (2, 3, 5) else "xvla"


def _sim(render: str):
    from vlar_harness.sim.libero_sim import LiberoSim, SimSpec
    return LiberoSim(TASK, SimSpec.from_config(load_config(render=render)))


def _driver(sim, **kw):
    from episode_driver import Driver, FakePI05, FakeXVLA
    from vlar_harness.adapters.pi05 import PI05Adapter
    from vlar_harness.adapters.xvla import XVLAAdapter
    xa, pa = XVLAAdapter(FakeXVLA()), PI05Adapter(sim)
    return Driver(sim, xa, pa, schedule, FakeXVLA(), FakePI05(), **kw), xa, pa


def _timers(sim) -> dict:
    inner = sim.env.env
    out = {name: tuple(getattr(o, f) for f in TIMER_FIELDS) for name, o in inner._observables.items()}
    out["env"] = (inner.timestep, inner.cur_time)
    return out


def _run(render: str) -> dict:
    """The scripted episode to the step limit, counting real camera renders and recording every step."""
    from episode_driver import EpisodeState
    from vlar_harness import snapshot as snap

    sim = _sim(render)
    sim.setup_episode(INIT, env_seed=ENV_SEED)
    inner = sim.env.env
    renders = [0]
    real_render, real_step = inner.sim.render, sim.step

    def counting_render(*a, **k):
        renders[0] += 1
        return real_render(*a, **k)

    timers = []

    def step(a):
        done = real_step(a)
        timers.append(_timers(sim))
        return done

    inner.sim.render, sim.step = counting_render, step
    snaps, requests, holder = {}, {}, {}

    def on_decision(d, ep):
        requests[d] = decision_requests(sim, holder["xa"], holder["pa"])
        snaps[d] = snap.to_bytes(snap.take(sim, holder["xa"], {"episode_seed": ep.episode_seed,
                                                               "prev_driver": ep.prev_driver,
                                                               "pi05_steps": ep.pi05_steps,
                                                               "decision_index": d, "step": sim.steps}))

    drv, xa, pa = _driver(sim, measure=True, on_decision=on_decision)
    holder["xa"], holder["pa"] = xa, pa
    xa.reset()
    t0 = time.perf_counter()
    records = drv.run(EpisodeState(EPISODE_SEED), UNTIL)
    seconds = time.perf_counter() - t0
    del inner.sim.render, sim.step
    return {"sim": sim, "records": records, "timers": timers, "snaps": snaps, "requests": requests,
            "renders": renders[0], "seconds": seconds}


@pytest.fixture(scope="module")
def runs():
    out = {"all": _run("all"), "used": _run("used")}
    yield out
    for r in out.values():
        r["sim"].close()


@requires_lerobot
def test_the_episode_covers_the_whole_horizon_and_both_models(runs):
    recs = runs["all"]["records"]
    assert len(recs) == UNTIL and not any(r.done for r in recs)
    assert {r.driver for r in recs} == {"xvla", "pi05"}
    assert len(runs["all"]["snaps"]) == UNTIL // INTERVAL


@requires_lerobot
def test_used_rendering_renders_only_the_steps_decision_points_read(runs):
    cams = len(runs["used"]["sim"].camera_keys)
    assert cams == 4
    assert runs["all"]["renders"] == cams * UNTIL
    assert runs["used"]["renders"] == cams * (UNTIL // INTERVAL)
    print(f"\n§12 timing, {UNTIL} steps incl. fake models and measurements: "
          f"all {runs['all']['seconds']:.1f} s, used {runs['used']['seconds']:.1f} s")


@requires_lerobot
def test_every_step_is_identical_apart_from_the_unused_images(runs):
    a, u = runs["all"], runs["used"]
    cams = set(u["sim"].camera_keys)
    assert len(a["records"]) == len(u["records"]) == UNTIL
    for ra, ru, ta, tu in zip(a["records"], u["records"], a["timers"], u["timers"]):
        where = f"step {ra.step}"
        assert (ra.step, ra.driver, ra.action, ra.mj, ra.done) == (ru.step, ru.driver, ru.action, ru.mj, ru.done), where
        assert ta == tu, f"{where}: sampling timers differ"
        assert {k: v for k, v in ra.obs.items() if k not in cams} == \
               {k: v for k, v in ru.obs.items() if k not in cams}, f"{where}: non-camera observations differ"
        assert cams <= set(ra.obs)
        if ra.step % INTERVAL == 0:
            assert {k: ru.obs[k] for k in cams} == {k: ra.obs[k] for k in cams}, f"{where}: images differ"
        else:
            assert not cams & set(ru.obs), f"{where}: an unrendered step exposes images"


@requires_lerobot
def test_decision_points_requests_and_snapshots_are_byte_identical(runs):
    a, u = runs["all"], runs["used"]
    assert sorted(a["snaps"]) == sorted(u["snaps"]) == list(range(UNTIL // INTERVAL))
    for d in a["snaps"]:
        assert a["requests"][d] == u["requests"][d], f"dp {d}: model requests differ"
        assert a["snaps"][d] == u["snaps"][d], f"dp {d}: snapshot files differ"


@requires_lerobot
def test_unrendered_steps_refuse_image_reads(runs):
    from vlar_harness.sim.libero_sim import ImagesNotRendered

    sim = runs["used"]["sim"]
    sim.setup_episode(0, env_seed=31)
    drv, xa, pa = _driver(sim)
    sim.set_control_mode("delta")
    noop = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
    for step in range(1, 2 * INTERVAL + 1):
        sim.step(noop)
        if step % INTERVAL:
            assert not sim.images_fresh
            for read in (lambda: xa.observation(sim), pa.raw_observation, sim.require_images):
                with pytest.raises(ImagesNotRendered):
                    read()
        else:
            assert sim.images_fresh
            xa.observation(sim)
            pa.raw_observation()


@requires_lerobot
def test_a_snapshot_between_decision_points_keeps_its_images_hidden(runs):
    """A capture where nothing rendered holds placeholders: restored into an always-on env, they stay
    hidden until the next rendering step, after which both envs agree exactly."""
    from episode_driver import mj_digest, obs_digest
    from vlar_harness import snapshot as snap
    from vlar_harness.sim.libero_sim import ImagesNotRendered

    used, every = runs["used"]["sim"], runs["all"]["sim"]
    used.setup_episode(1, env_seed=60)
    used.set_control_mode("delta")
    rng = np.random.default_rng(5)
    actions = np.clip(rng.normal(0, 0.4, size=(17, 7)), -1, 1).astype(np.float32)
    for a in actions[:3]:
        used.step(a)
    s = snap.from_bytes(snap.to_bytes(snap.take(used, None, {})))
    assert s.sim["sim"]["images"] is False
    snap.restore(s, every, None)
    assert not every.images_fresh and not set(every.camera_keys) & set(every.obs)
    with pytest.raises(ImagesNotRendered):
        every.require_images()
    for a in actions[3:]:
        used.step(a)
        every.step(a)
        assert mj_digest(used) == mj_digest(every)
        if used.steps % INTERVAL == 0:
            assert obs_digest(used.obs) == obs_digest(every.obs), f"step {used.steps}"
    assert used.steps == 17 and every.images_fresh


@requires_lerobot
@pytest.mark.parametrize("offset", [1, INTERVAL - 1])
def test_a_misplaced_render_step_is_refused_not_silent(offset):
    """Rendering one step late or early leaves the decision point without images: an error, never a
    stale or placeholder image."""
    from episode_driver import EpisodeState
    from vlar_harness.sim.libero_sim import ImagesNotRendered

    sim = _sim("used")
    try:
        sim.setup_episode(INIT, env_seed=ENV_SEED)
        sim._renders = lambda n: n % INTERVAL == offset
        drv, xa, _ = _driver(sim, measure=True)
        xa.reset()
        with pytest.raises(ImagesNotRendered):
            drv.run(EpisodeState(EPISODE_SEED), 2 * INTERVAL)
        assert sim.steps == INTERVAL
    finally:
        sim.close()


@requires_lerobot
def test_a_rendering_step_that_misses_its_sample_is_refused(runs):
    """If a camera's sampling phase moved so that a rendering step did not sample it exactly once, the
    step raises instead of leaving the previous (placeholder) value as the decision point's image."""
    from vlar_harness.sim.libero_sim import RenderPhaseError

    sim = runs["used"]["sim"]
    sim.setup_episode(0, env_seed=31)
    noop = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
    for _ in range(INTERVAL - 1):
        sim.step(noop)
    cam = sim.env.env._observables["agentview_image"]
    cam._sampled, cam._time_since_last_sample = True, 0.0      # no sample until the step's last substep
    with pytest.raises(RenderPhaseError):
        sim.step(noop)
