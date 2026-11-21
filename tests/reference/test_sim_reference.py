"""The simulator layer and both adapters against the reference harnesses' own code, step by step.

- X-VLA side vs the authors' `LIBEROEval._init_env` + `ClientModel` (libero_client.py).
- π0.5 side vs LeRobot's own vectorised LIBERO env, as `lerobot-eval` builds it.
Equality is exact: same machine, same calls, so any difference is a real preprocessing difference.
"""

import pytest

np = pytest.importorskip("numpy")

from conftest import requires_lerobot, requires_libero
from vlar_harness import reference
from vlar_harness.config import load_config

CFG = load_config()
TASK = 0


def _spec():
    from vlar_harness.sim.libero_sim import SimSpec
    return SimSpec.from_config(CFG)


def _lc():
    return reference.xvla_libero_client(CFG["reference"]["xvla_repo_files"]["evaluation/libero/libero_client.py"])


def assert_tree_equal(a, b, path="obs"):
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            assert_tree_equal(a[k], b[k], f"{path}.{k}")
    else:
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype, f"{path}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{path}: shape {a.shape} != {b.shape}"
        assert np.array_equal(a, b), f"{path}: max abs diff {np.abs(a.astype(float) - b.astype(float)).max()}"


def sim_state(env):
    return np.asarray(env.sim.get_state().flatten())


# ---------------------------------------------------------------- pins

@requires_libero
def test_xvla_checkout_matches_pins():
    reference.verify_xvla(CFG["reference"]["xvla_repo_files"])


@requires_lerobot
def test_lerobot_install_matches_pins():
    reference.verify_lerobot(CFG["reference"]["lerobot_files"])


# ---------------------------------------------------------------- X-VLA side

@requires_libero
@pytest.mark.parametrize("init_idx", [0, 7])
def test_episode_setup_and_absolute_control_match_xvla_client(init_idx):
    from vlar_harness.sim.libero_sim import LiberoSim

    lc = _lc()
    ev = lc.LIBEROEval(task_suite_name="libero_10", eval_horizon=900, act_type="abs", num_episodes=1, init_seed=42)
    env_ref, lang, obs_ref = ev._init_env(ev.task_suite_list[0], TASK, init_idx)   # seed 42+ep+100, init ep % 50

    sim = LiberoSim(TASK, _spec())
    obs = sim.setup_episode(init_idx, env_seed=42 + init_idx + 100)
    sim.set_control_mode("absolute")
    try:
        assert lang == sim.task_description
        assert np.array_equal(sim_state(env_ref), sim_state(sim.env))
        for key in ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat"):
            assert_tree_equal(obs_ref[key], obs[key], key)
        ref_ctrl, ctrl = env_ref.env.robots[0].controller, sim.env.robots[0].controller
        assert ref_ctrl.use_delta is False and ctrl.use_delta is False
        assert np.array_equal(ref_ctrl.ee_pos, ctrl.ee_pos) and np.array_equal(ref_ctrl.ee_ori_mat, ctrl.ee_ori_mat)

        # Absolute targets like X-VLA's: a slow move away from the start pose, gripper closing halfway.
        proc = lc.LiberoAbsActionProcessor()
        start_pos = np.array(ctrl.ee_pos)
        start_aa = lc.T.quat2axisangle(lc.T.mat2quat(np.array(ctrl.ee_ori_mat)))
        for i in range(20):
            target = np.concatenate([start_pos + np.array([0.004, -0.003, -0.002]) * i, start_aa,
                                     [1.0 if i > 10 else -1.0]]).astype(np.float32)
            obs_ref, _, done_ref, _ = env_ref.step(target)
            done = sim.step(target)
            assert done == done_ref
            assert np.array_equal(sim_state(env_ref), sim_state(sim.env)), f"state differs after step {i + 1}"
            for key in ("agentview_image", "robot0_eye_in_hand_image"):
                assert_tree_equal(obs_ref[key], sim.obs[key], f"step {i + 1} {key}")
        assert proc.Mat_to_Rotate6D(ctrl.ee_ori_mat).shape == (6,)
    finally:
        env_ref.close()
        sim.close()


class _FakeSim:
    def __init__(self, obs, pos, mat):
        self.obs, self._pose = obs, (pos, mat)

    def controller_pose(self):
        return self._pose


@requires_libero
def test_xvla_query_pop_split_matches_client_step():
    """Our query()/pop() split against the authors' ClientModel.step() on the same responses,
    over three chunks so the quirk (predicted pose as the next proprio) is exercised."""
    from vlar_harness.adapters.xvla import XVLAAdapter

    lc = _lc()
    rng = np.random.default_rng(0)
    chunks = [rng.normal(size=(30, 20)) for _ in range(3)]
    goal = "put both the alphabet soup and the tomato sauce in the basket"

    ref = lc.ClientModel.__new__(lc.ClientModel)
    ref.processor = lc.LiberoAbsActionProcessor()
    ref.reset()
    ref_sent, ref_iter = [], iter(chunks)
    ref._post = lambda payload: (ref_sent.append(payload), np.array(next(ref_iter).tolist()))[1]

    ours_sent, ours_iter = [], iter(chunks)
    adapter = XVLAAdapter(lambda payload, seed: (ours_sent.append(payload), {"action": next(ours_iter).tolist()})[1])
    adapter.reset()

    proc = lc.LiberoAbsActionProcessor()
    for i in range(75):
        imgs = {"agentview_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)}
        pos = rng.normal(size=3)
        q = rng.normal(size=4)
        mat = lc.T.quat2mat(q / np.linalg.norm(q))
        obs_ref = dict(imgs, robo_ori=proc.Mat_to_Rotate6D(mat), robo_pos=pos)   # libero_client.py:295-298
        a_ref = ref.step(obs_ref, goal)
        if adapter.queue_len() == 0:
            adapter.query(_FakeSim(imgs, pos, mat), goal)
        a = adapter.pop()
        assert a.dtype == a_ref.dtype and np.array_equal(a, a_ref), f"action {i} differs"
        assert np.array_equal(adapter.client.proprio, ref.proprio)
    assert len(ref_sent) == len(ours_sent) == 3
    for p_ref, p in zip(ref_sent, ours_sent):
        assert p_ref == p


# ---------------------------------------------------------------- π0.5 side

@requires_lerobot
def test_pi05_observation_and_delta_control_match_lerobot_env():
    """Our observation for π0.5 against LeRobot's own vectorised env (built at 360 px), from the
    first reset through 25 delta-mode steps."""
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION

    from vlar_harness.adapters.pi05 import PI05Adapter
    from vlar_harness.sim.libero_sim import LiberoSim

    env_cfg = LiberoEnvConfig(task="libero_10", task_ids=[TASK])
    vec = env_cfg.create_envs(n_envs=1, use_async_envs=False)["libero_10"][TASK]
    obs_ref, _ = vec.reset(seed=[1000], options={NEW_ROLLOUT_OPTION: True})   # init state 0, settle, relative

    sim = LiberoSim(TASK, _spec())
    sim.setup_episode(0, env_seed=1000)
    sim.set_control_mode("delta")
    adapter = PI05Adapter(sim)
    try:
        assert adapter.task_description == list(vec.call("task_description"))[0]
        assert_tree_equal(obs_ref, adapter.raw_observation())
        rng = np.random.default_rng(1)
        for i in range(25):
            a = np.clip(rng.normal(0, 0.4, size=7), -1, 1).astype(np.float32)
            a[-1] = 1.0 if i > 12 else -1.0
            obs_ref, _, term, _, _ = vec.step(a[None])
            done = sim.step(adapter.env_action(a))
            assert bool(term[0]) == done
            assert_tree_equal(obs_ref, adapter.raw_observation(), f"step {i + 1}")
    finally:
        vec.close()
        sim.close()


@requires_libero
def test_extra_cameras_stay_in_lockstep_across_resets():
    """The 360 px observables must sample at the same substep as robosuite's own, in every episode."""
    from vlar_harness.sim.libero_sim import LiberoSim, camera_key

    sim = LiberoSim(TASK, _spec())
    try:
        for ep in range(3):
            sim.setup_episode(ep, env_seed=ep)
            obs = sim.env.env._observables
            ref = obs["robot0_eef_pos"]
            for cam in ("agentview", "robot0_eye_in_hand"):
                o = obs[camera_key(cam, sim.spec.extra_camera_px)]
                assert (o._time_since_last_sample, o._sampled) == (ref._time_since_last_sample, ref._sampled)
            assert sim.obs[camera_key("agentview", 360)].shape == (360, 360, 3)
            assert sim.obs["agentview_image"].shape == (256, 256, 3)
    finally:
        sim.close()


@requires_libero
def test_observations_are_sampled_at_substep_24_of_25():
    """robosuite samples observables one physics substep before the end of env.step (docs/HARNESS.md)."""
    from vlar_harness.sim.libero_sim import LiberoSim, camera_key

    sim = LiberoSim(TASK, _spec())
    try:
        sim.setup_episode(0, env_seed=0)
        sim.set_control_mode("delta")
        inner = sim.env.env
        site = inner.sim.model.site_name2id(sim.env.robots[0].controller.eef_name)
        recorded = []
        real_step = inner.sim.step

        def step():
            real_step()
            recorded.append(np.array(inner.sim.data.site_xpos[site]))

        inner.sim.step = step
        for i in range(5):
            recorded.clear()
            sim.step(np.array([0.6, -0.4, -0.5, 0.0, 0.0, 0.3, -1.0]))
            assert len(recorded) == 25
            hits = [k + 1 for k, r in enumerate(recorded) if np.array_equal(r, sim.obs["robot0_eef_pos"])]
            assert hits == [24], f"step {i}: observation matches substep(s) {hits}"
        inner.sim.step = real_step

        # Hence a render after env.step is not the observation the models see.
        post = inner.sim.render(camera_name="robot0_eye_in_hand", width=360, height=360, depth=False)
        assert not np.array_equal(post, sim.obs[camera_key("robot0_eye_in_hand", 360)])
    finally:
        sim.close()
