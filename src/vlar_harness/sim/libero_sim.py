"""LIBERO simulator shared by both model adapters (docs/SPEC.md §3).

One episode-setup routine for every run mode, identical to both reference harnesses
(X-VLA's `LIBEROEval._init_env`, libero_client.py:245-278, and LeRobot's `LiberoEnv.reset`,
lerobot/envs/libero.py:339-364):

    env.seed(s); env.reset(); env.set_init_state(init_states[k]);
    10 x env.step([0, 0, 0, 0, 0, 0, -1])        # controller still in delta mode

then the clock starts. The controller mode is set by whichever model drives next.

Cameras. robosuite samples every observable (camera images and the proprio observables such
as `robot0_eef_pos`) once per env.step, at physics substep 24 of 25 in both references, i.e.
one substep before the state the step leaves behind (docs/HARNESS.md, "Observation timing").
A render taken after env.step returns would therefore not match what either model was
evaluated on. So each model's images come from robosuite observables:

- the env is built with its cameras at X-VLA's 256 px, exactly as the X-VLA client builds it;
- π0.5's 360 px images come from two extra observables using the same sensor code as
  robosuite's own camera observables (robot_env.py `_create_camera_sensors`). They are added
  once, right after the env is built; `_reset_internal` resets every observable in the env,
  so their sampling timers stay in lockstep with the native ones across resets.

Both render into the same 640x480 offscreen buffer, so an image rendered at a given size is the
image the corresponding reference harness produced (checked in tests/reference).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

CAMERAS = ("agentview", "robot0_eye_in_hand")
CONTROL_MODES = ("absolute", "delta")


def camera_key(camera: str, px: Optional[int] = None) -> str:
    """Observation key: robosuite's own name for the env cameras, `<name>_<px>` for the extra ones."""
    return f"{camera}_image" if px is None else f"{camera}_image_{px}"


@dataclass(frozen=True)
class SimSpec:
    suite: str = "libero_10"
    env_camera_px: int = 256           # X-VLA's native size: the env's own camera observables
    extra_camera_px: int = 360         # π0.5's native size: extra observables
    control_freq: int = 20
    settle_steps: int = 10
    settle_action: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)

    @staticmethod
    def from_config(cfg) -> "SimSpec":
        env = cfg["env"]
        return SimSpec(
            suite=env["suite"],
            env_camera_px=int(cfg["models"]["xvla"]["image_px"]),
            extra_camera_px=int(cfg["models"]["pi05"]["render_px"]),
            control_freq=int(round(1.0 / float(env["control_period_s"]))),
            settle_steps=int(env["settle_steps"]),
            settle_action=tuple(float(x) for x in env["settle_action"]),
        )


class LiberoSim:
    """One LIBERO-Long task in one process. Reused across episodes of that task (hard reset)."""

    def __init__(self, task_id: int, spec: SimSpec = SimSpec()):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.spec = spec
        self.suite = benchmark.get_benchmark_dict()[spec.suite]()
        task = self.suite.get_task(task_id)
        self.task_id = task_id
        self.task_name = task.name
        self.task_description = task.language
        self.bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        # LIBERO's own loader, as the X-VLA client uses (LeRobot reads the same file with torch.load).
        self.init_states = self.suite.get_task_init_states(task_id)
        # Same arguments as both references (X-VLA passes only the bddl file and camera size; the
        # other two are ControlEnv's defaults, which LeRobot passes explicitly).
        self.env = OffScreenRenderEnv(
            bddl_file_name=self.bddl_file,
            camera_heights=spec.env_camera_px,
            camera_widths=spec.env_camera_px,
            control_freq=spec.control_freq,
            hard_reset=True,
        )
        self._add_extra_cameras(spec.extra_camera_px)
        self.obs: Optional[dict] = None
        self.steps = 0
        self.done = False

    # ------------------------------------------------------------------ setup

    def _add_extra_cameras(self, px: int) -> None:
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        from robosuite.utils.observables import Observable, sensor

        inner = self.env.env
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        for cam in CAMERAS:
            # Mirrors robot_env.py `camera_rgb` (depth off); looks up `inner.sim` at call time because
            # a hard reset replaces the MjSim object.
            def make_sensor(cam_name: str):
                @sensor(modality="image")
                def camera_rgb(obs_cache):
                    img = inner.sim.render(camera_name=cam_name, width=px, height=px, depth=False)
                    return img[::convention]
                return camera_rgb

            inner.add_observable(Observable(name=camera_key(cam, px), sensor=make_sensor(cam),
                                            sampling_rate=inner.control_freq))

    def setup_episode(self, init_idx: int, env_seed: int) -> dict:
        """Build the scene, load initial state `init_idx`, settle. Returns the observation at step 0."""
        env = self.env
        env.seed(env_seed)                       # np.random.seed: only object placement, which the init state overrides
        env.reset()
        env.set_init_state(self.init_states[init_idx])
        obs = None
        settle = np.array(self.spec.settle_action, dtype=np.float64)
        for _ in range(self.spec.settle_steps):
            obs, _, _, _ = env.step(settle)
        self.obs, self.steps, self.done = obs, 0, False
        return obs

    # ------------------------------------------------------------------ control

    def set_control_mode(self, mode: str) -> None:
        """`absolute` for X-VLA, `delta` for π0.5, as each reference sets `controller.use_delta`."""
        if mode not in CONTROL_MODES:
            raise ValueError(f"mode must be one of {CONTROL_MODES}")
        for robot in self.env.robots:
            robot.controller.use_delta = mode == "delta"

    def control_mode(self) -> str:
        return "delta" if self.env.robots[0].controller.use_delta else "absolute"

    def step(self, action: np.ndarray) -> bool:
        """One environment step. Returns True once the task is done (LIBERO's success check)."""
        obs, _, done, _ = self.env.step(action)
        self.obs = obs
        self.steps += 1
        self.done = bool(done)
        return self.done

    def controller_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """The controller's cached end-effector pose, which is what the X-VLA client reads
        (libero_client.py:295-296). It is updated at the start of each physics substep, so after
        env.step it is the pose at the start of substep 25."""
        c = self.env.robots[0].controller
        return c.ee_pos, c.ee_ori_mat

    def check_success(self) -> bool:
        return bool(self.env.check_success())

    def close(self) -> None:
        self.env.close()


def describe(sim: LiberoSim) -> dict[str, Any]:
    """Facts worth recording with every episode."""
    inner = sim.env.env
    return {
        "task_id": sim.task_id,
        "task_name": sim.task_name,
        "task_description": sim.task_description,
        "model_timestep": inner.model_timestep,
        "control_timestep": inner.control_timestep,
        "substeps": int(inner.control_timestep / inner.model_timestep),
        "horizon": inner.horizon,
        "env_camera_px": sim.spec.env_camera_px,
        "extra_camera_px": sim.spec.extra_camera_px,
    }
