"""π0.5 adapter: the observation LeRobot's `lerobot-eval` hands to its processor chain, built by
LeRobot's own code (docs/SPEC.md §2-3).

In the reference, `LiberoEnv._format_raw_obs` (lerobot/envs/libero.py:287-337) turns the
robosuite observation into `{"pixels": {image, image2}, "robot_state": {...}}`, and gymnasium's
SyncVectorEnv batches it (leading dimension 1, dtypes from the observation space). The π0.5
server then runs `preprocess_observation`, `LiberoProcessorStep` (180° flip of both images,
quaternion -> axis-angle, 8-value state), the checkpoint's own pre-processor, the policy and the
post-processor.

Here a LeRobot `LiberoEnv` is constructed but never reset, so it never builds a simulator of
its own: it is used only for `_format_raw_obs` and its observation space. Its `_env` points at
the harness's env, and its camera names point at the 360 px observables (`sim.libero_sim`),
so the images are exactly what LeRobot's env (built at 360 px) would have returned.

Actions come back un-normalised, 7 values each, and go to env.step unchanged as float32, as in
`lerobot-eval` (the LIBERO env post-processor is empty). The controller must be in delta mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sim.libero_sim import CAMERAS, camera_key

LEROBOT_CAMERA_NAMES = ("image", "image2")   # LiberoEnv's default mapping: agentview -> image, wrist -> image2


class PI05Adapter:
    def __init__(self, sim):
        from gymnasium.vector.utils import batch_space
        from lerobot.envs.libero import LiberoEnv as LeRobotLiberoEnv

        px = sim.spec.extra_camera_px
        cams = [camera_key(c, px) for c in CAMERAS]
        self._lr = LeRobotLiberoEnv(
            task_suite=sim.suite,
            task_id=sim.task_id,
            task_suite_name=sim.spec.suite,
            camera_name=cams,
            camera_name_mapping=dict(zip(cams, LEROBOT_CAMERA_NAMES)),
            obs_type="pixels_agent_pos",
            observation_width=px,
            observation_height=px,
            init_states=False,          # never resets, so it never needs them
            control_mode="relative",
        )
        self._lr._env = sim.env          # _format_raw_obs reads the controller's ee_ori_mat from here
        self._space = self._lr.observation_space
        self._batched_space = batch_space(self._space, 1)
        self.task_description = self._lr.task_description
        self.sim = sim

    def raw_observation(self) -> dict[str, Any]:
        """The batched observation `lerobot-eval` passes to `preprocess_observation` at this step."""
        from gymnasium.vector.utils import concatenate, create_empty_array

        single = self._lr._format_raw_obs(self.sim.obs)
        out = create_empty_array(self._space, n=1, fn=np.zeros)
        return concatenate(self._space, [single], out)

    @staticmethod
    def env_action(action_row: np.ndarray) -> np.ndarray:
        """One un-normalised π0.5 action as LiberoEnv.step receives it from SyncVectorEnv."""
        a = np.asarray(action_row, dtype=np.float32)
        if a.shape != (7,):
            raise ValueError(f"expected a 7-value action, got {a.shape}")
        return a
