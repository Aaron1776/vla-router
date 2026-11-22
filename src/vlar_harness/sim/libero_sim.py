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

Rendering only what is used (SPEC §12, amendment 7; config option `render`). Models, measurements
and snapshots read images only at decision points, i.e. the observation left by the step that ends
on a multiple of the decision interval (and by the last settle step, for decision point 0). Under
`decision_points` every camera observable keeps its robosuite sampling timer and sampling logic
unchanged; only its sensor is gated: on the steps whose observation is used it renders, on all
others it returns a placeholder without rendering. The placeholder never reaches a reader: on those
steps `obs` has no camera keys (`require_images` raises), and snapshots record whether their
buffered images are real renders. Each rendering step must sample every camera exactly once, or the
step raises `RenderPhaseError`. `every_step` renders every step, as both reference harnesses do.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

CAMERAS = ("agentview", "robot0_eye_in_hand")
CONTROL_MODES = ("absolute", "delta")
RENDER_MODES = ("decision_points", "every_step")


class ImagesNotRendered(RuntimeError):
    """Camera images were read from a step whose observation is not rendered (§12)."""


class RenderPhaseError(RuntimeError):
    """A rendering step did not sample every camera exactly once: the sampling phase moved."""


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
    render: str = "decision_points"    # §12: `decision_points` or `every_step` (config option `render`)
    decision_interval: int = 10        # decision points at steps 0, 10, 20, ... after the clock starts

    def __post_init__(self):
        if self.render not in RENDER_MODES:
            raise ValueError(f"render must be one of {RENDER_MODES}, not {self.render!r}")
        if self.decision_interval < 1 or self.settle_steps < 1:
            raise ValueError("decision_interval and settle_steps must be >= 1")

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
            render=str(cfg.option("render")),
            decision_interval=int(cfg["decision"]["interval_steps"]),
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
        self._gate = _CameraGate()
        self._add_extra_cameras(spec.extra_camera_px)
        inner = self.env.env
        self.camera_keys = tuple(sorted(k for k, o in inner._observables.items() if o.modality == "image"))
        expected = sorted(camera_key(c, px) for c in CAMERAS for px in (None, spec.extra_camera_px))
        if list(self.camera_keys) != expected:
            raise RuntimeError(f"camera observables {self.camera_keys}, expected {expected}")
        self._hidden = frozenset(self.camera_keys) | {"image-state"}
        self._gate.renders = dict.fromkeys(self.camera_keys, 0)
        self._install_gates()
        self.obs: Optional[dict] = None
        self._images_fresh = False
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

    def _install_gates(self) -> None:
        """Wrap every camera observable's sensor in the gate (again after a hard reset, which gives the
        env's own cameras fresh sensors). Timers, sampling logic and the other observables are untouched."""
        gate, observables = self._gate, self.env.env._observables
        for key in self.camera_keys:
            o = observables[key]
            if getattr(o._sensor, "_vlar_gate", None) is gate:
                continue
            o.set_sensor(_gated_sensor(gate, key, o._sensor, np.zeros(o._data_shape, dtype=np.uint8)))

    def _renders(self, steps_after: int) -> bool:
        """Is the observation left at `steps_after` (steps since the clock started) rendered?"""
        return self.spec.render == "every_step" or steps_after % self.spec.decision_interval == 0

    def _env_step(self, action: np.ndarray, render: bool, where: str) -> tuple[dict, bool]:
        self._install_gates()
        gate = self._gate
        for k in gate.renders:
            gate.renders[k] = 0
        gate.open = render
        try:
            obs, _, done, _ = self.env.step(action)
        finally:
            gate.open = False
        if render and any(n != 1 for n in gate.renders.values()):
            raise RenderPhaseError(f"{where}: renders per camera {gate.renders}, expected 1 each")
        self._publish(obs, render)
        return obs, bool(done)

    def _publish(self, obs: dict, images_fresh: bool) -> None:
        self._images_fresh = bool(images_fresh)
        self.obs = obs if self._images_fresh else {k: v for k, v in obs.items() if k not in self._hidden}

    def setup_episode(self, init_idx: int, env_seed: int) -> dict:
        """Build the scene, load initial state `init_idx`, settle. Returns the observation at step 0."""
        env = self.env
        every = self.spec.render == "every_step"
        self._gate.open = every                  # forced observation updates render only under `every_step`
        try:
            env.seed(env_seed)                   # np.random.seed: only object placement, which the init state overrides
            env.reset()
            self._install_gates()
            env.set_init_state(self.init_states[init_idx])
        finally:
            self._gate.open = False
        self.steps = 0
        settle = np.array(self.spec.settle_action, dtype=np.float64)
        n = self.spec.settle_steps
        for i in range(n):
            self._env_step(settle, every or i == n - 1, f"settle step {i + 1} of {n}")
        self.done = False
        return self.obs

    @property
    def images_fresh(self) -> bool:
        """True when `obs` holds the camera images sampled during the step that produced it."""
        return self._images_fresh

    def require_images(self) -> None:
        """Called by the adapters before reading camera images."""
        if not self._images_fresh:
            raise ImagesNotRendered(f"no camera images at step {self.steps}: with render={self.spec.render!r} "
                                    f"only every {self.spec.decision_interval}th step's observation is rendered")

    def after_restore(self, obs: dict, images_fresh: bool) -> None:
        """`sim/state.restore` hands back the observation rebuilt from the restored buffers."""
        self._install_gates()
        self._publish(obs, images_fresh)

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
        _, done = self._env_step(action, self._renders(self.steps + 1), f"step {self.steps + 1}")
        self.steps += 1
        self.done = done
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
        "render": sim.spec.render,
        "decision_interval": sim.spec.decision_interval,
    }


class _CameraGate:
    """Open only while a step whose observation is used runs; counts real renders per camera."""

    def __init__(self):
        self.open = False
        self.renders: dict[str, int] = {}


def _gated_sensor(gate: _CameraGate, key: str, real, placeholder: np.ndarray):
    """A robosuite sensor that renders through `real` only while the gate is open. A plain function, so
    the snapshot tests' object-graph walk does not descend into it."""

    def camera(obs_cache):
        if gate.open:
            gate.renders[key] += 1
            return real(obs_cache)
        return placeholder

    camera.__modality__ = real.__modality__
    camera._vlar_gate = gate
    return camera
