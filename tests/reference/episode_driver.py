"""Test scaffolding for the snapshot and determinism tests (effort step 2).

A scripted routed episode with deterministic fake policies. It is not the harness's run loop
(effort steps 3-4); it exercises what a snapshot must carry: X-VLA chunks queued across decision
points (mid-chunk resumes, the quirk pose), π0.5 segments in delta mode, and control passing in
both directions.

The fake policies are pure functions of the exact request (images, pose, text) and the call's seed,
so any difference in a restored observation changes every later action and the replay diverges.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Optional

import json_numpy
import numpy as np

SEGMENT = 10


def call_seed(episode_seed: int, step: int, model: str, call: str, sample: int = 0) -> int:
    """Stand-in for the harness's per-call seed derivation (effort step 5)."""
    h = hashlib.sha256(f"{episode_seed}|{step}|{model}|{call}|{sample}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def digest(a) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes()).hexdigest()[:24]


class FakeXVLA:
    """Chunk = f(request, seed): a slow drift down from the requested pose, gripper from the hash."""

    def __call__(self, payload: dict, seed: Optional[int]) -> dict:
        img0 = json_numpy.loads(payload["image0"])
        img1 = json_numpy.loads(payload["image1"])
        proprio = np.asarray(json_numpy.loads(payload["proprio"]), dtype=np.float64)
        h = hashlib.sha256(img0.tobytes() + img1.tobytes() + proprio.tobytes()
                           + payload["language_instruction"].encode() + str(seed).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        steps = rng.normal(0.0, 0.004, size=(30, 3))
        steps[:, 2] -= 0.004                                    # towards the table and objects
        chunk = np.zeros((30, 20))
        chunk[:, :3] = proprio[:3] + np.cumsum(steps, axis=0)
        chunk[:, 3:9] = proprio[3:9] + rng.normal(0.0, 0.01, size=(30, 6))
        chunk[:, 9] = rng.uniform()
        return {"action": chunk.astype(np.float32).tolist()}    # the server sends float32 values


class FakePI05:
    """50 delta actions = f(raw observation, task, seed)."""

    def __call__(self, raw_obs: dict, task: str, seed: Optional[int]) -> np.ndarray:
        h = hashlib.sha256(task.encode() + str(seed).encode())

        def feed(x):
            if isinstance(x, dict):
                for k in sorted(x):
                    feed(x[k])
            else:
                h.update(np.ascontiguousarray(x).tobytes())

        feed(raw_obs)
        rng = np.random.default_rng(int.from_bytes(h.digest()[:8], "little"))
        actions = rng.uniform(-0.6, 0.6, size=(50, 7))
        actions[:, 2] -= 0.3
        actions[:, 6] = rng.choice([-1.0, 1.0])
        return actions.astype(np.float32)


@dataclass
class StepRecord:
    step: int
    driver: str
    action: str
    obs: dict
    mj: str
    done: bool


@dataclass
class EpisodeState:
    episode_seed: int
    prev_driver: Optional[str] = None
    pi05_steps: int = 0
    xvla_requests: list = field(default_factory=list)
    pi05_requests: list = field(default_factory=list)


def mj_digest(sim) -> str:
    import mujoco
    inner = sim.env.env
    m, d = inner.sim.model._model, inner.sim.data._data
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    s = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, s, spec)
    return digest(s)


def obs_digest(obs: dict) -> dict:
    return {k: digest(v) for k, v in obs.items()}


class Driver:
    """schedule(decision_index) -> 'xvla' or 'pi05'."""

    def __init__(self, sim, xvla_adapter, pi05_adapter, schedule: Callable[[int], str],
                 fake_xvla: FakeXVLA, fake_pi05: FakePI05, measure: bool = False,
                 on_decision: Optional[Callable] = None):
        self.sim, self.xa, self.pa = sim, xvla_adapter, pi05_adapter
        self.schedule = schedule
        self.fake_xvla, self.fake_pi05 = fake_xvla, fake_pi05
        self.measure = measure
        self.on_decision = on_decision
        self.records: list[StepRecord] = []

    def run(self, ep: EpisodeState, until_step: int) -> list[StepRecord]:
        sim, xa = self.sim, self.xa
        goal = sim.task_description
        while sim.steps < until_step and not sim.done:
            step = sim.steps
            d = step // SEGMENT
            driver = self.schedule(d)
            if xa.queue_len() == 0:                              # chunk point: /act before the decision
                xa.query(sim, goal, seed=call_seed(ep.episode_seed, step, "xvla", "act"))
                ep.xvla_requests.append(digest(np.frombuffer(repr(sorted(xa.client.last_payload.items())).encode(),
                                                             dtype=np.uint8)))
            if self.measure:                                     # measurement calls must not change anything
                payload = xa.measurement_payload(sim, goal)
                for k in range(3):
                    self.fake_xvla(payload, call_seed(ep.episode_seed, step, "xvla", "sample", k))
                self.pa.raw_observation()
            if self.on_decision is not None:
                self.on_decision(d, ep)
            if driver == "pi05":
                xa.discard_queue()
                xa.forget_predicted_pose()                       # X-VLA restarts from the measured pose
                sim.set_control_mode("delta")
                raw = self.pa.raw_observation()
                actions = self.fake_pi05(raw, goal, call_seed(ep.episode_seed, step, "pi05", "act"))
                ep.pi05_requests.append(digest(np.concatenate([np.ravel(raw["pixels"]["image"]),
                                                               np.ravel(raw["pixels"]["image2"])])))
                seq = [self.pa.env_action(a) for a in actions[:SEGMENT]]
                ep.pi05_steps += SEGMENT
            else:
                sim.set_control_mode("absolute")
                seq = None
            for i in range(SEGMENT):
                a = seq[i] if seq is not None else xa.pop()
                done = sim.step(a)
                self.records.append(StepRecord(sim.steps, driver, digest(a), obs_digest(sim.obs), mj_digest(sim), done))
                if done or sim.steps >= until_step:
                    break
            ep.prev_driver = driver
        return self.records
