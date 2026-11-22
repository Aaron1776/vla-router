"""X-VLA adapter: the authors' LIBERO client logic (libero_client.py at the pinned commit) with
the harness's transport (docs/SPEC.md §2-3).

Everything that decides what X-VLA sees and what the robot executes is the authors' code:
- `_format_query` (main view flipped 180°, wrist view as rendered, 20-value proprio
  `[pos3, rot6d, 0, zeros10]` from the controller's cached pose, `domain_id` 3, 10 steps);
- the quirk: after each `/act`, `proprio[:9]` becomes the last predicted pose of that chunk;
- 6D-rotation -> axis-angle conversion (`LiberoAbsActionProcessor`), queueing all 30 actions,
  and rounding the gripper to {-1, 1} when an action is popped.

The only change is that `ClientModel.step()` is split in two, because in routed mode the
harness asks for a chunk before the decision and executes it only if the decider continues:
`query()` is lines 192-207 of `ClientModel.step`, `pop()` is lines 209-212. tests/reference
checks the split against the authors' `step()` on the same responses.

Handoffs (step 3 of the effort plan) only use the two hooks the authors' class already has:
`proprio = None` makes the next query send the measured pose, and clearing `action_plan`
discards the queue.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import numpy as np

from .. import reference

Transport = Callable[[dict, Optional[int]], Mapping[str, Any]]   # (payload, seed) -> JSON response with "action"

STATE_MODES = ("quirk", "measured")


def _client_class(lc):
    """Subclass of the authors' ClientModel; built at runtime because the module is loaded from the checkout."""

    class HarnessClientModel(lc.ClientModel):
        def __init__(self, transport: Transport):
            # The authors' __init__ only builds the URL; keep its other two lines.
            self._transport = transport
            self.processor = lc.LiberoAbsActionProcessor()
            self.next_seed: Optional[int] = None
            self.last_payload: Optional[dict] = None
            self.last_response: Optional[Mapping[str, Any]] = None
            self.reset()

        def _post(self, payload: dict) -> np.ndarray:
            # Same validation as the authors' _post (libero_client.py:186-189), our transport.
            self.last_payload = payload
            data = self._transport(payload, self.next_seed)
            self.last_response = data
            action = np.array(data["action"])
            if action.ndim != 2 or action.shape[1] < 10:
                raise RuntimeError(f"Unexpected action shape from server: {action.shape}")
            return action

        def query(self, obs: dict, goal: str) -> np.ndarray:
            """ClientModel.step lines 192-207: request a chunk and queue it. Returns the raw chunk."""
            payload = self._format_query(obs, goal)
            action = self._post(payload)
            self.proprio[:9] = action[-1, :9].copy()
            target_eef = action[:, :3]
            target_axis = self.processor.Rotate6D_to_AxisAngle(action[:, 3:9])
            target_act = action[:, 9:10]
            final_action = np.concatenate([target_eef, target_axis, target_act], axis=-1)
            for row in final_action.tolist():
                self.action_plan.append(row)
            return action

        def pop(self) -> np.ndarray:
            """ClientModel.step lines 209-212: next queued action, gripper rounded to {-1, 1}."""
            action_predict = np.array(self.action_plan.popleft(), dtype=np.float32)
            action_predict[-1] = 1.0 if action_predict[-1] > 0.5 else -1.0
            return action_predict

    return HarnessClientModel


class XVLAAdapter:
    def __init__(self, transport: Transport, state_mode: str = "quirk", client_sha256: Optional[str] = None):
        if state_mode not in STATE_MODES:
            raise ValueError(f"state_mode must be one of {STATE_MODES}")
        self.lc = reference.xvla_libero_client(client_sha256)
        self.client = _client_class(self.lc)(transport)
        self.state_mode = state_mode

    # ------------------------------------------------------------ episode state

    def reset(self) -> None:
        """New episode: empty queue, next query sends the measured pose (ClientModel.reset)."""
        self.client.reset()

    def queue_len(self) -> int:
        return len(self.client.action_plan)

    def queued_actions(self) -> np.ndarray:
        """The queued actions as stored by the client: [pos3, axis-angle3, raw gripper] rows."""
        return np.array(list(self.client.action_plan), dtype=np.float64).reshape(-1, 7)

    def discard_queue(self) -> None:
        self.client.action_plan.clear()

    def get_state(self) -> dict:
        """The client's episode state for a snapshot: the quirk proprio (None once forgotten) and the
        queued actions exactly as stored (Python floats from the chunk conversion)."""
        proprio = self.client.proprio
        return {"proprio": None if proprio is None else np.array(proprio, copy=True),
                "queue": [list(row) for row in self.client.action_plan],
                "state_mode": self.state_mode}

    def set_state(self, state: dict) -> None:
        if state["state_mode"] != self.state_mode:
            raise ValueError(f"snapshot taken in state mode {state['state_mode']}, adapter is {self.state_mode}")
        p = state["proprio"]
        self.client.proprio = None if p is None else np.array(p, copy=True)
        self.client.action_plan.clear()
        self.client.action_plan.extend([list(row) for row in state["queue"]])

    def forget_predicted_pose(self) -> None:
        """Make the next query send the measured pose (handoff back from π0.5, §3)."""
        self.client.proprio = None

    # ------------------------------------------------------------ observations

    def observation(self, sim) -> dict:
        """The obs dict the authors' rollout loop passes to the client (libero_client.py:295-298):
        the env observation plus the controller's cached pose."""
        sim.require_images()
        ee_pos, ee_ori_mat = sim.controller_pose()
        obs = dict(sim.obs)
        obs["robo_ori"] = self.client.processor.Mat_to_Rotate6D(ee_ori_mat)
        obs["robo_pos"] = ee_pos
        return obs

    # ------------------------------------------------------------ calls

    def query(self, sim, goal: str, seed: Optional[int] = None) -> np.ndarray:
        """Request and queue a chunk (chunk point). In `measured` mode every query sends the measured pose."""
        if self.queue_len():
            raise RuntimeError("query() with actions still queued; discard them first")
        if self.state_mode == "measured":
            self.forget_predicted_pose()
        self.client.next_seed = seed
        try:
            return self.client.query(self.observation(sim), goal)
        finally:
            self.client.next_seed = None

    def pop(self) -> np.ndarray:
        return self.client.pop()

    def measurement_payload(self, sim, goal: str) -> dict:
        """Payload for /features and mid-point /sample probes: the authors' query format with the
        measured pose, built without touching the client's state (its proprio and queue)."""
        saved = self.client.proprio
        try:
            self.client.proprio = None
            return self.client._format_query(self.observation(sim), goal)
        finally:
            self.client.proprio = saved

    def last_query_payload(self) -> dict:
        """The exact payload of the last chunk-point query. The chunk-time `/sample` reuses it, so
        its K-1 extra chunks are samples of the same request as the executed one (§6)."""
        if self.client.last_payload is None:
            raise RuntimeError("no query has been made in this episode")
        return self.client.last_payload
