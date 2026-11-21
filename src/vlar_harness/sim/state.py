"""Exact capture and restore of the simulator state (docs/SPEC.md §7 as amended).

What has to come back for a replay to match the original step for step (docs/HARNESS.md,
"Snapshots"; found by walking the env's object graph, tests/reference/statewalk.py):

- MuJoCo: `mj_getState(..., mjSTATE_INTEGRATION)` — time, qpos, qvel, act, controls, applied
  forces, mocap, user data and the constraint solver's warm start.
- Compiled-model fields LIBERO writes at run time: `body_pos`/`body_quat` (fixtures such as the
  stove or microwave are placed at reset from NumPy's global RNG, so they depend on the env seed,
  not on the initial state) and `site_rgba` (the stove burner's glow, toggled every step). The rest
  of the compiled model must match the snapshot's digest (cached per compiled model).
- robosuite: the step counters, the observation cache, and every observation's sampling timer and
  buffered value, including the extra π0.5 cameras. The observation at a decision point was
  sampled one physics substep before the end of the step, so it is restored, never re-rendered.
- Per robot: the controller (goals, cached pose, Jacobians, mass matrix, `use_delta`), the
  gripper (its integrated command), the robot's history buffers and last torques.
- LIBERO: every object's, fixture's and site's `object_properties` (e.g. the stove's burner
  flag), restored in place because sites share their parent fixture's dicts, and the list of
  objects with visual state.

Never cache model or data objects across calls: a hard reset replaces both.
"""

from __future__ import annotations

import copy
import hashlib
from collections import deque
from typing import Any

import numpy as np

MODEL_FIELDS = ("body_pos", "body_quat", "site_rgba")
ENV_FIELDS = ("timestep", "cur_time", "done", "_obs_cache", "visualization_sites_list")
OBSERVABLE_FIELDS = ("_time_since_last_sample", "_sampled", "_current_delay", "_current_observed_value",
                     "_enabled", "_active")
SCHEMA = 1


class StateMismatch(RuntimeError):
    pass


def _mujoco():
    import mujoco
    return mujoco


def mj_state_spec() -> int:
    return int(_mujoco().mjtState.mjSTATE_INTEGRATION)


# ---------------------------------------------------------------- leaves

def _is_leaf(v: Any) -> bool:
    if isinstance(v, (np.ndarray, np.generic, bool, int, float, str, bytes, type(None))):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_leaf(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, (str, int)) and _is_leaf(x) for k, x in v.items())
    return False


def _copy(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return np.array(v, copy=True)
    return copy.deepcopy(v)


def capture_leaves(obj: Any, names: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Copies of `obj`'s plain-data attributes (all of them, or `names`)."""
    attrs = vars(obj)
    keys = names if names is not None else tuple(attrs)
    return {k: _copy(attrs[k]) for k in keys if k in attrs and _is_leaf(attrs[k])}


def _equal(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return a == b or (a != a and b != b)


def restore_leaves(obj: Any, saved: dict[str, Any]) -> None:
    """Set `obj`'s attributes back to `saved`. Attributes that already hold the saved value are left
    alone and lists/dicts are updated in place, so containers shared with other objects stay shared."""
    attrs = vars(obj)
    for k, v in saved.items():
        cur = attrs.get(k, _MISSING)
        if cur is not _MISSING and _equal(cur, v):
            continue
        if isinstance(cur, list) and isinstance(v, list):
            cur[:] = _copy(v)
        elif isinstance(cur, dict) and isinstance(v, dict):
            restore_dict_in_place(cur, v)
        else:
            setattr(obj, k, _copy(v))


_MISSING = object()


def restore_dict_in_place(current: dict, saved: dict) -> None:
    """Update `current` to equal `saved` without replacing any dict object, so dicts that other
    objects share stay shared (LIBERO's site objects alias their parent fixture's properties)."""
    for k in list(current):
        if k not in saved:
            del current[k]
    for k, v in saved.items():
        if isinstance(v, dict) and isinstance(current.get(k), dict):
            restore_dict_in_place(current[k], v)
        else:
            current[k] = _copy(v)


OBJECT_GROUPS = ("objects_dict", "fixtures_dict", "object_sites_dict")


def _buffers(robot) -> dict[str, Any]:
    """robosuite's history buffers hanging off a robot (recent_qpos, recent_ee_vel_buffer, ...)."""
    out = {}
    for k, v in vars(robot).items():
        if type(v).__module__.startswith("robosuite.utils.buffers"):
            out[k] = v
    return out


# ---------------------------------------------------------------- model

def model_digest(model, exclude: tuple[str, ...] = ()) -> str:
    """SHA-256 over every array field of a mujoco.MjModel (name, dtype, shape, bytes), minus `exclude`.
    Takes ~40 ms (mesh and texture data); see `static_model_digest` for the cached form."""
    h = hashlib.sha256()
    for name in sorted(dir(model)):
        if name.startswith("_") or name in exclude:
            continue
        try:
            v = getattr(model, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray):
            h.update(name.encode() + b"\0" + v.dtype.str.encode() + str(v.shape).encode())
            h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def static_model_digest(inner) -> str:
    """Digest of the compiled model without the fields LIBERO writes at run time (MODEL_FIELDS),
    computed once per compiled model: a hard reset builds a new model wrapper, which drops the cache.
    Nothing else in the model changes during an episode (tests/reference/test_snapshot_replay.py
    compares the full digest after every restore)."""
    wrapper = inner.sim.model
    cached = getattr(wrapper, "_vlar_static_digest", None)
    if cached is None or cached[0] is not wrapper._model:
        cached = (wrapper._model, model_digest(wrapper._model, exclude=MODEL_FIELDS))
        wrapper._vlar_static_digest = cached
    return cached[1]


# ---------------------------------------------------------------- capture / restore

def capture(sim) -> dict[str, Any]:
    """Everything needed to continue `sim` (a LiberoSim) exactly from the current decision point."""
    mj = _mujoco()
    inner = sim.env.env
    model, data = inner.sim.model._model, inner.sim.data._data
    spec = mj_state_spec()
    mj_state = np.empty(mj.mj_stateSize(model, spec), dtype=np.float64)
    mj.mj_getState(model, data, mj_state, spec)

    robots = []
    for robot in inner.robots:
        robots.append({
            "robot": capture_leaves(robot),
            "buffers": {name: capture_leaves(buf) for name, buf in _buffers(robot).items()},
            "controller": capture_leaves(robot.controller),
            "gripper": capture_leaves(robot.gripper) if getattr(robot, "gripper", None) is not None else {},
        })
    objects = {}
    for group in OBJECT_GROUPS:
        for name, obj in getattr(inner, group, {}).items():
            props = getattr(obj, "object_properties", None)
            if props is not None:
                objects[f"{group}/{name}"] = copy.deepcopy(props)

    return {
        "schema": SCHEMA,
        "task_id": sim.task_id,
        "mj_state_spec": spec,
        "mj_state": mj_state,
        "flat_state": np.asarray(inner.sim.get_state().flatten()),      # for inspection only
        "model_fields": {f: np.array(getattr(model, f)) for f in MODEL_FIELDS},
        "model_static_digest": static_model_digest(inner),
        "env": capture_leaves(inner, ENV_FIELDS),
        "observables": {name: capture_leaves(o, OBSERVABLE_FIELDS) for name, o in inner._observables.items()},
        "robots": robots,
        "object_properties": objects,
        "sim": {"steps": sim.steps, "done": sim.done},
    }


def restore(sim, state: dict[str, Any]) -> None:
    """Put `state` back into `sim`, an env of the same task (it may have run other episodes).

    Does not call set_init_state, reset or render. Raises StateMismatch if the env cannot hold the
    state exactly (different task, different compiled model, different observations or robots).
    """
    mj = _mujoco()
    if state.get("schema") != SCHEMA:
        raise StateMismatch(f"snapshot schema {state.get('schema')} != {SCHEMA}")
    if state["task_id"] != sim.task_id:
        raise StateMismatch(f"snapshot is for task {state['task_id']}, env is task {sim.task_id}")
    inner = sim.env.env
    if set(state["observables"]) != set(inner._observables):
        raise StateMismatch("the env's observations differ from the snapshot's")
    if len(state["robots"]) != len(inner.robots):
        raise StateMismatch("robot count differs")

    model, data = inner.sim.model._model, inner.sim.data._data
    if static_model_digest(inner) != state["model_static_digest"]:
        raise StateMismatch("compiled model differs from the snapshot's (task, LIBERO version or assets)")
    for f, v in state["model_fields"].items():
        getattr(model, f)[...] = v

    spec = int(state["mj_state_spec"])
    if mj.mj_stateSize(model, spec) != state["mj_state"].size:
        raise StateMismatch("MuJoCo state size differs")
    mj.mj_setState(model, data, np.asarray(state["mj_state"], dtype=np.float64), spec)
    # Make derived quantities consistent with the restored state. mj_forward does not touch the
    # warm start or any other integration state, and every env.step starts with a forward pass anyway.
    mj.mj_forward(model, data)

    restore_leaves(inner, state["env"])
    for name, saved in state["observables"].items():
        restore_leaves(inner._observables[name], saved)
    for robot, saved in zip(inner.robots, state["robots"]):
        restore_leaves(robot, saved["robot"])
        bufs = _buffers(robot)
        for name, leaves in saved["buffers"].items():
            restore_leaves(bufs[name], leaves)
        restore_leaves(robot.controller, saved["controller"])
        if saved["gripper"]:
            restore_leaves(robot.gripper, saved["gripper"])
    for key, props in state["object_properties"].items():
        group, name = key.split("/", 1)
        restore_dict_in_place(getattr(inner, group)[name].object_properties, props)

    sim.steps = int(state["sim"]["steps"])
    sim.done = bool(state["sim"]["done"])
    sim.obs = inner._get_observations()   # rebuilt from the restored buffered values, not re-rendered
