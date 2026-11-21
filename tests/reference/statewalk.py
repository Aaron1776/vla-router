"""Walk the simulator's Python object graph and fingerprint every leaf value.

Used by the snapshot tests to prove that restore leaves nothing behind: after a restore, every
leaf reachable from the env must equal its value at capture time, and the compiled model's arrays
and MuJoCo's integration state must be identical.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

OWN_MODULES = ("robosuite", "libero", "vlar_harness")
SKIP_ATTRS = {"viewer", "_destroy_viewer", "model_xml"}


def _leaf_digest(v):
    if isinstance(v, np.ndarray):
        return ("nd", v.dtype.str, v.shape, hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest())
    if isinstance(v, (bool, int, float, str, bytes, type(None), np.generic)):
        return ("v", repr(v))
    return None


def walk(root, max_nodes=200000):
    """{path: digest} for every leaf reachable through objects defined in robosuite/libero/harness code."""
    out, seen = {}, set()
    queue = deque([("env", root)])
    while queue and len(seen) < max_nodes:
        path, obj = queue.popleft()
        d = _leaf_digest(obj)
        if d is not None:
            out[path] = d
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                queue.append((f"{path}[{k!r}]", v))
        elif isinstance(obj, (list, tuple, deque)):
            out[path + ".__len__"] = ("v", len(obj))
            for i, v in enumerate(list(obj)):
                queue.append((f"{path}[{i}]", v))
        elif type(obj).__module__.startswith(OWN_MODULES) and hasattr(obj, "__dict__"):
            for k, v in list(vars(obj).items()):
                if k in SKIP_ATTRS or callable(v) and not hasattr(v, "__dict__"):
                    continue
                queue.append((f"{path}.{k}", v))
    return out


def model_digest(model) -> dict:
    """sha1 of every array field of a mujoco.MjModel."""
    out = {}
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            v = getattr(model, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray):
            out[name] = hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest()
    return out


def diff(a: dict, b: dict) -> list[str]:
    keys = set(a) | set(b)
    return sorted(k for k in keys if a.get(k) != b.get(k))
