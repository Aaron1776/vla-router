"""The two seeds of an episode (docs/SPEC.md §3, amended 2026-11-21).

Scene seed: the env seed given to `env.seed` before reset. With the initial-state file it defines the
scene: LIBERO places some fixtures at reset from NumPy's global RNG (the stove, cabinet, wine rack, desk
caddy and microwave move by up to ~19 mm with it). It is `offset + k` for initial state k, where the
offset is the named config option `scene_seed`: 142 in every project run, 1000 only in the π0.5 parity
run. It never feeds anything else.

Rollout seed: everything else random. Every consumer derives its own seed from it, so adding or removing
a call never shifts another call's randomness:
- `call_seed`: model noise for one call (X-VLA /act and /sample, π0.5), keyed by step, model, call type
  and sample index;
- `stream_seed`: a named stream (the network model, exploration's random escalations);
- `branch_rollout_seed`: the rollout seed of one branch continuation from a decision point, never equal
  to the parent rollout's.
"""

from __future__ import annotations

import hashlib

MODELS = ("xvla", "pi05")
# X-VLA: `act` at a chunk point, `act_fresh` after an abandoned escalation, `sample` for measurements;
# π0.5: `act`.
CALL_TYPES = ("act", "act_fresh", "sample")
STREAMS = ("network", "explore")
_MASK63 = (1 << 63) - 1


def _derive(*parts) -> int:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & _MASK63      # fits torch.manual_seed and NumPy


def scene_seed(init_idx: int, offset: int) -> int:
    if init_idx < 0:
        raise ValueError("initial-state index must be >= 0")
    return int(offset) + int(init_idx)


def call_seed(rollout_seed: int, step: int, model: str, call: str, sample: int = 0) -> int:
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if call not in CALL_TYPES:
        raise ValueError(f"call must be one of {CALL_TYPES}")
    return _derive("call", int(rollout_seed), int(step), model, call, int(sample))


def stream_seed(rollout_seed: int, name: str) -> int:
    if name not in STREAMS:
        raise ValueError(f"stream must be one of {STREAMS}")
    return _derive("stream", int(rollout_seed), name)


def branch_rollout_seed(parent_rollout_seed: int, decision_index: int, branch: str, seed_index: int) -> int:
    seed = _derive("branch", int(parent_rollout_seed), int(decision_index), branch, int(seed_index))
    return seed if seed != parent_rollout_seed else _derive("branch-retry", seed)
