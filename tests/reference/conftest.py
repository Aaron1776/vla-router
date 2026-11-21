"""Equivalence tests against the two reference harnesses' own code.

Sim-only tests need the simulator environment (LIBERO, robosuite, MuJoCo, LeRobot 0.6.1); run
them with that environment's Python, e.g. locally:

    .venv-sim/bin/python -m pytest tests/reference

Model tests also need the weights and run in the matching server environment (see each file).
Everything here is skipped when its environment is missing, so the plain test run stays fast.
"""

import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / "third_party" / "libero-home"))
os.environ.setdefault("MUJOCO_GL", "egl" if os.uname().sysname == "Linux" else "cgl")


def have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


LIBERO_READY = have("libero") and have("robosuite") and have("mujoco") and \
    (Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml").exists()

requires_libero = pytest.mark.skipif(not LIBERO_READY, reason="needs the simulator environment (LIBERO + robosuite)")
requires_lerobot = pytest.mark.skipif(not (LIBERO_READY and have("lerobot")), reason="needs lerobot 0.6.1 + LIBERO")
