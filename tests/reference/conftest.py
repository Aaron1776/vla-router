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

# LIBERO-10's instructions in task order (hf-libero 0.1.4, `task.language`; checked against the benchmark by
# test_batch_invariance.py where LIBERO is installed). Their lengths differ, which the batching tests rely on.
LIBERO_10 = [
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
]

requires_libero = pytest.mark.skipif(not LIBERO_READY, reason="needs the simulator environment (LIBERO + robosuite)")
requires_lerobot = pytest.mark.skipif(not (LIBERO_READY and have("lerobot")), reason="needs lerobot 0.6.1 + LIBERO")
