"""Leave-one-task-out cross-validation with a separate validation task (SPEC §7): in fold i, task i is the test task,
task (i + 1) mod 10 the validation task, the other eight the training tasks; always by whole task."""

from __future__ import annotations

import numpy as np

N_TASKS = 10


def fold(i: int, n_tasks: int = N_TASKS) -> dict[str, object]:
    if not 0 <= i < n_tasks:
        raise ValueError(f"fold {i} outside 0..{n_tasks - 1}")
    test, val = i, (i + 1) % n_tasks
    return {"fold": i, "test": test, "val": val, "train": [t for t in range(n_tasks) if t not in (test, val)]}


def masks(task: np.ndarray, i: int, n_tasks: int = N_TASKS) -> dict[str, np.ndarray]:
    """Boolean row masks for train, val and test in fold i (rows by their task)."""
    f = fold(i, n_tasks)
    task = np.asarray(task)
    return {"train": np.isin(task, f["train"]), "val": task == f["val"], "test": task == f["test"]}
