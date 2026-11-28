"""The setup count (SPEC §17 amendment 62 (2), rule (A) of amendment 65): recorded in every result marker, replayed
where a job asks for it.

The setup count is the simulator's model-load count for the episode's task when the episode is set up (a fresh
simulator is load 1 and each `setup_episode` one more; docs/HARNESS.md, "Fixture placement depends on the simulator's
load count"). In tasks 3, 8 and 9 the fixtures land somewhere new at every count, so two runs share a scene only at
equal counts.

- **Recording.** `SetupCounts.instrument` wraps a simulator factory so that every simulator it builds counts its own
  `setup_episode` calls (the bound method is wrapped on the instance: the object, and so everything the harness does
  with it, is unchanged). `job_meta(job)` gives the counts a job ended with: `setup_count` (the count at the job's
  last setup, i.e. the episode's, for an episode job) and `setups_in_job`. It goes into the job's result marker.
- **Replay.** An episode job whose spec records `setup_count: n` runs on a simulator whose next setup is load n: the
  task's cached simulator is kept if it is exactly there, else it is closed and a fresh one is built and warmed up
  (`pilot.warm_up`). Rule (A): every fresh-start run (Tranche B's screen) records 2, a fresh simulator; a re-run of a
  screened episode records the screen's count, so it also runs on a fresh simulator. After the job the count must
  be n, else the job fails (no marker, so no record of another scene is ever kept).
- Jobs without `setup_count` (branches, which restore the model fields from their snapshot and never call
  `setup_episode`; Tranche A) run as before on the cached simulator, their count recorded.

Ops code outside the common data digest: `vm_worker.ProductionEpisodes` is not changed (its sims cache is managed
from here, through its `sims` dict and `_make_sim`), and what the setup count changes is recorded per job (the spec's
`setup_count`, the marker's measured one), as amendment 49 does for tunables.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

FRESH = 2      # rule (A): a fresh simulator (load 1) set up once


class SetupCountError(RuntimeError):
    """A job's simulator did not set up at the count its spec records."""


def _count(sim) -> int:
    return int(getattr(sim, "_vlar_setups", 0))


class SetupCounts:
    """Setup counting and replay for one worker's `ProductionEpisodes` (anything with a `sims` dict of task ->
    simulator, a `_make_sim(task_id)` factory and simulators with `setup_episode` and `close`)."""

    def __init__(self, episodes, warm_up: Optional[Callable[[Any, int], None]] = None):
        self.episodes = episodes
        if warm_up is None:
            from .pilot import warm_up
        self.warm_up = warm_up
        self._before: dict[str, tuple[int, int]] = {}
        self.instrument()

    def instrument(self) -> None:
        make = self.episodes._make_sim
        if getattr(make, "_vlar_counted", False):
            return

        def counted(task_id: int):
            sim = make(task_id)
            sim._vlar_setups = 1                  # load 1: the simulator's construction
            inner = sim.setup_episode

            def setup_episode(*args, **kwargs):
                sim._vlar_setups += 1             # a load, even if the setup then fails
                return inner(*args, **kwargs)
            sim.setup_episode = setup_episode
            return sim
        counted._vlar_counted = True
        self.episodes._make_sim = counted

    def count(self, task_id: int) -> Optional[int]:
        sim = self.episodes.sims.get(task_id)
        return None if sim is None else _count(sim)

    def prepare(self, job) -> None:
        """Before a job: for an episode job recording `setup_count` n, a simulator of the task whose next setup is
        load n. Records the count the job starts from."""
        task = int(job.spec["task_id"])
        want = job.spec.get("setup_count") if job.kind == "episode" else None
        if want is not None:
            want = int(want)
            if want < FRESH:
                raise SetupCountError(f"{job.subject}: setup_count {want} < {FRESH} (a fresh simulator's first setup)")
            if self.count(task) != want - 1:
                old = self.episodes.sims.pop(task, None)
                if old is not None:
                    old.close()
                self.warm_up(self.episodes.sim(task), want)
                if self.count(task) != want - 1:
                    raise SetupCountError(f"{job.subject}: warm-up left the count at {self.count(task)}, not {want - 1}")
        self._before[job.job_id] = (task, self.count(task) or 0)

    def check(self, job) -> None:
        """After a job (before its marker): the episode set up at the count its spec records, exactly once."""
        want = job.spec.get("setup_count") if job.kind == "episode" else None
        if want is None:
            return
        m = self.job_meta(job, keep=True)
        if m["setups_in_job"] != 1 or m["setup_count"] != int(want):
            raise SetupCountError(f"{job.subject}: set up {m['setups_in_job']} time(s), last at count "
                                  f"{m['setup_count']}; the spec records {want}")

    def job_meta(self, job, *, keep: bool = False) -> dict[str, Any]:
        """The marker's setup fields for a job `prepare` saw: the count at the job's last setup (None if it set
        nothing up, e.g. a branch) and the number of setups during the job."""
        task, before = self._before[job.job_id] if keep else self._before.pop(job.job_id, (None, 0))
        after = self.count(task) if task is not None else None
        n = (after or 0) - before if after is not None else 0
        return {"setup_count": after if n > 0 else None, "setups_in_job": n, "setup_count_spec": job.spec.get("setup_count")}


class CountingRunner:
    """The worker's runner with the setup count around each job: prepare, run, check (a failed check fails the job)."""

    def __init__(self, runner: Callable, counts: SetupCounts):
        self.runner, self.counts = runner, counts

    def __call__(self, job):
        self.counts.prepare(job)
        out = self.runner(job)
        self.counts.check(job)
        return out

    def __getattr__(self, name):          # refuse, job_dir: the wrapped runner's
        return getattr(self.runner, name)


def meta_with_counts(meta: Mapping[str, Any], counts: SetupCounts) -> Callable[[Any], dict[str, Any]]:
    """`run_worker`'s per-job marker meta: the worker's meta plus the job's setup fields."""
    return lambda job: {**meta, **counts.job_meta(job)}
