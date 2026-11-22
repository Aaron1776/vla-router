"""Runs one queue job on a simulator worker (docs/SPEC.md §7, §12): writes its records and files with
`dataset.JobOutput` and returns them for upload. The worker loop (vlar_ops.worker) uploads them before it marks
the job done, so a job is done only once its snapshots, images and records are in the bucket.

Episode jobs (`kind == "episode"`, spec from `vlar_ops.plan.episode_plan`): routed runs decide by the phase's
scripted policy: `base` always continues locally; `explore` escalates with probability `p_escalate` while the budget
allows, drawn from the rollout seed's `explore` stream (seeding.uniform), so a re-run makes the same choices. With
`record="full"`, every measured decision point gets a snapshot (taken after its calls), its camera images and a
`points` row. Single-model runs (`xvla_only`, `pi05_only`) record the run, its steps and its events.

Branch jobs (kind = a branch type, spec from `vlar_ops.plan.branch_plan`): the snapshot is read from the bucket and
`Episode.run_branch` continues it; the run, its steps and its own events are recorded (no measurements).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from . import seeding
from .dataset import JobOutput, snapshot_key
from .snapshot import SnapshotStore, from_bytes

POLICIES = ("base", "explore")


class Runner:
    """`make_episode(task_id)` returns an `Episode` on a simulator of that task (the worker caches them);
    `read_bytes(key)` reads from the bucket. Calling the runner with a job returns (artifact records, summary)."""

    def __init__(self, cfg, make_episode: Callable[[int], Any], read_bytes: Callable[[str], Optional[bytes]],
                 staging: Path | str, *, p_escalate: float):
        self.cfg, self.make_episode, self.read_bytes = cfg, make_episode, read_bytes
        self.staging, self.p_escalate = Path(staging), float(p_escalate)
        self.header = {"harness_version": cfg["harness_version"], "config_hash": getattr(cfg, "hash", None)}

    def __call__(self, job) -> tuple[list[dict], dict]:
        out = JobOutput(self.staging, job.phase, job.job_id, header=self.header)
        summary = self._episode(job, out) if job.kind == "episode" else self._branch(job, out)
        return [a.as_record() for a in out.finish()], summary

    # ------------------------------------------------------------ episodes

    def _choice(self, phase: str, ep, dp) -> str:
        if phase == "base" or not dp.can_escalate:
            return "local"
        if phase == "explore":
            u = seeding.uniform(seeding.stream_seed(ep.state.rollout_seed, "explore"), f"dp{dp.index}")
            return "escalate" if u < self.p_escalate else "local"
        raise ValueError(f"no scripted policy for phase {phase!r} (one of {POLICIES}; DAgger and evaluation runs "
                         "need a learned decider)")

    def _episode(self, job, out: JobOutput) -> dict:
        spec = job.spec
        scene = self.cfg.selected.get("scene_seed")
        if spec.get("scene", "project") != scene:
            raise ValueError(f"job wants scene option {spec.get('scene')!r}, the worker's config has {scene!r}")
        ep = self.make_episode(int(spec["task_id"]))
        episode_id = job.subject
        dp = ep.start(int(spec["init_idx"]), int(spec["rollout_seed"]), spec["mode"], episode_id,
                      profile=spec["profile"], record=spec["record"])
        points = 0
        if spec["mode"] == "routed":
            while dp is not None:
                m = ep.point_meas
                if spec["record"] == "full" and m is not None and m["index"] == dp.index:
                    sid = SnapshotStore.make_id(episode_id, dp.index)
                    out.put_snapshot(sid, ep.snapshot(header=self.header))
                    out.put_images(episode_id, dp.index, {k: ep.sim.obs[k] for k in ep.sim.camera_keys})
                    out.add_point(episode_id, m, budget_remaining_steps=dp.budget_remaining_steps, snapshot_id=sid,
                                  thresholds=ep.gripper_thresholds, n_samples=ep.K)
                    points += 1
                dp = ep.decide(self._choice(job.phase, ep, dp)).next_point
        else:
            ep.run_episode()
        res, st = ep.result(), ep.state
        out.add_run(_run_row(episode_id, "episode", st, res, episode_id=episode_id))
        out.add_steps(episode_id, ep.trajectory)
        out.add_events(episode_id, ep.own_events)
        return {"run_id": episode_id, "success": res.success, "steps": res.steps, "points": points}

    # ------------------------------------------------------------ branches

    def _branch(self, job, out: JobOutput) -> dict:
        spec = job.spec
        sid = spec["snapshot_id"]
        data = self.read_bytes(snapshot_key(sid))
        if data is None:
            raise FileNotFoundError(f"snapshot {sid} is not in the bucket")
        ep = self.make_episode(int(spec["task_id"]))
        snapshot = from_bytes(data)
        parent = int(snapshot.episode["rollout_seed"])
        br = ep.run_branch(snapshot, branch=spec["branch"], pi05_calls=int(spec["pi05_calls"]),
                           seed_index=int(spec["seed_index"]), snapshot_id=sid)
        run_id = f"{sid}/{spec['branch']}/s{int(spec['seed_index'])}"
        row = _run_row(run_id, "branch", ep.state, br.result, episode_id=sid.rsplit("/", 1)[0], snapshot_id=sid,
                       branch=spec["branch"], seed_index=int(spec["seed_index"]), pi05_calls=int(spec["pi05_calls"]),
                       parent_rollout_seed=parent)
        out.add_run(row)
        out.add_steps(run_id, br.trajectory)
        out.add_events(run_id, br.events)
        return {"run_id": run_id, "success": br.result.success, "steps": br.result.steps}


def _run_row(run_id: str, kind: str, st, res, **extra) -> dict:
    row = {"run_id": run_id, "run_kind": kind, "snapshot_id": None, "branch": None, "seed_index": None,
           "pi05_calls": None, "parent_rollout_seed": None, "task_id": st.task_id, "init_idx": st.init_idx,
           "scene_seed": st.scene_seed, "rollout_seed": st.rollout_seed, "mode": st.mode, "profile": st.profile,
           "record": st.record, "time_limit": bool(res.time_limit), "time_limit_s": st.time_limit_s,
           "success": bool(res.success), "end_reason": res.end_reason, "steps": res.steps,
           "success_step": res.success_step, "pi05_steps": res.pi05_steps, "pi05_requests": res.pi05_requests,
           "pi05_failed_requests": res.pi05_failed_requests, "sim_time_s": res.sim_time_s}
    row.update(extra)
    return row
