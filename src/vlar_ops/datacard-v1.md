# Datacard: vla-router dataset v1

Data collected with harness v1 (`vlar_harness`) for budgeted escalation from X-VLA (local) to π0.5 (remote) on
LIBERO-Long (LIBERO-10). The design is `docs/SPEC.md` in the project repository; section numbers below (§) refer to
it. Where this card and the code differ, the code at the commit recorded in `MANIFEST.json` is the definition; the
module named in each section holds it.

## Layout

All keys are relative to the dataset root (a local copy of the project bucket).

| Path | Contents |
| --- | --- |
| `records/<table>/<phase>/<job_id>.parquet` | One Parquet part per job and table (`runs`, `steps`, `events`, `points`). Read a whole table with `vlar_harness.dataset.read_table(root, table)`. |
| `snapshots/<episode_id>/dpNN.npz` | One snapshot per measured decision point `NN` of a routed `record="full"` run (base, exploration, DAgger). |
| `images/<episode_id>.npz` | Every camera image at every measured decision point of that episode, as lossless PNG bytes. |
| `results/<phase>/<job_id>.json` | The job queue's result marker, written last: the job, its summary, every file it produced with SHA-256 and size, and the provenance of the code that ran it. |
| `MANIFEST.json` | Schema and harness versions, config hashes and options, the code's provenance, counts, and the SHA-256 and size of every file. |
| `DATACARD.md` | This file. |

Nothing else belongs in the tree. A job's files count only once its marker exists; files of a job that was
interrupted and never finished are removed before the manifest is built (`vlar-ops dataset manifest`).

Phases: `base` (X-VLA always local, routed mode, 500 episodes, §7), `explore` (routed, scripted random escalation
with probability `p_escalate` per decision point while budget remains, §7), `pi05_parity` (π0.5 alone, §11),
`branches` (continuations from sampled decision points, §7), later `dagger` and `eval` (§9, §10).

## Identifiers and seeds

- Episode id `<phase>/t<task>/i<init>`, e.g. `base/t3/i17`: task 0-9 in LIBERO-10's order, initial state 0-49.
- Snapshot id `<episode_id>/dpNN`, NN = the decision point's index, two digits.
- Branch run id `<snapshot_id>/<branch>/s<seed_index>`, branch one of `b1_continue` (X-VLA executes its stored
  queue, then continues locally), `b2_one_call` (one π0.5 call of 10 steps, then X-VLA), `b3_three_calls` (three
  consecutive π0.5 calls, truncated to the remaining budget), `b4_max` (π0.5 until the budget is spent or the episode
  ends). Branches with equal π0.5 call counts at a point are run once and shared (`vlar_ops.plan.branch_plan`).
- Job id: `vlar_ops.plan.job_id(subject, kind, seed_index)`; the marker holds the full job.
- Scene seed (the environment seed): 142 + initial-state index in every project run; 1000 + index only in the π0.5
  parity run (harness option `scene_seed`).
- Rollout seed (model noise, network draws, exploration): `namespace + 100 * task_id + init_idx`, namespaces in
  `vlar_ops/configs/collection-v1.yaml` (base 1,000,000, explore 2,000,000, DAgger 3,000,000, parity 5,000,000,
  evaluation 7,000,000; evaluation's are never used in data collection). A branch's rollout seed is
  `seeding.branch_rollout_seed(parent rollout seed, decision index, branch, seed index)`, never the parent's.
  Every model call's noise comes from `seeding.call_seed(rollout seed, step, model, call, sample)`.

## Provenance and integrity (§13, §15)

Every result marker's `meta.provenance` and every snapshot's header hold: `harness_version`, `config_name`,
`config_hash` (the harness config's fixed settings with its named options), `config_options`, `calibration_hash`
(SHA-256 of the Phase 0 calibration file in effect: latencies, batch geometry, worker count, gripper thresholds,
upload size, time limits; amendments 34, 35), `source_digest` (SHA-256 over the harness package's code and
configs), `commit` and `harness_dirty` (git), and `image_digests` (container images). Every `runs` row repeats
`harness_version`, `config_hash`, `calibration_hash` and `schema_version`. The manifest refuses to mix harness
versions, source digests or image digests; several config hashes may appear (named options, e.g. the parity run's
scene seeds) and are listed with their options. Given the freeze record (`--freeze freeze.json`), every calibration
hash must be the frozen file's or one in its lineage (earlier versions with the same write-once values: upload
size, π0.5 latency, batch geometry). Clocks and the gripper fractions are not stored: re-score the event log under
any profile, and compute the fractions from `prev_actions` and `prev_gripper_qpos` with the frozen thresholds
(`vlar_harness.measure.gripper_fractions_of_point`). `vlar-ops dataset verify --root <dir>` checks every file's size and
SHA-256 against the manifest and that no other file is present.

## Tables

Array columns are flattened row-major; shapes are given in the meaning column. Values are stored exactly as the run
produced them, in their own precision. Missing values are null.

### `runs`

One row per episode or branch run.

| Field | Type | Meaning |
| --- | --- | --- |
| `run_id` | string | Episode id or branch run id (above). |
| `run_kind` | string | `episode` or `branch`. |
| `phase` | string | The job's phase. |
| `job_id` | string | The job that produced the row. |
| `episode_id` | string | The episode, or for a branch the parent episode. |
| `snapshot_id` | string | Branch only: the snapshot it continues. |
| `branch` | string | Branch only: the branch type. |
| `seed_index` | int32 | Branch only: 1, 2 or 3. |
| `pi05_calls` | int32 | Branch only: π0.5 calls the branch makes at its first decision points (an abandoned escalation counts as one, §17 decision 17). |
| `task_id` | int32 | LIBERO-10 task, 0-9. |
| `init_idx` | int32 | Initial state, 0-49. |
| `scene_seed` | int64 | Environment seed (above). |
| `rollout_seed` | int64 | The run's rollout seed (for a branch, its own). |
| `parent_rollout_seed` | int64 | Branch only: the parent episode's rollout seed. |
| `mode` | string | `routed`, `xvla_only` or `pi05_only`. |
| `profile` | string | The decider profile the run was started with (`none` for scripted runs, §5). |
| `record` | string | `full` (every decision point measured and snapshotted) or `off`. |
| `time_limit` | bool | Whether the task's time limit was applied live (only routed runs with a live learned decider; false in every scripted run and branch). |
| `time_limit_s` | double | The live limit in seconds when `time_limit` is true, else null. |
| `success` | bool | The task succeeded before the step limit (and before the live time limit, if applied). Not a label: labels are re-scored per profile (below). |
| `end_reason` | string | `success`, `step_limit` (520 steps), `time_limit` (live cut) or `network` (π0.5 alone hit the 20-attempt cap). |
| `steps` | int32 | Environment steps executed after the settle steps. |
| `success_step` | int32 | The step at which the task succeeded, else null. |
| `pi05_steps` | int32 | Steps driven by π0.5 (the budget counts these, §4). |
| `pi05_requests` | int32 | π0.5 requests that got through. |
| `pi05_failed_requests` | int32 | π0.5 requests abandoned after their retries. |
| `config_hash` | string | Harness config hash: fixed settings with the named options (not the calibration). |
| `calibration_hash` | string | SHA-256 of the Phase 0 calibration file in effect (amendment 35); v1.0 records have the frozen file's hash or one in its lineage (`freeze.json`). |
| `harness_version` | string | Harness version. |
| `schema_version` | int32 | This schema's version, 1. |

### `steps`

One row per executed environment step.

| Field | Type | Meaning |
| --- | --- | --- |
| `run_id` | string | The run. |
| `step` | int32 | Steps executed so far, counting this one (1 = the first step after the settle steps). |
| `driver` | string | `xvla` or `pi05`. |
| `action` | float32[7] | The action as executed: X-VLA absolute target `[x, y, z, ax, ay, az, g]` (axis-angle rotation, gripper +1 close / -1 open); π0.5 delta action in LIBERO's convention. |
| `eef_pos` | float64[3] | End-effector position in the observation after the step (`robot0_eef_pos`). |
| `eef_quat` | float64[4] | End-effector quaternion in the observation after the step (`robot0_eef_quat`, robosuite's xyzw). |
| `gripper_qpos` | float64[2] | Finger joint positions in the observation after the step (`robot0_gripper_qpos`). |

A branch's `steps` start after its snapshot's step; the steps before it are the parent episode's.

### `events`

One row per event-log entry. The event log is complete: the clock under every profile, the outcome and the labels
are recomputed from it (`vlar_harness.clock`, `vlar_harness.labels`).

| Field | Type | Meaning |
| --- | --- | --- |
| `run_id` | string | The run. |
| `seq` | int32 | Position in the run's own log, from 0. A branch's own log starts with its `restore` and `branch` events; its prefix is the snapshot's `event_log`. |
| `t` | string | Event type (below). |
| `step` | int32 | The environment step the event happened at, where it has one. |
| `event` | string | The whole event as canonical JSON (sorted keys). |

Event types (`vlar_harness/episode.py`, `vlar_harness/eventlog.py`): `start` (identity, seeds, mode, profile,
`time_limit`, record, config hash, log version), `decision_point` (index, kind, `can_escalate`, budget), `decision`
(`local` or `escalate`), `segment` (a run of steps by one driver; the clock charges them one at a time), `xvla_act`
(call type, seed, context and the first target's distance), `pi05_request` (every network attempt with its
outcome and round trip), `handoff`, `escalation_abandoned`, `end` (reason), and in branches `restore` and `branch`
(branch, seed index, π0.5 calls, `time_limit` false).

### `points`

One row per measured decision point (routed runs with `record="full"`): everything any decider could use there (§6).

| Field | Type | Meaning |
| --- | --- | --- |
| `run_id` | string | The episode. |
| `index` | int32 | Decision point index (0, 1, ...; one every 10 steps). |
| `step` | int32 | Steps executed before the point. |
| `kind` | string | `chunk` (X-VLA fetched a new 30-step chunk here) or `mid` (10 or 20 steps into a chunk). |
| `chunk_pos` | int32 | Position in X-VLA's current chunk: 0, 10 or 20. |
| `prev_segment_pi05` | bool | Whether π0.5 drove the previous segment. |
| `budget_remaining_steps` | int32 | π0.5 steps left in the budget. |
| `snapshot_id` | string | The point's snapshot (taken after the point's calls). |
| `robot_state` | float64[11] | Controller end-effector position (3), its rotation in X-VLA's 6D form (6), finger positions (2). |
| `proposal` | float32[70] | The next 10 actions X-VLA would execute if the decider continues (10 x 7, executed form as in `steps.action`). |
| `features_main` | float32[1024] | Mean of X-VLA's encoder output tokens (main image and instruction). |
| `features_wrist` | float32[1024] | Mean of X-VLA's wrist-camera image tokens. |
| `n_samples` | int32 | K, the samples per spread measurement (5). |
| `chunk_samples_raw` | float32[] | The K raw chunks drawn when the current chunk was fetched (K x 30 x 20: the executed chunk and K-1 `/sample` chunks), stored once, at the first measured point after they were drawn; null at the other points. |
| `probe_raw` | float32[] | At a `mid` point, the K raw `/sample` chunks of the probe from the measured pose (K x 30 x 20); null otherwise. |
| `chunk_spread_pos_std_mean` | double | Spread of the chunk-time samples over the 10 steps ahead: mean over steps of the RMS distance of the K targets from their mean. |
| `chunk_spread_pos_std_max` | double | Same, the maximum over steps. |
| `chunk_spread_rot_dev_mean` | double | Mean over steps of the mean angle (rad) of the K rotations from their mean rotation. |
| `chunk_spread_grip_disagree_frac` | double | Mean over steps of min(#close, #open) / K. |
| `fresh_spread_pos_std_mean` | double | As `chunk_spread_pos_std_mean`, over fresh samples from the current pose (the probe at `mid` points; the new chunk's samples at `chunk` points). |
| `fresh_spread_pos_std_max` | double | As `chunk_spread_pos_std_max`, fresh samples. |
| `fresh_spread_rot_dev_mean` | double | As `chunk_spread_rot_dev_mean`, fresh samples. |
| `fresh_spread_grip_disagree_frac` | double | As `chunk_spread_grip_disagree_frac`, fresh samples. |
| `drift_pos_dist_mean` | double | Mean over the 10 queued steps of the distance from the queued target to the fresh samples' mean position. |
| `drift_rot_dist_mean` | double | Mean angle (rad) from the queued rotation to the fresh samples' mean rotation. |
| `drift_grip_disagree_frac` | double | Fraction of queued steps whose gripper command differs from the fresh samples' majority. |
| `prev_len` | int32 | Steps in the previous segment (0 at the first point). |
| `prev_driver` | string | Who drove the previous segment. |
| `prev_actions` | float32[] | The previous segment's executed actions (prev_len x 7). |
| `prev_ee_pos` | float64[] | Controller end-effector position after each of those steps (prev_len x 3). |
| `prev_ee_ori_mat` | float64[] | Controller end-effector rotation matrix after each step (prev_len x 9). |
| `prev_gripper_qpos` | float64[] | Finger positions after each step (prev_len x 2). |
| `track_pos_err` | double | X-VLA segments: mean distance between the controller's pose after each step and that step's target. |
| `track_rot_err` | double | Same, rotation angle (rad). |
| `tracking_valid` | bool | Whether the tracking errors apply (false after a π0.5 segment, whose actions are deltas). |

The spread, drift and free-signal definitions are `vlar_harness/measure.py` (its docstring states every choice);
they are pure functions of the raw samples and segments stored above, so new summaries can be computed offline.
Which of these each decider type may use is its profile's `observe_inputs` in the harness config (§6).

## Files

**Snapshots** (`vlar_harness/snapshot.py`, format `vlar-snapshot/1`): an `.npz` loaded with `allow_pickle=False`;
arrays are members (deduplicated by content), everything else JSON with type tags, so loading restores the same
types and bit-identical numbers (`snapshot.from_bytes`). Contents: MuJoCo's integration state with the solver warm
start and the compiled model's fingerprint, every observation's timer and buffered value (cameras included),
controller and gripper state, LIBERO's object states, the X-VLA client's queue and last predicted pose, the episode
state (task, initial state, seeds, step, budget used, driver), the event log up to the point, the point's
measurements, and the provenance header. Restoring needs harness v1 (`Episode.run_branch`).

**Images**: `images/<episode_id>.npz`, member `dpNN/<camera key>` = PNG bytes as uint8; decode with
`vlar_harness.dataset.read_images(root, episode_id)`. Camera keys: `agentview_image` and
`robot0_eye_in_hand_image` (256 px, X-VLA's native observations) and `agentview_image_360` and
`robot0_eye_in_hand_image_360` (360 px, π0.5's), as rendered for the observation the models read at that point
(before any model-side flip or resize).

## Labels (§7)

Labels are computed offline from the event logs, never stored as facts (`vlar_harness/labels.py`,
`labels.point_labels`). For a decider with profile P (one of `none`, `router`, `cascader_nospread`,
`cascader_chunkspread`, `cascader_full`) at decision point d, the label of branch b is

    label(b, d, P) = (number of b's seeds that succeed in time under P) / (number of seeds)

where a seed's full log is the snapshot's `event_log` (the parent's log up to and including d's calls) followed by
the branch run's own log, and "succeeds in time under P" means the log ends with `end` reason `success` and the
simulated clock replayed under P (`clock.score`) is at most the task's time limit at the success step
(`Score.within`; the clock never decreases, so this is what a live limit would have allowed). Seeds: three per
branch (two after the §12 reduction); for a base rollout, the rollout itself is seed 1 of `b1_continue` (its full
log is used). Branches that share a run (equal π0.5 calls at d) share its seeds.

- The clock (`vlar_harness/clock.py`): each step 0.05 s; each X-VLA `/act` its measured latency; each π0.5 request
  the sum over its recorded attempts (timeouts, backoff, then upload, round trip, download and inference for the
  attempt that got through); at each point where escalation is offered, the profile's decision overhead; nothing
  once the budget is spent. Latencies, overheads, the network model and time limits are in the harness config
  (`MANIFEST.json` gives its hash); changing any of them re-scores the same logs.
- Time limits (`vlar_harness/calibration.py`, Phase 0): limit(task) = p90(task) x m, where p90(task) is the 90th
  percentile (linear) of X-VLA's successful base-rollout completion times under profile `none`, and m is the
  smallest of 1.25, 1.5, 1.75, 2.0 such that at least 80% of π0.5's successful parity episodes, timed under `none`,
  finish within their task's limit. A completion time is the clock right after the success step.
- Exclusion: if the clock under P at d, replayed from the snapshot's log, already exceeds the limit (strictly), no
  live decider of type P can reach d, and d is left out of P's training and validation data
  (`PointLabels.excluded`, counted by `labels.exclusion_report`).
- Remaining time: a decider's `time_remaining_s` input at d is the limit minus the clock under its own profile from
  the snapshot's log (`labels.remaining_time_s`), never the value the collecting run saw.
- Runs cut live under a profile (DAgger, evaluation: `time_limit` true) are scored only under their own profile; their
  prefixes up to a decision point are re-scored under every profile, since branches run uncut.

## Split (§7)

Leave-one-task-out: in fold i, task i is the test task, task (i + 1) mod 10 the validation task, the other 8 tasks
train. Splits are by whole task, never by snapshot or episode.

## Known limits

- Mac-verified code paths; every equality and determinism result counts only once re-run on the L4 VM (§11 step 3).
- Values marked `placeholder` in the harness config (latencies, upload size, batching, time limits, gripper
  thresholds) are replaced in Phase 0; data is collected only after the freeze, so a dataset holds one set of them.
