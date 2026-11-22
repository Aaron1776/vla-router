# Harness design notes

How the harness in `src/vlar_harness` implements docs/SPEC.md, and what was learned from the two
reference harnesses' code. Every "verified" item names the test or probe that showed it.

## Reference stack (what the reference run actually used)

| Piece | Version | How we know |
| --- | --- | --- |
| LeRobot | 0.6.1 (tag commit 7e241bd6) | notebook pin; installed files match the tag (`reference.lerobot_files` hashes) |
| LIBERO | **hf-libero 0.1.4** | LeRobot's `libero` extra allows `>=0.1.4,<0.2.0`; 0.1.5 was published 2026-11-16, after the run |
| robosuite / MuJoCo | 1.4.0 / 3.8.1 | notebook printout; hf-libero pins `robosuite==1.4.0`, `mujoco<3.9` |
| LIBERO scene assets | `lerobot/libero-assets` @ 0b3ea86b | the repo's only commits are from 2025-11-03 |
| X-VLA code | 2toinf/X-VLA @ 6c4ecfb8 | notebook pin; file hashes in `reference.xvla_repo_files` |
| Weights | π0.5 7152f13e, X-VLA-Pt c1c4a64a, LoRA b6af8e31 | notebook download cells |

## The episode setup both references share

Verified by `tests/reference/test_sim_reference.py` (exact equality of MuJoCo state and images):

1. `OffScreenRenderEnv(bddl, camera_heights=px, camera_widths=px)` with the defaults `control_freq=20`,
   `hard_reset=True`. X-VLA uses px = 256, LeRobot px = 360; nothing else differs.
2. `env.seed(s)` (this only seeds NumPy's global RNG, which object placement uses and the initial state
   then overrides), `env.reset()`, `env.set_init_state(init_states[k])`.
3. 10 steps of `[0, 0, 0, 0, 0, 0, -1]` with the controller still in delta mode.
4. X-VLA then sets `use_delta = False`; LeRobot (relative control) leaves it True.

## Observation timing (important for snapshots)

robosuite samples every observable (camera images and the proprio observables such as
`robot0_eef_pos`) once per `env.step`, at **physics substep 24 of 25**, not at the end of the step.
The timer phase comes from the two forced updates in `reset()` and `set_init_state()`; `_reset_internal`
resets all observables on every reset, so the phase is the same in every episode of both references.
Verified by `test_observations_are_sampled_at_substep_24_of_25`: the observation's `robot0_eef_pos`
equals the end-effector site position recorded right after substep 24, to the bit (and a one-off probe
showed the same phase in a fresh env per episode and in one env reused across episodes).

Consequences:
- A render taken after `env.step` returns is *not* the observation the models were evaluated on
  (about half the wrist-camera pixels differ at every step of a moving arm). π0.5's 360 px images
  therefore come from two extra robosuite observables (same sensor code, same sampling rate),
  registered once on the env so `_reset_internal` keeps them in lockstep with the native ones.
- The X-VLA client's proprio comes from the controller's cached pose, which is the pose at the start
  of substep 25; LeRobot's comes from the observables (substep 24). Each adapter reads what its
  reference reads.
- **Snapshots:** the original SPEC §7 restore ("set_init_state, …, re-rendering observations") could not
  reproduce trajectories: a re-render shows the post-step state instead of the substep-24 observation,
  and `set_init_state`'s forced update shifts the sampling phase by one substep for the rest of the
  branch. The amended §7 (approved 2026-11-20) restores the full state instead; see "Snapshots" below.

## Snapshots (`sim/state.py`, `snapshot.py`; effort step 2)

A snapshot holds everything needed to continue an episode exactly from a decision point. The list came
from walking the env's whole Python object graph and diffing it across steps and episodes
(`tests/reference/statewalk.py`), not from reading code alone:

| Piece | Why it is needed |
| --- | --- |
| MuJoCo integration state (`mj_getState`, `mjSTATE_INTEGRATION`): time, qpos, qvel, controls, applied forces, **solver warm start** | Dropping only the warm start changes the replay at every decision point tested. |
| Model `body_pos`/`body_quat` | LIBERO places fixtures at reset from NumPy's global RNG, i.e. from the **env seed**, not the initial state (below). |
| Model `site_rgba` | The stove's burner glow, toggled by LIBERO every step and visible in the cameras. |
| Every observation's timer and buffered value (including the 360 px cameras), the observation cache, robosuite's step counters | The observation at a decision point was sampled at substep 24; it is restored, never re-rendered. |
| Controller (goals, cached pose, Jacobians, mass matrix, `use_delta`), gripper (integrated command), robot history buffers | The gripper's command integrates over steps; the controller's cached pose is what X-VLA's measured-pose requests and π0.5's observation read before the next step. |
| LIBERO `object_properties` of objects, fixtures and sites; the visual-state object list | Burner flags; sites share their parent fixture's dicts. |
| X-VLA client: quirk pose, queued actions | Mid-chunk resumes and the quirk. |

Rules that came out of the failures along the way:
- Restore writes into the target env's existing objects and only touches values that differ; lists and
  dicts are updated in place, because LIBERO and robosuite share some of them between objects (site
  properties alias their fixture's, `controller.joint_index` aliases the robot's list).
- Never hold on to a model or data object across a reset: a hard reset builds new ones.
- The compiled model must match: restore checks a digest of every model array except the three fields
  above (cached once per compiled model; computing it takes ~40 ms, a capture afterwards ~0.3 ms,
  a restore ~0.6 ms).
- Restore never calls `set_init_state` (its forced observation update shifts the sampling timers) and never
  re-renders.

File format: one compressed `.npz` per decision point, arrays deduplicated, all other values as tagged JSON
so types and numbers round-trip exactly; loaded with `allow_pickle=False`. About 0.36 MB each (the four
camera images dominate), so roughly 19 GB per 1,000 full-length episodes; storing the dataset's per-point
PNGs separately would duplicate them.

**Verified on the Mac** (`tests/reference/test_snapshot_replay.py`, 17 tests): a task-2 episode mixing X-VLA
chunks and π0.5 segments, with a snapshot at every decision point written through the file format, is
restored from nine decision points (mid-chunk, chunk points, before and after π0.5, both directions) into
the same env, into an env that ran a different episode with a different env seed, and into a never-reset
env. Every replayed step equals the original exactly in every observation value (both camera sizes), the
MuJoCo integration state, the executed action and the success flag; right after each restore, the object
graph, the full model digest and both models' requests equal capture time. Identical seeds reproduce the
run in a reused env with no measurements or snapshots, so measurement calls and snapshot capture change
nothing. The checks are shown to bite: the pre-amendment restore, and restores missing the warm start,
the fixture placement, the gripper command, the observation timers or the controller state, are each caught.

### Fixtures move with the env seed

In 5 of the 10 tasks a fixture's position depends on `env.seed(...)`: the stove (tasks 2 and 8), cabinet
and wine rack (3), desk caddy (5), microwave (9), by up to about 19 mm across seeds. The initial-state
file only holds joint positions. The two reference runs seeded differently (X-VLA `142 + k`, LeRobot
`1000 + k` for initial state k), so their scenes for the same initial state differed in those tasks. The
harness takes the env seed as an explicit input to `setup_episode`. Decision (2026-11-21, SPEC §3): the
**scene seed** is 142 + k in every project run (the X-VLA reference's convention), 1000 + k only in the π0.5
parity run; everything else random comes from a separate **rollout seed** (`vlar_harness/seeding.py`).

## Episode engine and handoffs (`episode.py`; effort step 3)

`Episode` runs one episode on one simulator in any of the three modes and enforces what a decider must not
be able to change: decision points every 10 steps, the 260-step π0.5 budget (counted in executed steps;
`BudgetExhausted` once fewer than 10 remain), the step limit, and the §3 handoff rules:
- escalate: X-VLA's queue is discarded and its remembered pose forgotten, delta mode, a fresh π0.5 call,
  its first 10 actions;
- back to X-VLA: absolute mode; the chunk-point `/act` at that decision point sends the measured pose;
- abandoned escalation (every attempt failed): absolute mode, a fresh X-VLA call (`act_fresh`) from the
  measured pose runs 10 actions and leaves 20 queued; no budget charged.
Before every segment the engine checks the controller is in that model's mode and refuses to run otherwise.
The π0.5 request goes through a `RemoteLink`: since effort step 4 the simulated network (next section) by
default; `DirectLink` (every request through on the first attempt) remains for tests.

Seeds (SPEC §3): `seeding.scene_seed` (142 + k, or 1000 + k under the `pi05_parity` option) feeds only
`env.seed`; every model call's noise seed comes from `seeding.call_seed(rollout_seed, step, model, call)`.

**Verified on the Mac** (`tests/reference/test_handoff.py`, fake policies): forced handoffs in both
directions (mid-chunk, chunk point, consecutive escalations, runs of four) show no arm jump, in both
`xvla_state_mode`s: each π0.5 first step's goal is within one delta step of the arm, X-VLA is always sent
exactly the measured pose after π0.5 and its first target is within 3 cm (the fake-model threshold; the
real-model check on the VM is relative, see the re-run list), and every step ran in its model's
mode. Abandoned escalations hand back from the measured pose and charge no budget. Exactly 26 escalations fit
the budget. Routed always-local is identical to `xvla_only`, which is identical to the authors' own rollout
loop; `pi05_only` calls π0.5 every 10 steps in delta mode. Restoring an engine snapshot into another env
continues exactly. Sensitivity: removing the "forget the remembered pose" rule is caught (stale pose sent,
first target far from the arm), and a skipped mode switch is refused. The same continuity checks run against
the real models on the VM (`test_model_handoff_continuity_on_the_vm`).

## Event log, clock, network, labels and time limits (effort step 4)

Modules: `network.py` (network model), `clock.py` (charges, live and offline clock, scoring), `episode.py`
(event log, live clocks, time cut, π0.5-alone retries, branch runs), `labels.py`, `calibration.py`,
`eventlog.py` (files, checks, handoff distances).

**Event log.** Plain JSON events, one list per run (table in `episode.py`'s docstring): `start`, `xvla_act`,
`decision_point`, `decision`, `pi05_request` (every attempt's draws and outcome), `handoff`,
`escalation_abandoned`, `segment` (a run of executed steps: driver, start step, steps), `end` (`success`,
`step_limit`, `time_limit` or `network`), and `restore` / `branch` for branch runs. A segment's steps are one
event (run-length encoded); everything else is one event per occurrence. The interim file format is JSON lines
(`eventlog.write_jsonl`), exact for every float; the dataset's Parquet tables come later.

**Clock** (`clock.Charges`, `Clock`). Step 0.05 s; X-VLA `/act` and `act_fresh` `latency.xvla_act_s`; a π0.5
request the sum over its attempts (failed: timeout, plus the backoff if another attempt follows; through:
upload + round trip + download + `latency.pi05_infer_s`); a decision point where escalation is offered its
profile's terms for its kind, and one where it is not (budget spent, `can_escalate` false in the log) nothing,
since the decider is not consulted there (SPEC §5, amendment 15); everything else 0. Unknown event types are
refused, so a new event type must be given a charge first. The clock charges the network outcome recorded in the
log and never re-decides it.

**Live equals offline, bit for bit.** The engine keeps one clock per profile. Live, every environment step is
charged as it runs and every other event as it is emitted; offline, `clock.replay` expands each segment into
its steps and charges them one at a time in log order, so both make the same floating-point additions in the
same order. At the end of every episode `Episode.verify_clocks` replays the log under all five profiles and
raises `ClockMismatch` on any difference; a restore rebuilds the clocks by replaying the snapshot's log.

**Network** (`network.SimulatedNetwork`, SPEC §5, amendments 13-14). Per attempt: a drop draw against `p_fail`
and a log-normal round trip (median 0.06 s, sigma = ln(0.30/0.06)/z(0.98) = 0.7837); an attempt fails if dropped
or if upload + round trip + download + π0.5 inference exceeds the 1.0 s timeout (with the placeholder latencies
an attempt fails with probability 0.0521). Draws come from `seeding.network_attempt_seed(rollout seed, step,
model, call, attempt)` through `seeding.uniform` (SHA-256, no generator state): a retry gets fresh draws, the same
rollout seed reproduces the same failure pattern, and nothing depends on the clock, the profile, or other
requests. The server is called once, for the attempt that gets through, with the request's own noise seed, so
retries never change actions. Routed: 3 attempts, then abandoned. `pi05_only`: up to `max_attempts_pi05_only`
(20), then end reason `network`.

Upload size: `upload_image_bytes` (median over requests of the two 224 px JPEGs' summed size, quality 90) +
`upload_other_bytes` (1,024, state, text, framing). The placeholder 20,290 is a Mac estimate (LeRobot's 180° flip
and `resize_with_pad_torch` to 224 px on 160 observations from all tasks; per camera about 11.0 KB main and
9.6 KB wrist). Phase 0 step 2 replaces it: the π0.5 server's `/upload_bytes` returns the JPEG sizes of the images
the policy itself produces (`PI05Policy._preprocess_images` after the same preprocessing as `/act`);
`network.median_upload_image_bytes` takes the median and `config.write_upload_size` stores it in every network
choice (calibration path `options.network.choices.*.upload_image_bytes`). Upload is about 8.5 ms at 20 Mbps.

**Time limit, live and offline.** Scripted runs (base, exploration, branches, scripted baselines) run uncut and
are scored offline (`clock.score(...).within(limit)`: success with the clock at the success step at most the
limit). Runs with a live learned decider start with `time_limit=True` (`reset()`'s flag, SPEC §15, amendment 16):
the run's profile clock is checked after
every step and every model call (`/act`, a decision point's overhead, a π0.5 request), and the episode ends with
`time_limit`; a success on the step that crosses the limit is not a success. Because the clock never decreases,
`clock.first_crossing` on an uncut log finds exactly where a live limit would have cut.

**Branches** (`Episode.run_branch`). Restore, rollout seed `seeding.branch_rollout_seed(parent, point, branch,
seed index)`, no time cut, escalate at the first `pi05_calls` decision points (0, 1, min(3, budget), or the
budget's maximum, as the planner counts them), then local to the step limit. An abandoned escalation counts as
one of the calls. This keeps branches the planner shares (equal call counts) identical; the cost is that in the
rare case of an abandonment (about 1.4e-4 per request) `b4_max` ends with 10 budget steps unused instead of
making one more call. A branch's own log starts with its `restore` and `branch` events; the snapshot's log is
its prefix.

**Labels** (`labels.point_labels`). For one decision point: the snapshot's log (ending with that point's
`decision_point` event), the planner's entry (calls, shares, parent seeds), the branch jobs' own logs and, for a
base rollout, the parent's full log (seed 1 of `b1_continue`). Everything is checked to fit (same point, branch,
seed, call count; the parent's log starts with the prefix), then each (branch, profile) label is the share of
seeds with `score(prefix + own, profile).within(limit)`, for every profile whose clock at the point is within the
limit (the others are excluded; see "Live-limit rules" below).

**Time-limit calibration** (`calibration.calibrate`, SPEC §5, rule approved 2026-11-21). Per task, p90 of X-VLA's
successful base-rollout completion times under `none` (NumPy's linear percentile, re-implemented in pure Python
and checked equal to NumPy bit for bit), times one margin m: the smallest of {1.25, 1.5, 1.75, 2.0} with at least
80% of π0.5's successful parity episodes (pooled, timed under `none` with the network model) within their task's
limit. Refuses cut runs, base rollouts that called π0.5, branches, and missing tasks; errors if no candidate
reaches 80%. `python -m vlar_harness.calibration --base ... --parity ... [--write]` prints the per-task report
and writes `time_limit.per_task_s`, `margin` and `status: calibrated`.

**Handoff diagnostics** (amendment 10). Each `xvla_act` records `first_target_dist_m`, the distance between the
chunk's first target (as it executes, float32) and the controller's measured position at query time, and its
context: `start`, `stretch` (an ordinary chunk boundary in an uninterrupted X-VLA stretch), `after_pi05` or
`after_abandon`. `python -m vlar_harness.eventlog first-targets base.jsonl ... --out first_targets.json` writes the
`stretch` distances of the base rollouts: the file the VM handoff check reads from `VLAR_BASE_FIRST_TARGETS`.

**Verified on the Mac.** Unit tests (`tests/test_network_clock.py`, `tests/test_labels_calibration.py`, pure
Python): the round-trip median and tail, the attempt failure rate and its split into drops and timeouts, the
abandonment rate p³ (with `p_fail` 0.3), retries that succeed (fresh draws; the server runs once), the same seed
giving the same failure pattern over 5,000 requests and another seed a different one, draws independent of
request order; request and overhead charges; scoring and crossing on synthetic logs; exact JSON round trips;
labels under every profile with planner entries, sharing and parent seeds, and refusals of logs that do not fit;
calibration choosing m = 1.5 on data built to discriminate, the no-candidate error, refusals, and an exact
calibration-file round trip; the upload-size calibration path. Simulator tests
(`tests/reference/test_clock_live.py`, fake policies): all five live clocks equal offline replay bit for bit over a
full routed episode with retries, abandonments and both point kinds, after a JSON round trip, and after restores
from four snapshots in another env (clocks restored exactly; continuing reproduces the rest of the log); once the
budget is spent every profile's clock grows by exactly the same charges (no decision overhead, amendment 15); π0.5
alone with a lossy network executes exactly the actions of a perfect network, only slower; a network that always
fails ends π0.5 alone after 20 attempts (23.8 s, end reason `network`, server never called); a live limit cuts at
the same float time and step as `first_crossing` on the uncut run, for a crossing inside a run of steps, at a
decision point's overhead and at a retried π0.5 request; branches reproduce across envs for the same seed index
and differ for another, `b1_continue` repeats the stored queue identically in every seed; labels from real branch
logs (with a hash-based stand-in success check) match direct scoring and never rise with more overhead; every
`xvla_act` records the context and a distance equal to the controller's goal after the next step minus the
measured position. Sensitivity (a scratch script, recorded here): keying network draws by step without the
attempt number fails the retry and abandonment tests; replay charging a segment as one multiplication, and a
restore that does not rebuild the clocks, both raise `ClockMismatch`.

## Live-limit rules (amendments 19-23, follow-ups to step 4)

**Where a live limit is allowed (19).** `api.check_reset(mode, profile, time_limit)` accepts `time_limit=True` only
in routed mode with a profile other than `none`, and only a real bool; `Episode.start` calls it before it touches
the simulator, and `eventlog.check` refuses a log whose start event breaks it.

**The flag (20).** The start event records `time_limit` true or false next to `time_limit_s` (they must agree), as
do `EpisodeState` (so snapshots carry it) and `EpisodeResult`. `run_branch` clears it and the `branch` event records
`time_limit: false`. `eventlog.check` and the clock refuse logs without the flag; there are no older logs to accept.

**Cut runs (21).** `clock.run_header` finds the event that set up the run a log's outcome belongs to: the start
event, or the last `branch` event for a snapshot's log followed by a branch's own log. If that header is a start
event with `time_limit: true`, `clock.score` and `clock.first_crossing` refuse every profile but the run's own
(`CutRunError`); `replay` (the raw clock: the engine's live-equals-offline check, a point's time) is not an outcome
and runs under any profile. So a DAgger parent's own log is scored only under its decider's profile, while its log
up to a decision point plus an uncut branch is scored under every profile. Calibration refuses runs with the flag
set; labels refuse a cut parent's full log used as a seed under any other profile (`LabelError`).

**Exclusions (22) and remaining time (23).** `labels.point_labels` returns a `PointLabels`: the labels for every
profile whose clock at the point, replayed from the snapshot's log, is within the limit; the profiles `excluded`
because that clock already exceeds it (strict, as the live cut); and per profile the clock at the point (`time_s`)
and the remaining time (`remaining_s`, the limit minus `time_s`). `labels.remaining_time_s` gives the same input for
one profile; it is the decider's training input, never the value the collecting decider saw (`observe()` in a
scripted run reports infinity; a DAgger rollout's decider saw its own profile's time). `labels.exclusion_report`
counts points, exclusions and kept points per profile.

**Verified on the Mac.** Unit tests (`tests/test_network_clock.py`, `tests/test_labels_calibration.py`): every
mode and profile combination for `check_reset`; logs without the flag, with a flag that disagrees with the value,
or with a live limit where none is allowed are refused; a cut run scores only under its own profile, a branch from
it under every profile, and a branch event claiming a limit is refused; exclusions at a limit between two profiles'
clocks, exactly at a clock (kept) and one ulp below (excluded), with the report's counts; labels from a parent cut
under `router` equal those from an uncut parent with the same events, and its full log as a seed is refused under
other profiles; the remaining time for every profile equals the limit minus that profile's replayed clock, and the
full cascader's is below what the router saw. Simulator tests (`tests/reference/test_clock_live.py`): `start` refuses
all 11 disallowed combinations before touching the simulator; real cut runs carry the flag in the log and result and
are refused under other profiles; a DAgger-style parent cut live under `router` (limit halfway between the router's
and the full cascader's clocks at point 12) gives at every point a `cascader_full` remaining time equal, bit for bit,
to the limit minus the engine's own live `cascader_full` clock and below what the router saw; branches from three of
its snapshots run uncut and give labels under every kept profile equal to direct scoring, with `cascader_full` kept
at point 4 and excluded at points 10 and 16.

Sensitivity (a scratch mutation script: each rule broken in the source, its tests run, the file restored byte for
byte). All 15 mutations are caught:

| Rule broken | Caught by |
| --- | --- |
| 19: `check_reset` accepts any combination | `test_reset_allows_a_live_limit_only_in_routed_runs_with_a_decider_profile` (11 cases), `test_every_log_records_the_live_limit_flag`, `test_start_refuses_a_live_limit_outside_routed_decider_runs` |
| 19: `Episode.start` skips the check | `test_start_refuses_a_live_limit_outside_routed_decider_runs` |
| 20: start event without the flag; result flag always false; branch keeps the parent's flag; `eventlog.check` accepts logs without it | `test_live_time_limit_cuts_where_offline_scoring_says`, `test_dagger_parent_under_router_rescored_for_cascader_full`, `test_every_log_records_the_live_limit_flag` |
| 21: `score` or `first_crossing` without the guard; calibration accepting cut runs; labels accepting a cut parent's full log as a seed | `test_a_cut_run_is_scored_only_under_its_own_profile`, `test_live_time_limit_cuts_where_offline_scoring_says`, `test_calibration_refuses_the_wrong_runs`, `test_labels_from_a_parent_cut_under_another_profile` |
| 21 applied literally (the header is always the start event, so a cut parent's prefix is refused under other profiles) | `test_a_branch_from_a_cut_run_is_scored_under_every_profile`, `test_labels_from_a_parent_cut_under_another_profile`, the DAgger-style test |
| 22: no exclusion; `>=` instead of `>` | `test_points_past_the_limit_are_excluded_per_profile`, `test_labels_follow_each_profiles_clock`, the DAgger-style test |
| 23: remaining time under the collector's profile (in `remaining_time_s`, in `PointLabels.remaining_s`) | `test_remaining_time_is_recomputed_under_each_deciders_profile`, the DAgger-style test |

## Time limit (SPEC §5, rule approved 2026-11-21)

Per task: the 90th percentile of X-VLA's successful-episode completion times under profile `none` (from the
base rollouts, uncut), times one margin m for all tasks. m is the smallest of {1.25, 1.5, 1.75, 2.0} such
that at least 80% of π0.5's successful parity episodes (pooled over tasks), timed under profile `none` with
the network model, finish within their task's limit. Computed in Phase 0 step 6 by `calibration.py` (above)
and written to the calibration file; reported per task with the share of π0.5 successes that fit.

## Reference-run files (`vlar_ops/reference_runs.py`)

`References/` (read-only): `task_<t>/eval_info.json` (π0.5, 50 successes per task in episode order) and
`progress.jsonl` (X-VLA), committed byte for byte (`.gitattributes`: `References/** -text`) with their SHA-256
pinned in `src/vlar_ops/configs/reference-run.sha256`. `vlar-ops reference check` first verifies those checksums
(a single changed byte fails), then the layout and the totals the user confirmed:
π0.5 478/500; X-VLA 412/500 at 900 steps, 405/500 within 520; 500 episodes each, no duplicates or gaps.
Episode e used initial state e, with scene seed 1000 + e (π0.5) or 142 + e (X-VLA). `compare()` pairs a
project run with the reference on the same initial states (Wilson intervals, paired difference with its
interval, exact McNemar test); on the π0.5 parity subset the reference scored 97/100.

## X-VLA adapter (`adapters/xvla.py`)

The authors' `ClientModel`, subclassed: same `_format_query`, same quirk, same 6D->axis-angle
conversion and gripper rounding. Only two changes: the transport, and `step()` split into
`query()` (lines 192-207) and `pop()` (lines 209-212) so routed mode can ask for a chunk before the
decision. Verified equal to `ClientModel.step()` over three chunks, quirk included.

Facts from the client worth knowing: the main view is flipped 180° but the **wrist view is not**;
the gripper slot of the proprio is ignored anyway (the `ee6d` action space zeroes channels 9 and 19
of the proprio and of the noisy actions before the transformer).

Measurement calls do not touch the client's state. Mid-point `/sample` probes and `/features` send
the **measured** pose (the authors' query with `proprio = None`), so fresh samples describe what
X-VLA would do from the current state; the chunk-point `/sample` reuses the exact `/act` payload, so
its extra chunks are samples of the same request as the executed chunk.

## π0.5 adapter (`adapters/pi05.py`)

A LeRobot `LiberoEnv` that is never reset (so it never builds its own simulator) supplies
`_format_raw_obs` and its observation space; gymnasium's `concatenate` batches the result as
`SyncVectorEnv` does. Verified equal to LeRobot's own vectorised env, built at 360 px, from the first
reset through 25 delta-mode steps (images, all robot-state arrays, dtypes, success flags).

## Model servers (`servers/`)

- **X-VLA** (`xvla_server.py`, X-VLA environment): loading is `deploy.py`; preprocessing is the
  authors' `/act` handler; generation is `generate_actions` split into the encoder pass and the
  denoising loop so the encoder output can be pooled and shared by `/sample`. Verified bit-equal to
  the authors' `/act` handler with the same noise (random-weight model on CPU; rerun with weights on
  the VM). TF32 and cuDNN flags are left at PyTorch's defaults, as `deploy.py` does.
- **π0.5** (`pi05_server.py`, LeRobot environment): `lerobot-eval`'s loading (TF32 matmuls and cuDNN
  benchmark on, the reference CLI overrides, the checkpoint's processors) and its per-step processing;
  returns all 50 un-normalised actions. The equivalence test against `select_action` +
  post-processors with pinned noise needs the weights and the gated tokenizer, so it runs on the VM.
- **Pooled features** (SPEC §3 as amended): `enc_mean`, the mean over all encoder output tokens (first
  image + 50 text tokens, padding included since the encoder attends to it), and `wrist_mean`, the mean
  over the wrist view's image tokens, which bypass the encoder. 1024 floats each; both are stored and
  every decider takes both.
- **Noise**: with seeds, each sample's initial noise comes from its own CPU generator, so it depends on
  nothing but its seed (not the batch, not the GPU). Without seeds the servers behave like the
  references (global RNG). Seeds are derived by the harness (`seeding.call_seed`).

### Batching (effort step 5, amendments 24 and 26)

`servers/batching.py`: one `Batcher` per endpoint, a request queue and one worker thread that owns the model. A
handler's `submit` prepares the request in its own thread (a bad request fails there, alone) and waits; the worker
takes the oldest request, then later ones in arrival order while they fit the endpoint's capacity, for up to the
wait window, and runs one batch. Capacity is a vector (X-VLA `/sample`: one encoder row and K denoising rows per
request); nothing overtakes the head of the queue; a request larger than the capacity is refused. If a batch raises,
each of its requests runs again alone, which is exact because of batch invariance, and only the ones that fail alone
fail. Unit tests (`tests/test_batching.py`, fake model): own results, the wait window, capacity splits, error
isolation, 8 concurrent clients.

X-VLA (`xvla_server.py`, `XVLABatching`): every seeded `/act`, every `/sample` and every `/features` goes through
its endpoint's batcher. Each batch runs at a fixed geometry: the encoder on exactly `max_batch` rows, `/act`'s
denoising on `max_batch` rows, `/sample`'s on `sample_rows` rows (a request's samples are consecutive rows), padded
with copies of the batch's first request; each row's noise comes from its own seed. Batched requests must carry
exactly the two views the harness sends (`forward_vlm` encodes only the valid views of the whole batch, so a third
view would change the image encoder's batch size) and the configured step count. The encoder geometry is shared by
the three endpoints, so their pooled features agree bit for bit (§3). An unseeded `/act` keeps the reference path
(`act_unbatched`), outside the batchers. `common.deterministic_torch()` (called by `load`): no cuDNN autotuning,
cuDNN deterministic, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; TF32 settings
stay as each reference has them. Config: `servers.batching` (`max_batch` 8, `xvla_sample_rows` 40, `wait_window_ms`
5), placeholders until Phase 0 step 2.

**Verified on the Mac** (`tests/reference/test_batch_invariance.py`, X-VLA environment, random weights, CPU,
geometry 3 requests / 6 sample rows / 10 steps): a request's `/act` result and features are bit-identical alone,
with different companions and at every position; the same for a `/sample` request with 2 samples among requests of
1 and 3 samples; `/act`, `/sample` and `/features` give bit-identical features for the same payload; three
concurrent clients through the real queue land in one batch and each gets its alone result; requests that break the
geometry are refused and unseeded ones bypass the batchers; at `max_batch` 1 the batched path equals the seeded
batch-of-one path (which `test_model_servers.py` ties to the authors' handler).

Findings:
- On the CPU the denoiser's result depends on the batch size: row 0 of an unpadded batch of 2 or 8 differs from a
  batch of one by up to 6e-8 (10 steps; a scratch probe). Fixed padding removes this; the mutation that drops the
  denoising rows' padding is caught (as is a batch-level noise generator). The encoder gave the same bits at every
  batch size tried on the CPU (1, 2, 3, 8), so dropping its padding is not caught on the Mac; it stays, because GPU
  kernel choice can change with batch size, and the VM run shows whether the L4 needs it.
- At the test geometry (3 rows) the padded result equals the batch of one exactly on the CPU; at 2 and 8 rows it
  does not. The production geometry's results therefore need not equal a batch of one, only each other: parity is
  judged by success rates, not bits.
- Deterministic mode makes PyTorch fill uninitialised memory with NaN. A randomly initialised X-VLA has one tensor
  random init never touches (Florence-2's raw `image_projection` Parameter, created with `torch.empty`); before,
  the random-weight tests read leftover memory there. The test builder now seeds every non-finite tensor; a
  checkpoint overwrites it in production. The NaN fill costs time, so Phase 0 step 2 measures it with the padding.

π0.5 (`pi05_server.py`, `PI05Batching`, part 2). `PI05Service.load` is lerobot-eval's loading with one recorded
deviation, `deterministic_torch()` instead of `cudnn.benchmark = True` (TF32 stays on, as in the reference). Seeded
`/act` requests go through the batcher: each is pre-processed alone (batch dimension 1) in its caller's thread and
checked against the fixed geometry (`check_geometry`: exactly the checkpoint's two cameras, the prompt exactly 200
tokens, right-padded, not filling them); a batch is padded to `max_batch` rows with copies of its first request,
tensors concatenated, each row's noise from its own seed, `predict_action_chunk` once, then each real row
post-processed alone, one action at a time, as the unbatched path does. Unseeded requests (`act_unbatched`, the
reference path) and `/upload_bytes` stay outside the batcher.

Reading LeRobot's π0.5 inference (`sample_actions`, `embed_prefix`, `denoise_step`, `euler_integrate`,
`_preprocess_images`, `make_att_2d_masks`): every operation is per row; the attention mask is additive with a finite
large negative value, so masked keys get exactly zero weight; the time is a scalar expanded over the batch; the empty
camera is the same padding image in every row. Nothing couples rows; only kernel choice can depend on the batch's
shape, which the fixed geometry pins.

Fixed token length (amendment 26). π0.5's prompt is the instruction plus the robot state as text (8 numbers in 256
bins, so its token count varies with the state as well as the task); LeRobot's tokenizer step pads it to the
checkpoint's 200 tokens (`padding="max_length"`, right) with an attention mask. X-VLA's processor pads every
instruction to 50 tokens and the encoder attends to the padding, as in the authors' model. Both servers now refuse a
request of another length or one that fills the whole length (possibly truncated). LIBERO-10's instructions take 10
to 21 of X-VLA's 50 tokens. X-VLA's part-1 tests used one instruction for every request; they now mix instructions
from different tasks.

Shared per-call state (audit of both servers' pre- and post-processing). LeRobot's `ProcessorStep.__call__` stores
each call's transition on the step object, and `TokenizerProcessorStep.observation` reads the prompt back from it
(`self.transition`), so two requests pre-processed at once could swap prompts; the (un)normaliser moves its
statistics to the input's device and dtype inside a call (`_apply_transform` → `to`). The checkpoint's pipelines
(rename, to-batch, normaliser, π0.5 state-prompt step, tokenizer, device; un-normaliser, device; LIBERO's env step;
an empty env post-processor) have no relative/absolute action steps, so pre- and post-processing share no step. The
authors' X-VLA processor and model keep no per-call state (an AST scan for attribute writes outside construction
found none on any call path), but its Hugging Face fast tokenizer holds mutable padding and truncation settings. Each
server therefore runs every processor call under a module-level `PROCESSING_LOCK`; the model runs under the device
lock, one batch at a time; lock order device, then processing.

**Verified on the Mac** (`tests/reference/test_batch_invariance.py -k pi05`, simulator environment with LeRobot's `pi`
extra; `pi05_testkit.py`): LeRobot's own π0.5 code at a tiny size (one layer in the vision tower, the language model
and the action expert; the checkpoint's widths and heads; a 4,096-entry vocabulary; random weights) with the
checkpoint's own processor pipelines loaded from its JSON files (synthetic statistics) and a stand-in tokenizer that
splits numbers into digits as Gemma's does. Bit for bit: a request alone, with companions and at every position, its
batch-mates from different tasks with prompts of 52, 39 and 24 tokens, in float32 and in the checkpoint's bfloat16;
other ids in the padding leave a result unchanged while one changed real token changes it; every LIBERO-10 prompt fits;
max_batch 1 equals `act_unbatched` (on the CPU the padded batch of 3 also equals it); six simultaneous requests keep
their own prompts (with LeRobot's race window widened by a sleep) and outputs; a stand-in policy whose rows change with
the batch size and the prompt tensor's length still gives every request its alone result. Requests that break the
geometry are refused. Mutation check (a scratch script, each rule broken in the source, the file restored byte for
byte), 6 of 6 caught:

| Rule broken | Caught by |
| --- | --- |
| rows not padded to `max_batch` | the stand-in policy test only (the tiny model's CPU results do not depend on the batch size) |
| one batch-level noise generator | invariance (float32, bfloat16), stand-in policy, simultaneous requests |
| no `PROCESSING_LOCK` | simultaneous requests (prompts swapped) |
| outputs handed back in the wrong order | invariance (float32, bfloat16), stand-in policy, simultaneous requests |
| prompts trimmed to the batch's longest | reference path, stand-in policy, simultaneous requests (the real code's result does depend on the token length) |
| geometry check skipped | the geometry test |

**Open (X-VLA, part 2).** X-VLA's tests now mix instructions of 10 to 21 tokens and pass, as does the token-length
test; the new simultaneous-requests test fails intermittently on the CPU: one of six requests sent at once gets an
action that is not bit-identical to its alone result, though its prompt is right (HANDOFF.md, "Test suites"). Until
it is explained, X-VLA's batch invariance under concurrent arrival is not shown, and its mutation check is not run.

## Rendering only what is used (SPEC §12, approved with conditions)

On this Mac, an env step costs 12.8 ms with the 256 px cameras and 19.9 ms with the 360 px ones added.
Models and snapshots only use images at decision points (every 10 steps), so enabling the camera
observables only on the step before a decision point, with their timers kept in step with an always-on
observation, could remove most rendering cost. On the VM, EGL rendering shares the L4 with model
inference, so the GPU time it frees matters more than the CPU saving. Approved only after restore is in
place, and only if it passes the same exact-equality tests as always-on rendering.

## Platform rule (SPEC §11 step 3 as amended)

The Mac runs (CPU MuJoCo, CGL rendering) show the code paths are right. Every equality and determinism
test is re-run on the L4 VM (Linux, EGL rendering, CUDA) before the freeze; only VM results count for
parity and determinism, because rendering and floating-point results can differ across platforms.

VM re-run list (each in its own environment, `MUJOCO_GL=egl`):

| Environment | Command | Covers |
| --- | --- | --- |
| simulator | `pytest tests/reference/test_sim_reference.py tests/reference/test_snapshot_replay.py tests/reference/test_handoff.py tests/reference/test_clock_live.py` | adapters vs both references; snapshot restore and replay; determinism; handoffs, budget, run modes; live clocks equal offline replay for every profile; network retries and the `pi05_only` cap; live time cut vs offline crossing; branch determinism; labels from branch logs; the live-limit rules (amendments 19-23: refused combinations, the flag, cut runs scored only under their own profile, the DAgger-style parent re-scored for `cascader_full`); handoff diagnostics; sensitivity checks |
| simulator, both servers up, after Phase 0 step 5 | `VLAR_XVLA_SERVER=127.0.0.1:8777 VLAR_PI05_SERVER=127.0.0.1:8778 VLAR_BASE_FIRST_TARGETS=<json list> pytest tests/reference/test_handoff.py -k vm` | handoff continuity with the real models, **relative** (SPEC §11 step 3, amendment 10): X-VLA's first target after a handoff must be at or below the 99th percentile of its first-step target distances at ordinary chunk boundaries (no handoff) in the base rollouts; the JSON list comes from the base rollouts' event logs with `python -m vlar_harness.eventlog first-targets <base logs> --out <json list>`. The fixed 3 cm check is for the fake-model tests on the Mac only. |
| simulator + π0.5 server, Phase 0 step 2 (a measurement, not a re-run) | observations at decision points of π0.5-alone episodes sent to `PI05Client.upload_bytes`; `network.median_upload_image_bytes`; `config.write_upload_size` | the upload size of decision 14 from the images the policy itself consumes; replaces the Mac placeholder (20,290 bytes) before the parity run |
| main | `vlar-ops reference check` | the reference files' pinned SHA-256 and totals, before any parity run |
| X-VLA | `VLAR_XVLA_MODEL_PATH=… VLAR_XVLA_LORA_PATH=… pytest tests/reference/test_model_servers.py -k xvla` | server vs the authors' handler with the real weights and LoRA |
| X-VLA | `VLAR_XVLA_MODEL_PATH=… VLAR_XVLA_LORA_PATH=… VLAR_TEST_DEVICE=cuda pytest tests/reference/test_batch_invariance.py -k xvla` | batch invariance on the L4 at the production geometry (8 requests, 40 sample rows), bit for bit: alone, with companions, at every position; features equal across endpoints (amendment 24) |
| LeRobot (with the `pi` extra) | `VLAR_PI05_CHECKPOINT=… VLAR_TEST_DEVICE=cuda pytest tests/reference/test_batch_invariance.py -k pi05_real` | π0.5 at the production geometry with the real weights and tokenizer, bit for bit: alone, with companions, at every position and in a full batch, batch-mates from different tasks with different prompt and state token lengths; padding ids cannot change a result; every LIBERO-10 prompt fits the 200 tokens; max_batch 1 equals `act_unbatched`; simultaneous requests keep their prompts and outputs (amendments 24, 26). Deterministic mode must not raise on any CUDA kernel π0.5 uses. |
| both servers, Phase 0 step 2 (a measurement) | throughput at the fixed geometry against unpadded batches, with deterministic mode on and off | the cost of fixed padding and of deterministic mode; if padding is too slow, stop and bring the numbers to the user before relaxing anything (amendment 24) |
| both servers, Phase 0 step 2 (a measurement) | servers started at `--max-batch 1` (X-VLA also `--sample-rows 5`): median latency of `/act`, `/features`, `/sample` K = 5 and π0.5 inference | the clock charges, unpadded, as a deployed robot runs the calls (amendment 25); written to `latency` in the config; the padded figures never are |
| LeRobot | `VLAR_PI05_CHECKPOINT=… pytest tests/reference/test_model_servers.py -k pi05` | server vs lerobot-eval's per-step processing (needs the gated tokenizer) |
| main | `pytest tests` | config, queue, spend guard, packaging; network model, clock charges, labels and calibration on synthetic logs |
| simulator | `pytest tests/test_labels_calibration.py -k numpy` | the pure-Python percentile equals NumPy's bit for bit (the main environment has no NumPy) |

## Running the reference tests

```bash
.venv-sim/bin/python -m pytest tests/reference                    # simulator + adapters, exact
VLAR_XVLA_CONFIG_DIR=third_party/hf/X-VLA-Pt VLAR_TEST_DEVICE=cpu \
  .venv-xvla/bin/python -m pytest tests/reference/test_model_servers.py -k xvla
```

The local environments (`.venv-sim`, `.venv-xvla`, `third_party/`) are git-ignored; the VM uses the
container images built from the same pins.
