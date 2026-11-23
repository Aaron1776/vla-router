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
1 and 3 samples; `/act`, `/sample` and `/features` give bit-identical features for the same payload; six requests of
different tasks give the same bits at every position of a full batch, and sent at once through the real queue they keep
their prompts and get their alone results; requests that break the geometry are refused and unseeded ones bypass the
batchers; at `max_batch` 1 the batched path equals the seeded
batch-of-one path (which `test_model_servers.py` ties to the authors' handler).

Findings:
- Batch size on the CPU (corrected after the concurrency fix below). Part 1 saw the denoiser depend on the batch size
  (row 0 of an unpadded batch of 2 or 8 differed from a batch of one by up to 6e-8). That was the whole-batch gripper
  sigmoid, not the model: with rows post-processed alone, dropping the denoising rows' padding changes no result of
  the real-model tests, and the encoder gave the same bits at every batch size tried (1, 2, 3, 8). On this CPU X-VLA's
  rows therefore do not depend on the batch size. The padding stays because GPU kernel choice can; on the Mac the
  stand-in model test (`test_xvla_fixed_geometry_makes_rows_independent_of_their_batch`, rows that change with the
  batch size) checks the padding mechanics, and the VM run shows whether the L4 needs it.
- The production geometry's results need not equal a batch of one, only each other (GPU kernels can differ by batch
  size): parity is judged by success rates, not bits.
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

**The X-VLA simultaneous-requests failure (part 2; explained and fixed).** The test failed intermittently on the CPU:
one of six requests sent at once got an action that was not bit-identical to its alone result, though its prompt was
right. A scratch probe logged every batch's composition (request ids by position, per-row hashes of every encoder and
denoiser input and output, padding rows included) over repeated concurrent runs: request 5 (task 2) failed whenever it
was last in its batch (`[0, 1, 5]`, `[1, 2, 5]`) and passed first or second (7 batches). Each captured batch replays
deterministically, in the main thread, in a fresh thread and from the alone-prepared inputs, so neither timing nor
threads were involved. In every batch the padding rows were exact copies of the first request (encoder inputs, proprio,
domain id, noise), and with forward hooks on every module no module's output for the request's rows differed from its
alone run (per-row hashes; the encoder's views and windows are batch-major). The one differing value, step 29 channel 9
(a gripper) by 5.96e-8 (one ulp), came after the model: the authors' `postprocess` applies a sigmoid to the gripper
channels of the whole batch at once, `[rows, 30, 2]`. PyTorch's CPU elementwise kernels compute the last (element count
mod 8) elements of a tensor with scalar code (`std::exp`) and the rest with the vectorised exp, and the two differ in
the last bit for about 4% of inputs; at 3 rows (180 elements) the last row's last 4 values take the scalar path. With
random gripper logits, 31 of 200 three-row batches of identical rows had a last row that differed from the first; none
at 2, 4, 6, 8 or 40 rows, whose element counts are multiples of 8, so the production geometry (8 and 40 rows) does not
show it on this CPU. The earlier tests missed it because it depends on the values (requests a and b happened to round
alike at the last position). Fix (`XVLAService.denoise`): the authors' post-processing runs on each row alone, exactly as
in the authors' batch of one, so every row is split the same way at any position and batch size. CUDA's elementwise
kernels apply one function to every element, so this is a CPU effect; the VM run checks it.

New tests: `test_xvla_fixed_geometry_makes_rows_independent_of_their_batch` (the batching mechanics with a stand-in
model whose rows change with the batch size; needs no model), `test_xvla_rows_are_post_processed_alone` (needs no model: a stand-in transformer gives every row the same
raw output, with gripper logits for which the scalar and vectorised sigmoids differ on the machine; every row at batch
sizes 1 to 8 equals the batch of one), `test_xvla_every_position_computes_the_same_bits` (each of the six requests at
every position of a full batch of copies of itself), `test_xvla_captured_concurrent_batches_replay_to_the_alone_results`
(the two captured batches). The simultaneous-requests test now records each batch's composition and prints it on
failure, so any failure can be replayed without the batcher's timing.

X-VLA mutation check after the fix (a scratch script; each rule broken in the source, the X-VLA tests run, the file
restored byte for byte), 7 of 8 caught:

| Rule broken | Caught by |
| --- | --- |
| post-processing over the whole batch (the old code) | rows post-processed alone, every position, captured batches, simultaneous requests |
| token length trimmed to the batch's longest instruction | `/act` invariance, captured batches, simultaneous requests, batch of one = reference path |
| no token-length check | the token-length test |
| encoder rows not padded | the stand-in model test only (X-VLA's own rows do not depend on the batch size on this CPU) |
| denoising rows not padded | the stand-in model test only (the same; before the fix the whole-batch sigmoid made it visible) |
| one batch-level noise generator | `/act` and `/sample` invariance, every position, captured batches, simultaneous requests, batch of one = reference path, the seeds test, the stand-in model test |
| results handed back in the wrong order | the stand-in model test (run against the fast tests only) |
| no `PROCESSING_LOCK` | not caught: not observable on the Mac |

`PROCESSING_LOCK`: the fast tokenizer changes its padding and truncation settings only when a call asks for other ones
(transformers' `set_truncation_and_padding`), and every X-VLA call asks for the same, so after the first call concurrent
calls only read them. The lock stays: it covers the first calls' update, where a collision would raise an error (never
a wrong prompt), and any later code that tokenizes with other settings.

### π0.5 padded to 8 vs the batch of one on the L4 (stage A failure, diagnosed 2026-11-23)

Stage A's row `pi05-batch-invariance-l4` failed only `check_pi05_reference_path`: the test's random-image request
(task 6, seed 105) padded to 8 rows differed from the batch of one by 1.94. Diagnosis on the L4 with the real
checkpoint (`ops/vm/phase0/debug/pi05_batch_debug.py`, run as step `095`; outputs in the bucket under
`phase0/logs/debug-pi05/` and locally in the git-ignored `outputs/debug-pi05/`):
- **Not a bug.** Every input of row 0 is bit-identical to the batch of one (noise, tokens, attention masks, the three
  images, the 4-D prefix mask, position ids); every padded row is bit-identical to row 0; the max_batch-1 batched path
  equals `act_unbatched` bit for bit on every request (5 requests, 2 real); repeats are bit-identical.
- **First difference:** SigLIP's patch-embedding convolution (float32, TF32 allowed as in the reference): identical
  input, row 0 changes once the batch has 2 or more images (max 5e-4, relative 2e-4); run alone, the same module on the
  same input repeats bit for bit at 1 row and gives one different result at 2-16 rows. The kernel profile shows cuDNN
  and cuBLAS picking other kernels at 1 row than at 8 (GEMV and split-K at 1 row, other tile shapes and implicit-GEMM
  convolution at 8). Every batch size from 2 to 16 differs slightly from the others too.
- **Growth, then one flip:** relative differences grow through the 27 vision layers (2e-5 to 3e-4) and the bfloat16
  language model (0.3% to 4%); the denoising steps carry them (x_t 0.004 to 0.02 by step 6), and on this request
  one gripper value of a far-ahead action (step 34 of 50; the first 10 are executed, max difference 0.008) sits on
  the open/close boundary and goes to -0.96 instead of +0.98 in the last three steps. Splitting the stages: the
  prefix at 8 rows with the denoising at 1 row already gives the flip; the reverse gives 0.03.
- **The request is on a knife edge either way:** in the batch of one alone, adding 2^-20 to a single pixel gives the
  same 1.94 flip; a real observation changes by 0.006-0.014 under the same perturbation.
- **Real observations** (50 decision points, 5 per task, following the batch of one's actions): max |padded -
  one| median 0.0037, p90 0.0057; executed actions (first 10) at most 0.0056; one flip (task 7, a gripper value at
  step 27). Random-image requests: median 0.0036, one flip at step 2 (task 3).
- **Which result is off:** neither. In float32 with TF32 off the two paths agree within 7e-7 on ordinary requests and
  within 0.0045 on the two flipping ones (no flip). Against that float32 result, the bfloat16 batch of one is the
  farther of the two on both flipping requests (1.94 and 1.54, against 0.32 and 0.42 for the padded result), and
  about as far on ordinary ones (0.004-0.010 against 0.003-0.005).
- `allow_bf16_reduced_precision_reduction = False` changes nothing material (it moves the batch of one by 1.95 on the
  flipping requests itself). Timing on the L4 (model span, deterministic): 0.80 s per request at 1 row, 1.80 s per
  batch of 8 (0.225 s per request), so padding to 8 gives 3.6x the π0.5 throughput of running it unbatched.

## Rendering only what is used (SPEC §12, amendment 7)

Models, measurements and snapshots read images only at decision points: the observation left by the
step that ends on a multiple of 10 (and by the last settle step, for decision point 0). Config option
`render`: `used` (default, value `decision_points`) or `all` (`every_step`, as both references do).

How (`sim/libero_sim.py`): each of the four camera observables (256 px and 360 px, both cameras) keeps
robosuite's sampling timer and sampling logic untouched; only its sensor is wrapped in a gate. During a
step whose observation is used the gate is open and the sensor renders; otherwise it returns a
placeholder without rendering. So the timers stay in step with the always-on observations by
construction (they never stop or reset), rather than by copying timer values. Safeguards:
- On a step that did not render, `sim.obs` has no camera keys, and both adapters call
  `sim.require_images()` first (`ImagesNotRendered`), so a placeholder can never reach a model.
- Each rendering step must sample every camera exactly once, or it raises `RenderPhaseError` (a moved
  sampling phase would otherwise leave a placeholder as the decision point's image).
- Snapshots record `sim.images` (are the buffered images this step's renders?); restore hides them when
  they are not, whatever the target env's mode. At decision points it is always true.
- A hard reset gives robosuite's own cameras fresh sensors; the gate is re-installed after every reset
  and checked before every step. The forced observation update in `set_init_state` renders only under
  `all`; `reset` still renders robosuite's own two cameras once per episode (it installs their fresh
  sensors before its forced update). The settle steps overwrite both.

**Verified on the Mac** (`tests/reference/test_render_used.py`, 9 tests): one scripted routed episode
(task 2, stove and burner light; X-VLA chunks, mid-chunk points, π0.5 segments, handoffs both ways;
measurements and a snapshot at every decision point) runs to the 520-step limit under `used` and
under `all`. At every step the action, MuJoCo's integration state, the success flag, every non-camera
observation and every observable's sampling timer are identical; at every decision point the images,
both models' requests and the snapshot file are byte-identical. `used` renders exactly 4 x 52 times
(`all`: 4 x 520). Unrendered steps refuse image reads; a snapshot between decision points restored into
an always-on env keeps its images hidden until the next rendering step; rendering one step late or
early is refused, not silent; a rendering step that misses its sample raises. The other reference tests
(snapshot replay, determinism, handoffs, live clocks) now run under `used`; the two reference-parity
tests run under both modes (`used`: images at decision points, state at every step). Mutation check: 3 of
3 caught (gate never closes; unrendered images not hidden; gating by `set_enabled`, which resets the
timers: `RenderPhaseError` at the first rendering step).

Cost on this Mac: the 520-step episode with fake models and measurements took 17.7 s under `all` and
5.4 s under `used`. On the VM, EGL rendering shares the L4 with model inference, so the GPU time freed
matters more than the CPU saving; the comparison is on the VM re-run list (EGL may differ).

## Decider inputs and measurement recording (SPEC §6, §7; §18 item 3, first part)

`measure.py` holds the arithmetic (pure NumPy, shared by the engine, the dataset tooling and the tests); its
docstring defines every choice the spec leaves open: rows in executed form (what `pop` returns), rotations by
Rodrigues' formula, the angle between rotations as 2 asin(|Ra - Rb|_F / 2 sqrt 2), the mean rotation as the SVD
projection of the mean matrix, the position std as the RMS distance from the mean (ddof 0), gripper disagreement
as min(#close, #open) / K, a "close" command as g > 0, finger width as qpos[0] - qpos[1].

Engine (`episode.py`):
- `Episode(..., measure_client=XVLAClient)`, `start(..., record="off" | "full")`. `record="full"` (base,
  exploration and DAgger rollouts) measures every input group at every decision point where escalation is offered;
  with `record="off"` a live decider measures only its own profile's groups (`observe_inputs`). Points where the
  budget is spent are not measured (no decider is consulted there, amendment 15). Branches never measure.
- Features: at a chunk point from the `/act` response (same encoder pass); at a mid point from the probe's
  `/sample` response when the probe is needed, else from `/features`.
- Chunk-time samples: after each `/act`, K-1 extra `/sample` chunks of the exact request (`last_query_payload`),
  seeds `call_seed(..., "sample", 1..K-1)`; after an abandoned escalation's `act_fresh` (only when `chunk_spread`
  is needed), seeds `"sample_fresh"`. Mid-point probes: K chunks of the measured-pose request, seeds
  `"probe", 0..K-1`. The three names keep seeds distinct (an `act_fresh` and a probe at the same step can send
  the same request).
  The clock charges the fresh chunk's samples to `cascader_chunkspread` only (amendment 27, `profiles.<p>.act_fresh`).
- Per point record (`Episode.measurements`, and the current one in every snapshot): index, step, kind, chunk
  position, previous driver, robot state ([controller position, its 6D rotation, finger positions], 11 values),
  proposal (10 x 7 executed form, `XVLAAdapter.peek`), both feature vectors (float32), the previous segment's
  per-step records (action, controller pose after the step, finger positions), the raw chunk-time samples (K x 30
  x 20 float32, at the point after they were drawn) or the raw probe, and the chunk spread, fresh spread and
  drift summaries. Free signals are computed in `observe()` from the stored segment, because the gripper
  thresholds are calibrated in Phase 0 (until then `observe()` raises for cascader profiles).
- `observe(profile=None)` returns `api.DeciderInputs` with exactly the profile's groups; another profile's view
  needs a recording run. Snapshots carry the point's record, the previous segment and the chunk-time samples.

**Verified on the Mac** (`tests/test_measure.py`, 6 tests, hand-computed values; `tests/reference/
test_measurements.py`, 15 tests, fake models in the simulator): `peek` and `executed_rows` equal what `pop`
executes, bit for bit; recording and every live profile leave trajectories and event logs unchanged; `record=full`
measures every offered point and nothing else; every stored summary recomputes from the stored raw chunks; the
first chunk-time sample is the executed chunk; a locally continued proposal is exactly what ran next; the
previous segment equals the executed one; the fresh chunk after an abandoned escalation has its samples; all
seeds in an episode are distinct; each live profile calls only its own endpoints and sees exactly what the
recording run measured for it; snapshots restore every profile's inputs and the next point's measurements;
branches call nothing but `/act`. Server chunks are stored as float32 and checked to be exact; they are converted
in float64 as the client converts the executed chunk (a float32 conversion differed in the last bits).

## Dataset records, job runner and worker (SPEC §12, §13)

`dataset.py` defines dataset v1's layout and Parquet schemas (documented field by field in the datacard, below):
`records/<table>/<phase>/<job_id>.parquet` (tables `runs`, `steps`, `events`, `points`), `snapshots/<episode_id>/
dpNN.npz`, `images/<episode_id>.npz` (lossless PNGs, member `dpNN/<camera>`), and the queue's marker
`results/<phase>/<job_id>.json`, which lists every file with its SHA-256 and size. Arrays are flattened row-major
in their own precision (features, proposals and raw chunks float32; poses float64). Points store the previous
segment raw, so the gripper fractions can be computed after Phase 0 calibrates the thresholds (null until then;
zero at episode start, as `observe()` gives). pyarrow 25.0.0 (LeRobot v0.6.1's lock) is in `.venv-sim` and in
`pyproject.toml`'s `dataset` extra.

`collect.Runner` runs one job: episode jobs (`base` always local; `explore` escalates with `p_escalate` from the
rollout seed's `explore` stream, so a re-run decides the same way; with `record="full"` a snapshot, the images and a
`points` row at every measured decision point) and branch jobs (snapshot read from the bucket, `run_branch`).
DAgger and evaluation need a learned decider and are refused for now. `vlar_ops.worker.run_worker` claims a job,
runs it, uploads every file and reads it back to check its SHA-256, and only then calls `JobQueue.complete`, which
writes the marker: a marker in the bucket means the job's snapshots, images and records are there (snapshot sync).
A preempted job leaves no marker and is re-run after `resume()`; failures retry up to the queue's limit. The
read-back doubles the bucket traffic; Cloud Storage's own checksums could replace it once the GCS path is tested.

**Verified on the Mac** (`tests/test_worker.py`, 5 tests: order of uploads, retries, a corrupted upload never gets
a marker, re-run after preemption, the stop check; `tests/reference/test_collect.py`, 7 tests: a base and an
exploration episode and a branch job through the real queue into a local bucket). Every marker's files are in the
bucket intact; the bucket's snapshots equal a direct run's byte for byte; the runs, steps, events and every point
row equal the direct run's values; the PNGs equal the observation at each point; exploration escalates exactly
where its seed says; the branch job continues the bucket's snapshot exactly as a direct `run_branch` does.

## VM worker entry point, dataset manifest and datacard (SPEC §12, §13, §15)

`vlar-ops worker` (`vlar_ops/vm_worker.py`) is one simulator worker per process; `ops/vm/vlar-worker@.service`
starts N of them by hand after `queue resume` (not enabled at boot). At startup it loads the harness config with
the given named options (`--option name=choice`), refuses a frozen config whose code differs from its freeze record,
and refuses any frozen config until the golden-set self-check (§15) exists (it is written with the freeze, Phase 0
step 7). It verifies LeRobot's pinned files, and the X-VLA adapter verifies the authors' `libero_client.py`. It
waits up to `--server-wait-s` for both servers' `/info` (new `info()` on both clients) and stops unless each
reports the configured fixed geometry (`max_batch`; X-VLA's `sample_rows` and flow steps), deterministic algorithms
and cuDNN deterministic on, and cuDNN benchmark off. A server started at `--max-batch 1` for Phase 0 step 2's
latency runs is refused. `ProductionEpisodes` builds a fresh `Episode` per job on one cached `LiberoSim` per task:
`XVLAClient.act` is the X-VLA transport, `XVLAClient` is the measurement client, π0.5's call returns
`PI05Client.act(...)["actions"]`, the state mode is the config's `xvla_state_mode`, and the network is the
configured model.

`run_worker` gained four hooks. `phases` restricts claims (`JobQueue.claim(phases=...)`). `refuse(job)`
(`Runner.refuse`: a scene option other than the worker's, or a routed DAgger or evaluation job, which needs a learned
decider) gives the job back untouched, spends none of its attempts and stops the worker with exit status 3; before
this, such jobs failed three times and ended `failed`. `meta` goes into every marker: provenance, worker id,
`p_escalate` and the servers' settings. `cleanup` deletes the job's staged files. A SystemExit or KeyboardInterrupt
during a job releases its lease and leaves no marker; the CLI turns SIGTERM into SystemExit. The spend guard's STOP
flag is checked before each claim, and a worker does not start while it is up. The `Runner` takes the provenance
header (`provenance.record_header`), so snapshots carry it too. The parity run needs its own workers:
`--phase pi05_parity --option scene_seed=pi05_parity`.

`vlar-ops dataset manifest --root <local copy>` (`vlar_ops/manifest.py`) checks every marker's files (size and
SHA-256, or sizes only with `--quick`) and refuses:
- data files no marker lists (from interrupted jobs; `--move-unlisted DIR` moves them out, so `read_table` never
  globs them);
- any other file;
- Parquet parts whose schema is not v1's;
- run rows whose schema version, harness version, config hash, phase or job id differ from their marker;
- markers without provenance;
- mixed harness versions, source digests or image digests (§15).

Several config hashes (named options) and git commits are allowed and listed. A dirty harness is a warning. Only a
passing tree gets `DATACARD.md` (from `vlar_ops/datacard-v1.md`, outside the harness source digest, since it is
documentation) and `MANIFEST.json` (versions, configs with their options, per-phase counts, table rows, every file's
SHA-256 and size). `vlar-ops dataset verify --root <dir>` re-checks a downloaded tree: every file present and intact,
nothing extra. The datacard documents every field of the four tables, the files, seeds, provenance and exactly how
labels, the clock, time limits, exclusions and remaining time are computed; a test checks that its field tables
equal the Parquet schemas, name for name and in order.

**Verified on the Mac.** `tests/test_worker.py`, 5 new tests: phase filter; refused job back untouched with 0
attempts; KeyboardInterrupt or SystemExit releases the job with no marker; meta in markers; cleanup.
`tests/test_vm_worker.py`, 12 tests: the server checks, waiting and options parsing.
`tests/reference/test_vm_worker_sim.py`, 3 tests: an exploration job (p_escalate 0.5) through `serve` with fake model
clients and a real `LiberoSim`. All four tables equal the test factory's run and the PNGs are byte-identical.
π0.5 is called. The marker and snapshot headers carry the provenance. Staging is emptied and a DAgger job is
refused with 0 attempts. A mutation (state mode forced to `measured`) fails the comparison.
`tests/test_manifest.py`, 8 tests, simulator env: clean build and verify; a changed byte (caught by SHA-256, not
by `--quick`); unlisted and stray files; mixed source digest; differing options and commits allowed, dirty code
warned; a run row against its marker; no provenance; verify finds missing, changed and extra files; the datacard's
fields.

## Container images and lockfiles (SPEC §15, §18 item 5; amendments 28-29)

`containers/Dockerfile` builds three targets from one pinned base (`ubuntu:24.04` by index digest) with CPython
3.13.16, the reference notebook's Python (amendment 30), from a hash-verified python-build-standalone archive:
`sim` (the queue workers), `pi05` (the π0.5 server) and `xvla` (the X-VLA server). `sim` and `pi05` share one
Python environment, `containers/lerobot.lock` (LeRobot 0.6.1 with `libero` and `pi`), because π0.5's own tests need
LIBERO and because both sides of π0.5's preprocessing then run identical torch, LeRobot and transformers.
`containers/xvla.lock` holds the authors' server pins. Both locks come from `containers/lock.sh`:
`uv pip compile` for linux/x86_64 (manylinux_2_28), CPython 3.13, with hashes, from PyPI only, with
`--exclude-newer 2026-11-07` as for the local environments. The locks pin:
- torch 2.11.0 and torchvision 0.26.0 in both;
- transformers 5.5.4 (LeRobot environment) and 4.51.3 (X-VLA);
- the config's sim pins (hf-libero 0.1.4, robosuite 1.4.0, MuJoCo 3.8.1, NumPy 2.2.6);
- pyarrow 25.0.0 and every X-VLA server pin.

`download.pytorch.org` was unreachable from the Mac's network (TLS reset for curl, Python and uv alike), but PyPI's
torch 2.11.0 Linux wheel is PyTorch's CUDA 13.0 build: it depends on `nvidia-cuda-runtime` 13.0.96. That is the
notebook's 2.11.0+cu130 under another file name. `check_env.py` asserts `torch.version.cuda == "13.0"` in every
image.

Five LeRobot-environment packages ship only as sdists (`lerobot.sdists`: antlr4-python3-runtime, bddl, egl-probe,
hf-egl-probe, robomimic). Pip does not hash-check an sdist's build dependencies, so a builder stage
installs the hashed build tools of `build.lock` (setuptools, wheel, cmake) and builds wheels from the hash-checked
sdists without build isolation. `locktool.py split` separates the lock. Everything else installs with
`--require-hashes --no-deps --only-binary :all:`.

The images also hold:
- the X-VLA checkout at 6c4ecfb8 (shallow fetch by commit, checked);
- for sim and pi05, LIBERO's scene assets at 0b3ea86b in `/root/.cache/libero/assets`, where hf-libero 0.1.4's
  `get_assets_path()` finds them offline as in the reference run (amendment 32; config.yaml's `assets` key is never
  read), checked against the tree digest pinned in the config as `sim.assets_sha256`; `check_env.py` checks that
  LIBERO resolves that directory and builds all ten LIBERO-Long scenes; LIBERO's config written by `libero_config.py`;
- the repo's `src`, `tests`, `References` and `containers`;
- `VLAR_GIT_COMMIT`, which `provenance.git_commit` reads when there is no `.git`;
- the apt package list.

Weights are mounted read-only and Hugging Face runs offline. Every build ends with `containers/check_env.py`:
- CPython 3.13.16;
- installed distributions equal the lock;
- torch 2.11.0 built for CUDA 13.0;
- the reference-file pins;
- the assets digest.

With `--gpu` it also checks the driver (580.65.06 or later), that torch sees the GPU and, for sim, a MuJoCo EGL
context.

Build and run (amendment 28). `ops/gcp/35_build_images.sh` builds on an e2-standard-8 Spot VM (deleted after at
most `BUILD_MAX_HOURS`), runs the CPU in-container tests with OSMesa (`pytest tests` in sim; π0.5's and X-VLA's
CPU model tests with config-only Hugging Face snapshots at the pinned revisions), and exports
`vlar-{sim,pi05,xvla}.tar.zst`, `image-digests.json` and `SHA256SUMS` to `gs://$BUCKET/containers/<commit>/`. On
the L4, `ops/vm/load_images.sh` does the driver check, loads and verifies the images, writes
`/etc/vlar/{image-digests.json,images.env,batching.env}` and runs the GPU checks. The service units
(`vlar-xvla`, `vlar-pi05`, `vlar-worker@`) run the images with host networking, servers on 127.0.0.1.

**Checked on the Mac (no container runtime).**
- `tests/test_containers.py`, 19 tests: base pinned by digest; COPY sources exist and are in the context; every
  pip install hash-checked; every lock entry hashed; the locks have every pin of their inputs and match the config,
  pyproject and the notebook; Dockerfile sources match the config; the sdist split; wheel tags; `check_env`
  helpers; one driver threshold everywhere; build settings defined; shell scripts parse; the units' images, ports
  and batching variables.
- `uv pip install --dry-run` of every lock for linux/x86_64 with hashes required: 172 wheels + 5 sdists, 69, 4 (cp313).
- `libero_config.py` reproduces the Mac's LIBERO config.
- X-VLA's tests use starlette's test client, which needs httpx2 (starlette 1.10.0's `full` extra; httpx is a
  deprecated fallback, which `.venv-xvla` happens to have). `containers/xvla-test.lock` (from `xvla-test.in`, resolved
  with `xvla.lock` as constraints, `locktool.py minus` keeps only what `xvla.lock` lacks: httpx2 2.16.0, httpcore2
  2.16.0, truststore 0.10.5) is installed with hashes into the throwaway container of each X-VLA test run, never into
  the image. Checked on the Mac: the hashes verify and the test passes through httpx2 with the httpx fallback's warning
  as an error.
- Batch invariance on the build VM's CPU (amendment 33): `tests/reference/invariance.py` (`matches_solo`, `exact`,
  `all_match_solo`): bit for bit except in an image on the CPU, where a difference within torch's default tolerance
  and closest to the request's own solo result is a `NotBitIdentical` warning with the measured difference; every
  failure measures max absolute and relative difference. `tests/reference/test_invariance_checks.py` (9) tests the
  rules, including an output within tolerance but closer to another request's. On the Mac: π0.5 and X-VLA pass in
  both modes; outputs swapped in `_run_batch` or inside the batcher fail in the image-CPU mode (max |diff| about 6-8).
  `build_vm_startup.sh` logs `lscpu` first and `torch.backends.cpu.get_cpu_capability()` per image.
- `tests/reference/test_libero_assets.py`: LIBERO's resolved assets directory and digest, all ten scenes; missing
  assets, the same files elsewhere and a changed file are each refused. On the Mac 4 passed in 7 s (the ten scenes
  about 5 s).
- `check_env.py` against `.venv-sim` passes the reference-file and assets checks; the rest differs by platform,
  as expected.

**Not checked until Phase 0:**
- the image builds themselves (the Python download, apt packages, the sdist builds, the shallow X-VLA fetch, the assets download);
- the in-container tests;
- that Ubuntu's GCE image has `gcloud` (the startup script fails clearly if not);
- the GPU checks.

**Python (amendment 30):** CPython 3.13.16, as printed by the notebook's runtime-check cell; python-build-standalone
release 20261020, its SHA-256 matched against that release's `SHA256SUMS` and a local download. Every locked package
supports 3.13 at its pinned version, and the 3.13 resolution has the same pinned versions as the 3.12 one.

## Platform rule (SPEC §11 step 3 as amended)

The Mac runs (CPU MuJoCo, CGL rendering) show the code paths are right. Every equality and determinism
test is re-run on the L4 VM (Linux, EGL rendering, CUDA) before the freeze; only VM results count for
parity and determinism, because rendering and floating-point results can differ across platforms.

VM re-run list (each in its own image, `MUJOCO_GL=egl`; the first two rows come first in Phase 0, amendment 28):

| Environment | Command | Covers |
| --- | --- | --- |
| CPU build VM, before the L4 (needs billing only) | `ops/gcp/35_build_images.sh` from the Mac on the commit to run | the three images build from the hashed locks; their environment checks; the in-container tests that need no GPU (sim: `pytest tests` with OSMesa; π0.5 and X-VLA CPU model tests); images exported to the bucket with SHA-256 sums |
| L4 VM, host, before anything else | `ops/vm/load_images.sh gs://<bucket>/containers/<commit>` | **driver check**: NVIDIA driver 580.65.06 or later (CUDA 13.0, amendment 29); images match `SHA256SUMS` and their recorded IDs; `check_env.py --gpu` in each image (torch sees the L4, CUDA 13.0, MuJoCo EGL context in sim) |
| simulator | `pytest tests/reference/test_sim_reference.py tests/reference/test_snapshot_replay.py tests/reference/test_handoff.py tests/reference/test_clock_live.py tests/reference/test_render_used.py tests/reference/test_measurements.py tests/test_measure.py tests/reference/test_collect.py tests/reference/test_vm_worker_sim.py tests/test_manifest.py` | decider inputs and measurement recording (§6, §7); dataset records, runner and worker end to end (§12, §13); the VM worker entry point with fake clients; the dataset manifest and datacard; adapters vs both references (both render modes); rendering only what is used vs always-on rendering, byte for byte at decision points (§12; EGL may differ from CGL); snapshot restore and replay; determinism; handoffs, budget, run modes; live clocks equal offline replay for every profile; network retries and the `pi05_only` cap; live time cut vs offline crossing; branch determinism; labels from branch logs; the live-limit rules (amendments 19-23: refused combinations, the flag, cut runs scored only under their own profile, the DAgger-style parent re-scored for `cascader_full`); handoff diagnostics; sensitivity checks |
| simulator, both servers up, after Phase 0 step 5 | `VLAR_XVLA_SERVER=127.0.0.1:8777 VLAR_PI05_SERVER=127.0.0.1:8778 VLAR_BASE_FIRST_TARGETS=<json list> pytest tests/reference/test_handoff.py -k vm` | handoff continuity with the real models, **relative** (SPEC §11 step 3, amendment 10): X-VLA's first target after a handoff must be at or below the 99th percentile of its first-step target distances at ordinary chunk boundaries (no handoff) in the base rollouts; the JSON list comes from the base rollouts' event logs with `python -m vlar_harness.eventlog first-targets <base logs> --out <json list>`. The fixed 3 cm check is for the fake-model tests on the Mac only. |
| simulator + π0.5 server, Phase 0 step 2 (a measurement, not a re-run) | `vlar-ops step2 upload` (in `ops/vm/phase0/steps/200-step2-measure.sh`) | the upload size of decision 14: π0.5-alone decision points (tasks 0-9, 16 calls each, no network model) sent to `PI05Client.upload_bytes`; `network.median_upload_image_bytes`; written to the calibration file's write-once group (amendment 35) before the parity run |
| simulator, both servers up, before the base rollouts (a smoke run) | `vlar-ops worker --phase base --max-jobs 1`, then `vlar-ops dataset manifest --root <copy of the bucket>` | the production worker with the real clients and servers (geometry and deterministic checks pass, one job's files and marker in the Cloud Storage bucket), the Cloud Storage backend, and a manifest over real data; delete that job's files and marker afterwards, or keep them as the first base job |
| main | `vlar-ops reference check` | the reference files' pinned SHA-256 and totals, before any parity run |
| X-VLA (throwaway container: first `pip install --require-hashes --no-deps --only-binary :all: -r containers/xvla-test.lock`) | `VLAR_XVLA_MODEL_PATH=… VLAR_XVLA_LORA_PATH=… pytest tests/reference/test_model_servers.py -k xvla` | server vs the authors' handler with the real weights and LoRA |
| X-VLA (throwaway container: first `pip install --require-hashes --no-deps --only-binary :all: -r containers/xvla-test.lock`) | `VLAR_XVLA_MODEL_PATH=… VLAR_XVLA_LORA_PATH=… VLAR_TEST_DEVICE=cuda pytest tests/reference/test_batch_invariance.py -k xvla` | batch invariance on the L4 at the production geometry (8 requests, 40 sample rows), bit for bit: alone, with companions, at every position (six tasks' requests, each at every position of a full batch); the batches captured from the Mac's concurrency failure; simultaneous requests through the queue; rows post-processed alone; features equal across endpoints (amendment 24) |
| LeRobot (with the `pi` extra) | `VLAR_PI05_CHECKPOINT=… VLAR_TEST_DEVICE=cuda pytest tests/reference/test_batch_invariance.py -k pi05_real` | π0.5 at the production geometry with the real weights and tokenizer, bit for bit: alone, with companions, at every position and in a full batch, batch-mates from different tasks with different prompt and state token lengths; padding ids cannot change a result; every LIBERO-10 prompt fits the 200 tokens; max_batch 1 equals `act_unbatched`; simultaneous requests keep their prompts and outputs (amendments 24, 26). Deterministic mode must not raise on any CUDA kernel π0.5 uses. |
| each model's image, Phase 0 step 2 (a measurement) | `python -m vlar_harness.servers.bench xvla|pi05` on requests captured by `vlar-ops step2 capture`, then `vlar-ops step2 choose` (step `200`) | GPU time per full batch at sizes 1-16, deterministic on and off; the geometry rule (below) picks `max_batch`, `xvla_sample_rows` = K x `max_batch` and the worker count; deterministic mode costing more than 25% stops the step (exit 75) with the numbers (amendment 24) |
| both servers, Phase 0 step 2 (a measurement) | servers started at `--max-batch 1` (X-VLA also `--sample-rows 5`), then `vlar-ops step2 latency` (step `200`) | the clock charges, unpadded (amendment 25): median client wall time over localhost of 30 seeded calls after 5 warm-ups of `/act`, `/features`, `/sample` K = 5 and π0.5 `/act`; written to `latency` in the calibration file (π0.5's in the write-once group) |
| LeRobot | `VLAR_PI05_CHECKPOINT=… pytest tests/reference/test_model_servers.py -k pi05` | server vs lerobot-eval's per-step processing (needs the gated tokenizer) |
| main | `pytest tests` | config, queue, spend guard, packaging; network model, clock charges, labels and calibration on synthetic logs |
| simulator | `pytest tests/test_labels_calibration.py -k numpy` | the pure-Python percentile equals NumPy's bit for bit (the main environment has no NumPy) |

## Calibration file and stage B (amendments 34, 35)

- **File.** `VLAR_CALIBRATION` names it (VM: `/etc/vlar/calibration/calibration.yaml`): `{schema, config, frozen,
  values, history}`; `values` only under `config.CALIBRATION_PATHS`, deep-merged over the base config's placeholders.
  Every version written is kept as `versions/<sha256>.yaml`. A named file that does not exist is an error (never
  silently the placeholders). The base config no longer has `frozen`; `harness_version` is 1.0.0 in the code that
  runs parity and the base rollouts, because the freeze no longer touches the package.
- **Identity.** `config_hash` covers the base config (fixed settings) and the chosen options only;
  `calibration_hash` (SHA-256 of the file) goes into every record: `provenance.record_header` (markers, snapshots),
  runs rows, the runner's header check.
- **Write-once group** (`config.WRITE_ONCE_PATHS`): `latency.pi05_infer_s`, every network choice's
  `upload_image_bytes`, `servers.batching.max_batch` and `xvla_sample_rows`. The writer refuses any change once set
  (the same value again is a no-op); `vlar-ops worker` refuses to start until all are set.
- **Freeze** (`provenance.write_freeze`): marks the file frozen, writes `freeze.json` beside it with the final hash and
  its lineage (`config.calibration_lineage`: every earlier version with the complete, identical write-once group);
  `verify_freeze` re-checks code, config, calibration and lineage; `vlar-ops dataset manifest --freeze freeze.json`
  requires every record's calibration hash in it.
- **Refusals.** `Episode.start(time_limit=True)` and `Episode.observe()` raise `CalibrationNotFrozen` until the file
  is frozen. Collection stores no `sim_time_s` (runs) and no gripper fractions (points): clocks are re-scored from the
  event log; `measure.gripper_fractions_of_point` computes the fractions from `prev_actions` and `prev_gripper_qpos`
  (equal to the live ones: `tests/test_measure.py`, `tests/reference/test_collect.py`).
- **Geometry rule** (`vlar_ops.step2.choose_geometry`, proposed 2026-11-22, approved with the memory change
  2026-11-23): feasible sizes are those both benches ran and whose joint memory measurement left at least 2 GB
  (`HEADROOM_MIB` 2048) of the card free: in step `200`, at each size B both production servers at that geometry,
  the planned 2 x B workers (at most 12) rendering every camera every step (`vlar-ops step2 render`, one container
  each), full batches to both servers from B client threads each (`step2 load`), the card's peak from `nvidia-smi`
  sampled every 250 ms (`step2 memory` -> `memory-B.json`); a size whose servers or renderers did not all run through
  it is infeasible. Of those, the smallest whose deterministic GPU cost per request (X-VLA `/act` + `/sample` +
  `/features` + π0.5, full batches) is within 10% of the best; stop if deterministic mode costs more than 1.25x the
  non-deterministic run there. Workers = 2 x `max_batch`, at most 12. Padding cost by fill is reported, not gated.
  The benches' per-model peaks are reported only. Tests: `tests/test_step2.py`, `tests/reference/test_step2_sim.py`.
- **Gripper thresholds** (`vlar-ops step2 gripper`, approved 2026-11-23): π0.5's delta mode, no arm motion, gripper
  open then closed on nothing for 30 steps each, tasks 0, 3, 6 at initial state 0; shut = widest closed width + 10% of
  the open-shut range, open = narrowest open width - 10%. Not write-once: only offline values and live `observe()`
  use them. **Grasp check** after the base rollouts, before the freeze (`vlar-ops grasps --root <local copy>`,
  `vlar_ops/grasps.py`, sim image or `.venv-sim`): the base runs' commanded-closed steps (`action[6] > 0`) and their
  finger widths against the thresholds, all of them and the settled ones (closed for at least 10 consecutive steps,
  which leaves out the closing sweep); fractions shut / partly open / open, quantiles, a 1 mm histogram, per task;
  exit 75 with the reasons when more than 2% of the settled steps lie within 5% of the open-shut range of either
  threshold (they do not separate cleanly: the user looks before the freeze). Tests: `tests/test_grasps.py`.
- **Steps on the L4** (`ops/vm/phase0/steps/`): `200` step 2, `210` invariance rows at the chosen geometry, `300`
  parity + gate (`vlar-ops parity gate`, `vlar_ops/parity.py`) + manifest over its records, queues `400` (base).
  Tests: `tests/test_config.py`, `tests/test_step2.py`, `tests/test_parity_gate.py`, `tests/test_manifest.py`,
  `tests/reference/test_measurements.py` / `test_clock_live.py` (refusals), `test_vm_worker_sim.py`.

## Phase 0 steps 6-7: calibration, golden set, freeze (scripted 2026-11-23)

- **Ops code over the frozen images.** Steps 6-7 and the golden-set self-check were written after the one image
  rebuild (`b4c6beb`), so they live in `vlar_ops` and the sim containers mount the host source's `vlar_ops` over the
  image's copy (`lib.sh` `OPS_RUN`, `vlar-worker@.service`); `vlar_harness`, the code v1.0 freezes (its source digest
  covers only that package), stays the image's. Result markers record the host source as `ops_source`.
- **The ops code is pinned by its digest** (amendment 46, the user's rule 2026-11-23): `provenance.ops_digest()`, the
  source digest over the `vlar_ops` the process imports (the mounted host copy on the VM), is in every record header
  (result markers' `meta.provenance` and snapshot headers; `runs` rows through their job's marker) and in
  `freeze.json`; `verify_freeze` (and so the workers' startup and step 7's check) refuses other ops code or a record
  without it, `write_freeze` refuses without ops code, and the manifest refuses mixed or missing ops digests. Every
  container that runs or checks ops code mounts the host copy (`tests/test_phase0_steps.py` checks it; step 7's
  verification lacked the mount before).
- **Step 6** (`ops/vm/phase0/steps/500-calibrate.sh`, queued by hand after `400`): records of `base` and
  `pi05_parity` from the bucket; the event logs rebuilt from the `runs` and `events` tables (`vlar_ops.finish`,
  every run complete in `seq` order); the relative handoff check (first-target distances, then the VM row with both
  servers up); `calibration.calibrate` + `write` (time limits and margin into the calibration file); the grasp check
  (its flag goes into the summary); `vlar-ops finish summary` (per-task limits and m read back from the calibration
  file and checked against this run's, π0.5's share within the limits, the grasp check), printed and uploaded to
  `phase0/step6/calibration-summary.{txt,json}`; then it **stops for the user** (exit 75; the user's rule,
  2026-11-23): `600` is queued only with the user's explicit approval after reading the summary. Exit 75 keeps the
  step queued; on a later boot `finish unchanged` (the summary's records digest and calibration hash against the
  current ones) ends it done (exit 0) with nothing recomputed, so an approved `600` queued behind it runs; with new
  records or another calibration file it recomputes and stops again (delete the summary to force a recompute).
- **Step 7** (`600-freeze.sh`): `vlar-ops golden select`: five base snapshots, tasks drawn with seed 20261123, one
  decision point (index >= 1) each, with the base run's recorded steps after it (`golden/golden.json` + `.npz`
  beside the calibration file); `golden check --mark-checked` replays each (restore, X-VLA local to the end,
  measurements off) through the production servers and compares every step's action, end-effector pose and finger
  positions and the outcome bit for bit; `finish freeze` (`provenance.write_freeze` with the golden ids), verified;
  the check again on the frozen file.
- **Every start of a frozen harness** (`lib.sh` `run_phase`): the golden check runs after the servers answer and
  before any worker, writing `/var/lib/vlar/golden-pass.json` (boot id, calibration hash, pass); `vlar-ops worker`
  refuses a frozen harness without this boot's pass for this calibration file.
- Tests: `tests/test_finish.py` (also the summary and the re-run check), `tests/reference/test_golden.py` (fake models:
  replay bit for bit; a changed action, pose or outcome caught at its step; the worker gate),
  `tests/test_phase0_steps.py` (the step chain, the ops mounts), `tests/test_config.py` and `tests/test_manifest.py`
  (the ops digest in the freeze record and the manifest).

## Phase 2 training (`vlar_train`, started 2026-11-23)

Outside the frozen harness (§15), for the Mac or the VM: `.venv-sim/bin/python -m vlar_train --records <local copy>
--label-map <branch planner's label map> --out <dir>` (`--synthetic` for a dry run). `data`: points of the label
map with their labels, exclusions and remaining time per profile from `vlar_harness.labels.point_labels` (prefix = the
parent's log up to the point's `decision_point` event), inputs from the points table, gripper fractions offline;
`deciders`: each decider's columns (§6; no task input, the router nothing derived from the proposal), one MLP family
(512, 256, GELU, dropout 0.1; under 5M parameters) with four sigmoid outputs; `folds`: test i, validation i + 1;
`train`: from scratch with a seed per fold and decider, AdamW, BCE against the label fractions over the branches a
point has, inputs standardised on the training points, a checkpoint every epoch, the stopping epoch at the lowest
validation loss (patience 8); `lam`: λ with the lowest escalation rate reaching 95% of the best estimated gain
(branch 2 - branch 1 labels where the rule escalates), optionally capped by `max_rate`. Results per fold and decider
in `<out>/fold<i>/<decider>/`, medians of the stopping epoch and λ in `summary.json` (for Phase 3). Not yet: the
evaluation on test tasks (§10) and the 8 GB deployment-memory measurement (§8). Tests: `tests/test_train.py`
(synthetic data). The label map's file is written by the branch planner when Phase 1 is planned.

## Running the reference tests

```bash
.venv-sim/bin/python -m pytest tests/reference                    # simulator + adapters, exact
VLAR_XVLA_CONFIG_DIR=third_party/hf/X-VLA-Pt VLAR_TEST_DEVICE=cpu \
  .venv-xvla/bin/python -m pytest tests/reference/test_model_servers.py -k xvla
```

The local environments (`.venv-sim`, `.venv-xvla`, `third_party/`) are git-ignored; the VM uses the
container images built from the same pins.
