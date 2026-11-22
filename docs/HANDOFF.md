# Handoff: state of the harness build (2026-11-21)

For a session that starts without the previous conversation. Read this file at startup; read `docs/SPEC.md` (the
design; approved amendments are logged in its §17) and `docs/HARNESS.md` (implementation notes, findings, and the VM
re-run list) only in the sections the task needs (`CLAUDE.md`, "Usage discipline").

## Working rules the user set

- **Effort checkpoints (SPEC §16).** Before starting any step on the §16 max list, stop and say exactly:
  "Effort checkpoint: please switch to max effort before I start [step]." and wait for confirmation. When the
  step is finished, say: "Effort checkpoint complete: you can switch back to xhigh." Also raise a checkpoint for
  any unexplained discrepancy that could silently corrupt labels or results.
- **Effort schedule** (set by the user 2026-11-21; tell the user the level the next task needs, not only when it is
  max): finishing step 5: max. Rendering only the observations that are used (§12): xhigh (max only if an
  exact-equality test fails). Measurement recording, decider inputs, dataset format and queue workers: xhigh (max
  only for unexplained discrepancies). Container images and lockfiles: high. Google Cloud settings, the VM re-run
  list and housekeeping: medium. Phase 0 on the VM: xhigh (max only for parity debugging).
- **Commits and pushes.** Only when the user asks. So far the user has asked, before each step, for the previous
  step to be committed on branch `harness-v1` with the test results in the commit message. End commit messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The user had `harness-v1` pushed to `origin` on
  2026-11-21 so the project partner can see it; push again only when asked or at the stopping rule's stop
  (`CLAUDE.md`, "Usage discipline"). Never commit to or push `main`.
- **docs/STATUS.md** is the overview for readers with no context (the project partner and their AI model); update it
  only when a §18 item or phase changes status. Update this file every session.
- **Spec changes** need the user's approval; log each approved one in SPEC §17 with its date.
- **Platform rule (SPEC §11 step 3).** Everything so far was verified on a Mac (CPU MuJoCo, CGL rendering). Mac
  results only show the code paths are right; every equality and determinism test must be re-run on the L4 VM
  (Linux, EGL, CUDA) before the freeze, and only VM results count.
- Repo convention: no hostnames, usernames or network addresses of the team's machines in the repo.

## Milestone checkpoint (SPEC §18, approved 2026-11-21)

The harness build is complete only when all of the following are true:
1. Effort steps 4 (event log, overhead profiles, offline clock re-scoring, labels, time-limit calibration code) and 5 (server batching, per-call seeding, batched-versus-unbatched equivalence) are done, with their tests passing on the Mac.
2. Both model servers implement every endpoint in §3 (/act, /sample, /features for X-VLA; π0.5 inference), with batching and per-call seeds.
3. Measurement recording (proposal, free signals, spread and drift, probes) and the snapshot store are implemented and tested.
4. The job queue (idempotent jobs, ordering tiers, bucket sync, resume after preemption), the spend guard with its ledger, and the packaging and verification scripts for §13 are implemented and tested.
5. The three container images and lockfiles build, and everything that can be tested locally in them passes.
6. Ready-to-run setup scripts exist for the Google Cloud side: the g2-standard-16 Spot VM, the bucket, and the budget alert at 50%, 80% and 100% of $50.
7. The VM re-run list in docs/HARNESS.md is complete, the parity tooling and reference JSON files are committed, everything is committed, git status is clean, and HANDOFF.md is up to date.

When all seven are true, stop and tell the user exactly: "Milestone checkpoint: the harness build is complete and Phase 0 on the L4 VM is next." With it, give a short checklist confirming each of the seven items with its commit, the estimated Phase 0 cost and VM hours, and the first commands Phase 0 will run. Then wait for the user's explicit go-ahead before creating any cloud resources or spending any money. Do not announce this milestone early or partially; if any item is incomplete, say which ones remain instead.

Where each item stands now (not complete; do not announce):

| Item | State |
| --- | --- |
| 1. Steps 4 and 5 | done on the Mac: step 4 (`272f64b`, follow-ups `a25053f`); step 5 (part 1 `f8d06d5`, part 2 `94b9297`, the X-VLA concurrency fix in the commit after `c1fcd78`); their equality tests are on the VM re-run list |
| 2. Server endpoints | X-VLA `/act`, `/sample`, `/features` and π0.5 `/act` with per-call seeds, all batched at a fixed geometry of rows and tokens (amendments 24, 26); unseeded requests and π0.5 `/upload_bytes` stay on the reference path |
| 3. Measurement recording, snapshot store | snapshot capture/restore and a local `SnapshotStore` done and tested; decider inputs and measurement recording (`observe()`, proposal, free signals, spread/drift, chunk-time samples, probes, `record="full"`) done and tested on the Mac with fake models (`6c08bf5`); Parquet dataset records and snapshot sync to the bucket (through the worker) done and tested against a local bucket (the commit after `6c08bf5`); the Cloud Storage path is untested |
| 4. Queue, spend guard, packaging | implemented and tested against a local stand-in for the bucket (`src/vlar_ops`); the engine is wired to the queue (`collect.Runner`, `vlar_ops.worker.run_worker`, the commit after `6c08bf5`); the Cloud Storage backend is untested without credentials; the VM worker entry point (real clients, spend-guard stop flag) is not written |
| 5. Container images, lockfiles | not started |
| 6. GCP scripts | written (`ops/gcp/`), never run; `IMAGE_FAMILY` and account values are placeholders in `env.example.sh`; the budget alert script covers 50/80/100% of $50 |
| 7. VM re-run list, commits, HANDOFF | the list covers steps 1-5 (both servers' batch invariance on the L4, the padding-cost and unpadded-latency measurements) and §12 rendering (`test_render_used.py`); the containers still to add; parity tooling and reference files are committed (`25794f7`) |

## Status

| Work | State | Commit |
| --- | --- | --- |
| Groundwork: versioned config with named options and calibration file, job planner and resumable queue, spend guard, archive packaging, GCP scripts and runbook (`ops/`) | done | `4960782` |
| Effort step 1: simulator layer, X-VLA and π0.5 adapters, both model servers, exact-equality tests against the reference code | done on Mac; VM re-run pending | `4960782` |
| Effort step 2: exact snapshot capture/restore, determinism tests | done on Mac; VM re-run pending | `5d92d97` |
| Effort step 3: episode engine (`src/vlar_harness/episode.py`): handoff rules, controller mode switching, budget, run modes; scene/rollout seed split; reference-run files and checks | done on Mac; VM re-run pending | `25794f7` |
| Decisions 9-12 recorded, relative VM handoff test, this file, `CLAUDE.md` | done | `b3e93da` |
| Decisions 13-16 recorded (network failures, upload size, no overhead without budget, `reset(time_limit)`), Google Cloud readiness noted | done | `272f64b` |
| Effort step 4: network model, event log, live clocks for every profile and offline replay, live time cut, π0.5-alone retries with the 20-attempt cap, `run_branch`, labels, time-limit calibration, handoff diagnostics, upload-size tooling (`network.py`, `clock.py`, `episode.py`, `labels.py`, `calibration.py`, `eventlog.py`) | done on Mac; VM re-run pending | `272f64b` |
| Decisions 17-18 recorded; STATUS.md added | done | `96ef752`, `11e999a` |
| Step-4 follow-ups, amendments 19-23: live limit only in routed runs with a decider profile; the explicit `time_limit` flag in start events, results, snapshots and branch events; cut runs scored only under their own profile (scoped so DAgger points get labels for every profile); per-profile exclusion of points already past the limit; the remaining-time input recomputed per profile (`api.py`, `clock.py`, `eventlog.py`, `episode.py`, `labels.py`, `calibration.py`); each rule shown to be caught by its tests when broken | done on Mac; VM re-run pending | `a25053f` |
| Effort step 5, part 1: the request batcher (`servers/batching.py`), deterministic settings (`common.deterministic_torch`), X-VLA's three endpoints batched at a fixed padded geometry, alone / companions / position checks on the CPU, the batching config placeholders | done on Mac; VM re-run pending | `f8d06d5` |
| Decisions 25-26 (unpadded clock charges; fixed token length); effort step 5, part 2: π0.5 batching at a fixed geometry of rows and tokens (`PI05Service.load`, `PI05Batching`, `act_unbatched`), the shared-state audit of both servers with `PROCESSING_LOCK`, X-VLA's token-length check, tests on LeRobot's real π0.5 code (tiny, random weights) and a stand-in policy, X-VLA's tests with mixed-length instructions, π0.5 mutation check 6 of 6 | done on Mac; VM re-run pending | `94b9297` |
| Usage discipline (`CLAUDE.md`) and the effort schedule (above) | done | `c1fcd78` |
| Effort step 5 completed: the X-VLA simultaneous-requests failure reproduced (batch compositions logged, the failing batches captured and replayed deterministically), traced to the authors' post-processing over the whole batch (a CPU elementwise kernel's scalar tail) and fixed by post-processing each row alone; new position, replay and post-processing tests; X-VLA mutation check (HARNESS.md "Batching") | done on Mac; VM re-run pending | the commit after `c1fcd78` |
| Render only the observations that are used (SPEC §12, amendment 7): config option `render` (`used` default, `all`); camera sensors gated, timers untouched; unrendered steps expose no images; snapshots record `sim.images`; `RenderPhaseError` if a rendering step misses its sample; full-episode comparison with always-on rendering byte-identical at decision points; mutation check 3 of 3 (HARNESS.md "Rendering only what is used") | done on Mac; VM re-run pending | the commit after `f283023` |
| Decider inputs and measurement recording (§6, §7): `measure.py` (definitions in its docstring), `Episode(measure_client=...)`, `start(record="off"/"full")`, `observe(profile)`, per-point records in `Episode.measurements` and in snapshots, seed names `sample_fresh` and `probe` (HARNESS.md "Decider inputs and measurement recording") | done on Mac; VM re-run pending | `6c08bf5` |
| Dataset records (`dataset.py`: Parquet `runs`/`steps`/`events`/`points`, snapshots, PNG images, markers listing every file), the job runner (`collect.py`: base, explore, branch jobs) and the worker loop (`vlar_ops/worker.py`: uploads and checks every file before the marker); pyarrow 25.0.0 in `.venv-sim` (HARNESS.md "Dataset records, job runner and worker") | done on Mac against a local bucket; VM re-run pending | the commit after `6c08bf5` |
| VM worker entry point (production factory with `XVLAClient`/`PI05Client`, one `LiberoSim` per task, the spend guard's `StopFlag`), dataset manifest and datacard | not started | |
| Container images and lockfiles; VM bring-up; Phase 0 runs (latency and upload measurement, parity, base rollouts, calibration, freeze) | not started | |
| Effort steps 6-7 (fold/leakage review; post-pilot re-budgeting and results) | later phases | |

**Test suites after dataset records, runner and worker (Mac, 2026-11-21).** Main venv 122 passed, 11 skipped; simulator env (`tests/reference tests/test_measure.py tests/test_labels_calibration.py`) 114 passed, 17 skipped.

**Test suites after decider inputs, measurement recording and amendment 27 (Mac, 2026-11-21).** Main venv 117 passed, 10 skipped; simulator env (`tests/reference tests/test_measure.py tests/test_labels_calibration.py`) 107 passed, 17 skipped.

**Test suites after §12 (Mac, 2026-11-21).** Main venv 112 passed, 8 skipped; simulator env 69 passed, 17 skipped (about 3 minutes; the 17 skips as before); X-VLA env not re-run (no code it runs changed).

**Test suites after step 5 (Mac).** Main venv 112 passed, 7 skipped; simulator env 57 passed, 17 skipped (the skips
need weights, both servers, the VM or the X-VLA environment); X-VLA env (random weights, CPU) 12 passed. Part 2's
intermittent X-VLA failure (one of six simultaneous requests not bit-identical to its alone result) is explained and
fixed (HARNESS.md "Batching"): logging every batch's composition showed request 5 failing whenever it was last in its
batch, and the captured batches (`[0, 1, 5]`, `[1, 2, 5]`) replay deterministically, so timing and threads were not
involved. Padding rows were exact copies of the first request and no module's output differed; the one differing value
(a gripper, one ulp) came from the authors' post-processing, a sigmoid over the whole batch, whose CPU kernel computes
the last (count mod 8) elements with scalar code. The server now post-processes each row alone, as the authors' batch
of one does. After the fix: 0 failures in 10 concurrent trials (2 in 4 before). X-VLA mutation check 7 of 8 caught (no
`PROCESSING_LOCK` is not observable: the tokenizer only reads its settings after the first call). Part 1's "CPU denoiser
depends on the batch size" was the same sigmoid, so a stand-in model test now checks the padding mechanics on the Mac.

Test suites after step 5 part 1 (Mac): main venv 112 passed, 7 skipped; simulator env 48 passed, 11 skipped (the
5 X-VLA batch-invariance tests skip there); X-VLA env (random weights, CPU) 7 passed in about 4 minutes. Mutation
check of step 5: 2 of 3 caught (dropping the encoder's padding is not observable on the CPU; HARNESS.md,
"Batching"). After the step-4 follow-ups (`a25053f`): main venv 106 passed, 6 skipped (the simulator modules and the
NumPy cross-check need NumPy); simulator env 48 passed, 6 skipped (the same 6 as before: 3 need model weights,
3 need both servers on the VM); X-VLA env (random weights, CPU) 2 passed. Mutation check of rules 19-23: 15 of 15
caught (HARNESS.md, "Live-limit rules"). At `272f64b`: main 84 passed and 6 skipped, simulator 46 passed and 6
skipped, X-VLA 2 passed.

**VM re-run list:** `docs/HARNESS.md`, section "Platform rule (SPEC §11 step 3 as amended)".

## Running things locally

The local environments are git-ignored and live in the repo root:
- `.venv`: Python 3.13, the unit tests: `.venv/bin/python -m pytest` (no NumPy: the clock, network, labels and
  calibration modules are pure Python on purpose).
- `.venv-sim`: LIBERO (hf-libero 0.1.4), robosuite 1.4.0, MuJoCo 3.8.1, LeRobot 0.6.1, CPU torch. Reference tests:
  `.venv-sim/bin/python -m pytest tests/reference` (about 6 minutes; `tests/reference/conftest.py` sets
  `LIBERO_CONFIG_PATH=third_party/libero-home` and `MUJOCO_GL`).
- `.venv-xvla`: the X-VLA authors' pins, CPU. `VLAR_XVLA_CONFIG_DIR=third_party/hf/X-VLA-Pt VLAR_TEST_DEVICE=cpu
  .venv-xvla/bin/python -m pytest tests/reference/test_model_servers.py -k xvla` (random weights).
- `third_party/` (git-ignored): the X-VLA checkout at 6c4ecfb8, a LeRobot v0.6.1 clone, unpacked hf-libero 0.1.4
  and robosuite 1.4.0 wheels, the LIBERO scene assets at 0b3ea86b, LIBERO's config, small Hugging Face config files.
- `vlar-ops reference check --dir References` verifies the reference files' pinned SHA-256 and totals.
- `python -m vlar_harness.calibration --base ... --parity ... [--write]` (Phase 0 step 6) and
  `python -m vlar_harness.eventlog first-targets <base logs> --out <json>` (the VM handoff check's input).
- Do not wrap commands in `timeout`: the one on this Mac is an x86_64 binary and starts the universal Python as
  x86_64, which breaks the arm64 wheels.

## Decisions so far, and where they are recorded

Spec amendments, all approved by the user and logged in SPEC §17:
1. Pins add hf-libero 0.1.4 and the LIBERO scene assets at 0b3ea86b (§2).
2. Pooled features are two 1,024-value vectors, the main encoder output and the wrist camera's tokens; both stored,
   every decider takes both (§3, §6, §13).
3. Mid-chunk `/sample` probes send the measured pose; the extra samples at a chunk point reuse the exact `/act`
   request, quirk pose included (§3, §6).
4. π0.5's 360 px images come from extra camera observations kept in step with the native ones (§3).
5. Restore puts back the full state: MuJoCo state with the solver warm start, observation timers and buffered
   values, controller and gripper state, LIBERO object states; the replay test compares observations exactly (§7).
6. Every Mac equality/determinism test is re-run on the L4 VM before the freeze; only VM results count (§11 step 3).
7. Rendering only the observations that are used is approved after restore, if it passes the same exact tests (§12).
8. Two seeds per episode: scene seed (the env seed) 142 + k in every project run, 1000 + k only for the π0.5
   parity run; a separate rollout seed for model noise, the network model and exploration; different rollout
   seeds per run type, evaluation's never used in data collection; paired per-episode comparisons in §10
   (§3, §7, §10, §11, §15).
9. Time-limit calibration rule approved: m is the smallest of {1.25, 1.5, 1.75, 2.0} such that at least 80% of
   π0.5's successful parity episodes, timed under profile `none` with the network model, finish within their
   task's limit (§5, §11 step 6). Implemented in `calibration.py`.
10. The real-model handoff check on the VM is relative: X-VLA's first target after a handoff must be at or below
    the 99th percentile of its first-step target distances at ordinary chunk boundaries (no handoff) in the base
    rollouts; the fixed 3 cm stays for the fake-model tests on the Mac (§11 step 3). Implemented in
    `tests/reference/test_handoff.py::test_model_handoff_continuity_on_the_vm`, which reads the distances from
    the JSON file named by `VLAR_BASE_FIRST_TARGETS`; every `xvla_act` event now records the distance and its
    context, and `python -m vlar_harness.eventlog first-targets` writes the file from the base rollouts' logs.
11. The reference run's 11 per-episode files are committed byte for byte in `References/` with their SHA-256
    pinned in `src/vlar_ops/configs/reference-run.sha256`; the parity tooling verifies them (§2, §11).
12. The milestone checkpoint for the end of the harness build (§18, quoted in full above).
13. Network failures (2026-11-21, before step 4; §5, §10): a retry is the same request and never re-draws model
    noise (in every mode, including retries inside a routed escalation), so executed actions equal a run without
    failures and only the clock is charged. Each attempt's network draws are keyed by rollout seed, step, request
    identity and attempt number, never by the clock or the profile. In `pi05_only` a failed request is retried
    until it succeeds, with a safety cap of 20 attempts, after which the episode ends with end reason `network`.
14. Upload size (2026-11-21, before step 4; §5, §11 step 2): measured once in Phase 0 step 2 as the median JPEG
    size of the two 224 px images π0.5 actually consumes, quality 90, stored as a byte count in the `network`
    option. Clock only; the model's input is unchanged.
15. No decision overhead where escalation is not offered (2026-11-21, during step 4; §5): once the π0.5 budget is
    spent the decider is not consulted, so those decision points cost nothing under any profile (X-VLA's `/act`
    is still charged). Implemented in `clock.event_s` from the `can_escalate` flag every `decision_point` logs.
    Without it, b4_max from point 5 would have paid 41.2 s of cascader_full overhead after the budget ran out
    (placeholder latencies; time limit placeholder 37.2 s).
16. `reset()` gains `time_limit` (2026-11-21, during step 4; §15): off by default; applies the task's limit live
    under the run's profile; set only for runs with a live learned decider (DAgger, evaluation). In the engine it
    is `Episode.start(..., time_limit=True)`.
17. Branch call counting (2026-11-21, after step 4; §7): an abandoned escalation counts as one of a branch's calls,
    so planner-shared branches stay identical (already implemented in `Episode.run_branch`).
18. Upload size definition (2026-11-21, after step 4; §5): median over requests of the combined JPEG size of the
    request's two images (already implemented in `network.median_upload_image_bytes`).
19. Live limit only where a live decider runs (2026-11-21, step-4 follow-up; §15): `time_limit=True` only in routed
    mode with a profile other than `none`; `api.check_reset`, called by `Episode.start` before anything else.
20. The flag is explicit (§5, §15): `time_limit` true/false in the start event, `EpisodeState` (snapshots) and
    `EpisodeResult`; `branch` events record false.
21. Cut runs (§5, §7), scoped by the user: a cut run's outcome is scored only under its own profile (`clock.score`,
    `first_crossing`, calibration, a cut parent's full log as a label seed); a cut parent's log up to a decision
    point plus an uncut branch is re-scored under every profile, so DAgger points get labels for every decider type
    (§9). Implemented with `clock.run_header`: the governing header is the start event or the last `branch` event.
22. Exclusion (§7): a point whose clock under P from the parent's log already exceeds the limit (strictly) is left
    out of P's training and validation data; `PointLabels.excluded`, `labels.exclusion_report`.
23. Remaining time (§6, §7), added by the user: at every training point, recomputed from the event log under the
    decider's own profile (`labels.remaining_time_s`, `PointLabels.remaining_s`), never copied from the collector.
24. Batch invariance (2026-11-21, before step 5; §3, §11, §12): a request's output is bit-identical whatever its
    batch-mates, their number or its position; one fixed padded batch size per endpoint, deterministic GPU settings,
    each request's noise from its own generator seeded by its per-call seed. CPU tests now; on the L4 the same three
    checks bit for bit for both servers plus the throughput cost of fixed padding; if padding is too slow, stop and
    bring the numbers to the user before relaxing anything.
25. Unpadded clock charges (2026-11-21, during step 5; §5, §11 step 2): the clock's per-call charges are measured as
    a deployed robot runs them, unpadded: X-VLA `/act`, `/features` and π0.5 inference as a batch of one, `/sample` as
    a batch of exactly K. Padding exists only to make data collection reproducible and is never charged. Phase 0 step
    2 measures these separately from the padded throughput figures that size the worker count (practical route: start
    the servers at `max_batch` 1 and `xvla_sample_rows` K, which is exactly an unpadded batch of one / of K).
26. Fixed token length (2026-11-21, during step 5; §3, §11 step 3): the fixed geometry includes the prompt's token
    length: π0.5 right-padded to the checkpoint's 200 tokens with attention masks (padding cannot affect results;
    tested by changing the padding's ids), X-VLA padded to the authors' 50 tokens (its encoder attends to the padding
    as in the authors' model, so the padding is the request's own). Requests of another length, or filling the whole
    length (possibly truncated), are refused; the invariance tests use batch-mates from different tasks with different
    instruction and state token lengths.
27. Fresh-chunk samples (2026-11-21, during §18 item 3; §5): after an abandoned escalation, X-VLA's fresh chunk
    (`act_fresh`) is charged t(`/sample` K) − t(`/act`) on top of `/act` under `cascader_chunkspread` only (it reuses
    that chunk's chunk-time spread at the next points); `cascader_full` probes fresh there and is not charged.
    Config `profiles.<p>.act_fresh`; `clock.event_s`; tests in `test_network_clock.py` and `test_measurements.py`
    (each profile is charged exactly when its live decider draws those samples).

Implementation decisions and findings, recorded in `docs/HARNESS.md` (with the tests that show them):
- The reference stack: LeRobot 0.6.1, hf-libero 0.1.4, robosuite 1.4.0, MuJoCo 3.8.1, file hashes pinned in
  `src/vlar_harness/configs/harness-v1.yaml` (`reference`).
- robosuite samples observations at physics substep 24 of 25, so a render after a step is not what the models saw.
- In tasks 2, 3, 5, 8 and 9 a fixture moves by up to ~19 mm with the env seed (hence decision 8).
- The X-VLA adapter is the authors' `ClientModel` with `step()` split into query and pop; the π0.5 adapter uses
  LeRobot's own `_format_raw_obs` and gymnasium batching; both servers reuse the reference code, and model noise
  from per-call seeds is drawn on the CPU from one generator per seed.
- Snapshot contents and restore rules (in-place, only differing values, cached static model digest).
- The engine counts the π0.5 budget in executed steps and checks each segment's controller mode.
- `xvla_state_mode` (quirk/measured), `scene_seed` (project/pi05_parity), network and probe choices are named
  options in the harness config; rollout-seed namespaces are in `src/vlar_ops/configs/collection-v1.yaml`.
- §12 (HARNESS.md "Rendering only what is used"): the camera sensors are gated, never the observables, so the
  sampling timers run exactly as with always-on rendering; option `render` (`used`/`all`) is recorded with the
  other config options.
- Step 4 (HARNESS.md "Event log, clock, network, labels and time limits"): the engine keeps one live clock per
  profile and checks each against the offline replay at the end of every episode (`ClockMismatch`); a segment's
  steps are one log event, charged one step at a time both live and offline; network draws via
  `seeding.network_attempt_seed` + `seeding.uniform` (no generator state); the default link is the simulated
  network, `DirectLink` is for tests only; branches escalate at their first `pi05_calls` decision points, an
  abandoned escalation counting as one (keeps planner-shared branches identical); the clock charges the logged
  network outcome and never re-decides it; calibration's p90 is NumPy's linear percentile re-implemented in pure
  Python and checked equal bit for bit; the upload-size placeholder is a Mac estimate of 20,290 bytes.

## Open questions for the user

None open. (The fresh-chunk charge question raised during §18 item 3 was decided as amendment 27.)

**Google Cloud readiness (user's action, reported 2026-11-21).** The user is upgrading from the free trial if
needed and requesting GPU quota: global GPU quota (`GPUS_ALL_REGIONS`) ≥ 1, preemptible L4
(`PREEMPTIBLE_NVIDIA_L4_GPUS`) ≥ 1 and on-demand L4 (`NVIDIA_L4_GPUS`) ≥ 1 in us-central1 (the region in
`ops/gcp/env.example.sh`; the on-demand quota covers the §12 fallback when Spot capacity is short). The user will
confirm when it is approved. Nothing in effort steps 4-5 depends on it; it gates only the VM bring-up, which in
any case waits for the §18 milestone and the user's go-ahead.

## Next step

Done this session (2026-11-21): §12 rendering (`76a0392`), decider inputs and measurement recording with amendment
27 (`6c08bf5`), dataset records, the job runner and the worker loop (the commit after `6c08bf5`). Next, at
**xhigh**: the VM worker entry point (a `vlar-ops` command that builds the production `Episode` factory with
`XVLAClient`, `PI05Client`, one `LiberoSim` per task, and runs `run_worker` with the spend guard's `StopFlag`), then
the dataset manifest and datacard (§13). After that, the container images and lockfiles (§18 item 5) at **high**.

Worth knowing:
- `.venv-sim` now also has LeRobot's `pi` extra (transformers 5.5.4, tokenizers 0.22.2; installed 2026-11-21 with
  pip's `--uploaded-prior-to 2026-11-07`, additive only), so it runs LeRobot's real π0.5 code at a tiny size:
  `.venv-sim/bin/python -m pytest tests/reference/test_batch_invariance.py -k pi05` (about 30 s). Without the extra
  those tests skip.
- The reference run's exact transformers version for the LeRobot environment is not recorded (LeRobot 0.6.1 allows
  5.4.0 to 5.5.x); the lockfiles (§18 item 5) must pin it, and the VM's LeRobot environment needs the `pi` extra.
- Phase 0 step 2 now measures two separate things (amendment 25): the clock charges, unpadded (servers started at
  `max_batch` 1 and `xvla_sample_rows` K), and the throughput at the production geometry, padded and unpadded, with
  deterministic mode on and off (sizes the worker count; never charged).
- Engine side: no change. Each simulator worker sends single requests; the servers batch across workers.
- Dataset format: pyarrow is in neither local environment. LeRobot v0.6.1's uv.lock pins pyarrow 25.0.0; install
  it additively in `.venv-sim` (as the `pi` extra was) and add it to the simulator lockfile. Snapshot sync to the
  bucket belongs with the queue workers: an episode job is `done` only once its snapshots are uploaded.
- §12: the sim renders only the step before each decision point by default. Anything that reads images
  elsewhere gets `ImagesNotRendered`; a snapshot taken between decision points holds placeholders
  (`sim.images` false). Use `render="all"` (`load_config(render="all")`) for per-step image work.

Toward the §18 milestone, in order:
1. Done: render only the observations that are used (§12); decider inputs, measurement recording, dataset records,
   runner and worker loop (§18 item 3; item 4 wired). Left from item 4: the VM worker entry point.
2. (done) Decider inputs and measurement recording (§18 item 3): `observe()` (features, proposal, free signals, spread and
   drift, the K-sample probes), the dataset format (Parquet), and the queue workers that run episodes and branches
   through `run_branch` (item 4: the engine is not wired to the queue yet; the Cloud Storage backend is untested).
3. The three container images and lockfiles (item 5).
4. The GCP values in `ops/gcp/env.sh` once the account and GPU quota are ready (item 6; scripts written, never run).
5. The VM re-run list completed with the containers, everything committed, a clean tree (item 7); then the milestone.
None of these is on the §16 max list as such; debugging an exact-equality or replay failure in step 1 is.
