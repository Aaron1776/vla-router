# Handoff: state of the harness build (2026-11-21)

For a session that starts without the previous conversation. Read this file, then `docs/SPEC.md` (the design;
approved amendments are logged in its §17) and `docs/HARNESS.md` (implementation notes, findings, and the VM
re-run list).

## Working rules the user set

- **Effort checkpoints (SPEC §16).** Before starting any step on the §16 max list, stop and say exactly:
  "Effort checkpoint: please switch to max effort before I start [step]." and wait for confirmation. When the
  step is finished, say: "Effort checkpoint complete: you can switch back to xhigh." Also raise a checkpoint for
  any unexplained discrepancy that could silently corrupt labels or results.
- **Commits and pushes.** Only when the user asks. So far the user has asked, before each step, for the previous
  step to be committed on branch `harness-v1` with the test results in the commit message. End commit messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The user had `harness-v1` pushed to `origin` on
  2026-11-21 so the project partner can see it; push again only when asked. Never commit to or push `main`.
- **docs/STATUS.md** is the overview for readers with no context (the project partner and their AI model); update it
  whenever the state changes, together with this file.
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
| 1. Steps 4 and 5 | step 4 done on the Mac (uncommitted until the user asks); step 5 not started |
| 2. Server endpoints | X-VLA `/act`, `/sample`, `/features` and π0.5 `/act` exist, single-request, with per-call seeds (CPU generators); π0.5 `/upload_bytes` added for Phase 0 step 2; dynamic batching missing (step 5) |
| 3. Measurement recording, snapshot store | snapshot capture/restore and a local `SnapshotStore` done and tested; proposal, free signals, spread/drift and probe recording not started; snapshot sync to the bucket not wired |
| 4. Queue, spend guard, packaging | implemented and unit-tested against a local stand-in for the bucket (`src/vlar_ops`); the Cloud Storage backend is untested without credentials; the episode engine (now with `run_branch`) is not yet wired to the queue |
| 5. Container images, lockfiles | not started |
| 6. GCP scripts | written (`ops/gcp/`), never run; `IMAGE_FAMILY` and account values are placeholders in `env.example.sh`; the budget alert script covers 50/80/100% of $50 |
| 7. VM re-run list, commits, HANDOFF | the list covers steps 1-4 (`test_clock_live.py` and the Phase 0 step 2 upload measurement added); step 5 and the containers still to add; parity tooling and reference files are committed (`25794f7`) |

## Status

| Work | State | Commit |
| --- | --- | --- |
| Groundwork: versioned config with named options and calibration file, job planner and resumable queue, spend guard, archive packaging, GCP scripts and runbook (`ops/`) | done | `4960782` |
| Effort step 1: simulator layer, X-VLA and π0.5 adapters, both model servers, exact-equality tests against the reference code | done on Mac; VM re-run pending | `4960782` |
| Effort step 2: exact snapshot capture/restore, determinism tests | done on Mac; VM re-run pending | `5d92d97` |
| Effort step 3: episode engine (`src/vlar_harness/episode.py`): handoff rules, controller mode switching, budget, run modes; scene/rollout seed split; reference-run files and checks | done on Mac; VM re-run pending | `25794f7` |
| Decisions 9-12 recorded, relative VM handoff test, this file, `CLAUDE.md` | done | `b3e93da` |
| Decisions 13-16 recorded (network failures, upload size, no overhead without budget, `reset(time_limit)`), Google Cloud readiness noted | done | uncommitted |
| Effort step 4: network model, event log, live clocks for every profile and offline replay, live time cut, π0.5-alone retries with the 20-attempt cap, `run_branch`, labels, time-limit calibration, handoff diagnostics, upload-size tooling (`network.py`, `clock.py`, `episode.py`, `labels.py`, `calibration.py`, `eventlog.py`) | done on Mac; VM re-run pending | uncommitted |
| Effort step 5: request batching in both servers, batched-vs-unbatched equivalence (per-call seeds already supported) | **next** (not started) | |
| Render only the observations that are used (SPEC §12) | approved, may start now that restore exists; must pass the same exact-equality tests | |
| Decider inputs (`observe()`: features, proposal, free signals, spread, drift), dataset storage format (Parquet), queue worker wiring (`run_branch` exists) | not started | |
| Container images and lockfiles; VM bring-up; Phase 0 runs (latency and upload measurement, parity, base rollouts, calibration, freeze) | not started | |
| Effort steps 6-7 (fold/leakage review; post-pilot re-budgeting and results) | later phases | |

Test suites after step 4 (Mac, uncommitted): main venv 84 passed, 6 skipped (the simulator modules and the
NumPy cross-check need NumPy); simulator env 46 passed, 6 skipped (the same 6 as before: 3 need model weights,
3 need both servers on the VM); X-VLA env (random weights, CPU) 2 passed. At `25794f7`:
main 48 passed, simulator 36 passed and 6 skipped, X-VLA 2 passed.

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
- Step 4 (HARNESS.md "Event log, clock, network, labels and time limits"): the engine keeps one live clock per
  profile and checks each against the offline replay at the end of every episode (`ClockMismatch`); a segment's
  steps are one log event, charged one step at a time both live and offline; network draws via
  `seeding.network_attempt_seed` + `seeding.uniform` (no generator state); the default link is the simulated
  network, `DirectLink` is for tests only; branches escalate at their first `pi05_calls` decision points, an
  abandoned escalation counting as one (keeps planner-shared branches identical); the clock charges the logged
  network outcome and never re-decides it; calibration's p90 is NumPy's linear percentile re-implemented in pure
  Python and checked equal bit for bit; the upload-size placeholder is a Mac estimate of 20,290 bytes.

## Open questions for the user

None open. Decisions 13-16 above settled the questions raised before and during step 4.

**Google Cloud readiness (user's action, reported 2026-11-21).** The user is upgrading from the free trial if
needed and requesting GPU quota: global GPU quota (`GPUS_ALL_REGIONS`) ≥ 1, preemptible L4
(`PREEMPTIBLE_NVIDIA_L4_GPUS`) ≥ 1 and on-demand L4 (`NVIDIA_L4_GPUS`) ≥ 1 in us-central1 (the region in
`ops/gcp/env.example.sh`; the on-demand quota covers the §12 fallback when Spot capacity is short). The user will
confirm when it is approved. Nothing in effort steps 4-5 depends on it; it gates only the VM bring-up, which in
any case waits for the §18 milestone and the user's go-ahead.

## Next step 1: four approved follow-ups to step 4 (do these first)

Approved by the user on 2026-11-21, not yet implemented (the session ran out of usage). Implement each with tests,
log them in SPEC §17 (as items 19-22), then update this file and `docs/STATUS.md`, commit and push.
1. `time_limit=True` is allowed only in routed mode with a profile other than `none`; `Episode.start` (and the
   client API's `reset`) reject every other combination with a clear error.
2. Record the flag in every episode's metadata and event log (an explicit `time_limit` true/false in the `start`
   event, which today has only `time_limit_s`, and in the episode result/record).
3. A cut run can be scored only under its own profile: `clock.score`, `labels` and `calibration` (and any re-scoring
   tool) must refuse any other profile for a log whose `start` event has the flag set.
4. Label rule (SPEC §7): when a decision point's clock under profile P, computed from the parent's event log (the
   snapshot's prefix), already exceeds the task's time limit, exclude that point from P's training and validation
   data. Count the exclusions per profile and report them (e.g. from `labels.point_labels` or a summary helper).

## Next step 2: effort step 5 (max effort)

Scope (SPEC §16 item 5): request batching and per-call seeding in both model servers (§3, §12), including the
batched-versus-unbatched equivalence check. Raise the effort checkpoint before starting.

Design notes (none of this is implemented yet):
- **What exists.** Both servers take per-call seeds and draw each sample's initial noise on the CPU from its own
  generator (`servers/common.seeded_noise`), so a sample's noise depends only on its seed, not on its batch. Each
  service holds a lock and runs one request at a time. The X-VLA server's `/sample` already batches the K samples
  of one request (one encoder pass, K denoising rows). Config `servers.batching` has placeholders (`max_batch` 8,
  `wait_window_ms` 5) to be set from Phase 0 step 2.
- **Dynamic batching.** Per server, one GPU worker thread and a request queue: handlers enqueue (payload, future)
  and await; the worker takes the first request, waits up to `wait_window_ms` for more (up to `max_batch` rows),
  runs one batched pass, and resolves each future with its own slice. Batch only requests of the same endpoint
  (X-VLA `/act`, `/sample` and `/features` have different work; `/sample` rows count K each). Keep the reference
  behaviour for a request without seeds (global RNG) out of batches, or refuse it in batched mode.
- **X-VLA.** Batch the processor outputs (images, 50 padded text tokens, proprio, domain id) along dim 0; encoder
  once; denoising with stacked per-seed noise; pooled features per row (the image mask is per row). Check the
  authors' `forward_vlm`/`generate_actions` code for anything that assumes batch size 1.
- **π0.5.** LeRobot's processors take a batch dimension (the rollout uses gymnasium batching). The prompt includes
  the tokenised state but is padded on the right to `tokenizer_max_length` 200 (`processor_pi05.py:142-144`), so
  rows have equal length; confirm the attention masks keep rows independent and that `predict_action_chunk` takes
  a [B, 50, 32] noise tensor (`modeling_pi05.py:620-638`). Post-processing is per action row, as now.
- **Equivalence (§11 step 3).** Batched and unbatched calls must give the same executed actions within numerical
  tolerance: on the Mac with random weights on the CPU (X-VLA env; the π0.5 side needs weights, so on the VM),
  compare each row of a batch of mixed requests with the same request alone, and show that a row's result does not
  depend on its batch-mates or position. Decide the tolerance from the measured differences (CPU may well be exact;
  cuBLAS kernel choice can change with batch size on the GPU) and record it.
- **Engine side.** No change: each simulator worker sends single requests; the servers batch across workers.
  `run_branch` and the episode engine are ready for the queue workers.
- **Tests to write.** The batcher (ordering, wait window, max batch, error isolation: one bad request must not fail
  its batch-mates), equivalence as above, the reference equivalence tests still passing through the batched path,
  and a load test with several concurrent clients.

After step 5: rendering only what is used, decider inputs (`observe()`) and measurement recording, the dataset
format and queue workers, container images and lockfiles, and the VM bring-up for the Phase 0 runs.
