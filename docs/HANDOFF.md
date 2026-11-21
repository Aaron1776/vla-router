# Handoff: state of the harness build (2026-11-21)

For a session that starts without the previous conversation. Read this file, then `docs/SPEC.md` (the design;
approved amendments are logged in its §17) and `docs/HARNESS.md` (implementation notes, findings, and the VM
re-run list).

## Working rules the user set

- **Effort checkpoints (SPEC §16).** Before starting any step on the §16 max list, stop and say exactly:
  "Effort checkpoint: please switch to max effort before I start [step]." and wait for confirmation. When the
  step is finished, say: "Effort checkpoint complete: you can switch back to xhigh." Also raise a checkpoint for
  any unexplained discrepancy that could silently corrupt labels or results.
- **Commits.** Only when the user asks. So far the user has asked, before each step, for the previous step to be
  committed on branch `harness-v1` with the test results in the commit message. End commit messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never commit to `main`; never push.
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
| 1. Steps 4 and 5 | not started |
| 2. Server endpoints | X-VLA `/act`, `/sample`, `/features` and π0.5 `/act` exist, single-request, with per-call seeds (CPU generators); dynamic batching missing (step 5) |
| 3. Measurement recording, snapshot store | snapshot capture/restore and a local `SnapshotStore` done and tested; proposal, free signals, spread/drift and probe recording not started; snapshot sync to the bucket not wired |
| 4. Queue, spend guard, packaging | implemented and unit-tested against a local stand-in for the bucket (`src/vlar_ops`); the Cloud Storage backend is untested without credentials; the episode engine is not yet wired to the queue |
| 5. Container images, lockfiles | not started |
| 6. GCP scripts | written (`ops/gcp/`), never run; `IMAGE_FAMILY` and account values are placeholders in `env.example.sh`; the budget alert script covers 50/80/100% of $50 |
| 7. VM re-run list, commits, HANDOFF | the list will need steps 4-5 and the containers added; parity tooling and reference files are committed (`25794f7`) |

## Status

| Work | State | Commit |
| --- | --- | --- |
| Groundwork: versioned config with named options and calibration file, job planner and resumable queue, spend guard, archive packaging, GCP scripts and runbook (`ops/`) | done | `4960782` |
| Effort step 1: simulator layer, X-VLA and π0.5 adapters, both model servers, exact-equality tests against the reference code | done on Mac; VM re-run pending | `4960782` |
| Effort step 2: exact snapshot capture/restore, determinism tests | done on Mac; VM re-run pending | `5d92d97` |
| Effort step 3: episode engine (`src/vlar_harness/episode.py`): handoff rules, controller mode switching, budget, run modes; scene/rollout seed split; reference-run files and checks | done on Mac; VM re-run pending | `25794f7` |
| Decisions 9-11 recorded, relative VM handoff test, this file, `CLAUDE.md` | done | the commit that added this file |
| Effort step 4: event log, overhead profiles, offline clock re-scoring, labels, time-limit calibration | **next** (not started) | |
| Effort step 5: request batching in both servers, batched-vs-unbatched equivalence (per-call seeds already supported) | not started | |
| Render only the observations that are used (SPEC §12) | approved, may start now that restore exists; must pass the same exact-equality tests | |
| Decider inputs (`observe()`: features, proposal, free signals, spread, drift), `run_branch`, dataset storage format (Parquet) | not started | |
| Container images and lockfiles; VM bring-up; Phase 0 runs (latency measurement, parity, base rollouts, calibration, freeze) | not started | |
| Effort steps 6-7 (fold/leakage review; post-pilot re-budgeting and results) | later phases | |

Test suites at `25794f7` (Mac): main venv 48 passed, 4 modules skipped; simulator env 36 passed, 6 skipped
(3 need model weights, 3 need both servers on the VM); X-VLA env (random weights, CPU) 2 passed.

**VM re-run list:** `docs/HARNESS.md`, section "Platform rule (SPEC §11 step 3 as amended)".

## Running things locally

The local environments are git-ignored and live in the repo root:
- `.venv`: Python 3.13, the unit tests: `.venv/bin/python -m pytest`.
- `.venv-sim`: LIBERO (hf-libero 0.1.4), robosuite 1.4.0, MuJoCo 3.8.1, LeRobot 0.6.1, CPU torch. Reference tests:
  `.venv-sim/bin/python -m pytest tests/reference` (about 4 minutes; `tests/reference/conftest.py` sets
  `LIBERO_CONFIG_PATH=third_party/libero-home` and `MUJOCO_GL`).
- `.venv-xvla`: the X-VLA authors' pins, CPU. `VLAR_XVLA_CONFIG_DIR=third_party/hf/X-VLA-Pt VLAR_TEST_DEVICE=cpu
  .venv-xvla/bin/python -m pytest tests/reference/test_model_servers.py -k xvla` (random weights).
- `third_party/` (git-ignored): the X-VLA checkout at 6c4ecfb8, a LeRobot v0.6.1 clone, unpacked hf-libero 0.1.4
  and robosuite 1.4.0 wheels, the LIBERO scene assets at 0b3ea86b, LIBERO's config, small Hugging Face config files.
- `vlar-ops reference check --dir References` verifies the reference files' pinned SHA-256 and totals.
- Do not wrap commands in `timeout`: the one on this Mac is an x86_64 binary and starts the universal Python as
  x86_64, which breaks the arm64 wheels.

## Decisions made in the previous session, and where they are recorded

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
   task's limit (§5, §11 step 6).
10. The real-model handoff check on the VM is relative: X-VLA's first target after a handoff must be at or below
    the 99th percentile of its first-step target distances at ordinary chunk boundaries (no handoff) in the base
    rollouts; the fixed 3 cm stays for the fake-model tests on the Mac (§11 step 3). Implemented in
    `tests/reference/test_handoff.py::test_model_handoff_continuity_on_the_vm`, which reads the distances from
    the JSON file named by `VLAR_BASE_FIRST_TARGETS` (produced from the base rollouts' event logs, see step 4 below).
11. The reference run's 11 per-episode files are committed byte for byte in `References/` with their SHA-256
    pinned in `src/vlar_ops/configs/reference-run.sha256`; the parity tooling verifies them (§2, §11).
12. The milestone checkpoint for the end of the harness build (§18, quoted in full above).

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

## Open questions for the user

- **π0.5-alone runs and network failures.** Proposal (not yet put to the user): in `pi05_only` mode a failed
  request costs time (timeout plus backoff) and is retried until it succeeds, since there is no local model to
  fall back to; π0.5's actions, and so parity, are unaffected. Needs approval and a line in SPEC §5.
- **Upload size in the network model.** The spec says "two JPEG-compressed images"; proposal: measure the median
  JPEG size of the two images at the upload resolution (224 px, quality 90, both are `[ASSUMPTION]`s in the
  config) in Phase 0 step 2 and store it as a byte count in the `network` option, rather than encoding every request.
- **GCP readiness** (from the first session): GPU quota (`GPUS_ALL_REGIONS` and `PREEMPTIBLE_NVIDIA_L4_GPUS` ≥ 1)
  and a billing account that allows GPUs; the user has not reported their status.

## Next step: effort step 4 (max effort)

Scope (SPEC §16 item 4): the event log, overhead profiles, offline clock re-scoring, label computation and
time-limit calibration (SPEC §5, §7, §11 step 6). Raise the effort checkpoint before starting.

Design notes already formed (none of this is implemented yet):
- **Seams that exist.** `Episode(remote=...)` takes a `RemoteLink`; `DirectLink` is the no-network stand-in.
  `Episode._emit` collects provisional events (`start`, `decision_point`, `xvla_act`, `pi05_request`, `handoff`,
  `escalation_abandoned`, `segment`, `end`, `restore`). `seeding.call_seed` and `seeding.stream_seed` exist.
  Config already has `latency` (placeholders), `profiles` (overhead terms per point kind) and the `network` option.
- **Network model** (`network.py`). Per request, an RNG seeded from (rollout seed, step), not a sequential stream,
  so one request's draws never shift another's. Per attempt: a failure draw with `p_fail`, and a log-normal RTT
  with median 0.06 s and sigma = ln(0.30/0.06)/z(0.98) ≈ 0.784; attempt time = upload + RTT + download + π0.5
  inference; it fails if the failure draw fires or that time exceeds the 1.0 s timeout. A failure is noticed at the
  timeout and followed by a 0.2 s backoff before the next attempt; after 1 + 2 failed attempts the escalation is
  abandoned. The π0.5 server is only called for the attempt that succeeds. Record the draws and outcomes per attempt.
- **Clock** (`clock.py`). One `charge(event, profile, charges)` function used by both the live engine and offline
  replay: step 0.05 s; X-VLA `/act` and `act_fresh` t_act; π0.5 request = Σ attempts (failed: timeout, plus backoff
  if another attempt follows; ok: upload + RTT + download + t_pi05); decision point = the profile's terms for that
  point kind (`features`, `sample_k`, `sample_k_less_act`); everything else 0; decider inference 0. Emit one event
  per environment step live (so the clock can cut after any step) and store runs of steps run-length encoded;
  replay expands them and adds in the same order, so live and offline floats are identical.
- **Live cut.** With a time limit (live learned deciders), check after every step and every model call; end with
  reason `time_limit`. Scripted runs never cut; they are scored offline for any profile. The clock is monotone, so
  "success within the limit" offline is "clock at the success step ≤ limit".
- **Labels** (`labels.py`). Branch b at point d under profile P: the fraction of b's seeds whose concatenated log
  (the snapshot's event log up to d, then the branch's own log, which starts with a zero-cost `restore`) shows
  success by step 520 with clock ≤ the task's limit. A base rollout is seed 1 of `b1_continue`.
- **Branches** (`Episode.run_branch`). Restore the snapshot, set the rollout seed to
  `seeding.branch_rollout_seed(parent, d, branch, seed_index)`, then scripted decisions to step 520 with no time
  cut: b1 local throughout; b2 and b3 escalate at the first 1 and min(3, budget) decision points (an abandoned
  escalation counts as one of them); b4 escalates while the budget allows. Shared branches (`vlar_ops.plan`) run once.
- **Time-limit calibration** (`calibration.py`). Per task, p90 (NumPy's default linear method) of X-VLA
  success times under profile `none` from the base rollouts; margin from decision 9, pooled over tasks; raise an
  error if no candidate reaches 80%; write `time_limit.per_task_s`, `margin` and `status` with
  `config.write_calibration`; report per task the limit and the share of π0.5 successes that fit.
- **Diagnostics for decision 10.** Each `xvla_act` event should record the first queued target's distance from the
  measured arm position at query time, and its context (`start`, `stretch`, `after_pi05`, `after_abandon`); the
  `stretch` values from the base rollouts make the `VLAR_BASE_FIRST_TARGETS` file.
- **Tests to write.** Live clock equals offline replay for every profile, including network failures, retries,
  abandonment and a time-limit cut; network statistics (median RTT, tail probability, failure rate, abandonment
  ≈ p_fail³); labels under several profiles; calibration on synthetic records and a calibration-file round trip;
  branch determinism (same seed reproduces, a different seed index differs).

After step 4: effort step 5 (dynamic batching in both servers, batched vs unbatched equivalence), then rendering
only what is used, container images and lockfiles, and the VM bring-up for the Phase 0 runs.
