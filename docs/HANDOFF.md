# Handoff: state of the harness build (2026-11-24: stage B done through step 6, freeze waits for the user)

Account values (project, bucket, VM name, billing account, service account) are written as `$PROJECT_ID`, `$BUCKET`, `$VM_NAME`, `$BILLING_ACCOUNT`,
`$SA_NAME`: the git-ignored `ops/gcp/env.sh` holds them (`source ops/gcp/env.sh`); `tests/test_no_identifiers.py`
fails if they reappear in a tracked file.

For a session that starts without the previous conversation. Read this file at startup; read `docs/SPEC.md` (the
design; approved amendments are logged in its §17) and `docs/HARNESS.md` (implementation notes, findings, and the VM
re-run list) only in the sections the task needs (`CLAUDE.md`, "Usage discipline").

## Done 2026-11-23 (late evening): the ops-digest rule (amendment 49, refining 46), committed and pushed

- **Verified first (the user's condition):** nothing read `branches.seeds` or `branches.enabled` from the config.
  Enabled branches came from the label entries (`shares`). The seed count came from neither the config nor the entry:
  it was a caller argument (`complete_points(seeds=)`, `vlar_train.data.build(seeds=)`, `vlar_train.run --seeds`
  default 3). Now every branch-plan entry and job spec records the planner's `settings`, and `complete_points` and
  `vlar_train` read the seed count from the entry. `labels.point_labels` (harness, frozen code) is unchanged: its
  caller passes the entry's value.
- **Code:** `vlar_ops/datadigest.py` (common and per-kind digests; definition-level closure), `tunables.py` +
  `configs/collection-v1-tunables.yaml` (the eight tunables; `collection-v1.yaml` keeps `schema` and
  `rollout_seed_namespaces`), `branchplan.py` (planner moved out of `plan.py`; records `settings`, applies the dp
  sampling; the future branch-plan command goes here), `waiver.py` (worker-side waiver), `vm_worker.serve` (waiver,
  `meta.boot_id`, `meta.golden_pass`, `meta.ops_waiver`, p_escalate from the tunables, exploration jobs with another
  spec p_escalate given back), `cli.py` (exploration plans record `p_escalate`, `episodes_per_state`; refuses != 1;
  `dataset manifest --ops-archives`), `manifest.py` (freeze rules), `vlar_train` (seeds from entries, `--seeds` gone).
  `ProductionEpisodes`, `XVLA_CLIENT_FILE`, `rollout_seed`, `episode_plan` untouched: **common data digest equals
  `0a0cd4c`'s** (`4951f7f0...`, pinned in `tests/test_datadigest.py`). Base/explore generator digests differ from
  `0a0cd4c`'s (`cmd_plan` changed); allowed, since kinds compare only among their own records and all base records are
  at the freeze's ops digest.
- **Tests:** `tests/test_datadigest.py` (16), `tests/test_waiver.py` (28: every other freeze problem alone and with an
  ops mismatch, source/archive checks, data-digest changes, serve wiring), 6 new in `test_manifest.py`; planner tests
  moved to `branchplan`. Main venv 300 passed, 28 skipped; simulator env (all of `tests`) 473 passed, 21 skipped (the same 21); X-VLA env not re-run (nothing it runs changed). Mutation check of the digest tests: 5 of 5 caught (imports not followed, definitions dropped, tunables not excluded, docstrings kept, branch root narrowed).
- **Deployment:** none. `CURRENT` stays `0a0cd4c` through stage B, `500` and the freeze; the new ops code runs only
  after the freeze, under the waiver. Exploration plans must be made with the new `vlar-ops plan` (records p_escalate).
- **Before Phase 1 queues real work:** one smoke job on the frozen harness under the new ops code; confirm the
  marker's `meta.ops_waiver`, `boot_id` and `golden_pass` (the waiver wiring in `serve` is tested only with stubs).

## Done 2026-11-23 (evening): the must-do changes and the partner's HTTP finding, merged into `harness-v1`

- **π0.5 HTTP 422 (amendment 45, the partner's review):** `pi05_server.py` imported FastAPI's `Request` only inside
  `create_app`; with string annotations every `/act` and `/upload_bytes` call failed validation (422). Fixed at module
  level (`071a2a7`); X-VLA and the other server modules checked, no other case. `tests/reference/test_server_http.py`:
  every endpoint of both servers through the production clients; the old import fails the 3 π0.5 tests with 422.
  The images' test runs install hashed test-only client locks (`containers/lerobot-test.lock`, new;
  `xvla-test.lock`) and run these tests in all three images. `test_vm_startup.py` and `test_phase0_steps.py` now skip
  inside an image (the cause of the `b4c6beb` rebuild's 6 failures).
- **Must-do 1, ops code pinned (amendment 46):** `provenance.ops_digest()` in every record header, `freeze.json`,
  `verify_freeze` (workers' startup), the manifest; `600-freeze.sh`'s verification container now mounts the host
  `vlar_ops`. **Must-do 2:** step `500` ends with exit 75 after writing the calibration and uploading
  `phase0/step6/calibration-summary.{txt,json}`; it never queues `600`. A re-run with the same records and calibration
  file ends done (exit 0) so an approved `600` queued behind it runs. **Queue `600` only with the user's explicit
  approval after they read the summary.** (`phase2-work` `1e78ede`, merged.)
- The L4 was found STOPPED with an empty queue (stage B never queued): nothing to stop.
- Suites on the merge (Mac): main venv 250 passed, 24 skipped; simulator env, all of `tests` (as the sim image
  runs it) 406 passed, 21 skipped; X-VLA env (`test_model_servers`, `test_batch_invariance`, `test_server_http`,
  `-k xvla`, CPU random weights) 16 passed.

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
5. The three Dockerfiles and their lockfiles (with hashes) are complete, and everything that can be checked without a container runtime is checked on the Mac; building the images and the in-container tests are the first step of Phase 0 (§11 step 0, amendment 28).
6. Ready-to-run setup scripts exist for the Google Cloud side: the g2-standard-16 Spot VM, the bucket, and the budget alert at 50%, 80% and 100% of $50.
7. The VM re-run list in docs/HARNESS.md is complete, the parity tooling and reference JSON files are committed, everything is committed, git status is clean, and HANDOFF.md is up to date.

When all seven are true, stop and tell the user exactly: "Milestone checkpoint: the harness build is complete and Phase 0 on the L4 VM is next." With it, give a short checklist confirming each of the seven items with its commit, the estimated Phase 0 cost and VM hours, and the first commands Phase 0 will run. Then wait for the user's explicit go-ahead before creating any cloud resources or spending any money. Do not announce this milestone early or partially; if any item is incomplete, say which ones remain instead.

Where each item stands now (all seven hold as of the commit after `c6d49ed`; milestone announced 2026-11-22, waiting for the user's go-ahead before any cloud resource):

| Item | State |
| --- | --- |
| 1. Steps 4 and 5 | done on the Mac: step 4 (`272f64b`, follow-ups `a25053f`); step 5 (part 1 `f8d06d5`, part 2 `94b9297`, the X-VLA concurrency fix in the commit after `c1fcd78`); their equality tests are on the VM re-run list |
| 2. Server endpoints | X-VLA `/act`, `/sample`, `/features` and π0.5 `/act` with per-call seeds, all batched at a fixed geometry of rows and tokens (amendments 24, 26); unseeded requests and π0.5 `/upload_bytes` stay on the reference path |
| 3. Measurement recording, snapshot store | snapshot capture/restore and a local `SnapshotStore` done and tested; decider inputs and measurement recording (`observe()`, proposal, free signals, spread/drift, chunk-time samples, probes, `record="full"`) done and tested on the Mac with fake models (`6c08bf5`); Parquet dataset records and snapshot sync to the bucket (through the worker) done and tested against a local bucket (the commit after `6c08bf5`); the Cloud Storage path is untested |
| 4. Queue, spend guard, packaging | done on the Mac against a local stand-in for the bucket (`src/vlar_ops`): queue, spend guard, packaging, runner and worker loop (the commit after `6c08bf5`), the VM worker command `vlar-ops worker` and the dataset manifest and datacard (`vlar-ops dataset manifest`/`verify`; the commit after `1683b56`); the Cloud Storage backend is first exercised by the VM smoke run on the re-run list |
| 5. Dockerfiles, lockfiles | done on the Mac, the commit after `1e92964` (wording changed by amendment 28): `containers/` (Dockerfile with targets sim/pi05/xvla, hashed linux/x86_64 locks, `check_env.py`, `locktool.py`, `lock.sh`), `ops/gcp/35_build_images.sh` + `build_vm_startup.sh` (CPU build VM), `ops/vm/load_images.sh` (driver check, load, GPU checks), service units; image builds and in-container tests are Phase 0's first step |
| 6. GCP scripts | done 2026-11-22: scripts ready to run, `ops/gcp/env.sh` and `ops/ops.yaml` filled in (git-ignored; billing account `$BILLING_ACCOUNT`, the Education account; `IMAGE_FAMILY` common-cu129-ubuntu-2404-nvidia-580, driver 580); `05_check_account.sh` (read-only) confirmed billing linked and enabled, GPUS_ALL_REGIONS 1, NVIDIA_L4_GPUS 1, PREEMPTIBLE_NVIDIA_L4_GPUS 1 (unused), compute/storage/logging/monitoring APIs on (iam and billingbudgets enabled by `10_bucket_and_account.sh`); budget excludes credits (coupon account). Nothing created |
| 7. VM re-run list, commits, HANDOFF | VM re-run list complete (image build and driver check first, then steps 1-5, rendering, worker, manifest, smoke run); parity tooling and reference files committed (`25794f7`); everything committed with the item-6 commit after `c6d49ed`, tree clean |

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
| VM worker entry point (`vlar_ops/vm_worker.py`, `vlar-ops worker`: production factory with `XVLAClient`/`PI05Client`, one `LiberoSim` per task, server geometry and deterministic checks, the spend guard's `StopFlag`, clean abandon on SIGTERM, refuse-and-stop for jobs the worker's options cannot run, provenance in markers and snapshots, `ops/vm/vlar-worker@.service`); dataset manifest and datacard (`vlar_ops/manifest.py`, `vlar_ops/datacard-v1.md`, `vlar-ops dataset manifest`/`verify`) (HARNESS.md "VM worker entry point, dataset manifest and datacard") | done on Mac; VM smoke run pending | the commit after `1683b56` |
| Container definitions (amendments 28-30; HARNESS.md "Container images and lockfiles"): Dockerfile, hashed locks at the notebook's versions, CPU build VM, L4 loader with driver check, service units, `tests/test_containers.py` | done on the Mac | the commit after `1e92964` |
| Phase 0 step 0: container images built and tested on a CPU build VM, exported to the bucket (amendments 28, 31-33) | done | `be7b95a` |
| Phase 0 stage A on the L4 (images `be7b95a`): image load, host checks, weights, VM re-run rows, verdict | failed at 090: row pi05-batch-invariance-l4 (padded vs batch of one, max |diff| 1.94); all other rows passed; VM stopped 05:21:59Z; spend $0.73 | `94b1966`, `6690b19` |
| π0.5 diagnosis (step 095, 2026-11-23): numerical, not a bug (HARNESS.md); host-package freeze (amendment 36); approved rules (joint memory, grasp check) | done, uncommitted; option for π0.5 open; spend $0.86 | |
| Phase 0 stage B on the L4 (images and host source `03f311c`, 2026-11-23): rebuild, image load, re-run rows, verdict, step 2 | 010-090 done, every re-run row passed (π0.5 batch invariance included); **200 failed** (UnboundLocalError in `vlar_ops/step2.py`:390, not debugged); VM stopped 14:48Z; spend $1.70 | `03f311c` |
| Amendment 35 and stage B code: calibration file with write-once group, freeze lineage, refusals, step 2 tools, parity gate, base step (HARNESS.md "Calibration file and stage B") | done on the Mac; main 204 passed, 19 skipped; simulator env 150 passed, 17 skipped (the same 17); X-VLA env `-k xvla` 2 passed | the commit after `c991493` |
| Phase 0 steps 2-7 on the L4 (step 2, parity, base rollouts, time limits, freeze) | step 2 (`200`, `210`) done; parity (`300`) passed 2026-11-23 22:26Z; base rollouts (`400`) done 2026-11-24 08:41Z, X-VLA parity 398/500 vs 405 (McNemar p 0.47); step 6 (`500`) done 08:48Z: m = 2.0, handoff check passed, grasp check flagged; `600` (freeze) waits for the user's approval; spend $7.43 | ops `0a0cd4c`, images `03f311c` |
| Effort steps 6-7 (fold/leakage review; post-pilot re-budgeting and results) | later phases | |

**Tests after the container definitions (Mac, 2026-11-22).** Main venv 158 passed, 13 skipped (`tests/test_containers.py` 19); simulator env 125 passed, 17 skipped; `uv pip install --dry-run` of every lock for cp313 linux/x86_64 with hashes required: 172 wheels + 5 sdists, 69, 4.

**Tests after the VM worker entry point, manifest and datacard (Mac, 2026-11-22).** Main venv 139 passed, 13 skipped; simulator env (`tests/reference tests/test_measure.py tests/test_labels_calibration.py tests/test_manifest.py`) 125 passed, 17 skipped (the same 17 as before), about 4.5 minutes. Mutation: forcing the X-VLA state mode to `measured` in the production factory fails the record comparison.

**Test suites after dataset records, runner and worker (Mac, 2026-11-21).** Main venv 122 passed, 11 skipped; simulator env (`tests/reference tests/test_measure.py tests/test_labels_calibration.py`) 114 passed, 17 skipped. From 2026-11-22 the simulator suite also includes `tests/test_manifest.py`.

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
- `.venv-sim`: LIBERO (hf-libero 0.1.4), robosuite 1.4.0, MuJoCo 3.8.1, LeRobot 0.6.1, CPU torch. Its LIBERO finds
  its assets through a hand-made symlink `.venv-sim/lib/python3.13/site-packages/libero/libero/assets ->
  third_party/libero-assets` (made 2026-11-20; recreate it if the venv is rebuilt; amendment 32). Reference tests:
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
28. Container builds (2026-11-22, during §18 item 5; §11, §13, §15, §18): the Dockerfiles and hashed lockfiles are
    completed and checked on the Mac; building the images and the in-container tests move to the start of Phase 0
    (§11 step 0), on a small CPU-only VM (billing only, no GPU quota); images exported as files to the bucket, not
    Artifact Registry; the L4 VM loads them; GPU-only tests run on the L4.
29. Versions (2026-11-22; §2, §11): from the notebook's install output: lerobot 0.6.1, torch 2.11.0+cu130 (also the
    X-VLA server's, reused from Colab), transformers 5.5.4, robosuite 1.4.0, mujoco 3.8.1; possibly from a later
    re-run than the evaluation runs, so version mismatch heads the parity-debugging checklist (§11 step 4); NVIDIA
    driver 580.65.06+ for CUDA 13.0, checked on the VM (`ops/vm/load_images.sh`, `check_env.py --gpu`).
30. Python (2026-11-22; §2, §11, §15): all three images run CPython 3.13.16, the notebook's (runtime-check cell; the
    X-VLA venv was built on it with `--system-site-packages`), a hash-pinned standalone build; locks re-resolved for
    cp313 (same pinned versions); the Python version is part of the parity checklist's version check.
31-33. (2026-11-22, image builds) sdists built as the notebook did (CMake <4.2, policy minimum 3.5); LIBERO reads
    the pinned assets at `~/.cache/libero/assets`; CPU batch invariance in the images within tolerance, exact on the L4.
34. (2026-11-22) Phase 0 calibration in its own file (hashed, in the bucket, read-only in the containers, in every
    record, pinned by the freeze); no rebuild after the measurements; base rollouts count as v1.0; live
    calibration-dependent values refuse until frozen; tokenizer pinned at 35e4f46.
35. (2026-11-22) The four exceptions: write-once group (π0.5 latency, upload size, batch geometry) set in step 2
    before parity; calibration hash in every record; freeze pins the final file plus its lineage and verifies it;
    `sim_time_s` dropped; gripper fractions offline, `observe()` refuses until frozen; batch geometry in the
    calibration file, one image rebuild in all.

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

None open. The π0.5 question was decided as option A (amendment 37); the partner's review became amendments 39-44.

**Billing and spending rules (user, 2026-11-22).**
- The billing account is a university education coupon for **$50**, on a paid billing account with no payment
  method, credits expiring **2027-08-20**. $50 is a hard ceiling: the spend guard's hard stop stays at $45, and the
  remaining $5 must cover bucket storage and the final download before the expiry (download within a month or two
  of finishing; Cloud Storage's free tier covers 100 GB/month of outbound transfer from North America). Budget
  alerts and ledger reconciliation use the cost before credits.
- **Pauses:** the user works consistently, but if work pauses for more than a day, delete the L4 VM and its disk and
  recreate it from the bucket to resume (ops/README.md, "Costs the spend guard counts"). The disk is 100 GB
  (about $0.33/day; breakdown in ops/README.md, "Boot disk size").
- **Decider training on the Mac where possible** (SPEC §12, §13 already allow it): download only
  `records/{runs,events,points}` (`gcloud storage cp -r gs://$BUCKET/vla-router/records/<table> ...`),
  compute labels locally (`labels.py` is pure Python; a point's prefix is the parent run's `events` rows up to its
  `decision_point` event, so snapshots and images are not needed; check this on first use), train locally, and
  upload the checkpoints for Phase 3 and evaluation. The VM then only collects data and runs evaluation.

**Google Cloud account (user, 2026-11-22).** Project `$PROJECT_ID`, us-central1, bucket
`$BUCKET`; billing is a $50 university education coupon on a paid billing account with no
payment method, credits expiring 2027-08-20: $50 is a hard ceiling, the hard stop stays at $45, and the rest must
cover storage and the final download before the expiry. GPU quota reported approved (GPUS_ALL_REGIONS 1,
NVIDIA_L4_GPUS 1, PREEMPTIBLE_NVIDIA_L4_GPUS 1); to be confirmed with `05_check_account.sh` before anything is
created. The alert email is only in the git-ignored `env.sh`.

**Google Cloud readiness (user's action, reported 2026-11-21).** The user is upgrading from the free trial if
needed and requesting GPU quota: global GPU quota (`GPUS_ALL_REGIONS`) ≥ 1, preemptible L4
(`PREEMPTIBLE_NVIDIA_L4_GPUS`) ≥ 1 and on-demand L4 (`NVIDIA_L4_GPUS`) ≥ 1 in us-central1 (the region in
`ops/gcp/env.example.sh`; the on-demand quota covers the §12 fallback when Spot capacity is short). The user will
confirm when it is approved. Nothing in effort steps 4-5 depends on it; it gates only the VM bring-up, which in
any case waits for the §18 milestone and the user's go-ahead.

## Next step

**NOW (2026-11-24, 08:55Z): stage B is done through step 6. The L4 is STOPPED (TERMINATED 08:48Z) and waits for
the user's decision on the freeze (`600`).** `400` done 08:41:41Z (36925 s; all 500 episodes; no failure, time cap
or preemption, so the pre-approved recovery was not needed). `500` ended with exit 75 at 08:48:34Z as designed and
stays queued: a re-run with the same records and calibration file ends done, so a `600` queued behind it runs.
Spend guard $7.43 (VM $7.03 over 14.93 h, disk $0.40). All results are in `phase0/step6/`:
- `calibration-summary.{txt,json}` (calibration file `1d61294a7f51`, records `888916b00172`: 500 base, 100 π0.5
  parity runs): **margin m = 2.0**, 93.6% of π0.5's 94 successful parity episodes finish within their task's limit
  (candidates 1.25: 9.6%, 1.5: 35.1%, 1.75: 71.3%). Per-task limits 30.2-62.3 s (task 0 57.1, 1 51.8, 2 38.9, 3 39.0,
  4 34.5, 5 30.2, 6 42.6, 7 39.6, 8 62.3, 9 45.9); π0.5 successes over the limit only in task 4 (5 of 9 fit) and
  task 8 (6 of 8).
- Relative handoff check (`logs/results/03f311c.../handoff-relative-l4.*`): **passed** (3 passed; the base rollouts'
  ordinary first-target distances in `first_targets.json`: n = 4941, median 23.6 mm, p99 83.6 mm).
- **Grasp check: FLAGGED** (`grasps.json`): 9.1% of 55,915 settled commanded-closed steps lie within 3.13 mm of the
  9.00 mm shut threshold (limit 2%); shut 25.9%, partly open 74.1%, open 0.0%.
- **X-VLA parity (SPEC §11 step 5):** `xvla-parity.{json,txt}` (`ops/xvla_parity.py`): 398/500 against the
  reference's 405/500 at 520 steps; paired 367 both, 31 only ours, 38 only the reference, 64 neither; exact McNemar
  p = 0.47: within sampling error. Per task, ours vs reference: 30/36, 34/36, 45/44, 46/46, 35/34, 44/42, 34/35, 43/40,
  39/43, 48/49.

**Freeze review (the user, 2026-11-24):** margin 2.0 and the time limits are approved; the handoff check, X-VLA
parity (398 vs 405, p = 0.47) and π0.5 parity are accepted. **Before the freeze, fix the grasp thresholds offline on
the Mac (no VM):** (1) plot the raw finger widths at settled closed-gripper steps of the base rollouts, overall and
per task; (2) ground truth where feasible: restore a few hundred base-rollout snapshots across all tasks in the Mac's
simulator and decide whether an object is held (both fingers in contact with an object, or an object lifted from
its resting place), and report how the current thresholds classify held and not-held cases; (3) propose shut and
open thresholds at the clearest separation, with the resulting grasp-check numbers and misclassification rates on
the ground truth, or, if nothing separates cleanly (thin objects overlapping "closed on nothing"), say so and give
the best achievable rates. **Show the user before writing a new calibration version.** Also (the user): record the
actual post-handoff distances in the handoff check from now on. **To queue `600` once approved:** `gcloud storage cp ops/vm/phase0/steps/600-freeze.sh gs://$BUCKET/vla-router/phase0/queue/` (the script is
unchanged since `0a0cd4c`), then `gcloud compute instances start $VM_NAME --zone $ZONE` and watch. Do NOT use the
summary's printed `31_phase0_queue.sh` command: it would move `CURRENT` off `0a0cd4c`. Pause rule: delete the VM and
its disk if it is still stopped at 08:48Z on 2026-11-25.

**Base-rollout time budget (for Phase 1, measured 2026-11-24).** The records hold no per-call or per-component
wall-clock times (the event log is logical), so this combines the records' call counts, per-episode times from the
markers, and step `200`'s L4 benchmarks (`phase0/step2/03f311c.../bench-xvla.json`):
- Fleet: 500 episodes in 36,925 s on 8 workers = **73.9 s of VM time per episode**; mean 312 steps and 31.5 decision
  points (10.9 chunk, 20.6 mid).
- Calls (record `full`; `episode._chunk_samples`, `_measure_point`): a chunk point makes `/act` (it also returns the
  features) and one `/sample` with K-1 = 4 seeds; a mid point makes one probe `/sample` with K = 5 seeds (it also
  returns the features). `/features` is never called in full recording. Totals: 5,441 `/act`, 5,441 chunk `/sample`,
  10,323 probe `/sample`, 0 `/features`.
- Per worker (gaps between a worker's consecutive markers, n = 492): mean 527 s, median 437 s per episode. A least-
  squares fit gives wall = 0.08 s x steps + 15.7 s x decision points + 6.8 s (R^2 0.993). So the decision points take
  about 94% of a worker's time (including waiting at the X-VLA server), per-step work (simulation, rendering the used
  observations, encoding images for `/act`) about 5%, and the per-episode constant (start, uploads, marker) about 1%.
- X-VLA GPU time per batch at the fixed geometry (batch 4, deterministic): `/act` 1.42 s, `/sample` 5.66 s (20 rows),
  `/features` 0.30 s. `/sample` costs about 0.28 s per row at every size, so batching saves nothing on it. With every
  batch full, per episode: probe `/sample` 29.2 s, chunk `/sample` 15.4 s, `/act` 3.9 s, total 48.5 s of the 73.9 s.
  So the GPU was busy at least 66% of the time; `/sample` is at least 92% of its work and the probes alone about 60%.
- **For the pilot:** throughput is bound by X-VLA `/sample` on the one GPU, mostly the mid-point probes. Recording
  probes at a sampled fraction f of mid points cuts the GPU work by about 0.6 x (1 - f). Simulation, rendering, image
  encoding and uploads are not the bottleneck, but these records cannot separate them. Timing them would need timers
  in the worker and the servers (a code change, not made).
- **For the pilot (the user, 2026-11-24):** investigate why worker `w8` finished only 15 episodes (the others
  66-75), and whether more workers would raise the GPU utilization above its 66% lower bound.

**Before Phase 1 queues real work (the user, 2026-11-23):** run a single smoke job on the frozen harness
under the post-freeze ops code and confirm its marker's `meta.ops_waiver` (and `boot_id`, `golden_pass`):
`serve`'s waiver wiring has only been tested with stubs. Earlier notes: The VM runs `0a0cd4c`'s ops code
(`phase0/src/CURRENT`) on images `03f311c`. At 21:49Z step `200` had passed the geometry rule: batch size 4,
8 workers, 20 sample rows, every size 1-16 feasible (16 leaves 3108 MiB free); it was then measuring latency,
upload size and gripper widths, before the write-once calibration write. `210` and `300` (parity: the first dataset
records) follow, and `300` queues `400` on a pass. The VM stops itself on any failure, gate, needs-input or an
empty queue. **First thing next session:** `export PATH=$HOME/google-cloud-sdk/bin:$PATH; source ops/gcp/env.sh;
gcloud storage cat gs://$BUCKET/vla-router/phase0/STATE.json` and `.../history.jsonl`. Report where it stopped and
why, or reattach the watcher: `caffeinate -i -s ops/gcp/32_watch_vm.sh`; this session's watcher ends with it. On a
stop: don't debug; report. `500` is queued by hand only with the user's OK, by copying the step script so
`CURRENT` stays `0a0cd4c`; `600` only with explicit approval after the step-6 summary. Spend guard: $2.07 at
21:31Z, now including the disk. The start at 21:17Z came after a zone stockout (21:06Z, two failed starts,
`outputs/build-logs/l4-start-retries.log`). The zone fallback below was pre-approved but not needed.
Once the VM is up: `caffeinate -i -s ops/gcp/32_watch_vm.sh`.
**Zone fallback (pre-approved by the user, 2026-11-23 ~21:15Z), if the start still fails when the retry loop ends:**
1. Queue `010,015,020,030,040,050,090` beside `200,210,300`, all from a clean HEAD. This must happen before the VM
   exists, because its runner starts at boot. `090` needs every row on the new disk, and `020` fetches the weights
   and the tokenizer from the bucket.
2. `ZONE=us-central1-{b,c,f} IMAGES_COMMIT=03f311c4cd18a392285709987426bdfc42413fcd ops/gcp/30_create_vm.sh`: Spot,
   one zone after another. The VM name stays the same; names are unique per zone.
3. Once the new VM is RUNNING, delete the old VM and its disk in `us-central1-a`. Set `ZONE` in the git-ignored
   `ops/gcp/env.sh` to the new zone, then watch.
4. If Spot is unavailable in every zone: take `010`-`090` out of the queue again, leave the old VM stopped, and give
   the user the on-demand estimate. On-demand is never used without asking.

**Spend-guard fixes (2026-11-23, 21:20-21:45Z, SPEC §17 amendments 47-48), done while stage B ran:**
- **Spend CLI from the Mac** (amendment 47): `.venv-ops` (git-ignored; google-cloud-storage 3.17.1) with the user's
  application-default credentials. `status`, `disk-created`, `reconcile` and `disk-deleted` were checked against
  the bucket under a scratch prefix, which was deleted afterwards. Usage is in `ops/README.md`, "From the Mac".
  `disk-created --since` and `disk-deleted --until` take real times. Two open records with one name need `--since`.
- **The uncounted disk** (amendment 48): the cause was the pre-amendment-39 race at the first boot (03:52Z). The
  record was written and reported ("recorded disk $VM_NAME-boot" in `phase0/logs/startup.log`) and the marker
  touched; then the guard daemon's concurrent whole-object rewrite dropped the record, and the marker blocked every
  later boot. The user's guess was right. **Recorded at 21:31Z** from the Mac through `disk-created --since`
  gcloud's `creationTimestamp` (2026-11-22T19:51:45.921-08:00). The guard total went $1.83 -> **$2.07** ($0.24 of
  disk), and the daemon's later ticks kept the record. `disk-created` now reads its record back and exits 1 if it is
  missing. `startup.sh`'s `record_boot_disk` touches `.disk-recorded` only after that, with tests and mutation
  checks.
- **Deployment, deliberately partial:** these changes alter `vlar_ops`, and so the ops digest (amendment 46).
  **Keep `phase0/src/CURRENT` at `0a0cd4c` through stage B, `500` and the freeze**: moving it would mix ops digests
  if a preemption rebooted the VM mid-stage. The VM keeps the old `disk-created`. That is harmless, since its disk
  is now recorded and its marker exists. Queue `500` by copying the step script, as planned. The Mac's CLI already
  runs the new code; the ledger format is unchanged. Committed and pushed as `b8ad746` (the user approved). The
  startup-script metadata was re-set from `b8ad746`'s `startup.sh` and checked equal; `CURRENT` stays `0a0cd4c`.
- **The ops-digest rule: done (amendment 49 in SPEC §17; summary at the top of this file).**
 To queue by hand later, from a clean `harness-v1`:
```bash
export PATH=$HOME/google-cloud-sdk/bin:$PATH; source ops/gcp/env.sh
ops/gcp/31_phase0_queue.sh ops/vm/phase0/steps/{200,210,300}-*.sh    # host source and CURRENT -> HEAD; 200 sorts first
gcloud compute instances start $VM_NAME --zone $ZONE
caffeinate -i -s ops/gcp/32_watch_vm.sh
```
The VM installs `CURRENT` on boot, and step `200`'s `vlar-ops` runs mount that host `vlar_ops` over the image's copy
(`OPS_RUN`), so the fix needs no image rebuild. The images stay `03f311c` (`vlar-images-url` unchanged;
`startup-script` unchanged since `03f311c`). On the VM, `200` reuses `obs/` and both benches from 14:00Z. It measures
every size again, because each `memory-B.json` there holds the `load` error (the script change in `44cb3e5`). Then
it runs the geometry rule, latency, upload, gripper and the write-once calibration write.
**Spend guard $1.70**; the build VM's about $0.65 is outside the ledger. The L4's disk bills about $0.33/day.
**Pause rule:** delete the L4 and its disk if it is still stopped at 14:48Z on 2026-11-24.
When stage B resumes: `400` ends with an empty queue, so the VM stops itself. `500` is queued by hand afterwards;
copy the step script straight into `phase0/queue/`, as `300` does for `400`. `600` is queued only with the user's
explicit approval after they read the step-6 summary. Images: `gs://$BUCKET/containers/03f311c4cd18a392285709987426bdfc42413fcd/`.

**Step `200` fix (2026-11-23 evening, `44cb3e5`).**
- **Cause:** `step2.run` assigned a local `load` lambda in its `updates` branch, which made `load` local in the
  whole function. So `vlar-ops step2 load` raised `UnboundLocalError` at every size on the L4.
- **Why no test caught it:** the tests called the functions directly, only `choose` went through the command, and
  nothing ran the step script, which reported the crash as a possible out-of-memory.
- **New tests:** `tests/reference/test_step200_e2e.py` runs the real step script end to end on the Mac against
  stand-ins (HARNESS.md, "Step `200` end to end on the Mac"). It has three tests, and all three fail on the old
  `step2.py` with the L4's error.
- **Step script:** a re-run now measures again a size whose `memory-B.json` holds an error.
- **Suites:** main venv 250 passed, 27 skipped; simulator env 409 passed, 21 skipped; X-VLA env 16 passed.

The stop at 14:52Z, for the record: stage B STOPPED at step `200` (FAILED, exit 1). The geometry rule found no
feasible batch size: at every size the memory probe recorded "full batches failed (a server may have run out of
memory)", while the log showed `UnboundLocalError` in `src/vlar_ops/step2.py`, line 390, once per size. The size-16
probe peaked at 13056 of 23034 MiB. **Nothing was written to the calibration file** (`vla-router/calibration/` is
empty). Partial results are in `phase0/step2/03f311c.../`; logs are in `outputs/build-logs/stageB-03f311c/`.

Run log (UTC; spend = the L4 spend guard's total from `history.jsonl`; the build VM is outside the ledger):
- 10:41Z: merge `03f311c` pushed to `harness-v1`. Rebuild started on-demand (`BUILD_PROVISIONING=STANDARD caffeinate
  -i -s ops/gcp/35_build_images.sh`, local log `outputs/build-logs/rebuild-03f311c.log`).
- 11:11Z (the user's direction): the failed `b4c6beb` folder and, once the new images are verified, the superseded
  `be7b95a` images are to be deleted, logs saved first. Both folders' non-image files were copied to
  `outputs/build-logs/b4c6beb/` and `outputs/build-logs/be7b95a5d944d351428b5d79978864dc62ad4d1a/`, with sizes checked
  against the bucket. `containers/b4c6beb.../` was deleted (6 objects, 1.2 MiB) and confirmed gone.
- 13:01Z: rebuild **DONE**. The build VM ran 10:44:21Z-13:01:46Z, about 137 on-demand minutes; the 85-90 estimated
  did not allow for the in-container tests' length (sim 35 min, xvla 23 min, pi05 3 min). In-container tests, no
  failures: pi05 11 passed, 1 skipped (+3 HTTP); xvla 16 passed (+4 HTTP); sim 364 passed, 63 skipped (the repo-only
  files skip). Digests: sim `sha256:20d19481...`, pi05 `sha256:39acf6e7...`, xvla `sha256:83896f24...`
  (`image-digests.json`). The build VM and its disk were deleted and confirmed gone. Logs are in
  `outputs/build-logs/03f311c/`. **Build cost about $0.65** (list prices, before credits): e2-standard-8 for 2.29 h at
  about $0.27/h, plus 150 GB pd-balanced and an external IP. The user was told about $0.40.
- 13:02Z: L4 metadata re-set. `vlar-images-url` now points at `containers/03f311c.../`; `startup-script` was re-set
  from `ops/vm/phase0/startup.sh` at `03f311c` and checked equal by hash. The queue was empty. Queued
  `010,015,030,040,050,090,200,210,300`. **13:03:16Z: L4 started (Spot); spend $0.86.**
- Step results (seconds, spend after): `010` done (433 s, $0.95); checksums OK, all three images loaded and
  GPU-checked. `015` done (23 s, $0.96). `030` done (2095 s, $1.23). `040` done (191 s, $1.25). `050` done (593 s,
  $1.33). **`090` done at 14:00:40Z: every re-run row passed on the L4**, including `pi05-batch-invariance-l4`
  (1 passed), the row that failed in stage A. Also passed: host unit tests 247 passed, 27 skipped; sim-reference
  106 passed, 3 skipped; xvla-batch-invariance-l4 10 passed; xvla-vs-authors; pi05-vs-lerobot-eval; reference-check;
  sim-numpy-percentile.
- 14:20:41Z: `containers/be7b95a.../` deleted (11 objects, 10.2 GiB) and confirmed gone. The bucket has no versioning
  and soft delete is off, so nothing is retained. `containers/` now holds only `03f311c.../`.
- 14:00:48Z: `200` (step 2 measurement, cap 3 h) started. **14:48:25Z: `200` failed (exit 1, 2855 s, spend $1.70)**;
  the reason was "geometry rule failed (exit 1)". The VM stopped itself (TERMINATED at 14:48:38Z); the watcher
  exited. Details are in the NOW paragraph above.

Pause rule: if the L4 is STOPPED for a day, delete it and its disk (then recreate it with `30_create_vm.sh` and
queue `020` too).

**REBUILD RESULT (2026-11-23, 08:43Z-10:05Z): FAILED in the in-container tests; stage B NOT queued.** The build VM
and its disk were deleted and confirmed gone (`gcloud compute instances list`: only the stopped L4). Images built;
tests: pi05 8 passed, 1 skipped; xvla 12 passed; **sim 6 failed**, 337 passed, 48 skipped. All 6 failures are
`tests/test_vm_startup.py` (new in `b4c6beb`), which reads `ops/vm/phase0/startup.sh`; `ops/` is not copied into the
images, and the repo's other repo-only tests skip there (`VLAR_IN_IMAGE`). Likely fix (not verified, not debugged, as
the user asked): mark the file repo-only like `tests/test_containers.py`'s `repo_only`, and check `phase2-work`'s new
repo-only tests (`test_phase0_steps.py`, `test_no_identifiers.py`) the same way; then a second rebuild (about 85 min,
about $0.40) needs the user's go-ahead, since it was to be the only one. Logs: `gs://$BUCKET/containers/b4c6beb51cd1ea04fe002fb4be95a0ca3d60dd99/` (deleted 2026-11-23 11:11Z)
and locally `outputs/build-logs/b4c6beb/` (git-ignored). Spend guard still **$0.86** (L4); this build VM (about 80
on-demand minutes, roughly $0.35) is outside the ledger until a reconcile. L4 still STOPPED since 08:16Z: pause rule,
delete it and its disk if still stopped at 08:16Z on 2026-11-24.

**STATE AT THE STOP (2026-11-23 ~09:05Z, user at 88% of the 5-hour limit).**
- `harness-v1` = `b4c6beb` + this HANDOFF commit, pushed. Images: the one rebuild of `b4c6beb` was **running** at the
  stop (on-demand CPU build VM `vlar-build` created 08:43Z; `35_build_images.sh` polls from the Mac and deletes the VM
  when it ends; the VM also deletes itself after `BUILD_MAX_HOURS` = 4 h). Log: `outputs/build-logs/rebuild-b4c6beb.log`
  (git-ignored). First check: `gcloud compute instances list` shows no `vlar-build` (else delete it and its disk),
  and `gs://$BUCKET/containers/b4c6beb51cd1ea04fe002fb4be95a0ca3d60dd99/` holds `DONE` (or `FAILED`).
- Stage B: **not queued** unless a line below says so. To queue it (images DONE, from `b4c6beb` exactly: the queue
  script archives HEAD, so `git checkout --detach b4c6beb` first and back after): re-set the L4's metadata
  (`gcloud compute instances add-metadata $VM_NAME --zone $ZONE --metadata vlar-images-url=gs://$BUCKET/containers/b4c6beb51cd1ea04fe002fb4be95a0ca3d60dd99`
  and `--metadata-from-file startup-script=ops/vm/phase0/startup.sh`), then
  `ops/gcp/31_phase0_queue.sh ops/vm/phase0/steps/{010,015,030,040,050,090,200,210,300}-*.sh`, then
  `gcloud compute instances start $VM_NAME --zone $ZONE` (stage B ~$3-4 + disk, approved by the user).
- Spend guard: **$0.86** (L4 only; the six earlier build VMs and this rebuild, about $1.50 before credits, are
  outside the ledger until a `spend reconcile`). L4 `$VM_NAME` STOPPED since 08:16Z on 2026-11-23: pause rule, delete
  it and its disk if still stopped at 08:16Z on 2026-11-24 (then recreate it and queue `020` too).
- The L4's startup-script metadata holds an **older** `startup.sh` (the package freeze, without the failed-boot record
  of amendment 42): it must be re-set from `b4c6beb`'s file before stage B (the queue instructions above do).

**SESSION 2026-11-23 (read first).** Decided and done on `harness-v1` (commit after `01dc2dc`):
- π0.5 padded vs batch of one, diagnosed at max effort on the L4 (step `095`, `ops/vm/phase0/debug/`; HARNESS.md "π0.5
  padded to 8 vs the batch of one"): numerical, not a bug. The user chose option A (amendment 37): both exact checks
  kept; the random-image 0.05 check replaced by 20 real decision points (executed actions within 0.05 everywhere, all
  50 within 0.05 at >= 90%); parity (step `300`) runs the production padded path (confirmed: production servers at
  the calibrated geometry, every π0.5 request seeded, `episode.py` `_pi05_request`).
- Approved: host packages frozen on every boot (amendment 36; the journal confirmed apt-daily-upgrade restarting the
  runner at 04:35:50Z); the step-2 rules with the joint memory measurement and the grasp check with its defaults
  (amendment 38).
- The partner's review, all six points (amendments 39-44): conditional ledger writes with retry and no StopIteration;
  lease-checked `complete`/`fail`/`release` with renewal; exit 124/137 told apart (time cap, OOM, kill); sessions
  from boot time plus failed-boot records; alerts per limit with re-alerts; identifiers replaced by env.sh
  placeholders with `tests/test_no_identifiers.py`. Bucket checked: uniform access, public access prevention
  enforced, no public members.
- Spend guard **$0.86**. VM `$VM_NAME` STOPPED since 08:16Z on 2026-11-23: pause rule, delete it and its disk if still
  stopped at 08:16Z on 2026-11-24. Queue empty. **Its startup-script metadata holds an uncommitted `startup.sh`**:
  re-set it from the committed file (`gcloud compute instances add-metadata $VM_NAME --zone $ZONE
  --metadata-from-file startup-script=ops/vm/phase0/startup.sh`) before stage B.
- Next (xhigh): the one rebuild from this commit (`BUILD_PROVISIONING=STANDARD caffeinate -i -s
  ops/gcp/35_build_images.sh`, about 85 min, about $0.40, cost told to the user first), the metadata re-set, then
  stage B as in "START HERE" below, step 4 (queue `010,015,030,040,050,090,200,210,300`), each cost announced first.

**OVERNIGHT LOG (2026-11-23, superseded by the session above).** Stage A **failed** at its verdict, so per the user's pre-approved plan
nothing more ran: no rebuild, nothing queued, no debugging. The L4 `$VM_NAME` is **STOPPED** (TERMINATED since
05:21:59Z on 2026-11-23); pause rule: delete it and its disk if it is still stopped at 05:22Z on 2026-11-24. Spend
guard: **$0.73** total (VM $0.73, 1.38 VM hours; disk not yet charged in the ledger). Steps: 010 done (451 s),
015 done, 020 done (113 s), 030 done (2117 s; sim-reference 101 passed, 3 skipped), 040 done (172 s), 050 done
(362 s), 090 **failed**: row `pi05-batch-invariance-l4` (logs: `gs://$BUCKET/vla-router/phase0/logs/results/`).
Every other row passed: host unit tests, reference check, sim-reference, sim-numpy-percentile, xvla-vs-authors,
xvla-batch-invariance-l4, pi05-vs-lerobot-eval. The failure, as the row printed it (not debugged): the real π0.5
checkpoint on the L4 reached `check_pi05_reference_path` (so the alone/companions/position and padding-token checks
before it passed); a batch of one equals `act_unbatched` exactly, but the same request padded to 8 rows differs from
it by max |diff| = **1.94** (allowed 0.05): not last-bit noise. **Effort checkpoint (SPEC §16): an unexplained
discrepancy that could corrupt parity and all π0.5 results; debugging needs max effort and the user's word.**
**The 27-minute gap:** not a hang. `runner.log` shows 030 started at 04:08:34Z, then "phase 0 runner started" at
04:35:50Z with no reboot (serial log: one boot) and no new startup-script run (startup.log: two boots, both before):
the runner service was restarted from outside mid-step and 030 re-ran from scratch (27 min, about $0.20). Nothing in
the repo restarts it; the likely cause is Ubuntu's unattended-upgrades with needrestart restarting services after a
package update (unconfirmed: the VM is stopped and its serial buffer is gone). The runner did not cause it, but it
logged nothing on SIGTERM; fixed (`ops/vm/phase0/runner.sh`, committed after `65b5b42`; tests/test_phase0_runner.py 9 passed): it now logs the SIGTERM and an
`interrupted` history entry, and at every start writes `logs/system-<time>.log` (runner/apt journal, apt history,
needrestart lines), so the next boot records what happened at 04:35Z. If confirmed, disable unattended-upgrades on
the VM (a decision for the user).
**Approved by the user 2026-11-23, NOT implemented yet (do before the rebuild; xhigh):** (1) the batch-size rule's
memory limit must be the measured total L4 memory with both models loaded at the candidate size AND the planned
number of workers rendering at the same time, keeping at least 2 GB headroom on the 24 GB card (replace
`MEMORY_BUDGET_GB` in `vlar_ops/step2.choose_geometry` with a joint measurement in step `200`: servers at each
candidate geometry, 2 x B workers rendering, full batches to both servers, peak from `nvidia-smi`); the rest of the
rule stands. (2) Gripper procedure approved; add a check after the base rollouts, before the freeze: report how the
base rollouts' real grasps (steps table: commanded-closed steps' finger widths) fall relative to the thresholds and
flag it if they do not separate cleanly. The overnight go-ahead (rebuild, re-point, queue stage B unattended) applied
only if stage A passed; it does not carry over.

**START HERE (2026-11-22, superseded in part by the log above): stage B of Phase 0, at xhigh.** Stage B's code is
committed on `harness-v1` (`65b5b42`; amendment 35, HARNESS.md "Calibration file and stage B", ops/README.md
"Stage B"). Nothing of stage B is built or queued yet. In order, once the π0.5 failure is resolved with the user:
1. Implement the two approved changes above (the original rule text, for reference): the geometry rule
   (`choose_geometry`: sizes 1-16 whose two models' deterministic peak memory sums to <= 20 GB; the smallest whose
   deterministic GPU cost per request is within 10% of the best; stop with exit 75 if deterministic mode costs
   > 1.25x the non-deterministic run there; workers = 2 x max_batch, at most 12) and the gripper procedure
   (`gripper`: delta mode, no arm motion, open then closed on nothing for 30 steps each, tasks 0, 3, 6 at initial
   state 0; thresholds 10% of the range inside the measured widths). Tests: `tests/test_step2.py`.
2. **Stage A's verdict.** `export PATH=$HOME/google-cloud-sdk/bin:$PATH; source ops/gcp/env.sh`, then
   `gcloud storage cat gs://$BUCKET/vla-router/phase0/STATE.json` and `.../history.jsonl`, or `ops/gcp/32_watch_vm.sh`.
   At 05:05Z on 2026-11-23 it was running step 030 (started 04:35:51Z, cap 3 h; spend $0.39). Tell the user when and
   why the VM stopped. Look in `gs://$BUCKET/vla-router/phase0/logs/runner.log` for the unexplained 27-minute gap
   between 020's end (04:08:25Z) and 030's start (04:35:51Z); there was no reboot. If any re-run row failed: stop,
   report, no debugging without the user (an exact-equality failure is an effort checkpoint, SPEC §16); a fix goes
   into the one rebuild. A stopped VM may stay stopped at most a day (pause rule below).
3. **The one rebuild** (only with the user's go-ahead; about 85 min, about $0.40), from the clean stage B commit:
   `BUILD_PROVISIONING=STANDARD caffeinate -i -s ops/gcp/35_build_images.sh`. It builds `git rev-parse HEAD`, runs the
   in-container tests on the CPU VM and uploads to `gs://$BUCKET/containers/<full commit>/`; it deletes its VM and
   confirms. Then record the commit, image digests and test counts here.
4. **Queue stage B** (only once the L4 is STOPPED, and from the same commit as the images, tree clean):
   ```bash
   gcloud compute instances add-metadata $VM_NAME --zone $ZONE \
     --metadata vlar-images-url=gs://$BUCKET/containers/<full commit>
   ops/gcp/31_phase0_queue.sh ops/vm/phase0/steps/{010,015,030,040,050,090,200,210,300}-*.sh
   gcloud compute instances start $VM_NAME --zone $ZONE && ops/gcp/32_watch_vm.sh
   ```
   (If the VM was deleted under the pause rule: recreate it with `IMAGES_COMMIT=<full commit> ops/gcp/30_create_vm.sh`
   and queue `020-weights.sh` too.) The VM installs the new host source on boot, `010` loads the new images (the old
   ones stay loaded), `030`-`090` re-run the test rows on them (results in `phase0/logs/results/<commit>/`), `200`
   measures step 2 and writes the calibration file (`gs://$BUCKET/vla-router/calibration/`; sub-results in
   `phase0/step2/<commit>/`), `210` re-checks batch invariance at the chosen geometry, `300` runs parity, its gate
   (`phase0/parity/gate.json`) and a manifest check, and on a pass queues `400` (500 base rollouts). Estimated 6-8 L4
   hours, about $3-4 plus disk $0.33/day.
5. **While it runs:** report to the user which steps are safe unattended (all of them run under systemd; the VM stops
   itself on a failure, exit 75 or 76, or an empty queue) and the spend guard's total at each major step
   (`history.jsonl`). Exit 75 at `200` means deterministic fixed batching is too slow: bring
   `phase0/step2/<commit>/geometry.json` to the user (amendment 24). Exit 76 at `300` means parity failed: stop, no
   debugging. The write-once group, once written, never changes; to redo it the calibration file would have to be
   discarded before any record exists, which only the user decides.
6. **After the base rollouts (not scripted yet):** the relative handoff check (HARNESS.md re-run list, "after Phase 0
   step 5"), step 6 (`python -m vlar_harness.calibration --base ... --parity ... --write` in the sim image with the
   calibration directory mounted read-write and `VLAR_CALIBRATION` set), step 7 (golden set, `provenance.write_freeze`,
   the golden-set self-check the worker requires before it runs a frozen harness).

**Created in Google Cloud (Phase 0 stage 1, 2026-11-22, approved by the user; nothing that bills by the hour):**
- APIs enabled: compute, storage, iam, logging, monitoring, billingbudgets.
- Bucket `gs://$BUCKET` (us-central1, Standard, public access prevention enforced, uniform
  access, soft delete off). Empty; the queue and data go under `vla-router/` (`ops/ops.yaml`).
- Service account `$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com`: `roles/storage.objectAdmin` on the
  bucket, `roles/logging.logWriter` on the project. (The first run of `10_bucket_and_account.sh` failed at the
  bucket binding because the new account had not reached IAM yet; the two bindings were applied by hand and the
  script now retries for up to two minutes.)
- Budget `vla-router-50usd`: $50, alerts at 50/80/100%, credits excluded,
  one custom period 2026-11-01 to 2027-08-20 (created as a calendar-month budget, which would reset monthly; changed
  in place, and `20_budget_alerts.sh` now sets the period from `BUDGET_START`/`BUDGET_END`), alerts also sent to the
  email channel below (`40_spend_alert_policy.sh` now attaches it).
- A monitoring email channel (the alert email in `env.sh`) and an alert policy
  (the spend guard's soft and hard limit log events).

**Phase 0 stage 2, image build (2026-11-22, approved by the user).** First attempt on `0046252` (build script
hardened against network drops on the Mac: three-way checks, deletion only after DONE/FAILED is read, the cap, or a
confirmed-gone VM, deletion confirmed; `VLAR_IN_IMAGE=1` marker with in-image-only skips for repo-only tests;
`tests/reference/test_vm_worker.py` renamed `test_vm_worker_sim.py` after a base-name clash stopped `pytest tests`
in the sim environment; image-layout dry run 267 passed / 27 skipped): FAILED about 2 minutes in at `docker build
sim`, stage `sdist-wheels`: egl-probe's CMakeLists.txt (`cmake_minimum_required(VERSION 2.8.12)`) does not configure
with the locked CMake 4.4.4. VM and disk deleted and confirmed gone; cost cents; `containers/0046252.../` holds only
`src.tar.gz`, `build.log`, `FAILED` (1 MB).

Fix (amendment 31, the user's direction): build the sdists as the reference notebook did: `cmake>=3.29,<4.2` in
`containers/build.in` (locked 4.1.3; `lock.sh` re-run with `PY=.venv-sim/bin/python`, which has certifi: `.venv`'s
Python cannot verify PyPI's certificate and its failed run empties `lerobot.sdists`), `CMAKE_POLICY_VERSION_MINIMUM=3.5`
in the `sdist-wheels` stage, `--no-build-isolation` (already); SPEC §2 provenance note and §17 amendment 31; guard
test `test_sdists_build_as_the_reference_notebook_did`. Local check on the Mac (arm64, CPython 3.13, locked build
tools by version, sdists hash-checked, no pip cache): all five sdists build (antlr4-python3-runtime, bddl, egl-probe,
hf-egl-probe, robomimic); egl-probe and hf-egl-probe configure and compile. Without the policy setting both CMake
4.1.3 and 4.4.4 fail with the VM's error; with it 4.4.4 also configures, so the setting is the fix and the pin
matches the notebook. egl-probe bundles its EGL/KHR/X11 headers and links only dl and pthread (build-essential).

Second attempt on `6b3fccd` (this fix): the Spot VM was **preempted by Google** about 2 minutes after creation
(audit log: `compute.instances.preempted`, 14:38:31Z), before the build uploaded anything; termination action DELETE
removed it and its disk; the script reported "gone without a result" and confirmed both gone (also checked by hand:
no instances or disks in any zone). Not a build failure. `containers/6b3fccd.../` holds only `src.tar.gz`.
The user chose on-demand for the build (a preempted Spot run costs more in time and usage than the ~$0.25
difference): `BUILD_PROVISIONING` option in script 35 (`SPOT` default, `STANDARD` on-demand; same cap, polling and
confirmed deletion; tests in `tests/test_build_script.py`). Third attempt: on-demand on `8b6e3a9`, 07:44-07:54 local: **FAILED at the sim image's last step**, after the
sdist builds (the CMake fix works), the LeRobot environment and the LIBERO setup had all succeeded:
`check_env.py --role sim` reported `thop: locked 0.1.1.post2209072238, installed 0.1.1-2209072238`, its only
mismatch. Both strings are the same version under PEP 440 normalization (the hash-checked locked wheel was installed;
its metadata spells the version differently); `check_env.py` compares the strings literally (`version_mismatches`,
around line 51). VM and disk deleted and confirmed gone (also checked by hand); cost about 10 on-demand minutes.
`containers/8b6e3a9.../` holds `src.tar.gz`, `build.log`, `FAILED`.

Fix (approved by the user): `check_env.same_version` compares with packaging's `Version` equality (PEP 440), strict
otherwise (other releases, post-releases and local versions differ; unparsable versions must match exactly);
packaging is in all three locks. Scan of all three locks: every pin is already written in normalized form (250 of
250), so the lock text cannot show this; the locked wheels' own core metadata (PyPI per-file metadata matched by the
lock's hashes, and the locally built sdist wheels) spells only thop differently. Tests: `METADATA_SPELLINGS` in
`tests/test_containers.py` (the thop case, checked against the lock pin) and the strictness cases; text comparison
and always-equal are both caught. Fourth attempt: on-demand on `e1f10ee`, 14:36-15:26 local: **all three images built and passed `check_env.py`;
the in-container tests failed** (`FAILED: in-container tests`); VM and disk deleted and confirmed gone (also checked
by hand); about 50 on-demand minutes. Logs in `containers/e1f10ee.../` (`build.log`, `test-{sim,pi05,xvla}.log`).
- pi05: 8 passed, 1 skipped.
- sim: 193 passed, 32 skipped, 10 failed, 73 errors, all 83 from one cause: LIBERO opens its scene XML from the
  package directory, `/opt/venv/lib/python3.13/site-packages/libero/libero/assets/scenes/libero_{living_room,kitchen}
  _tabletop_base_style.xml` (FileNotFoundError), not from the assets the image installs at `/opt/libero-assets`
  (`LIBERO_CONFIG_PATH=/opt/libero-home`, written by `containers/libero_config.py`; `check_env.py` verified the
  assets' tree digest at build time).
- xvla: 11 passed, 1 failed: `test_xvla_act_matches_authors_handler`: "The starlette.testclient module requires the
  httpx2 package to be installed" (not in `xvla.lock`).
Waiting for the user before debugging (xhigh).

Fixes (approved by the user, at xhigh): (1) LIBERO assets, amendment 32: hf-libero's `get_assets_path()` ignores
config.yaml and takes `<package>/libero/assets`, else its download cache `~/.cache/libero/assets` (the reference run's
path; the Mac uses the package-relative symlink); the images now put the pinned snapshot at
`/root/.cache/libero/assets` (`HOME=/root`), and `check_env.py` (sim, pi05) checks the resolved directory, its digest
and builds all ten LIBERO-Long scenes at build time (OSMesa, no rendering; about 5 s on the Mac);
`tests/reference/test_libero_assets.py`. (2) X-VLA test dependency: the locked starlette 1.10.0's metadata lists
httpx2>=2.0.0 (and httpx<0.29 as a deprecated fallback) in its `full` extra; `containers/xvla-test.lock` (httpx2,
httpcore2, truststore) is installed with hashes only into the throwaway X-VLA test container
(`build_vm_startup.sh`; the L4 re-run rows say the same). Fifth attempt: on-demand on `f93c714`, 16:00-17:26 local (about 86 minutes): images built; the build-time checks
passed in all three (sim and pi05 now include LIBERO's resolved assets directory, digest and the ten scenes); **the
in-container tests failed on bit-level batch invariance on the CPU**; VM and disk deleted and confirmed gone (also
checked by hand). Logs in `containers/f93c714.../`. The LIBERO and httpx2 fixes work (sim 281 passed, including
`test_libero_assets.py`; `test_xvla_act_matches_authors_handler` passes).
- sim and pi05 (the same three tests in each): `test_pi05_act_is_batch_invariant`,
  `test_pi05_act_is_batch_invariant_in_bfloat16`, `test_pi05_simultaneous_requests_keep_their_own_prompts_and_outputs`
  ("requests [0, 1, 2, 4] got another request's output": batched bytes differ from alone).
- xvla (5): `test_xvla_every_position_computes_the_same_bits`, `..._captured_concurrent_batches_replay_to_the_alone_results`,
  `test_xvla_act_is_batch_invariant`, `test_xvla_sample_is_batch_invariant_and_features_agree_across_endpoints`,
  `test_xvla_simultaneous_requests_keep_their_own_prompts_and_outputs`.
**Unexplained:** the same tests passed in the fourth run (`e1f10ee`: pi05 8 passed; xvla all but the httpx2 test), and
nothing in `src/` or these test files changed between the two commits, so the outcome depends on the build VM (e2 can
land on different Intel/AMD CPU platforms with different SIMD kernels; the build log does not record the CPU) or on
nondeterminism. This is an exact-equality discrepancy (SPEC §16): effort checkpoint raised, waiting for max effort
before debugging. Open question for the user: whether CPU batch invariance on x86_64 must hold for the image build,
or these checks belong to the L4 (amendment 24 requires them bit for bit on the L4).

Decision (user, 2026-11-22; amendment 33): exact CPU batch invariance on x86_64 does not block the image build; the
L4's bit-for-bit checks stay blocking. Size of run 5's differences: the logs show only the first differing values,
1-5 units in the last place of float32 (tiny), not the maximum; the tests now print max absolute and relative
difference on any failure. In an image on the CPU the blocking checks are torch's default tolerance (float32 rtol
1.3e-6, atol 1e-5; bfloat16 rtol 1.6e-2) and closeness to the request's own solo result; bit inequality is a
warning. Build logs start with `lscpu`.

**Sixth attempt: built and tested (DONE).** On-demand on `be7b95a`, 18:20-19:46 local (about 85 minutes); VM and
disk deleted and confirmed gone (also checked by hand). Build VM CPU: AMD EPYC 7B12 (Rome), 8 vCPUs, AVX2 and FMA, no
AVX-512; PyTorch's CPU capability AVX2 in all three images. Tests: sim 293 passed, 33 skipped; pi05 8 passed,
1 skipped; xvla 12 passed; **no NotBitIdentical warnings: every batch-invariance check was bit-exact on this CPU**,
as in run 4. Run 5's CPU was not recorded, so the maximum size of its differences is still unmeasured (only its
first differing values, 1-5 ulp, are known); a later build on another CPU records them as warnings.
Images in `gs://$BUCKET/containers/be7b95a5d944d351428b5d79978864dc62ad4d1a/` (deleted 2026-11-23 14:20Z, superseded by `03f311c`; logs kept locally): vlar-sim 3994 MB,
vlar-pi05 3963 MB, vlar-xvla 2998 MB (about 11 GB, about $0.22/month in Standard storage), image-digests.json
(sim sha256:3dfe75be..., pi05 sha256:fceeb4b8..., xvla sha256:94508c13...), SHA256SUMS, logs. The earlier attempts'
prefixes hold only sources and logs (a few MB). Six build attempts in all cost roughly $1.10 before credits.

**Next: `30_create_vm.sh` (the L4, $0.46/h + $0.33/day disk) only with the user's go-ahead**, then the VM setup in
ops/README.md starting with `ops/vm/load_images.sh` for commit `be7b95a` (driver check, load, GPU checks), then the
VM re-run list (HARNESS.md) at **xhigh** (max only for parity debugging). The L4's batch-invariance checks are bit
for bit and blocking (amendment 24); its X-VLA test rows install `containers/xvla-test.lock` in a throwaway container.
**Go-ahead given (user, 2026-11-22) for `30_create_vm.sh` and Phase 0 on the L4, at xhigh**, with these rules: every
long job runs on the VM under systemd (never tied to an SSH session from the Mac); π0.5 parity gates the base rollouts
automatically (McNemar p >= 0.05 in `reference_runs.compare`, i.e. `within_sampling_error`: queue the X-VLA base
rollouts without waiting; otherwise stop, don't debug); whenever the VM runs out of queued work or needs the user, it
syncs everything to the bucket and stops itself (stopped, not deleted), and the user is told when and why; report
which steps are safe unattended and the spend guard's running total at each major step. The pause rule stands: a
pause of more than a day means deleting the VM and its disk. The failed build attempts' folders under `containers/` in the
bucket were deleted at the user's request (only `be7b95a...` kept); their logs are kept locally in the git-ignored
`outputs/build-logs/<commit>/` (run 5's are the only record of its CPU differences).

**Phase 0 step runner (2026-11-22, ops/README.md "Phase 0 on the L4").** `ops/vm/phase0/runner.sh` as
`vlar-phase0.service`, started on every boot by the startup script `ops/vm/phase0/startup.sh` (installs the host side
from `phase0/src/CURRENT` in the bucket, spend guard first, disk recorded in the ledger once); steps queued in the
bucket by `ops/gcp/31_phase0_queue.sh`; `ops/gcp/32_watch_vm.sh` follows `STATE.json` from the Mac. Stage A steps
(`ops/vm/phase0/steps/010-090`): image load, host checks, weights and tokenizer, the VM re-run rows, the verdict.
Tests: `tests/test_phase0_runner.py` (fake gcloud/systemctl/vlar-ops/timeout; order, queueing from a step,
needs-input stays queued, gate failure, failure and time cap, STOP flag, SIGTERM).

**Stage B written (2026-11-22, at xhigh; committed and pushed on `harness-v1`, not built, nothing queued).** The user decided amendment
34's four exceptions (amendment 35, SPEC §17): a write-once group in the calibration file (π0.5 latency, upload size,
batch geometry), the record's calibration hash, the freeze pinning the final file plus its lineage; `sim_time_s`
dropped; gripper fractions offline, `observe()` refusing until frozen; the batch geometry in the write-once group,
not the images (no reason found against it: the servers already take it as start-up flags and the workers check it).
Implemented (HARNESS.md "Calibration file and stage B"): `config.py` (file via `VLAR_CALIBRATION`, write-once
writer, lineage; config hash without calibration; `harness_version` 1.0.0, `frozen` out of the base config),
`provenance.py` (freeze record and checks), `calfile.py` (VM CLI), refusals in `episode.py`, `measure.py`
(`gripper_fractions_of_point`), `dataset.py`/`collect.py` (calibration hash in runs, no `sim_time_s`, no fractions),
`manifest.py` (`--freeze` lineage check), `vm_worker.py` (refuses without the write-once group),
`servers/bench.py`, `vlar_ops/step2.py`, `vlar_ops/parity.py`, `vlar-ops plan|parity|step2`, units, `lib.sh`,
`startup.sh` (restores the calibration from the bucket), `load_images.sh`, steps `200`, `210`, `300`, `400`.
Tests (Mac): see the stage B line in the status table. **Waiting for the user:** approval of the two proposed rules
(geometry rule and gripper procedure, HARNESS.md), then the one rebuild (`35_build_images.sh`, on-demand,
about 85 min, about $0.40) once stage A has its verdict, then the queue commands in ops/README.md "Stage B".
Estimated stage B on the L4: about 5-8 VM hours, about $2.50-3.70, plus disk $0.33/day.
**Stage A at 05:05Z:** still running step 030 (sim re-run rows) since 04:35:51Z (cap 3 h), VM up, spend $0.39.
Unexplained: 020 finished at 04:08:25Z but 030's start was written at 04:35:51Z; there was no reboot (serial log:
one boot, 03:57:53Z). Check `logs/runner.log` for the gap when 030 ends (`pull_queue` or `vlar-ops spend status`
hanging are the candidates).
**Tokenizer (2026-11-22):** the user logged in to Hugging Face; revision `35e4f46` (the then-latest; the reference
run's is unrecorded) with SHA-256 in SPEC §2, cached at `gs://$BUCKET/vla-router/hf-tokenizer/` (flattened, with
`refs/main`); stage A's step 050 fetches it. Note for downloads: the new `hf` CLI needs one `--include` per pattern and
`hf_xet` for these files. **Stage A** was running at handoff (step 030; spend guard $0.15); its first boot stopped at
010 because the VM image has no Docker (fixed in `6690b19`: docker.io + `nvidia-ctk runtime configure`). Check
`gs://$BUCKET/vla-router/phase0/STATE.json` and `history.jsonl` for where it ended; a stopped VM stays within the pause
rule for one day.

**L4 VM created (2026-11-22, user's go-ahead):** `$VM_NAME` (us-central1-a, Spot, 100 GB pd-balanced) with stage A
queued (host source `94b1966`, images `be7b95a`); it stops itself when stage A ends (`32_watch_vm.sh` follows it).
Spend guard ledger starts with this VM ($0 before it; the six build VMs, about $1.10 before credits, are outside the
ledger until the weekly reconcile). Pause rule: if the VM stays stopped for more than a day, delete it and its disk
(`gcloud compute instances delete $VM_NAME --zone $ZONE --delete-disks=all`, then
`vlar-ops spend disk-deleted --name ${VM_NAME}-boot`).

**Milestone (2026-11-22):** all seven §18 items hold; the checkpoint was announced and the user's explicit go-ahead
was given for stage 1 (scripts 10, 20, 40, done). Then Phase 0 at **xhigh**, in this order, each script
approved by the user after seeing what it creates and costs: `10_bucket_and_account.sh`, `20_budget_alerts.sh`,
`40_spend_alert_policy.sh`, `35_build_images.sh` (CPU VM), then `30_create_vm.sh` and the VM setup in ops/README.md
(`load_images.sh` first). Commands need `export PATH=$HOME/google-cloud-sdk/bin:$PATH` and `source ops/gcp/env.sh`.

Done 2026-11-22: the VM worker entry point and the dataset manifest and datacard (`1e92964`, §18 item 4); the
container definitions (§18 item 5 as amended by 28-30, the commit after `1e92964`). Next, at **medium**: §18 item 6, the Google
Cloud values in `ops/gcp/env.sh` once the account has billing and the GPU quota (the CPU build VM needs billing
only), then item 7 (everything committed, clean tree), then the milestone check. Container notes for whoever runs
Phase 0's first step:
- `download.pytorch.org` was unreachable from the Mac's network; the locks take PyPI's torch 2.11.0, which is the
  CUDA 13.0 build (it depends on nvidia-cuda-runtime 13.0.96). `check_env.py` asserts CUDA 13.0 in each image.
- CPython 3.13.16 (amendment 30): the notebook's runtime check printed it; standalone build 20261020, hash-pinned.
- Unverified until the build: Ubuntu's GCE image carrying `gcloud` (the startup script fails clearly if not), the
  five sdist builds (egl-probe needs cmake and a compiler, both provided), the shallow fetch of the X-VLA commit,
  and the π0.5 CPU test running offline with config-only snapshots.
- `.venv-tools` (git-ignored) holds uv 0.13.9 for `containers/lock.sh`.

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
- VM worker (`vlar-ops worker`): refuses a frozen config until the golden-set self-check (§15) is written with the
  freeze (Phase 0 step 7); the self-check belongs before the workers in the runbook's "Every start". Run parity
  workers with `--phase pi05_parity --option scene_seed=pi05_parity`; any worker given a job its options cannot run
  gives it back (no attempt spent) and exits 3. Workers are not enabled at boot (resume first).
- Manifest: `vlar-ops dataset manifest --root <local copy>` needs pyarrow (simulator env). It refuses unlisted data
  files (use `--move-unlisted DIR`), stray files, schema or marker mismatches and mixed harness code; a dirty harness
  is only a warning. The datacard (`src/vlar_ops/datacard-v1.md`) is outside the harness source digest; a test keeps
  its field tables equal to the Parquet schemas.
- Amendments 28-30 (2026-11-22): images built on a CPU-only VM and stored as files in the bucket; versions from
  the notebook's install output (may be from a later re-run), version mismatch first on the parity-debugging
  checklist (§11 step 4), driver 580.65.06+ for CUDA 13.0; CPython 3.13.16 in every image.
- Housekeeping candidate (medium): `vlar_ops.queue.run_worker` (groundwork, handler-based, used only by
  `tests/test_queue_spend_packaging.py`) duplicates `vlar_ops.worker.run_worker`.
- §12: the sim renders only the step before each decision point by default. Anything that reads images
  elsewhere gets `ImagesNotRendered`; a snapshot taken between decision points holds placeholders
  (`sim.images` false). Use `render="all"` (`load_config(render="all")`) for per-step image work.

Toward the §18 milestone, in order:
1. Done: render only the observations that are used (§12); decider inputs, measurement recording, dataset records,
   runner and worker loop (§18 item 3); the VM worker entry point, dataset manifest and datacard (item 4, Mac).
2. (done) Decider inputs and measurement recording (§18 item 3): `observe()` (features, proposal, free signals, spread and
   drift, the K-sample probes), the dataset format (Parquet), and the queue workers that run episodes and branches
   through `run_branch` (item 4: the engine is not wired to the queue yet; the Cloud Storage backend is untested).
3. The three container images and lockfiles (item 5).
4. The GCP values in `ops/gcp/env.sh` once the account and GPU quota are ready (item 6; scripts written, never run).
5. The VM re-run list completed with the containers, everything committed, a clean tree (item 7); then the milestone.
None of these is on the §16 max list as such; debugging an exact-equality or replay failure in step 1 is.
