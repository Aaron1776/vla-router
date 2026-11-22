# Project status

Last updated: 2026-11-21, describing the code on branch `harness-v1` after part 1 of effort step 5 (`f8d06d5`); later
commits change only documents.

This page is for someone who has seen nothing of the project so far: what it is, where things are, how far the
work has got, what comes next and how to run it. It is a summary; where it and [SPEC.md](SPEC.md) disagree, the spec
wins.

## What the project is

A robot arm in simulation must finish long household tasks from a written instruction ("put both moka pots on the
stove"). It is controlled by a **VLA** (vision-language-action model: a neural network that takes camera images, the
robot's joint state and the instruction, and outputs the next arm movements). The tasks are the ten tasks of
**LIBERO-Long** (`libero_10`), a standard simulated benchmark, each with 50 fixed starting arrangements ("initial
states"), so 500 episodes cover the benchmark. An **episode** is one attempt at one task from one initial state,
at most 520 control steps (20 steps per simulated second).

Two VLAs are used. **X-VLA** (0.9 billion parameters) is small enough to run on the robot's own GPU and drives by
default; it succeeded on 405 of the 500 episodes in our reference run. **π0.5** (a 3-billion-parameter
vision-language backbone plus an action module) is much stronger (478 of 500) but too large for the robot, so it sits
on a "remote" server behind a simulated network.
Every 10 steps, at a **decision point**, a small learned model called the **decider** chooses between continuing with
X-VLA and **escalating**: sending the current observation to π0.5 and executing its next 10 actions. Escalation is
limited three ways: a **budget** of at most 260 π0.5 steps per episode (26 escalations), a per-task **time limit**
on simulated wall-clock time, and the **simulated network**, which adds upload time, a random round trip and
occasional failures to every π0.5 request. The goal is to get close to π0.5's success rate while using π0.5 as
little as possible.

Two kinds of decider are trained on identical data and compared. A **router** decides from the observation alone
(features from X-VLA's image and text encoder, the robot state, the remaining budget and time). A **cascader** also
sees what X-VLA is about to do (its next 10 proposed actions) and signals derived from it, such as how much several
sampled X-VLA plans disagree ("spread"). The headline cascader pays for those extra samples in simulated time; two
ablations remove or reuse them. The question is whether seeing X-VLA's plan and confidence makes a better escalation
decision once its cost is counted.

The work has four phases:
- **Phase 0, build and calibrate the harness.** The **harness** is the simulator, both models, the decision loop,
  the clock and the network model. Phase 0 checks that each model reproduces its own published harness's results
  ("parity") and that runs are exactly reproducible, measures the real model latencies on the cloud GPU, runs X-VLA
  alone on all 500 episodes ("base rollouts"), sets the time limits from them, and then freezes the harness as v1.0.
- **Phase 1, collect labelled data.** From snapshots of many decision points, short scripted continuations
  ("branches": keep X-VLA, one π0.5 call, three calls, or as many as the budget allows) are run several times each,
  and the share that succeed becomes the training label for that decision point.
- **Phase 2, train the deciders.** Small networks (under 5 million parameters) trained with leave-one-task-out
  cross-validation, so every decider is tested on a task it never saw.
- **Phase 3, one DAgger round.** The trained deciders drive episodes themselves, the new states they reach are
  labelled the same way and added to training, and everything is retrained. Then the final evaluation.

Deliverables: the repository (harness, configs, training and evaluation scripts, packaging scripts), a short results
report with the comparisons and sample videos, and two sets of zip archives delivered to the team's own computer:
`dataset-v1` (every episode, snapshot, feature and label) and `harness-v1.0` (the three container images and the
model weights), so the study can be re-analysed or extended without the cloud. Compute is one Google Cloud
g2-standard-16 Spot VM with one NVIDIA L4 GPU, within a $50 budget.

## Where things live

Documents (in `docs/`):
- [SPEC.md](SPEC.md): the design and the source of truth. §17 logs every approved change with its date; §18 defines
  the milestone that ends the harness build.
- [HANDOFF.md](HANDOFF.md): the working state for the next work session: status of every step, decisions, the exact
  next step with design notes.
- [HARNESS.md](HARNESS.md): how the harness implements the spec, what was learned from the reference code, which test
  shows each claim, and the list of tests to re-run on the cloud VM.
- [STATUS.md](STATUS.md): this page.
- [ASSUMPTIONS.md](ASSUMPTIONS.md): every claim labelled verified, stated in a source, assumed or unknown, with evidence.
- [PLAN.md](PLAN.md) and [CONCEPT.md](CONCEPT.md): the original plan and concept document (an earlier design with a
  different local model); superseded by SPEC.md where they differ.

Code and data:
- `src/vlar_harness/`: the harness. `sim/` (LIBERO simulator layer), `adapters/` (each model's preprocessing,
  reused from the reference code), `servers/` (the X-VLA and π0.5 model servers), `episode.py` (decision loop,
  handoffs, budget, event log, time limit, branches), `network.py`, `clock.py`, `labels.py`, `calibration.py`,
  `snapshot.py`, `seeding.py`, `config.py` with `configs/harness-v1.yaml`, and `api.py` (the interface deciders use).
- `src/vlar_ops/`: running at scale: job planner, resumable job queue, spend guard, archive packaging, the reference
  run checks, and the `vlar-ops` command; `configs/collection-v1.yaml` holds data-collection choices.
- `ops/`: Google Cloud setup scripts (`ops/gcp/`, never run yet), VM services and the runbook (`ops/README.md`).
- `tests/`: unit tests that need only Python; `tests/reference/`: simulator and model tests against the reference
  code (need the local simulator and model environments).
- `References/`: the per-episode results of the reference run that parity is checked against. Read-only.
- `envs/`: how an earlier server machine was set up (from the first design).

## Current state

Done, on branch `harness-v1` (all of it tested **on a Mac only**; nothing has run on the cloud L4 VM yet, and only VM
results count for parity and determinism, because rendering and floating-point results can differ by platform):

| Commit | What |
| --- | --- |
| `4960782` | Groundwork (versioned config, job planner and queue, spend guard, packaging, cloud scripts) and effort step 1: the simulator layer, both model adapters and servers, exact-equality tests against each model's own reference code |
| `5d92d97` | Effort step 2: exact snapshot capture and restore of a decision point; determinism tests |
| `25794f7` | Effort step 3: the episode engine (decision points, handoff rules between the models, budget, run modes); the scene/rollout seed split; the reference run's result files and checks |
| `b3e93da` | Decisions 9-12 recorded, the milestone rule, the first HANDOFF.md |
| `272f64b` | Effort step 4: network model, event log, simulated clock under every overhead profile with offline re-scoring, live time limit, branch runs, label computation, time-limit calibration, handoff diagnostics |
| `a25053f` | Step-4 follow-ups (amendments 19-23): a live time limit only in routed runs with a decider profile, recorded as an explicit flag in every log and result; a time-cut run scored only under its own profile, while labels from a cut parent's log up to a decision point are computed for every profile; decision points already past the limit under a profile left out of that profile's data, counted per profile; the remaining-time input recomputed under each decider's own profile |
| `f8d06d5` | Effort step 5, part 1: a request queue that batches concurrent requests; X-VLA's three endpoints run every batch at one fixed padded size with deterministic settings and per-request noise, so a request's result is bit-identical alone, with other requests or at any position (checked on the CPU) |

Test results after step 5 part 1 (Mac): unit tests 112 passed, 7 skipped; simulator tests 48 passed, 11 skipped (the 5
new X-VLA batch tests skip there); X-VLA server and batch-invariance tests 7 passed (CPU, random weights). The skipped tests need model weights or both servers running, so they run on the VM.

Not started: request batching in the servers (effort step 5), rendering only the camera images that are used, the
decider inputs and measurement recording, the dataset format and queue workers, the container images, and
everything on the VM.

The §18 milestone (all seven must hold before the harness build counts as complete):

| §18 item | State |
| --- | --- |
| 1. Effort steps 4 and 5 done, tests passing on the Mac | partial: step 4 done (`272f64b`, follow-ups `a25053f`); step 5 in progress (X-VLA batching done, π0.5's next) |
| 2. Both servers implement every endpoint, with batching and per-call seeds | partial: all endpoints and per-call seeds exist; X-VLA batched, π0.5 batching missing |
| 3. Measurement recording and the snapshot store | partial: snapshot store done; measurement recording (proposal, spread, drift, probes) not started |
| 4. Job queue, spend guard and packaging implemented and tested | partial: implemented and unit-tested against a local stand-in for cloud storage; the real cloud storage path is untested; the engine is not yet wired to the queue |
| 5. Three container images and lockfiles build and pass local tests | not started |
| 6. Ready-to-run Google Cloud setup scripts (Spot VM, bucket, budget alerts) | partial: written, never run; account values are placeholders |
| 7. VM re-run list complete, everything committed, HANDOFF.md current | partial: the list covers steps 1-4; parity tooling and reference files committed |

By phase: Phase 0 is in progress (the build, on the Mac; none of its VM steps have started). Phases 1 to 3 and the
evaluation have not started.

## What's next, in order

1. The rest of effort step 5: batch-invariant batching in the π0.5 server, as already done for X-VLA (amendment 24:
   a request's output must be bit-identical whatever else shares its batch, so every batch runs at one fixed padded
   size with deterministic GPU settings and per-request noise generators); checked bit for bit on the L4, where the
   cost of the padding is also measured.
2. Render camera images only when they are used, provided it passes the same exact-equality tests.
3. Decider inputs (`observe()`) and measurement recording at every decision point; the dataset format (Parquet);
   queue workers that run episodes and branches.
4. The three container images with lockfiles.
5. Complete the VM re-run list and announce the §18 milestone, then wait for an explicit go-ahead.
6. Only then: create the Google Cloud bucket, budget alerts and Spot VM, and run Phase 0 on the VM: measure latencies,
   batching, gripper thresholds and upload size; re-run every equality and determinism test; π0.5 parity (100
   episodes); X-VLA base rollouts (500 episodes); compute the time limits; choose five "golden" snapshots replayed at
   every start-up; freeze the harness as v1.0.

How the work is run. Each work session (so far, a Claude Code session) starts by reading HANDOFF.md and ends by
updating it. The hardest steps (listed in SPEC §16) are done at the AI model's highest reasoning effort: the session
stops and asks the user to switch before starting one, and says when the user can switch back. Commits happen only
when the user asks, with test results in the message. Any spec change needs the user's approval and a dated §17
entry. Nothing that spends cloud money happens without the user's explicit go-ahead, and the harness's own spend
guard stops all work at $45.

## Key decisions so far

The full list, with dates, is SPEC §17 (24 entries). The most important:
- **Two seeds per episode.** The scene seed fixes the scene (some LIBERO fixtures move by up to 19 mm with it) and
  is 142 + k for initial state k in every run except π0.5's parity check (1000 + k, its own reference's value). A
  separate rollout seed drives all other randomness (model noise, the network, random exploration), different for
  each kind of run, so methods face identical scenes and can be compared episode by episode.
- **Time limits.** Per task: the 90th percentile of X-VLA's successful completion times (base rollouts) times one
  margin m, the smallest of 1.25, 1.5, 1.75, 2.0 that lets at least 80% of π0.5's successful parity episodes finish
  in time.
- **Network retries.** A failed request is noticed at the 1 s timeout and retried after a 0.2 s backoff. A retry is
  the same request, so it never changes what the robot does, only the clock. Routed runs give up after 3 attempts and
  hand the segment back to X-VLA; π0.5 alone retries up to 20 attempts. Each attempt's randomness depends only on
  the rollout seed, the step and the attempt number.
- **Overhead profiles.** Each decider type pays simulated time for what it needs at a decision point: the router an
  encoder-only X-VLA call between X-VLA's action chunks (at a chunk's start the features come free with X-VLA's own
  call), the cascaders extra X-VLA samples. Nothing is charged once the budget is spent, since no decision is left to
  make. Data is collected once and scored under every profile afterwards, exactly.
- **Live time limits.** Only runs driven by a live learned decider (DAgger rollouts, evaluation) are cut at the time
  limit, under that decider's profile; every log records whether its run was cut. A cut run's outcome is scored only
  under its own profile. Labels and the "remaining time" input at a training point are recomputed from the event
  log under each decider type's own profile, and a point already past the limit under a profile is left out of that
  profile's data.
- **Batch invariance.** The model servers batch requests from many simulator workers, but a request's result must
  not depend on its batch: otherwise exact replay of snapshots and runs would break whenever batching differed.
- **Leave-one-task-out folds.** Ten folds: task i is the test task, task i+1 (mod 10) the validation task, the other
  eight the training tasks; splits are always by whole task.

## Open questions and blockers

- **Clock charges under fixed padding (needs a decision).** The servers now always run padded batches, so the
  simulated clock's per-call charges (meant to be what a robot serving one request at a time would wait) should be
  measured in Phase 0 as a true batch of one, unpadded. Recommended in HANDOFF.md; not yet confirmed.

- **Cloud GPU quota (pending).** Google Cloud needs a GPU quota of at least 1 globally, 1 preemptible L4 and 1
  on-demand L4 in us-central1, possibly after upgrading from the free trial. Requested by Miguel; not confirmed yet.
  It blocks only the VM work, which waits for the §18 milestone anyway.
- **Account values.** The cloud scripts read project, billing, bucket, image family and alert email from a
  git-ignored `ops/gcp/env.sh` copied from `ops/gcp/env.example.sh`; they are not filled in yet.
- **Gated model files.** π0.5 needs the PaliGemma tokenizer, which requires accepting Google's terms on a Hugging
  Face account; the VM needs that account's access.
- **Placeholders until Phase 0.** Model latencies, batching settings, gripper thresholds, the upload size (a Mac
  estimate of 20,290 bytes) and the time limits are placeholders marked as such in `src/vlar_harness/configs/harness-v1.yaml`.

## Running the tests

The three local Python environments are not in git and there is no one-command setup yet; the container images and
lockfiles (§18 item 5) will provide it.
- Unit tests need only Python 3.10+ with PyYAML and pytest: `python3 -m venv .venv`, then
  `.venv/bin/pip install -e '.[dev]'` and `.venv/bin/python -m pytest`.
- Simulator tests: `.venv-sim/bin/python -m pytest tests/reference` (about 6 minutes). The environment holds LIBERO
  (hf-libero 0.1.4), robosuite 1.4.0, MuJoCo 3.8.1, LeRobot 0.6.1 and CPU PyTorch, and `third_party/` holds the pinned
  X-VLA and LeRobot code, LIBERO's scene assets and configuration (details in HANDOFF.md).
- X-VLA server tests (random weights, CPU): `VLAR_XVLA_CONFIG_DIR=third_party/hf/X-VLA-Pt VLAR_TEST_DEVICE=cpu
  .venv-xvla/bin/python -m pytest tests/reference/test_model_servers.py -k xvla`.
- On this Mac, do not run commands through `timeout`: it is an x86_64 binary and breaks the arm64 Python packages.

VM only: the server tests with the real weights (X-VLA with its LoRA adapter; π0.5, which also needs the gated
tokenizer), the handoff test with both real models (it also needs the base rollouts), the upload-size measurement,
and the re-run of every equality and determinism test; the list is in HARNESS.md under "Platform rule".

## Ground rules

- SPEC.md is the source of truth. Every change to it is approved first and logged in §17 with its date.
- After Phase 0 the harness is frozen as v1.0. Any later change to harness code makes a new version and means
  re-running the Phase 0 checks; data from different versions are never mixed.
- Never modify the files in `References/`: they are the reference run's results, pinned by SHA-256.
- No secrets in the repository (tokens, keys, `.env` files, credentials), and no hostnames, usernames or network
  addresses of the team's machines.
- Work on branches, never directly on `main`.
- Results from the Mac show only that the code paths are right; parity and determinism count only on the VM.
