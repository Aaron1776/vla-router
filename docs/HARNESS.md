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
The π0.5 request goes through a `RemoteLink` (`DirectLink` until the network model of effort step 4); the
engine emits provisional events that step 4 turns into the event log and clock.

Seeds (SPEC §3): `seeding.scene_seed` (142 + k, or 1000 + k under the `pi05_parity` option) feeds only
`env.seed`; every model call's noise seed comes from `seeding.call_seed(rollout_seed, step, model, call)`.

**Verified on the Mac** (`tests/reference/test_handoff.py`, fake policies): forced handoffs in both
directions (mid-chunk, chunk point, consecutive escalations, runs of four) show no arm jump, in both
`xvla_state_mode`s: each π0.5 first step's goal is within one delta step of the arm, X-VLA is always sent
exactly the measured pose after π0.5 and its first target is within 3 cm, and every step ran in its model's
mode. Abandoned escalations hand back from the measured pose and charge no budget. Exactly 26 escalations fit
the budget. Routed always-local is identical to `xvla_only`, which is identical to the authors' own rollout
loop; `pi05_only` calls π0.5 every 10 steps in delta mode. Restoring an engine snapshot into another env
continues exactly. Sensitivity: removing the "forget the remembered pose" rule is caught (stale pose sent,
first target far from the arm), and a skipped mode switch is refused. The same continuity checks run against
the real models on the VM (`test_model_handoff_continuity_on_the_vm`).

## Reference-run files (`vlar_ops/reference_runs.py`)

`References/` (read-only): `task_<t>/eval_info.json` (π0.5, 50 successes per task in episode order) and
`progress.jsonl` (X-VLA). `vlar-ops reference check` validates the layout and the totals the user confirmed:
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
  references (global RNG). Seed derivation and batching are effort step 5.

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
| simulator | `pytest tests/reference/test_sim_reference.py tests/reference/test_snapshot_replay.py tests/reference/test_handoff.py` | adapters vs both references; snapshot restore and replay; determinism; handoffs, budget, run modes; sensitivity checks |
| simulator, both servers up | `VLAR_XVLA_SERVER=127.0.0.1:8777 VLAR_PI05_SERVER=127.0.0.1:8778 pytest tests/reference/test_handoff.py -k vm` | handoff continuity with the real models |
| X-VLA | `VLAR_XVLA_MODEL_PATH=… VLAR_XVLA_LORA_PATH=… pytest tests/reference/test_model_servers.py -k xvla` | server vs the authors' handler with the real weights and LoRA |
| LeRobot | `VLAR_PI05_CHECKPOINT=… pytest tests/reference/test_model_servers.py -k pi05` | server vs lerobot-eval's per-step processing (needs the gated tokenizer) |
| main | `pytest tests` | config, queue, spend guard, packaging |

## Running the reference tests

```bash
.venv-sim/bin/python -m pytest tests/reference                    # simulator + adapters, exact
VLAR_XVLA_CONFIG_DIR=third_party/hf/X-VLA-Pt VLAR_TEST_DEVICE=cpu \
  .venv-xvla/bin/python -m pytest tests/reference/test_model_servers.py -k xvla
```

The local environments (`.venv-sim`, `.venv-xvla`, `third_party/`) are git-ignored; the VM uses the
container images built from the same pins.
