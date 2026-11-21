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
- **For effort step 2 (snapshots):** the restore procedure in SPEC §7 ("set_init_state, …, re-rendering
  observations") cannot reproduce trajectories: a re-render shows the post-step state instead of the
  substep-24 observation, and `set_init_state`'s forced update shifts the sampling phase by one
  substep for the rest of the branch. A faithful restore must also bring back the observables' timers
  and current values, the controller's internal state (goals, cached pose, `use_delta`), the gripper's
  integrated command (`current_action`), LIBERO's Python-side object states (e.g. the stove), and
  MuJoCo's warm-start (`mj_getState` with the integration spec). This is designed and tested in step 2.

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
- **Pooled features** (spec assumption): `enc_mean`, the mean over all encoder output tokens (first
  image + 50 text tokens, padding included since the encoder attends to it), and `wrist_mean`, the mean
  over the wrist view's image tokens, which bypass the encoder. 1024 floats each.
- **Noise**: with seeds, each sample's initial noise comes from its own CPU generator, so it depends on
  nothing but its seed (not the batch, not the GPU). Without seeds the servers behave like the
  references (global RNG). Seed derivation and batching are effort step 5.

## Throughput note (decide in Phase 0)

On this Mac, an env step costs 12.8 ms with the 256 px cameras and 19.9 ms with the 360 px ones added.
Models only consume images at decision points (every 10 steps), so enabling the camera observables only
on the step before a decision point (with their timers re-synced from an always-on observable) could
remove most rendering cost. It changes nothing physical, but it must pass the same exact-equality tests
before it is used.

## Running the reference tests

```bash
.venv-sim/bin/python -m pytest tests/reference                    # simulator + adapters, exact
VLAR_XVLA_CONFIG_DIR=third_party/hf/X-VLA-Pt VLAR_TEST_DEVICE=cpu \
  .venv-xvla/bin/python -m pytest tests/reference/test_model_servers.py -k xvla
```

The local environments (`.venv-sim`, `.venv-xvla`, `third_party/`) are git-ignored; the VM uses the
container images built from the same pins.
