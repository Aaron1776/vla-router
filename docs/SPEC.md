# Spec: Budgeted escalation to a remote policy on LIBERO-Long — router vs. cascader (X-VLA local, pi0.5 remote)

> Design reference for the harness build. Supersedes the model choice and phase plan in `PLAN.md` (which predates the switch from SmolVLA to X-VLA). Items marked [ASSUMPTION] are the spec author's working choices and are tracked in `ASSUMPTIONS.md` once they are tested. Approved amendments are listed in §17.

## 1. Goal

Build a single LIBERO-Long (`libero_10`) harness in which a small local policy (X-VLA) controls the robot by default and a learned decision-maker ("decider") chooses, every 10 environment steps, whether to escalate to a large "remote" policy (pi0.5) for one call (10 actions). The aim is to approach pi0.5's success rate while using pi0.5 for as few steps as possible, under a hard pi0.5 step budget, a per-episode time limit, and a simulated network that adds latency and can drop requests. The deployment story is that X-VLA plus the decider fit comfortably on a small GPU where pi0.5 would not, and pi0.5 sits behind a network.

Two kinds of decider are trained on identical data and compared: a router, which decides from the observation alone, and a cascader, which decides after seeing the actions X-VLA has proposed and signals derived from them. The cascader has a headline variant and two ablations (§6).

The work has four phases: (0) build the harness, verify parity with each model's standalone results and determinism, run the X-VLA base rollouts that calibrate the time limits, then freeze the harness; (1) a throughput pilot, exploration rollouts and branch rollouts that label decision points with measured outcomes; (2) training all deciders with leave-one-task-out cross-validation; (3) a shared DAgger-style round in which trained deciders are rolled out and the new states they reach are labelled and added to training. All collected data is packaged as zip archives and kept on the user's own computer (§13).

## 2. Models and their native conventions (must be preserved)

pi0.5 is `lerobot/pi05_libero_finetuned` (PaliGemma 3B backbone plus ~300M flow-matching action expert, PyTorch port by LeRobot, fine-tuned on all four LIBERO suites). Its native harness is LeRobot's `lerobot-eval`. Inputs are two camera images (rendered at 360 px, resized to 224 px, rotated 180°), the task text, and an 8-value robot state that is normalized and tokenized into the prompt; gripper rotation is converted by LeRobot's processor chain. Each call runs 10 flow-matching steps and returns 50 actions of 7 values (relative arm move plus gripper); only the first 10 are executed. It uses relative (delta) control.

X-VLA is X-VLA-Pt (0.9B, Florence-2 encoder only) with the LoRA adapter `2toINF/X-VLA-libero-long-peft` (trained on LIBERO-Long only), loaded in float32 with `domain_id=3`. Its native harness is the authors' client–server setup: `deploy.py` serves `POST /act` on port 8777 in its own Python environment (transformers 4.51.3), and `libero_client.py` drives raw LIBERO. Inputs are main and wrist images at 256 px, task text and a 20-value state. Each call returns 30 actions of 20 values (absolute target pose plus gripper, padded); the client converts rotations to LIBERO's format, rounds the gripper to open/closed, and queues all 30. It switches the controller to absolute mode after the settle steps. Known quirk: after the first call of an episode the client sends the final pose of the model's own previous chunk as the state instead of the measured pose, and the gripper slot is always 0. The client's native step limit is 900; the reference 520-step score was obtained by scoring the same 900-step run at 520 steps from per-episode step counts.

Both standalone harnesses already exist and work; they are the reference for preprocessing and must be reused, not rewritten from memory.

Reference run. The user's standalone notebook (`libero_long_pi05_xvla_eval.ipynb`) is the reference for both harnesses, their pinned versions and their scores. Pins: LeRobot 0.6.1 (`lerobot[pi,xvla,libero]`), whose `libero` extra installed hf-libero 0.1.4 (0.1.5 was published after the run) on robosuite 1.4.0 and MuJoCo 3.8.1, with the LIBERO scene assets `lerobot/libero-assets` at revision 0b3ea86b; π0.5 run with `--policy.n_action_steps=10 --policy.compile_model=false`, batch size 1, seed 1000, 520-step limit; X-VLA from `2toinf/X-VLA` at commit `6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2`, server environment transformers 4.51.3, peft 0.17.1, timm 1.0.12, accelerate 1.2.1, base model `2toINF/X-VLA-Pt` with the LoRA adapter (r=8, alpha 16), 10 flow-matching steps per request, client init seed 42 (episode seed 42 + ep + 100, initial state ep mod 50). Reference scores on all 500 episodes: π0.5 478/500 (95.6%, 95% CI 93–97); X-VLA 405/500 (81.0%, 95% CI 77–84) at 520 steps and 412/500 at 900; X-VLA successes used a median of 247 steps (90th percentile 387). Reference measurements: π0.5 median 435 ms per call and 9.5 GB peak allocated GPU memory; X-VLA median 420 ms per HTTP request and about 4.0 GB for the server process; one X-VLA client process sustained about 21 environment steps per second. These were measured on an NVIDIA L4 (the notebook's saved Colab metadata says T4, but all runs used an L4). They serve as reliable placeholders until Phase 0 re-measures them in the project's own harness, whose batching and request format differ from the notebook's.

## 3. Harness architecture

Simulator. One simulation layer using raw LIBERO (not LeRobot's wrapper) with a single episode-setup routine: build the task scene, load initial state k, take 10 no-op settle steps, then start the clock. Per-model adapters handle rendering at each model's native resolution and orientation, building each model's state vector, converting each model's actions into LIBERO controller commands, and switching the controller between absolute mode (X-VLA) and delta mode (pi0.5) at every handoff.

Images. Each model's images are rendered at its native resolution and taken from robosuite's observation pipeline, which samples every observation at physics substep 24 of the 25 in a step (a render taken after the step shows a different arm pose from what either model was evaluated on). The env's own cameras run at X-VLA's 256 px. π0.5's 360 px images come from two extra cameras registered on the env as observations and kept in step with the native ones. Snapshots and restore (§7) cover the extra cameras too.

pi0.5 server. Runs in the LeRobot environment and wraps the policy with LeRobot's exact processor chain, so it accepts raw observations and returns un-normalized 7-D delta actions.

X-VLA server. The authors' `deploy.py` in its own environment, with three endpoints:
- `POST /act`: one chunk generated with the call's own seed; returns the 30 actions and the pooled features of the same pass. Only chunks from `/act` are ever executed.
- `POST /sample`: n chunks for a given observation, batched, each with its own seed; returns the chunks and the pooled features. Measurement only: these chunks are never executed.
- `POST /features`: encoder-only pass over the current images and task text; returns the pooled features.
Because executed chunks always come from single-sample `/act`, X-VLA's executed behaviour is identical in every mode and run type and matches the parity check; measurement calls cannot change what the robot does.

Pooled features are two vectors of 1,024 values each: the mean of the encoder's output tokens (the main camera's image tokens and the 50 text tokens, padding included, since the encoder attends to all of them), and the mean of the wrist camera's image tokens, which reach the action transformer directly rather than through the encoder. Both are stored, and every decider takes both.

Request poses. The extra samples at a chunk point reuse the exact `/act` request, including the quirk pose, so they are samples of the same request as the executed chunk. Mid-chunk `/sample` probes and `/features` calls send the measured end-effector pose, so fresh samples describe what X-VLA would do from the current state.

Both servers run on the same machine on localhost and batch concurrent requests from all simulator workers (dynamic batching with a short wait window); see §12. The "remote" behaviour of pi0.5 is simulated in the clock (§5), not by real networking.

Run modes. `routed` (default): decision points every 10 steps, a decider (learned, or a scripted rule for data collection and scripted baselines) chooses local or escalate, and the pi0.5 budget is enforced. `xvla_only`: X-VLA alone. `pi05_only`: pi0.5 alone, called every 10 steps, no budget, X-VLA never called.

Calls at decision points (routed mode). A decision point is a chunk point if X-VLA's queue is empty (start of episode, its chunk is used up, or control has just returned from pi0.5) and a mid point otherwise. At a chunk point the harness calls `/act` before the decision and passes on its features; if the decider escalates, that chunk is discarded. At a mid point the harness calls `/features`, unless a `/sample` probe is being made (§6), which also returns the features. The encoder is deterministic, so features for the same observation are identical whichever call produced them.

Handoff rules. When the decider escalates, X-VLA's remaining queued actions are discarded, the controller switches to delta mode and pi0.5 is called fresh; of its 50 returned actions the first 10 are executed and the rest discarded. When control returns to X-VLA, the controller switches to absolute mode and X-VLA is called fresh, and that first call must send the measured end-effector pose (its remembered "last predicted pose" is stale after pi0.5 has moved the arm). Within an uninterrupted X-VLA stretch the original quirk is kept by default to match how X-VLA was evaluated, behind a flag `xvla_state_mode ∈ {quirk, measured}`.

Randomness. Each episode has two seeds.
- Scene seed: the env seed (`env.seed`), which together with the initial-state file defines the scene. LIBERO places some fixtures at reset from NumPy's global RNG, so in tasks 2, 3, 5, 8 and 9 the stove, cabinet, wine rack, desk caddy or microwave moves by up to about 19 mm with it; anything else the env seed controls is part of the scene too. It is 142 + k for initial state k in every project run: base, exploration, branches, DAgger, evaluation, and every baseline including π0.5 alone. The only exception is the π0.5 parity run (§11 step 4), which uses 1000 + k to match its own reference. 142 + k is also the X-VLA reference run's convention, so base rollouts face exactly its scenes.
- Rollout seed: everything else random. Every model call (X-VLA `/act` and `/sample`, π0.5) gets its own noise seed, derived deterministically from (rollout seed, step index, model, call type, sample index); the network model and exploration's random escalations draw from their own streams derived from the rollout seed. Rollout seeds differ between the base, exploration, DAgger and evaluation runs of the same task and initial state, and evaluation uses rollout seeds never used during data collection. Without this split, the X-VLA-alone baseline would replay the base rollouts exactly and exploration would replay them until its first escalation.

Adding or removing a call (a `/features` pass, a `/sample` measurement, or a branch) never shifts the randomness of any other call, so trajectories are reproducible across run types.

## 4. Decision points, escalation and budget

In routed mode a decision point occurs every 10 environment steps regardless of which model is driving: steps 0, 10, …, 510, up to 52 per 520-step episode. At each decision point the decider chooses "continue local" (execute the next 10 actions from X-VLA's queue) or "escalate" (discard X-VLA's queue, make one pi0.5 call, execute the first 10 of its actions). Escalation is allowed at consecutive decision points.

The pi0.5 budget is counted in executed steps: at most 260 pi0.5 steps per episode out of the 520-step limit, which is at most 26 escalations. Escalate is not offered once the budget is exhausted. Using fewer escalations is preferred even under the cap; this is enforced through the decision threshold (§8). A failed remote request costs time but not budget.

## 5. Simulated clock, latency, failures and overhead profiles

Event log. Every run records an event log: each environment step, each X-VLA `/act` call, each pi0.5 request with its network draws, failures and retries, and each decision point with its type (chunk or mid). The simulated clock under any overhead profile is a deterministic function of the event log and the measured latencies.

Base charges (every profile). Each environment step adds 0.05 s (LIBERO's 20 Hz control rate). Each X-VLA `/act` call adds X-VLA's measured median `/act` time. Each pi0.5 request adds pi0.5's measured median inference time plus the simulated network time below. Clock charges use batch-size-1 latencies, because a deployed robot serves one request at a time. The robot is stationary during every model call, as in both existing harnesses, so waiting is real cost.

Decision overhead (charged per decision point, by profile). This is what a deployed decider of that type would pay:
- `none` (xvla_only, pi05_only, time-limit calibration): nothing.
- `router` (the router, and all scripted baselines in routed mode): mid point: `/features` time; chunk point: nothing extra (features come with `/act`).
- `cascader_nospread` (ablation 1): same as `router`.
- `cascader_chunkspread` (ablation 2): chunk point: t(`/sample` K) − t(`/act`), since a deployed system would generate all K samples in one batched call; mid point: `/features` time.
- `cascader_full` (headline): chunk point: t(`/sample` K) − t(`/act`); mid point: t(`/sample` K) for the fresh probe.
Decider inference itself is assumed negligible and not charged; Phase 2 must verify it is under 5 ms.

Live versus offline clocks. In runs with a live learned decider (DAgger rollouts, evaluation), the clock runs live under that decider's profile, the decider sees its own remaining time, and the episode is cut at the time limit. Scripted runs (base, exploration, branches, scripted baselines) do not observe time: they run to 520 steps without a time cutoff, and success within the time limit is computed afterwards for any profile from the event log. This is exact because profiles only add waiting time while the robot is stationary.

[ASSUMPTION] Network model per pi0.5 request: upload time for two JPEG-compressed images plus state and text at a configurable uplink bandwidth (default 20 Mbps), plus a round-trip latency drawn from a log-normal distribution (default median 60 ms, heavy-tailed so roughly 1 in 50 requests exceeds 300 ms), plus download of the action chunk. Each request independently fails with probability `p_fail` (default 0.05), and a request whose sampled total time exceeds the timeout is also treated as failed. A failure is detected only when the timeout expires (default 1.0 s), after which the request is retried after a short backoff (default 0.2 s). If 2 retries also fail, the escalation is abandoned: X-VLA is called fresh with the measured pose and drives that 10-step segment, its remaining 20 actions stay queued for later decision points, and no budget is charged.

Time limit: per task, the 90th percentile of X-VLA's successful-episode completion time under profile `none`, multiplied by a margin m that is the same for all tasks. m is chosen in Phase 0 (§11 step 6) as the smallest value in {1.25, 1.5, 1.75, 2.0} such that at least 80% of π0.5's successful parity episodes (§11 step 4), timed under profile `none` with the network model, finish within their task's limit (rule approved 2026-11-21, §17). This keeps the limit binding (heavy escalation costs real time) without making escalation pointless; with the reference latencies, a π0.5 step costs about 1.6 times the simulated time of an X-VLA step, so a margin of 1.25 would let π0.5 alone complete only about 160–340 steps depending on the task. The limit is an environment property applied identically to every method; each method pays its own overhead against it. The live clock is checked after every environment step and every model call. A run succeeds only if the task is completed before 520 steps and before the time limit.

## 6. Decider inputs

Common inputs (router and every cascader variant): both pooled X-VLA feature vectors of the current observation (main encoder output and wrist camera tokens, §3), the robot state, the remaining pi0.5 budget, the remaining time under the decider's own profile, and the steps elapsed. There is no task-ID or separate task-text input: the encoder features already reflect the instruction, and an explicit task input would let a decider memorize its 8 training tasks, which cannot transfer to an unseen test task.

Router: the common inputs only. It never sees X-VLA's proposed actions or anything derived from them.

All cascader variants add:
- Proposal: the 10 X-VLA actions that would execute next if the decider continues (the next 10 from the queue at a mid point; the first 10 of the fresh chunk at a chunk point), the position within the chunk (0, 10 or 20), and whether the previous segment was driven by pi0.5.
- Free signals (no model call, no clock cost): tracking error, the position and rotation distance between the measured end-effector pose and X-VLA's planned target pose, averaged over the previous segment's 10 steps (set to zero and flagged when the previous segment was pi0.5's or at episode start); and gripper mismatch, the fraction of the previous segment's steps with a "closed" command and fully shut fingers (grasp missed or object dropped) and with a "closed" command and partly open fingers (object held). Finger-width thresholds are calibrated in Phase 0.

Variant-specific inputs (K = 5 [ASSUMPTION]):
- `cascader_nospread` (ablation 1): no spread inputs.
- `cascader_chunkspread` (ablation 2): chunk-time spread: at each chunk point, the executed chunk plus 4 `/sample` chunks give K samples; the spread summary is computed separately for each 10-step segment of the chunk and reused at that segment's decision point.
- `cascader_full` (headline): fresh spread and drift at every decision point. At a chunk point these come from the K chunk-time samples. At a mid point a `/sample` probe draws K fresh chunks for the current observation and the measured end-effector pose (§3); fresh spread is computed over their first 10 actions, and drift is the distance between the 10 queued actions about to execute and the mean of the K fresh first-10 actions. The robot still executes the queued actions.

Spread summary for a 10-step segment: the mean over steps of the standard deviation of target end-effector positions across samples, the maximum over steps of that standard deviation, the mean angular deviation of target rotations from their mean, and the fraction of samples disagreeing with the majority gripper command. Drift: mean position distance, mean angular distance, and fraction of steps with gripper disagreement.

## 7. Phase 1 data: snapshots, rollouts, branches, labels and split

Full measurement. In base, exploration and DAgger rollouts, the harness records at every decision point everything any decider could need: features, the proposal, free signals, the 4 extra `/sample` chunks at chunk points, and a K-sample `/sample` probe at mid points. Measuring at every point (not only at points later branched) keeps the dataset complete for future deciders (§13). If the budget requires it (§12), probes are recorded only at sampled decision points and the chunk points their spread depends on. Branch rollouts record outcomes only, not measurements.

Snapshots. In base, exploration and DAgger rollouts, a snapshot is taken at every decision point, after that point's calls. It stores the full simulator state and the episode state:
- MuJoCo's integration state including the constraint solver's warm start (`mj_getState` with `mjSTATE_INTEGRATION`; the flattened `env.sim.get_state()` is kept alongside for inspection), and a fingerprint of the compiled model;
- the observation timers and buffered observation values of every observation, including the extra π0.5 cameras, so the observation at the decision point is restored exactly rather than re-rendered;
- the controller's internal state (goals, cached pose and Jacobians, control mode) and the gripper's internal state (its integrated command);
- LIBERO's Python-side object states (for example, whether the stove is on) and robosuite's step counters;
- task id, initial-state index, episode seed, step count, pi0.5 budget used, which model was driving, X-VLA's remaining queued actions (for a chunk point, the fresh chunk), X-VLA's last predicted pose (for the quirk), the event log up to that point, and the measurements above.

Restoring a snapshot puts all of this back into an env of the same task whose compiled model matches the fingerprint. It never calls `set_init_state` (whose forced observation update would shift the sampling timers) and never re-renders; no settle steps are taken after a restore. The replay test compares observations exactly (every camera image at both resolutions and every proprio value), not only the physics state.

Base rollouts (run in Phase 0, §11): X-VLA always-local in routed mode on every task and initial state (10 × 50 = 500 episodes), run to 520 steps without a time cutoff. Exploration rollouts: one additional routed episode per task and initial state, in the same scene with a fresh rollout seed (500 episodes [ASSUMPTION]), in which each decision is a random escalation with probability 0.1 [ASSUMPTION], respecting the budget, so the data contains decision points with partly used budgets and mixed control history.

Decision-point sampling. Branches are run from a random sample of each base, exploration and DAgger episode's decision points, a fixed fraction per episode chosen with a recorded seed. The fraction is set by the pilot (§12). Decision points with no budget remaining are skipped.

Branches. From each sampled decision point, four continuations are run in routed mode with scripted decisions, to 520 steps without a time cutoff:
1. Continue: X-VLA executes its stored queue, then continues with no escalation.
2. One call: discard X-VLA's queue, one pi0.5 call (10 steps), then X-VLA to the end.
3. Three calls: three consecutive pi0.5 calls (30 steps), then X-VLA to the end.
4. Max: consecutive pi0.5 calls until the budget is exhausted or the episode ends, then X-VLA to the end.
In branch 1 the stored queue is executed identically in every seed, so the label refers to exactly the proposal a cascader saw; seeds vary only the calls made after it. If the remaining budget is below 30 steps, branch 3 is truncated to the remaining budget; branches that become identical as a result are run once and shared. Each branch runs with 3 rollout seeds, in the snapshot's scene. A base rollout counts as one seed of branch 1 at every decision point (it continued locally with its own rollout seed); every other branch seed is derived from (snapshot, branch, seed index) and never equals the parent rollout's. Each branch records its event log, its end-effector and gripper trajectory per step, success and the step of success, pi0.5 steps and requests, and failed requests.

Labels. For a decider with profile P, the label of branch b at decision point d is the fraction of b's seeds that succeeded before 520 steps and before the time limit, with time computed under P from the parent rollout's event log up to d plus the branch's event log. One set of branch rollouts therefore yields correct labels for every decider type.

Data split: leave-one-task-out cross-validation with a separate validation task. Labels are collected once for all 10 tasks and all 50 initial states and shared by every fold. There are 10 folds; in fold i, task i is the test task, task (i+1 mod 10) is the validation task, and the remaining 8 tasks are the training tasks. Splits are always by whole task, never by snapshot or episode. Test episodes use rollout seeds never used during data collection (§3); their scenes are the same as everywhere else (scene seed 142 + k).

## 8. Training (Phase 2)

In each fold, four deciders are trained: the router, `cascader_full` (headline), `cascader_nospread` (ablation 1) and `cascader_chunkspread` (ablation 2). Each is an MLP or tiny transformer under about 5M parameters, sharing one architecture family and differing only in input width. Each outputs four probabilities, one per branch: success if X-VLA continues, after one pi0.5 call, after three pi0.5 calls, and after maximum escalation. The decision rule is: escalate if p_one_call − p_continue > λ and budget remains. Heads 3 and 4 are auxiliary targets that teach whether a state is rescuable and whether one call is enough; an auxiliary time-to-success head is optional. Labels are scored under each decider's own profile (§7). Branch labels measure a one-step deviation against X-VLA continuing alone, not against the decider continuing; the DAgger round (§9) exposes deciders to the states their own repeated decisions produce.

Protocol, per fold and per decider: train from scratch (fresh random initialization, no weights carried between folds or deciders) on the 8 training tasks, checkpointing every epoch. Early stopping uses prediction loss on the validation task. λ is chosen on the validation task offline from branch labels: for each candidate λ, estimate the success gain over always-local (branch 2 versus branch 1 outcomes at the decision points where the decider would escalate) and the resulting escalation rate; choose the λ with the lowest escalation rate that achieves at least 95% of the maximum estimated gain [ASSUMPTION], subject to the escalation rate staying within budget. The final decider is the one trained on the 8 training tasks; it is not retrained with the validation task. Hyperparameters other than the stopping epoch and λ are identical across folds. Training needs only stored features, measurements and labels, so it can run either on the cloud VM or on the user's own computer (§13).

Deployment memory target: X-VLA plus any decider must run within 8 GB of GPU memory at inference, measured as peak allocated memory, including the K-sample batch for the cascader variants that use it. In the reference run the X-VLA server used about 4.0 GB and π0.5 peaked at 9.5 GB, consistent with π0.5 not fitting this target.

## 9. DAgger round (Phase 3)

One shared collection serves all deciders. Train a router and a headline cascader from scratch on all 10 tasks using the Phase 1 labels, each with the median stopping epoch and median λ from its 10 Phase 2 folds. Roll them out once per task and initial state (500 episodes): even initial-state indices driven by the router (live clock, `router` profile) and odd indices by the headline cascader (live clock, `cascader_full` profile), so both deciders cover every task. Full measurement and snapshots are recorded at every decision point; decision points are sampled as in §7 and the same four branches run with 3 seeds. Because labels do not depend on the decider, every decider type trains on all the new labels.

Each fold then adds the new labelled points from its 8 training tasks to training and from its validation task to validation, drops its test task's points, retrains all four deciders from scratch, and re-selects the stopping epoch and λ as in §8. The collecting deciders were trained on every task, so the states they visit on a fold's training tasks are slightly influenced by that fold's test task; no test-task labels reach a fold's deciders, and this indirect influence is accepted as a budget compromise. One round is in scope.

## 10. Evaluation

Learned deciders are evaluated live on their fold's test task: all 50 initial states, one fresh rollout seed each (scene seed 142 + k, as in data collection), at the validation-chosen λ, under their own profile, cut at the step and time limits. Before DAgger: the router and the headline cascader (1,000 episodes across folds). After DAgger: the router, the headline cascader and ablation 1 (1,500 episodes), plus ablation 2 (500 episodes) if the reserve allows.

Scripted baselines run once on all tasks and initial states with fresh rollout seeds (the same evaluation rollout seed for a given task and initial state as every learned decider; scene seed 142 + k, including for π0.5 alone), to 520 steps without a time cutoff, and are scored both with and without the time limit from their event logs: X-VLA alone (`xvla_only`, profile `none`); pi0.5 alone (`pi05_only`, profile `none`, no budget), reported both as an unconstrained reference and under the time limit; a fixed-schedule rule (routed, `router` profile) that escalates at evenly spaced decision points, with spacing chosen so its pi0.5 usage matches the post-DAgger headline cascader's mean usage; and random escalation (routed, `router` profile) matched per fold to the pi0.5 usage of the post-DAgger router, and separately to that of the post-DAgger headline cascader.

The key comparisons are: headline cascader versus router (the value of seeing X-VLA's plan and confidence); ablation 1 versus router (the plan and free signals alone, at identical overhead); headline versus ablation 1 (the value of the spread signals); headline versus ablation 2 (fresh versus reused spread); each learned decider versus its usage-matched random baseline (the value of learning); and before versus after DAgger.

Report, per task and as the mean and spread across folds: success rate, pi0.5 steps as a fraction of executed steps, pi0.5 requests and failed requests per episode, simulated time used, end-reason breakdown (success, step limit, time limit), and peak GPU memory of X-VLA plus each decider. If the reserve allows, also evaluate the post-DAgger router and headline cascader at two additional λ values (one lower and one higher pi0.5 usage) to show the success-versus-usage trade-off; the headline result is always the validation-chosen λ.

Paired comparisons. Every method faces identical scenes for each task and initial state (scene seed 142 + k) and the same evaluation rollout seed, so alongside the per-method rates, report for each key comparison the paired per-episode differences (success, pi0.5 steps, simulated time) over the same task and initial-state pairs, with paired confidence intervals.

## 11. Phase 0: build, verify, calibrate, freeze

In order:
1. Build the harness (§3–§7) with placeholder latencies and time limits taken from the reference run (§2).
2. Measure on the L4, for batch sizes 1, 2, 4 and 8: median latency and GPU time for X-VLA `/act`, `/features` and `/sample` with K = 5, and for π0.5 inference. Write the batch-size-1 medians to the config as the clock charges, and use the batched figures to set the number of simulator workers (§12). Calibrate the finger-width thresholds for the gripper-mismatch signal.
3. Determinism and handoff tests: snapshot restore-and-replay must reproduce trajectories, including resuming X-VLA mid-chunk from a restored queue; identical seeds must reproduce identical runs; adding or removing measurement calls must not change executed trajectories; batched and unbatched inference must give the same executed actions within numerical tolerance; the clock recomputed offline from an event log must equal the live clock for every profile; forced handoffs in both directions must produce no arm jump (end-effector pose continuity): with the fake models on the Mac, X-VLA's first target after a handoff is within 3 cm of the arm; with the real models on the VM, it must be at or below the 99th percentile of the first-step target distances X-VLA produces at ordinary chunk boundaries without a handoff in the base rollouts (and π0.5's first goal within one delta step). Every equality and determinism test that passed on the Mac (including the adapter and server equivalence tests against the reference code) must be re-run on the L4 VM (Linux, EGL rendering, CUDA) before the freeze, because rendering and floating-point results can differ across platforms. Mac results show the code paths are right; only VM results count for parity and determinism.
4. π0.5 parity: `pi05_only` on initial states 0–9 of every task (100 episodes) with scene seed 1000 + k, the reference's own (§3), 520-step limit, no time cutoff, event-logged, compared episode by episode with the reference run's results on the same initial states. The success rate must lie within sampling error of the reference on those episodes; otherwise find and fix the preprocessing difference first.
5. Base rollouts: X-VLA always-local in routed mode on all 500 episodes (scene seed 142 + k, the same scenes as the X-VLA reference run), 520-step limit, no time cutoff, full measurement and snapshots at every decision point. Their success at 520 steps is the X-VLA parity check against the reference 405/500, within sampling error, and is also compared episode by episode with the reference.
6. Compute the per-task time limits and the margin m (§5) from the base rollouts and the π0.5 parity episodes, and write them to the config. Report, per task, the limit and the share of π0.5 successes that fit within it.
7. Choose the golden set (§15) and freeze the harness as v1.0. Base rollouts need no re-running: being uncut with full event logs, they are scored offline under v1.0's time limits.

## 12. Compute, cost, spend guard and pausing ($50 Google Cloud)

Use one g2-standard-16 Spot VM (1× NVIDIA L4 24 GB, 16 vCPUs, 64 GiB RAM), about $0.46/hour Spot at the time of writing, roughly 100 hours. Run parallel simulator workers against both model servers on the single GPU. Spot VMs can be preempted, so all data collection runs from a resumable job queue: every rollout and branch is an idempotent job keyed by (snapshot id or episode id, branch type, seed), results are appended to disk and synced to a Cloud Storage bucket as each job finishes, and a restarted VM resumes where it stopped.

Throughput. The reference run shows simulation takes about 1 s of CPU time per 30 X-VLA steps per process, while each model call occupies the GPU for roughly 0.4 s unbatched, so with many workers the GPU, not the CPU, is the bottleneck. Both model servers must batch concurrent requests from all workers (dynamic batching with a short wait window), and the worker count is set from the Phase 0 batched measurements so the GPU stays saturated without long queues. Rough pre-pilot estimate: about 50 unbatched GPU-seconds per sampled decision point across its 11 branch rollouts (dominated by branch 4), which at the planned Phase 1 budget suggests a branch-sampling fraction of roughly 10–15% of the ~30,000 base and exploration decision points; the pilot replaces this estimate.

Indicative budget split (estimates, to be replaced by the pilot's measurements): Phase 0 including base rollouts about $8; a pilot at the start of Phase 1 about $2 (a small set of exploration rollouts and branches that measures rollouts per hour); the rest of Phase 1 about $16; training all deciders about $2, or $0 if run on the user's own computer from downloaded features and labels (§13); Phase 3 DAgger about $8; evaluation about $11; storage during the project about $1; about $2 in reserve. After the pilot, recompute the full plan from measured throughput and choose reductions before committing. If the plan will not fit, reduce in this order: lower the decision-point sampling fraction, drop branch 3, record `/sample` probes only at sampled decision points (and the chunk points their spread depends on) instead of at every decision point, halve the exploration rollouts, use 2 seeds instead of 3, skip the ablation 2 evaluation, skip the pre-DAgger router evaluation, then reduce initial states per task.

Rendering. Cameras may be rendered only on the steps whose observations a model or a snapshot actually uses, with their sampling timers kept in step with the always-on observations. This is approved only after restore (§7) is in place, and only if it passes the same exact-equality tests as always-on rendering. On the VM, EGL rendering shares the L4 with model inference, so this also frees GPU time, which matters more than the CPU saving.

Job ordering for graceful stopping. Within each phase, jobs are ordered so that stopping at any point leaves a usable, balanced dataset. Episode-level jobs (exploration, DAgger, evaluation) run in a random order across all tasks and initial states. Branch jobs run in three tiers: first, seeds 1 and 2 of branches 1, 2 and 4 for every sampled decision point; second, seed 3 of branches 1, 2 and 4; third, branch 3. Within each tier, decision points run in a random order across all tasks, so a partial tier is equivalent to a lower sampling fraction. Training and evaluation use only decision points whose required branches and seeds are complete.

Spend guard. Google Cloud budgets only send alerts and do not stop a running VM, so the harness enforces its own limit. A spend ledger stored in the bucket records cumulative VM hours, the hourly rate (set in the config from the current Spot price) and estimated storage cost, and survives VM restarts. At a soft limit of $40 the guard emails the user; at a hard limit of $45 it stops taking new jobs, lets in-progress jobs finish or abandons them cleanly, syncs all results to the bucket, and shuts the VM down. A Google Cloud budget alert at 50%, 80% and 100% of $50 is configured as a backup, and the ledger is reconciled against the actual billing report at least weekly, since alerts can lag by hours.

Pausing and resuming. Work can be paused at any time (spend guard, preemption, or by choice) and resumed later, possibly after adding funds, from the job queue on the same frozen harness version. While paused, only bucket storage is charged (about $1–2 per month); the VM incurs no cost while stopped apart from its disk, and should be deleted if the pause is long, since everything needed to resume is in the bucket. Spot capacity is not guaranteed at resume time; if none is available, wait or temporarily use an on-demand VM at its higher hourly rate.

## 13. Data retention and reuse

During the project, all data lives in a Cloud Storage bucket. At the end, everything is packaged as zip archives, downloaded to the user's own computer, verified, and the bucket and VM are deleted.

Dataset contents (dataset v1, tied to harness v1.0):
- Per episode (base, exploration, DAgger, evaluation): metadata, full event log, per-step end-effector pose and gripper trajectory, and outcome.
- Per decision point: the snapshot, the camera images at each model's native resolution (lossless PNG), both pooled feature vectors, every raw `/act` and `/sample` chunk, the free signals, and the spread and drift summaries.
- Per branch: event log, per-step trajectory and outcome.
- Per trained decider: checkpoint, config, fold, chosen stopping epoch and λ, and a manifest of the training data used.
- Sample videos from evaluation, and the results report.

Records are stored as Parquet tables; images and simulator states are stored in sharded files keyed by id. A manifest records the schema version, harness version, config hash and file checksums, and a datacard documents every field and exactly how labels are computed.

Packaging for download. Two archive sets are produced: `dataset-v1` (everything above) and `harness-v1.0` (the three container images exported as files, and the model weights with their SHA-256 hashes). Archives use zip "store" mode (no recompression, since PNG and Parquet are already compressed), are split into parts of about 10 GB so a failed download can be retried part by part, and come with a SHA-256 checksum file covering every part. Expected total size is 60–80 GB. The download is verified against the checksums on the user's computer before the bucket is deleted. If the account's free monthly data-transfer allowance covers the download, it is free; otherwise it is billed per GB, and the download can be split across two months to stay within the allowance.

Reuse from the local copy without new simulation: training new decider types (the deciders are small and train on stored features, so this runs on the user's own computer with no cloud cost); new inputs derived from stored data (new spread summaries from the raw samples, or a different image encoder run on the stored images); different time limits or overhead profiles (re-scored from event logs); and new label definitions (for example time-to-success regression). Mid-project, decider training can also run locally by downloading only the features, measurements and labels (a small fraction of the dataset) and uploading the trained checkpoints back to the bucket for Phase 3 and evaluation.

Reuse that needs new simulation (new rollouts or branches from stored snapshots) requires restoring harness v1.0 from the `harness-v1.0` archive, either on a cloud VM (uploading the archive back to a bucket) or on a local machine with a suitable GPU.

## 14. Deliverables

A repository containing the harness (simulator layer, both model servers with request batching, adapters, run modes, overhead profiles, clock and network model, snapshot store, job queue, spend guard), configs for every parameter above, training and evaluation scripts including the fold logic for all four deciders, the packaging and verification scripts for §13, and a short results report with the metrics and comparisons from §10, including sample videos of escalation episodes. The `dataset-v1` and `harness-v1.0` zip archives are delivered to the user's own computer.

## 15. Harness isolation (applies once Phase 0 completes)

At the end of Phase 0 the harness is tagged as harness v1.0 and frozen. From then on it is treated as a fixed instrument, separate from the decider, data-collection and evaluation code that use it.

Packaging. The harness ships as three container images, one per Python environment: the simulator image (LIBERO, robosuite, MuJoCo, adapters, run modes, profiles, clock and network model, snapshot store), the X-VLA server image (transformers 4.51.3 and the other pins in §2), and the pi0.5 server image (LeRobot 0.6.1). All dependencies are pinned with lockfiles. Model weights are downloaded once, recorded with SHA-256 hashes, and mounted read-only; containers run with Hugging Face offline mode and no outbound network access at runtime.

Interface. Code outside the harness may use only a small client API: `reset(task_id, init_idx, rollout_seed, mode, profile, record)` (the scene seed is not a parameter: the harness sets it from the initial state, 142 + k, or 1000 + k under the named option for the π0.5 parity run) with mode ∈ {`routed`, `xvla_only`, `pi05_only`}, profile ∈ {`none`, `router`, `cascader_nospread`, `cascader_chunkspread`, `cascader_full`}, and record ∈ {`off`, `full`}; `restore(snapshot_id)`; `observe()` (routed mode only; returns the inputs for the active profile's decider type); `decide(choice)` where choice is `local` or `escalate` (runs the next 10-step segment and returns its outcome and the next decision point); `run_episode()` (for `xvla_only` and `pi05_only`); `snapshot()`; and `run_branch(job)` for queue workers. Deciders live in a separate package that depends only on this client. The step limit, time limit, pi0.5 budget, controller switching, handoff rules, clock charges and network model are enforced inside the harness, so a buggy decider cannot exceed the budget, skip the clock or alter preprocessing.

Configuration. Harness parameters (latencies, time limits and margin, step limit, budget, K, network model, profiles) are baked into a versioned config inside the frozen harness. Experiments may only choose among named, pre-defined options the harness exposes (for example `network=default` or `network=lossy`); they cannot set arbitrary values.

Versioning. Every record the harness writes carries the harness version, git commit, container image digests and config hash. Data from different harness versions are never mixed. Any change to harness code creates a new version, requires re-running the Phase 0 determinism and parity tests, and either regenerates or separately stores affected data.

Startup self-check. A golden set of 5 saved snapshots with recorded trajectories and outcomes is replayed every time the harness starts, including after every Spot preemption. If any replay does not match, the job queue does not start.

## 16. Effort checkpoints (instructions for the building chat)

The user runs this project at xhigh effort by default and switches to max only for the hardest steps, because max uses up usage limits faster. You cannot change your own effort setting, so before starting any step below, stop and tell the user: "Effort checkpoint: please switch to max effort before I start [step]." Wait for confirmation before proceeding. When that step is finished, tell the user: "Effort checkpoint complete: you can switch back to xhigh."

Steps that require max effort:
1. Designing and implementing the per-model adapters and preprocessing (§3), and any debugging when a parity check in §11 steps 4–5 fails or lands outside sampling error.
2. Implementing snapshot restore and the determinism tests (§7, §11 step 3), and any debugging when a determinism or replay test fails.
3. Implementing the handoff rules and controller mode switching (§3), and any debugging when the handoff continuity test fails.
4. Implementing the event log, overhead profiles, offline clock re-scoring, label computation and time-limit calibration (§5, §7, §11 step 6).
5. Implementing request batching and per-call seeding in both model servers (§3, §12), including the batched-versus-unbatched equivalence check.
6. A review pass, after implementation, of the fold logic, data-split enforcement and λ selection (§7, §8), checking specifically for leakage between training, validation and test tasks.
7. Re-planning the budget after the pilot (§12), and interpreting the final results (§10).

Also raise an effort checkpoint whenever an unexplained discrepancy appears that could silently corrupt labels or results, even if it falls outside the steps above.

## 17. Amendment log

Approved by the user on 2026-11-20, after effort step 1 (findings in `docs/HARNESS.md`):
1. §2: pins add hf-libero 0.1.4 and the LIBERO scene assets at revision 0b3ea86b.
2. §3, §6, §13: pooled features are two 1,024-value vectors (main encoder output, wrist camera tokens); both are stored and every decider takes both. Replaces the mean-pooling assumption.
3. §3, §6: mid-chunk `/sample` probes send the measured end-effector pose; the extra samples at a chunk point reuse the exact `/act` request, including the quirk pose.
4. §3: π0.5's 360 px images come from extra cameras kept in step with the native ones; snapshots and restore cover them.
5. §7: restore restores the full simulator state (MuJoCo state with solver warm start, observation timers and buffered values, controller and gripper state, LIBERO object states) instead of re-rendering; the replay test compares observations exactly.
6. §11 step 3: every equality and determinism test that passed on the Mac is re-run on the L4 VM before the freeze; only VM results count for parity and determinism.
7. §12: rendering only the observations that are used is approved after restore is done, if it passes the same exact-equality tests.

Approved by the user on 2026-11-21:

8. §3, §7, §10, §11, §15: two seeds per episode. The scene seed (the env seed) is 142 + k for initial state k in every project run, including every baseline; only the π0.5 parity run uses 1000 + k. The rollout seed drives model noise, the network model and exploration's random escalations; it differs between the base, exploration, DAgger and evaluation runs of the same task and initial state, and evaluation's are never used in data collection. §10 adds paired per-episode differences with paired confidence intervals, since every method faces the same scenes.
9. §5, §11 step 6: the time-limit calibration rule is approved: m is the smallest value in {1.25, 1.5, 1.75, 2.0} such that at least 80% of π0.5's successful parity episodes, timed under profile `none` with the network model, finish within their task's limit. The "pending" marker is removed.
10. §11 step 3: the real-model handoff check on the VM is relative: X-VLA's first target after a handoff must be at or below the 99th percentile of the first-step target distances it produces at ordinary chunk boundaries without a handoff in the base rollouts. The fixed 3 cm threshold stays for the fake-model tests on the Mac.
11. §2, §11: the reference run's 11 per-episode files are committed byte for byte in `References/`, with their SHA-256 pinned in `src/vlar_ops/configs/reference-run.sha256`; the parity tooling verifies them before use.
12. §18 (new): the milestone checkpoint for the end of the harness build. All seven conditions must hold before announcing "Milestone checkpoint: the harness build is complete and Phase 0 on the L4 VM is next."; then wait for the user's explicit go-ahead before creating any cloud resources or spending any money.

## 18. Milestone checkpoint: harness build complete

The harness build is complete only when all of the following are true:
1. Effort steps 4 (event log, overhead profiles, offline clock re-scoring, labels, time-limit calibration code) and 5 (server batching, per-call seeding, batched-versus-unbatched equivalence) are done, with their tests passing on the Mac.
2. Both model servers implement every endpoint in §3 (/act, /sample, /features for X-VLA; π0.5 inference), with batching and per-call seeds.
3. Measurement recording (proposal, free signals, spread and drift, probes) and the snapshot store are implemented and tested.
4. The job queue (idempotent jobs, ordering tiers, bucket sync, resume after preemption), the spend guard with its ledger, and the packaging and verification scripts for §13 are implemented and tested.
5. The three container images and lockfiles build, and everything that can be tested locally in them passes.
6. Ready-to-run setup scripts exist for the Google Cloud side: the g2-standard-16 Spot VM, the bucket, and the budget alert at 50%, 80% and 100% of $50.
7. The VM re-run list in docs/HARNESS.md is complete, the parity tooling and reference JSON files are committed, everything is committed, git status is clean, and HANDOFF.md is up to date.

When all seven are true, stop and tell the user exactly: "Milestone checkpoint: the harness build is complete and Phase 0 on the L4 VM is next." With it, give a short checklist confirming each of the seven items with its commit, the estimated Phase 0 cost and VM hours, and the first commands Phase 0 will run. Then wait for the user's explicit go-ahead before creating any cloud resources or spending any money. Do not announce this milestone early or partially; if any item is incomplete, say which ones remain instead.
