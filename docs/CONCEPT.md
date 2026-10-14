# Concept Document: Edge-Cloud Router for Robot Action Models

CSCI 5942, Mega Assignment. Oct 13, 2026. Team (pair): Oscar Rodriguez and Miguel Ramirez-Aleman. Direction: A6, a router over two models.

We build a router that decides, every 10 robot actions, whether to use a small onboard robot model or pay a network round trip for a large remote one. It works if it keeps success within 3 points of always using the large model while calling it on at most half of the decisions.

## 1. What we will build

A router between a simulated robot and two vision-language-action models (VLAs): SmolVLA (450M parameters) onboard and π0.5 (4.1B parameters) on a remote server [1, 2]. Each model predicts a chunk of actions; the robot executes the first 10, then needs a new chunk. At each of these decisions the router chooses, under a deadline, between the small model's chunk and paying the network round trip for the large model's. We build three things:

1. **A routing model server** for AllenAI's vla-evaluation-harness. It owns the action buffer, calls both models, applies an escalation policy and logs every decision.
2. **A decision-record dataset**, harvested by running that server on LIBERO tasks: observation, both models' chunks, measured latencies, and a label for whether the small model's chunk was good enough.
3. **Four escalation policies and a scorer.** The policies form a 2×2: decide before running the small model (router) or after inspecting its output (cascade), each as a hand-written rule or a learned classifier. The classifiers are small models over a few signals and train in seconds on a CPU. The scorer replays the records under a simulated clock with network delay, jitter and loss.

## 2. On what data

We train no VLA. Our data is what the LIBERO simulator [3] and the two released checkpoints (`lerobot/smolvla_libero`, `lerobot/pi05_libero_finetuned`) produce. At 10 actions per decision, each suite's step limit caps the decisions per episode.

| Suite | Role | Episodes | Max decision records |
| --- | --- | --- | --- |
| LIBERO-10 (long-horizon) | Headline | 10 tasks × 50 starts = 500 | 26,000 (52 per episode) |
| LIBERO-Plus [4], LIBERO-10 variants | Headline | 2,519 perturbed | 131,000 (52 per episode) |
| Spatial, Object, Goal | Control; extra training data if needed | 1,500 | 40,000 (22 to 30 per episode) |

LIBERO-Plus perturbs camera, lighting, background, layout, language, robot initial state and sensor noise.

**Labels.** For 2,000 decisions we restore the saved simulator state, execute each model's chunk, finish both branches with the large model and compare task success. A cheaper proxy label (the distance between the two chunks) is validated against that subset before we use it on the rest.

**Split.** Learned policies are tested only on tasks they never trained on: train on 9 of the 10 LIBERO-10 tasks, test on the tenth, rotate through all 10. The held-out task's LIBERO-Plus variants stay out of training, and hand-written rules tune their thresholds on the same 9 tasks. We add the Spatial, Object and Goal suites only if held-out AUROC is below 0.65 with LIBERO-10 alone.

**Is the gap there?** The project needs the small model to fail where the large one succeeds. We plan for a gap of 15 to 25 points: below 15 there is too little to route, and above 25 the escalation cap is hard to meet. Published LIBERO-10 scores put SmolVLA at 71% [1] and π0.5 at 92.4% (OpenPI) to 96.0% (our checkpoint) [6], a gap of 21 to 25 points. The released SmolVLA checkpoint scores 76% against the paper's 87% averaged over four suites, attributed to an eighth of the training budget [5], so our gap may land higher; and the SmolVLA paper replans after every action [1], so its scores do not tell us how it does at 10 actions per decision. Our first experiment measures the gap: 10 episodes per task on LIBERO-10 (100 per model) and 100 LIBERO-Plus episodes per model.

**If it is not enough.** Gap below 15 points on both headline suites by Oct 20: we switch the remote model to GR00T N1.7 or move to the harder RoboCasa365 tasks before building further. Gap above 25: we report the escalation rate needed to stay within 3 points. Exact simulator state restore fails: proxy labels only, validated on fresh rollouts.

## 3. On what compute

The headline plan needs about **85 GPU-hours**, roughly 3.5 days of the Spark running continuously; the extra training data raises it to about 103.

**Machines.** An NVIDIA DGX Spark (GB10, 128 GB) as the server, a MacBook M2 Max as the robot-side node, and the course cluster (4× RTX 8000, 48 GB) for overflow.

**Measured by us on the Spark** (one 50-action chunk, median of 100 calls, uncompiled): SmolVLA takes 0.107 s with real instructions through its full preprocessing (0.082 s for the model call alone) and 1.0 GB; π0.5 in bfloat16 takes 0.268 s and 9.5 GB; eight SmolVLA samples in one batch, for a spread signal, take 0.236 s (model call) and 1.2 GB.

**Arithmetic.** Decisions D = 26,000 + 131,000 = 157,000 on the headline suites, an upper bound since successful episodes end early.

| Step | Calculation | GPU-hours |
| --- | --- | --- |
| Harvest: one small and one large call per decision | 157,000 × (0.107 + 0.268) s | 16.4 |
| Seven extra small-model samples (model call, batch of 8 minus batch of 1) | 157,000 × (0.236 − 0.082) s | 6.7 |
| Always-remote baseline episodes | 157,000 × 0.268 s | 11.7 |
| Gold labels: 2,000 decisions × 2 branches × 26 remaining large calls | 104,000 × 0.268 s | 7.7 |
| **Headline total, and with 2× for reruns and debugging** | | **42.5, 85** |

The gap checks' episodes are part of the harvest and baseline rows. Extra training data, only if LIBERO-10 alone is not enough, costs the first three rows again per decision: 40,000 × (0.375 + 0.154 + 0.268) s = 8.9 GPU-hours, for 51.4 in total and 103 with 2× (about 4.3 days).

**Caveats.** Simulator stepping time is not in these numbers yet; we measure it this week. The cluster's throughput is not measured either. We expect the RTX 8000 to lack native bfloat16, so π0.5 may need float32 there, which costs 1.65× on the Spark (0.411 s against 0.249 s per model call). If time runs short we drop the extra training data first, then cut gold labels and episodes per task, before cutting headline suites.

## 4. How we will know it worked

**Measurement.** Success within deadline: a decision counts only if the chosen chunk is good and arrives on time, with the policy's own overhead charged to the clock. On time means the new chunk arrives before the robot finishes executing the 10 actions it is running. On a fast local network π0.5's extra 0.16 s rarely misses that deadline, so the network sweep (added delay, jitter and loss) is what makes deadlines bind. We report success against mean and 95th-percentile latency per decision, per suite, with confidence intervals and paired comparisons on the same episodes.

**Baselines.** Always-local, always-remote, random escalation at the same rate as each policy, and an oracle that escalates exactly the decisions the small model gets wrong.

**It worked if** the best policy, on held-out tasks, loses at most 3 percentage points of success to always-remote (for example 77% if always-remote scores 80%), escalates at most half of the decisions, and beats random escalation at the same rate.

**The numbers that make us stop:**

- **Gap below 15 points** between the two models on every headline suite, after the fallbacks in section 2: there is nothing to route. With 100 episodes per model, about 14 points is the smallest difference we can tell from noise.
- **AUROC below 0.65** for every routing signal at predicting small-model failure on held-out tasks, after adding the extra training data: the signals cannot tell easy decisions from hard ones, and we report that as the finding.

## References

1. Shukor et al. SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. arXiv:2506.01844, 2025.
2. Black et al. (Physical Intelligence). π0.5: a Vision-Language-Action Model with Open-World Generalization. arXiv:2504.16054, 2025.
3. Liu et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310, 2023.
4. Fei et al. LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models. arXiv:2510.13626, 2025.
5. Sawada and Kasahara. Beyond Single-Axis Testing: Paired Evaluation of Compound Robustness in Vision-Language-Action Policies. arXiv:2609.15940, 2026.
6. Hugging Face. LeRobot documentation: LIBERO (π0.5 reproduction results). https://huggingface.co/docs/lerobot/libero
