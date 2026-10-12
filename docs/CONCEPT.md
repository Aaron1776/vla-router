# Concept document: an edge-cloud router for robot action models

**Working draft, 2026-10-12. Due Oct 13 on Gradescope, two pages.**

CSCI 5942, Mega Assignment. Team (pair): Oscar Rodriguez and Miguel Ramirez-Aleman. Direction: A6, a router over two models.

Numbers marked *(proposed)* are placeholders for the two of us to confirm. Everything marked *measured* was run by us; the evidence is in [ASSUMPTIONS.md](ASSUMPTIONS.md).

## 1. What we will build

A router that sits between a simulated robot and two vision-language-action models (VLAs): a small one onboard (SmolVLA, 450M parameters) and a large one on a remote server (π0.5, 4.1B parameters). Every 10 control steps the robot needs a new chunk of actions. The router decides, under a deadline, whether to use the small model's chunk or pay the network round trip for the large model's.

Concretely we build three things:

1. **A routing model server** for AllenAI's `vla-evaluation-harness`. It owns the action-chunk buffer, calls both models, applies an escalation policy, and logs every decision.
2. **A decision-record dataset**, harvested by running that server on LIBERO manipulation tasks: observation, both models' chunks, measured latencies, and a label for whether the small model's chunk was good enough.
3. **Four escalation policies and a scorer.** The policies form a 2x2: decide before running the small model (router) or after inspecting its output (cascade), each as a hand-written rule or a learned classifier. The scorer replays the records under a simulated clock with network delay, jitter and loss.

## 2. On what data

We do not train a VLA. Our data is what the simulator and the two released checkpoints (`lerobot/smolvla_libero`, `lerobot/pi05_libero_finetuned`) produce.

- **LIBERO-10** (long-horizon suite): 10 tasks x 50 fixed initial states = 500 episodes. With a 520-step limit and 10 actions per chunk, that is at most 52 decisions per episode, so at most 26,000 decision records.
- **LIBERO-Plus**, the long-horizon variants: 2,519 perturbed episodes (camera, lighting, layout, language and other shifts), at most 131,000 records.
- **Controls:** the Spatial, Object and Goal suites, where recent models are saturated and we expect little to route.

Labels come from the simulator. For a subset of 2,000 decisions *(proposed)* we restore the saved simulator state, execute each model's chunk, finish both branches with the large model, and compare task success. A cheaper proxy label (distance between the two chunks) is validated against that subset before we use it on the rest.

**If it is not enough.** The whole project depends on the small model failing where the large one succeeds. We have not measured that gap yet; it is the first experiment. If the gap is below 15 points *(proposed)* on both headline suites by Oct 20, we switch the remote model to GR00T N1.7, or move to the harder RoboCasa365 tasks, before building further. If exact simulator state restore fails, we use proxy labels only, validated on fresh rollouts.

## 3. On what compute

**Machines.** An NVIDIA DGX Spark (GB10, 128 GB) as the server, a MacBook M2 Max as the robot-side node, and the course cluster (4x RTX 8000) for parallel runs.

**Measured throughput** (Spark, one 50-action chunk, real instructions through each model's full preprocessing, median of 100 calls, uncompiled):

| Model | Time per decision | Peak GPU memory |
| --- | --- | --- |
| SmolVLA | 0.107 s | 1.0 GB |
| π0.5 (bfloat16) | 0.268 s | 9.5 GB |
| SmolVLA, 8 samples in one batch (for a spread signal) | 0.236 s | 1.2 GB |

**Arithmetic.** Decisions D = 26,000 + 131,000 = 157,000 (upper bound; successful episodes end early).

| Step | Calculation | GPU-hours |
| --- | --- | --- |
| Harvest: one small and one large call per decision | 157,000 x (0.107 + 0.268) s | 16.4 |
| Extra small-model samples for the spread signal | 157,000 x (0.236 - 0.082) s | 6.7 |
| Always-remote baseline episodes | 157,000 x 0.268 s | 11.7 |
| Gold labels: 2,000 decisions x 2 branches x 26 remaining large calls | 104,000 x 0.268 s | 7.7 |
| **Total** | | **42.5** |
| **With 2x for reruns and debugging** | | **85** |

That is about 3.5 days of the Spark running continuously. Two caveats: simulator stepping time is not in this number (we measure it this week), and the RTX 8000 throughput is unmeasured, so the cluster share is an extrapolation. If the budget does not fit, we cut gold labels and episodes per task before cutting suites.

## 4. How we will know it worked

**Measurement.** Success within deadline: a decision counts only if the chosen chunk is good *and* arrives on time, with the policy's own overhead charged to the clock. We report it against mean and 95th-percentile latency per decision, per suite, with confidence intervals and paired comparisons on the same episodes.

**Baselines.** Always-local, always-remote, random escalation at the same escalation rate as each policy, and an oracle that escalates exactly the decisions the small model gets wrong.

**It worked if** the best policy, on held-out tasks, loses at most 3 points *(proposed)* of success to always-remote while escalating at most half of the decisions *(proposed)*, and beats random escalation at the same rate.

**The numbers that make us stop:**

- **Gap below 15 points** *(proposed)* between the two models on every headline suite, after the fallbacks in section 2. There is nothing to route. With 100 episodes per model, about 14 points is the smallest difference we can tell from noise.
- **AUROC below 0.65** *(proposed)* for every routing signal at predicting small-model failure on held-out tasks. The signals cannot tell easy decisions from hard ones, and we report that as the finding.

---

## Open items before submitting (not part of the two pages)

- [ ] Confirm or change the five *(proposed)* numbers: gap threshold (15 points), AUROC stop (0.65), win margin (3 points), escalation cap (half), gold-label subset (2,000).
- [ ] Confirm 10 actions per decision for both models. π0.5's LIBERO protocol uses 10; SmolVLA's default is 50.
- [ ] How the deadline is set. On a fast local network the large model's extra 0.16 s may rarely miss a deadline, in which case the network sweep (added delay and loss) carries the result.
- [ ] Cluster facts: container support, scheduler, and our realistic share of GPU-hours.
- [ ] LIBERO-Para is in PLAN.md as a headline suite but has no integration in the harness or LeRobot. This draft leaves it out; decide whether it is a stretch goal.
- [ ] Cite the two papers the course asks for.
- [ ] Trim to two pages in the submission format.
