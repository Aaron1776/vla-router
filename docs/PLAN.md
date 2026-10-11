# Edge-Cloud VLA Router: Implementation Plan

Oct 1, 2026 · @Oscar Rodriguez

## Overview

We build a router that lets a simulated robot run a small VLA onboard and escalate hard decisions to a large remote VLA, then measure when waiting for the bigger model is worth it. The core result is open-loop and due before Thanksgiving; the closed-loop demo and stretch goals sit on top.

**Research question:** given a deadline per decision, can a routing signal from the small model decide when to escalate, beating both always-local and always-remote on success within deadline?

**Deliverables (CSCI 5942):**

| Deliverable | Date | Points | What it contains |
| --- | --- | --- | --- |
| Concept document | Oct 13 | 5 | Two pages: build, data, compute arithmetic, stop number |
| Written feedback | Oct 20 | 0 | Last free chance to change the plan |
| Pitch day | Oct 27 | 5 | Four slides, five minutes, defend the numbers |
| Poster session | Dec 3 | 10 | Code, README, results, demo video |

**Success looks like:** a curve of success-within-deadline vs mean latency where the router sits above the line between always-local and always-remote, plus a closed-loop video of the robot escalating on a hard step.

## Models and benchmarks

The model pair is SmolVLA (local) and π0.5 (remote), evaluated headline-first on the hard LIBERO variants, because clean LIBERO no longer separates models. Updated Oct 10 after a review of 2026 sources.

**Selection rule.** A benchmark is usable for this project only if (1) both models have checkpoints on it, (2) there is a measurable success gap between them, and (3) the large model succeeds often enough that escalating pays off.

| Role | Model | Params | Why | Status |
| --- | --- | --- | --- | --- |
| Local (onboard) | SmolVLA (`lerobot/smolvla_base`, Apache-2.0) | \~0.5B | Small enough for a real onboard budget; supported by the eval harness | LIBERO-finetuned checkpoint not listed on the official card: find or fine-tune in Phase 0 |
| Remote (server) | π0.5 via openpi (`pi05_libero`) | 3.6B | Most robust on LIBERO-Plus among protocol-faithful models; built-in websocket policy server; PyTorch validated on LIBERO; >8 GB for inference | Checkpoint released |
| Remote, alternative | GR00T N1.7 | 3.4B | About a third of π0.5's latency in the same comparison | Supported by the eval harness |
| Remote, fallback | OpenVLA-OFT | 7B | 2025 model; use only if both above fail on our hardware | Checkpoints released |
| Teacher (exploratory) | RSI-Jev v6.0-VL | 4B | Calibrated escalation probabilities for distillation | Apache-2.0 weights; unaffiliated with TypeSafe |

Published scores for the remote candidates, from IndustrialVLA-Bench (Sep 2026, three-seed averages; latencies reflect implementation and hardware, not only the model):

| Model | Clean LIBERO | LIBERO-Plus | LIBERO-Para | Latency | Peak VRAM |
| --- | --- | --- | --- | --- | --- |
| π0.5 (openpi) | 96.52% | 85.31% | 71.33% | 58.8 ms | 14.6 GB |
| GR00T N1.7 | 97.88% | 75.12% | 74.26% | 19.1 ms | 7.8 GB |

| Benchmark | Role in the project | Why |
| --- | --- | --- |
| LIBERO-Long | Headline | Most discriminative clean suite |
| LIBERO-Plus | Headline | Visual, language and geometric perturbations; models spread out here |
| LIBERO-Para | Headline | Paraphrased instructions; reported drops of 22 to 52 points |
| LIBERO Spatial, Object, Goal | Controls | Saturated: six recent models sit within 1.58 points; shortcut-solvable per a 2026 audit |
| RoboCasa365 Atomic split | Stretch: harder, more credible check | Same robosuite/MuJoCo stack; far from saturated (GR00T N1.5 at 43.0% on Atomic) |

**Why not LIBERO alone:** a June 2026 audit of the five most-reported manipulation benchmarks found a \~0.09B vision probe with no language encoder within one point of the best LIBERO results on three of four suites, and only about 20% of LIBERO SOTA claims statistically significant. We cite it in the concept document and follow its reporting advice (see Evaluation protocol).

**Evaluation infrastructure:** AllenAI's vla-evaluation-harness runs a model server on a GPU host and the benchmark as a separate client, with a batch server that groups requests from parallel episodes. It supports SmolVLA, π0.5 and GR00T N1.7 and covers LIBERO, LIBERO-Plus and RoboCasa365. Our router sits between its client and server, so the contribution stays ours while the plumbing is reused.

## System architecture

Three parts talk over one interface: the robot (simulator, small VLA, router, executor), the network, and the server (request queue, large VLA). The same code runs on the cluster with an emulated network and on the Mac and Spark over a real one.

&#91;embedded content: system architecture · robot, network, server\]

At each chunk boundary the router either lets the executor run the local chunk or sends a request and waits, falling back if the reply misses the deadline.

**The shared interface** between Track A (simulation) and Track B (systems) is the decision record from Phase 1: observation in, action chunk plus latency and uncertainty out. Agree on its fields in week one so both tracks can build in parallel.

## Hardware and environments

The course cluster carries the reproducible experiments; the MacBook and DGX Spark carry the real-network demo. If either personal device fails its smoke test, the core result does not depend on it.

| Machine | Role | Memory | Known constraint |
| --- | --- | --- | --- |
| Course cluster, 4x RTX 8000 | Open-loop experiments, emulated network, partner access | 48 GB per GPU | Turing: no native BF16 or FP8 as far as we know; models must run in FP16 |
| MacBook M2 Max | Edge node: simulator, small VLA, router | 32 GB unified | No CUDA; PyTorch via MPS; LIBERO on macOS unverified |
| DGX Spark (GB10) | Server node: large VLA for the demo | 128 GB unified | SM121 support gaps seen before (PyTorch, flash-attention) |
| Google Cloud credits | Fallback GPUs if Turing or SM121 blocks a model | Varies | Costs credits; set a budget cap |

**Software stack (to confirm in Phase 0):**

- **Simulator:** LIBERO and its variants on robosuite and MuJoCo. LIBERO pins numpy below 1.24, robosuite 1.4.0, bddl 1.0.1 and mujoco 3.2.3, so it lives in its own environment.
- **Headless rendering on Linux:** set `MUJOCO_GL=egl`, which tells MuJoCo to render offscreen on the GPU without a display. macOS needs a different backend; to be determined.
- **Evaluation harness:** AllenAI vla-evaluation-harness. `vla-eval serve` runs a model server on a GPU host; `vla-eval run` runs the benchmark client, in Docker by default. Check whether the course cluster allows Docker; if not, use Apptainer or a native install.
- **Local model:** SmolVLA through LeRobot, \~0.5B, about 1 GB of weights in FP16.
- **Remote model:** π0.5 through openpi. Its policy server (`scripts/serve_policy.py`, port 8000 by default) streams actions to a client over a websocket.
- **Router placement:** a proxy between the benchmark client and the two model servers. It sees every observation, calls the local model, and decides whether to forward to the remote server.
- **Network emulation on Linux:** traffic control (`tc`) with its network-emulator module to add delay, jitter and loss on the proxy-to-remote link.

**Onboard budget (design choice):** the robot gets a fixed memory and per-decision latency budget, and the small model is what fits it. The Mac could likely hold the remote model too, so the split is a stated constraint, not a hardware accident.

## Phase 0: smoke tests (Oct 1 to Oct 12)

Phase 0 answers one question per machine: does each model run, how fast, and in how much memory. Its measured numbers become section 3 of the concept document.

- [ ] Install the vla-evaluation-harness on the cluster; confirm its model-server and benchmark-client split works there (Docker, Apptainer, or native)
- [ ] Create the LIBERO environment with its pinned versions; install LIBERO-Plus and LIBERO-Para
- [ ] Run one LIBERO-Long episode headless and save a rendered video
- [ ] Find a SmolVLA checkpoint fine-tuned on LIBERO; if none exists, estimate the cost of fine-tuning one
- [ ] Serve π0.5 (`pi05_libero`) with the openpi policy server on one RTX 8000; run one inference step from a client
- [ ] For each model, record peak GPU memory (`torch.cuda.max_memory_allocated()`) and median latency over 100 steps; for SmolVLA, also time k samples run as one batch vs one after another
- [ ] **Measure the gap first:** SmolVLA vs π0.5 success on LIBERO-Long, LIBERO-Plus and LIBERO-Para (a few episodes per task). This decides whether the router has anything to exploit
- [ ] Check FP16 on Turing: clean-suite success should land near published numbers
- [ ] Mac: install LIBERO and SmolVLA, run one inference step on MPS, record latency
- [ ] Spark: serve π0.5 with the openpi server, run one inference step, record latency
- [ ] Measure round-trip time between Mac and Spark on the network the demo will use
- [ ] Optional: load RSI-Jev v6.0-VL in FP16 and run one query
- [ ] Read two papers on edge-cloud or cascaded inference; note one baseline we can borrow
- [ ] Agree with partner on the decision-record fields (the interface between Track A and Track B)

**Exit criteria:** a measured SmolVLA vs π0.5 gap of at least \[\_\_\] points on at least one headline suite, plus latency and memory for both models. No gap anywhere: switch the remote to GR00T N1.7, try a weaker local model, or move to RoboCasa365 Atomic, before Oct 20.

## Phase 1: open-loop (Oct 13 to Nov 25)

Phase 1 produces the guaranteed result: a labeled set of decision points harvested from LIBERO, a routing signal scored against those labels, and a success-within-deadline curve under emulated network conditions.

**1. Harvest decision points.** Run full episodes of the small model on LIBERO-Long, LIBERO-Plus and LIBERO-Para, plus the clean suites as controls. At every action-chunk boundary, save a decision record:

| Field | Contents |
| --- | --- |
| Simulator state | Full MuJoCo state, restorable (verify restore works in Phase 0) |
| Observation | Camera frames, robot proprioception, instruction text |
| Context | Suite, task id, episode id, step index |
| Local output | Small model's action chunk + its uncertainty features |
| Remote output | Large model's action chunk from the same observation |
| Measured latency | Local inference time, remote inference time |

**2. Label each decision.** Two labeling methods, decided in week one:

- **Gold (expensive):** restore the state, branch twice (local chunk vs remote chunk), finish both branches with the large model, compare episode success. Local is "good enough" if its branch succeeds when the remote branch does.
- **Proxy (cheap):** local is "good enough" if its chunk is within a distance threshold of the remote chunk. Validate the proxy against gold labels on a subset before trusting it.

**3. Build four escalation policies (a 2×2).** Two questions define each policy: when it decides, and how. All four are trained and scored on the same decision records, so the extra arms cost analysis time, not new rollouts.

|  | Programmatic (hand-written rule) | Learned (classifier on labels) |
| --- | --- | --- |
| **Router:** decides before running the small model, from observation and context only | Stall heuristic, unfamiliar-input check | Classifier on context features; can skip the small model on hard steps |
| **Cascade:** runs the small model first, then inspects its output | Threshold on output confidence or sampling spread | Classifier on output features + context |

The router arms ignore the local-output fields of each record. The cascade arms pay for one local attempt on every decision, so a cascade that escalates often wastes that attempt; the router avoids it but sees less.

**Signals available to these policies,** cheapest first, because a policy's own cost counts against the deadline:

1. **Free signals (one pass, no extra inference):** stall heuristic (no progress or gripper change for N chunks); output confidence such as token entropy, if the small model exposes probabilities; an unfamiliar-input check comparing the current internal representation to easy steps.
2. **Paid signal: sampling spread.** Run the small model k times (batched where possible, the first sample doubling as the action) and measure action variance. Works only if action generation is stochastic or noise is injected; a deterministic model gives zero spread. Cascade arms only.
3. **Learned classifiers:** logistic regression or a small MLP on the signals above + context, trained on the labels, tested on held-out tasks.

**Signal cost vs signal quality.** For every signal, record its overhead in milliseconds and its AUROC for predicting local failure, and plot one against the other. A paid signal earns its place only if it beats the free ones by enough to cover its overhead; if k samples cost close to an escalation, escalating is the better use of the time.

**4. Score under a simulated clock.** Each decision gets a deadline. Local cost = local inference + router overhead (extra samples or checks). Remote cost = router overhead + network round trip + remote inference + queueing. Router overhead is never treated as free. A decision scores as correct only if it is good and on time. Sweep the network grid (round-trip delay, jitter, loss) using `tc` on the cluster.

**5. Avoid overfitting the router.** Tune thresholds on some LIBERO tasks, report on held-out tasks.

## Phase 2: closed-loop (Nov 9 to Dec 2)

Phase 2 turns the Phase 1 router on inside live episodes, first on the cluster with emulated network, then on the Mac and Spark over a real network for the poster video. It starts while Phase 1 analysis finishes, so it overlaps by two weeks.

**Runtime behavior at each chunk boundary:**

1. The robot computes the local action chunk and the routing signal.
2. Signal below threshold: execute the local chunk.
3. Signal above threshold: send the observation to the server and keep executing a safe fallback (hold position, or continue the local chunk) while waiting.
4. Remote reply arrives before the deadline: execute it. Reply late or lost: execute the fallback and log a timeout.

**Experiments:**

| Run | Where | Network | Purpose |
| --- | --- | --- | --- |
| Closed-loop on cluster | Cluster | Emulated grid from Phase 1 | Does the open-loop ranking of routers hold when errors compound? |
| Real-network demo | Mac + Spark | Actual Wi-Fi or campus network | Validation on two physical machines, poster video |
| Local π0.5 baseline (optional) | Mac | None | Is the big model too slow to meet the deadline onboard? |

**Videos:** render episodes with an on-screen marker when the robot escalates and when a remote reply lands, so a viewer sees the decisions happen.

**Exit criteria:** end-to-end task success for all three policies (always-local, always-remote, router) on at least one suite, and one rendered video per policy.

## Stretch goals

Stretch goals start only after the open-loop result is locked, in this order, because each reuses what Phases 1 and 2 built.

1. **RoboCasa365 Atomic split.** Same robosuite/MuJoCo stack, far from saturated, and it fails fewer of the audit's diagnostics than LIBERO. Checks whether the router's advantage holds on a harder, more credible benchmark. Needs SmolVLA and π0.5 checkpoints for it; we have not found released ones yet.
2. **Fleet.** Several simulated robots share one remote server. Queueing delay rises with load, so the router should escalate less as the server gets busier. Measures success vs number of robots, with and without a load-aware threshold. The harness's batch server already groups requests from parallel episodes.
3. **Degraded networks in closed-loop.** Push the emulated network past where remote help arrives on time, and show the robot falls back gracefully instead of stalling.

### Exploratory arm: Jev as teacher

Jev is TypeSafe's decision model: it takes a structured state plus named questions and returns typed answers, including yes/no with a calibrated probability, which is exactly the escalation question. Jev itself is proprietary with no published weights; we use **RSI-Jev v6.0-VL**, an unaffiliated open project (4B parameters on Qwen3.5-4B-Base, Apache-2.0 weights, benchmarks self-reported). Open-loop only.

At 4B parameters it is larger than SmolVLA, so it cannot be the onboard router. It runs offline as a **teacher**, and only a small distilled student runs on the robot.

1. **Zero-shot gate.** Convert each decision record to Jev's structured-state format, ask "does this step need the remote model?", and score its probability against gold labels on held-out tasks. If AUROC is below the stop number, report that and stop here.
2. **Label at scale.** Run the teacher over every harvested record, including the many with no gold label, to get a soft escalation probability for each.
3. **Distill.** Train the same small student architecture as the learned router twice: on our labels only, and on teacher soft labels (plus our labels where they exist).
4. **Compare** both students, the zero-shot teacher, and the 2×2 arms on AUROC, success within deadline, and overhead.

The question it answers: *does a general calibrated decision model's judgment beat a task-specific classifier trained on our own labels, once its cost is counted?* Use RSI-Jev, not the TypeSafe API, as the teacher: we have not checked TypeSafe's terms on training models from its outputs.

Out of scope: Isaac Sim or Isaac Lab. The LIBERO checkpoints are tuned to LIBERO's scenes and cameras, so a new simulator means new demonstrations and re-fine-tuning both models.

## Evaluation protocol

The headline metric is success within deadline vs mean latency, compared against three baselines, with one stop number that ends the idea if missed.

| Item | Definition |
| --- | --- |
| Primary metric | Success within deadline: open-loop = fraction of decisions that are good and on time; closed-loop = task success |
| Cost metric | Mean and 95th-percentile latency per decision, including router overhead |
| Secondary | Fraction escalated; overhead vs AUROC for each routing signal; success as network quality drops |
| Baseline: always-local | Small model on every decision |
| Baseline: always-remote | Large model on every decision, paying network and remote latency |
| Baseline: oracle router | Escalates exactly the decisions the local model gets wrong (upper bound) |
| Stop number | Routing-signal AUROC below \[\_\_\] on held-out tasks: the signal cannot tell easy decisions from hard ones |
| Win condition | Router beats always-remote on latency while losing at most \[\_\_\] points of success to it |

**Statistics:** report each number with a confidence interval over episodes or decisions, and fix random seeds per suite. With LIBERO's 10 tasks per suite, a task-level held-out split is small, so prefer leave-tasks-out cross-validation.

**Also report the gap first.** Before any routing, measure always-local vs always-remote success per suite on our hardware. If the gap is near zero on a suite, the router has nothing to exploit there, and that suite becomes a control.

**Reporting, following the 2026 benchmark audit:** save per-episode and per-decision outcomes so paired tests can compare policies on the same instances; state in advance the gap that counts as a real improvement; and report results per suite, never only a pooled LIBERO average.

## Compute arithmetic

Section 3 of the concept document needs one multiplication from measured throughput; this is the template, with every input to be filled from Phase 0 measurements.

Total GPU time is the number of model inferences times the measured time per inference, summed over models:

```latex
\text{GPU-hours} = \frac{\sum_{m} N_m \cdot t_m}{3600}
```

where N is the number of inferences for model m and t is its measured seconds per inference.

| Input | Symbol | Value | Source |
| --- | --- | --- | --- |
| Suites used | S | 3 headline (Long, Plus, Para) + 3 clean controls | LIBERO |
| Tasks per suite | T | \[\_\_\] | LIBERO docs |
| Episodes per task | E | \[\_\_\] | Our choice |
| Chunks per episode (mean) | C | \[\_\_\] | Measured in Phase 0 |
| Small-model samples per decision (for sampling spread) | k | \[\_\_\] | Our choice |
| Small-model seconds per inference | t\_small | \[\_\_\] | Measured in Phase 0 |
| Large-model seconds per inference | t\_large | \[\_\_\] | Measured in Phase 0 |
| Gold-label branch rollouts (subset) | B | \[\_\_\] | Our choice |
| Network conditions in the sweep | G | \[\_\_\] | Our choice |

**How the counts combine:** decisions D = S × T × E × C. Harvesting costs D × k small inferences plus D large inferences. Gold labels cost B branch rollouts, each about C large inferences. The network sweep reuses recorded latencies, so it adds almost no GPU time; closed-loop runs add their own episodes on top.

**Budget rule:** multiply the total by 2 for reruns and debugging, then check it against the cluster share we can realistically get. If it does not fit, cut E or B first, not suites.

## Schedule and work split

Phase 1 runs from the concept document to Nov 25, Phase 2 overlaps its last two weeks, and the poster is Dec 3. Nov 25 is our own milestone, not a course deadline.

&#91;embedded content: schedule · course deadlines and internal milestones\]

The two tracks build in parallel and meet at the decision-record interface.

| Weeks | Track A: simulation | Track B: systems | Together |
| --- | --- | --- | --- |
| Oct 1 to Oct 12 | LIBERO environment, first rendered episodes, state restore check | Load both VLAs, measure latency and memory, Mac and Spark smoke tests | Two papers, interface fields, concept document |
| Oct 13 to Oct 26 | Harvest decision records across suites | Router proxy between harness client and model servers, `tc` network emulation | Pitch slides, compute numbers |
| Oct 27 to Nov 8 | Gold and proxy labels | Routing signals, simulated-clock scoring | First success vs latency curve |
| Nov 9 to Nov 25 | Closed-loop harness, escalation overlay on videos | Network sweep, learned router if needed | Lock the open-loop result |
| Nov 26 to Dec 3 | Real-network demo videos on the Mac | Server on the Spark for the demo | Poster, README, code cleanup |

Thanksgiving falls on Nov 26, so the final week is short; nothing on the critical path should land there.

## Risks and decision points

Every risk below has a fallback that keeps the open-loop result intact; the decision dates are when we stop and choose.

| Risk | Signal | Decide by | Fallback |
| --- | --- | --- | --- |
| No usable SmolVLA vs π0.5 gap | Gap below the exit threshold on all headline suites | Oct 20 | GR00T N1.7 as remote, a weaker local model, or RoboCasa365 Atomic |
| No SmolVLA checkpoint for LIBERO | Nothing found in Phase 0 | Oct 12 | Fine-tune SmolVLA on LIBERO demos (small model, single GPU) |
| A model fails or underperforms on Turing in FP16 | Clean-suite success far below published numbers | Oct 12 | Swap that model, or run it on cloud credits |
| Harness needs Docker; cluster disallows it | Docker unavailable on the course cluster | Oct 12 | Apptainer, or run the benchmark client natively |
| LIBERO will not run on the Mac | Install or rendering errors on macOS | Oct 12 | Mac becomes a thin client; simulator stays on Linux |
| π0.5 server will not run on the Spark | SM121 errors or very slow inference | Oct 12 | Remote served from the cluster for the demo |
| Simulator state restore is unreliable | Replayed state diverges from the original | Oct 20 | Proxy labels only, validated on fresh rollouts |
| Routing signal is weak | AUROC near the stop number | Nov 10 | Learned policies; if still weak, report it as the finding |
| RSI-Jev will not run or is poor zero-shot | Load errors in FP16, or AUROC below the stop number | Nov 10 | Drop the Jev arm; report the zero-shot result if we have it |
| Closed-loop runs long | Phase 2 not working by Nov 25 | Nov 25 | Open-loop result stands alone; video from a few episodes |
| LIBERO dependency conflicts | Version clashes with model code | Oct 6 | Separate environments for benchmark and models, connected through the harness |

## Open questions

These need answers before Oct 13; most are settled by Phase 0 runs or one conversation with the partner.

- [ ] Which SmolVLA LIBERO checkpoint, and does one exist for LIBERO-Plus and LIBERO-Para conditions?
- [ ] π0.5 or GR00T N1.7 as the remote, once latency is counted? Decide from Phase 0 measurements
- [ ] Does the small model expose anything usable as uncertainty, or is sampling spread the only option?
- [ ] How is the per-decision deadline set: a fixed time, or tied to the chunk's execution time?
- [ ] Gold labels, proxy labels, or both? Decide after measuring the cost of one branch rollout
- [ ] Phase 0 gap threshold, AUROC stop threshold and the win-condition margin (the \[\_\_\] values)
- [ ] Router features: which context features can the before-acting router see without running the small model?
- [ ] RSI-Jev: does v6.0-VL load in FP16 on the RTX 8000s, does it take camera frames or only structured state, and how should a decision record be written in its format?
- [ ] Router placement: a proxy between the harness client and both model servers, or inside the client?
- [ ] Which network the real demo uses (home Wi-Fi, campus, or Tailscale between Mac and Spark)
- [ ] Partner's track choice (A: simulation, B: systems)
- [ ] Course cluster share: how many GPU-hours per week can we realistically get, and is Docker or Apptainer allowed?

## Sources

- [IndustrialVLA-Bench (arXiv 2609.25562)](https://arxiv.org/html/2609.25562)
- [What Are We Actually Benchmarking in Robot Manipulation? (arXiv 2606.04233)](https://arxiv.org/html/2606.04233v1)
- [LIBERO-Para (arXiv 2603.28301)](https://arxiv.org/abs/2603.28301)
- [RoboCasa365 (arXiv 2603.04356)](https://arxiv.org/pdf/2603.04356)
- [allenai/vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness)
- [openpi README](https://cdn.jsdelivr.net/gh/Physical-Intelligence/openpi/README.md)
- [SmolVLA base model card](https://huggingface.co/lerobot/smolvla_base)
- [LIBERO dependency pins (RoboVerse integration)](https://roboverse.wiki/_sources/dataset_benchmark/integrations/libero.md.txt)
- [OrcaRouter: Jev open source](https://www.orcarouter.ai/vi/blog/jev-open-source)
