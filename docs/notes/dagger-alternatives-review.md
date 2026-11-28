# Alternatives to DAgger for the escalation deciders: literature review

Written 2026-11-27 (night, -07:00) by a side session, for the main session and Oscar. Read-only on the repo. The Spark was not touched. Nothing ran on a GPU. Not committed.

What I did: read the five notes and SPEC §6 to §10 with amendments 55, 63, 65 and 78 (`origin/harness-v1`), the datacard, and the sibling notes on the signal study and the denoising signals; verified every cited paper by web search; ran one small CPU check on the pilot table (section 4).

**Added later the same night, at Oscar's request: neurosymbolic options.** Section 3.10, a second check in section 4, and changes to the shortlist. For it I read nine papers from Oscar's neurosymbolic NLP folder (listed in 3.10) and verified six more by search.

## Labels

- **[V]** verified: I ran it or looked it up in this session.
- **[S]** from a source I read (named), not re-run.
- **[A]** assumption or my own reasoning. Every "chance" figure is [A].
- **[U]** unknown.

For papers, [V] means a search in this session returned the paper's own page (arXiv or proceedings) under that title and ID. I read abstracts and search summaries only, not full papers. [S-design] means `cascader-design.md` verified it on 2026-11-23 and I did not repeat the check.

## 1. Bottom line

1. **No published method is a likely substitute for DAgger here, because DAgger's job is not the missing piece.** DAgger fixes the state distribution of a decider that already ranks well. Ours rank benefit at chance on held-out tasks. [S, `2x2-learning-curve.md`] The same argument rules out most of the "more or better training" family.
2. **The literature splits into two groups, and neither fits our gap.**
   - Failure and uncertainty monitors (SAFE, FAIL-Detect, FIPER, Sentinel) predict "this episode is failing". We can already do that from step 170 on (held-out AUROC 0.76 to 0.81 in my check), but benefit sits before step 170, where the same predictor scores 0.55 to 0.61. [V, section 4]
   - Deferral, cascade, uplift and off-policy methods learn "where does the big model help". They assume the inputs carry that information. Ours expose "early" (a clock) and not "this scene will fail". [S, `2x2-learning-plan.md`]
3. **One idea is new to the plan and costs no GPU: treat the random-escalation runs as a micro-randomized trial.** The exploration screens (1,974 episodes, escalation probability 0.1) and the planned random baselines randomize the escalation at every decision point with a known probability. That gives an unbiased estimate of the effect of one call by step and by state feature, on states that follow earlier escalations. It is the closest thing to what DAgger's data would have measured.
4. **My estimate that any learned decider clearly beats "escalate earliest first" on held-out tasks by Nov 30: about 15%.** [A] The plan in `amendment-followup-draft.md` (matched baselines, timing baselines, positive control) is still the right use of the GPU. The shortlist in section 5 adds CPU work beside it and at most one optional live run.

5. **Neurosymbolic methods do not change this, but they sharpen it.** Logic rules can only inject what we already know, and what we know is the clock. A rule-constrained decider should tie earliest-first instead of losing to it, which the MLPs do. The one neurosymbolic idea with new information is a symbolic state (LIBERO's goal predicates), and it is in the snapshots, not in what a live decider sees. Hand-written failure rules point the wrong way: after a failed grasp a call helps less, not more (section 4). [V]

## 2. What signals exist in our data and logs

| source | size | what it holds | missing |
| --- | --- | --- | --- |
| Branch-labelled points (L4) | 1,390 points, 1,296 used; about 600 episodes [S] | Every `points` column (embeddings, robot state, proposal, free signals, K=5 chunk spread, fresh spread, drift, history features) plus b1 and b2 with 3 seeds each [V, `points.npz`] | Nothing; it is small |
| Base rollouts, record `full` | 500 episodes on the L4 (102 fail), 500 on the Spark (97 fail, 15,671 decision points) [S] | Every `points` column at every decision point, snapshots, and a free label: the episode's own outcome is one seed of "continue" [S, SPEC §7] | No b2 outcome |
| Exploration screens, record `off` (L4) | 1,974 episodes, random escalation with probability 0.1 at each decision point [S, amendments 56, 67] | `runs`, `steps` (action, end-effector pose, fingers per step, driver) and `events` (decision points, `can_escalate`, decisions, π0.5 requests) [S, datacard] | No embeddings, no `points` rows. Whether these tables came over in the handoff is unknown to me [U]: amendment 66 (5) says training exports keep the record-`full` row |
| Exploration re-runs, record `full` | about 320 episodes [S] | As base rollouts, with escalations | Already the source of batch B's labels |
| Live round 1, record `off` (Spark) | 2,000 episodes, four deciders [S] | `runs`, `steps`, `events` | No embeddings. Decisions are a deterministic threshold on a score, so there is no randomization. Whether the score itself is logged: [U] |
| Planned random baselines (a) and (d) | 500 episodes each [S, draft amendment] | As round 1, plus `can_escalate` and budget at every point, with escalation drawn from a recorded stream at a known rate | Not run yet |

**Symbolic state.** The `points` rows hold no task predicates or object states. [V, datacard column list] Each snapshot holds MuJoCo's full state and LIBERO's Python-side object states, and the harness builds each env from the task's BDDL file, which defines the goal predicates. [S, SPEC §7; V, `sim/libero_sim.py:109`] So predicates can be computed offline by restoring snapshots on CPU [A, not tried]. The Spark has its own 15,671 base snapshots; the L4 snapshots (handoff part 3) are not on hand. [S] A live decider cannot see any of it: `observe()` is frozen and §6 lists no simulator state. [S]

Low-dimensional features that need no embeddings (all memory features, tracking, gripper events, executed commands) can be rebuilt from `steps` and `events` alone. [A, from the datacard's column list; not tried]

Already tested here and weak, so I do not propose them again [S, `cascader-signal-study.md`, `perstep-denoising-signal.md`]:
- K=5 chunk spread, fresh spread and drift: 0.54 to 0.57 step-stratified for failure, and +0.000 to +0.008 on top of the cheap signals.
- Per-iteration denoising signals: at most 0.61 step-stratified, 0.56 within task.
- Spread profiles cost more clock time than they return (amendment 63). [S]

## 3. Candidates by family

Reference to beat, from `2x2-learning-plan.md` [S]: "earliest first" has benefit AUROC 0.63, within-third AUROC 0.60, gain lens +0.0006 at 10% and +0.0078 at 20%. The unobservable "true parent outcome, then earliest" has 0.77 and +0.0208. Shuffle standard deviations: 0.0034, 0.0044, 0.04.

### 3.1 Micro-randomized analysis of the random-escalation runs

- **Papers.** Boruvka, Almirall, Witkiewitz, Murphy, "Assessing Time-Varying Causal Effect Moderation in Mobile Health", JASA 2018, [arXiv:1601.00237](https://arxiv.org/abs/1601.00237) [V]. It defines the "causal excursion effect" of a treatment randomized at each of many decision times and gives the weighted and centered least-squares estimator for how state features moderate it. Related: Nie, Brunskill, Wager, "Learning When-to-Treat Policies", JASA 2021, [arXiv:1905.09751](https://arxiv.org/abs/1905.09751) [V].
- **Signal.** The randomized escalation indicator at each decision point, the known probability, the episode outcome, and any feature computable up to that point.
- **In our data?** Yes for the planned baselines (a) and (d). Yes for the exploration screens if their `steps` and `events` tables are reachable [U]. Low-dimensional features only.
- **Size.** [A, arithmetic] 1,974 screens at roughly 30 to 50 offered points each and probability 0.1 give several thousand randomized escalations, a third or more of them early. A main effect of one early call on final success of about +0.05 (the +0.25 at failing points times a 20% failure rate) would stand 3 to 4 standard errors out. Moderation by a feature is weaker: expect to resolve "early against late" and perhaps one stall feature, not a full ranking.
- **Cost.** CPU minutes. No GPU. No new runs.
- **Chance.** As a measurement of timing on independent, partly on-policy data: about 70% that it shows a clear early effect, given the logs exist. As a route to a decider that beats earliest-first: about 10%, because branch labels are paired and much more efficient per point than whole-episode outcomes.
- **Why it is the nearest substitute for DAgger.** Its states come after earlier escalations, and the effect it estimates is "one more call, then the logged policy continues", which is what a deployed decider faces. Branch labels measure "one call, then X-VLA alone". [S, SPEC §8]
- **Status.** On the exploration screens: exploratory. On the planned (a) and (d) runs it can be pre-registered for free by adding one sentence to the draft amendment before they start: the estimand (effect of an escalation on success, by step third), the estimator, episode-clustered intervals.

### 3.2 Failure hazard from the free labels, times a clock prior

- **Papers.** Gu et al., SAFE, [arXiv:2506.09937](https://arxiv.org/abs/2506.09937) [S-design; V by search]: a failure detector on VLA internal features, trained on successful and failed rollouts, tested on unseen tasks, thresholded with functional conformal prediction. Gensheimer and Narasimhan, "A scalable discrete-time survival model for neural networks", PeerJ 2019, [arXiv:1805.00917](https://arxiv.org/abs/1805.00917) [V], for the hazard form. Hoque et al., ThriftyDAgger, CoRL 2021, [arXiv:2109.08273](https://arxiv.org/abs/2109.08273) [V], gates on novelty plus a learned risk of failure under a query budget.
- **Idea.** Benefit is roughly "probability the episode fails" times "probability a call still rescues it". The second factor is a clock (+0.25, +0.10, +0.06 by third). [S] Train the first factor on every base decision point (about 31,000 with the Spark's, against 1,296), as a discrete-time hazard of success so that the time confound is handled, and multiply.
- **In our data?** Yes. The Spark's 15,671 points with features are already extracted for the signal study. [S]
- **Cost.** CPU minutes to hours. No GPU unless it is run live.
- **Chance it beats earliest-first: about 15%.** The whole question is whether failure is predictable before step 170 on a held-out task. On the 1,296 labelled points it is not: 0.55 to 0.61 (section 4). The signal study's 0.70 is over all steps, and its gain came from motion history, which is neutral at the start of an episode. [S] More data may lift the early figure; the learning curve for the linear cascader says slowly. [S]
- **A use that does not need the early signal.** From step 170 failure is predictable (0.76 to 0.81) and a call still adds about +0.10 at failing points. [V, S] A rule "after step 170, escalate only when the failure score is high" could beat step-matched random in the middle third. Small prize, but it is the one place a learned signal demonstrably exists.
- **Status.** Offline: exploratory. Live: it would be the form of the optional lite decider in item (6) of the draft amendment, and needs to be fixed there before training.

### 3.3 Monitors that need no branch labels: OOD scores, conformal bands, progress estimators

- **Papers.** Xu et al., FAIL-Detect, [arXiv:2503.08558](https://arxiv.org/abs/2503.08558) [S-design; V by search]: scores learned from successful rollouts only, time-varying conformal thresholds. Römer et al., FIPER, NeurIPS 2025, [arXiv:2510.09459](https://arxiv.org/abs/2510.09459) [V]: an observation-embedding OOD score and an action-chunk entropy score, both calibrated by conformal prediction on a few successful rollouts. Agia et al., Sentinel, [arXiv:2410.04640](https://arxiv.org/abs/2410.04640) [S-design]: temporal consistency of successive action chunks. Lee et al., Mahalanobis score, NeurIPS 2018, [arXiv:1807.03888](https://arxiv.org/abs/1807.03888) [V]. Sun et al., deep nearest neighbours, ICML 2022, [arXiv:2204.06507](https://arxiv.org/abs/2204.06507) [V]. Ma et al., "Vision Language Models are In-Context Value Learners", ICLR 2025, [arXiv:2411.04549](https://arxiv.org/abs/2411.04549) [V], for progress estimation. Angelopoulos et al., "Conformal Risk Control", ICLR 2024, [arXiv:2208.02814](https://arxiv.org/abs/2208.02814) [V], for the threshold.
- **Signal.** Distance of the current embedding to states of successful rollouts at the same step; chunk-to-chunk consistency; a progress score.
- **In our data?** Embeddings and K=5 samples: yes, in every record-`full` rollout. Live: embeddings yes (the router profile pays for them); sample-based scores only under the spread profiles.
- **Cost.** CPU minutes for a kNN or Mahalanobis score on stored embeddings.
- **Chance: about 10% on held-out tasks.**
  - These methods calibrate on successful rollouts of the same task. FIPER and FAIL-Detect say so. [V, abstracts] Under leave-one-task-out there are none, and a new task's embeddings are far from every other task's by construction. [A]
  - Within a task they would work as scene memory, which the positive control already covers.
  - The action-uncertainty half is the spread family, already measured as weak here. [S]
  - A progress estimator needs the same task's successes too. Its transferable part, "the arm has stalled", is already in the history features. [A]
- **What is worth taking.** The threshold: a per-step-third quantile on calibration data in place of the "95% of maximum gain" rule, which picked noise. [S, second opinion] The draft amendment's fixed 20% rate already does this in simpler form.
- **Status.** Exploratory. I would not run it.

### 3.4 Scene recurrence: episodic memory and retrieval

- **Papers.** Pritzel et al., "Neural Episodic Control", ICML 2017, [arXiv:1703.01988](https://arxiv.org/abs/1703.01988) [V]. Sun et al. (above) for the nearest-neighbour machinery.
- **Signal.** The outcome of earlier rollouts in the nearest stored scenes, retrieved by embedding at step 0.
- **In our data?** Yes. The pilot's "scene memory" input reached AUROC 0.73 and +0.017 at 20% with seen scenes, and failure repeats by scene (60% against 8%). [S]
- **Cost.** CPU minutes offline. One live run (about 3 to 5 GPU hours) if tested live.
- **Chance.** About 50% to beat earliest-first under a "seen scenes" protocol; zero on held-out tasks by construction. [A]
- **Is it legitimate?** As a decider with a memory of past deployments in the same scenes: yes, and that is a real deployment setting. As an answer to the registered question: no. §6 forbids task identity as an input for exactly this reason. [S] A retrieval decider that looks up by embedding, not by scene index, is the honest version: it needs no identity input, and it should be evaluated with the evaluation rollout seeds held out, as the positive control is.
- **Status.** The draft amendment's item (4) is the privileged version. A retrieval decider would be a new pre-registered item. It mostly re-measures (4), so I rank it below the shortlist.

### 3.5 Online adaptation within a task

- **Papers.** Nie et al., "Online Cascade Learning for Efficient Inference over Streams", ICML 2024, [arXiv:2402.04513](https://arxiv.org/abs/2402.04513) [V]: the deferral rule is learned online from the large model's outputs as the stream arrives.
- **Idea.** Over the 50 episodes of a test task, track X-VLA's running failure rate and scale early escalation with it. No scene identity; only outcomes seen so far.
- **In our data?** Offline it can only be approximated. X-VLA-alone failures on the evaluation seeds are 20, 12, 5, 3, 16, 7, 12, 6, 10, 1 per task (92 of 500). [V, `baselines.jsonl`] The five worst tasks hold 70 of the 92.
- **Chance: about 20% to beat earliest-first at equal usage.** Task-level targeting enriches failures by about 1.5 times (76% of failures in half the episodes), against about 7 times for scene memory. [A, arithmetic] The live noise is about ±2.8 points. [S]
- **Cost and status.** One live run, an engine change so that episodes of a task run in order and share state, and an amendment. Episodes stop being independent, which changes the paired tests. Not worth it by Nov 30.

### 3.6 Learning to defer, cascades and routers

- **Papers.** Mozannar and Sontag, ICML 2020, [arXiv:2006.01862](https://arxiv.org/abs/2006.01862) [V]. Geifman and El-Yaniv, SelectiveNet, ICML 2019, [arXiv:1901.09192](https://arxiv.org/abs/1901.09192) [V]. Jitkrittum et al., "When Does Confidence-Based Cascade Deferral Suffice?", NeurIPS 2023, [arXiv:2307.02764](https://arxiv.org/abs/2307.02764) [V]. Ong et al., RouteLLM, ICLR 2025, [arXiv:2406.18665](https://arxiv.org/abs/2406.18665) [V]. Dekoninck et al., [arXiv:2410.10347](https://arxiv.org/abs/2410.10347), and Chen et al., FrugalGPT, [arXiv:2305.05176](https://arxiv.org/abs/2305.05176) [S-design]. Ren et al., KnowNo, CoRL 2023, [arXiv:2307.01928](https://arxiv.org/abs/2307.01928) [V].
- **What they offer.** Consistent surrogate losses for training the defer decision jointly; post-hoc deferral rules trained on "would the large model do better here"; calibrated ask-for-help thresholds.
- **Fit.** Jitkrittum et al. is the relevant one: confidence-based deferral fails when the large model is a specialist or under distribution shift, and a post-hoc rule trained on the benefit does better. [V, abstract] That is our case (failure risk is anti-correlated with benefit among failing points [S]), and a post-hoc benefit model is what the pilot fitted: ridge and boosting on b2 minus b1, about 170 fits. [S]
- **Chance of an incremental gain: under 5%.** The loss is not the bottleneck. In robotics I found edge-cloud VLA routers only as 2026 search hits (for example [arXiv:2603.07949](https://arxiv.org/abs/2603.07949), which reports that kinematic triggers beat action-entropy triggers; search summary only, not opened). Useful as related work for the poster, not as a method to run.
- **Status.** Nothing to run.

### 3.7 Uplift and treatment-effect learners

- **Papers.** Künzel et al., metalearners, PNAS 2019, [arXiv:1706.03461](https://arxiv.org/abs/1706.03461) [V]. Wager and Athey, causal forests, JASA 2018, [arXiv:1510.04342](https://arxiv.org/abs/1510.04342) [V].
- **Fit.** The decider is an uplift model: the score is an estimated effect of treatment. These methods exist because normally only one arm is observed per unit. We observe both arms from the same snapshot, three seeds each, at 0.8 reliability. [S] The two-head MLP is a "T-learner"; the pilot's direct regression on b2 minus b1 is already the best case these learners aim for.
- **Chance: under 10%.** A causal forest on the 131 inputs is one more nonlinear model; boosting did not help. [S]
- **Status.** Exploratory if tried. I would not.

### 3.8 Off-policy and bandit learning from logged data

- **Papers.** Dudík, Langford, Li, "Doubly Robust Policy Evaluation and Learning", ICML 2011, [arXiv:1103.4601](https://arxiv.org/abs/1103.4601) [V]. Liu et al., "Breaking the Curse of Horizon", NeurIPS 2018, [arXiv:1810.12429](https://arxiv.org/abs/1810.12429) [V].
- **Round 1 logs: no.** The four deciders escalate by a deterministic threshold, so each action has probability 0 or 1 given the state. Inverse-propensity and doubly robust estimators need every action to have positive probability. [A, standard condition] Recording was off, so there are no embeddings either. [S]
- **Random-escalation data: unbiased, with limits.**
  - Evaluating a *deterministic* decider from random logs by weighting whole episodes fails: an episode matches the decider's 30 to 50 choices with vanishing probability. This is the "curse of horizon". [S, Liu et al.; A for our numbers]
  - Per-decision effects are estimable without that problem. That is 3.1.
  - Policy learning from them is unbiased in principle and low-powered in practice. [A]
- **Status.** Covered by 3.1.

### 3.9 Semi-supervised use of the base rollouts, sequence models, active learning

- **Semi-supervised.** The useful part is the free "continue" label, which is 3.2. Using the unlabelled embeddings for a better representation is unpromising: adding embedding components made the pilot's models slightly worse. [S]
- **Sequence models.** A recurrent or attention model over the decision-point history could replace the hand-built history features. With about 600 labelled episodes it would overfit, and on the 31,000 free-labelled points it becomes 3.2 with a different model. Chance of beating the linear history model: about 10%. [A]
- **Active learning and new labels.** This is rank 4 of `2x2-learning-plan.md`: 8 to 12 GPU hours for 1,500 to 1,900 new equal-weight points on the Spark. [S] The learning curve says the linear cascader needs about three times today's episodes. [S] It competes with the baselines for the GPU and needs an amendment before any label is opened. Unchanged advice: only if the baselines finish early.
- **Diff-DAgger** (Lee et al., [arXiv:2410.14868](https://arxiv.org/abs/2410.14868) [S-design]) and the other DAgger variants change *when to query*, using a loss-based uncertainty. They need a new server endpoint and still need a labelling round. No.

### 3.10 Neurosymbolic options

**Papers read from Oscar's folder.** The folder holds 22 PDFs, not 27. [V] Nine apply. I read each one's abstract and introduction, and the method section of the first four; I did not study their experiments in detail. Titles and authors are from the PDFs' first pages. Where a PDF does not print its venue (the two Li et al. papers), the venue and year are from my memory. [A]

| paper | what it contributes here |
| --- | --- |
| Hu, Ma, Liu, Hovy, Xing. "Harnessing Deep Neural Networks with Logic Rules". ACL 2016 | The recipe for rule distillation: build a rule-regularized teacher by posterior regularization, train the student to imitate it with strength π, and use unlabelled data to absorb the rule. This is the template for N1 |
| Li, Srikumar. "Augmenting Neural Networks with First-order Logic". ACL 2019 | Compiles rules into the network as constrained neurons with no new parameters. Its stated finding is that the gain is largest in low-data regimes and shrinks with more data. We are in the low-data regime, which is the argument for trying rules at all |
| Li, Gupta, Mehta, Srikumar. "A Logic-Driven Framework for Consistency of Neural Models". EMNLP 2019 | Turns logical invariants into differentiable losses with a t-norm, enforceable on unlabelled examples. Template for constraints over the 31,000 unlabelled base points (N1) |
| Calanzone, Teso, Vergari. "Logically Consistent Language Models via Neuro-Symbolic Integration". 2024, arXiv:2409.13724 | Semantic loss (the probability that a constraint holds) as a fine-tuning regularizer, with several constraints combined in one loss. Same role as the previous row, in the semantic-loss form |
| Zhang, Kung, Yoshida, Van den Broeck, Peng. "Adaptable Logical Control for Large Language Models" (Ctrl-G). NeurIPS 2024 | Constraints written as a finite automaton and enforced at inference, with the model left untrained. Template for N2: a rule mask over the decision sequence |
| Lu, West, Zellers, Le Bras, Bhagavatula, Choi. "NeuroLogic Decoding". NAACL 2021 | Same idea at decoding time; its claim is that models do not learn constraints reliably from examples, even many. That matches our MLPs failing to learn the clock from 1,000 points |
| Mao, Gan, Kohli, Tenenbaum, Wu. "The Neuro-Symbolic Concept Learner". ICLR 2019 | A decision made by a symbolic program over learned concepts instead of raw features. Template for N3 |
| Kamali, Barezi, Kordjamshidi. "NeSyCoCo". AAAI 2025 | Names the obstacle for N3: getting the right predicates out of raw inputs is the hard part, and predefined predicate sets limit transfer |
| Pacheco, Somenzi, Srinivas, Trivedi. "Explaining Hitori Puzzles: Neurosymbolic Proof Staging for Sequential Decisions". 2025, arXiv:2508.14294 | Explaining a sequence of decisions by short symbolic derivations. Useful for presenting a rule-based decider on the poster, not for improving one |

Not applicable (read the first page only): Dalvi et al. 2021 (entailment trees), Gururangan et al. 2022 (DEMix), Das et al. (media narratives), Liu et al. (autoregressive structured prediction), Andreas et al. 2016 (neural module networks), Liu et al. (Disco-RAG), Akkaya, Wiebe, Mihalcea 2014 (constrained clustering), Wang et al. (Logical Transformers), Pauk and Pacheco (prompt-based structured prediction), Han and Gardent (question decompositions), West et al. (symbolic knowledge distillation), Lee, Pacheco, Goldwasser (event embeddings), the anonymous CommunityMix submission.

Other papers, verified by search [V]: Xu et al., "A Semantic Loss Function for Deep Learning with Symbolic Knowledge", ICML 2018, [arXiv:1711.11157](https://arxiv.org/abs/1711.11157). Koh et al., "Concept Bottleneck Models", ICML 2020, [arXiv:2007.04612](https://arxiv.org/abs/2007.04612). Ratner et al., "Data Programming", NeurIPS 2016, [arXiv:1605.07723](https://arxiv.org/abs/1605.07723). Angelino et al., CORELS, JMLR 2018, [arXiv:1704.01701](https://arxiv.org/abs/1704.01701). Zhou et al., "Code-as-Monitor", CVPR 2025, [arXiv:2412.04455](https://arxiv.org/abs/2412.04455). Liu et al., LIBERO, NeurIPS 2023, [arXiv:2306.03310](https://arxiv.org/abs/2306.03310).

**N1. Rules as soft constraints or a distilled teacher (semantic loss, posterior regularization).**
- **Knowledge needed.** The rules. Candidates: (a) benefit does not rise with the step; (b) benefit is at most zero where X-VLA would succeed anyway; (c) benefit is about "fails" times the clock.
- **Do we have it?** Yes, but every rule was read off the unblinded labels, so a rule-trained decider is exploratory by construction. [A] Unlabelled points to enforce it on: yes, about 31,000, and for (b) and (c) their outcome is known at training time, so the teacher can use it and the student cannot. That is Hu et al.'s teacher and student exactly.
- **Cost.** CPU. A loss term in the existing training code, or labelling functions in the sense of Ratner et al.
- **Chance.** About 70% that a rule-constrained MLP ties earliest-first (today it is below it at 20%, -0.005 to -0.009 [S]). About 10% that it beats it. Rule (a) is the clock. Rules (b) and (c) ask the student to predict failure early from its inputs, which is option 2 under another name and has the same bottleneck (0.55 to 0.61). Li and Srikumar's low-data result says rules help most where we are; it does not say they add information the rule lacks.
- **Status.** Offline exploratory. Live only through item (6) of the draft amendment.

**N2. Rules enforced at decision time (a mask over the decision sequence).**
- **Knowledge needed.** A small automaton: no escalation after step 340; at most r1 and r2 of offered points in the first and second third; the learned score only breaks ties inside a third.
- **Do we have it?** Yes. It needs no training and runs in the live wrapper, since the step count and budget are decider inputs. [S, SPEC §6]
- **Cost.** CPU minutes offline. One live run if tested.
- **Chance.** By construction it cannot do worse than its schedule. It beats earliest-first only if the tie-break score carries within-third signal: the linear cascader has 0.59 to 0.62 within thirds against a shuffle spread of 0.04 [S], so about 15%. As a fix for the live deciders' wild usage (3.5% to 51% from λ alone [S]) it is close to certain.
- **Status.** This replaces the per-third threshold of the earlier shortlist. It is the form I would give the lite decider in item (6): "early-first schedule, learned tie-break". Its test is then exactly the draft's "decider minus step-matched random".

**N3. Symbolic-state or concept-bottleneck decider over task predicates.**
- **Knowledge needed.** Per decision point: which goal conjuncts hold, whether an object is grasped or lifted, how many subgoals remain. LIBERO-Long goals are conjunctions of two subgoals, so "share of goal conjuncts satisfied" is a task-agnostic progress measure. [A, from the task suite's design; S for BDDL goals]
- **Do we have it?** Offline: computable from snapshots, with work (see section 2). Live: no.
- **Cost.** About half a day to write and verify a predicate extractor, then CPU on the Spark over 15,671 snapshots while it is running live rounds. No GPU. A live version would need either a harness change (out of scope) or a concept predictor from embeddings in the style of Koh et al., trained and registered: not possible by Nov 30. [A]
- **Chance.** As an offline, privileged reference: about 25% that "no subgoal done by now, relative to the task's usual time" beats earliest-first. The limit is the same as everywhere: before step 170 most episodes, failing or not, have no subgoal done. [A] From step 170 it should be strong, where failure is already predictable without it.
- **Why it is still the most interesting neurosymbolic idea.** It is the only one that adds information the inputs lack, and it attacks the stated gap ("which episode fails is a property of the scene"). Kamali et al.'s caution applies: the predicates are easy in the simulator and hard from pixels.
- **Status.** Exploratory, privileged, offline only. A good "what would fix it" item for the poster.

**N4. Rule or program induction from the labelled points.**
- **Knowledge needed.** None beyond the labels.
- **Result.** I fitted one shallow tree (depth 3, at least 30 points per leaf) on the fresh benefit, leave one task out. Gain lens at 20%: +0.0016 with the 27 inputs and -0.0050 with the 131, against +0.0078 for earliest-first. The first split is the clock (time remaining). [V, section 4]
- **Chance.** Under 10%. A certified-optimal rule list (Angelino et al.) searches the same space more thoroughly on the same 1,296 points.
- **Status.** Exploratory; done.

**N5. Logic-guided failure monitors.**
- **Papers.** Zhou et al. (Code-as-Monitor) has a vision-language model write constraint-checking code; Sentinel (above) pairs a statistical check with a vision-language model. Both need a large model in the loop, which is the thing we are rationing.
- **Our version.** Hand-written temporal rules on signals we log: a missed grasp in the last segment, a failed close so far, a close with nothing ever held.
- **Result.** They fire almost only in failing episodes (failed close: 37% of failed-parent points against 4%), and they mark the points where a call helps least. In the middle third a call adds +0.03 after a failed close against +0.17 without one. As rankings they are at or below earliest-first. [V, section 4]
- **Chance.** Under 5% as positive triggers. The reverse rule ("skip points after a failed grasp") is suggested by the data and was not fixed in advance; treat it as a hypothesis only.
- **Status.** Exploratory; done. Worth one sentence on the poster: visible failure evidence arrives after the rescuable window.

## 4. The checks I ran [V]

Exploratory, on the unblinded pilot table (`points.npz`, 1,296 used points). Fixed before running: one model (logistic, C = 0.1), two feature sets, leave one task out, no tuning. It adds 2 fits to the roughly 170 already made. Script: `dagger-alternatives-hurdle.py` in the shared notes folder.

Held-out AUROC for "the parent episode failed", by step third (mean over tasks):

| low-dimensional inputs | steps 0-169 | 170-339 | 340 on (3 tasks only) |
| --- | --- | --- | --- |
| router with memory (27) | 0.55 | 0.76 | 0.90 |
| cascader with history (131) | 0.56 (0.61 on base parents) | 0.81 | 0.91 |

Per task the early figure runs from 0.32 to 0.89, so it is noise around a small effect.

Hurdle score (predicted failure times the third's mean benefit) on the registered lens:

| rule | gain 10% | gain 20% | benefit AUROC | within-third AUROC |
| --- | --- | --- | --- | --- |
| earliest first | +0.0006 | +0.0078 | 0.63 | 0.62 |
| hurdle, 27 inputs | +0.0028 | +0.0033 | 0.53 | 0.52 |
| hurdle, 131 inputs | -0.0013 | +0.0024 | 0.57 | 0.54 |
| predicted failure alone, 27 inputs | +0.0088 | +0.0092 | 0.51 | 0.52 |
| true parent outcome times the clock (not observable) | +0.0166 | +0.0208 | 0.77 | 0.81 |

Reading:
- **With today's labelled points the hurdle model does not beat earliest-first.** The failure score has almost no early signal, so multiplying by it only adds noise to the clock.
- **"Predicted failure alone" gets +0.009 on the registered lens while ranking benefit at chance.** The lens is weight-dominated by successful-parent points, so avoiding them is rewarded. It is about 2.5 shuffle standard deviations at 10%, one of two cells; do not lean on it.
- **This check does not settle 3.2.** It used 1,296 points. The proposal is to train the failure score on about 31,000. The decisive number is the early-third held-out AUROC on that data.

### Second check: hand-written rules and a shallow tree

Same table and lenses. Fixed before running: three event rules and one tree setting. Script: `dagger-alternatives-rules.py` in the shared notes folder. With the first check this brings the exploratory count to about 180 fits or rules.

Fresh benefit at failed-parent points, rule on against rule off:

| rule | steps 0-169 | 170-339 | 340 on |
| --- | --- | --- | --- |
| missed grasp in the last segment (fires at 6% of points) | +0.00 (n=5) against +0.25 | +0.03 (n=45) against +0.12 | +0.04 against +0.07 |
| at least one failed close so far (25%) | +0.15 (n=26) against +0.26 | +0.03 (n=145) against +0.17 | +0.05 against +0.09 |
| a close command and nothing ever held (2%) | +0.39 (n=12) against +0.24 | no points | no points |

| ranking | gain 10% | gain 20% | benefit AUROC |
| --- | --- | --- | --- |
| earliest first | +0.0006 | +0.0078 | 0.63 |
| missed grasp, then earliest | -0.0013 | +0.0051 | 0.61 |
| failed close, then earliest | -0.0030 | -0.0017 | 0.57 |
| closed and never held, then earliest | +0.0012 | +0.0070 | 0.62 |
| tree, depth 3, 27 inputs | +0.0003 | +0.0016 | 0.66 |
| tree, depth 3, 131 inputs | -0.0036 | -0.0050 | 0.60 |

Reading:
- **A failed grasp is a sign that one call will not help.** In all three thirds the benefit is lower after one. One 10-step call is probably too short to recover a dropped or missed object. [A for the reason]
- **The third rule's +0.39 rests on 12 points.** It is the only hint of an early positive trigger and is not distinguishable from noise.
- **Nothing here beats earliest-first.**

## 5. Ranked shortlist

All five leave the GPU plan of `amendment-followup-draft.md` untouched unless stated.

| rank | option | GPU | first check on CPU | go or stop | status |
| --- | --- | --- | --- | --- | --- |
| 1 | **Micro-randomized analysis** of random-escalation runs (3.1) | none | Find out whether the 1,974 screens' `steps` and `events` tables are in the handoff parts on hand. Count randomized decisions per step third with `can_escalate` true. Then one regression: final success on the escalation indicator minus 0.1, by third, episode-clustered errors. | Stop if the tables are not on hand; the planned (a) and (d) runs still give the same analysis later | Exploratory on the screens. Add one sentence to the draft amendment before (a) and (d) start to make their version registered |
| 2 | **Failure hazard from free labels, times the clock** (3.2), trained as a rule-distilled student (N1): the teacher sees the episode outcome, the student does not | none offline; 3 to 5 h if run live as the lite decider | On the signal study's extracted Spark points (and the L4 base points if on hand): held-out AUROC for "episode fails" restricted to steps 0-169, low-dimensional inputs, leave one task out | Go on only if the early figure is at least 0.65. Then score the hurdle rule on the labelled points against earliest-first. If the early figure fails, keep only the middle-third rule as a note | Offline exploratory. Live only through item (6) of the draft amendment, fixed before training |
| 3 | **Retrieval decider over seen scenes** (3.4) | 3 to 5 h, only if the list in the draft amendment finishes early | Offline, by episode folds: nearest-scene lookup on the step-0 embedding, then the pilot's scene-memory score. Does embedding lookup recover the scene (top-1 match rate), and does the score keep the pilot's 0.73? | Go only if lookup is near-perfect and (4) of the draft amendment has shown a live effect | New pre-registered item, labelled "seen scenes", never as the 2 x 2's answer |
| 4 | **Rule-masked decider** (N2): early-first schedule with fixed rates per third, the linear cascader's score only breaking ties within a third | none offline; it is the lite decider's live run if item (6) runs | On the labelled points: gain lens of "schedule with learned tie-break" against "schedule with random tie-break", leave one task out, 20 random tie-break draws | Adopt as the form of item (6) only if the learned tie-break is above the random one in at least 8 of 10 tasks; otherwise item (6) is not worth its GPU hours | Fixed in item (6) before training. Replaces the per-third threshold of the first version of this list |
| 5 | **Symbolic-state reference** (N3): goal-predicate progress from snapshots, offline and privileged | none; CPU on the Spark, competing with live rounds | Restore 20 Spark base snapshots from one task on CPU and evaluate the BDDL goal conjuncts; confirm the final snapshot of a successful episode satisfies all of them | Stop if a restore needs the GPU or takes over about 2 s each (15,671 snapshots). Otherwise tabulate "conjuncts satisfied" by step and parent outcome | Exploratory, privileged, offline only. Poster material for "what would fix it" |

What I would tell Oscar in one line: there is no drop-in replacement for DAgger that is likely to make these deciders work by Nov 30; option 1 recovers, for free, most of what a DAgger round would have taught us about on-policy states, and option 2 is the only learned-decider idea with a clear cheap go or stop test. Of the neurosymbolic ideas, the rule mask (option 4) is the one to adopt if anything is trained, because it guarantees the clock; the symbolic state (option 5) is the one that could add real information, and it cannot run live under the frozen harness.

## 6. Not recommended

- Any DAgger variant (ThriftyDAgger, Diff-DAgger and similar): same labelling round, same inputs.
- Sample-based uncertainty live (spread, temporal consistency, entropy): measured weak here and too costly on the clock. [S]
- OOD scores on embeddings for held-out tasks: no same-task calibration data exists by design.
- Learning-to-defer losses, metalearners, causal forests, more model classes on the 1,296 points: the pilot covers their best case, and each further fit adds to an exploratory count of about 180.
- Off-policy learning from round 1: deterministic policies, no recording.
- Online per-task adaptation: needs an engine change and breaks episode independence.
- Hand-written failure triggers and rule induction on the labelled points: checked, at or below earliest-first.
- A concept predictor from embeddings for live use, and monitors that call a vision-language model: not feasible by Nov 30.

## 7. Not checked

- Whether the exploration screens' `steps` and `events` are on the Mac or the Spark. [U]
- Whether round 1's event logs hold the decider's score. [U]
- The early-third failure AUROC on the full base rollouts. This is option 2's first check; I did not open the Spark. [U]
- Whether a snapshot restores and evaluates goal predicates on CPU without rendering. [U]
- The vocabulary of LIBERO's BDDL predicates; I confirmed only that the harness loads the BDDL file. [U]
- Full texts of the papers. Claims about them come from abstracts and search summaries.
- The 2026 search hits on VLA failure detection ([arXiv:2609.04277](https://arxiv.org/abs/2609.04277), [arXiv:2606.21386](https://arxiv.org/abs/2606.21386), [arXiv:2609.21246](https://arxiv.org/abs/2609.21246)) were not opened. From the search summaries they extend SAFE's recipe (same-task or seen-task calibration), so I expect the same limit; that is [A].

## References

Verified in this session by search unless marked.

- Boruvka, Almirall, Witkiewitz, Murphy. Assessing Time-Varying Causal Effect Moderation in Mobile Health. JASA 2018. https://arxiv.org/abs/1601.00237
- Nie, Brunskill, Wager. Learning When-to-Treat Policies. JASA 2021 (vol. 116). https://arxiv.org/abs/1905.09751
- Gu et al. SAFE: Multitask Failure Detection for Vision-Language-Action Models. 2025. https://arxiv.org/abs/2506.09937
- Xu et al. Can We Detect Failures Without Failure Data? (FAIL-Detect). 2025. https://arxiv.org/abs/2503.08558
- Römer, Kobras, Worbis, Schoellig. Failure Prediction at Runtime for Generative Robot Policies (FIPER). NeurIPS 2025. https://arxiv.org/abs/2510.09459
- Agia et al. Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress (Sentinel). 2024. https://arxiv.org/abs/2410.04640 [S-design]
- Lee et al. Diff-DAgger. 2024. https://arxiv.org/abs/2410.14868 [S-design]
- Hoque et al. ThriftyDAgger. CoRL 2021. https://arxiv.org/abs/2109.08273
- Ren et al. Robots That Ask For Help (KnowNo). CoRL 2023. https://arxiv.org/abs/2307.01928
- Gensheimer, Narasimhan. A scalable discrete-time survival model for neural networks. PeerJ 2019. https://arxiv.org/abs/1805.00917
- Lee, Lee, Lee, Shin. A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. NeurIPS 2018. https://arxiv.org/abs/1807.03888
- Sun, Ming, Zhu, Li. Out-of-Distribution Detection with Deep Nearest Neighbors. ICML 2022. https://arxiv.org/abs/2204.06507
- Ma et al. Vision Language Models are In-Context Value Learners. ICLR 2025. https://arxiv.org/abs/2411.04549
- Angelopoulos, Bates, Fisch, Lei, Schuster. Conformal Risk Control. ICLR 2024. https://arxiv.org/abs/2208.02814
- Pritzel et al. Neural Episodic Control. ICML 2017. https://arxiv.org/abs/1703.01988
- Nie, Ding, Hu, Jermaine, Chaudhuri. Online Cascade Learning for Efficient Inference over Streams. ICML 2024. https://arxiv.org/abs/2402.04513
- Mozannar, Sontag. Consistent Estimators for Learning to Defer to an Expert. ICML 2020. https://arxiv.org/abs/2006.01862
- Geifman, El-Yaniv. SelectiveNet. ICML 2019. https://arxiv.org/abs/1901.09192
- Jitkrittum et al. When Does Confidence-Based Cascade Deferral Suffice? NeurIPS 2023. https://arxiv.org/abs/2307.02764
- Ong et al. RouteLLM. ICLR 2025. https://arxiv.org/abs/2406.18665
- Dekoninck et al. A Unified Approach to Routing and Cascading for LLMs. 2024. https://arxiv.org/abs/2410.10347 [S-design]
- Chen et al. FrugalGPT. 2023. https://arxiv.org/abs/2305.05176 [S-design]
- Künzel, Sekhon, Bickel, Yu. Metalearners for estimating heterogeneous treatment effects using machine learning. PNAS 2019. https://arxiv.org/abs/1706.03461
- Wager, Athey. Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. JASA 2018. https://arxiv.org/abs/1510.04342
- Dudík, Langford, Li. Doubly Robust Policy Evaluation and Learning. ICML 2011. https://arxiv.org/abs/1103.4601
- Xu, Zhang, Friedman, Liang, Van den Broeck. A Semantic Loss Function for Deep Learning with Symbolic Knowledge. ICML 2018. https://arxiv.org/abs/1711.11157
- Koh et al. Concept Bottleneck Models. ICML 2020. https://arxiv.org/abs/2007.04612
- Ratner, De Sa, Wu, Selsam, Ré. Data Programming: Creating Large Training Sets, Quickly. NeurIPS 2016. https://arxiv.org/abs/1605.07723
- Angelino, Larus-Stone, Alabi, Seltzer, Rudin. Learning Certifiably Optimal Rule Lists for Categorical Data. JMLR 2018. https://arxiv.org/abs/1704.01701
- Zhou et al. Code-as-Monitor. CVPR 2025. https://arxiv.org/abs/2412.04455
- Liu et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. NeurIPS 2023. https://arxiv.org/abs/2306.03310
- From Oscar's folder (local PDFs, read as described in 3.10): Hu et al. ACL 2016; Li and Srikumar ACL 2019; Li, Gupta, Mehta, Srikumar EMNLP 2019; Calanzone, Teso, Vergari 2024 (https://arxiv.org/abs/2409.13724); Zhang et al. NeurIPS 2024; Lu et al. NAACL 2021; Mao et al. ICLR 2019; Kamali, Barezi, Kordjamshidi AAAI 2025; Pacheco et al. 2025 (https://arxiv.org/abs/2508.14294).
- Liu, Li, Tang, Zhou. Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation. NeurIPS 2018. https://arxiv.org/abs/1810.12429
