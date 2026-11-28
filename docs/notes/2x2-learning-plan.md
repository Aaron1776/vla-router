# Helping the deciders learn something: pilot and ranked plan

Written 2026-11-27 (late evening, -07:00) by the same side session as `2x2-second-opinion.md`, answering Oscar's follow-up. Read-only on the repo and the Spark, nothing on the GPU, nothing on the Spark's CPU.

What I did instead of only listing ideas: I copied the records tables, label maps, inclusion file and calibration (1.1 GB) from the Spark to a scratch folder on the Mac, built the training table there with the pinned code (`cascader-2x2`, exported with `git archive`), and piloted the cheap ideas on the existing labels. The copy cost the Spark about a minute of disk and network during live round 1. All fitting ran on the Mac.

Files, all in the shared notes folder (outside the repo, which is why they are not in the project folder):
- `2x2-second-opinion.md` and `2x2-second-opinion-scripts/`: the first note and its scripts.
- `2x2-learning-plan.md`: this note.
- `2x2-learning-plan-pilot/`: `points.npz` (the 1,390-point training table with every input column), `seeds.json` (per-seed outcomes, strata), `baselines.jsonl` (per-episode X-VLA-alone and π0.5-alone results), the scripts (`build.py`, `pilot2.py`, `reliab.py`, `scene.py`, `seen.py`, `strat.py`) and `pilot2.log`. With these, further CPU work needs neither the Spark nor a rebuild.

Labels: **[V]** I ran it. **[S]** stated in a source or read in code. **[A]** my reasoning. **[U]** unknown.

Everything here is exploratory. The pilot fitted about 170 model variants on labels that are already unblinded. A best cell picked from it means nothing by itself; I report patterns across cells and nothing was tuned.

## What the pilot found

### A correction to my first note: the labels are not noisy [V]

I had assumed three seeds made the benefit label noisy. The per-seed outcomes say otherwise. Two independent seed pairs at the same point give benefit estimates that correlate at 0.57, which makes the 3-seed label about 0.8 reliable (0.72 for early failed-parent points, 0.84 later). Outcomes from a snapshot are close to deterministic: two seeds of the same branch agree at r = 0.87 to 0.89.

So whether a call rescues a given state is a stable property of that state. **More seeds per point would buy little.** The deciders' failure is a modelling and measurement problem.

### The benefit is real and the same in every task [V]

With the parent's own seed removed from b1 (fresh seeds only, unweighted):

| | X-VLA continues | one π0.5 call | difference | points |
| --- | --- | --- | --- | --- |
| failed parent, steps 0-170 | 0.285 | 0.533 | +0.248 | 291 |
| failed parent, 170-340 | 0.052 | 0.157 | +0.105 | 324 |
| failed parent, 340-520 | 0.009 | 0.073 | +0.063 | 211 |
| successful parent, 0-170 | 0.950 | 0.935 | -0.015 | 298 |
| successful parent, 170-340 | 0.987 | 0.975 | -0.011 | 162 |

In the early failed-parent cell the difference is positive in all ten tasks (+0.12 to +0.34).

### Which episodes fail is mostly a property of the scene [V]

The same 50 initial scenes per task are used in base, exploration and evaluation runs, with different rollout seeds. [S, SPEC §7 and §10] Of the 102 scenes where the L4 base rollout failed, X-VLA alone failed again in 60% on the Spark's evaluation seeds. Of the 398 where it succeeded, 8%.

A decider needs two things: "is this episode failing" and "is it still early". The second is a clock. The first is mostly the scene, and the pooled features do not expose it across tasks.

### Cheap redesigns, on held-out tasks [V]

References on the same data. "Gain lens" is the registered one: inclusion-weighted gain minus random at the same rate, mean over tasks. Shuffled scores give a standard deviation of 0.0034 at 10% and 0.0044 at 20% on it, and 0.04 on the benefit AUROC.

| ranking rule | gain lens at 10% | at 20% | benefit AUROC | within-third AUROC |
| --- | --- | --- | --- | --- |
| registered MLPs (four variants) | -0.006 to +0.002 | -0.009 to -0.005 | 0.48 to 0.50 | 0.50 to 0.53 |
| escalate earliest first | +0.0006 | +0.0078 | 0.63 | 0.60 |
| true parent outcome, then earliest (not observable) | +0.0166 | +0.0208 | 0.77 | 0.79 |
| ranking by the labels themselves | +0.0496 | +0.0502 | 1.00 | |

Models tried, all leave-one-task-out, hyperparameters fixed in advance: ridge on the benefit directly, two logistic heads, gradient boosting on the benefit; inputs without the 2,048 embedding values (router 14, router with memory 27, cascader 91, cascader with history 131), each with and without 16 principal components of the embeddings; three weightings (inclusion weights with registered labels, square-root weights with fresh labels, no weights with fresh labels).

- **Nothing clearly beats "escalate earliest first" on the gain lens.** The best cells reach +0.007 to +0.010 at 20%, about two shuffle standard deviations, which is what the best of 60 cells does by luck.
- **One pattern is consistent.** The low-dimensional cascader-with-history inputs give a benefit AUROC of 0.61 to 0.65 in all 12 linear cells. The router's low-dimensional inputs give 0.41 to 0.56. Within step thirds the order is router 0.47 to 0.54, router with memory 0.54 to 0.59, cascader 0.53 to 0.54, cascader with history 0.59 to 0.62. That is the 2 x 2's hoped-for order, at 2.5 to 3 shuffle standard deviations for the top cells.
- **But it only matches the clock.** "Earliest first" alone scores 0.63 and 0.60 on the same two measures.
- **Levers that did nothing consistent:** the direct benefit target, tempering or dropping the weights, fresh b1 labels, gradient boosting, adding embedding components (slightly worse).
- **Splitting within tasks does not help.** Holding out initial states of seen tasks gives the same picture as holding out tasks.

### Seen scenes [V]

Folding by episode, so that most test scenes appear in training through another episode (61% do):

- Cascader-with-history inputs, logistic, inclusion weights: +0.012 at 10% and +0.011 at 20%, AUROC 0.70.
- Adding a "scene memory" input (how other labelled episodes of the same scene fared, from training folds only): +0.007 and +0.017, AUROC 0.73, positive in 9 of 10 tasks, within-third AUROC 0.67. This is the clearest signal in the pilot, about four shuffle standard deviations at 20%.
- Router inputs stay near chance here too (AUROC 0.49 to 0.59).

This is recognising recurring scenes, not generalising. It is a legitimate, weaker claim only if labelled as such.

### Code review: `data.py`, `history.py`, the rest of `variants.py` [S, read; build reproduced V]

No bug that would suppress signal. Rebuilding the table locally with this code gave 1,390 points and the registered input widths (2,062, 2,075, 2,139, 2,179), with 0 incomplete points.

Design points that work against learning:

- **The parent rollout counts as seed 1 of b1 at base points** (`labels.py:138-146`; 709 of the 1,296 used points). The parent's outcome is also the sampling stratum, so at failed-parent base points a third of b1 is zero by construction. The inclusion weights undo this on average (weighted mean benefit 0.0230 registered against 0.0223 fresh), at the price of the heavy weights. Using fresh labels made no consistent difference in the pilot.
- **Model selection never looks at the benefit.** Early stopping and λ both use weighted quantities on one validation task whose effective sample size is 28 to 82.
- **History features** take neutral values after π0.5 segments and at episode start, as documented. The gripper sign (`proposal[:, 6] > 0` is "closed") matches the harness's own `gripper_fractions`. I ran no unit tests on the feature functions.

## Ranked plan

GPU from about 17:00Z Nov 28 to the end of Nov 30 is about 55 hours. A 500-episode live run is about 2.6 to 5.5 hours. [A, from 946 to 1,963 s per 50 episodes, V] Scripted runs go to 520 steps with no time cut, so they may be slower. [U]

CPU cost during round 1: none is needed on the Spark. For scale, the registered training of all 60 deciders took about ten minutes at four threads there. [V, file times]

| rank | what | GPU | chance of a clear positive [A] | pre-register? |
| --- | --- | --- | --- | --- |
| 1 | Timing: the §10 random baselines, plus an early-first schedule and a step-matched random | 16 to 24 h (six runs; the two low-usage ones are short) | moderate to good | §10 part is registered; add the rest as a short amendment |
| 2 | Positive control: prior-failure schedule (privileged scene memory) | 3 to 5 h | good | yes, one paragraph |
| 3 | A low-dimensional linear decider pair, tested live against step-matched random | 6 to 10 h | low to moderate | yes if run live |
| 4 | New equal-weight labels on the Spark as a sealed test set | 8 to 12 h | moderate for timing, low for features | yes, before any label is opened |
| no | More seeds at existing points; DAgger; more tuning on existing labels | | | |

Ranks 1 and 2 together are about 19 to 29 hours and fit. Adding one of 3 or 4 fits. All four is about 33 to 51 hours and does not fit safely; drop 3 first.

### 1. Measure the timing the deciders already learned

- **Why it should work.** Help is early: 47% of early failed-parent points are positives against 8% late. On task 0 the live deciders already escalate early (0.36 to 0.71 of decision points in steps 0-170, 0.06 to 0.08 after 340). [V] So "the deciders learned when to call" is a positive that may already be in hand.
- **Runs.** (a) Uniform random escalation matched per fold to `router_hist`'s and to `cascader_hist`'s usage, as §10 requires. (b) An early-first schedule at the same usage. (c) Random escalation with the decider's own rate in each third of the episode. (d) Early-first against uniform random at a low usage, about 15%, where timing should matter most. [A]
- **Comparisons.** Decider minus uniform random is §10's "value of learning". Early-first minus uniform random tests timing. Decider minus step-matched random tests whether anything beyond timing was learned.
- **Quick check.** After two tasks, the paired difference between early-first and uniform random. Fix 500 episodes in advance and do not stop on a peek.
- **Risk.** At 40% to 50% usage a uniform random policy also covers the early steps, so the difference may be small against a noise of about ±2.8 points. [S, `cascader-design.md`] That is why (d) is in.
- **Amendment.** Define (b), (c), (d) and their paired tests before they run. Worth doing: it is cheap and the data are fresh.

### 2. Positive control: remember where X-VLA failed

- **What.** A scripted schedule, not a decider: in the 102 scenes where the L4 base rollout failed, escalate at the early decision points; elsewhere never. Compare with uniform random and early-first at the same total usage.
- **Why.** Failure repeats by scene (60% against 8%). Offline, "hard scene, then earliest" gives +0.011 at 20% against random, interval [-0.001, +0.024], AUROC 0.68. [V]
- **What it shows.** How much selective routing is worth when the selection is right. It gives the poster a non-null that is true: "where to call matters, and a router that remembers failures uses far less π0.5".
- **Rough size.** Of 102 flagged scenes about 60 fail again; if early calls rescue a third to a half of those, that is 4 to 6 points of success at a few percent usage. [A, crude]
- **Risk.** It uses scene identity and past outcomes, which §6 forbids as decider inputs. It must be labelled privileged, and it says nothing about unseen scenes.
- **Quick check.** None needed beyond the first two tasks' paired difference.

### 3. A decider small enough for the data

- **What.** `router_hist_lite` and `cascader_hist_lite`: the same inputs without the two embedding vectors (27 and 131 values), a linear model, no early stopping, λ set to a fixed escalation rate on the validation task instead of the best point of a noisy curve.
- **Why.** It is the only learned ranking in the pilot that is above chance on held-out tasks on every cell (AUROC 0.61 to 0.65 against the MLP's 0.48 to 0.50), and it reproduces the 2 x 2's order.
- **Honest limit.** It is not shown to beat the clock. Expect it to tie the step-matched random baseline live.
- **How.** `add_variant` with a column list without `features_main` and `features_wrist`; `TrainConfig(hidden=())` gives a linear model through the existing training code. The live loader builds the model from the checkpoint's `hidden` field (`ops/spark/live.py:171` on `spark-eval`). [S] Not tried. [U]
- **Quick check.** On the Mac, through the real training code: within-third benefit AUROC of at least 0.58 on held-out tasks, or stop.
- **Amendment.** If it runs live: primary, `cascader_hist_lite` minus step-matched random at its usage; secondary, `cascader_hist_lite` minus `router_hist_lite`, which is the 2 x 2's question again on fresh data. It was chosen after about 170 exploratory fits, so only the live result counts.

### 4. New labels, as a test set with power

- **Cost.** The dry run did 240 branch jobs in 1,155 s with 8 shards: 4.8 s per job. [S, `dryrun-writeup.md`] A 3-seed point at a base parent is 5 jobs, about 24 s (29 s with the parent check). With 2 fresh seeds per branch it is 4 jobs, about 19 s. Ten hours is roughly 1,500 to 1,900 points, more than Phase 1's 1,390. [A, arithmetic]
- **Design.** A uniform random sample of eligible decision points from the Spark's own 500 base episodes (15,671 snapshots are on disk [V]), 2 fresh seeds per branch, the parent not counted in b1. Equal weights, so the effective sample size is the sample size, against about 550 over all ten test tasks now.
- **Use.** First as a sealed test: the registered four deciders, the lite pair and the early-first rule, frozen, scored on it. Only afterwards as extra training data, exploratory.
- **Why it could give a positive.** The offline lens would be two or more times sharper. Early-first's +0.008 would become clearly visible; the lite decider's edge might. [A]
- **Limits.** Only about 97 failing base episodes on the Spark [S, amendment 78 (4)], so episodes, not points, bound the precision. Base states only. Spark labels against L4-trained deciders. The branch runner is step 660 on the local `spark-dryrun` branch and has run 40 points, not 1,500. [S]
- **Quick check.** After 300 points, compare label rates by stratum with Phase 1's base points (early failed-parent benefit near +0.25). Look at no decider score before the amendment's analysis.
- **Amendment.** Yes. This is the only route left to a confirmatory offline claim.

### Not recommended

- **More seeds at existing points.** The label is already about 0.8 reliable. [V] The L4 snapshots (handoff part 3) are not on the Spark; only parts 1 and 2 are. [V]
- **DAgger.** Still no. The labels are reliable and the limit is not which states were visited. The missing piece is which episodes fail, a scene property these inputs do not expose, and DAgger adds nothing for that.
- **More fitting on the existing labels** in search of a significant offline number. After about 170 variants the honest summary is "matches the clock". Further search would manufacture a result.

## What can and cannot be claimed if this works

- Ranks 1 and 2 can support "timing matters" and "knowing where the local model fails is worth this much". Neither is "the cascader beats the router".
- Rank 3 can support the 2 x 2's question on fresh data, in a smaller model. I would not bet on it.
- The registered result stays as printed. These are additions beside it, each marked exploratory or tied to its own amendment.

## Not checked

- Whether the live loader accepts a linear, embedding-free variant end to end. [U]
- How long scripted baselines take on the Spark. [U]
- Training-seed variation of the registered primary (now runnable on the Mac from `points.npz`; not run). [U]
- Whether a nonlinear small model on the low-dimensional inputs does better than the linear one. Gradient boosting did not. [V]
