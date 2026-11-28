# Second opinion on the first 2 x 2 results

Written 2026-11-27 (evening, -07:00) by a side session, for the main session and Oscar. Read-only work: nothing was committed, pushed, posted or run on the Spark's GPU. From the Spark I read result files under `sealed/` and the baseline result files, copied the small ones (test predictions, `result.json`, live result lines for task 0) to a scratch folder on the Mac, and analysed them there with NumPy. One mistake to disclose: my first copy command included the checkpoint folders and pulled about 6 GB over the network for two minutes before I stopped it and deleted the partial copy. Nothing on the Spark was written or changed.

Scripts are in `2x2-second-opinion-scripts/` beside this note (`a1.py` to `a5.py`). They expect the copied files under `spark/` and `live/` in the working folder.

**Correction, added later the same evening.** The "label noise" row in section 2 was an assumption and is wrong. Per-seed outcomes show the 3-seed benefit label is about 0.8 reliable (two independent seed pairs correlate at 0.57). More seeds per point would buy little. See `2x2-learning-plan.md`, which also supersedes the ranked steps in section 4 where they differ.

## Labels

- **[V]** verified: I ran it in this session and saw the result.
- **[S]** stated in a source I read (named), not re-run.
- **[A]** assumption or my own reasoning.
- **[U]** unknown.

Everything I computed beyond the pre-registered table is **exploratory**: chosen after the results were unblinded, several dozen numbers in all, with no multiplicity correction. Treat p-values below as descriptive.

## Short answers

1. **The four deciders learned to predict whether X-VLA will succeed, weakly. None of them learned where a π0.5 call helps.** On the population the metric uses, the escalation score ranks benefit at chance. [V]
2. **The primary's verdict under the logged rule is "reversed", and it must be reported as that. As evidence that X-VLA's information hurts, it is not credible.** It rests on about five test points in tasks 4 and 5. Removing one episode moves the interval across zero. [V]
3. **The main cause is variance, in three places: the inclusion weights, three seeds per branch, and a 2,000-input network trained on an effective 200 points.** The features and the 2 x 2 design are not shown to be wrong; the experiment cannot see them either way. [V for the numbers, A for the ranking of causes]
4. **Task 0 live numbers are not a ranking of deciders.** Usage follows each decider's λ, and success follows usage. They become interpretable only against usage-matched baselines. [V for task 0, A for the generalisation]
5. **Recommendation: no DAgger. Spend the GPU on the matched-usage baselines, the CPU on a handful of ten-minute checks, and stop new runs by the end of Nov 30.**

## 1. What the results support

### 1.1 The table reproduces [V]

From the saved test predictions I recomputed gain at 10% per task and pooled, the primary (-0.0069) and its interval ([-0.0159, -0.0003], same bootstrap seeds). All match `benefit.txt`.

A correction to the brief: the logged rule in amendment 78 (6) has three outcomes, and `benefit.txt` prints `VERDICT: reversed`. [S] The brief quoted only "supported if the interval is above zero".

### 1.2 The primary is driven by a handful of heavy points [V]

Per-task primary contrast (`cascader_hist` minus `router_hist`, gain at 10%):

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contrast | +0.0049 | -0.0094 | -0.0005 | -0.0001 | -0.0284 | -0.0342 | +0.0058 | 0.0000 | +0.0007 | -0.0077 |

- **Task 5 is one point.** `explore/t5/i1/r1/dp05` has weight 282 (3.3% of the task's weight) and labels b1 = 3/3, b2 = 0/3. `cascader_hist` puts it inside its top 10%, `router_hist` does not. That point alone is -0.0327 of the task's -0.0342.
- **Task 4 is four points** from four episodes (two are exploration runs of the same initial state), all at decision point 1, all successful-parent points where the call hurt (`base/t4/i11`, `base/t4/i48`, `explore/t4/i1/r0`, `explore/t4/i1/r1`). `cascader_hist` has them in its top 10% and `router_hist` does not; they sum to the task's -0.0284.
- **Task 9 is a fraction of one point.** `explore/t9/i12/r0/dp15` (weight 807, 4.6% of the task's weight, labels 3/3 and 0/3) sits on the 10% boundary. `router_hist` is charged 6% of it and `cascader_hist` 22%, by the tie interpolation.

Sensitivity, same bootstrap otherwise:

| change | primary | 95% interval | verdict by the rule |
| --- | --- | --- | --- |
| none (as registered) | -0.0069 | [-0.0159, -0.0003] | reversed |
| drop episode `explore/t5/i1/r1` | -0.0036 | [-0.0082, +0.0003] | not shown |
| drop episode `explore/t4/i1/r1` | -0.0057 | [-0.0137, +0.0002] | not shown |
| leave one task out (range over the 10) | -0.0038 to -0.0083 | not computed | |

Other views of the same contrast:

- **Tasks as the unit:** mean -0.0069, standard deviation across tasks 0.0138, t = -1.58, sign-flip p = 0.18. Six tasks negative, three positive, one zero.
- **Other registered summaries of the same contrast point the other way:** gain at 20% is +0.0040 [-0.0008, +0.0096], gain at λ +0.0022, failure AUROC +0.037. [S, `benefit.txt`]
- **What the interval does not cover:** it resamples test episodes with the trained deciders and thresholds held fixed. It has no training-seed variation and no task-to-task variation. [S, `benefit.py`] So it is narrower than the uncertainty about "does X-VLA information help a decider trained this way".

### 1.3 Neither side of the primary beats chance [V]

- **Shuffled scores** (within task, 4,000 draws) give a pooled gain at 10% of +0.0028, standard deviation 0.0034, 95% range [-0.0037, +0.0098]. All four observed gains (-0.0034, +0.0043, +0.0041, -0.0026) are inside it.
- **The difference of two independent shuffles** has standard deviation 0.0048, and exceeds 0.0069 in size 16% of the time. The real variants' scores are correlated (rank correlation 0.63 to 0.97 within task), so this overstates the null spread somewhat. [A]
- **Benefit ranking:** the weighted AUROC of the score for "b2 beat b1" is 0.497, 0.498, 0.483, 0.502 (mean over tasks). Per task it swings from 0.17 to 0.87.
- **Ceiling:** ranking by the realised label difference would give +0.052 at 10% (label-peeking, so inflated by seed noise). Random gives +0.0026. The deciders are at random, not partway to the ceiling.

So the primary compares two rankings that carry no detectable benefit signal on this population. Its sign is which of them happened to place a few heavy harmful points above the 10% line.

### 1.4 What signal there is [V]

- **Success level, yes.** Test loss beats the constant predictor on average for both heads (continue: -0.026 to -0.041; one call: -0.055 to -0.067, weighted), in 5 to 8 of 10 tasks. Weighted AUROC of 1 - p_continue for "b1 at most 1/3" is 0.68 to 0.72 without step stratification (the registered step-stratified figure is 0.61 to 0.65).
- **Benefit, faintly, inside failing episodes only.** Restricted to failed-parent points and unweighted, the score beats shuffling at 10% by +0.004 to +0.009 for all four variants (permutation p 0.01 to 0.11). No difference between variants. One of ten lens-by-rate combinations I tried; do not lean on it.
- **A trivial rule does better at 20%.** "Escalate the earliest decision points first" gives +0.0130 at 20% against +0.0052 for random, and beats each learned decider there by +0.013 to +0.017 (intervals exclude zero, same episode bootstrap). At 10% it ties them.

### 1.5 Facts about the labels worth keeping [V]

| | steps 0-170 | 170-340 | 340-520 |
| --- | --- | --- | --- |
| failed parent: share of points with b2 - b1 ≥ 1/3 | 47% (138/291) | 18% (57/324) | 8% (17/211) |
| successful parent: same | 4% (13/298) | 2% (3/162) | 0% (0/10) |
| successful parent: share with b2 - b1 ≤ -1/3 | 7% (20/298) | 4% (7/162) | 0% |

- One π0.5 call at a failed-parent point raises success by +0.02 to +0.18 depending on the task (weighted mean of b2 - b1; +0.13 to +0.18 in seven tasks, lower in tasks 2, 4 and 7). At successful-parent points it is about zero, and -0.11 in task 4.
- Parent outcome is inferred here from each point's weight (the lighter stratum of each parent kind), not read from the plan. [A]
- **Benefit is not monotone in failure risk.** Among failed-parent points, a lower p_continue goes with less benefit (AUROC 0.33 to 0.35 for 1 - p_continue against "b2 beat b1"). The target is "failing but still rescuable", which mostly means early. A failure predictor alone does not find it.

## 2. Likely causes, and a quick check for each

Ordered by how much I think each contributes. [A for the order]

| cause | evidence | quick check |
| --- | --- | --- |
| **Inclusion weights** | Weights span 2 to 1,078; within a task the ratio reaches 259 (task 3) and 388 (task 9). Test effective sample size 27 to 82 of 111 to 139 points; training 187 to 257 of 1,018. Successful-parent points hold 59% to 94% of a task's weight. One point can be 5.6% of a task. [V] | Report the effective sample size and the heaviest point's share beside every number. Redo the table by stratum, and with weights capped (say at 10 times the task median). Exploratory. Minutes on CPU. |
| **Metric definition** | "10% of the weight" is two or three points in tasks 3, 5 and 9. The failed-parent share of the weight is below 10% in tasks 3 and 9, so a perfect decider cannot fill its 10% with useful points there. Ties on heavy points are interpolated. [V] | Same as above. Also report gain against rate within the failed-parent stratum. |
| **Label noise** | Three seeds give labels in thirds. At a point with true success near 0.5 on both branches, the difference has standard deviation about 0.41. [A, binomial arithmetic] Among successful-parent points, 2% to 4% show a rescue and 4% to 7% a harm of a third or more, much of it plausibly seed luck. [V for the counts, A for the reading] | Split-seed reliability: from the branch records, estimate b2 - b1 from seed 1 alone and from seeds 2-3 alone, and correlate. Low correlation means the target is mostly noise at this seed count. Minutes on CPU. |
| **Too few points for the model** | 2,062 to 2,179 inputs, about 1.2 million parameters, 1,018 training points. One epoch is four gradient steps. Members of one fold stop anywhere from epoch 0 to 26; fold 0 stops every member at epoch 0. Validation effective sample size is 28 to 82, so the stopping epoch is mostly noise. [V] | Learning curve: retrain on 25%, 50%, 75% of the training episodes and see whether anything rises. About ten minutes per full training run. [V, from the file times of `train.start` and `train.end`: six deciders, ten folds, five members each] |
| **Training-seed noise in the contrast** | Not measured. Per-member predictions were not saved; the interval holds the models fixed. [U] | **Do this one first.** Retrain everything with a different seed base five to ten times and rerun `benefit.py` each time. If the primary changes sign across replicates, that settles question 1. About an hour of CPU. The seed base is not a command-line option (`member_seeds(..., base=0)`), so it needs a small patch on a scratch copy, outside the pinned code. |
| **Features drowned by embeddings** | History and X-VLA features are 13 to 117 of roughly 2,100 standardised inputs. Variants' scores correlate at 0.63 to 0.97. [V] The signal study that motivated them used logistic models, not this network. [S, amendment 78 (1)] | Fit the 2 x 2 with a logistic or ridge model on the non-embedding inputs only (optionally plus a few principal components of the embeddings). Exploratory. |
| **Score as a difference of two heads** | The score's spread is 0.02 to 0.10 against 0.09 to 0.32 for p_continue. Both heads mostly track "will this succeed". [V] | Train one head on the difference directly, or on "b2 - b1 ≥ 1/3". Exploratory. |
| **λ rule** | It takes the best point of a noisy curve. Chosen validation gains of 0.02 to 0.06 at rates of 0.3 to 0.5 came back as about 0.00 on test. Router and `router_hist` are below random at λ. [V, S] | Nothing to run; report it. A λ fixed by target rate would remove this, as an exploratory variant. |
| **Platform** | Labels come from the L4, live runs from the Spark. V49 reports statistical agreement. [S] It cannot explain a weak offline signal, since training and test labels are both from the L4. [A] | None needed now. |

I found no bug. The profile check passed on all 1,390 points, the four variants are on identical rows, and no point was left out. [S, `train.log`]

## 3. The live results

### 3.1 Task 0 as it stands [V]

| | successes | paired against X-VLA alone (30/50): only decider, only X-VLA, exact p | π0.5 share | λ | escalation rate in steps 0-170 |
| --- | --- | --- | --- | --- | --- |
| router | 31 | 8, 7, p = 1.00 | 24% | 0.0068 | 0.36 |
| router_hist | 38 | 12, 4, p = 0.08 | 51% | 0.0021 | 0.71 |
| cascader_nospread | 29 | 1, 2, p = 1.00 | 3.5% | 0.0426 | 0.04 |
| cascader_hist | 32 | 8, 6, p = 0.79 | 44% | 0.0134 | 0.57 |

- π0.5 alone on task 0 is 49/50 with no time limit. [V] I did not rescore it under the limit.
- **Usage is set by λ.** The live score has mean -0.010 to +0.011 and standard deviation about 0.02 for all four. A λ of 0.002 against 0.043 on a score that narrow is the difference between 51% and 3.5% usage. Fold 0's deciders are the ones that stopped at epoch 0, after four gradient steps.
- **Success tracks usage.** `router_hist` used the most π0.5 and succeeded most. A straight line from X-VLA alone (30 at 0%) to π0.5 alone (49 at 100%) passes 39.7 at 51% and 38.4 at 44%. `router_hist` is at 38, `cascader_hist` at 32. The straight line is a crude stand-in for a random baseline, and ignores the time limit. [A]
- Between deciders, only `router_hist` against `cascader_nospread` reaches p = 0.035 (12 against 3), which is the highest usage against the lowest.

### 3.2 `pi05_failed_requests` is not a fault [S, read in code]

`episode.py:379` adds the failed attempts of the simulated network (`network.py`, the `p_fail` draw and the timeout). SPEC §5 says the server is called once per request, for the attempt that succeeds. So a gate log of all-200 with a nonzero counter is the expected picture. The rates fit: 32 failed against 446 requests for the router (6.7% of attempts), 7.3% for π0.5 alone over all tasks. [V] That is above `p_fail` = 5% because slow draws also count as failures. I did not compute the expected slow share. [U]

### 3.3 How to analyse and report

1. **Report the registered comparisons as registered:** each decider at its validation λ, all 500 episodes, paired differences, with usage, simulated time and end reasons beside success. Do not hide the usage gap; put it in the same row.
2. **Success against π0.5 usage is the main figure.** X-VLA alone at 0%, π0.5 alone at 100% (with and without the limit), random escalation at several usage levels as a line, the four deciders as points.
3. **Each decider against random escalation at its own usage** is the test of whether anything was learned. §10 already specifies it, matched per fold. With no DAgger it is matched to the round-1 deciders; say so as a deviation.
4. **Add a step-matched random baseline** (exploratory): random escalation with the decider's own escalation rate in each third of the episode. The deciders escalate mostly early, and early is where the benefit is. This control separates "knows to escalate early" from "knows which states".
5. **Decider against decider at equal usage:** compare each one's success minus the random line at its usage. Only a λ sweep gives a direct equal-usage comparison; §10 lists it as optional.
6. **Paired test:** exact McNemar on the 500 pairs for each key comparison, and beside it the per-task differences with a task-level sign-flip test, since deciders differ by fold. Report the eight tasks without 4 and 8 and those two apart, as §10 requires.
7. **Power:** π0.5 alone under the limit is 451/500 against X-VLA's 408/500, so the whole gap is 8.6 points. A smart allocation can beat a random one at the same usage by a fraction of that. `cascader-design.md` estimated a 95% half-width of about 2.8 points for 500 pairs. [S] Expect "not shown" for decider-against-decider differences, and plan the text for it.

## 4. Next steps, ranked

Measured cost: 50 live episodes took 946 to 1,963 s with 8 shards. [V, `live.status`] So a 500-episode method is about 2.6 to 5.5 hours. [A]

| rank | what | cost | status |
| --- | --- | --- | --- |
| 1 | Let live round 1 finish untouched. | running, to about 17:00Z Nov 28 | registered |
| 2 | Random escalation matched to each of the four deciders' usage, per fold, and the fixed schedule. | 4 to 5 runs, about a day | registered in §10; matching to pre-DAgger deciders is a deviation to log |
| 3 | CPU checks from section 2, in this order: seed replicates of the primary, weight sensitivity and by-stratum table, split-seed reliability, low-dimensional logistic 2 x 2, direct benefit target, learning curve. | about half a day of someone's attention, little compute | all exploratory |
| 4 | Step-matched random baseline for `router_hist` and `cascader_hist`; random at two more fixed usage levels for the frontier. | 3 to 4 runs, under a day | exploratory |
| 5 | Optional, only if 2 and 4 are done by Nov 30: two more λ values for `router_hist` and `cascader_hist`. | 4 runs | §10 lists it as optional |
| 6 | Stop new runs at the end of Nov 30 (UTC). Dec 1-2 for figures, README, poster. | | |
| not recommended | DAgger round. | 2.5 days or more | see below |
| not recommended | More branch labels on the Spark. | unknown | see below |

**Why not DAgger.** [A]
- DAgger corrects the state distribution of a decider that already works. These do not beat random offline, so the collectors would be near-random early escalators.
- It adds perhaps half again as many points with the same three-seed noise and the same weighting. Section 2 says the problem is variance in the estimator and the model, not which states were visited.
- It takes all the remaining GPU time and displaces the baselines, without which the live results cannot be read.
- It mixes L4 and Spark labels (U27 in `ASSUMPTIONS.md`).
- `plan-to-dec3.md` already names it as the first cut and says the pre-DAgger deciders are a complete result. [S] Dropping it is the plan working, not a deviation from it. The roles in amendment 78 (7) for DAgger simply go unused; log that.

**Why not more labels now.** Whether L4 snapshots restore on the Spark is unknown to me. [U] More labels would be analysed after unblinding, so they are exploratory unless a new amendment fixes the analysis before they are opened. The one version I would consider is a replication: freeze today's deciders, collect fresh test points, register the same primary beforehand. Given section 1.3, the likely outcome is "not shown", and it costs a day. Rank it below everything in the table.

**What must stay as registered.**
- The `benefit.txt` table in full, all 64 intervals, the primary and the verdict "reversed" as printed. No points, episodes or tasks dropped from it.
- The four live deciders at their validation λ, the §10 baselines and paired comparisons, tasks 4 and 8 reported apart.
- `cascader_chunkspread` and `cascader_full` offline only.

**What must be labelled exploratory.**
- Everything in sections 1.2 to 1.5 and 2 of this note, any other rate, model, weighting, target or rule, the step-matched baseline, and any retraining.
- The count of exploratory looks should be stated, as amendment 78 (9) implies.

## 5. Poster and README

**Lead with what happened, in this order.**
1. The question, and that the analysis was fixed and logged before any label was opened (amendment 78, with its timeline).
2. The registered result, verbatim: primary -0.0069 [-0.0159, -0.0003], verdict "reversed" under the three-outcome rule. The claim "seeing X-VLA's commands improves the decision, memory held equal" is not supported.
3. Straight after it, in a box marked exploratory: no decider beats random escalation at the same rate; the primary rests on about five heavily weighted points; removing one episode gives "not shown"; with tasks as the unit p = 0.18; and the seed-replicate result once it exists. Conclusion in plain words: "We could not measure the value of X-VLA's information with 1,390 labelled points. The registered contrast is negative but fragile."
4. The live figure: success against π0.5 usage with the matched random baselines, and paired differences with intervals.
5. What the data did show: a single π0.5 call rescues a failing episode often when it comes early (47% of early failed-parent points) and rarely late (8%); it does nothing or slight harm in episodes that would have succeeded; success is predictable at AUROC 0.61 to 0.65. Mark the step table exploratory.
6. Why the experiment was underpowered, as numbers: effective sample size, weight range, three seeds, four gradient steps per epoch.
7. What would fix it: more seeds per branch at fewer, better-chosen points; a low-dimensional decider; a target that is the benefit itself; capped weights or a stratified estimand, registered in advance.

**Things to avoid.**
- Presenting `router_hist`'s +0.0043 [+0.0009, +0.0083] as a finding. It is one of 64 unadjusted intervals, and its excess over random includes zero.
- Saying "the cascader is worse than the router". The evidence does not carry that.
- Rewording the verdict to "not shown". The rule said "reversed"; report it, then show why it is fragile.
- Ranking deciders from live success without usage beside it.
- Calling the nonzero `pi05_failed_requests` an error.

**Deviations to state.** No DAgger round; §10 baselines matched to pre-DAgger deciders; labels from the L4 and live runs on the Spark; any exploratory analysis that makes it onto the poster.

## Not checked

- Training-seed variation of the primary. [U]
- Whether a simpler model or a direct benefit target finds signal. [U]
- π0.5 alone on task 0 under the time limit. [U]
- Live results beyond task 0; the round was still running.
- `cascader_chunkspread` and `cascader_full` predictions.
- The feature code in `variants.py` beyond its header and feature list, and `data.py`, `history.py`, `collectors.py`. I did not review them for bugs.
