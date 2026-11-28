# Learning curve: are the deciders data-limited?

Written 2026-11-27 (night, -07:00) by the side session that wrote `2x2-second-opinion.md` and `2x2-learning-plan.md`. CPU only, on the Mac. The Spark was not touched. Everything here is exploratory, on labels that are already unblinded. The registered result is unchanged.

Scripts, logs and saved scores are in `2x2-learning-curve-scripts/` in the shared notes folder. They read `points.npz` and `seeds.json` from `2x2-learning-plan-pilot/`.

Labels: **[V]** I ran it. **[S]** stated in a source or read in code. **[A]** my reasoning. **[U]** unknown.

## Verdict

**No. The evidence does not support "more data, from DAgger or new labels, would make the deciders learn a usable benefit ranking in time".**

- **The registered MLP deciders are not data-limited in the range DAgger can reach.** They are at chance with 272 training points and at chance with 1,037. No slope is distinguishable from zero. [V]
- **The low-dimensional linear cascader is data-limited, mildly.** Its ranking improves with data, and that slope is distinguishable from zero. But a DAgger-sized addition is worth about +0.03 of AUROC, and it would still not be distinguishable from "escalate earliest first". [V for the fit, A for the extrapolation]
- **Exploration-parent points, the nearest thing we have to on-policy states, carry less benefit than base points, not more.** The comparison has little power, but nothing in it favours DAgger. [V]
- **A by-product settles an open question: the "reversed" primary was a training-seed draw.** Across ten seed sets it is the most negative of the ten, and the mean is +0.0009. [V]

My confidence:
- That +50% more Phase-1-like data leaves the registered MLPs near chance: about 90%. [A]
- That more data improves the linear cascader's ranking at all: about 85%. [A]
- That +50% makes any of these clearly better than "earliest first": about 15%. [A]

So dropping DAgger is justified on this evidence. What would change my mind is in the last section.

## Method

- **Code.** The registered training, unchanged: `vlar_train.train.train_seeds` with `TrainConfig()` defaults and `member_seeds`, from an export of `cascader-2x2` at `77516ae`. Five members, early stopping on the fold's validation task, weighted loss. The λ step is skipped; none of the metrics needs it.
- **Check.** With the registered seeds and all the data, the local run reproduces the Spark's saved test predictions to 2e-7 and the same stopping epochs (fold 1, both deciders), and the same metrics for both deciders. [V]
- **Splits.** Leave one task out, as registered: 8 training tasks, 1 validation task, 1 test task.
- **Subsampling.** Whole episodes, the same share of episodes in every stratum (task by parent kind by parent outcome) of the 8 training tasks. The validation and test tasks are left whole. Both deciders get the same episodes in a draw.
- **Draws.** MLPs: 10 per fraction, each with its own seed base as well. At 100% the draws differ only in training seeds; seed base 0 is the registered run. Linear models: 20 per fraction below 100%; at 100% they are deterministic, so one value.
- **Training sizes** (mean per fold): 272, 522, 784 and 1,037 points; 120, 240, 358 and 478 episodes.
- **Linear low-dimensional models.** As in the pilot, in scikit-learn, not through the training code: (a) two logistic heads with inclusion weights on the registered labels; (b) ridge on the fresh benefit, unweighted. Inputs: 27 (`router_hist` without embeddings) and 131 (`cascader_hist` without embeddings). They train on the fold's 8 training tasks here; the pilot used 9.
- **Metrics on the held-out tasks**, mean over the ten tasks: benefit AUROC (inclusion-weighted, registered labels); the same within step thirds; the gain lens at 10% and 20% (inclusion-weighted gain minus random at the same rate).
- **Trend.** Least squares of the metric on the log of the training points, over all draws. Intervals: 95% from 300 joint resamples of test episodes (within task) and of draws.

References on the same data [V]:

| | benefit AUROC | within-third AUROC | gain lens 10% | gain lens 20% |
| --- | --- | --- | --- | --- |
| shuffled scores, standard deviation | 0.043 | 0.041 | 0.0034 | 0.0044 |
| escalate earliest first | 0.630 | 0.602 | +0.0006 | +0.0078 |

## 1. Learning curves

Mean over draws, with the standard deviation over draws in brackets. [V]

**Registered MLP, `router_hist`**

| training points | benefit AUROC | within-third AUROC | gain lens 10% | gain lens 20% |
| --- | --- | --- | --- | --- |
| 272 | 0.482 (0.063) | 0.499 (0.057) | -0.0014 (0.0025) | -0.0041 (0.0053) |
| 522 | 0.506 (0.037) | 0.512 (0.026) | -0.0002 (0.0027) | -0.0028 (0.0023) |
| 784 | 0.499 (0.023) | 0.514 (0.025) | -0.0004 (0.0036) | -0.0032 (0.0043) |
| 1,037 | 0.489 (0.010) | 0.512 (0.012) | -0.0006 (0.0021) | -0.0074 (0.0018) |

**Registered MLP, `cascader_hist`**

| training points | benefit AUROC | within-third AUROC | gain lens 10% | gain lens 20% |
| --- | --- | --- | --- | --- |
| 272 | 0.487 (0.044) | 0.500 (0.055) | +0.0001 (0.0028) | -0.0025 (0.0046) |
| 522 | 0.517 (0.036) | 0.513 (0.028) | -0.0005 (0.0027) | -0.0030 (0.0040) |
| 784 | 0.510 (0.016) | 0.521 (0.019) | -0.0001 (0.0025) | -0.0031 (0.0036) |
| 1,037 | 0.508 (0.014) | 0.523 (0.015) | +0.0002 (0.0022) | -0.0039 (0.0017) |

**Linear, 131 inputs (cascader with history), logistic heads**

| training points | benefit AUROC | within-third AUROC | gain lens 10% | gain lens 20% |
| --- | --- | --- | --- | --- |
| 272 | 0.575 (0.044) | 0.560 (0.042) | +0.0019 (0.0045) | +0.0027 (0.0039) |
| 522 | 0.626 (0.042) | 0.586 (0.035) | +0.0050 (0.0037) | +0.0042 (0.0043) |
| 784 | 0.649 (0.040) | 0.608 (0.031) | +0.0059 (0.0032) | +0.0064 (0.0041) |
| 1,037 | 0.687 | 0.623 | +0.0080 | +0.0043 |

**Linear, 27 inputs (router with memory), logistic heads**

| training points | benefit AUROC | within-third AUROC | gain lens 10% | gain lens 20% |
| --- | --- | --- | --- | --- |
| 272 | 0.515 (0.051) | 0.523 (0.039) | +0.0030 (0.0033) | +0.0028 (0.0036) |
| 522 | 0.512 (0.042) | 0.531 (0.030) | +0.0027 (0.0034) | +0.0019 (0.0056) |
| 784 | 0.536 (0.035) | 0.550 (0.026) | +0.0049 (0.0032) | +0.0034 (0.0040) |
| 1,037 | 0.543 | 0.566 | +0.0054 | +0.0026 |

The ridge versions are in `analyse.out`. Their AUROC rises the same way (131 inputs: 0.563 to 0.622; 27 inputs: 0.510 to 0.572) and their gain lens stays near zero.

### Slopes [V]

Change per doubling of the training data, with the 95% interval.

| model | benefit AUROC | within-third AUROC | gain lens 10% | gain lens 20% |
| --- | --- | --- | --- | --- |
| MLP `router_hist` | +0.005 [-0.017, +0.033] | +0.008 [-0.018, +0.038] | +0.0004 [-0.0025, +0.0024] | -0.0012 [-0.0045, +0.0018] |
| MLP `cascader_hist` | +0.010 [-0.010, +0.036] | +0.012 [-0.014, +0.040] | +0.0000 [-0.0029, +0.0024] | -0.0006 [-0.0043, +0.0024] |
| linear 131, logistic | **+0.050 [+0.014, +0.070]** | **+0.031 [+0.006, +0.052]** | +0.0027 [-0.0011, +0.0054] | +0.0023 [-0.0020, +0.0068] |
| linear 131, ridge | +0.020 [-0.005, +0.041] | **+0.030 [+0.002, +0.051]** | -0.0015 [-0.0056, +0.0018] | -0.0022 [-0.0085, +0.0022] |
| linear 27, logistic | +0.012 [-0.011, +0.038] | +0.018 [-0.004, +0.043] | +0.0011 [-0.0021, +0.0033] | +0.0003 [-0.0034, +0.0043] |
| linear 27, ridge | +0.025 [-0.000, +0.048] | **+0.029 [+0.001, +0.051]** | +0.0001 [-0.0021, +0.0015] | -0.0007 [-0.0036, +0.0024] |

- **MLPs: no slope is distinguishable from zero on any metric.**
- **Linear models: the AUROC slopes are positive, and four of the eight intervals exclude zero (bold).** No gain-lens slope is distinguishable from zero.

### Extrapolation to a DAgger-sized addition

**The number I assume: +640 labelled points from about +320 episodes, which is +49% of the 1,296 points used.** [A] Reasons: §9 samples DAgger's decision points "as in §7", and Phase 1's tranches A and B were each 64 points per task. Tranche B's 587 used points came from 312 episodes. [V] Per fold that takes training from about 1,037 to about 1,545 points.

Predicted value at +49%, and that prediction minus "earliest first" on the same resample. [V for the arithmetic; the log-linear form is A]

| model | metric | fitted now | predicted at +49% | minus earliest first |
| --- | --- | --- | --- | --- |
| MLP `router_hist` | benefit AUROC | 0.498 | 0.501 [0.452, 0.574] | [-0.195, -0.038] |
| MLP `cascader_hist` | benefit AUROC | 0.514 | 0.520 [0.476, 0.590] | [-0.180, -0.015] |
| MLP `cascader_hist` | gain lens 20% | -0.0036 | -0.0040 [-0.0137, +0.0049] | [-0.0230, -0.0006] |
| linear 131, logistic | benefit AUROC | 0.672 | 0.701 [0.615, 0.743] | [-0.053, +0.125] |
| linear 131, logistic | within-third AUROC | 0.619 | 0.637 [0.567, 0.687] | [-0.067, +0.117] |
| linear 131, logistic | gain lens 10% | +0.0073 | +0.0088 [-0.0013, +0.0153] | [-0.0072, +0.0158] |
| linear 131, logistic | gain lens 20% | +0.0069 | +0.0083 [-0.0032, +0.0214] | [-0.0141, +0.0197] |

- With +49% the MLPs are predicted to stay below "earliest first" on AUROC (both intervals exclude zero).
- The best linear model is predicted to gain +0.029 of AUROC and +0.0013 to +0.0016 on the gain lens. Its difference from "earliest first" includes zero on every metric.
- On the log-linear fit, the linear cascader would need about 1.6 doublings, roughly three times today's data, to reach an AUROC of 0.75. [A] The curve is by episodes, so that means roughly 1,000 more parent episodes, not more points from the same ones. [A]

### Cautions on the curve

- **The validation task is not subsampled.** DAgger would also add validation points, which would steady early stopping and λ. That benefit is not in the curve. [A]
- **Subsampling tests "more data like Phase 1".** It says nothing direct about on-policy states; section 2 is the only proxy.
- **The 100% point of the linear models is one value from a noisy quantity.** Trained on 9 tasks instead of 8 (the pilot), the 131-input logistic model scores 0.613, not 0.687. [V] Adding 12% more data of a different task moved it down by about as much as the fitted trend moves it up.
- **At 100% the MLP's spread over draws is seeds only**, so it is smaller than at lower fractions by construction.

## 2. Base against exploration points

Used points only (1,296). Parent outcome and kind are read from the label entries. [V]

| group | points | episodes | points with b2 - b1 ≥ 1/3 | share | mean fresh benefit |
| --- | --- | --- | --- | --- | --- |
| base parents | 709 | 285 | 154 | 21.7% | +0.108 |
| exploration parents | 587 | 312 | 74 | 12.6% | +0.062 |
| exploration, before any π0.5 call | 153 | 120 | 19 | 12.4% | +0.039 |
| exploration, after at least one call | 434 | 268 | 55 | 12.7% | +0.071 |

Within the cell where benefit is largest (failed parent, steps 0-170) the fresh benefit is +0.303 at base points (n = 158) and +0.183 at exploration points (n = 133).

Held-out benefit AUROC within each group (unweighted, fresh labels, mean over tasks):

| score | base parents | exploration parents | exploration, after a call |
| --- | --- | --- | --- |
| shuffle standard deviation | 0.031 | 0.044 | 0.048 |
| escalate earliest first | 0.704 | 0.687 | 0.729 |
| MLP `router_hist`, mean of 10 seed sets | 0.583 | 0.468 | 0.493 |
| MLP `cascader_hist`, mean of 10 seed sets | 0.565 | 0.483 | 0.485 |
| linear 27, logistic | 0.586 | 0.504 | 0.532 |
| linear 131, logistic | 0.593 | 0.621 | 0.643 |

Exploration minus base, with a 95% interval from episode resamples: earliest first -0.018 [-0.124, +0.058]; MLP `cascader_hist` -0.082 [-0.180, +0.052]; linear 131 +0.027 [-0.053, +0.131]; linear 27 -0.082 [-0.162, +0.033].

What this says:

- **The base rate of benefit is lower after random escalations**, about 0.6 of the base-point rate. Some of the rescue has already happened by then, or what is left is harder. [A for the reason]
- **No score ranks exploration points detectably better than base points.** The MLPs do nominally worse there; only the 131-input linear model does nominally better.
- **Power is low.** Each group's AUROC has a shuffle standard deviation of 0.03 to 0.05, and the intervals on the differences are about ±0.1 wide. Only a difference of 0.1 or more would show. The exploration parents escalated at random with probability 0.1 [S, SPEC §7], which is a weak stand-in for a decider's own states.

## 3. By-product: seed replicates of the registered primary [V]

At 100% of the data, ten seed sets for each decider. Seed set 0 is the registered run and reproduces it.

`cascader_hist` minus `router_hist`, gain at 10%:

| seed set | 0 (registered) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contrast | -0.0069 | +0.0035 | +0.0047 | +0.0041 | +0.0040 | -0.0006 | -0.0003 | +0.0006 | +0.0004 | -0.0006 |

- Mean +0.0009, standard deviation 0.0035, four of ten negative. The registered value is the lowest of the ten, about two standard deviations below the mean.
- The same contrast at 20% has mean +0.0035 and standard deviation 0.0031.
- These are point estimates; I did not rerun the registered bootstrap for each seed set.

This answers question 1 of the first note: the registered interval held the trained models fixed, and training-seed variation alone is as large as the effect. The registered verdict stays "reversed" and is reported as printed. This table belongs beside it, marked exploratory.

## What would change the verdict

- **A sharper test.** On an equal-weight test set (rank 4 of `2x2-learning-plan.md`), if the linear cascader's gain lens rose with data and cleared "earliest first", more labels would be worth buying. Today's gain lens cannot show a slope of this size.
- **A model with a steeper curve.** I tried linear models and one gradient-boosting setting. A small nonlinear model on the 131 inputs with a clearly rising gain lens would change the picture. [U]
- **Much more data than DAgger gives.** About three times today's episodes, by the log-linear fit, and only for the linear cascader. That is new parent rollouts plus labels, roughly a day or more of Spark GPU, with no guarantee the fit holds that far. [A]
- **On-policy states behaving unlike exploration states.** Section 2 cannot rule that out. Only a small labelled sample from live round 1's deciders would, and round 1 ran with recording off. [V, the task 0 result lines say `record: off`]

## Not checked

- Subsampling the validation task as well.
- The other two registered variants (`router`, `cascader_nospread`) and the four-variant contrasts under seed replicates.
- The registered bootstrap interval for each seed set.
- Nonlinear small models on the low-dimensional inputs beyond one gradient-boosting setting.
