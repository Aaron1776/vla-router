# Draft amendment: after the 2 x 2, without DAgger (for Miguel's review)

Draft by Oscar, 2026-11-28. Number-neutral: "[next number]" is given at logging.

This was written **after** amendment 78's offline results and part of its live round 1 were unblinded, and after exploratory analyses on the unblinded labels (`docs/notes/`). Everything below is fixed before any run it governs starts and before any of its results are seen. The amendment 78 result is reported verbatim, unchanged, and nothing here reinterprets it.

[next number]. §10, §17 amendments 55, 61, 63, 65, 78 (the user, 2026-11-28).

(1) **No DAgger round.** The DAgger round (amendments 61, 65) and the post-DAgger evaluation are not run. Amendment 78's live round 1 (the four 2 x 2 deciders at their validation λ, 2,000 episodes) is the evaluation of the learned deciders.

Reasons:
- the time left before the deadline (Dec 2);
- the offline result: no decider ranks branch benefit above chance (benefit AUROC 0.48 to 0.50).

An exploratory learning curve (`docs/notes/2x2-learning-curve.md`) supports this. The registered MLPs stay at chance from 272 to 1,037 training points, with no slope distinguishable from zero. A DAgger-sized addition of about 49% more points is predicted to leave them below "escalate earliest first". This decision was made after unblinding and is disclosed as such.

(2) **Four scripted runs.** These replace §10's random and fixed-schedule baselines, which §10 matches to post-DAgger deciders that will not exist. Each run covers all 500 task and initial-state pairs, with the evaluation rollout seeds and the `router` profile. Runs go to 520 steps uncut and are scored with and without the time limit from the event logs, as §10 states.
- **R1, positive control (privileged, not a decider).** The scenes are the 102 (task, initial state) pairs where the L4 base rollout failed (`References/l4-phase0/runs/base-runs.csv`; those runs used base-namespace seeds, not the evaluation seeds). In those scenes, escalate at every offered decision point in steps 0-169. Elsewhere, never.
  - That is at most 17 escalations per episode, so the budget never binds.
  - It uses scene identity and a past outcome, which §6 forbids as decider inputs. So it is reported as an upper reference for "knowing where X-VLA fails before", never as a decider.
  - Its measured usage, u_c, sets the level for R2 and R3.
- **R2, uniform random at u_c.** Each task's escalation rate is R1's escalation rate among offered points on that task.
- **R3, early-first at u_c.** Escalate at the first N offered decision points of every episode, and never after. N per task = round half up (R2's usage x R2's mean decision points per episode).
- **R4, uniform random matched to `cascader_hist`.** Per fold, the rate is `cascader_hist`'s round 1 escalation rate among offered points on its test task.

**Conventions.**
- **Usage** is pooled per task: π0.5 steps over executed steps.
- **Escalation rate** is escalation decisions over offered points, counting an escalation even if its request is later abandoned. For round 1, a point is offered when at least 10 steps of budget are left, rebuilt from the π0.5 steps so far.
- **Matching is in expectation only**, so realized usage and rate are reported beside every target.
- **Random draws** come from one stream, `vlar_ops.plan.unit_uniform(rollout_seed, "eval_random", decision index)`, for R2 and R4. A point escalates when its draw is below the rate. That gives common random numbers across the two runs, independent of model noise.
- **Whole episodes run live,** as in round 1, not amendment 55's shared trunk.
- **New runs log `can_escalate` and the budget left** at every decision point.
- **Reused, not rerun:** X-VLA alone (408/500) and π0.5 alone (479/500, 451 within the limit) are the existing Spark runs on the evaluation seeds.
- **Code:** branch `spark-eval` at the commit used, recorded in each run's provenance.
- **Runner:** the runs go through step 690, whose equality with step 680 is checked on the GPU before use.

(3) **Key comparisons.** These are paired over the same task and initial-state pairs and scored with the time limit. They are Holm-corrected as a family of three.
- **K1:** `cascader_hist` minus R4. The value of learning, §10.
- **K2:** R3 minus R2. Timing: does calling early beat calling at random at the same usage?
- **K3:** R1 minus R2. Where to call: does knowing which scenes fail beat random at the same usage?

Each uses the exact McNemar test on success (the test K1 to K3 are judged by) and a task-level sign-flip test on the per-task differences. Each is reported over 10 tasks, over the 8 without tasks 4 and 8, and for tasks 4 and 8 apart.

(4) **Secondary, registered, not corrected.**
- **Micro-randomized estimate of one call's effect,** on R2 and R4. In each run, every offered decision point escalates independently with a known probability, so at offered points in step third k (steps 0-169, 170-339, 340-519), the difference in final success between points that escalated and points that did not is an unbiased estimate of the excursion effect of one more call, with the random policy continuing after it.
  - Estimated per run and pooled, with task fixed effects.
  - 95% intervals by an episode-clustered bootstrap (2,000 resamples of episodes within task).
  - It measures the timing effect live, on states that follow earlier escalations, which is what the DAgger round would have measured.
- **Other comparisons:**
  - R1 minus R3;
  - the four 2 x 2 deciders against X-VLA alone;
  - `router_hist` minus R4 (unmatched; reported with both usages);
  - success against π0.5 usage for every method, with the R2 and R4 points.

(5) **Training-seed spread of amendment 78's primary (descriptive, already run, exploratory).** The deciders were retrained on the Mac with 9 new seed sets through the registered training code; seed set 0 reproduces the registered run. Across the 10 sets the primary's point estimate has mean +0.0009 and standard deviation 0.0035, and the registered -0.0069 is the lowest of them. This is reported beside the registered value and does not change the registered verdict.

(6) **Order and cut.** The runs go R1, R2, R3, R4; R3 needs R2's output. No new run starts after Nov 30 23:59 MST. What has not started by then is cut and reported. Every run is fixed at 500 episodes, with no stopping on a look.

Not run, and reported as not run:
- §10's fixed schedule;
- step-matched random;
- random matched to `router_hist`;
- the post-DAgger λ sweep.

The code for the fixed schedule and step-matched random exists but is unused.

(7) **Disclosures.**
- (i) The pilot-phase run records repeat batch A's run ids, so amendment 78's records view excluded the 1,221 pilot-phase runs. The pilot has no labelled points, and all 1,390 were used.
- (ii) Live round 1 stopped once (05:05Z Nov 28, an X-VLA server connection reset on one shard). It was relaunched unchanged, and one episode of `cascader_hist` fold 1 ran in a second server session.
- (iii) Within each task, round 1 runs the four deciders in a fixed order, with the same seeds for every method and the servers restarted for each job (step 680).
- (iv) `pi05_failed_requests` counts failed simulated network attempts, not requests (§5, amendment 13). Reports give failed attempts per request, and abandoned escalations from the event logs.
- (v) Exploratory analyses on the unblinded labels (about 180 fits) are in `docs/notes/`. None of them is a result of this amendment.
