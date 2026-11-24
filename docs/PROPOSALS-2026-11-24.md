# Proposals for the user's approval (2026-11-24): items 6 and 9 of the post-freeze list

**Status (2026-11-24, later): the user's decisions are in §0, which supersedes §2-§4 and §6 where they differ.** Originally: nothing here was approved or implemented. Numbers marked *est.* are estimates from the measurements named; the pilot
replaces them with measured values. Prices: $0.472 per VM hour (g2-standard-16 Spot with its L4); the spend ledger
reads **$9.76** (VM $7.12, disk $0.44, the $2.20 of image builds and other non-VM charges added with `spend charge`),
so about **$30 remains to the $40 soft limit** (the hard stop stays at $45; $5 is kept for storage and the download).


## 0. Decisions and revised plan (the user, 2026-11-24)

**Decisions.** 3a (common random numbers, shared trunk) approved: amendment 55 in SPEC §17, with a 20-episode live
audit per decider and round. 3b approved in principle, keeping outcome-stratified selection possible (§0.3 says how,
on the frozen harness). Pilot and stop rule approved, with the additions in §0.4. Worker fixes approved: `--max-jobs`
with supervisor restarts plus per-worker memory limits, and progress-gated leases. All are required before the pilot
boots. Task-affine claiming only if the pilot shows that restart overhead matters. Restarter (§7) approved; build it
before the first long unattended Phase 1 run. Keep the VM and disk until ~09:45Z 2026-11-25. Pilot boot approved
up to $2.50 once the worker fixes are tested and this section is written.

### 0.1 Budget, revised

Prices as in §1: $0.472 per VM hour; disk ~$0.33 per day while the VM exists. Ledger **$9.94** (VM $7.28, disk
$0.45, charges $2.20), so **$30.06 remains to $40**. Per-run VM-s are §1's estimates; the pilot replaces them.

**(a) Evaluation at q = 0.7.** The profile costs assume ~5 escalations per learned-decider episode, so q = 0.4 was
optimistic. The trunk plan at q = 0.7, first escalation 40% into the episode, with scripted baselines at their upper
bound (escalating from the first point):

| Item | VM-s | $ |
| --- | --- | --- |
| Trunk: 500 × 76 | 38,000 | 4.98 |
| Pre-DAgger suffixes: router 500 × 0.7 × 0.6 × 16, `cascader_full` × 80 | 3,360 + 16,800 | 2.64 |
| Post-DAgger suffixes: router, `cascader_full`, `cascader_nospread` | 23,520 | 3.08 |
| Fixed-schedule (≤ whole episode, router profile) | ≤ 8,000 | 1.05 |
| Random × 2 (same) | ≤ 16,000 | 2.10 |
| π0.5 alone: 500 × 12.7 | 6,350 | 0.83 |
| Audit (amendment 55): 20 fully live episodes per decider and round | 1,920 + 2,240 | 0.55 |
| **Evaluation** | | **15.23** |

The proposal's ~$15.0 plus the audit. Contingencies paid from the reserve: `xvla_only` run separately if the trunk
check fails ($0.46); `/features` recomputed at trunk snapshots if `/sample` features differ from `/features`
(~$0.7 *est.*). `cascader_chunkspread` (post-DAgger, ~$0.40-1.15 for q 0.2-0.7, with its audit) and the λ sweep
(router and `cascader_full` at 2 more λ: ~$1.5-5.3) are funded only from what a lower measured q saves. Once the
pre-DAgger evaluation has measured q for the router and `cascader_full`, the post-DAgger block is re-projected at
that q. The difference from its $6.52 line funds chunkspread first, then the λ sweep if enough is left.
For reference, at q = 0.4 the evaluation is $12.78 (saving $2.45); at q = 0.2, $11.14.

**(b) Preemption reserve: a real 10%** of planned VM spend: **$2.58**.

**(c) Branch 3 dropped from Phase 1 and from DAgger branch collection** (`branches.enabled` without
`b3_three_calls`, §12 reduction step 2; a tunable, so the data digest is unchanged, amendment 49). It stays in the
pilot, which reports b3 against b1. Its share goes into more points. Per point without b3: base parents 8 runs
(b1 needs 2 more seeds), exploration and DAgger parents 9 runs. At 7 VM-s per run (*est.*), $0.0073 and $0.0083 per
point. **Training already supports a missing head 3**, checked in the code and now tested. The planner leaves a
disabled branch out of the entry's `shares` (`branchplan.branch_plan`; `tests/test_plan.py`). Labels exist only for
the entry's branches (`labels.point_labels`). `vlar_train.data` fills absent labels with NaN, and `PointSet.targets`
turns them into mask 0. `train._loss` multiplies each head's BCE by the mask and divides by the mask's sum. λ uses
heads 1 and 2 only (`run.py`). New test: `test_absent_branch_labels_are_masked_out_of_training` (a head-3 label at
masked rows changes nothing, bit for bit).

**(d) Per-phase stop lines.** Each phase gets an allocation and a stop line on the ledger's VM spend since the phase
began. The workers check it before each claim (the existing stop flag, raised by the spend service for that phase)
and the runner checks it before each step. Disk is tracked on its own line. A branch phase stops cleanly at any point
(tiered job order, §12), so its line equals its allocation. Rollout and evaluation phases may overrun by 8%, and
the pilot up to its cap. Together these use the reserve, since 10% each would exceed it.

| Phase (in order) | Allocation | Stop line |
| --- | --- | --- |
| Pilot | 1.60 | 2.50 (the user's cap) |
| Phase 1: exploration screening (§0.3) | 0.98 | 1.06 |
| Phase 1: exploration re-runs and branches, base and exploration points | 3.24 | 3.24 |
| Phase 2 training (on the Mac) | 0 | |
| Pre-DAgger evaluation: trunk, router, `cascader_full`, π0.5 alone, audits | 8.71 | 9.41 |
| DAgger rollouts: `cascader_full` half record `full`; router half screened (§0.3) | 3.14 | 3.39 |
| DAgger re-runs and branches | 1.62 | 1.62 |
| Post-DAgger evaluation and baselines, audits | 6.52 | 7.04 |
| Disk (~$0.33/day; ~55 VM-h over ~5 days) | 1.65 | pause rule |
| Reserve (10% of planned VM spend) | 2.58 = 0.90 pilot headroom + 1.55 overrun shares + 0.13 unassigned | |
| **Total** | **30.06** | |

The pilot's overrun share ($0.90, up to its cap) is the user's approval; if the pilot comes in at $1.6, the
difference stays in the reserve. The lines need code: a phase name and line in the spend service, checked as the
hard limit is now. The pilot can use it.

**(e) The DAgger round in labelled-point equivalents.** Apart from its own labels, the round costs DAgger rollouts
$3.14 plus the post-DAgger router and `cascader_full` suffixes $2.64 plus their audits $0.25, **$6.04 in all**.
Post-DAgger `cascader_nospread` and the baselines are needed either way. That is **~590 Phase 1 points** at Phase
1's mixed price ($0.0102), or ~820 base-parent points, against **~317 Phase 1 points in the whole plan**. Its own
150 points cost $1.62 (~160 Phase 1 points), and its reserve share is ~$0.77. Without the round, Phase 1 could have
~1,070 points instead of 317 (n ≈ 107 per task instead of 47 in all). §10's before/after-DAgger comparison would
then be lost.

**Branches and points at this budget.** $30.06 − fixed lines (pilot, rollouts, evaluation, disk: $22.61) − reserve
$2.58 = **$4.86 for branches and re-runs**: two thirds to Phase 1, one third to DAgger (the §3 ratio).
- Phase 1: ~159 base points + ~159 exploration points (re-run cost shared by k = 2 points per re-run episode; ~79
  exploration episodes re-run): **~317 points**, ~32 per task.
- DAgger: ~75 `cascader_full`-parent + ~75 router-parent points (~37 router episodes re-run): **~149 points**, ~15
  per task.
- **~466 labelled points in all, ~47 per task.**

### 0.2 What this means for the stop rule (before the pilot)

**Decided 2026-11-24: thresholds kept; the DAgger round's fate follows the pilot's projections at ~47 and ~107
points per task (SPEC §17 amendment 58).**

With ~47 points per task, ≥ 20 positives per task needs a positive rate of **43% of all points** under uniform
sampling. A positive needs p_1 ≤ 2/3. At a base point in a successful episode, b1's first seed is the parent itself,
a success, so p_1 ≤ 2/3 also needs one of the 2 new seeds to fail. Positives therefore come mostly from the ~20%
of points in failed parents, and uniform sampling will almost certainly fail the stop rule. With stratified sampling
taking 2/3 of each task's points from failed parents, the failed-parent positive rate π^F must reach **≥ 64%**
(if positives elsewhere are rare). ≥ 100 per fold (8 × 47 = 376 points) needs 27% overall, or π^F ≥ 40% stratified.
So the per-task threshold binds before the per-fold one. **The thresholds (20 per task, 100 per fold) were set
against §3's 660 points. At ~466 they are near the edge even with stratification. This is a decision for the user,
before or after the pilot:** keep them, lower them, or move points from elsewhere (e.g. the DAgger round, (e)).

### 0.3 Probes only at sampled points, with outcome-stratified selection kept possible (decision 3)

**Approved 2026-11-24 (SPEC §17 amendment 56)**, with stop on any re-run that differs from its screen and the
router-half fallback to record `full` if the pilot's check (i) fails.

**Neither proposed method runs on the frozen harness.** The harness has two record modes, `off` and `full`
(`episode.RECORD_MODES`). `collect.Runner` writes snapshots and `points` rows only under `full`. Recording probes
at a candidate subset would therefore need a new record mode in `Episode._needs` and `Runner._episode`. That is a
harness change: a v1.1 freeze, a new golden set and the VM re-run rows. A run recorded `off` leaves no snapshots,
so there is nothing to replay probes from either. Replaying from a `full` run's snapshots saves nothing, because
that run already recorded every probe.

**What does run on the frozen harness: screen, then re-run whole episodes.**
1. **Screen.** Every exploration episode runs with record `off` and profile `none` (scripted random escalation,
   no measurement calls): ~15 VM-s instead of 74 (*est.*), **$0.98** for 500. It records the run, steps and events,
   so the outcome is known when it ends.
2. **Select.** After screening, choose the episodes and points to branch with a recorded seed, uniform or stratified
   by outcome (§0.5), k points per chosen episode (k = 2 planned).
3. **Re-run.** Each chosen episode runs again with the same rollout seed and record `full`. This is the §7 episode
   exactly: snapshots and full measurement at every point, which also serves §13. It costs 74 VM-s per chosen
   episode, i.e. 37 VM-s per point at k = 2. The re-run must equal the screened run bit for bit (steps, events,
   outcome): probe neutrality on the real models (item 5) says it does. Every re-run is compared with its screened
   run (free), and any difference stops the re-runs (max-effort checkpoint).
- **DAgger:** the `cascader_full` half makes all the measurement calls for its own decisions anyway, so it records
  `full` from the start ($2.62). The router half is screened with record `off` under its live router ($0.52) and
  its chosen episodes are re-run with record `full`. The router's mid-point features then come from the `/sample`
  probe, not `/features`, so the re-run reproduces the screened run only if the two are equal bit for bit (the
  pilot's check, §0.4). Otherwise the router half also records `full` from the start: +$2.1, the same mismatch §9's
  plan has.
- **Cost against the alternatives:** exploration plus DAgger rollouts **$5.30** (screening $0.98 + exploration
  re-runs $0.77 + DAgger $3.14 + router re-runs $0.39), against $10.09 recording `full` everywhere and $4.9 for
  §3b as first proposed (which needed the harness change). Base parents need nothing: they are already `full`, and
  their outcomes are known.

### 0.4 Pilot (§4): additions

Unchanged otherwise: 150 base points, 11 runs each with b3 kept, ~$1.6, stopped at $2.50.
- **Projected positives per task and per fold, under both sampling schemes.** From the pilot's per-task rates in
  each stratum, π_t^F (points in failed parents) and π_t^S (in successful ones), with 95% intervals. The pilot
  samples both strata in every task and records the weights. At the budgeted n_t (~47 per task: Phase 1 + DAgger):
  uniform E[pos_t] = n_t (φ_t π_t^F + (1 − φ_t) π_t^S), with φ_t the task's base failure rate; stratified
  E[pos_t] = n_t (s π_t^F + (1 − s) π_t^S), with s = 2/3 (capped by the failed points available). Per fold: the
  sum over its 8 training tasks. The stop rule is evaluated for both. Exploration and DAgger parents are projected
  with base rates, as no others exist yet (stated in the report).
- **Measured cost per branch type:** VM-s per run for b1, b2, b3 and b4 (from the GPU/VM-time ratio and each job's
  wall time and calls), per point with and without b3, and b4's share of each point's cost.
- **Cheap checks the revised plan rests on:** (i) `/sample` probe features against `/features` on the same
  restored base snapshots, bit for bit (amendment 55 (5); §0.3's router re-runs); (ii) routed always-local against
  `xvla_only`, 3 episodes, bit for bit (amendment 55 (2)); (iii) 3 base episodes re-run with record `off` against
  their `full` records, bit for bit (§0.3's screening); (iv) peak memory per worker and restart overhead under
  `--max-jobs` (§0.6). Together well under $0.05.

### 0.5 Outcome-stratified decision-point sampling with weighted training (approved 2026-11-24: SPEC §17 amendment 57)

1. **Strata:** (parent kind, task, parent outcome). The outcome is success at 520 steps without a time cutoff for
   scripted parents (base, exploration screening), and success under the parent's own live profile for DAgger
   parents. It is known once the parent episode ends.
2. **Selection:** in each stratum, eligible points (budget ≥ 10 steps) are drawn uniformly with a recorded seed.
   Base parents are drawn by point. Re-run parents (exploration, DAgger router half) are drawn by episode, then k
   points within it. Per task, a share s of the points (proposed 2/3) comes from failed parents, capped by the
   failed points available.
3. **Weights:** each point's inclusion probability π_i is recorded in its branch-plan entry (for re-run parents,
   P(episode) × P(point | episode)). w_i = 1/π_i, normalised to mean 1 within each fold's training rows and,
   separately, its validation rows.
4. **Training:** BCE per head × mask × w_i, divided by Σ mask × w_i. Early stopping uses the same weighted loss on
   the validation task. λ: the estimated gain (b2 − b1 where the decider escalates) and the escalation rate are
   weighted means over validation points. Points from the same episode share weights and are not treated as
   independent in any interval (cluster bootstrap by episode).
5. **Reporting:** label statistics (rescue and harm rates, mean p_2 − p_1) re-weighted to the population, and the
   effective sample size (Σw)²/Σw² per fold. Evaluation is unchanged: live (with amendment 55), on every initial
   state.
6. **Bias note:** at a base point b1's first seed is the parent rollout, so the stratum is correlated with b1's
   label. Within a stratum points are uniform, so weighted estimates are unbiased for the population. Unweighted
   training would be biased toward rescue.
7. **Leakage:** none new. Selection uses only the parent's own outcome, and folds still drop the test task's points.
8. **Code:** `branchplan` records the stratum and π_i per entry. `vlar_train.data` reads w_i. `train._loss`,
   validation loss and `lam` take weights. Tests: weights all 1 reproduce unweighted training bit for bit, and
   synthetic strata re-weight to the known population rates. The data digest is unchanged (selection and weights
   are in branch-plan entries; training is not covered).

### 0.6 Worker fixes (decision 5): implemented with tests (see `docs/HANDOFF.md` for the commit)

- **Progress-gated leases:** `LEASE_S` 10 min, renewed every 60 s, but only when the job has made progress since the
  last renewal. Progress means the episode's step count or event count changed, or an artifact was uploaded. A
  worker stuck on a request that never returns stops renewing and loses the job within ~11 minutes.
- **`--max-jobs` recycling with supervisor restarts** in `run_phase`. A worker that took its N jobs exits with a
  distinct status, and the supervisor starts it again while work remains (not counted as a restart). A worker that
  died (any other non-zero status; 137 = out of memory) has its job counted as a failed attempt. It is then
  restarted, at most 3 times per worker per phase. A worker that is alive but holds an expired lease is stopped,
  its job counted, and restarted. So is one whose job another worker has meanwhile claimed: a progressing worker
  never loses a lease. Exit 3 (refused job) is never restarted. Every event goes to the step log and
  `history.jsonl`.
- **Per-worker memory limit:** `docker run --memory` (swap equal), sized when the phase starts from the servers'
  measured memory (`docker stats`, after the golden check): limit = (0.95 × RAM − 1.25 × servers − 4 GiB host) /
  workers, rounded down to 256 MiB, and refused below 3 GiB. `--max-jobs` defaults to (limit − 2 GiB) / 0.75 GiB,
  the tasks' simulators that fit (*est.* from w8: 8-9 GB at up to 10 tasks). With servers at ~12 GiB and 8 workers
  that is ~5 GiB and 4 jobs per process. The pilot logs peak memory per worker and restart overhead, which replace
  the estimates. If restarts every ~4 branch jobs cost too much, there are two options: task-affine claiming (the
  user's fallback), or evicting cached simulators between jobs from the ops loop. Eviction changes no code in the
  data digest; the smoke re-runs on fresh workers matched cached-simulator records bit for bit.
- Code: `vlar_ops/queue.py` (`keep_lease(progress=)`, `leases`, `reap`), `worker.py` (`Progress`),
  `vm_worker.py` (`TrackedEpisodes`, `RECYCLE_EXIT` 4), `supervisor.py`, `cli.py` (`vlar-ops supervise`, worker
  exit statuses), `ops/vm/phase0/lib.sh` `run_phase`, `ops/vm/vlar-worker@.service` (`$VLAR_WORKER_DOCKER_ARGS`).
  Tests: `tests/test_supervisor.py`, `tests/test_worker.py` (stuck job loses its lease; progress and uploads renew
  it; tracked-episode token; exit statuses).

## 1. Measured speeds this plan rests on

| Source | Value |
| --- | --- |
| Base rollouts (`400`, record `full`, 8 workers) | 36,925 s / 500 = **73.9 VM-s per episode**; 312 steps, 10.9 chunk + 20.6 mid points |
| GPU per request at the fixed geometry, full batches (step 2) | `/act` 0.355 s, `/sample` 1.415 s, `/features` 0.075 s, π0.5 1.07 s per padded batch of 4 (0.27 s per request when full) |
| GPU share of the base rollouts | 48.5 GPU-s per episode; VM-s / GPU-s = **1.52** (GPU busy ≥ 66%) |
| π0.5 alone (`300`, 100 parity episodes) | 1,274 s / 100 = **12.7 VM-s per episode** (includes server start) |

Per-episode VM time by profile (*est.*: GPU work × 1.52, 5 escalations per learned-decider episode):

| Profile | Calls per episode | VM-s |
| --- | --- | --- |
| `cascader_full` (and any record-`full` run) | `/act` + chunk `/sample` at chunk points, probe `/sample` at mid points | **80** |
| `cascader_chunkspread` | `/act` + chunk `/sample`, `/features` at mid points | 38 |
| `router`, `cascader_nospread`, fixed-schedule, random (router profile) | `/act`, `/features` at mid points | 16 (the non-GPU floor is unmeasured) |
| `xvla_only` | `/act` every 30 steps | 7 |
| `pi05_only` | π0.5 only | 12.7 (measured) |
| one branch run (record off, scripted) | `/act`, 0-26 π0.5 calls | 7 (unmeasured; the pilot measures it) |

## 2. Budget re-plan: the §10 plan as written does not fit

**Evaluation, profile by profile, no sharing** (each learned decider: 10 folds × 50 = 500 episodes per round):

| Block | Episodes | VM-h | $ |
| --- | --- | --- | --- |
| Pre-DAgger: router, `cascader_full` | 500 + 500 | 2.2 + 11.1 | 6.29 |
| Post-DAgger: router, `cascader_full`, `cascader_nospread` | 1,500 | 2.2 + 11.1 + 2.2 | 7.34 |
| Baselines: `xvla_only`, `pi05_only`, fixed-schedule, random × 2 | 2,500 | 1.0 + 1.8 + 2.2 + 4.4 | 4.44 |
| **Required evaluation** | 5,500 | 38.3 | **18.07** |
| Optional: `cascader_chunkspread` post-DAgger | 500 | 5.3 | 2.49 |
| Optional: λ trade-off (router and `cascader_full` at 2 more λ) | 2,000 | 26.7 | 12.59 |

**Data:** exploration rollouts 500 × 74 s = $4.85; DAgger rollouts about $4.98; branches about 11-12 runs × 7 s
≈ 80 VM-s per sampled point = $0.0105 per point, and the points are 47,250 × f (1,500 parent episodes × 31.5
points × the sampling fraction f), i.e. **$496 × f**.

Total without sharing: pilot ($1.6, §4) + exploration $4.85 + DAgger rollouts $4.98 + evaluation $18.07 + $496 f
= $29.5 + $496 f. Against ~$30 this leaves f ≈ 0.001 (≈ 50 labelled points in all): **infeasible**.

## 3. Proposed savings

### 3a. Shared-prefix evaluation with common random numbers (proposed amendment)

All methods already share each (task, initial state)'s scene and evaluation rollout seed within a round (§10).
**Amendment proposed:** the same evaluation rollout seeds in both rounds (pre- and post-DAgger), and the X-VLA-only
run as a shared trunk:
1. **Trunk.** For every test (task, initial state), one routed always-local X-VLA run with the evaluation seed,
   record `full`, `/features` also called at mid points (see the trunk-data rule below), snapshots at every decision
   point. It is also the `xvla_only` baseline if routed always-local equals `xvla_only` mode bit for bit (to check
   once; otherwise `xvla_only` runs separately for $0.46).
2. **Offline prefix.** Each decider is run offline along the trunk: at each point its inputs come from the trunk's
   record, its remaining time and live cut from the trunk's event log under its own profile (exact, §5), and its
   network draws are the trunk's (keyed by rollout seed, step, call and attempt, never by profile). Until its first
   escalation a decider executes exactly the trunk (probe neutrality, item 5), so its run is the trunk's up to there,
   cut where its own clock passes the limit.
3. **Live suffix.** At its first escalation the decider's run is continued live from the trunk's snapshot at that
   point, under its own profile and live time limit, in ops code (restore, then set `profile`, `time_limit`,
   `time_limit_s`; no harness change). A decider that never escalates costs nothing beyond the trunk.

**Trunk-data rule (exactness).** The offline prefix is exact only if every input equals what the live decider would
have computed, bit for bit: (i) features at mid points: the live router/nospread/chunkspread call `/features`, but
record `full` takes them from the probe `/sample` (`/features` is never called in full recording); the trunk records
both, or the pilot shows the two equal on the L4; (ii) the decider runs with the same code, device and batch of one
offline and live; (iii) an audit: 20 episodes per decider and round run fully live and compared bit for bit with the
trunk reconstruction; any difference stops the shared method (max-effort checkpoint). Scoring needs one record-format
decision: a suffix's log begins with the trunk's prefix, whose start event says `time_limit: false`; the suffix
records the live limit from its divergence point (as `branch` events do), and scoring treats the combined log as a
live-limit run under the suffix's profile.

**Estimate** (q = fraction of a decider's episodes that ever escalate; first escalation on average 40% into the
episode, *est.*): trunk 500 × 76 s = $5.0; suffixes at q = 0.4: pre-DAgger router $0.25 + full $1.26; post-DAgger
router, full, nospread $1.76; fixed-schedule (q ≈ 0.8) $0.84; random × 2 (q ≈ 1, earlier) $1.47; π0.5 alone (not
shareable) $0.83. **Evaluation $11.4 instead of $18.1** (−37%); the λ sweep becomes $3.0 instead of $12.6.
Sensitivity: q = 0.2 → $8.6; q = 0.7 → $15.0.

**Snapshot storage against replay.** Trunk snapshots: ~0.6 MB × 31.5 × 500 ≈ 9.5 GB ≈ $0.19 per month, kept until
the post-DAgger evaluation is done, then all but the divergence points deleted. Replaying a prefix instead (X-VLA
only, record off, ~8 VM-s each) would cost ≈ q × 9 decider-rounds × 500 × 8 s ≈ 4 VM-h ≈ $1.9 at q = 0.4.
**Proposed: store** (cheaper, and no second source of error).

### 3b. Probes only at sampled points in exploration and DAgger rollouts (allowed by §7, "if the budget requires it")

Points without branches get no labels, so their probes train nothing (they serve only §13's future deciders).
Choosing the branch sample before each rollout (seeded per episode, as now) and recording probes only there and at
the chunk points their spread uses cuts exploration from ~74 to ~25 VM-s per episode (*est.*): exploration $1.64
instead of $4.85; DAgger rollouts about $3.3 instead of $5.0 (the live `cascader_full` half still probes everywhere
for its own decisions). Trunk runs keep full probes.

### Re-planned budget

| Item | $ |
| --- | --- |
| Pilot (§4) | 1.6 |
| Exploration rollouts (3b) | 1.6 |
| DAgger rollouts (3b) | 3.3 |
| Evaluation, shared trunk (3a), q = 0.4 | 11.4 |
| Branches at f | 496 f |
| Reserve: chunkspread $1.3, λ sweep $3.0, preemption losses ~10% | 5.0 |
| **Available to $40** | **≈ 30** |

So **f ≈ 0.014: about 660 labelled points in all** (≈ 44 per task from Phase 1, ≈ 22 from DAgger). Whether that
is enough depends on how often escalation helps, which is what the pilot measures.

## 4. Pilot design

- **Parents:** the 500 base rollouts' snapshots (in the bucket; no new rollouts). Per task 15 decision points with
  budget ≥ 30 steps, chosen with a recorded seed: up to 10 from base episodes that failed and the rest from ones that
  succeeded, uniform over points within an episode; the strata weights (per-task base success rates, e.g. task 0
  30/50, task 9 48/50) are recorded so rates can be re-weighted to the population. 150 points × 11 runs (b1 needs 2
  more seeds; b2, b3, b4 3 each) ≈ 1,650 runs ≈ 3.3 VM-h ≈ **$1.6**.
- **Per point:** labels p_b = fraction of b's seeds succeeding, scored under `none` and under each decider profile
  (offline). **Rescue:** p_2 > p_1 (one call helps); **harm:** p_2 < p_1. Per task: rescue rate and harm rate
  (re-weighted), mean p_2 − p_1, and the same for b3 and b4 against b1.
- **Tasks 4 and 8 separately:** the only tasks where π0.5's successful parity episodes overran the limit (4: 5 of 9
  fit; 8: 6 of 8), so escalation can hurt there through time alone; their rescue and harm rates, and how many
  rescues are lost to the cut under `cascader_full`'s profile, are reported apart from the other eight.
- **Measured costs:** VM-s per branch run by branch type, and per decision point; these replace the *est.* values.
- **Stop rule:** a positive label is a point with p_2 − p_1 ≥ 1/3 under `cascader_full`'s profile. Projected to the
  full collection at the f the budget allows, **if fewer than 20 positive points are expected for a task, that task
  is flagged; if more than 3 tasks are flagged, or any training fold (8 tasks) is expected to have fewer than 100
  positives, stop before Phase 1 and re-plan** (e.g. a stratified branch sample, which would need an amendment).

## 5. Spec confirmations

- **Deciders see remaining time and π0.5 budget:** §6 "Common inputs": the remaining π0.5 budget and the remaining
  time under the decider's own profile (recomputed from the event log per profile for training, amendment 23). Code:
  `Episode.observe` → `CommonInputs(budget_remaining_steps, time_remaining_s, steps_elapsed)`; training uses
  `ps.remaining_s[decider]` (`vlar_train/run.py`).
- **No fold trains on its own test task:** §7 (fold i: test task i, validation i+1, train the other 8; splits by
  whole task) and §9 (each fold drops its test task's DAgger points). Code: `vlar_train/folds.py` masks by task;
  standardisation is fitted on training rows only, λ on the validation task, the test rows are unused. Two indirect
  influences, both stated in the spec: the DAgger collectors are trained on all 10 tasks (§9, accepted), and the time
  limits and margin are environment properties calibrated on all tasks (§5). The shared trunk adds none: a fold's
  decider runs only on its own test task's trunk episodes, and trunk records never enter training.

## 6. Silent worker death (w8): cause found, proposed fix

**Cause (from the disk's journals, `phase0/w8-evidence/`):** a global out-of-memory kill. At 00:49:31Z the kernel
killed w8's worker (python, 8.2 GB resident); its unit exited 137 a second later. The same happened at 22:17:51Z
during the parity step (another worker, 9.1 GB; that step's next round re-ran its job). At both moments the 8
workers held 5-9 GB each beside the two servers, together more than the VM's 64 GB. A worker keeps one simulator per
task it has met (`ProductionEpisodes.sims`), so its memory grows toward ~10 tasks' worth: **8 workers will run out
of memory again in Phase 1.** A worker-count increase is therefore off. Memory bound, proposed (no change to
`ProductionEpisodes`, which is in the common data digest): run each worker with `--max-jobs N` (exists) and let the
supervisor below restart it when it exits with work left, so a process never accumulates many tasks; or claim jobs
task-affinely (a worker takes jobs of at most 2-3 tasks; a queue change, not in any data digest); plus a per-worker
`docker --memory` limit so an overrun kills that worker, not a random one. The pilot measures memory per worker.

- **Heartbeat leases:** `LEASE_S` 3 h → 10 min, renewed every 60 s by the existing `keep_lease` thread. A dead or
  frozen process (its renewal thread stops too, as w8's did) loses its job within 10 minutes instead of 3 hours; a
  live worker never does (renewal is independent of the job's progress).
- **Supervisor in `run_phase`:** at each 120 s poll, (i) a worker unit that is no longer active while the phase has
  pending or leased jobs is restarted, at most 3 times per worker per phase, logging its exit status and journal
  tail; (ii) a unit still active whose job's lease has expired (alive but frozen) is stopped (`docker stop`, then
  kill) and restarted; (iii) every restart is recorded in the step log and `history.jsonl`. Exit status 3 (refused
  job) is never restarted.
- **Worker count:** not increased (memory, above).
  `servers.workers` is in the frozen write-once group, so a change needs a dated supplement. The padded sizes
  (`max_batch` 4, 20 sample rows) do not depend on the worker count; if a change ever alters any padded batch size,
  the bit-exact batch checks are re-run first (float32 alone does not make results shape-independent).

## 7. Item 9: preemption restarter (design only)

- **Trigger:** Cloud Scheduler every 15 minutes runs a small Cloud Run job (free tier) under its own service
  account. It acts only on `$VM_NAME` in its zone.
- **Preemption only:** restart only if all hold: the instance is `TERMINATED`; its most recent terminating operation
  is `compute.instances.preempted` (not `stop`, which a person or script issues, and not `guestTerminate`, which the
  runner's own `shutdown -h` and the spend guard's stop produce); `STATE.json` is `running` or `interrupted` and was
  written before that operation; the guard's `STOP` state is absent (`STATE.json` not `stopped`, no hard alert at
  the current limit in the ledger). Never after `idle`, `failed`, `gate_failed`, `needs_input` or `stopped`.
- **Spend check first:** the ledger estimate (`spend.estimate`, with failed boots folded in) plus one more VM hour
  must stay below the hard limit minus $2, else no restart and a log-based alert.
- **Rate limit:** at most 3 successful restarts per UTC day, counted in `ops/restarter/<date>.json` (conditional
  writes); start attempts that fail on a stockout are retried at the next tick, at most 6 per day; same zone only.
- **Least privilege:** a custom role with `compute.instances.get`, `compute.instances.start` and
  `compute.zoneOperations.list`, bound on the one instance (IAM condition on its resource name) and the zone;
  `storage.objects.get` on the ledger, `STATE.json` and the queue prefix; `storage.objects.create` only under
  `ops/restarter/`. No stop, delete, create, metadata or IAM permissions.
- **Kill switch:** an object `ops/restarter/DISABLED` stops it; the pause rule's VM deletion leaves it nothing to
  start (it checks the instance exists and does nothing otherwise).
