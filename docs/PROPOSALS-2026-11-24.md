# Proposals for the user's approval (2026-11-24): items 6 and 9 of the post-freeze list

Nothing here is approved or implemented. Numbers marked *est.* are estimates from the measurements named; the pilot
replaces them with measured values. Prices: $0.472 per VM hour (g2-standard-16 Spot with its L4); the spend ledger
reads **$9.76** (VM $7.12, disk $0.44, the $2.20 of image builds and other non-VM charges added with `spend charge`),
so about **$30 remains to the $40 soft limit** (the hard stop stays at $45; $5 is kept for storage and the download).

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

## 6. Silent worker death (w8): proposed fix, held until w8's cause is known

- **Heartbeat leases:** `LEASE_S` 3 h → 10 min, renewed every 60 s by the existing `keep_lease` thread. A dead or
  frozen process (its renewal thread stops too, as w8's did) loses its job within 10 minutes instead of 3 hours; a
  live worker never does (renewal is independent of the job's progress).
- **Supervisor in `run_phase`:** at each 120 s poll, (i) a worker unit that is no longer active while the phase has
  pending or leased jobs is restarted, at most 3 times per worker per phase, logging its exit status and journal
  tail; (ii) a unit still active whose job's lease has expired (alive but frozen) is stopped (`docker stop`, then
  kill) and restarted; (iii) every restart is recorded in the step log and `history.jsonl`. Exit status 3 (refused
  job) is never restarted.
- **Worker count:** unchanged until w8's cause is known (its evidence is the smoke step's first action).
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
