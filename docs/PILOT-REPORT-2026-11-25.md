# Pilot report (2026-11-25)

Computed by `vlar-ops pilot report` (`src/vlar_ops/pilot.py` at `1fc1857`) from the records, markers and reports of
both pilot boots (local copies under the git-ignored `outputs/pilot/`; full output `outputs/pilot/report.json`).
**The rules for this report were set before its results (SPEC §17 amendment 62, logged 2026-11-24 22:34Z and committed
as `1fc1857` at 22:47Z; the resumed boot's results were downloaded from 00:37Z).** Everything not stated otherwise
follows amendments 57, 58, 61 and 62.

**After this report (2026-11-25):** the user's re-plan is SPEC §17 amendment 63 (`cascader_nospread` the headline cascader; positives scored under its profile; b3 and b4 dropped; own-rate flags only from 10 failed-parent points) and 64 (7 workers by the RAM rule). Re-applied at the new budgets the rule says **keep**: `docs/REPLAN-2026-11-25.md`. This report is unchanged; it is reproduced by `pilot report --stop-profile cascader_full --own-basis all` (checked: same projections, flags and decision).

## Outcome: the amendment 58 rule says **stop** before Phase 1 and re-plan with the user

| | positives / labelled (unscored under `cascader_full`) | pooled rate, 95% CI |
| --- | --- | --- |
| failed parents | 0 / 59 (34) | r_f 0.000 [0.000, 0.000] |
| successful parents | 2 / 50 (15) | r_s 0.040 [0.000, 0.083] |

Expected positives per task under stratified sampling (pooled rates, each task's own φ and failed points): **2.6** at
~194 points per task (current plan), **3.0** at ~227 (DAgger dropped); at the measured prices' ~131 and ~154, 1.8 and
2.1. Every task is flagged, every fold is short, and no bootstrap replicate passes (share 0.00); uniform sampling
fails too. The decision is the same at declared and measured prices, and not close, so no pilot extension (amendment
61 (2)).

Why: a positive needs p_2 − p_1 ≥ 1/3 **under `cascader_full`'s profile** (§4). At 34 of the 59 failed-parent points
that profile's clock has already passed the task's time limit at the point (the profile is excluded there, so the
point cannot be positive), and on the 25 it scores none reaches 1/3. π0.5 does rescue failures when the clock is
not charged for measurement:

| at the 109 labelled points, 520-step limit, no clock | b1 continue | b2 one call | b3 three calls | b4 max |
| --- | --- | --- | --- | --- |
| failed parents: successes / runs | 12/118 (10%) | 43/177 (24%) | 55/177 (31%) | 69/174 (40%) |
| successful parents | 97/100 (97%) | 147/150 (98%) | 144/150 (96%) | 148/150 (99%) |

Re-weighted (w = 1/π, amendment 57 (6)), b2 against b1: under `none` rescue 6.0%, harm 1.6%, mean gain +0.027
(ESS 72); under `cascader_full` rescue 4.2%, harm 6.8% (60 scored). b4 against b1 is negative under every profile
(mean −0.065 under `none`, −0.134 under `cascader_full`; harm 53% under `cascader_chunkspread`). 15 rescues under
`none` are lost to `cascader_full`'s cut (5 of them excluded at the point). Full tables: `report.json` → `rates`.

**Information for the re-plan only (not the rule):** with positives scored under the other profiles, the same pooled
projection gives

| positives scored under | failed-parent positives | r_f (95% CI) | E per task at 194 / 227 / 131 / 154 points | flagged by own rate (≥ 10 points) | passes at 194 / 227 / 131 / 154 |
| --- | --- | --- | --- | --- | --- |
| `cascader_full` (the rule) | 0/25 scored of 59 | 0.000 | 2.6 / 3.0 / 1.8 / 2.1 | all | no / no / no / no |
| `router`, `cascader_nospread` | 13/56 | 0.220 (0.132-0.311) | 29.8 / 34.9 / 20.2 / 23.6 | 4, 7, 8 (at 131/154: 0, 1, 4, 5, 7, 8) | yes / yes / no / no |
| `none` | 14/57 | 0.237 (0.150-0.323) | 31.9 / 37.5 / 21.7 / 25.4 | as `router` | yes / yes / no / no |
| `cascader_chunkspread` (added 2026-11-25, amendment 63 (3b)) | 13/47 (12 unscored) | 0.220 (0.132-0.311); r_s 0.060 | 32.3 / 37.9 / 21.9 / 25.7 | 4, 7, 8 (at 131/154: 0, 1, 4, 7, 8) | yes / yes / no / no |

Bootstrap pass shares under `router` are low (0.12 / 0.20 / 0.01 / 0.02) because the own-rate flags move with the
resample. The pilot labelled 10-11 points per task, not the ~7 amendment 62 assumed, so its own-rate test applies to
every task with only 5-6 failed-parent points each: at the pooled 22%, a task shows 0 of 6 with probability
0.78^6 ≈ 0.23.

## Checks

- **Golden set and image cross-check:** passed on the resumed boot (`929f6ca2`).
- **Corrected check (iii): passed, 7 of 7 bit for bit** at the replicated load count, fixtures equal to the base
  run's after setup: base/t8/i30 (count 4; 396 steps, success, as recorded), t3/i25, t9/i8, t2/i26, t4/i49, t5/i7
  (count 2) and t8/i13 (count 7). The first boot's t8/i30 mismatch is explained by the load count (HARNESS.md); the
  screen's premise holds once the count is replayed (amendment 62 (2)).
- Checks (i) and (ii) passed on the first boot (`/features` equals the probe; `xvla_only` equals the routed run).

## Labels and jobs

1,221 of 1,638 jobs done (first boot 345, resumed boot 876); the spend guard stopped the step at the phase's stop
point (exit 75, `needs_input`, 00:34:31Z) with 417 pending. 109 of the 150 points are labelled (11 per task, task 8
10); 41 are not, all for missing b1 seeds 2-3 (36 and 5).

## Costs (resumed boot only, amendment 62 (1))

6 workers at 5.5 GiB, `max_jobs` 2; branch window 9,449 s; 876 jobs.

| per run | b1 continue | b2 one call | b3 three calls | b4 max |
| --- | --- | --- | --- | --- |
| VM-s | 8.3 | 8.1 | 9.7 | 17.2 |
| $ | 0.0011 | 0.0011 | 0.0013 | 0.0023 |
| steps | 378 | 350 | 365 | 327 |
| VM-s per step | 0.0219 | 0.0232 | 0.0266 | 0.0527 |

Per point: base parent without b3 **92.6 VM-s, $0.0121** (plan 56 VM-s, $0.0073: ×1.65); exploration parent without
b3 100.9 VM-s, $0.0132 (plan 63); base with b3 121.8 VM-s. **b4 is 56% of a base point's cost.** At these prices
amendment 58's budgets buy **~131 points per task (current) and ~154 (DAgger dropped)**, against ~194 and ~227 at the
plan's prices (the re-run's 74 VM-s is still an estimate).

First boot (reported apart, not used: 8 workers dying at the 4 GiB limit): b1 14.0, b2 13.2, b4 29.1 VM-s per run;
155 VM-s per base point.

Restart overhead (resumed boot): 44 gaps over 20 s between a worker's jobs (median 20.8 s, 1,147 s in all, ~2% of
worker time); 434 recycles (`max_jobs` 2); one frozen worker restarted (w5, 22:28Z; its lease re-run).

**Screening cost (the user's addition):** the corrected check (iii) ran alone on the servers before the branch phase:
501 s for 7 record-off episodes (3,516 steps), including container start, the count searches and warm-up loads, so
at most 0.143 s per step alone (this boot's code has no per-episode split). Under load a b1 run takes 0.117 worker-s
per step, and with 6 workers 0.0219 VM-s per step: **a screened episode costs ~6.8 VM-s at the base rollouts' mean
312 steps and ~11.4 VM-s at 520** (plan: 7), plus whatever warm-up loads the setup-count rule adds.

## Memory (resumed boot; amendment 62 (4))

RAM 67.42 GB; servers 18.96 GB at start, **peak 20.45 GB** (π0.5 14.90, X-VLA 5.55); worker peaks 4.04-4.54 GB (the
5.5 GiB = 5.91 GB limit is 1.30× the largest). N × 5.91 GB + 20.45 GB inside RAM: **6 fit (11.5 GB to spare), 7 fit
(5.6 GB), 8 do not (0.28 GB over).** The boot-time sizing rule allows 6; running 7 needs it changed.

## Spend

Pilot phase VM **$2.3702** (allocation $1.60, stop point $2.3467, line $2.50); the phase is left open (state `stop`).
Ledger total **$12.48** (VM $9.65, disk $0.61, storage $0.01, charges $2.20): $27.52 left to $40. The VM is
terminated; its disk still exists (~$0.33 per day).
