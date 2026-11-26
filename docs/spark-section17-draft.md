# Draft: SPEC §17 entries for the DGX Spark (for Miguel's review)

Draft by Oscar, 2026-11-25. Not approved; numbers are placeholders (S1-S7) to be renumbered after Miguel's latest amendment. Evidence references point to `docs/ASSUMPTIONS.md` on `main` and to the `spark-port` branch (`fac4245`). Items marked **[decide]** are proposals that need both of us to agree before any run they govern.

## S1. Phase 0 reproduction on the Spark (dated supplement)

The Spark (NVIDIA DGX Spark, GB10, linux/aarch64) ran Phase 0's parity and base rollouts with harness v1.0 (source digest `845d6933`, equal to the freeze), the frozen calibration `b80124b2` and the calibration the Phase 0 runs were recorded under (`aa38aac7`), the L4 runs' seeds and reconstructed setup counts, and aarch64 images built from `9b8d076` at the lockfiles' versions. Pass criteria were written into `ops/spark/README.md` before either run: (1) paired exact McNemar against the L4's per-episode results, p >= 0.05, every episode re-run; (2) the L4's own gate against the reference run (`reference_runs.compare`, limit 520). Results (ASSUMPTIONS V49):

| Phase | Spark vs L4 (p) | Spark vs reference (p) | Result |
| --- | --- | --- | --- |
| π0.5 parity, 100 episodes | 90 vs 94 (0.289) | 90 vs 97 (0.065) | pass |
| X-VLA base, 500 episodes | 403 vs 398 (0.625) | 403 vs 405 (0.878) | pass |

Every task is within 5 successes between the two machines. Known exceptions stated before running: trajectories and step counts are not expected to match the L4's, and Spark re-runs are not expected to repeat bit for bit (S4).

## S2. Spark platform rule: π0.5 runs one request per batch

On the GB10, π0.5 at `max_batch` 4 is not position-invariant: cuBLAS's bfloat16 GEMM uses a small-matrix kernel at 2-6 rows that rounds alternate row groups differently, so a request's output depends on its row position (deterministic, no cross-row leakage; 1, 7, 8, 12 and 16 rows clean; executed actions differ by at most 0.0029 between positions against amendment 37's 0.05; ASSUMPTIONS V50). No library setting removes it on this GPU. Rule: on the Spark, π0.5 serves one request per batch (an ops-side gate keeps one `/act` in flight), enforced and recorded in every run's provenance. The π0.5 batch-invariance row is replaced on the Spark by a check that the gate is in force and that a batch of one equals `act_unbatched` exactly. Measured cost per 500 episodes with 8 workers: +3% at 10% π0.5 steps, +14% at 31%, +103% at 90% (ASSUMPTIONS V51). The batch geometry in the frozen calibration is unchanged.

## S3. aarch64 image supplement

A dated aarch64 image supplement sits beside `freeze.json`, listing the three aarch64 image digests built from the frozen code. The image check selects the supplement by the `platform` key in the running image digests. An L4 run still requires the L4 images; there is no fallback in either direction. The dataset manifest keeps one image set per dataset, so Spark results form their own datasets.

## S4. Spark golden set

4x MSAA rendering is not bit-repeatable on the GB10 (about 9-10% of renders at one frozen state differ by one level at one pixel; ASSUMPTIONS V44). In 120 repeat pairs of base-style X-VLA episodes, divergence appeared only in tasks 3 and 8 (V45). The Spark golden set is chosen from the Spark's own Phase 0 runs, outside tasks 3 and 8, and keeps exact replay. **[decide]** Acceptance before adopting it: each candidate point replayed 10 times from fresh processes, all 10 bit-identical in steps and events. If any candidate fails, it is replaced; if no exact set of 5 can be found, a statistical rule is written down before any further run.

## S5. Spark verdict step for the MSAA exact-replay rows

The 5 exact-replay tests that fail on the Spark because of MSAA (snapshot replay, render-only-what-is-used) are reported, not hidden. On the Spark, step 090's verdict treats them as known exceptions and adds one statistical row. **[decide]** Proposed row: a fixed set of base-style episodes (for example 30: tasks 0-9 x initial states 0-2), each run twice from fresh processes; pass if at least 90% of pairs are bit-identical and no more than 2 pairs change outcome. Observed so far: 109 of 120 identical, 4 outcome changes, over four passes (V45).

## S6. DAgger collection on the Spark

The DAgger round runs on the Spark (amendment 65). Its rollouts record in full from the start (the mode amendment 56 allows as a fallback), so there is no screen and no screen-to-re-run bit-for-bit comparison to trip on MSAA. The cost is Spark time only. Every record carries `platform: linux/aarch64` in its provenance header.

## S7. DAgger labels from another platform

Phase 1 labels come from the L4, DAgger labels from the Spark. Every training point carries its platform. Each decider is trained twice after the DAgger round, with and without the Spark points, and both are evaluated. **[decide]** "The two disagree" is defined before any DAgger result as: on the paired evaluation episodes, the headline comparison (`cascader_nospread` vs the router) changes sign, or either decider's success rate differs between its two versions by more than the 95% paired interval of that difference. If they disagree, the pre-DAgger deciders are the headline result and the DAgger results are reported as a secondary analysis. Planned bias check, beyond the Phase 0 reproduction: re-score a sample of Phase 1 points on the Spark from their stored snapshots and compare positive rates by stratum, if the Spark can restore L4 snapshots (under check: the compiled-model fingerprint may differ across platforms).
