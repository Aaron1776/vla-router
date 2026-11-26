# Spark Phase 0 reproduction results (DGX Spark, linux/aarch64)

The DGX Spark's re-run of the L4's Phase 0 (`References/l4-phase0/`), episode by episode, compared paired. Small
text files only: no snapshots, images, steps tables or records. Every file's SHA-256 is pinned in
`ops/spark/spark-phase0.sha256` (outside the `vlar_ops` package, so the ops digest is unchanged); verify from the repo
root with `grep -v '^#' ops/spark/spark-phase0.sha256 | shasum -a 256 -c`. Kept byte for byte (`.gitattributes`:
`References/** -text`).

- Code: `spark-port` at `fac4245` (clean tree), `ops/spark/repro.py` and step `ops/spark/steps/620-repro.sh`; images
  built from harness-v1 `9b8d076` (harness source digest `845d6933`, the v1.0 freeze's). Pass criteria: written in
  `ops/spark/README.md` ("Phase 0 reproduction") before either run, unchanged since.
- Runs: π0.5 parity 2026-11-25 07:56-08:51Z, one shard (one episode at a time); base 08:51-17:10Z, 8 shards. Each
  episode on a fresh simulator brought to the L4 row's reconstructed setup count (`vlar_ops.pilot.warm_up`), under
  calibration `aa38aac7` (the L4 runs'), against both production servers at the calibrated geometry (`max_batch` 4,
  `xvla_sample_rows` 20). The job id, task, init, rollout seed, mode, profile and record of every row equal the L4's.

## Results

| phase | episodes | Spark | L4 | reference (520 steps) | vs L4: exact McNemar p (L4 only / Spark only) | vs reference: p | criteria |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| π0.5 parity | 100 | 90 | 94 | 97 | 0.289 (6 / 2) | 0.065 | both pass |
| X-VLA base | 500 | 403 | 398 | 405 | 0.625 (31 / 36) | 0.878 | both pass |

## Files

| file | contents |
| --- | --- |
| `compare/pi05-parity-compare.json`, `compare/base-compare.json` | `repro.py compare` output as computed: paired counts, exact McNemar against the L4, the per-task counts, and `vlar_ops.reference_runs.compare(..., limit=520)` against the reference run |
| `tables/pi05-parity-per-task.csv`, `tables/base-per-task.csv` | per task: episodes, Spark, L4 and reference successes (reference: `reference_runs.load_pi05` / `load_xvla`, within 520 steps) |
| `runs/pi05-parity-runs.csv`, `runs/base-runs.csv` | one row per Spark episode: `run_id`, `job_id`, `task_id`, `init_idx`, `scene_seed`, `rollout_seed`, `mode`, `profile`, `record`, `success`, `success_step` (empty on failure), `steps`, `end_reason`, π0.5 steps, requests and failed requests (the harness's network model), `calibration_hash` (8 hex), `setup_count` (the load count the episode was set up at), `shard` (0-7; parity 0) |
| `network.json` | π0.5 requests and failed requests per phase, Spark and L4 (base rollouts make none) |
| `platform.json` | machine, architecture, kernel, GPU and driver, the image digests the runs used (with `"platform": "linux/aarch64"`), and the repo commit, as every result line recorded it (identical across both phases) |

To redo the paired tests: join `runs/<phase>-runs.csv` here with `References/l4-phase0/runs/<phase>-runs.csv` on
`run_id`, count the discordant pairs, and take the two-sided exact binomial (n = discordant, p = 1/2); or run
`python ops/spark/repro.py compare` in the sim image.

Notes: outcomes are compared, not trajectories (other GPU kernels; 4x MSAA, ASSUMPTIONS D5). The parity setup counts
are the L4's reconstruction and unchecked (record `off`, no snapshots); in task 8 the Spark has 4 of 10 against the
L4's 8. π0.5 at `max_batch` 4 is not batch-invariant on the GB10 (step `210`, row `pi05-batch-invariance-geometry`);
the parity run kept every π0.5 request alone in its batch, and the base rollouts make no π0.5 request.
