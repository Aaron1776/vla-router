# L4 Phase 0 reference files (for the DGX Spark)

Small reference files from Phase 0 on the L4 (harness v1.0, frozen 2026-11-24T09:46:21Z, images `03f311c`, ops
`0a0cd4c`), for replaying and comparing on the DGX Spark. Separate from the Phase 1 handoff package. Not here (they stay
in the bucket, `gs://$BUCKET/vla-router/`): snapshots, PNGs, steps tables, points tables, container images and weights.
Paths below are relative to `gs://$BUCKET/`.

Every file's SHA-256 is pinned in `src/vlar_ops/configs/l4-phase0.sha256` (this README included); verify from the repo
root with `grep -v '^#' src/vlar_ops/configs/l4-phase0.sha256 | shasum -a 256 -c`, or run
`tests/test_l4_phase0_refs.py`. Like the rest of `References/`, the files are kept byte for byte (`.gitattributes`:
`-text`).

## Files

| file | bytes | source | SHA-256 |
| --- | ---: | --- | --- |
| `calibration/calibration.redacted.yaml` | 3,087 | `vla-router/calibration/calibration.yaml` (the frozen file, `b80124b2`), **redacted** | `931310675e959dab07382d766332d1d9bf7f5460326c1e240ec683cea99ec08a` |
| `calibration/redactions.json` | 1,169 | built here: each redacted file's original name and SHA-256 (below) | `1c090f54194a24bacdf1616d861663a83350614f6b01b22506ac336e1052f21e` |
| `calibration/versions/1d61294a7f511e7fd52298236ac85431bb81c1d7f94c32139e6387b9f5c360bc.yaml` | 2,310 | `vla-router/calibration/versions/1d61294a….yaml`: step 6 (`500`: time limits, margin m = 2.0) | `1d61294a7f511e7fd52298236ac85431bb81c1d7f94c32139e6387b9f5c360bc` |
| `calibration/versions/236439b4f06c68e2d4ecef86e8c57d71876f1bbd6dca8e58a75a9f9125c6657c.redacted.yaml` | 2,937 | `vla-router/calibration/versions/236439b4….yaml`: step `550` (only `shut_width_m` 0.0090040 -> 0.003) **redacted** | `1d313cd43a7618d75dc46c219436defab732a1b27bb5d3e496d0e9d65ac7da9b` |
| `calibration/versions/aa38aac7fb61850e3550a256885fe634f48810aaeeb7f182f00b9552cf5e3890.yaml` | 1,461 | `vla-router/calibration/versions/aa38aac7….yaml`: step 2 (`200`) | `aa38aac7fb61850e3550a256885fe634f48810aaeeb7f182f00b9552cf5e3890` |
| `calibration/versions/b80124b2c19ba8a5e7dcc538eb777a6ca033463020e77b28464b068118c5f51c.redacted.yaml` | 3,087 | `vla-router/calibration/versions/b80124b2….yaml`: step `600` (only `frozen` added; the frozen file) **redacted** | `931310675e959dab07382d766332d1d9bf7f5460326c1e240ec683cea99ec08a` |
| `events/base-events.sha256` | 55,625 | built here: each archive member's SHA-256, equal to its result marker's artifact hash | `6c0b024575fa0c671990a7a6e3f4a64e5e47bb201e17870b8d6c0e5500e745f2` |
| `events/base-events.tar.gz` | 541,439 | `vla-router/records/events/base/*.parquet`, the 500 files byte for byte in a deterministic tar.gz | `e950902d7407adc683d1e8ef645b5dbe1c88172190de12b31328cd9b546b2e79` |
| `freeze/freeze-supplement-2026-11-24.json` | 1,420 | `vla-router/calibration/freeze-supplement-2026-11-24.json` (amendment 54) | `bc3ba2c3d89c538564178c69be43cfba0fbd3338b9d3b4674adcbb69aa659478` |
| `freeze/freeze.json` | 833 | `vla-router/calibration/freeze.json` (step `600`) | `5e2826f6a4864a4cbbd7e1542b9a1f4dbd2502276b842d793b94676efe4585ad` |
| `handoff/handoff-relative-l4.log` | 1,209 | `vla-router/phase0/logs/results/03f311c4cd18a392285709987426bdfc42413fcd/handoff-relative-l4.log` (step `500`: 3 passed) | `4cf1dcb1deb45591c5a8e6203544aff47ac365767e654d5a053695646d8d2b51` |
| `handoff/handoff-relative-l4.status` | 7 | `vla-router/phase0/logs/results/03f311c4cd18a392285709987426bdfc42413fcd/handoff-relative-l4.status` | `2700165975f68815c97d605c56eca8e90d497ade1264b6282401d13fee99ac27` |
| `images/image-digests.json` | 303 | `containers/03f311c4cd18a392285709987426bdfc42413fcd/image-digests.json` (the image build's record) | `c509448a07ad620ccc0bb6b52e216c8a417aa66daab559f4550e20b8e8c90838` |
| `runs/base-runs.csv` | 73,364 | built here from `vla-router/records/runs/base/*.parquet` and `vla-router/results/base/*.json` (markers) | `7c152690711ea6463061cc88fb63faf0ea003a9e64f6f499abd67a66c026e5b9` |
| `runs/pi05-parity-runs.csv` | 15,682 | built here from `vla-router/records/runs/pi05_parity/*.parquet` and `vla-router/results/pi05_parity/*.json` | `f986826df0b8dd6a52cf755ec41d80a342d331d0d2e3bfff7cb91e769cb973c5` |
| `runs/setup-count-check.json` | 12,642 | built here: the reconstructed counts checked against the base dp00 snapshots (below) | `ad484500dfd20216b7cad64f8ae9fe0fe452cd2abe35d2dc118248498b1bb47f` |
| `step2/geometry.json` | 2,047 | `vla-router/phase0/step2/03f311c4cd18a392285709987426bdfc42413fcd/geometry.json` (steps `200`/`210`) | `15261bc1e544a434a66f33d2cfed778769279afde3962bf0c018b460c6eaaabc` |
| `step2/gripper.json` | 326 | `vla-router/phase0/step2/03f311c4cd18a392285709987426bdfc42413fcd/gripper.json` (step `200`) | `936518e1c94ca8c392fd27cf7744d2ab56968e2216f43e145163f97b6dd065de` |
| `step2/latency.json` | 4,530 | `vla-router/phase0/step2/03f311c4cd18a392285709987426bdfc42413fcd/latency.json` (step `200`) | `cd06d9504a14ca61756c64f7aeff500483da87dfec64e45be5a14ae737744ccd` |
| `step2/upload.json` | 4,447 | `vla-router/phase0/step2/03f311c4cd18a392285709987426bdfc42413fcd/upload.json` (step `200`) | `1879804267a9df83b337a0bedda3d3465fe58211b42ca4a61c3b1d921974225a` |
| `step6/first_targets.json` | 106,582 | `vla-router/phase0/step6/first_targets.json` (the handoff check's base-rollout first-target distances) | `042f839daebae53f930250fbac5caf9c1e4b1451db803b1145c82c2f6e46e9cd` |
| `step6/grasps.json` | 13,218 | `vla-router/phase0/step6/grasps.json` (step `500`'s grasp check, at the step-2 shut width) | `a765a83b40003969325fef4df6327774e474d9427e2e72f9658895fe6f6362c0` |
| `step6/xvla-parity.json` | 9,724 | `vla-router/phase0/step6/xvla-parity.json` (`ops/xvla_parity.py`, SPEC §11 step 5) | `d611f19bca420b8c51adf105ffe923044ad8fd43f548b554efa3e648e2160040` |
| `step6/xvla-parity.txt` | 2,170 | `vla-router/phase0/step6/xvla-parity.txt` (the same, as text) | `a09df498ed97fdc1bec9f47f66efa4c9e3d10dcac4cc90891d4c5a18c5051607` |

The calibration versions are named by their own SHA-256 (the harness's `calibration_hash`). Lineage, from the freeze
record: `aa38aac7` → `1d61294a` → `236439b4` → `b80124b2` (frozen). All 600 Phase 0 runs (500 base, 100 π0.5 parity)
were recorded under `aa38aac7`, so their rows say so. `freeze.json` has `image_digests: {}` (the container that wrote
it could not see them); the supplement pins them (amendment 54), and `images/image-digests.json` is the image build's
own record, identical to the supplement's digests and to every Phase 0 marker's.

**Redacted calibration files.** The frozen file (`calibration.yaml` = `b80124b2`) and `236439b4` carry the bucket's
name once, in a free-text `note` of their change history (step `550`'s pointer to the grasp-review files), and the
repo may not hold it (`tests/test_no_identifiers.py`). They are committed with that name replaced by the literal
`$BUCKET` and named `*.redacted.yaml`; the originals contain no `$`, so the restore is exact. Their SHA-256 here is
the redacted file's; `calibration/redactions.json` pins each original's name and SHA-256 (the `calibration_hash` the
harness and the freeze record expect). To use the frozen file, restore it first (with `BUCKET` from
`ops/gcp/env.sh`), or copy the original from `gs://$BUCKET/vla-router/calibration/`:

```bash
sed "s#\$BUCKET#${BUCKET}#" References/l4-phase0/calibration/calibration.redacted.yaml > calibration.yaml && shasum -a 256 calibration.yaml
```

The printed hash must be `b80124b2c19ba8a5e7dcc538eb777a6ca033463020e77b28464b068118c5f51c`; the test checks the
restore wherever `ops/gcp/env.sh` is present.

## Runs tables (`runs/*.csv`)

One row per episode, from its `runs` table row: `run_id`, `job_id`, `phase`, `task_id`, `init_idx`, `scene_seed`
(base 142 + init, X-VLA's convention; π0.5 parity 1000 + init, lerobot-eval's), `rollout_seed`, `mode`, `profile`,
`record`, `success`, `success_step` (empty on failure), `steps`, `end_reason` (`success` or `step_limit` at 520),
`pi05_steps`, `pi05_requests`, `pi05_failed_requests`, `calibration_hash` (first 8 hex). From the result markers:
`worker` (`w1`-`w8`), `worker_seq` (the worker's n-th episode of the step, in finish order), `worker_task_seq` (its
n-th of this task), `finished_at_unix`. Base: 500 episodes, 398 successes (X-VLA parity 398/500 against the
reference's 405 at 520 steps). π0.5 parity: 100 episodes (tasks 0-9, inits 0-9), 94 successes.

### Setup (load) counts: **reconstructed**, not recorded

Phase 0 did not record the simulator's model-load count (the "setup count", amendment 62 (2)); the fixtures of tasks
3, 8 and 9 move with it (`docs/HARNESS.md`, "Fixture placement depends on the simulator's model-load count"). The
column `setup_count_reconstructed` is rebuilt from the markers' worker order:

- `ProductionEpisodes` keeps one `LiberoSim` per task per worker process; a fresh simulator is load 1 and each
  `setup_episode` adds one. So the worker's k-th episode of a task was set up at load **k + 1**.
- Assumptions: each worker id was one process for the whole step (systemd `Restart=no`; each step starts its own
  workers, so the parity and base counts both start fresh), a worker runs one job at a time (finish order = start
  order), and no episode was set up without leaving a marker. The one lost job (w8 died at ~00:48Z holding
  `base/t1/i15`, which w3 then ran) does not touch any recorded count: w8 never ran again.
- **Checked (base):** tasks 2 and 8 by the dp00 snapshot's `visualization_sites_list`, which LIBERO extends once per
  load (100/100 agree); tasks 3 and 9 by fixture poses (`vlar_ops.pilot.base_loads` on the Mac
  against each dp00 snapshot's model fields: 100/100 agree, 48 bit for bit and the rest at
  ≤ 5.6e-17 m, the Mac's last-bit difference, while neighbouring counts differ by millimetres). Columns
  `setup_count_check` and `setup_count_check_method` (`sites_list`, `fixture_poses`, or `none` for tasks 0, 1, 4, 5,
  6, 7, whose scenes do not depend on the count). Details in `runs/setup-count-check.json`.
- **π0.5 parity counts are unchecked:** those runs used record `off`, so there are no snapshots. Same rule, same
  assumptions.
- To replay a count n on a fresh simulator: `vlar_ops.pilot.warm_up(sim, n)` (n - 2 extra setups), then
  `setup_episode(init_idx, scene_seed)` is load n. Snapshot restores are unaffected by the count.

## Event logs (`events/`)

The base rollouts' 500 `events` tables (parquet; columns `run_id`, `seq`, `t`, `step`, `event` as JSON), unchanged,
in one tar.gz (541,439 bytes; members `events/base/<job_id>.parquet`; `job_id` → `run_id`
in `runs/base-runs.csv`). The π0.5 parity event logs are not included (ask; they are in the bucket).

## See also

- `docs/HARNESS.md`: "Fixture placement depends on the simulator's model-load count" (the finding, the Mac survey of
  counts 2-51, and the proposed rule for fresh-start runs).
- `docs/SPEC.md` §17, amendments 54-64: 54 the image-digest supplement; 55 common random numbers and the shared
  evaluation trunk; 56 screen then re-run; 57 stratified sampling; 58 the pilot's stop rule; 59 per-phase spend lines;
  60 bucket soft delete; 61 evaluation on another billing account; 62 pooled projections and the setup count; 63 the
  re-plan after the pilot (`cascader_nospread` headline); 64 the RAM worker-sizing rule.
