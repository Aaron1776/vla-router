# Running the harness on the DGX Spark (linux/aarch64)

The Spark runs the evaluation (ASSUMPTIONS D3). Nothing here changes the harness, the x86_64 images or the cloud
path (`ops/gcp/`, `ops/vm/`); it reuses the VM's step scripts unchanged.

## Images (`containers/aarch64/`)
- `lock.sh` resolves `containers/*.in` for linux/aarch64 with hashes (same cutoff and index as `containers/lock.sh`).
  Every lock has the same versions as its x86_64 lock; the same five packages are built from sdists.
- `build.sh COMMIT` builds `vlar-{sim,pi05,xvla}:<commit>` offline (`--network none`) from `git archive` of the commit
  plus this folder: wheels from a local wheelhouse, hash-checked against the aarch64 locks; the base image
  `vlar-aarch64-base` (`Dockerfile.base`: ubuntu:24.04 by the harness's index digest, the same apt rendering
  libraries, CPython 3.13.16 aarch64 by SHA-256); the X-VLA checkout at 6c4ecfb8 and LIBERO's assets at 0b3ea86b.
  Each image ends with `check_env.py` against the aarch64 locks. It writes `image-digests.json` (Docker image IDs, as
  `ops/vm/load_images.sh` records them) with `"platform": "linux/aarch64"`, which `provenance.record_header` copies
  into every record and marker.
- The wheelhouse is filled with `pip download --require-hashes` of the aarch64 locks (stage 1 notes).

## Step runner (`ops/spark/run.sh STEP...`)
The local stand-in for `ops/vm/phase0/runner.sh`: runs the named steps in order in tmux, stopping at the first exit
other than 0 (75 needs the user, 76 gate failed). Each step is rewritten by `transform.py` (the VM paths
`/opt/vla-router`, `/etc/vlar`, `/opt/weights`, `/opt/hf`, `/var/lib/vlar` become folders under `VLAR_SPARK_ROOT` on
the host side only; containers see the L4's paths) and runs with `shim.py`'s stand-ins on PATH:
- `gcloud storage`: `gs://<bucket>/<key>` is `VLAR_SPARK_ROOT/var/bucket/<key>`, which the containers see as
  `/var/lib/vlar/bucket/` (the ops config's `storage_url`);
- `systemctl`: `vlar-xvla`, `vlar-pi05`, `vlar-worker@N` from `ops/vm/*.service` as named detached containers;
- `docker run`: adds the Spark root at the same path and the wheelhouse as the only pip index (no network);
- `sudo` runs the command; `shutdown` does nothing. There is no spend guard.
Spark versions of three steps (`ops/spark/steps/`): `010` (the locally built images, not the bucket's), `015` (host
rows in a throwaway container) and `020` (weights already on the machine, hard-linked; no download, no token).
Every step's log, `history.jsonl` entry and results folder carry `platform.json`.

```bash
containers/aarch64/build.sh <commit>
tmux new -d -s vlar 'ops/spark/run.sh 010 015 020 030 040 050 090 > ~/vlar-run.log 2>&1'
```

## Phase 0 reproduction (`ops/spark/repro.py`, step `620`)
The L4's π0.5 parity (100 episodes) and X-VLA base rollouts (500), re-run on the Spark episode by episode and
compared paired, from `References/l4-phase0/`. `repro.py plan` rebuilds both plans with `vlar_ops.plan` exactly as
steps 300 and 400 did and refuses unless every L4 row matches its job (job id, task, init, rollout seed, mode,
profile, record). `run` brings a fresh simulator to the row's reconstructed setup count (`vlar_ops.pilot.warm_up`)
and runs the job with `vlar_harness.collect.Runner`, under the calibration the L4 runs used (`aa38aac7`), against
both production servers at the calibrated geometry; records stay in the output folder.

```bash
REPRO_PHASE=pi05_parity ops/spark/run.sh 620      # 1 shard: π0.5 requests stay alone in their batch
REPRO_PHASE=base ops/spark/run.sh 620             # 8 shards
```

**Pass criteria (written 2026-11-25, before any reproduction run).** For each phase, both must hold:
1. Against the L4, paired by episode: every L4 episode re-run, and the two-sided exact McNemar test on the
   discordant pairs (L4 success and Spark failure, and the reverse) gives p >= 0.05.
2. Against the reference run, the L4's own gate: `vlar_ops.reference_runs.compare(..., limit=520)` (π0.5:
   `load_pi05`, as `vlar_ops.parity`; X-VLA: `load_xvla`, as the X-VLA parity of SPEC §11 step 5) has
   `within_sampling_error` (exact McNemar p >= 0.05).

**Known exceptions, not failures:** episodes are compared by outcome only. Trajectories, success steps and step
counts are not expected to match the L4's (other GPU kernels), and a Spark episode re-run is not expected to repeat
bit for bit (4x MSAA, ASSUMPTIONS D5; 5 exact-replay tests of step 030 fail for this reason). The parity runs' setup
counts are reconstructed and unchecked (no snapshots); base counts were checked on the L4's snapshots.
**Open:** π0.5 at the calibrated `max_batch` 4 is not batch-invariant on the GB10 (step 210: 6 of 10 exact checks,
up to 0.017), while it is at 8 and on the L4 at 4. The parity run keeps every request alone (one shard) and the base
rollouts never call π0.5, so the reproduction is not affected; concurrent π0.5 use (evaluation) is.
