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
