# Operations runbook (Google Cloud, $50)

How the VM, bucket and spend guard are set up and run (docs/SPEC.md §12–§13). Nothing here
runs automatically; each script is run by hand from a machine logged in with `gcloud`.

## Before anything: lead-time items

- **GPU quota.** New projects have no GPU quota. In the console (IAM & Admin → Quotas) request
  `GPUS_ALL_REGIONS` ≥ 1 (global) and `PREEMPTIBLE_NVIDIA_L4_GPUS` ≥ 1 in the chosen region.
  Approval can take from minutes to a few days.
- **Billing type.** Free-trial billing accounts cannot attach GPUs. Check that the credits
  sit on a billing account that allows them.
- **Gated tokenizer.** π0.5 needs `google/paligemma-3b-pt-224` (see `envs/README.md`).

## Costs the spend guard counts

| Item | Rate used | Note |
| --- | --- | --- |
| VM (g2-standard-16 Spot + L4) | `spend.vm_hourly_usd`, about $0.46/h | Only while running. Set from the current Spot price before each start. |
| Boot disk | `DISK_GB` × $0.10/GB-month (pd-balanced) | **Bills every hour it exists, running or stopped.** 100 GB ≈ $10/month ≈ $0.33/day. |
| Bucket | $0.020/GB-month (Standard, regional) | Scanned every 30 min while the VM runs. |
| Anything else | weekly reconcile | Set the ledger to the billing report's figure: `vlar-ops spend reconcile --actual-usd X`. |

The disk is the cost that grows while nothing is happening. **If work pauses for more than a day,
delete the L4 VM and its disk** (`gcloud compute instances delete … --delete-disks=all`, then
`vlar-ops spend disk-deleted --name <vm>-boot`) and recreate it from the bucket to resume
(`30_create_vm.sh`, the one-time VM setup, `load_images.sh`, `vlar-ops queue resume`): the images,
the queue's results, the snapshots and the ledger are all in the bucket. Only the weights are downloaded
again (11 GB, free inbound).

### Boot disk size (measured 2026-11-22)

| Part | GB | Source |
| --- | --- | --- |
| Deep Learning VM image (OS, driver, CUDA toolkit) | 50 | the image family's minimum disk size (`diskSizeGb`), counted whole |
| The three images, loaded | 14 | locked wheels 3.5 GB (LeRobot env, shared by sim and pi05) + 2.8 GB (X-VLA) compressed, about twice that installed, plus Python, LIBERO assets (0.4) and code |
| Image files while loading | 7 | the `.tar.zst` files; `load_images.sh` deletes them after loading |
| Weights and tokenizer | 11 | π0.5 checkpoint 7.47, X-VLA-Pt 3.52, LoRA 0.05 (Hugging Face, pinned revisions) |
| Working space | 5 | queue database, result markers, each worker's staged files until upload (about 0.15 GB per job), logs |
| Margin | 13 | |
| **Total (`DISK_GB`)** | **100** | about $0.33/day, half the earlier 200 GB |

The figure is checked on the VM: `df -h /` after `load_images.sh` and the weight download (Phase 0's first step).

## One-time setup

```bash
cp ops/gcp/env.example.sh ops/gcp/env.sh   # fill in; git-ignored
source ops/gcp/env.sh
ops/gcp/05_check_account.sh                 # read-only: sign-in, billing, APIs, GPU quotas, driver image families
ops/gcp/10_bucket_and_account.sh            # APIs, bucket (soft delete off) + VM service account
ops/gcp/20_budget_alerts.sh                 # backup email alerts at 50/80/100% of $50
ops/gcp/40_spend_alert_policy.sh            # email when the guard logs its soft/hard event
ops/gcp/35_build_images.sh                  # CPU-only Spot VM: build, test, export the images; then deleted
ops/gcp/30_create_vm.sh                     # L4 Spot VM; prints the disk-ledger command to run next
```

`35_build_images.sh` needs only billing, not the GPU quota (amendment 28). It runs from a clean commit,
uploads `git archive` of it, and waits for the build VM's result in `gs://$BUCKET/containers/<commit>/`:
the three images as `.tar.zst` files, `image-digests.json`, `SHA256SUMS`, and the build and in-container test
logs. The VM deletes itself after `BUILD_MAX_HOURS` at the latest; the script deletes it when it is done.
Rebuild after any harness change (a new commit means new images). The lockfiles are regenerated with
`containers/lock.sh` (needs `uv`).

On the VM: clone the repo to `/opt/vla-router`, create `.venv` and `pip install -e ".[gcs]"` (the spend
guard and the queue commands run on the host; everything else runs in the images), copy
`ops/ops.example.yaml` to `ops/ops.yaml` and set `storage_url`, then install the guard
**before anything else can run**:

```bash
sudo cp ops/vm/vlar-spend.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now vlar-spend
```

Then load and check the images (the first step of Phase 0): `ops/vm/load_images.sh gs://$BUCKET/containers/<commit>`
checks that the NVIDIA driver supports CUDA 13.0 (580.65.06 or later), loads the images, checks them against
`SHA256SUMS` and their image IDs, writes `/etc/vlar/image-digests.json` (provenance), `/etc/vlar/images.env` and
`/etc/vlar/batching.env`, and runs each image's environment check with the GPU. Install the service units
(`sudo cp ops/vm/vlar-{xvla,pi05}.service ops/vm/vlar-worker@.service /etc/systemd/system/ && sudo systemctl
daemon-reload`); they are started by hand, never at boot. Weights go to `/opt/weights` and the PaliGemma
tokenizer's cache to `/opt/hf` (both mounted read-only).

## Every start (including after a preemption)

1. Check the current Spot price and update `spend.vm_hourly_usd` if it moved.
2. The spend daemon starts on boot. If the ledger is already over the hard limit it shuts the VM
   straight down again; raise the limits in `ops/ops.yaml` first only if funds were added.
3. `vlar-ops queue resume`: release stale leases, re-upload local-only results, mark everything
   already in the bucket as done.
4. Start both model servers: `sudo systemctl start vlar-xvla vlar-pi05` (batch geometry from
   `/etc/vlar/batching.env`, written from the harness config by `load_images.sh`).
   The golden-set self-check (§15) runs here once the harness is frozen; it is written with the freeze, and
   until then `vlar-ops worker` refuses a frozen config.
5. Start the workers: `sudo systemctl start vlar-worker@{1..N}` (N from the Phase 0 throughput measurements;
   `ops/vm/vlar-worker@.service`). Each worker checks both servers' `/info` (fixed batch geometry,
   deterministic settings) and stops if they differ, claims jobs until none is left or the spend guard's
   STOP flag is up, and uploads every file before a job's marker. Options: `--phase P` (repeatable) to claim
   only one phase's jobs, `--option scene_seed=pi05_parity` for the π0.5 parity run (start those workers with
   `--phase pi05_parity`, as `docker run` with the unit's mounts, or a unit drop-in). A worker given a job its
   options cannot run gives it back and exits with status 3.

## Weekly

- `vlar-ops spend status`, then compare with the billing report and run `vlar-ops spend reconcile`. Use the
  cost **before credits**: on a coupon-funded account the cost after credits is $0 until the coupon runs out.
  The budget alerts (`20_budget_alerts.sh`) track the gross cost for the same reason.

## End of project

1. Package on the VM: `vlar-ops package build --src <tree> --out <dir> --name dataset-v1`
   (and `harness-v1.0`), upload the parts to the bucket.
2. Download the parts and `SHA256SUMS` to your computer, then
   `vlar-ops package verify --out <dir>` (or `shasum -a 256 -c SHA256SUMS`).
3. Only then: `ops/gcp/90_teardown.sh --archives-verified`.
