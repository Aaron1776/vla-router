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
| Boot disk | `DISK_GB` × $0.10/GB-month (pd-balanced) | **Bills every hour it exists, running or stopped.** 200 GB ≈ $20/month ≈ $0.66/day. |
| Bucket | $0.020/GB-month (Standard, regional) | Scanned every 30 min while the VM runs. |
| Anything else | weekly reconcile | Set the ledger to the billing report's figure: `vlar-ops spend reconcile --actual-usd X`. |

The disk is the cost that grows while nothing is happening. If work pauses for more than a
couple of days, delete the VM and its disk (`gcloud compute instances delete … --delete-disks=all`,
then `vlar-ops spend disk-deleted --name <vm>-boot`) and recreate it on resume; everything needed
to resume is in the bucket.

## One-time setup

```bash
cp ops/gcp/env.example.sh ops/gcp/env.sh   # fill in; git-ignored
source ops/gcp/env.sh
ops/gcp/10_bucket_and_account.sh            # bucket (soft delete off) + VM service account
ops/gcp/20_budget_alerts.sh                 # backup email alerts at 50/80/100% of $50
ops/gcp/40_spend_alert_policy.sh            # email when the guard logs its soft/hard event
ops/gcp/30_create_vm.sh                     # Spot VM; prints the disk-ledger command to run next
```

On the VM: clone the repo to `/opt/vla-router`, create `.venv` and `pip install -e ".[gcs]"`,
copy `ops/ops.example.yaml` to `ops/ops.yaml` and set `storage_url`, then install the guard
**before anything else can run**:

```bash
sudo cp ops/vm/vlar-spend.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now vlar-spend
```

## Every start (including after a preemption)

1. Check the current Spot price and update `spend.vm_hourly_usd` if it moved.
2. The spend daemon starts on boot. If the ledger is already over the hard limit it shuts the VM
   straight down again; raise the limits in `ops/ops.yaml` first only if funds were added.
3. `vlar-ops queue resume`: release stale leases, re-upload local-only results, mark everything
   already in the bucket as done.
4. Model servers, the golden-set self-check and the workers start here (added with the harness;
   the queue must not start if the self-check fails, §15).

## Weekly

- `vlar-ops spend status`, then compare with the billing report and run `vlar-ops spend reconcile`.

## End of project

1. Package on the VM: `vlar-ops package build --src <tree> --out <dir> --name dataset-v1`
   (and `harness-v1.0`), upload the parts to the bucket.
2. Download the parts and `SHA256SUMS` to your computer, then
   `vlar-ops package verify --out <dir>` (or `shasum -a 256 -c SHA256SUMS`).
3. Only then: `ops/gcp/90_teardown.sh --archives-verified`.
