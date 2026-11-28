#!/usr/bin/env bash
# cap-seconds: 43200
# Phase 0 step 5 (SPEC §11), queued by the parity gate on a pass: the base rollouts, X-VLA always local in routed
# mode on all 500 episodes (tasks 0-9 x initial states 0-49, scene seed 142 + k, rollout namespace 1000000), 520
# steps, no time cutoff, full measurement and snapshots at every decision point. They count as v1.0 (amendments 34,
# 35): their calibration hash is in the freeze's lineage. Re-runnable after a preemption: the job queue resumes.
# Steps 6-7 (time limits, golden set, freeze) come after, at the user's word.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
calfile require-write-once || fail "no write-once calibration"
PL="${P0D}/plans"; mkdir -p "${PL}"
vlar-ops plan episodes --phase base --inits 0-49 --mode routed --profile none --record full \
  --out "${PL}/base.json" || fail "base plan"
vlar-ops queue add "${PL}/base.json" || fail "queue add"
run_phase base "--phase base"
echo "base rollouts done: $(vlar-ops queue status --phase base | tail -n 1)"
