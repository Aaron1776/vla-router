#!/usr/bin/env bash
# GCE shutdown script (runs on preemption and on normal shutdown): stop the workers so no new job
# starts, then stop the spend daemon so its final tick lands in the ledger in the bucket.
systemctl stop 'vlar-worker@*' 2>/dev/null || true
systemctl stop vlar-spend 2>/dev/null || true
