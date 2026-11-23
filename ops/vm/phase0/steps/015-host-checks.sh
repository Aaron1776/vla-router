#!/usr/bin/env bash
# cap-seconds: 1800
# VM re-run list, host rows: the unit tests in the main environment (config, queue, spend guard, packaging, network
# model, clock, labels, calibration on synthetic logs) and the reference files' pinned SHA-256 and totals.
source /opt/vla-router/ops/vm/phase0/lib.sh
cd "${R}"
row host-unit-tests "${R}/.venv/bin/python" -m pytest -q -rs -p no:cacheprovider tests
row reference-check "${R}/.venv/bin/vlar-ops" reference check --dir References
exit 0
