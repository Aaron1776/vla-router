#!/usr/bin/env bash
# cap-seconds: 10800
# VM re-run list, simulator rows (docs/HARNESS.md "Platform rule"): every equality and determinism test of the
# simulator image on Linux with EGL rendering and CUDA; only these results count for parity and determinism.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
row sim-reference docker run "${GPU_RUN[@]}" "vlar-sim:${C}" python -m pytest -q -rs -p no:cacheprovider \
  tests/reference/test_sim_reference.py tests/reference/test_snapshot_replay.py tests/reference/test_handoff.py \
  tests/reference/test_clock_live.py tests/reference/test_render_used.py tests/reference/test_measurements.py \
  tests/test_measure.py tests/reference/test_collect.py tests/reference/test_vm_worker_sim.py tests/test_manifest.py
row sim-numpy-percentile docker run "${GPU_RUN[@]}" "vlar-sim:${C}" python -m pytest -q -rs -p no:cacheprovider \
  tests/test_labels_calibration.py -k numpy
exit 0
