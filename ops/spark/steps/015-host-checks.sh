#!/usr/bin/env bash
# cap-seconds: 1800
# The Spark's step 015: the VM's host rows (the unit tests in the main environment, the reference files' pinned
# SHA-256). The L4 ran them in a host venv with `pip install -e .[gcs,dev]`; here a throwaway container of the aarch64
# base image (CPython 3.13.16, with python3 on /usr/bin as on the VM host) installs the same extras offline from the
# wheelhouse (the docker shim) and imports the repo from PYTHONPATH, writing nothing into the clone.
source /opt/vla-router/ops/vm/phase0/lib.sh
HV=(docker run --rm -v /opt/vla-router:/opt/vla-router:ro -w /opt/vla-router -e PYTHONPATH=/opt/vla-router/src
    -e PYTHONDONTWRITEBYTECODE=1 vlar-aarch64-base sh -c)
SETUP="git config --system --add safe.directory '*' && ln -sf /opt/python/bin/python3 /usr/bin/python3 && python3 -m venv /tmp/v && /tmp/v/bin/pip install -q 'pyyaml>=6' 'google-cloud-storage>=2.14' 'pytest>=8' &&"
row host-unit-tests "${HV[@]}" "${SETUP} /tmp/v/bin/python -m pytest -q -rs -p no:cacheprovider tests"
row reference-check "${HV[@]}" "${SETUP} /tmp/v/bin/python -m vlar_ops.cli reference check --dir References"
exit 0
