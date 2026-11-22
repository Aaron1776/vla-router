# The harness's three images, one per Python environment (docs/SPEC.md §15; amendments 28-29):
#   sim   LIBERO, robosuite, MuJoCo, the harness's simulator side and vlar_ops (the queue workers)
#   pi05  the π0.5 server (LeRobot 0.6.1); same Python environment as sim (containers/lerobot.lock)
# All three run CPython 3.13.16, the reference notebook's Python (amendment 30).
#   xvla  the X-VLA server (the authors' pins, containers/xvla.lock)
# Built on the CPU build VM from `git archive` of a clean commit (ops/gcp/35_build_images.sh), from the repo root:
#   docker build -f containers/Dockerfile --target sim --build-arg GIT_COMMIT=<commit> -t vlar-sim:<commit> .
# Python packages install only from the hashed lockfiles. Weights are not in the images: they are mounted read-only
# at run time, and Hugging Face runs offline. Each image ends by running containers/check_env.py (no GPU); the L4 VM
# runs it again with --gpu before anything else. On the VM the containers use host networking and the servers
# listen on 127.0.0.1 only (ops/vm/*.service).

ARG UBUNTU=ubuntu:24.04@sha256:9c7f200c07e3d4a03e7249d2c16bf66a51120fb876aa2242cf31451412543289
# CPython 3.13.16, the reference notebook's Python (amendment 30): Astral's python-build-standalone, release
# 20261020, its SHA-256 checked against the release's SHA256SUMS on 2026-11-22.
ARG PYTHON_VERSION=3.13.16
ARG PYTHON_RELEASE=20261020
ARG PYTHON_SHA256=0ad293ecda8ea93a28e4a6b5e9cc20ae74be15aafed75be04418b8f82bd0d0da

# ---------------------------------------------------------------- base: standalone CPython 3.13.16 in a venv
FROM --platform=linux/amd64 ${UBUNTU} AS base
ARG PYTHON_VERSION
ARG PYTHON_RELEASE
ARG PYTHON_SHA256
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 \
    VLAR_IN_IMAGE=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL -o /tmp/python.tar.gz "https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/cpython-${PYTHON_VERSION}%2B${PYTHON_RELEASE}-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "${PYTHON_SHA256}  /tmp/python.tar.gz" | sha256sum -c - \
 && mkdir -p /opt/python && tar -xzf /tmp/python.tar.gz -C /opt/python --strip-components=1 && rm /tmp/python.tar.gz \
 && test "$(/opt/python/bin/python3 -c 'import platform; print(platform.python_version())')" = "${PYTHON_VERSION}" \
 && /opt/python/bin/python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv

# ---------------------------------------------------------------- the authors' X-VLA code at the pinned commit
FROM base AS xvla-src
ARG XVLA_REPO=https://github.com/2toinf/X-VLA
ARG XVLA_COMMIT=6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
 && git init -q /opt/X-VLA && cd /opt/X-VLA && git remote add origin "${XVLA_REPO}" \
 && git fetch -q --depth 1 origin "${XVLA_COMMIT}" && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${XVLA_COMMIT}" && rm -rf .git

# ---------------------------------------------------------------- wheels for lerobot.lock's sdist-only packages
# As the reference notebook built them (SPEC §2): cmake>=3.29,<4.2 (build.lock), CMAKE_POLICY_VERSION_MINIMUM=3.5,
# --no-build-isolation.
FROM base AS sdist-wheels
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY containers/build.lock containers/lerobot.lock containers/lerobot.sdists containers/locktool.py /tmp/lock/
RUN /opt/python/bin/python3 -m venv /opt/build \
 && /opt/build/bin/pip install --require-hashes --no-deps --only-binary :all: -r /tmp/lock/build.lock \
 && python /tmp/lock/locktool.py split /tmp/lock/lerobot.lock /tmp/lock/lerobot.sdists \
      --wheels /tmp/lock/wheels.txt --sdists /tmp/lock/sdists.txt \
 && CMAKE_POLICY_VERSION_MINIMUM=3.5 PATH=/opt/build/bin:$PATH \
    /opt/build/bin/pip wheel --no-build-isolation --no-deps --no-binary :all: \
      --require-hashes -r /tmp/lock/sdists.txt -w /wheels

# ---------------------------------------------------------------- the LeRobot environment (sim and pi05)
FROM base AS lerobot-env
COPY containers/lerobot.lock containers/lerobot.sdists containers/locktool.py /opt/vla-router/containers/
COPY --from=sdist-wheels /wheels /tmp/wheels
RUN cd /opt/vla-router/containers \
 && python locktool.py split lerobot.lock lerobot.sdists --wheels /tmp/wheels.txt --sdists /tmp/sdists.txt \
 && pip install --require-hashes --no-deps --only-binary :all: -r /tmp/wheels.txt \
 && pip install --no-deps --no-index /tmp/wheels/*.whl \
 && rm -rf /tmp/wheels /tmp/wheels.txt /tmp/sdists.txt

# LIBERO's scene assets at the pinned revision; their tree digest is checked by check_env.py.
FROM lerobot-env AS libero-assets
ARG ASSETS_REPO=lerobot/libero-assets
ARG ASSETS_REVISION=0b3ea86be5fe169d0fd036ae63d1070ec09e90f6
RUN python -c "from huggingface_hub import snapshot_download as d; d('${ASSETS_REPO}', repo_type='dataset', \
      revision='${ASSETS_REVISION}', local_dir='/opt/libero-assets')" \
 && rm -rf /opt/libero-assets/.cache

# Rendering libraries (EGL with the NVIDIA driver on the VM; OSMesa for the CPU tests on the build VM), the assets
# and LIBERO's config. pi05 needs LIBERO for its tests against lerobot-eval's processing.
FROM lerobot-env AS lerobot-runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libegl1 libgl1 libgles2 libglvnd0 libosmesa6 libglib2.0-0t64 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=libero-assets /opt/libero-assets /opt/libero-assets
COPY containers/libero_config.py /opt/vla-router/containers/
RUN python /opt/vla-router/containers/libero_config.py /opt/libero-home /opt/libero-assets
ENV LIBERO_CONFIG_PATH=/opt/libero-home VLAR_LIBERO_ASSETS=/opt/libero-assets \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/opt/vla-router/src

# ---------------------------------------------------------------- targets
FROM lerobot-runtime AS sim
COPY --from=xvla-src /opt/X-VLA /opt/X-VLA
ENV VLAR_XVLA_REPO=/opt/X-VLA
COPY pyproject.toml /opt/vla-router/
COPY References /opt/vla-router/References
COPY tests /opt/vla-router/tests
COPY src /opt/vla-router/src
COPY containers /opt/vla-router/containers
RUN printf '#!/bin/sh\nexec python -m vlar_ops.cli "$@"\n' > /usr/local/bin/vlar-ops && chmod +x /usr/local/bin/vlar-ops
ARG GIT_COMMIT
RUN test -n "${GIT_COMMIT}" || { echo "--build-arg GIT_COMMIT=<commit> is required" >&2; exit 1; }
ENV VLAR_GIT_COMMIT=${GIT_COMMIT}
WORKDIR /opt/vla-router
RUN dpkg-query -W > /opt/vla-router/apt-packages.txt \
 && python containers/check_env.py --role sim --lock containers/lerobot.lock
CMD ["vlar-ops", "worker"]

FROM lerobot-runtime AS pi05
COPY pyproject.toml /opt/vla-router/
COPY References /opt/vla-router/References
COPY tests /opt/vla-router/tests
COPY src /opt/vla-router/src
COPY containers /opt/vla-router/containers
ARG GIT_COMMIT
RUN test -n "${GIT_COMMIT}" || { echo "--build-arg GIT_COMMIT=<commit> is required" >&2; exit 1; }
ENV VLAR_GIT_COMMIT=${GIT_COMMIT} HF_HOME=/hf
WORKDIR /opt/vla-router
RUN dpkg-query -W > /opt/vla-router/apt-packages.txt \
 && python containers/check_env.py --role pi05 --lock containers/lerobot.lock
CMD ["python", "-m", "vlar_harness.servers.pi05_server", "--checkpoint", "/weights/pi05_libero_finetuned", \
     "--host", "127.0.0.1"]

FROM base AS xvla
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0t64 && rm -rf /var/lib/apt/lists/*
COPY containers/xvla.lock /opt/vla-router/containers/
RUN pip install --require-hashes --no-deps --only-binary :all: -r /opt/vla-router/containers/xvla.lock
COPY --from=xvla-src /opt/X-VLA /opt/X-VLA
ENV VLAR_XVLA_REPO=/opt/X-VLA HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/opt/vla-router/src
COPY pyproject.toml /opt/vla-router/
COPY References /opt/vla-router/References
COPY tests /opt/vla-router/tests
COPY src /opt/vla-router/src
COPY containers /opt/vla-router/containers
ARG GIT_COMMIT
RUN test -n "${GIT_COMMIT}" || { echo "--build-arg GIT_COMMIT=<commit> is required" >&2; exit 1; }
ENV VLAR_GIT_COMMIT=${GIT_COMMIT}
WORKDIR /opt/vla-router
RUN dpkg-query -W > /opt/vla-router/apt-packages.txt \
 && python containers/check_env.py --role xvla --lock containers/xvla.lock
CMD ["python", "-m", "vlar_harness.servers.xvla_server", "--model-path", "/weights/X-VLA-Pt", \
     "--lora-path", "/weights/X-VLA-libero-long-peft", "--host", "127.0.0.1"]
