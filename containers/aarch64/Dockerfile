# The harness's three images for linux/aarch64 (DGX Spark): the same targets, environment and commands as
# containers/Dockerfile (sim, pi05, xvla), from the aarch64 locks in this folder (same versions as the x86_64 locks).
# Built offline by containers/aarch64/build.sh from `git archive` of a commit plus this folder:
#   - BASE: ubuntu:24.04 by the same index digest, the same apt rendering libraries, CPython 3.13.16 aarch64
#     (python-build-standalone 20261020, SHA-256 908e16f2...); built by build.sh from Dockerfile.base;
#   - every Python package from the wheelhouse (named context `wh`), hash-checked against the aarch64 locks;
#   - the X-VLA checkout at 6c4ecfb8 and LIBERO's assets at 0b3ea86b from the build context (checked here and by
#     check_env.py: commit, tree digest).
ARG BASE=vlar-aarch64-base

FROM ${BASE} AS base
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 \
    VLAR_IN_IMAGE=1
RUN test "$(/opt/python/bin/python3 -c 'import platform; print(platform.python_version())')" = "3.13.16" \
 && /opt/python/bin/python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:/opt/python/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin VIRTUAL_ENV=/opt/venv

FROM base AS xvla-src
ARG XVLA_COMMIT=6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2
COPY xvla-src /opt/X-VLA
RUN cd /opt/X-VLA && test "$(git rev-parse HEAD)" = "${XVLA_COMMIT}" && git diff --quiet HEAD && rm -rf .git

FROM base AS sdist-wheels
COPY containers/aarch64/build.lock containers/aarch64/lerobot.lock containers/aarch64/lerobot.sdists containers/locktool.py /tmp/lock/
RUN --mount=type=bind,from=wh,target=/wh \
    /opt/python/bin/python3 -m venv /opt/build \
 && /opt/build/bin/pip install --no-index --find-links /wh --require-hashes --no-deps --only-binary :all: -r /tmp/lock/build.lock \
 && python /tmp/lock/locktool.py split /tmp/lock/lerobot.lock /tmp/lock/lerobot.sdists \
      --wheels /tmp/lock/wheels.txt --sdists /tmp/lock/sdists.txt \
 && CMAKE_POLICY_VERSION_MINIMUM=3.5 PATH=/opt/build/bin:$PATH \
    /opt/build/bin/pip wheel --no-index --find-links /wh --no-build-isolation --no-deps --no-binary :all: \
      --require-hashes -r /tmp/lock/sdists.txt -w /wheels

FROM base AS lerobot-env
COPY containers/aarch64/lerobot.lock containers/aarch64/lerobot.sdists containers/locktool.py /opt/vla-router/containers/aarch64/
COPY --from=sdist-wheels /wheels /tmp/wheels
RUN --mount=type=bind,from=wh,target=/wh \
    cd /opt/vla-router/containers/aarch64 \
 && python locktool.py split lerobot.lock lerobot.sdists --wheels /tmp/wheels.txt --sdists /tmp/sdists.txt \
 && pip install --no-index --find-links /wh --require-hashes --no-deps --only-binary :all: -r /tmp/wheels.txt \
 && pip install --no-deps --no-index /tmp/wheels/*.whl \
 && rm -rf /tmp/wheels /tmp/wheels.txt /tmp/sdists.txt /opt/vla-router/containers

FROM lerobot-env AS lerobot-runtime
COPY libero-assets /root/.cache/libero/assets
COPY containers/libero_config.py /opt/vla-router/containers/
RUN python /opt/vla-router/containers/libero_config.py /opt/libero-home /root/.cache/libero/assets
ENV HOME=/root LIBERO_CONFIG_PATH=/opt/libero-home VLAR_LIBERO_ASSETS=/root/.cache/libero/assets \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/opt/vla-router/src

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
 && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python containers/check_env.py --role sim --lock containers/aarch64/lerobot.lock
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
 && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python containers/check_env.py --role pi05 --lock containers/aarch64/lerobot.lock
CMD ["python", "-m", "vlar_harness.servers.pi05_server", "--checkpoint", "/weights/pi05_libero_finetuned", \
     "--host", "127.0.0.1"]

FROM base AS xvla
COPY containers/aarch64/xvla.lock /tmp/xvla.lock
RUN --mount=type=bind,from=wh,target=/wh \
    pip install --no-index --find-links /wh --require-hashes --no-deps --only-binary :all: -r /tmp/xvla.lock && rm /tmp/xvla.lock
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
 && python containers/check_env.py --role xvla --lock containers/aarch64/xvla.lock
CMD ["python", "-m", "vlar_harness.servers.xvla_server", "--model-path", "/weights/X-VLA-Pt", \
     "--lora-path", "/weights/X-VLA-libero-long-peft", "--host", "127.0.0.1"]
