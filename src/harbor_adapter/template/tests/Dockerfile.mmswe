# Verifier image for harbor's separate-verifier mode — mmswe (SWE-bench
# Multimodal) variant.
#
# Same shape as the generic OMNI verifier (tests/Dockerfile): FROM the
# eval-deliver "omni-eval" base + node/codex (contamination judge) + /tests baked
# in. mmswe adds ONE thing the perception benches don't need: a self-contained
# `swebench` venv for the grading stage.
#
# Why a separate venv (not the base python): mmswe's evaluate.py is a two-stage
# seam —
#   stage 1  GENERATE  base python3 (omni stack: torch/transformers/qwen-omni/
#            datasets) runs /opt/eval/runners/run_mmswe_official.py -> patches.
#   stage 2  GRADE      $SWE_VENV_PY runs dlc_native_grade.py (Route B, daemon-
#            free): registry-v2 manifest digest (trust anchor) -> mirror blob +
#            sha256 verify -> unpack rootfs to /dev/shm -> unshare+chroot -> apply
#            patch + run the instance's test suite -> swebench.get_eval_report.
# swebench must NOT be installed into the omni stack (it drags its own dep pins
# that would fight transformers/datasets), hence the isolated venv here.
#
# Runtime needs (verify in the actual grader environment):
#   - egress at grade time: docker registry (registry-1.docker.io) for manifest
#     digests + a blob mirror (docker.1panel.live) for layer bytes + the issue
#     screenshot hosts. task.toml sets [verifier.environment] allow_internet=true.
#   - working mount/PID namespaces, chroot and group switching for official
#     test scripts. The grader tries real-root unshare, then userns; merely
#     starting unshare --map-root-user is insufficient if su/setgroups fails.
#
# Build the base first from eval_omni/, tag it, then pass it here:
#   DOCKER_BUILDKIT=1 docker build -t omni-eval:local eval_omni/
#   docker build --build-arg OMNI_EVAL_IMAGE=omni-eval:local -t <verifier> tests/
ARG OMNI_EVAL_IMAGE=omni-eval:local
FROM ${OMNI_EVAL_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV NO_PROXY="localhost,127.0.0.1"
ENV no_proxy="localhost,127.0.0.1"

# Node.js 22.x — only needed for the codex CLI contamination judge invoked from
# test.sh. curl/ca-certificates: the omni base ships python3.12 + ffmpeg + git
# but not curl. python3-venv: defensive — the base is python3.12; venv is stdlib
# but Debian/Ubuntu split it into a package (no-op on conda-style bases).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates python3-venv && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Codex CLI (contamination judge). Agent CLIs deliberately omitted.
RUN npm install -g @openai/codex@0.98.0

# swebench grading venv (stage 2). Isolated from the omni stack. swebench 5.0.2
# is the validated pin; it pulls its own datasets/GitPython/etc. into the venv,
# so the grader's `from datasets import load_dataset` + `from swebench.harness
# ...` resolve here, not against the base's transformers-pinned datasets.
RUN python3 -m venv /opt/swe-venv && \
    /opt/swe-venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/swe-venv/bin/pip install --no-cache-dir swebench==5.0.2

# mmswe eval env (all overridable at run time):
#   SWE_VENV_PY   grader interpreter (stage 2).
#   SWE_WORK      grader home; blob_cache defaults under it (we point it elsewhere).
#   SWE_SPLIT / MMSWE_SPLIT must not be baked to dev or test: split_util derives
#                 both stages from effective EVAL_SPLIT (val -> dev, eval -> test).
#                 Explicit operator overrides are validated; conflicts fail.
#   MMPTB_SRC_EVAL  where run_mmswe_official.py finds split_util.py — the bundle
#                 stages it flat at /tests (its own default /opt/.../src/eval is
#                 absent in the bundle layout), so pin it here.
#   HF_*_OFFLINE=0  the grader/runner pull the SWE dataset online at grade time
#                 (the grader would otherwise default HF_HUB_OFFLINE=1).
#   MMSWE_BLOB_CACHE  writable dir for sha256-verified docker layers (shared
#                 across instances so common bases pull once).
ENV SWE_VENV_PY=/opt/swe-venv/bin/python \
    SWE_WORK=/opt/eval/mmswe \
    MMPTB_SRC_EVAL=/tests \
    HF_HUB_OFFLINE=0 \
    HF_DATASETS_OFFLINE=0 \
    MMSWE_BLOB_CACHE=/tmp/mmswe_blob_cache
RUN mkdir -p /opt/eval/mmswe /tmp/mmswe_blob_cache

# Log-streamer entrypoint (parity with harbor's Modal dashboard) + system
# monitor. Overrides the omni base's ENTRYPOINT.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY system_monitor.sh /usr/local/bin/system_monitor.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/system_monitor.sh

# Workspace exists but stays empty until the artifact transfer fills it
# (harbor) or the docker orchestrator bind-mounts the agent's final_model.
RUN mkdir -p /home/agent/workspace
WORKDIR /home/agent/workspace

# Bake the tests/ build context into /tests/. For mmswe this includes
# dlc_native_grade.py (the stage-2 grader) alongside evaluate.py, split_util.py,
# templates/, contamination_judge.py, metadata.json. Strip the build-only files
# (Dockerfile / entrypoint / monitor / requirements) to keep /tests clean;
# dlc_native_grade.py is NOT stripped — evaluate.py resolves it next to itself.
COPY . /tests/
RUN rm -f /tests/Dockerfile \
          /tests/entrypoint.sh \
          /tests/system_monitor.sh \
          /tests/requirements-direct.txt && \
    chmod +x /tests/test.sh

# The stage-1 runner (run_mmswe_official.py) is baked into the omni-eval base at
# /opt/eval/runners/; evaluate.py's DEFAULT_RUNNER resolves it there.

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
