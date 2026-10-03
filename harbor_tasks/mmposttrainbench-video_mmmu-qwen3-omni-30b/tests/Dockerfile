# Verifier image for harbor's separate-verifier mode (PR #1655), OMNI variant.
#
# Built from `tests/` as context. Harbor builds this whenever the task's
# [verifier].environment_mode = "separate", and uploads the agent's
# [[artifacts]] into the resulting container at the same paths. For
# mmposttrainbench we also run it directly with `docker run` (see src/docker/),
# bypassing the log-streamer ENTRYPOINT with `--entrypoint /bin/bash ... /tests/test.sh`.
#
# UNLIKE the text PostTrainBench verifier (which was FROM cuda + vllm 0.11 +
# inspect_evals), the omni verifier is FROM the eval-deliver "omni-eval"
# image — the delivered isolation core (transformers 4.57 / datasets 3.6 /
# qwen-omni-utils / lmms-eval editable / self-runners at /opt/eval/runners).
# That image is the single source of truth for the pinned omni env; we only
# layer the codex CLI (contamination judge) and bake /tests on top.
#
# Build the base first from eval_omni/, tag it, then pass it here:
#   DOCKER_BUILDKIT=1 docker build -t omni-eval:local eval_omni/
#   docker build --build-arg OMNI_EVAL_IMAGE=omni-eval:local -t <verifier> tests/
ARG OMNI_EVAL_IMAGE=omni-eval:local
FROM ${OMNI_EVAL_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV NO_PROXY="localhost,127.0.0.1"
ENV no_proxy="localhost,127.0.0.1"

# Node.js 22.x — only needed for the codex CLI contamination judge invoked
# from test.sh. The omni base ships python3.12 + ffmpeg + git but not curl.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Codex CLI (contamination judge). Agent CLIs deliberately omitted — the
# verifier doesn't run an agent.
RUN npm install -g @openai/codex@0.98.0

# Log-streamer entrypoint (parity with the text verifier / harbor's Modal
# dashboard) + system monitor. Overrides the omni base's ENTRYPOINT.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY system_monitor.sh /usr/local/bin/system_monitor.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/system_monitor.sh

# Workspace exists but stays empty until the artifact transfer fills it
# (harbor) or the docker orchestrator bind-mounts the agent's final_model.
RUN mkdir -p /home/agent/workspace
WORKDIR /home/agent/workspace

# Bake the tests/ build context (test.sh, evaluate.py, templates/,
# contamination_judge.py, metadata.json, task_context/*) into /tests/.
# entrypoint.sh / system_monitor.sh / requirements-direct.txt land here too
# via `COPY .`; strip them to keep test.sh's listing clean.
COPY . /tests/
RUN rm -f /tests/Dockerfile \
          /tests/entrypoint.sh \
          /tests/system_monitor.sh \
          /tests/requirements-direct.txt && \
    chmod +x /tests/test.sh

# MMAU_RUNNER default already points at /opt/eval/runners/run_mmau_official.py
# (present in the omni base); evaluate.py resolves it there.

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
