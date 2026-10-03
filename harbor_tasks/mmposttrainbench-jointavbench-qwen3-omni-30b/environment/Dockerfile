# Agent image for harbor, OMNI variant.
#
# This is where the CLI agent (claude-code / codex / gemini-cli) lives and
# post-trains Qwen3-Omni-30B. It works in /home/agent/workspace and must
# leave the trained weights at final_model/ (harbor uploads that single
# artifact into the separate verifier container; see tests/Dockerfile).
#
# UNLIKE the text PostTrainBench agent (FROM cuda + vllm 0.11 + inspect_evals),
# the omni agent is FROM the omni *training* image `omni-train:local`, built by
# train_omni/Dockerfile (public ms-swift Megatron-SWIFT stack; CUDA 12.8 + the
# Qwen3-Omni SFT/RL training env). We only layer node + the agent CLIs + the
# harbor log-streamer on top; we do NOT install vllm/flash-attn/inspect (those
# are the text stack and would fight the training env).
#
# Build train_omni/ first (build_images.sh does this as step [1/4]), then this
# image builds FROM it. Override to any registry-tagged copy if you pushed one:
#   docker build --build-arg OMNI_TRAIN_IMAGE=<registry>/omni-train:<tag> -t <agent> environment/
ARG OMNI_TRAIN_IMAGE=omni-train:local
FROM ${OMNI_TRAIN_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"
ENV NO_PROXY="localhost,127.0.0.1"
ENV no_proxy="localhost,127.0.0.1"

# Node.js 22.x for the agent CLIs. The training base has CUDA/python/git but
# not node; install curl/ca-certificates defensively (present on most bases).
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Pinned agent CLIs (same versions as the text PostTrainBench agent).
RUN npm install -g \
    @anthropic-ai/claude-code@2.1.76 \
    @openai/codex@0.98.0 \
    @google/gemini-cli@0.18.4 \
    opencode-ai@1.1.59

# Entrypoint streams /logs/{agent,verifier}/*.txt to PID 1 stdout for the
# harbor dashboard; system_monitor.sh mirrors condor's monitor daemon.
# task.toml's healthcheck writes /timer_start once the streamer is up.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY system_monitor.sh /usr/local/bin/system_monitor.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/system_monitor.sh

# Workspace
RUN mkdir -p /home/agent/workspace
WORKDIR /home/agent/workspace

# Task files. entrypoint.sh / system_monitor.sh / requirements-direct.txt land
# here too via `COPY .`; they're container-internal — strip from workspace.
# contamination_judge.py is stripped for a different reason: it is the verifier's
# anti-cheat probe, and handing the probe to the party being probed is a one-sided
# leak. The adapter already withholds it from environment/ (agent_side=True); this
# rm is belt-and-braces so an older, un-regenerated bundle is still clean.
COPY . /home/agent/workspace/
RUN rm -f /home/agent/workspace/Dockerfile \
          /home/agent/workspace/entrypoint.sh \
          /home/agent/workspace/system_monitor.sh \
          /home/agent/workspace/requirements-direct.txt \
          /home/agent/workspace/contamination_judge.py

# The agent self-evaluates on the VAL split; the sealed EVAL split is scored only
# by the verifier, which passes EVAL_SPLIT=eval explicitly (src/docker/run_verifier.sh).
# Pinning the default here rather than hiding split_util.py is deliberate: keep()
# is symmetric (val is the complement of eval), so knowing one is knowing the
# other, and hiding the module would only break the self-evaluation instruction.md
# explicitly asks the agent to do.
ENV EVAL_SPLIT=val

RUN chmod -R a+rw /home/agent/workspace/ && \
    chmod +x /home/agent/workspace/timer.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
