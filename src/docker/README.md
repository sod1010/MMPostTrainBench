# MMPostTrainBench — Docker orchestration for the omni pilot loop

Docker-native driver for the harbor task bundle: build the images, run the
agent and the **separate** verifier as containers, prove the omni harbor
evaluation plugs into the loop. Mirrors harbor's separate-verifier contract
(agent → `final_model` artifact → isolated verifier → `reward.txt`) but drives
it with plain `docker run` + bind mounts (harbor's Modal env can't hold a
68G model + 134G of media; a host with a shared large-disk volume can).

## Research use and safe execution

This benchmark is intended for evaluation and research. The benchmark protocol does not add restrictions to the MIT license. Users are responsible for ensuring compliance with the terms of service of any third-party services they employ. Agents autonomously execute code and training workflows; running in an isolated environment is recommended.

External-model distillation is disallowed. Agents may not call external model APIs to generate or synthesize training data. See [research-use guidance](../../docs/research-use.md) for policy scope, audit evidence, and isolation recommendations.

## Where to run
Any host with a reachable docker daemon works — a single GPU box, a workstation
with a remote daemon, or a scheduler node. The daemon may be **remote** (`docker
info` Name = a different host), in which case `-v` mount sources resolve on the
*daemon* host and `docker build` streams the context over the socket. Point
everything (model, data, workspace, logs) at a single shared filesystem the
daemon can bind-mount, configured via `MMPTB_ROOT` (default `/path/to/mmptb_runs`;
see `config.env`). The Qwen3-Omni-30B weights are an **open model**, downloaded
from HF into that root by `prepare_data.sh`. You still need GPUs reachable to the
daemon (≥1 for the verifier, ≥8 for real agent training).

Nothing here binds you to a particular cluster or scheduler: the whole loop is
plain `docker run`, so one GPU host runs it end-to-end. To scale out, wrap the
same three stages (`run_agent.sh` → `run_verifier.sh` → `cheat_gate.sh`) in your
own k8s / Slurm / platform submitter — see the top-level README's
"bring-your-own-scheduler" note.

## Two "Harbor"s (don't conflate)
- **Registry** (goharbor.io): where `omni-eval` images are pushed/pulled. Not used by these scripts directly — they build images locally.
- **Framework** (harborframework.com): the task-bundle format (`environment/`, `tests/`, `task.toml`) these scripts consume. `harbor run --env modal` is the upstream driver; here we substitute plain `docker run` for the reasons above.

## Steps
```bash
# 0. Generate the task bundle (run anywhere; it's pure files)
python src/harbor_adapter/run_adapter.py -b mmau -m qwen3-omni-30b \
       -o src/../harbor_tasks --num-hours 24
#    -> harbor_tasks/mmposttrainbench-mmau-qwen3-omni-30b/

# From the repo root:
cd src/docker
# (edit config.env if your paths/tags differ)

# 1a. Download the open model + MMAU dataset into MMPTB_ROOT (shared storage, ~68G model)
bash prepare_data.sh                # HF_ENDPOINT=hf-mirror; skips if present

# 1b. Build images (omni-train + omni-eval -> verifier + agent)
bash build_images.sh                # --skip-agent for eval-only; --skip-train if
                                    # omni-train is already built/pushed to a registry

# 2. CRITERION #1 — the seam: evaluate.py -> omni runner -> {"accuracy": ...}
bash run_seam_test.sh               # expect ~0.375 at limit=8

# 3. CRITERION #2 — verifier container: test.sh -> metrics.json + reward.txt
bash run_verifier.sh                # scores base model as final_model (floor)

# 4. CRITERION #3 — end-to-end loop: agent -> verifier -> reward.txt
bash run_loop.sh                    # AGENT_ENGINE=placeholder (base-model plumbing check; FRESH=0)
AGENT_ENGINE=claude-code ANTHROPIC_API_KEY=... bash run_loop.sh   # real run
```

## Files
| file | role |
|---|---|
| `config.env` | all paths + image tags + knobs (source of truth) |
| `prepare_data.sh` | HF-downloads the open 30B model + MMAU dataset into `MMPTB_ROOT` |
| `build_images.sh` | builds `omni-train` (public ms-swift Megatron-SWIFT) → `omni-eval` → `verifier` → `agent` (`--skip-train` / `--skip-agent`) |
| `run_seam_test.sh` | criterion #1: `evaluate.py` alone → normalized accuracy |
| `run_verifier.sh` | criterion #2: full `test.sh` in the verifier container |
| `run_agent.sh` | agent half; `placeholder` (default) or a real CLI agent |
| `run_loop.sh` | criterion #3: agent → verifier end-to-end |
| `cheat_gate.sh` | anti-cheat gate: LLM judge + contamination vs all benches + data provenance |

## Isolation contract (read before trusting a number)

Isolation has **two** sides, and both matter:

**Evaluation side** — the agent must never see the sealed test set. Enforced
structurally: the eval `DATA_DIR` and the verifier's `HF_CACHE_DIR` are mounted
**only** in the verifier (`run_verifier.sh`); the agent container gets a separate,
clean `agent_hf_cache`. `EVAL_SPLIT` defaults to `val` for the agent (agent image
`ENV`, and `-e EVAL_SPLIT` in `run_agent.sh`) while the verifier is passed `eval`
explicitly. Note this is a *default*, not a secret: `split_util.keep()` is
symmetric, so val and eval determine each other — the seal is that the agent never
has the eval **media/answers**, not that the split function is hidden. The
verifier's own anti-cheat probe (`contamination_judge.py`) is withheld from the
agent workspace.

**Execution side** — the agent must not read the operator's filesystem: previous
runs' workspaces, their training data, their notes and conclusions. Otherwise the
benchmark stops measuring "can an agent post-train this model" and starts measuring
"can an agent find our homework". This is enforced by the **container boundary**:

`run_agent.sh` / `run_loop.sh` run the agent inside its own container that mounts
**only** `/home/agent/workspace`, `/models:ro`, a clean `/hf_cache`, and an optional
`/train_data:ro`. The operator's host filesystem — previous workspaces, the eval
`DATA_DIR`, the verifier's `HF_CACHE_DIR` — is never bind-mounted in, so it simply
does not exist inside the agent's mount view. Training the agent triggers runs in
that **same** container (`run_sft_agent.sh` via the public `megatron sft` stack baked
into the `omni-train` image), so it inherits the same boundary — no extra dependency
set is exposed and the base model / output dir / workspace are the only paths present.
A dataset the agent points training at that lives outside the workspace simply is not
mountable, which is exactly the operator-data shortcut the gate would otherwise flag.

**Networking stays open, deliberately** — the paradigm requires the agent to source
its own training data. So the container boundary alone cannot stop it from
downloading one of the eval benches off the internet. `cheat_gate.sh` is what
closes that: the loop runs it after the verifier scores, and `GATE_DISABLE=1` skips
it while saying so loudly — a score produced that way is not reportable.

**Residual risk, by design:** the container boundary is a filesystem seal, not a
network ACL. If you scale training out to your own scheduler (see "Where to run"),
give each job the **same** minimal mount set — workspace + base model + output —
and nothing else; do not hand it a credential or mount that exposes prior runs or
eval data. The provenance check in `cheat_gate.sh` (the dataset actually trained on
must resolve under the workspace) is the backstop for that path.

## Notes / gotchas
- **Workspace shadowing**: the agent image bakes task files into
  `/home/agent/workspace`; bind-mounting the host workspace shadows them, so
  `run_agent.sh` *seeds* the host workspace from the bundle's `environment/`
  before mounting. `final_model` then lands on the host for the verifier — no
  `docker cp` of 68G weights.
- **Entrypoint bypass**: images keep harbor's log-streamer as `ENTRYPOINT`;
  direct runs use `--entrypoint /bin/bash ... /tests/test.sh` to exec the
  verifier immediately.
- **MMAU data**: `run_mmau_official.py` does `load_dataset("lmms-lab-audio/mmau",
  split="test_mini")` from `HF_HOME=/hf_cache`. The cache must contain that
  dataset before running the verifier. Offline env
  vars are set so no network is attempted.
- **final_model staging** hardlinks (`cp -al`) when source and workspace share
  a filesystem (both under the shared `MMPTB_ROOT`), avoiding a 68G copy.
- **Contamination judge** runs only if `CODEX_API_KEY` (or `OPENAI_API_KEY`) is
  set; otherwise `test.sh` records "judge skipped" and proceeds. It ships in
  `tests/` only — never in the agent workspace (see the isolation contract).

Run directories must be dedicated children of `$MMPTB_ROOT/workspaces/` and `$MMPTB_ROOT/logs/`. Use a fresh pair per experiment; only `FRESH=1` explicitly resets them. Task generation defaults to 24 hours. Install host `python3`, `huggingface_hub`, `pyarrow`, and `ffprobe`; upstream gated datasets require independently approved access.
