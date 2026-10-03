# `omni-train` — public Qwen3-Omni-30B post-training image

This directory builds `omni-train:local`, the environment the mmposttrainbench
**agent** runs inside and the image **any training node** uses to run the
Megatron-SWIFT SFT. It is the training counterpart of [`../eval_omni`](../eval_omni)
(the eval/verifier image): both ship only the environment recipe — the model and
data are mounted at runtime, never baked in.

## Why this exists

The agent's job is to post-train the **public** `Qwen/Qwen3-Omni-30B-A3B-Instruct`.
Full-parameter SFT of a 30B-A3B MoE needs TP/EP across 8 GPUs, which the public
[Megatron-SWIFT](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html)
(modelscope/ms-swift ≥ 4.6) supports natively for Qwen3-Omni, including
`use_audio_in_video`, packing, and `loss_scale`. This image is just that public
stack plus the Qwen3-Omni A/V decode extras.

## Build

```bash
# on a host with docker + network (aliyun CR + pypi reachable)
DOCKER_BUILDKIT=1 docker build -t omni-train:local .
```

The default base is a **public** ModelScope×SWIFT image that already ships the
hard-to-compile CUDA deps prebuilt (torch + TransformerEngine + apex + flash-attn
+ megatron-core), so the build only `pip install -U`s the training stack and the
Qwen3-Omni extras — no `nvcc` compile step, no internal registry. Override the
base with `--build-arg BASE_IMAGE=<tag>` to track a newer/closer ModelScope tag
(see the [ms-swift Mirror list](https://swift.readthedocs.io/en/latest/GetStarted/SWIFT-installation.html#mirror)).

`src/docker/config.env` wires this image via `OMNI_TRAIN_IMAGE=omni-train:local` —
the base the **agent** image is built FROM (`build_images.sh` step `[1/4]`). If you
run training on a remote node instead of the local host, push `omni-train:local` to
a registry that node can pull.

## Training entrypoint

`run_sft_agent.sh` (shipped here, copied into the image at `/opt/train/`) is the
public megatron SFT entrypoint. Point `SFT_SCRIPT` at it in `config.env`:

```bash
export SFT_SCRIPT=$REPO_ROOT/train_omni/run_sft_agent.sh
```

It reads `MODEL_PATH` / `DATASET_PATH` / `OUTPUT_DIR` / `MODEL_TYPE`
(`qwen3_omni_moe`) / `MTP_NUM_LAYERS` (`0`) and the TP/EP/GBS/iters knobs by env;
`src/docker/run_agent.sh` sets them when the agent triggers training. It writes HF
safetensors directly (`--save_safetensors true`), so the checkpoint loads into
the eval/verifier image with no conversion.

## From-scratch base (optional)

If you cannot pull the ModelScope base, build on `nvidia/cuda:12.8.0-cudnn-devel`
(**devel**, not runtime — you need `nvcc`) and install TransformerEngine, apex,
flash-attn, and megatron-core per the
[Megatron-SWIFT install doc](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html#environment-setup).
apex is optional: pass `--gradient_accumulation_fusion false` to `megatron sft`
(add it in `run_sft_agent.sh`) to run without it.
