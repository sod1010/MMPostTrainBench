# MMPostTrainBench

**MMPostTrainBench: Benchmarking Autonomous Research for Multimodal Post-Training**

论文中的基准统一称为 **MMPostTrainBench**，多模态研究框架称为 **MMResearch**。本仓库提供 benchmark 评测代码；任务 ID、目录名及命令中的 `mmposttrainbench` / `mmptb` 前缀保留，以兼容现有配置。

**CLI 智能体能否对多模态大模型做后训练?** MMPostTrainBench 衡量编码智能体(Claude Code、Codex,或让 omni 模型驱动自己)能否**后训练 Qwen3-Omni-30B-A3B**、提升它在某个多模态基准上的分数。智能体拿到基座模型、评测脚本和 GPU 预算,交付一个微调后的 `final_model`,在**密封的留出集**上打分。这考察的是智能体做真实多模态 AI 研发的能力(数据获取、训练方法、迭代),而不只是写代码。

它把智能体后训练从文本 LLM 扩展到音频/图像/视频,并用 **Docker** 的分离式 verifier 流水线做隔离评测(在你自己的 GPU 主机上跑,调度器自选)。

## Main results

![MMPostTrainBench Figure 1](assets/figure1.png)

Figure 1 from the paper. Left: final model outcomes averaged across eight tasks,
compared with the common base model. Right: best-so-far held-out candidate
accuracy on MMMU-Pro; labels show the best candidate score.

## Source release scope

This source release provides the eight-task MMPostTrainBench evaluation suite,
training and verifier container recipes, configuration examples, and CPU regression
tests. MMResearch is the research framework described in the paper; its controller
and memory implementation are not included in this benchmark-code distribution.

Dataset files, model weights, private credentials, run logs, and Git history are not
included. Download third-party resources from their upstream providers under the
applicable licenses. See [NOTICE](NOTICE), [credential handling](docs/disclosure-security.md),
and [validation and reference scores](docs/validation.md).

**Reference-score scope:** `suite.yaml` and `src/eval/baselines.json` now match
Tables 2 and 7 of the paper. The MMSWE development/test sizes are 100/480, and its
published final-test baseline is 2.29%. Explicit `n_dev` and `n_eval` fields separate
research feedback from final scoring; `n_full` is the combined pool size.
`run_task.sh <bench> oracle` checks raw full-test metrics, and the audit reset path
requires a matching task, split, and denominator. Small smoke runs cannot certify
baseline reproduction. Freeze dataset IDs/revisions, checkpoint bytes, media, and
grader configuration for actual experiments; matching counts alone is insufficient.
See [validation](docs/validation.md) for what has and has not been executed.

## Research use and compliance / 研究用途与合规

This benchmark is intended for evaluation and research. The benchmark protocol does not add restrictions to the MIT license. Users are responsible for ensuring compliance with the terms of service of any third-party services they employ. Agents autonomously execute code and training workflows; running in an isolated environment is recommended.

**Agent rule:** External-model distillation is disallowed. Agents may not call external model APIs to generate or synthesize training data.

本基准面向评测与学术研究；评测规则不构成对 MIT 许可证的额外限制。使用者须自行遵守所用第三方服务的 ToS；智能体会自主执行代码和训练流程，建议在隔离环境中运行。禁止通过外部模型 API 生成或合成训练数据。适用范围、审计证据及运行建议见 [研究使用说明](docs/research-use.md)。

## 8 个基准

| 基准 | 模态 | 公开 HF 来源 |
|---|---|---|
| `mmau` | 音频理解 | `lmms-lab-audio/mmau` |
| `mmar` | 音频推理(选择题) | `ngqtrung/mmar` |
| `mmmu_pro` | 图像理解 | `MMMU/MMMU_Pro` |
| `video_mmmu` | 视频理解 | `lmms-eval/VideoMMMU` |
| `videomme_v2` | 视频理解 | `MME-Benchmarks/Video-MME-v2` |
| `jointavbench` | 音视频联合 | `roverx12345/jointavbench` |
| `omnivideobench` | 音视频视频问答 | `NJU-LINK/OmniVideoBench` |
| `mmswe` | 图像 + 代码(SWE) | `SWE-bench/SWE-bench_Multimodal` |

前 7 个是**感知/推理选择题**,分数 = 准确率。第 8 个 `mmswe`(SWE-bench Multimodal)是**代码/软件工程**维度:模型读 issue 文本 + 截图、产出统一 diff,在实例官方镜像的文件系统内打补丁跑测试(`FAIL_TO_PASS` + `PASS_TO_PASS` 全过才算解决),分数为 resolved 率,归一到同一个 `accuracy` 契约。原生判定器免 Docker 守护进程(见 [`dlc_native_grade.py`](src/eval/tasks/mmswe/dlc_native_grade.py));发布版的外层 verifier 仍由 Docker 启动。MMSWE 另需装有 `swebench` 的 Python 环境、拉取官方镜像层的网络,以及实际判定环境中的 namespace/chroot 和用户组切换权限。仅能启动非特权 user namespace 不够,官方测试中的 `su`/`setgroups` 也必须正常执行。基础设施错误不计为模型零分。运行前须在目标主机完成镜像与 grader 校准，验证范围见[验证说明](docs/validation.md)。

## 架构 —— 隔离的分离式 verifier 评测

```
智能体容器 (mmptb-agent)  ──训练──►  final_model  ──►  独立的 verifier 容器  ──►  reward.txt
   可见:工作区 + 基座模型                              可见:final_model + 评测数据(只读)、
   不可见:评测数据                                     HF 离线、烘焙好的 evaluate.py/judge
```

智能体和打分器运行在**不同的容器**里:智能体对评测数据和打分代码**没有 shell 访问权**;verifier 在一个只读、HF 离线的数据挂载上,运行烘焙进镜像的 `evaluate.py` + 污染判定,产出 `reward.txt`。整套用普通的 `docker run` + bind mount 实现;68G 的 omni 模型 + 100G+ 媒体需要一块共享大存储承载(可使用支持所需容量的共享文件系统)。

**自带 GPU / 调度器。** 训练与评测都是纯 `docker run`,在**单台 GPU 主机**上即可跑完整循环(`run_loop.sh`:agent → verifier → 反作弊闸门 → reward,全在本地容器内完成,不依赖任何集群)。要扩到多机/排队,自行用 k8s / Slurm / 你自家的平台把 `run_agent.sh` 与 `run_verifier.sh` 这两段包一层提交即可 —— 本仓库不绑定任何特定调度器。

## 快速开始

运行前提：Linux、可运行容器的 Docker daemon、NVIDIA Container Toolkit，以及可访问公开模型／数据和依赖源的网络。评测需足够 GPU 显存容纳模型；完整训练采用八卡配置。模型和媒体需至少数百 GB 存储空间。MMSWE 官方实例另外需要浏览器、namespace/chroot 和用户组切换支持。

先编辑 `src/docker/config.env` 中的仓库、模型、数据、缓存、工作区和日志路径。API 密钥由使用者在仓库外保存；源码仅提供配置变量。给已有模型评分时设置 `FINAL_MODEL_SRC`；工作区已有 `final_model` 会被复用，因此更换模型请使用新工作区。

完整操作指南(在哪跑、docker 需求、GPU 需求、注意事项):**[`src/docker/README.md`](src/docker/README.md)**。简言之:

```bash
# 0. 配置
cp src/docker/config.env.example src/docker/config.env    # 填入路径 / HF token

# 1. 生成 verifier 构建上下文 bundle(全部 8 个 bench)
python3 scripts/sync_eval_bundles.py
#   单个基准: python3 src/harbor_adapter/run_adapter.py -b mmau -m qwen3-omni-30b -o harbor_tasks

# 2. 从公开 HF 下载开源模型 + 8 个基准的数据
bash src/docker/prepare_data.sh                         # 或:prepare_data.sh mmau mmar

# 3. 获取固定版本的第三方 harness，并构建镜像
#    omni-train = 公开 ms-swift Megatron-SWIFT 训练环境(train_omni/);omni-eval = 评测环境(eval_omni/)
python3 eval_omni/prepare_harnesses.py
python3 eval_omni/check_build_inputs.py
bash src/docker/build_images.sh
#    已把 omni-train 推到 registry 的话:bash src/docker/build_images.sh --skip-train

# 4. 在隔离 verifier 中给模型打分(基座模型 = 一个下限分)
BENCH=mmau bash src/docker/run_verifier.sh              # -> $LOGS/verifier/{metrics.json,reward.txt}

# 5. 跑完整的 智能体-训练-评测 循环
bash src/docker/run_loop.sh                             # 见 src/docker/README.md
```

`run_verifier.sh` 的 `BENCH=<id>` 会把该 bench 的 bundle `tests/` 挂载覆盖镜像里的 `/tests`,因此**一个镜像即可给 7 个感知 bench 中任意一个打分**(无需按 bench 重建镜像);`mmswe` 用自己的 verifier 镜像(前置见上文「8 个基准」)。

## 仓库结构

| 路径 | 说明 |
|---|---|
| `src/docker/` | Docker 驱动(纯 `docker run`,不绑定调度器):`build_images.sh`、`prepare_data.sh` + `resources.json`、`run_verifier.sh`、`run_agent.sh`、`run_loop.sh`、`config.env.example`、`cheat_gate.sh` |
| `src/harbor_adapter/` | 从 `template/` + `src/eval/tasks/<bench>/` 生成每个 bench 的 verifier 构建上下文 bundle |
| `train_omni/` | omni-train Docker 镜像(公开 ms-swift Megatron-SWIFT 训练环境)+ 公有 megatron SFT 入口 `run_sft_agent.sh`;是智能体镜像与训练容器的共同基座 |
| `eval_omni/` | omni-eval Docker 镜像 + 各 bench 的 runner(`run_mmau_official.py`、…、`run_mmswe_official.py`、`convert_omnivideobench.py`) |
| `src/eval/` | 评测逻辑:各 bench 的 `tasks/<bench>/evaluate.py`、共享 `lmms_common/`、`tasks/mmswe/dlc_native_grade.py`(免 docker 守护进程的 SWE 判定)、`judge.py`、`contamination.py`、`baselines.json`、`factors.json`、split/diag 工具 |
| `harbor_tasks/` | 已生成的 bundle,8 个 bench 各一个(即上文架构里 verifier 的**构建上下文**:`tests/` = evaluate.py + test.sh + metadata,被 `build_images.sh` 烘焙进 verifier 镜像)。7 个感知 bench 共用同一通用 `tests/Dockerfile`;`mmswe` 用专属 `tests/Dockerfile`(多一层 `swebench` venv + `dlc_native_grade.py`),见上文说明 |

## 添加 / 复现一个基准

每个 bench 需要:(a) `src/eval/tasks/<bench>/evaluate.py`(音频/音视频类是 self-runner 薄封装,lmms-eval 类是委托给 `lmms_common` 的薄封装);(b) 在 `src/docker/resources.json` 里的公开数据源。然后:

```bash
python src/harbor_adapter/run_adapter.py -b <bench> -m qwen3-omni-30b -o harbor_tasks
bash src/docker/prepare_data.sh <bench>
BENCH=<bench> bash src/docker/run_verifier.sh
```

## 数据

各基准数据集归其各自作者所有,在准备阶段由 `prepare_data.sh` 从其公开 HuggingFace 仓库下载,本仓库**不转发**这些数据 —— 包括派生物:反作弊用的**污染比对参照**(全部 8 个 bench 的评测问题文本;`mmswe` 每个实例出两条 —— 问题侧 = issue+截图 URL、答案侧 = gold patch/test_patch/`FAIL_TO_PASS`,因为在 mmswe 上 gold patch 就是答案)不入库,准备好数据后本地生成一次即可:

```bash
python src/eval/build_contam_reference.py --out src/eval/contam_ref_all.jsonl
# 只补一个 bench(其余 bench 的源缓存可能已不在本机):
python src/eval/build_contam_reference.py --only mmswe --append --out src/eval/contam_ref_all.jsonl
```

缺这个文件时 `cheat_gate.sh` 的污染比对腿会报 `skipped`(**不会**静默算作"干净"),judge 与 provenance 两腿照常。同理,OOD 反 benchmax 探针的配置是 operator-only 的:`cp src/eval/ood_map.example.json src/eval/ood_map.json` 后自行填写(填好的那份也不入库——智能体若读到它就知道哪个 bench 在守自己)。

## 反作弊闸门(`cheat_gate.sh`)——四腿,命中即回落 baseline

每个真实结果都必须过 `cheat_gate.sh`;任一腿命中,该 bench 的 reward 被覆盖为基座模型 baseline(`src/eval/baselines.json`),与文本版 PostTrainBench 的 reward-hack 处理一致。四腿:

1. **LLM judge**(`judge.py`)——审阅智能体交付的代码,判定是否作弊;
2. **污染比对**(`contamination.py`)——训练数据 vs 全 8 bench 的评测问题(参照见上文,缺失报 `skipped`);
3. **数据来源**(provenance)——实际训练用的 `DATASET_PATH` 必须落在 workspace 内;用 operator 本地/预存数据集 = 作弊;
4. **自测 split 审计**——智能体的每次自测(其 `cmd.sh`,或 `SPLIT_AUDIT_SPECS` 指向的不可变 job spec)必须 `EVAL_SPLIT=val`;出现 `eval`/空/`all`/`full` 说明它在**密封留出集**上自测,该分数不可信。

任何一腿因参照/密钥缺失而没跑,报 `skipped` 并在 `note` 里点名,绝不把"某腿没跑"读成"干净"。`GATE_DISABLE=1` 显式跳过整个闸门(此时分数**不可上报**,日志会写明)。
