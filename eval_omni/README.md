# 多模态评测镜像

该目录提供 8 个基准所需的评测运行环境。正式评分由 `src/eval/tasks/` 的适配器完成，推荐从仓库根目录使用 `src/docker/run_verifier.sh`；MMSWE 另需带 SWE-bench 依赖的 verifier 镜像。

## 构建输入

模型与数据不打入镜像，通过运行时挂载提供。第三方 harness 源码也没有纳入 Git，需要在构建前单独准备到以下目录：

- `harness_repos/lmms-eval/`：须包含本项目使用的 Qwen3-Omni 模型接口及 MMMU-Pro、Video-MMMU、Video-MME v2 任务。
- `harness_repos/OmniVideoBench/`：须包含 `eval/qwen3_omni_eval.py` 及其依赖。

版本及源码树校验值固定在 `harnesses.lock.json`：

| Harness | 固定提交 | 本地修改 |
|---|---|---|
| LMMS | `9a337637931c6817ab794d7a32f2f135fb17ca1b` | 无；保留快照的 3,323 个文件与该 Git tree 完全一致 |
| OmniVideoBench | `fb7a5603bad0c08e28754a670001f04b45983ce3` | `patches/omnivideobench-qwen3-omni.patch`，重建 35 个运行时源码文件 |

OVB 补丁保留外部 GPU 选择、使用 SDPA、统一生成结果解包与解码，并支持冒烟题量限制。脚本校验 commit、补丁 SHA256 和所有交付源码的组合 SHA256；现有目录不一致时直接失败，不覆盖本地修改。上游源码与许可说明随获取流程保留，数据和模型不由此脚本获取。

在仓库根目录执行：

```bash
python3 scripts/sync_eval_bundles.py
python3 eval_omni/prepare_harnesses.py
python3 eval_omni/check_build_inputs.py
bash src/docker/build_images.sh
```

输入检查验证关键文件及锁定源码树，不代表镜像已构建成功或真实评测已通过。构建脚本会在任何镜像构建开始前执行检查，缺失时明确退出。

单独构建基础评测镜像时：

```bash
cd eval_omni
python3 prepare_harnesses.py
python3 check_build_inputs.py
docker build -t omni-eval:local .
```

默认使用 Ubuntu 22.04 可用的 `python3`。基础镜像、Python 和包版本变更后，需要重新验证实际运行环境。

## 评分入口与输出隔离

正常评分必须经过适配器。其行为为：

- 普通调用默认使用 `val`；空值、全量别名或普通调用中的 `eval` 请求会归一到 `val`。非法 split 直接失败。
- 独立 verifier 入口设置 `MMPTB_ROLE=verifier`，默认评分 `eval`。
- runner 的原始输出保存到临时评测目录，适配器只输出当前 split 的汇总；逐题诊断只在 `val` 写出。
- 缺失、损坏或不完整结果不产生有效分数；正常零分仍然有效。

`MMPTB_ROLE` 是防误配标志，不是权限认证。真实隔离依赖独立容器、评测端控制的目录与挂载权限。LMMS/OVB 仍然先推理后筛选；推理前预筛是另行可选加固，本次未实施。

`entrypoint.sh` 是遗留的原生 harness 调试入口，不执行正式的 split 评分。现仅接受显式 `MMPTB_ROLE=verifier EVAL_SPLIT=all` 的运营侧全量诊断；不能用它代替 val 自测或 sealed 评分。

## 本轮验证范围

本轮仅进行了 CPU fixtures、适配器回归、补丁应用脚本验证与打包一致性检查，没有启动新的 loop、训练、GPU 评测或完整镜像构建。详细记录见 [验证范围说明](../docs/validation.md)。

## 运行环境预检

```bash
python3 scripts/check_eval_runtime.py --out runtime-preflight.json
```

命令仅查询 Docker、GPU，并在子命名空间测试权限，不加载模型或运行研究 loop。此前 GPT 会话中 GPU 可见、未找到 Docker 入口，namespace/setgroups 探针失败；后续 Claude 报告其构建环境可用 Docker，但新本机自检为 2/4，两个标准补丁均因基础设施故障失败。环境能力须按实际会话和执行位置核实，不能把宿主机预检等同于最终容器权限。需要在支持所需权限的评测环境完成构建与正负补丁校准，不能通过跳过官方测试中的权限切换来冒充通过。
