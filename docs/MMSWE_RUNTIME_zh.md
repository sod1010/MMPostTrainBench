# MMSWE 运行与验收约定

此文档区分评分器代码修复与正式研究 loop 可用性。CPU 回归通过不代表容器权限验收完成，也不代表原始 shell 队列具备 sealed 隔离能力。

## 失败不能变成分数

MMSWE adapter 使用独立临时目录，校验生成与评分的样本集合、划分、失败标记和逐项报告。基础设施失败、生成失败、部分结果、评分进程失败均不产生指标。有效的模型零分仍然可以返回。

批处理运行环境可以使用 `src/dlc/score_job.py` 的 `run` 与 `wait` 子命令。它为成功、失败、超时生成原子的状态文件；等待端检查状态、worker 退出码、心跳与超时。仅有 DONE 或旧 reward 不代表成功。运行端固定 validation 角色；正式 sealed verifier 应直接使用 adapter，不应调用自检包装器。

该工具是结果交付协议，不是沙箱。不得把允许智能体提交任意 shell 的队列当成可信评分服务。正式 loop 的训练、评分必须通过操作员控制的执行入口；隔离策略应在执行端强制实施，角色环境变量只用于防止配置错误。

## 冻结测试资产

从数据集的 `image_assets.test_patch` 和 `image_assets.patch` 提取官方 URL，生成含 `urls` 列表及数据版本信息的 JSON。操作员在有网络的环境运行：

```sh
python scripts/prefetch_mmswe_assets.py --urls asset_urls.json --out assets
```

仅在所有资产成功取回后生成 `assets/manifest.json`。记录输出的 SHA256；将 manifest 和 blobs 作为操作员只读资源挂载到评分环境，并设置：

```sh
export MMSWE_ASSET_MANIFEST=/operator/assets/manifest.json
export MMSWE_ASSET_MANIFEST_SHA256=<实际清单哈希>
```

配置 manifest 后，缺失或损坏资产立即产生基础设施失败，不再退回联网下载。未配置时保留原有在线方式。此缓存解决下载可用性，不能替代 namespace、chroot、组权限或标准补丁的真实运行验收。

## 冻结问题截图

从数据集 `image_assets.problem_statement` 提取 URL。对已有截图缓存执行：

```sh
python scripts/freeze_mmswe_images.py --urls image_urls.json --cache-dir image_cache --out images
```

此脚本需要 Pillow；它只复制并校验已有图片，保留已有 `.miss` 缺失记录，不重新下载图片。既无图片也无缺失记录的 URL 会导致冻结失败。

```sh
export MMSWE_IMAGE_MANIFEST=/operator/images/manifest.json
export MMSWE_IMAGE_MANIFEST_SHA256=<实际清单哈希>
export MMSWE_REQUIRE_FROZEN_IMAGES=1
```

四臂和基座必须使用同一份只读清单。未知 URL、内容变化或清单哈希变化均拒绝生成；明确记录为不可用的图片在所有运行中保持相同缺失状态。生成结果包含每条样本的实际图片数和内容哈希，指标记录清单哈希、总图片数与无图片样本数。清单中的 URL 数量不是样本数。

这些设置不改变单次补丁生成任务、hints、默认最多 4 张截图和 3072 个新 token。若要加入仓库源码或交互工具，应作为协议变更另行建立基线。

## 必须保留的真实验收

1. 标准补丁通过，空补丁和坏补丁未解决，且均无基础设施失败。
2. Chart.js 含二进制资产的标准补丁通过；仅正确检测到 infra-fail 不算通过。
3. 使用最终镜像验证 val → dev、verifier eval → test，以及生成与评分的实际样本 ID 集合。
4. 在真实执行端验证训练代码无法访问操作员测试数据或其它实验工作区；原始 shell 队列未完成替换前不启动正式四臂 loop。
