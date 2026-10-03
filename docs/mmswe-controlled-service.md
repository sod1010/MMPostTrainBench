# MMSWE 受控执行服务：待真实环境验收

本目录新增独立服务与普通 Docker 实例评分后端。旧队列、旧启动入口和已有研究结果不变。完整服务链尚未在真实 Docker/GPU 环境验收，不能据此解除旧入口的 78 防护，也不能声明四配置 loop 已可重跑。

## 本版完成的代码

- Linux 本地 Unix socket 使用 `SO_PEERCRED` 获取真实用户身份；每个研究运行必须分配不同的非 root 宿主 UID，不能用请求中的 uid、role 或 run 字段冒充。操作员使用独立的 0600 socket。
- 只接受 `train`、`import_model`、`score`、`status`；`seal` 仅操作员可调用。未知字段、命令、环境变量、评分脚本路径、密封划分覆盖均拒绝。
- 输入采用逐级目录描述符打开，拒绝路径穿越、符号链接和特殊文件。模型复制到服务私有目录，记录每个文件的完整 SHA-256 和包含配置的整体身份；不是硬链接或文件名指纹。
- 模型接受扁平 Qwen3-Omni safetensors 导出；拒绝 pickle、自定义 Python、auto_map 与不匹配的分片。结构校验不等于实际加载成功。
- 训练与推理在固定镜像、固定资源配置的普通容器中运行，默认非 root、只读根文件系统、无网络、无附加 capability，只挂载本次任务所需目录。训练代码可以自由编写，不能替换服务端评分入口。
- 请求编号防重复收费、禁止复用编号改变内容；服务端预留任务预算并记录实际耗时和资源配置。失败无分数；重启不自动重放任务，失去准确计时的任务按预留预算保守记账。清理失败阻止继续接单。
- 服务端冻结模型后才评分，记录模型、策略、截图／资产清单及预测哈希。开发评分只返回标量和分母；密封任务状态、结果只能由操作员读取。封存后关闭该运行的研究请求。

## 普通容器评分后端解决什么

`src/mmswe_service/docker_grade.py` 复用原评分器的官方镜像校验、解包、补丁应用、资产还原和成绩解析，只替换测试执行阶段：将已核验并准备好的实例文件系统导入普通 Docker 容器，在容器已有的命名空间中执行官方脚本。

实例容器不挂宿主目录或 Docker socket，不使用 host PID，不申请 privileged、CAP_SYS_ADMIN 或任何 cap-add。保留默认 root 容器中的用户组切换能力，由容器运行时建立隔离，避免在受限 pod 中再次 unshare。若普通 Docker 操作被平台拒绝，直接失败，没有提权或无隔离回退。

当前为每个实例导入一个临时镜像，结束后清理。该实现优先用于正确性验收；大规模开发集重复评分的磁盘配额、导入耗时与缓存策略仍需实测，不能直接承担正式 loop 的吞吐承诺。评分端不需要 GPU，生成端需要模型运行环境。

## 平台与部署前置条件

1. 平台明确允许调用的本地 Docker daemon。Docker socket 只能由操作员服务访问，不能传入智能体、训练容器、生成容器或实例测试容器。
2. 服务源码、配置、状态、缓存和基础模型由操作员控制；智能体不能写这些位置。每个研究运行使用独立宿主 UID。当前全 root／同 UID 的旧 launcher 不满足本版身份契约，尚未接入。
3. 状态目录必须位于有实际硬配额的独立存储区。导入／复制有字节上限，但不替代训练输出、评分 scratch 和日志的文件系统硬配额；无配额不能启动不可信训练。本版不负责配置平台存储配额。
4. 基础模型和缓存应先冻结；记录其内容身份并核查离线数据缓存能在只读挂载下工作。`baseline_manifest` 将基座清单及逐文件哈希绑定到策略，服务启动时验证全部模型字节；这仍不替代真实模型加载验收。
5. 按真实镜像、实际运行 UID 验证文件／进程／网络隔离、子进程清理、组切换、官方正负例、模型加载和结果归属。本地 CPU 替身不证明这些条件。

这些前置条件尚未全部完成，因此本版服务没有部署启动，旧 loop 仍禁用。

## 操作员配置

参见 `src/mmswe_service/policy.example.json`。替换所有路径、镜像摘要及清单哈希。镜像必须预构建并固定 registry digest（`repo@sha256:...`）或本地不可变镜像 ID（`sha256:...`）；运行时不自动拉取镜像，不从研究请求接受 Docker 参数或评分环境配置。

`container_profile` 可选 `strict-v1`（默认）或 `platform-managed-v1`。前者显式传递原隔离参数；后者仅省略平台接管的 `ipc`、`cap-drop`、`security-opt` 参数，**并不取消这些要求**。训练／生成和实例评分都在 `create` 后、`start` 前读取 Docker inspect：禁止特权、附加 capability、host 网络／PID／UTS，要求 private IPC、no-new-privileges 和对应 capability 删除；训练／生成还要求非 root 与只读根文件系统。实例不得有宿主／卷挂载。未达标时留下检查记录、清理容器并失败，不自动换 profile 重试。

平台若只在内核层实施策略而不在 inspect 暴露，当前保守门禁不能证明它等价，仍拒绝。inspect 检查也不能替代真实进程和浏览器验收。旧 `MMSWE_DOCKER_HARDENING` 开关显式拒绝，避免脚本仍设为 0 却无效或悄悄弱化。独立评分入口由操作员设置 `MMSWE_CONTAINER_PROFILE`；broker 从固定配置传入，不接收研究请求的环境覆盖。

发布入口在解包前检查上述设置。`MMSWE_DOCKER_NETWORK` 仅允许不设置或设为 `none`；`bridge`、`host` 和自定义网络不会被静默忽略，也没有自动放网回退。测试执行的是候选补丁修改后的程序，评分阶段仍有候选代码执行。需要在线资源的官方测试必须先完成受控资源访问方案并验收；`none` 下这类测试的失败不能用来证明标准补丁无效。

真实镜像摘要必须替换模板中的全零占位符，否则配置加载失败。服务启动后将配置哈希与配置快照保存到私有 `policy.json`；重启使用不同的运行 UID、预算、镜像、划分或资源设置时，必须另开状态目录。同一状态目录中的历史任务也必须匹配此哈希。基础模型内容及评分数据现在由必填的 `baseline_manifest`、`evaluation_contract` 绑定；真实实验身份、隔离和模型加载仍需验收。

部署前可运行只读盘点，不会创建容器、启动服务或申请 GPU：

```sh
python3 scripts/mmswe_deployment_check.py \
  --config /etc/mmptb/service.json \
  --contract /private/evaluation_contract.json \
  --out /private/deployment-check-unique.json
```

`--contract` 可省略；提供时核对已有 dev100/test480 契约的文件字节、样本 ID 哈希和仓库数量。此 CLI 参数是额外参考盘点；实际生成和评分已经由配置中必填的 `evaluation_contract` 接入运行时。配置不存在、模板摘要、目录身份、镜像或清单不匹配会报阻塞；即使静态检查全过，收据仍标明 `loop_ready=false`，等待真实评分及服务全链路验收。退出码 0 仅代表静态检查通过，2 表示配置阻塞。

正式数据划分为 dev 100 / test 480；示例已更正 `test_limit`，数据文件字节、样本 ID 和顺序由 `evaluation_contract` 校验；技术自检子集使用冻结顺序的前缀，不能只匹配分母。示例中的镜像摘要、资源与总超时不是已测部署参数。特别是 `score.timeout=7200` 包含生成与串行评分，不能默认足以处理完整集合；先测冷／热耗时，再依照研究预算确定上限。模板不构成正式运行放行。

构建上下文为仓库根目录，Dockerfile 位于 `src/mmswe_service/Dockerfile`。训练镜像基于既有训练依赖环境，生成镜像基于既有评测依赖环境，分别构建并记录镜像摘要。容器内 Python 必须包含对应依赖；构建此层不会自动安装模型训练依赖。所有模型文件均在容器中读取；主服务不导入 torch 或模型代码。

在满足上述前置条件的专用执行主机上，入口为：

```sh
PYTHONPATH=src python3 -m mmswe_service.service --config /etc/mmptb/service.json
```

服务公开 socket 使用 0666，仅为允许本地连接；运行权限由内核传递的 UID 判断。操作员 socket 为同路径加 `.operator`，权限为 0600。状态目录 0700，socket 父目录不可被研究 UID 修改。服务需要读取各研究 workspace；建议由平台配置只读访问权限。不要让 agent 获得服务 UID 或 root。

请求示例（JSON 文件）：

```json
{"op":"train","run":"gpt56","request_id":"train_001","code":"submission_code","data":"training_data"}
```

训练代码目录内必须有 `train.py`，容器内使用 `MMPTB_DATA=/input/data`、`MMPTB_BASE_MODEL=/input/base`、`MMPTB_OUTPUT=/output/model`。输出为扁平推理模型。不能直接提交包含 optimizer/checkpoint 嵌套目录的整个训练目录。

```json
{"op":"import_model","run":"gpt56","request_id":"import_001","model":"inference_export"}
```

模型冻结完成后，返回 artifact 标识。评分只接收该标识：

```json
{"op":"score","run":"gpt56","request_id":"score_001","artifact":"服务返回的标识"}
```

```json
{"op":"status","run":"gpt56","job":"服务返回的任务标识"}
```

客户端：`PYTHONPATH=src python3 -m mmswe_service.client --socket /run/mmptb/broker.sock --request request.json`。

`seal` 使用相同 artifact 字段，经操作员 socket 提交。封存前必须等待该运行其他任务完成；封存后不再接受该运行的研究任务。若密封评测因基础设施故障失败，当前版保守保持封存，不自动重试；由操作员先保留证据并处理后端，不能用新研究结果替代已封存的模型。

预算目前按运行总执行秒数预留／扣减。字段 `profile_gpu_seconds_upper_estimate` 为任务墙钟时间乘以配置 GPU 数的保守估算，纯模型导入记为零；它包含复制和 CPU 评分时间，不能当作实测 GPU 用时或论文 GPU 成本。正式实验需要单独记录各阶段实际分配时长，分离操作员最终评测开销。生成容器使用评分 profile；实例测试资源由独立的 `instance_resources` 配置定义，默认仍为 8 CPU、16 GiB、512 PID、0 GPU。多个研究请求在单 worker 中顺序执行；四臂并行调度尚未接入。

### 实例评分资源与 EAGAIN 排查

操作员配置增加 `instance_resources`，完整字段为 `{"cpus":8,"memory_gib":16,"pids":512}`。资源值进入已有策略哈希，由 broker 传给评分器；不能由研究请求覆盖。同一状态目录不能在实验中途更改资源。若省略此字段，使用上述默认值；默认值并非所有仓库都已验收。

独立验收入口可使用 `MMSWE_INSTANCE_RESOURCES_JSON` 传同一对象。兼容 Claude 的 `MMSWE_DOCKER_PIDS=8192`、`MMSWE_DOCKER_MEM=32g` 作为独立入口的有限资源别名；不能与 JSON 混用。负数、零、无限制、异常单位或超出允许范围均在镜像准备前拒绝。broker 不继承这些宿主别名。`8192/32g` 已在 远端测试环境 诊断配置的两个 diegomura gold 上通过；它不是正式默认配置，也未证明完整开发集或 publish 隔离配置通过。

容器在 create 后、start 前核对 Docker inspect 的 `PidsLimit`、`Memory`、`MemorySwap` 和 `NanoCpus`，将请求值与实际值记入 pull.log；不匹配则清理并拒绝运行。原有硬化及网络门禁继续有效，这次资源支持不等于接受 `HARDENING=platform / NETWORK=bridge`。

`spawn node EAGAIN` 可能涉及 PID/线程上限、用户级进程限制或其他资源压力。`--cpus` 是 CPU 时间配额，不保证旧 Node 的 `os.cpus()` 返回配额内核数；远端测试环境 记录中配额 8 CPU，Node 仍报告 40。提高 pids 和内存后通过，只证明新资源组合可行，不能区分两个参数的因果作用。preflight.log 增加测试前后的 cgroup PID、内存事件与用户进程上限（文件可读时），不混入官方测试日志。诊断日志由候选代码所在容器产生，仅作工程线索，不能替代宿主记录。正式资源配置应使用验收过的统一值，并检查并发任务的总内存/PID预算。

## Claude 可以执行的下一步验收

在平台允许普通 Docker 容器的评分主机上，复用现有标准补丁预测和冻结资产，使用安装了 swebench 的操作员 Python 运行 `src/mmswe_service/docker_grade.py`。参数与旧 `dlc_native_grade.py` 一致；设置 `MMSWE_DOCKER` 为获准的 Docker 可执行文件绝对路径，`MMSWE_DOCKER_HOME` 为操作员私有 Docker 配置目录。仍使用原来的资产 manifest 及 SHA-256。

每次输出到新目录。先验证 Chart.js 标准补丁，随后在同一后端完整重跑 4 项自检。接受条件为 Chart.js applied=true、resolved=true、infra_failure=false，两个负例仍拒绝。容器拒绝、缓存失败、超时或清理失败都不能计为模型零分。不要求授予 cap，也不改变平台授权策略。

后续再完成服务镜像构建、真实身份边界测试和受限开发／密封样本。没有执行这些步骤前，不更新论文结果、不启动正式四臂。

### 浏览器沙箱失败与零分的区别

Chrome 的 `FATAL:...zygote_host_impl_linux.cc... No usable sandbox` 表示浏览器尚未正常启动。即使补丁已应用、日志有开始／结束标记、外层脚本退出零，评分器仍将其记为 `infra_failure=true`，不能计为模型零分。该检测仅分类错误，不修改 Chrome 参数、官方测试或容器安全策略。

优先在平台批准、支持原浏览器沙箱的执行环境中完成 Chart.js 原用例。新增非 Chrome 的带资产用例可以补充覆盖，但原 Chart.js 项继续记为未通过；不能靠替换用例把原四项验收改成通过。改变浏览器沙箱参数会形成单独的运行配置，不能直接冒充原配置验收。

批量重判也须检查每实例基础设施状态和测试证据；调度器显示 Succeeded、处理进度到达分母，以及 resolved 全零，都不单独证明这些零分有效。

Karma 的终止记录 `Firefox failed N times (cannot start). Giving up.`（或 Chrome）也判为基础设施失败，即使日志有完整首尾标记。识别兼容日志颜色转义；单次失败后成功连接、测试中引用错误文字或补丁中的同名文本不触发该规则。

### 在解包和正式评分前做 CPU 预检

运行 `scripts/mmswe_runtime_preflight.py --help`。工具仅使用已经存在的不可变镜像引用，不拉镜像、不挂宿主目录、不申请 GPU、不更改系统策略。每次输出到新目录，产生配置收据和退出状态；失败不会自动省略参数。两个模式必须分别检查：

```sh
python3 scripts/mmswe_runtime_preflight.py --docker /trusted/bin/docker \
  --image 'repository@sha256:实际摘要' --kind workload --profile strict-v1 \
  --out /private/preflight/workload-unique
python3 scripts/mmswe_runtime_preflight.py --docker /trusted/bin/docker \
  --image 'sha256:实际实例镜像ID' --kind instance --profile strict-v1 \
  --probes browsers --test-user chromeuser --out /private/preflight/browser-unique
```

上面的镜像值是说明占位符，必须替换。浏览器检查须使用实际官方实例环境，而非仅含评分器的 verifier 镜像；可以用 `--chrome` / `--firefox` 指定实例内的绝对二进制路径。身份检查默认为 `nobody`，浏览器检查默认为 `chromeuser`。工具记录 CPU 可见性，并验证进程中的 no-new-privileges 和 capability 删除；浏览器探针验证 Chrome 页面、收集 `chrome://sandbox` 状态、验证 Firefox 截图。操作员仍须检查沙箱状态及二进制来源，探针通过不构成官方 4/4。

退出码：0 为探针通过但不代表官方验收；2 为创建或配置拒绝；3 为运行探针失败；4 为清理失败，需要操作员处理。始终保留 `result.json`；清理失败禁止接着做验收。若 Docker 二进制或任一父目录全员／组可写，不能作为服务可信执行入口；应由操作员准备验证过哈希、位于可信私有路径的副本，不修改平台共享文件权限。

### 非 UTF-8 测试日志

测试程序可能输出不完整的 UTF-8 字节，官方解析器再次按严格 UTF-8 打开日志时会抛出异常。评分器保留原始 `test_output.txt`；仅在解码失败时生成以替换字符解码的 `test_output.utf8.txt`，将该副本传给官方解析器，并把原始／解析副本的 SHA256 与首次错误偏移记入 `test_output.decoding.json` 和实例报告的 `log_decoding` 字段。合法 UTF-8 日志仍直接使用原文件。

这项处理只修复日志读取，不更改测试结果、测试名单、浏览器沙箱或完整性判断。存在权限错误、缺失测试标记等问题时仍拒绝评分。历史日志复解析应保存到独立目录，保留原始报告，不把离线解析当作重新通过了真实环境验收。


### 冻结输入与离线 HTTPS 资源快照

服务必填 `baseline_manifest` 与 `evaluation_contract`，格式均为操作员私有文件的 `{"path":"绝对路径","sha256":"文件SHA256"}`。前者在启动时核对完整基座文件集、逐文件哈希及模型结构；后者核对 Arrow 数据字节、dev100/test480 ID 与顺序。生成容器使用指定的 Arrow 文件，不再依赖 Hugging Face 缓存中“最新”快照。生成返回的 ID、预测顺序和契约哈希不一致时，broker 不启动 grader。基座、数据、镜像、截图/资产及资源限额共同进入策略冻结；仍须另外冻结四个提供方的模型版本、运行 UID 和研究预算。

可选 `offline_resources` 使用同样的文件 pin；默认 `null`。`scripts/freeze_mmswe_resources.py --url HTTPS_URL --out NEW_PRIVATE_DIR` 只在准备阶段抓取明确指定的公共资源，并保存时间、字节哈希及独立 TLS 材料。正式容器仍 `--network none`，固定域名映射到容器 loopback；内置服务器只为确切 URL 提供固定 GET/HEAD 响应，未知 URL 返回 404，其他方法返回 405，无转发或在线回退。Docker inspect 必须确认 host mapping。

资源运行包装器不改官方 eval.sh，而在隔离实例内启动固定 HTTPS 服务，向 Node 进程传入局部 `NODE_EXTRA_CA_CERTS`。签名 CA 的私钥在准备后丢弃，叶证书只用于该快照；不修改宿主信任库。客户端容器 root 可以读到叶密钥，它不是外部服务凭证。该模式当前针对 Node HTTPS 资源，未宣称覆盖 Chrome/Firefox 信任库、动态多 URL 下载、npm/yarn 安装或完整网络需求。

这会冻结官方测试使用的外部资源，属于需要单独验收的环境协议。必须先对使用这些资源的原 gold/负例做对照，证明测试语义适用和 su 后信任配置有效；在此之前仅可称为“网络快照原型”，不能报网络闭合或用其失败给模型记零。报告同时记录 `evaluation_contract_sha256` 与 `offline_snapshot_sha256`，确保四臂使用同一资源版本。

### 明确选择入口守卫：entry-guard-v1

部分平台拒绝客户端配置 NNP/capability 参数，但允许进程主动降低自身权限。本模式不依赖 inspect 中的两个字段来证明进程权限；它使用固定静态 ELF 守卫和一次宿主握手。必须明确选择 `container_profile=entry-guard-v1`，没有失败后自动切换。

操作员先用 `scripts/build_mmswe_guard.py --out 新的私有目录` 构建 Linux x86-64 静态程序，记录源码、编译器和二进制哈希。策略增加 `runtime_guard={"path":"实际guard路径","sha256":"实际哈希"}`；独立 grader 对应 `MMSWE_RUNTIME_GUARD` / `MMSWE_RUNTIME_GUARD_SHA256`。其他模式不接受守卫 pin。训练/生成镜像使用 `src/mmswe_service/Dockerfile.guard`，构建上下文根目录须包含该二进制 `guard`。实例 rootfs 由操作员在导入前放入同一个守卫，不要求实例内有 Python。

容器以 root 启动的第一段代码只能是已核对字节的静态守卫；不通过 shell 或候选入口启动。守卫先设置 NNP、清空补充组和 ambient/inheritable capabilities。训练/生成进一步删除全部 bounding/permitted/effective capabilities 并永久降到 UID/GID65534；评分保留固定 Docker 默认能力子集，排除 NET_RAW、MKNOD 和额外能力，以保留 su/groups。进程通过系统调用核实自己的实际状态。

执行顺序为：检查 Docker 配置和精确入口/参数/镜像 → 从未启动容器复制守卫并核验哈希 → 启动守卫 → 核验第一条权限证明及本任务随机 nonce → 发出一次性放行信号 → 守卫执行固定任务入口。未收到放行信号、nonce 错误或任一步失败都不执行候选代码。证明产生时没有候选进程，因此不接受运行后由候选代码自行报告的隔离状态。

其余约束不变：无 cap-add/privileged/unconfined、无任意外网、私有 IPC/PID/UTS、训练只读根文件系统与固定挂载、实例无宿主挂载。每次正常/异常退出都清理；守卫日志有大小和时间上限。静态 inspect 的 compliant 只表示通过第一阶段，entry-guard 模式必须另外完成二进制验证及握手。

CPU 守卫验收使用 `scripts/mmswe_guard_preflight.py`，分别跑 workload 和 instance；它验证 exec/su 后状态，不证明浏览器或模型可运行。原 `mmswe_runtime_preflight.py` 保留旧两个模式，避免把尚不支持的浏览器探针错误地当成守卫验收。真实训练/生成、官方 gold/负例、网络快照和资源负载仍需分别验收。该模式并不解开容器 user namespace 限制，也不关闭 Chrome 沙箱。

### p5.js 浏览器内 HTTPS 信任

部分 p5.js 官方用例由 Puppeteer 执行，即使静态 eval.sh 只显示 grunt；不能据此标为纯 Node。其 stock mocha-chrome.js 本身可能包含浏览器参数，验收必须保留并记录原始内容，不能把这些用例的运行结果泛化为 Chart.js 沙箱已可用。

`NODE_EXTRA_CA_CERTS` 只配置 Node，不保证 Chromium 信任快照 CA。快照可显式增加 `browser_nss={"homes":["/root"],"sha256":{"cert9.db":"实际哈希","key4.db":"实际哈希"}}`，相应 NSS 数据库置于快照目录 `nssdb/`。操作员用受信任 certutil 创建仅包含该 CA 的数据库、核对导出证书与快照 CA 一致，固定工具/包/数据库哈希。评分阶段把数据库放入实例 `/root/.pki/nssdb`；拒绝已有数据库或符号链接，不改宿主证书库，不关闭 TLS 校验，不改浏览器启动参数。此版本只支持 stock p5.js 的 root HOME，不能推断 su 用户或 Firefox 已配置；启用后仍须原 gold/负例验收。

### 操作员之间的 共享存储 文件交接组件

`mmswe_service.handoff.HandoffQueue` 是独立的可信控制面传输组件，尚未接入正式 broker 的 GPU 调度。它不新增研究 API，不依赖请求内的 run/uid 字段鉴权，也不把 共享存储 文件所有者当作跨节点身份。队列根目录必须由操作员私有控制；GPU 生成进程的真实挂载/UID 边界仍须另外验收。当前组件只注册 dev 请求，不对研究客户端开放密封划分。

控制器先注册 run/request_id、固定顺序的实例 ID、策略/评测契约摘要与冻结产物身份；可信生成程序调用 `publish()`，以临时目录写入 predictions.jsonl 与 ready.json，刷盘后原子发布。消费者只处理已经登记的任务，使用 no-follow 私有复制冻结输入，校验任务/摘要/样本顺序后，调用操作员代码中固定的 grader 回调。请求文件不能提供命令、脚本路径或环境变量。

回调须在收到的 deadline 内终止子进程、清理本任务容器，并独立核对官方报告的完整性、infra 和覆盖。只有整数 correct/n 通过检查后，队列才计算并回写标量反馈。合法零分与基础设施失败分开；失败没有数值分数、逐实例日志不进入反馈。同一请求编号改内容被拒绝；同一完成任务重复消费只返回已保存结果，不再评分。持久化为 running 的任务在失去 worker 后记为 infra_failed，不自动重放；状态已提交但反馈未写完时可重建反馈。清理失败会阻止下一任务执行，须操作员先处理后端。

单消费者锁基于文件系统 flock，反馈按原子替换落盘。它不提供硬配额、跨节点时钟/锁语义证明、四臂预算调度或独立研究身份认证；不能仅凭这些 CPU 文件传输测试就宣布分布式 loop 已部署。正式接入要将它与已有 broker 的请求归属、模型冻结、预算与密封机制连接，而不是旁路这些约束。
