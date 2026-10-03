# TrainLab 单机 GPU 预训练部署

适用：Linux x86_64 + NVIDIA GPU，用户目标为 8 张 RTX PRO 6000。实际型号、显存、驱动、系统内存和磁盘需要在机器上确认。本包实现从随机权重初始化 GPT 文本模型，不下载基础模型、不做 LoRA。

## 当前能力和边界

- 字节级 BPE 分词器只用训练集学习；GPT-2 架构随机初始化。
- 每行一篇文档的 UTF-8 JSONL；上传上限 100 MiB；按文档精确去重，seeded SHA256 约 90/10 划分训练和验证。
- EOS 连接文档，连续定长打包，丢弃末尾不足一段的 token；相邻文档之间没有额外 attention 隔离。
- 1 至 8 张 GPU 的单机 DDP、BF16/FP32、梯度累积、激活检查点、AdamW 和 cosine 调度。
- 串行持久任务队列；训练日志、loss、验证 perplexity、模型文件下载；完整检查点恢复。
- 个人 API 凭据、admin/operator/viewer 三角色、项目级 API 权限、凭据过期与吊销、请求审计。第一阶段仅供本人使用，权限模型为未来可信团队协作保留；尚无 SSO、进程/文件系统租户隔离或高可用。
- 不包含模型推理 API。训练产物是 Hugging Face 模型与分词器目录；对外推理服务需单独部署。
- 不是海量语料预训练集群：尚无流式/分片语料、对象存储、FSDP/ZeRO、跨节点训练、自动评测与模型服务。

8 卡 DDP 在每张卡保存完整模型。8×单卡显存不是一块可任意使用的合并显存，也不能据此保证某个大模型能训练。先单卡、短序列、小模型实测，再增加规模；PCIe 拓扑和通信会影响扩卡收益。

## 1 在目标机器检查环境

```bash
nvidia-smi
nvidia-smi topo -m
uname -m
free -h
df -h
```

需要 Docker Engine、Docker Compose v2 和 NVIDIA Container Toolkit。使用 NVIDIA 的官方安装说明配置容器运行时；不要在不确认现有业务的情况下升级或重启宿主机驱动。

官方文档：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

基础镜像为 `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime`，训练依赖在 `server/requirements-train.txt` 中固定直接版本。Dockerfile 已固定经 registry 查询的基础镜像 digest；这是选定的兼容基线，不宣称是最新版。目标驱动必须支持该 CUDA 运行时及显卡；用下面的真实算子测试验收，不能只凭 nvidia-smi 显示了 GPU 就判定兼容。

## 2 复制目录并初始化

将整个 trainlab 项目复制到服务器，例如 `/opt/trainlab`，在该目录运行：

```bash
python3 scripts/init_env.py
docker compose build
```

脚本仅首次生成 `.env`，不会覆盖已有配置。升级旧版本时，手动移除 `.env` 中的 `TRAINLAB_API_TOKEN`；生产模式检测到共享令牌会拒绝启动。历史无项目归属的任务/数据仅管理员可见，其他用户需重新上传并提交。

首次签发管理员凭据（30 天有效）；原文只写入指定的 0600 文件，命令输出不会显示令牌：

```bash
docker compose run --rm trainlab python -m scripts.users issue --subject admin --project research --role admin --days 30 --credential-file /state/credentials/admin.key
```

管理员在服务器上安全读取该文件，将凭据通过密码管理器交付对应本人。浏览器登录时粘贴文件中的值，不要放入 URL、日志或 Git。可以通过容器内的文件管理工具读取 `/state/credentials/admin.key`。其他成员分别签发，不得共用：

```bash
docker compose run --rm trainlab python -m scripts.users issue --subject researcher-01 --project research --role operator --days 30 --credential-file /state/credentials/researcher-01.key
docker compose run --rm trainlab python -m scripts.users issue --subject reviewer-01 --project research --role viewer --days 30 --credential-file /state/credentials/reviewer-01.key
docker compose run --rm trainlab python -m scripts.users list
```

operator 可操作本项目数据和训练，viewer 只读本项目，admin 能跨项目访问并查看审计/监控。项目名决定共享边界；同项目成员可读彼此数据和日志。API 数据库只存高熵令牌的 SHA256 摘要。妥善交付后移走凭据原文文件；备份工具不包含 credentials 目录。

吊销命令：`docker compose run --rm trainlab python -m scripts.users revoke --key-id <list返回的id>`。下一次请求立即失效；已在传输中的下载不会被主动断开。轮换应签发新凭据、更新调用方后吊销旧凭据。

运行数据使用 Docker named volume `trainlab-state`，不在镜像里；更新镜像不会删除数据。不要使用 `docker compose down -v`，那会删除该 volume。

## 3 验证容器和全部显卡

```bash
docker compose run --rm trainlab python scripts/preflight.py
docker compose run --rm trainlab torchrun --standalone --nproc_per_node=8 scripts/ddp_check.py
```

第一个命令逐卡执行 BF16 矩阵运算及反向传播。第二个验证 8 卡 NCCL all-reduce。必须全部通过后再提交多卡训练。如果只有部分卡可见，请先确认容器 GPU 授权和设备状态；不要随意关闭硬件或网络保护来掩盖错误。

## 4 启动并访问

```bash
docker compose up -d
docker compose ps
docker compose logs --tail=100 trainlab
```

默认只绑定服务器的 `127.0.0.1:8000`。在你自己的电脑建立 SSH 隧道：

```bash
ssh -N -L 8000:127.0.0.1:8000 your-user@your-server
```

浏览器打开 `http://127.0.0.1:8000/gpu.html`，输入分配给本人的个人凭据。浏览器只请求同源 API，令牌仅存在页面内存，刷新后重新输入。现有 chatgpt.site 地址只提供界面预览，不代理访问你的服务器。

如需局域网直连，可在 `.env` 将 `TRAINLAB_BIND` 改为机器的内网 IP，并配置访问范围；令牌不应通过不可信的明文网络传输。团队/公网访问建议使用下节 HTTPS 入口。

## 5 可选的域名 HTTPS 发布

有域名、DNS 已指向该服务器，并允许 80/443 后，在 `.env` 增加真实域名：

```text
TRAINLAB_DOMAIN=trainlab.example.com
```

将上面的示例替换为你自己的域名，然后运行：

```bash
docker compose -f compose.yaml -f deploy/compose.https.yaml up -d --build
```

Caddy 负责 TLS 和反向代理。训练 API 仍要求令牌，8000 默认保持 loopback 绑定。该方式发布的是训练控制台，不是已训练模型的推理端点。个人凭据和项目权限已实现，但本版本仍不适合公开接受不受信任客户。需要企业 SSO、独立作业沙箱和网关限流后再扩大边界。

## 6 第一次验收

1. 进入 GPU 预训练，确认实际显示 8 张 GPU 的型号、显存和驱动。
2. 下载页面提供的流程测试语料，上传并确认格式校验通过。示例是人工合成数据，仅验证链路，不用于能力评估。
3. 选择第一张 GPU，默认 6 层/384 维、序列 256、词表 4096、micro batch 2、累积 8、100 步，保存/验证间隔 25。
4. 提交并确认训练 loss、验证 loss 和日志来自真实进程；完成后下载 `final` 下的配置、权重与分词器文件，保持目录结构。
5. 再提交一次，在第一个完整检查点写完后点击停止，然后恢复；新任务应从原步数继续，配置和数据不变。
6. 选择 8 张 GPU 做短跑并比较吞吐。保持有效 batch 不变做扩卡对照时，相应降低梯度累积，避免同时改变优化条件。

默认模型按 4096 词表上限约 12.32M 参数。词表实际训练结果可能更小；平台显示估计值，日志报告真实参数数。

## 数据准备

```json
{"text":"第一篇完整文档正文……"}
{"text":"第二篇不同来源的完整文档正文……"}
```

至少 20 行，单行不超过 1 MiB，仅接受非空 `text` 字段。训练前还会检查去重后的文档数和打包后的序列数；语料不足会明确失败。精确去重不能发现改写或近重复，同源文档应先在数据准备阶段归组和审查。验证集用于调参与曲线，最终业务能力还需另设测试集。

数据上限 100 MiB 适合开始学习和验证流程。大规模语料需要改用分片、流式读取与独立数据流水线，不能用反复训练小语料代替足够的数据多样性。

## 恢复与运维

- 取消会终止 torchrun 及其子进程；恢复点为最近一个带完整写入标记的检查点，不是点击停止的精确时刻。
- 保留最近两个检查点。恢复会新建任务，继承配置、不可变数据摘要和原分词器，使用 Trainer 恢复优化器/调度器/随机状态。
- 服务停止会中断当前任务；重启标记为 interrupted。排队中的任务保留并按序继续。不要对同一数据目录启动多个 API 进程。
- 一个任务失败不会阻塞后续任务。OOM 通常先减少序列长度和 micro batch；DDP 不解决单卡模型状态装不下的问题。
- 控制台显示本平台任务，没有跨其他用户/进程的 GPU 排他锁；安排专用卡或由统一资源管理器分配。
- 日志 API 只返回最后 128 KiB，完整文件留在 volume。指标页面返回最近一段记录；最终统计保存在 summary.json。
- 停机备份、恢复演练、监控和发布门禁见 [PRODUCTION.md](PRODUCTION.md)。先备份再升级；目前不承诺零停机或高可用。
- 管理员可用 `docker compose exec trainlab sh` 进入容器查看 `/state/jobs/<任务ID>/output`。不要把数据目录暴露为无鉴权静态目录。

## 开发验证

Python 3.12 环境：

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.13.0 -r server/requirements-train.txt pytest==9.0.3 httpx==0.28.1
.venv/bin/python -m pytest -q tests
.venv/bin/python -m scripts.smoke_train
```

在 Linux 安装 torch 时应使用目标 CUDA 对应的官方 wheel；Docker 路径已经固定 CUDA 运行时。smoke_train 使用离线 CPU 验证训练、保存和恢复，不能替代 CUDA/NCCL 验收。

## 技术资料

- https://huggingface.co/docs/transformers/v5.18.0/en/main_classes/trainer
- https://pytorch.org/get-started/locally/
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- https://www.nvidia.com/en-gb/products/workstations/professional-desktop-gpus/rtx-pro-6000/

浏览器下载为避免大文件占满页面内存，限制单文件 256 MiB；更大的权重在服务器的 output/final 目录复制，或使用带 Authorization 请求头的 `/api/jobs/{job_id}/artifact/{path}` 下载 API。文件不对匿名用户开放。

## v3 运行保护

默认最多 20 个排队/运行中任务；磁盘空闲低于 5 GiB 拒绝新任务和上传，并终止运行任务。可以调整 compose 中的 `TRAINLAB_MAX_PENDING_JOBS`、`TRAINLAB_MIN_FREE_BYTES`。水位应按最大检查点实测尺寸提高，5 GiB 并不保证足够。

每个任务默认最多运行 6 小时；API 的 `max_runtime_seconds` 可设置为 60 至 604800 秒。控制台可按分钟调整，默认 360 分钟。只有执行开始才计时，排队不计入。超时作为 failed，完整检查点可恢复。上传同时最多 2 个，每文件 100 MiB；普通配置请求最多 64 KiB。容器限制 1024 PID、64 个服务并发连接；并发超限可能返回 503。上述是单实例资源保护，不是按用户计费配额或网关 DDoS 防护。

`/api/health` 表示 HTTP 进程存活；`/api/ready` 检查调度线程与磁盘水位。Docker healthcheck 使用 ready。unhealthy 不会被 Docker restart 策略自动重启，须由监控通知管理员处理。监控接口 `/api/metrics` 和审计接口 `/api/audit` 需要管理员凭据。服务不记录 Authorization/请求正文/查询参数；训练日志仍可能含语料衍生信息，应按训练数据等级保护。

训练栈已升级至 PyTorch 2.13.0 / Transformers 5.18.0。不同版本间的 optimizer/RNG 检查点兼容性未经保证；旧版训练请在配套旧镜像中完成恢复，不跨版本直接续训。CUDA 13.0 需要匹配的宿主驱动，目标机器未实测前不要直接升级驱动。


## v4 工作台操作

1. 连接后先检查计算资源与磁盘水位；无可用 GPU、调度器未就绪、配置非法或只读角色时，提交按钮不可用。
2. 上传语料后点击「预览语料」，查看 SHA256 和前三篇文本。每篇最多 800 字符，服务端最多扫描 64 行；这不是完整数据质量检查。
3. 模型起点可选预设或自定义。高级参数允许调整层数、隐藏维度、头数、随机种子、warmup、weight decay 和激活检查点；保存与评估间隔各自独立。
4. 「保存本机草稿」只在明确点击后写入浏览器 localStorage，包含任务名和训练参数，不包含凭据、数据内容、数据集 ID 或 GPU 选择；载入时保留当前数据集与 GPU。凭据仍只保存在页面内存中。
5. 「复用配置」把历史任务的完整配置放回表单，生成副本名称；不会立即开始训练。请确认数据集权限、GPU 和运行时限后提交。提交失败重试保留相同幂等标识，刷新页面后应先检查任务列表。
6. 任务按名称或 ID 搜索、按状态筛选，每页 12 条。统计显示当前身份可访问的全部任务，不受分页影响。管理员可查看所有项目。
7. 任务详情分为指标、日志、配置、模型文件。取消「跟随最新」可阅读已有日志；关闭自动刷新可暂停轮询。指标 CSV 导出的是接口当前返回的记录，日志导出的是当前尾部（最多 128 KiB），都不是无限历史。
8. 支持导出配置 JSON 与指标 CSV，浏览器阻止下载时请查看下载设置或使用带鉴权的 API。大于 256 MiB 的文件仍需服务器复制。

更新此版前等待当前训练完成或主动停止并确认检查点，按 PRODUCTION.md 做停机备份，再执行 `git pull --ff-only` 和 `docker compose up -d --build`。本版不改变数据库结构或训练依赖，前端与 API 必须一起更新；浏览器刷新后重新输入凭据。

开发校验增加 `node --test tests/ui.test.cjs`，可用 `make check PYTHON=.venv/bin/python` 一并运行 API/权限测试、前端逻辑测试、语法检查与 pip 依赖检查。需要支持 node:test 的 Node.js。
