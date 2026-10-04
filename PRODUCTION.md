# TrainLab 生产验收与运行手册

版本定位：单机预训练平台候选版。第一阶段按用户确认，仅供本人使用，Linux + 8 张 RTX PRO 6000。未获得字节跳动内部标准，不能宣称通过其标准认证。生产可用性取决于目标环境、业务数据、测试证据和团队验收，不能由界面或代码量证明。

## 架构与信任边界

浏览器 → HTTPS/Caddy 或 SSH 隧道 → 单 worker FastAPI → SQLite WAL 持久队列 → torchrun → DDP Trainer → 本地 volume 中的检查点/权重。

同一 volume 只允许一个调度器，串行执行训练及模型测试；没有多机、高可用或自动故障转移。项目权限在 API 层校验，包含数据列表、提交、日志、取消、恢复与下载。训练使用固定代码、结构化参数，禁止客户端指定 shell、模型下载路径或 Python 代码。子进程环境只继承必要变量，不继承服务访问凭据/云密钥。容器非 root、只读根文件系统、去除 capabilities，训练输出写入 `/state`。

管理员、宿主机 root、容器进程和共享数据卷仍属于同一信任域。API 项目权限不构成文件系统租户隔离；不接受外部不受信任代码或训练镜像。个人 API key 不是 SSO/MFA。当前 100 MiB JSONL 上限、可缩放 Qwen3.5/Qwen3/GPT-2 的小模型路线适合流程验证，不能代表大规模基础模型生产流水线。

## 发布门禁与证据

| 验收域 | 应提供的证据/通过条件 | 当前状态 |
|---|---|---|
| 身份与权限 | 未授权 401、只读写入 403、跨项目 404、吊销/过期立即拒绝后续请求 | 本地自动化测试通过 |
| 数据与任务正确性 | 摘要固定、文档分割无精确重复泄漏、重复提交不重复执行、恢复继承权限 | 本地测试通过；近重复治理待建 |
| CPU 训练链路 | 随机初始化、tokenizer、优化器更新、完整保存与恢复一致 | 已验证小样本，不是能力评估 |
| 备份恢复 | 停服务后 SQLite 一致性备份、逐文件 SHA256、空目录恢复、损坏拒绝 | 本地自动化测试通过；真实语料恢复时间待测 |
| GPU 兼容性 | 保存型号/显存/驱动/拓扑；逐卡 BF16 前后向；8 卡 NCCL | 待目标机器执行 |
| 性能与稳定性 | 固定配置 1/2/4/8 卡 tokens/s、峰值显存；建议至少 24h 训练稳定性观察 | 未执行；24h 是拟定门槛，不是字节标准 |
| 故障演练 | OOM、磁盘水位、容器重启、worker 异常、检查点中断、停电后恢复 | 部分本地模拟；目标机器演练待执行 |
| 模型质量 | 独立测试集、污染排查、loss/PPL 基线、业务任务评测 | 已实现续写与有界 Loss/PPL；独立业务质量验收未完成 |
| 安全与供应链 | 依赖/CUDA 镜像扫描、SBOM、批准镜像 digest、密钥轮换、网关接入 | 本地 API 与 Python 训练环境扫描为 0 已知漏洞；目标镜像扫描待完成 |
| 运维 | 告警收到并响应、异机备份、恢复演练、值班责任人与回滚记录 | 工具已提供；运行制度与演练未完成 |
| 企业集成 | 企业 SSO、集中不可篡改审计、项目配额、审批/变更制度 | 未实现 |

个人试运行优先通过身份保护、数据正确性、GPU、稳定性、备份恢复和供应链检查。SSO/MFA、集中审计及团队配额列为未来多人协作扩展，不作为当前个人使用的阻断条件。对外客户、多节点或无人值守长期服务应单独立项补齐隔离、HA 和组织控制。吞吐、扩卡效率、RPO/RTO/SLO 阈值需由业务确认后测量，不编造承诺。

## 监控与排障

管理员凭据访问 `/api/metrics`，Prometheus 文本提供任务状态计数、空闲磁盘、调度线程存活。指标没有用户名或任务 ID 标签。给采集器单独签发管理员凭据并安全保管（当前没有独立 monitor 角色）。

建议告警：ready 连续失败 1 分钟；scheduler_alive=0；磁盘接近设定水位的两倍；failed/interrupted 数量增长。阈值为初始建议，需演练调整。尚未部署 Prometheus/Alertmanager 或发送实际告警。

审计记录存储在 `audit_events`，包含服务端请求 ID、时间、主体、项目、方法、路由/任务 ID、响应状态；`/api/audit` 返回最新 200 条。数据在本地 SQLite，管理员可修改，因此不具备防篡改保障。需后续接入集中日志、保留策略、不可变归档。当前不自动删除审计与历史训练文件；磁盘监控和人工保留策略必须安排。

故障处置：先保存任务 ID 与请求 ID，检查 ready 和容器日志，确认磁盘/驱动，再处理。不要直接改数据库把失败任务标为成功。训练从最近完整检查点恢复；不完整检查点不承诺可恢复。无法通过 GPU 验收时禁止放大模型掩盖环境错误。生产进程必须由容器管理，直接 kill 单独 uvicorn 可能留下训练子进程，需要确认清理后再重启。

## 停机备份

先安排维护窗口并停止训练。snapshot 包含数据库、全部 datasets、jobs、tests；不包含 HF cache、凭据原文、TLS 私钥、`.env` 或镜像。数据库包含凭据摘要，训练文件包含业务数据：备份也应限制访问并加密存储。

以下路径以 Linux 服务器为例，UID 10001 对备份目录必须有写权限：

```bash
sudo install -d -m 700 -o 10001 -g 10001 /srv/trainlab-backups
docker compose stop trainlab
docker compose run --rm -v /srv/trainlab-backups:/backup trainlab python -m scripts.backup backup --state /state --snapshot /backup/release-v3
docker compose run --rm -v /srv/trainlab-backups:/backup:ro trainlab python -m scripts.backup verify --snapshot /backup/release-v3
docker compose start trainlab
```

snapshot 目录必须不存在，不覆盖既有备份。每个文件校验 SHA256，数据库执行 integrity_check。校验和用于发现意外损坏，不提供对恶意修改的签名保护。运行中的调度器持锁，备份工具会拒绝；不得绕开锁在线复制数据库。复制成功后再做异机加密备份，至少演练一次恢复。

## 恢复与升级回滚

恢复必须指向新的空 volume，并保持容器内绝对挂载路径 `/state`，因为检查点和配置记录了绝对路径。不要覆盖正在使用的数据。可在隔离环境下执行：

```bash
docker volume create trainlab-restore-v3
docker run --rm --user 0:0 -v trainlab-restore-v3:/state trainlab-pretrain:local chown 10001:10001 /state
docker run --rm --user 10001:10001 -v trainlab-restore-v3:/state -v /srv/trainlab-backups:/backup:ro trainlab-pretrain:local python -m scripts.backup restore --state /state --snapshot /backup/release-v3
```

随后用与备份配套的镜像及临时 compose override 将服务挂到该恢复 volume，先在隔离端口启动。重启会把旧 running/cancelling 标记 interrupted，但 queued 任务会继续调度；演练应隔离真实 GPU 或先在受控备份中安排空队列。验证文档、权限、检查点恢复后再切换入口。恢复会回滚凭据吊销状态；重新核对吊销清单并轮换凭据，避免恢复已失效访问。

发布流程：记录源 commit → 运行测试和扫描 → 目标 Linux 构建/验收 → 固定镜像 digest → 备份 → 维护窗口发布 → health/ready/短训练验收 → 观察告警。失败时先停新版本，恢复配套数据与旧镜像。数据库未来变化需提供显式迁移与回滚；不要混用新数据库与不兼容旧镜像。

## 8 卡验收记录模板

记录日期/执行人、源码 commit、镜像 digest、OS/驱动/CUDA/PyTorch、每卡 UUID/显存、拓扑、CPU/RAM/磁盘、数据 sha256、tokenizer、seed、模型参数、序列长度、precision、micro batch/accumulation/world size、实际 tokens/s、峰值显存、开始/结束时间、退出码、验证 loss、检查点恢复步数、故障与处置。

扩卡比较需要保持有效 batch 与 token 数、数据划分等实验条件一致：有效 batch = micro batch × gradient accumulation × GPU 数。DDP 每卡保存完整模型，8 卡总显存不是单模型可用的连续显存。先验证小规模配置，再扩大模型；没有实测前不承诺 7B/更大规模的可训练性或工期。

参考框架：[OWASP ASVS](https://owasp.org/projects/asvs) 用于建立应用安全检查项；[PyTorch torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html) 用于核对分布式启动语义；[NVIDIA DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html) 用于规划 GPU 监控。引用这些资料不等于通过认证，也不等于获得字节内部规范。

## 本次依赖整改记录

原 API 框架依赖和旧训练栈扫描检出已知公告后，升级到 FastAPI 0.142.2 / Starlette 1.7.0 / PyTorch 2.13.0 / Transformers 5.18.0 / Accelerate 1.15.0。补齐 Transformers v5 的 warmup_steps 接口适配，并重新执行训练和恢复一致性验证。

`reports/api-dependency-audit.json`：15 个已解析包，0 条已知漏洞报告；`reports/local-training-dependency-audit.json`：本地环境 51 个包，0 条。扫描日期 2026-10-03，使用 pip-audit 默认漏洞数据源。它们不包含目标 Linux 镜像内的系统/CUDA 库，也不是永久安全保证。`reports/local-python-environment.txt` 记录验证环境，不作为跨平台 lock 文件。发布前在最终镜像重新生成 SBOM、扫描并固定产物 digest。

可用 `make check` 执行 API/权限/恢复测试、前端语法与依赖一致性检查，`make smoke` 执行离线 CPU 训练和恢复权重比较。`make audit AUDITOR=<pip-audit可执行路径>` 扫描 API 依赖及 `PYTHON` 指定的已安装环境（包含 torch）；发现问题会返回非零退出码，不自动忽略公告。已接入 GitHub Actions：测试、Linux amd64 镜像构建、三种架构的镜像内 CPU 训练/恢复/续写/评分与生产 API 启动检查，通过后发布 GHCR；CI 无 NVIDIA GPU。
