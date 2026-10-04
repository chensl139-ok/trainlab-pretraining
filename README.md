# TrainLab 模型训练与学习平台

可视化学习实验 + 可自部署的单机 NVIDIA GPU 文本预训练平台。

- `/`：浏览器真实 MLP 小实验、训练曲线、数据集和知识库。
- `/gpu.html`：从随机初始化 GPT 开始的服务端训练。上传 JSONL、训练 BPE tokenizer、选择 1–8 张 GPU、监控、取消、恢复和下载模型。
- `server/`：FastAPI + SQLite + torchrun + Transformers Trainer。
- `Dockerfile` / `compose.yaml`：Linux GPU 部署；`deploy/` 可选 HTTPS 入口。

请按 [DEPLOY.md](DEPLOY.md) 检查机器、生成配置、启动和验收。目标环境是 Linux x86_64、Docker Engine / Compose v2、NVIDIA 驱动和 Container Toolkit。CUDA 13.0 兼容性需要在目标机器验证。

## 常用数据集

在 **GPU 预训练 → 准备语料 → 选择数据集** 导入小样本，完成后会自动选中：

| 数据源 | 用途 | 下载方式 |
| --- | --- | --- |
| 合成流程示例 | 检查训练与恢复链路，不代表模型能力 | 随项目提供，离线可用 |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 英文合成故事、小模型预训练练习 | 服务器在线获取 |
| [Wikipedia 中文 / 英文](https://huggingface.co/datasets/wikimedia/wikipedia) | 完整百科文章、中文或英文预训练练习 | 服务器在线获取 20231101 对应语言配置 |

支持 50 / 200 / 500 篇，默认 50 篇；合成示例最多 240 篇。在线源只读取公开 train 集，保留整篇文章，精确去重，最多 20 MiB。小样本采用顺序选取，不是随机代表性样本或官方测试集，不能用来宣称基准性能。

语料预览可查看来源与许可并导出来源 JSON；记录每篇原始行号、标题/URL（源有提供时）、内容 SHA256、下载响应摘要和时间。下载服务提供当前快照，本实现不声称锁定上游版本；已导入的本地内容有摘要校验。相同项目、来源和规模重复导入会复用已有数据。TinyStories 数据卡标注 CDLA-Sharing-1.0，Wikipedia 标注 CC BY-SA 3.0 / GFDL，实际使用与分发应遵循相应许可。

网络要求及边界见 [DEPLOY.md](DEPLOY.md)。原始公开语料按需下载，不随 GitHub 仓库分发。

## 导航交互更新（2026-10-04）

学习页与 GPU 页使用一致的导航名称和顺序，按服务器训练、浏览器学习实验、学习资料分组。当前页重复点击不会重置或刷新；教学模块切换支持浏览器返回/前进，知识库章节可直接链接。微调练习参数在同一页面内切换后保留。手机端使用带文字的抽屉导航，支持 Esc、遮罩关闭和键盘焦点约束。

跨到 GPU 页面或离开 GPU 页面仍属于整页导航；GPU 凭据只在页面内存中，重新进入后需要连接，服务端训练不受影响。部署和运维资料在新标签页打开。

## v4 工作台更新

- 资源概览、磁盘水位、任务状态、任务名称/ID 搜索、状态筛选与服务端分页。
- JSONL 拖拽上传、进度与取消、前三篇语料预览和 SHA256 摘要。
- 自定义模型架构、独立评估/保存间隔、优化器参数、参数量和有效 batch 估计。
- 本机参数草稿、完整配置复用、提交/停止/恢复确认、重复提交保护。
- 指标曲线和数值表、日志跟随开关、配置与指标导出、模型文件列表；支持窄屏和键盘操作。

所有任务和资源数字来自 API。无 GPU 时可以准备数据和参数，不能提交训练。详见 [验证记录](VALIDATION.md)。

## 在服务器部署

这是私有仓库。服务器需要先获得该仓库的访问权限；使用 GitHub CLI 时可先运行 `gh auth login`，再用下面的命令克隆。不要在 URL 中写入访问令牌。

```bash
gh repo clone chensl139-ok/trainlab-pretraining
cd trainlab-pretraining
```

已配置 GitHub SSH key 的服务器也可使用 `git clone git@github.com:chensl139-ok/trainlab-pretraining.git`。随后执行：

```bash
python3 scripts/init_env.py
docker compose build
docker compose run --rm trainlab python scripts/preflight.py
docker compose run --rm trainlab torchrun --standalone --nproc_per_node=8 scripts/ddp_check.py
docker compose run --rm trainlab python -m scripts.users issue --subject admin --project research --role admin --credential-file /state/credentials/admin.key
docker compose up -d
```

默认只监听服务器 loopback。从个人电脑建立隧道：

```bash
ssh -N -L 8000:127.0.0.1:8000 your-user@your-server
```

打开 `http://127.0.0.1:8000/gpu.html`，输入服务器 `/state/credentials/admin.key` 文件中的个人凭据。凭据只在服务器上生成，本仓库不包含任何部署凭据或业务训练数据；合成示例语料仅供流程测试。

第一次先单卡跑通短训练和检查点恢复，再扩到八卡。完整的鉴权、备份、HTTPS、升级与故障处理见部署手册。升级前先备份；`git pull --ff-only` 只更新源码，镜像需重新构建后生效。不要执行 `docker compose down -v`，它会删除训练数据卷。

## 验证与状态

旧的在线 Sites 链接是独立静态预览，本次 v4 更新以此 GitHub 仓库为准。GPU 服务必须部署到 Linux 机器；自部署会同时提供当前前端和 API。当前是经过权限和运维加固的单机候选版，支持个人凭据、角色和项目 API 权限；未完成目标机器生产验收，不是多租户训练集群，不包含模型推理服务。

校验说明见 [VALIDATION.md](VALIDATION.md)。完整环境和部署限制见 DEPLOY.md。生产门禁、风险与恢复流程见 [PRODUCTION.md](PRODUCTION.md)。
