# TrainLab 模型训练与学习平台

可视化学习实验 + 可自部署的单机 NVIDIA GPU 文本预训练平台。

- `/`：浏览器真实 MLP 小实验、训练曲线、数据集和知识库。
- `/gpu.html`：从随机初始化 Qwen3.5 / Qwen3 / GPT-2 开始的服务端训练。上传 JSONL、训练 BPE tokenizer、选择 1–8 张 GPU、监控、取消、恢复和下载模型。
- `server/`：FastAPI + SQLite + torchrun + Transformers Trainer。
- `Dockerfile` / `compose.yaml`：Linux GPU 部署；`deploy/` 可选 HTTPS 入口。

请按 [DEPLOY.md](DEPLOY.md) 检查机器、生成配置、启动和验收。目标环境是 Linux x86_64、Docker Engine / Compose v2、NVIDIA 驱动和 Container Toolkit。CUDA 13.0 兼容性需要在目标机器验证。

## v5：选择架构、训练后测试、ModelScope 数据

在 **GPU 预训练 → 模型架构** 选择 Qwen3.5 文本稠密混合注意力、Qwen3 稠密或兼容的 GPT-2。全部从随机权重开始，BPE 分词器由自己的训练集学习；不下载官方预训练权重，也不访问 Hugging Face 数据服务。层数、宽度、KV heads、MLP 大小可配置，缩小配置用于验证训练流程，不等同于官方模型规模或能力。

训练完成后，打开 **任务详情 → 模型测试**，选择最终模型或完整检查点：

- **文本续写**：输入提示词，设置生成长度、温度和 seed，查看真实输出及 token 数。
- **语料评估**：选择原验证集或另行上传/导入的测试语料，查看 Loss、困惑度、评估 token 数和数据摘要。
- 历史结果持久保存，可复用条件比较检查点、下载结果 JSON、取消排队或运行中的测试。训练与测试共用串行队列，避免同时争抢 GPU。

CPU 测试限权重 400 MiB 以内；GPU 测试单卡执行。测试限时 30–600 秒、最多生成 256 tokens 或评估 128 块。提示词与输出总长度不能超过训练上下文。Qwen3.5 当前采用 PyTorch 参考内核，尚未验证八卡性能；详见部署手册。

## 常用数据集（ModelScope）

在 **准备语料 → 选择数据集** 导入 50 / 200 / 500 篇小样本，完成后自动选中。已将原 Hugging Face 在线目录替换为以下数据源，已有本地语料仍可继续使用。

| 数据源 | 用途 | 数据卡许可 |
| --- | --- | --- |
| [中文预训练样本](https://modelscope.cn/datasets/BazingaLyn/mini_pretrain_dataset)（默认） | 中文文本预训练流程练习 | Apache License 2.0 |
| [MiniMind 预训练样本](https://modelscope.cn/datasets/gongjy/minimind_dataset) | 轻量预训练实验 | CC-BY-NC-4.0，非商业用途 |
| 合成流程示例 | 离线检查训练与恢复，不代表模型能力 | 随项目提供 |

服务器直接读取 ModelScope JSONL 文件的有限前缀，不下载完整 GB 级数据文件，不需要 ModelScope token。只保留完整文档，精确去重；最多读取 20 MiB，至少获得 20 篇才成功。顺序小样本不是代表性随机样本或官方测试集。

来源 JSON 保存仓库、文件、revision、ETag、下载前缀及文档 SHA256。上游 master 会变化，可复现实验请保留已导入语料和备份。使用与分发应遵守对应数据许可。网络要求见 [DEPLOY.md](DEPLOY.md)。

## 导航交互更新（2026-10-04）

学习页与 GPU 页使用一致的导航名称和顺序，按服务器训练、浏览器学习实验、学习资料分组。当前页重复点击不会重置或刷新；教学模块切换支持浏览器返回/前进，知识库章节可直接链接。微调练习参数在同一页面内切换后保留。手机端使用带文字的抽屉导航，支持 Esc、遮罩关闭和键盘焦点约束。

跨到 GPU 页面或离开 GPU 页面仍属于整页导航；GPU 凭据保存在当前标签页 sessionStorage，刷新或同标签页返回会验证并恢复连接；断开连接或凭据失效时清除，服务端训练不受影响。部署和运维资料在新标签页打开。

## v4 工作台更新

- 资源概览、磁盘水位、任务状态、任务名称/ID 搜索、状态筛选与服务端分页。
- JSONL 拖拽上传、进度与取消、前三篇语料预览和 SHA256 摘要。
- 自定义模型架构、独立评估/保存间隔、优化器参数、参数量和有效 batch 估计。
- 本机参数草稿、完整配置复用、提交/停止/恢复确认、重复提交保护。
- 指标曲线和数值表、日志跟随开关、配置与指标导出、模型文件列表；支持窄屏和键盘操作。

所有任务和资源数字来自 API。无 GPU 时可以准备数据和参数，不能提交训练。详见 [验证记录](VALIDATION.md)。

## 从 GitHub 镜像部署

GPU 镜像由 [GitHub Actions](https://github.com/chensl139-ok/trainlab-pretraining/actions/workflows/publish-image.yml) 构建并发布到 `ghcr.io/chensl139-ok/trainlab-pretraining`，目标架构为 **Linux amd64**。只有通过 API/权限/前端测试、镜像内 CPU 训练恢复检查及生产模式 API 启动检查的镜像才会推送。CI 没有 GPU，因此不代表 CUDA/NCCL 实机验收。

- `latest`：最近成功发布的主分支构建。
- `sha-<完整提交 SHA>`：对应源码提交的镜像标签；正式部署建议使用构建摘要中的 `@sha256:...` 固定镜像。
- 源码或 Dockerfile 更新会自动构建，也可以在 Actions 手动运行。发布使用短期 `GITHUB_TOKEN`，不需要把个人访问令牌提交到仓库。

首次部署，在已安装 NVIDIA Container Toolkit 的 Linux 服务器执行：

```bash
git clone https://github.com/chensl139-ok/trainlab-pretraining.git
cd trainlab-pretraining
python3 scripts/init_env.py
docker compose -f compose.image.yaml pull
docker compose -f compose.image.yaml run --rm trainlab python scripts/preflight.py
docker compose -f compose.image.yaml run --rm trainlab torchrun --standalone --nproc_per_node=8 scripts/ddp_check.py
docker compose -f compose.image.yaml run --rm trainlab python -m scripts.users issue --subject admin --project research --role admin --credential-file /state/credentials/admin.key
docker compose -f compose.image.yaml up -d
```

已有部署保留原 `.env`、项目目录及 `trainlab-state` 数据卷；不要重复签发同名凭据文件，也不要用 `down -v` 删除数据。升级前停机备份，再 `pull` 和 `up -d`。`compose.image.yaml` 与源码构建版的服务、卷名相同，请在同一项目目录执行并保持原 Compose 项目名。可以在 `.env` 设置 `TRAINLAB_IMAGE=ghcr.io/chensl139-ok/trainlab-pretraining@sha256:实际摘要`。

[GHCR 镜像包](https://github.com/chensl139-ok/trainlab-pretraining/pkgs/container/trainlab-pretraining) 已公开，已验证匿名读取镜像 manifest，无需 GitHub 登录即可拉取。首次镜像压缩层合计约 3.14 GB；服务器还需预留解压、镜像更新及训练数据空间。发布摘要与验证记录见 [VALIDATION.md](VALIDATION.md)。

## 在服务器从源码部署

这是公开仓库，服务器可直接通过 HTTPS 克隆，无需 GitHub 登录。训练服务仍需要个人凭据；公开源码不会开放服务器上的语料、模型或任务。

```bash
git clone https://github.com/chensl139-ok/trainlab-pretraining.git
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

旧的在线 Sites 链接是独立静态预览，当前更新以此 GitHub 仓库为准。GPU 服务必须部署到 Linux 机器；自部署会同时提供当前前端和 API。当前是经过权限和运维加固的单机候选版，支持个人凭据、角色和项目 API 权限；未完成目标机器生产验收，不是多租户训练集群，提供内部模型测试，不包含对外模型推理服务。

校验说明见 [VALIDATION.md](VALIDATION.md)。完整环境和部署限制见 DEPLOY.md。生产门禁、风险与恢复流程见 [PRODUCTION.md](PRODUCTION.md)。
