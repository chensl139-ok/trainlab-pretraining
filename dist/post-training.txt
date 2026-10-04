# 后训练：ModelScope 基座 → SFT → DPO → 模型测试

平台支持从零预训练，以及在现有权重上做全参数 SFT、DPO。后训练使用现有队列、维护模式、超时、取消、完整检查点恢复和项目权限。未接入 LoRA/QLoRA、PPO/GRPO、多机或对外推理服务。

## 从 ModelScope 导入

第一批适配 Qwen3 0.6B / 1.7B 的普通稠密 safetensors 模型，包括 `-Base` 版本；不接受 MoE、量化权重或自定义远程代码。建议先用 [Qwen3-0.6B-Base](https://modelscope.cn/models/Qwen/Qwen3-0.6B-Base) 跑通 SFT，再将 SFT 的最终模型作为 DPO 基础模型。

在 Linux 主机的独立下载环境安装 ModelScope 官方 CLI，不需要在训练服务中联网，也不读取 Hugging Face 数据源。下载命令依据 [ModelScope CLI 官方文档](https://github.com/modelscope/modelscope/blob/master/docs/source/command.md)。示例使用 master；实际交付应替换为所选模型的固定 revision，并保留下载记录。

```bash
python3 -m venv .modelscope-download
.modelscope-download/bin/pip install modelscope
.modelscope-download/bin/modelscope download \
  --model Qwen/Qwen3-0.6B-Base --revision master \
  --local_dir ./models/Qwen3-0.6B-Base
```

在部署目录运行导入。`--project` 必须与个人凭据所属项目一致；下例为 `default`。Compose 使用已有 state 卷，并将下载目录只读挂载。导入会复制模型到持久卷，预留原下载目录和持久副本两份磁盘空间。使用源码版部署时将 `compose.image.yaml` 替换为 `compose.yaml`。

```bash
docker compose -f compose.image.yaml run --rm --no-deps \
  -v "$PWD/models:/imports:ro" trainlab \
  python -m scripts.import_model \
  --path /imports/Qwen3-0.6B-Base \
  --source Qwen/Qwen3-0.6B-Base --revision master \
  --project default --context 1024
```

导入校验 config、safetensors 张量名称/形状、分片索引与分词器；注册内容 SHA256 和下载来源标识，不执行下载目录中的 Python 文件。SHA256 是完整性记录，并不证明来源声明或替代模型许可证审核。刷新页面，在“训练阶段”选择 SFT / DPO，再从“基础模型”选择导入结果。也可选择同项目已有任务的 final 或完整 checkpoint。训练时重新校验基础模型摘要。

原模型结构和分词器保持不变。导入时配置本次平台训练使用的上下文范围（64–2048）；后训练任务继承该范围。Qwen3 原生 head_dim 等参数从原始 config 加载，不通过页面中的缩小模型配置重建。参数量由实际权重张量统计。

## 数据格式

UTF-8 JSONL，每行一个完整样本，至少 20 行，最多 100 MiB / 10 万行。同文件不允许混合格式。问答支持单轮 user → assistant，当前未提供多轮 messages 数据格式。

SFT：

```json
{"prompt":"简述梯度累积的作用。","response":"将多个 micro batch 的梯度累积后再更新参数，增加有效 batch。"}
```

DPO：

```json
{"prompt":"训练损失下降是否代表模型能上线？","chosen":"还需要独立评估集、业务质量、安全性及稳定性验收。","rejected":"只要训练损失下降就可以直接上线。"}
```

页面可下载格式示例；示例为合成算术小数据，只用于验证流程，不是正式训练集。现有 ModelScope 常用文本目录用于预训练；后训练请上传自己的问答或偏好语料。

同一标准化 prompt 的所有回答被哈希分到同一集合，约 90% 训练 / 10% 验证；精确去重。至少保留两个不同训练/验证样本，训练样本需覆盖每卡 batch。无法排除语义重复、预训练污染或业务数据泄漏，正式效果需要独立测试集。

有原生 chat template 时使用该模板并关闭 thinking 提示；否则使用固定 `### User` / `### Assistant` 模板。训练与测试使用同一格式。仅回答和 EOS 参与 SFT 损失，提示词和 padding 被屏蔽。不拼接不同问答，不静默截断超长样本，超长时在日志中给出行号。

## 训练和测试口径

- **SFT**：载入基础模型和原分词器，更新全部参数；训练/验证 Loss 为回答部分的交叉熵。
- **DPO**：策略模型与冻结参考模型初始均来自所选基础版本；仅回答部分的总 log 概率进入标准 sigmoid DPO 损失，β 可配置，关闭 dropout。参考模型不会随策略更新；恢复时保持同一参考摘要。[DPO 算法说明](https://huggingface.co/docs/trl/dpo_trainer)。
- **指令回答**：训练完成后选择模型版本，输入用户问题，自动应用相同模板。当前为单轮测试。
- **语料评估**：SFT 对回答评分；DPO 对 chosen 回答评分并展示偏好命中率（chosen 总 log 概率高于 rejected 的比例）。偏好命中率受回答长度影响，不是人工胜率。这里的回答 CE / PPL 与训练期 DPO loss 不是同一指标，DPO loss 不换算为困惑度。

全参数 DDP 不会将 8 张卡显存合并，每卡保存完整模型及优化器。DPO 还在每卡保存一份冻结参考模型，并处理两个回答。0.6B / 1.7B 是当前导入范围，不代表任何配置已保证适配显存。先用单卡、micro batch 1、短上下文与少量步数测量峰值，再扩展到 8 卡。

## 验证与升级

代码内有真实 CPU smoke：`python -m scripts.smoke_train --architecture qwen3 --post-training`。覆盖本地小模型导入校验、预训练 → SFT → DPO、权重更新、基础模型不变、恢复后的权重逐项一致，以及回答生成和仅回答计分。该测试不会冒称已训练官方 ModelScope 权重或通过 8 卡验收。

state 备份包含 `models/`、数据、训练任务与模型测试。升级前按运维文档进入维护模式，等待当前工作结束，停止服务并备份，再拉取新镜像。不要在训练运行中覆盖或删除基础模型目录。
