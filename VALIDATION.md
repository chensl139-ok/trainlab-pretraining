# 验证记录

2026-10-03，macOS ARM64 / Python 3.12。目标为用户确认的 Linux + 8 张 RTX PRO 6000，第一阶段仅本人使用，尚未连接目标服务器。本版是经过加固的候选版，不能宣称达到字节内部标准或完成生产验收。

## 本次执行

- 22 项 pytest 通过，覆盖旧版训练控制流程，以及项目权限、只读角色、凭据过期/吊销、管理员恢复保留原项目、审计脱敏、队列限额、磁盘拒绝、请求大小限制、子进程环境隔离、真实进程超时终止、停机备份/恢复、损坏与在线备份拒绝。
- PyTorch 2.13.0 + Transformers 5.18.0 + Accelerate 1.15.0，离线 CPU 随机初始化两层 GPT（470,528 参数）训练 10 步，从 checkpoint-5 恢复到 10 步，最终 safetensors 逐项完全相同。两次验证 loss 均为 5.289914608001709。此数字仅证明流程，不代表语言能力。
- smoke 脚本改为每次使用全新目录，并自动断言最终权重相同，避免旧产物干扰。
- pip-audit：API 已解析依赖 15 包、已安装 Python 验证环境 51 包，均为 0 条已知漏洞报告；原始 JSON 和环境版本见 reports。该范围不包括 Linux/CUDA 系统库。
- FastAPI 0.142.2 / Starlette 1.7.0 回归通过；pip check 无冲突。测试有 httpx TestClient 弃用警告，不影响测试通过。
- Docker Compose 基础与 HTTPS 合并配置解析通过；官方 PyTorch 2.13.0 / CUDA 13.0 镜像 manifest 可读取。没有 Docker daemon，未实际构建/运行 Linux 镜像。
- 浏览器连接本地 API，身份、调度状态与当前机器无 NVIDIA GPU 的提示来自真实后端。不会填充假 GPU、假吞吐或假 loss。

## 仍需验收

实际每卡显存/驱动/拓扑、逐卡 BF16 forward/backward、8 卡 NCCL、训练吞吐及显存峰值、长时间稳定性、OOM/磁盘/容器重启演练、真实数据恢复时间、目标镜像扫描、域名 HTTPS 和告警演练均未执行。CPU 测试不能替代这些证据。

个人使用无需先引入企业 SSO、多人配额或多机 HA；扩大使用范围前再按 PRODUCTION.md 执行相应门禁。当前只有 API 项目隔离，没有 OS 租户隔离、海量语料流水线、独立模型能力评测或推理 API。
