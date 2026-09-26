# DeepSeek-V4.1-Flash on A100

本项目 fork 自 [shi3z/deepseekv4.1-A100-custom](https://github.com/shi3z/deepseekv4.1-A100-custom)，结合本地部署的需求对原项目做了一些扩展。

## 主要扩展

- **模型与网关分离**：新增常驻模型 worker，将模型加载和推理与 HTTP 网关分开。修改接口或输出处理后，可单独重启网关并复用已加载的模型，减少重载权重的等待。
- **OpenAI 兼容接口改进**：在原有聊天、补全及流式接口基础上，完善思考内容和工具调用解析，修复字符输出、结束时的尾部输出及 token 用量统计，并补充请求参数校验。
- **图文推理修复**：在原有图片输入能力基础上，修复长图文输入分块处理中的问题，完善图片请求在 worker 和 DSpark 推理路径中的处理。
- **DSpark/MTP 与批处理优化**：完善多请求投机解码和请求状态管理，支持客户端断开后取消批处理请求。同时调整预填充调度，缓解新请求加入时已有请求生成停顿的问题。
- **五卡专家分片**：批处理配方将部分路由专家的权重与计算迁移到第五张卡，与 DSpark 和视觉塔共同使用该卡，为四张主干卡上的并行请求缓存腾出显存。
- **索引与前缀缓存改进**：修复多卡推理中的索引缓存问题，减少 MTP 验证时的重复缓存复制。在原有前缀缓存基础上扩展图文前缀复用，完善持久化、容量限制和过期清理，减少重复输入的预填充开销。
- **监控与管理改进**：完善请求文本、图片预览、生成内容和预填充进度的展示，补充吞吐、GPU 状态及 MTP 接受情况等指标，并提供 worker 状态查询和 DSpark 配置重载入口。
- **部署与回归测试**：提供 Docker 开发容器配置，整理依赖安装、单序列与批处理启动、状态检查和停止脚本。补充接口输出、批处理、缓存、图文推理及网关通信等路径的回归测试。

## 部署

参考部署使用五张 A100 80GB，主机内存需容纳约 189 GiB Engram 表并留出运行余量。环境为 Ubuntu 24.04、CUDA 12.8 和匹配的 NVIDIA 驱动。

语言主干使用四张卡，第五张卡承载 DSpark 与视觉塔。批处理入口还将部分路由专家放到第五张卡，为主干卡上的并行请求缓存腾出显存；单序列入口保留四卡专家分片。GPU 排布与分片参数应根据目标机器调整。

两个入口的语言主干顺序均为 `2,0,1,3`，每卡 10 层；DSpark 与视觉塔位于 `cuda:4`。专家采用 EP 分片，分片卡可以与主干卡不同。默认布局如下：

| 入口 | 专家 GPU 顺序 | 每层专家分片数 | GPU4 职责 |
|---|---|---|---|
| `run_batch_local.sh` | `2,0,1,3,4` | `76,76,84,84,64` | 64 个路由专家、DSpark、视觉塔 |
| `run_mtp_local.sh` | `2,0,1,3` | `92,99,98,95` | DSpark、视觉塔 |

### Docker 开发容器

仓库附带 [compose.yaml](compose.yaml)，基于 `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel`。宿主机需先准备 NVIDIA 驱动、Docker Compose 与 NVIDIA Container Toolkit。

模型权重需自行下载，可从 HuggingFace 或 ModelScope 官方仓库获取。模型目录名为 `DeepSeek-V4.1-Flash`，默认放在宿主机 `/var/models/DeepSeek-V4.1-Flash`，挂载后容器内路径为 `/models/DeepSeek-V4.1-Flash`。

Compose 启动的是 Bash 开发容器，**需进入容器安装依赖并启动服务**。重建容器后需重新安装依赖，挂载目录中的文件会保留。

在宿主机的仓库根目录执行：

```bash
# 默认将宿主机 /var/models 挂到 /models，将当前仓库挂到 /workspace。
# 若路径不同，可先 export DSV41_MODELS_DIR=/your/model-directory
# 工作区也可用 DSV41_WORKSPACE_DIR 覆盖。
docker compose config
docker compose up -d
docker compose exec deepseek bash
```

后续命令均在容器内的仓库根目录执行，默认位置为 `/workspace`。若自定义了工作区挂载路径，请进入实际的仓库目录。

### 容器内依赖与配置

进入容器后，以 root 安装系统与 Python 依赖：

```bash
cd /workspace
./setup_mtp_local.sh
```

若依赖已准备好，可跳过 setup。

如需调整模型路径、监听地址、专家分片或 DSpark 配置，可分别设置 `DSV41_CKPT`、`DSV41_HOST`、`DSV41_PORT`、`DSV41_EP_DEVICES`、`DSV41_EP_SHARDS`、`DSV41_DSPARK_CONFIG`。

## 服务启停

完成上面的依赖与配置准备后，选择一个入口启动：

```bash
# 批处理入口。
./run_batch_local.sh --background

# 如需单序列，改用下面的入口。
# ./run_mtp_local.sh --background
```

省略 `--background` 可在前台运行。两个入口选择其一；切换时先停止现有服务。启动脚本会检查端口占用及 worker 配置兼容性。

服务默认监听 `host.docker.internal:40033`。若该地址无法绑定，可通过 `DSV41_HOST` 指定地址，例如 `127.0.0.1`。其他容器配置 `host.docker.internal=host-gateway` 映射后，可通过默认地址访问服务。

接口包括 `/health`、`/v1/models`、`/v1/chat/completions`、`/v1/completions` 和 `/dashboard`。请求中的 `model` 应使用 `/v1/models` 返回的名称。

查看状态并停止服务：

```bash
./stop_mtp_local.sh --status
./stop_mtp_local.sh
```

有活跃请求时停止脚本会拒绝停止，`--force` 可强制中断。日志位于 `logs/latest/`，图文缓存位于 `cache/image-prefix/`。

仅修改网关时，可在请求结束后终止 `logs/server.pid` 对应的进程，再执行启动脚本复用模型 worker。修改推理或 worker 代码后需重启整个服务。

## 使用说明

当前服务面向可信环境中的本机部署与调试，对外开放前需自行配置访问控制。监控页面和指标接口未设置鉴权，会展示请求文本、图片预览及生成内容；部分错误响应包含 traceback。图片输入支持本地路径和 HTTP/HTTPS URL，服务会以自身权限读取文件或访问指定地址。

本 fork 的代码修改、测试与文档是在 AI 辅助下编写和整理的，可能存在错误、遗漏或未经充分验证的行为。已有测试只覆盖特定条件，不构成对正确性、稳定性或适用于任何部署环境的保证。使用前请根据自己的硬件、模型和负载进行审查与验证。
