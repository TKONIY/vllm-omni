# `serve.py` — serve 子命令实现

## 文件概述

实现了 `vllm serve --omni` 命令，是启动 vLLM-Omni HTTP 服务器的主要方式。该命令自动检测模型类型（LLM 多阶段管线或扩散模型），配置相应的服务参数，并启动 FastAPI 服务器。

## 关键代码解析

### 命令执行

```python
class OmniServeCommand(CLISubcommand):
    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if not os.environ.get("VLLM_DISABLE_LOG_LOGO"):
            os.environ["VLLM_DISABLE_LOG_LOGO"] = "1"
            log_logo()  # 显示 vLLM-Omni Logo

        if args.headless:
            run_headless(args)  # 已废弃
        else:
            uvloop.run(omni_run_server(args))
```

启动流程：显示 Logo -> 使用 uvloop 运行异步服务器。

### 参数验证

```python
def validate(self, args):
    if args.stage_id is not None and (args.omni_master_address is None or ...):
        raise ValueError("--stage-id 需要同时指定 master 地址和端口")

    # 自动检测扩散模型并跳过 vLLM 标准验证
    if model and is_diffusion_model(model):
        logger.info("检测到扩散模型: %s", model)
        return
    validate_parsed_serve_args(args)
```

### 丰富的命令行参数

该文件定义了大量的 CLI 参数，按功能分组：

**Omni 核心参数：**
- `--omni`: 启用 Omni 模式
- `--stage-configs-path`: 阶段配置文件路径
- `--stage-id`: 单阶段启动模式
- `--stage-init-timeout` / `--init-timeout`: 超时设置

**扩散模型参数：**
- `--num-gpus`: GPU 数量
- `--usp` / `--ring`: 序列并行度
- `--cache-backend`: 缓存优化后端
- `--vae-use-slicing` / `--vae-use-tiling`: VAE 内存优化

**TTS 参数：**
- `--task-type`: TTS 任务类型
- `--tts-max-instructions-length`: 指令最大长度

**分布式参数：**
- `--worker-backend`: 工作器后端（multi_process / ray）
- `--omni-master-address` / `--omni-master-port`: 编排器地址

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniServeCommand` | 类 | serve 子命令，实现 CLISubcommand 接口 |
| `cmd()` | 静态方法 | 执行服务启动 |
| `validate()` | 方法 | 验证命令行参数 |
| `subparser_init()` | 方法 | 注册所有 CLI 参数 |
| `run_headless()` | 函数 | 已废弃的无头模式 |
| `cmd_init()` | 函数 | 命令工厂方法 |

## 与其他模块的关系

- 调用 `openai/api_server.py` 的 `omni_run_server()` 启动服务器
- 调用 `logo.py` 的 `log_logo()` 显示启动 Logo
- 使用 vLLM 的 `make_arg_parser()` 继承标准 vLLM 参数
- 被 `main.py` 通过 `cmd_init()` 注册

## 总结

`serve.py` 是 vLLM-Omni 服务启动的完整入口，提供了覆盖多阶段 LLM、扩散模型、TTS 和分布式部署的全面 CLI 参数配置，自动检测模型类型并选择合适的服务模式。
