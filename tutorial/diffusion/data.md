# `data.py` — 核心数据结构与配置

## 文件概述

`data.py` 是 diffusion 模块中最重要的数据定义文件，定义了扩散模型运行所需的所有配置类和数据结构。包括并行配置（`DiffusionParallelConfig`）、Transformer 配置容器（`TransformerConfig`）、缓存配置（`DiffusionCacheConfig`）、全局扩散配置（`OmniDiffusionConfig`）、输出结构（`DiffusionOutput`）以及注意力后端枚举（`AttentionBackendEnum`）。

## 关键代码解析

### DiffusionParallelConfig — 并行策略配置

```python
@config
@dataclass
class DiffusionParallelConfig:
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    sequence_parallel_size: int | None = None
    ulysses_degree: int = 1
    ring_degree: int = 1
    cfg_parallel_size: int = 1
    vae_patch_parallel_size: int = 1
    use_hsdp: bool = False
    hsdp_shard_size: int = -1
    hsdp_replicate_size: int = 1
```

该类涵盖了所有并行维度：流水线并行、数据并行、张量并行、序列并行（Ulysses + Ring）、CFG 并行、VAE Patch 并行以及 HSDP（混合分片数据并行）。`__post_init__` 中会自动计算 `world_size`。

### OmniDiffusionConfig — 全局扩散配置

这是最核心的配置类，包含以下关键字段分组：

| 分组 | 关键字段 | 说明 |
|------|----------|------|
| 模型路径 | `model`, `model_class_name`, `dtype` | 模型标识与精度 |
| 并行配置 | `parallel_config` | `DiffusionParallelConfig` 实例 |
| 缓存策略 | `cache_backend`, `cache_config` | TeaCache/cache-dit 等缓存加速 |
| 分布式 | `distributed_executor_backend`, `master_port` | 执行器后端与通信端口 |
| 内存优化 | `enable_cpu_offload`, `enable_layerwise_offload`, `vae_use_tiling` | CPU 卸载与 VAE 优化 |
| LoRA | `lora_path`, `lora_scale`, `max_cpu_loras` | LoRA 适配器配置 |
| 量化 | `quantization`, `quantization_config` | FP8 等量化支持 |
| 编译 | `enforce_eager` | 是否强制 eager 模式 |

`__post_init__` 中执行大量的类型转换和验证逻辑，例如将字符串 dtype 转为 `torch.dtype`，将字典转为 `DiffusionCacheConfig` 实例，以及自动寻找可用端口。

### DiffusionCacheConfig — 缓存加速配置

```python
@dataclass
class DiffusionCacheConfig:
    rel_l1_thresh: float = 0.2           # TeaCache 阈值
    Fn_compute_blocks: int = 1           # cache-dit 前向计算块数
    max_warmup_steps: int = 4            # 预热步数
    residual_diff_threshold: float = 0.24 # 残差差分阈值
    max_continuous_cached_steps: int = 3  # 最大连续缓存步数
    enable_taylorseer: bool = False       # TaylorSeer 开关
```

支持 TeaCache 和 cache-dit 两种缓存加速策略，通过 `from_dict` 类方法从字典创建，并通过 `__getattr__` 支持访问额外参数。

### DiffusionOutput — 输出结构

```python
@dataclass
class DiffusionOutput:
    output: torch.Tensor | None = None
    trajectory_timesteps: list[torch.Tensor] | None = None
    trajectory_latents: torch.Tensor | None = None
    error: str | None = None
    post_process_func: Callable[..., Any] | None = None
    custom_output: dict[str, Any] = field(default_factory=dict)
```

封装扩散模型的推理输出，包含生成结果、轨迹潜码、错误信息等。

### AttentionBackendEnum — 注意力后端枚举

```python
class AttentionBackendEnum(enum.Enum):
    FA = enum.auto()
    SLIDING_TILE_ATTN = enum.auto()
    TORCH_SDPA = enum.auto()
    SAGE_ATTN = enum.auto()
    VIDEO_SPARSE_ATTN = enum.auto()
    # ...
```

枚举所有支持的注意力后端实现。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionParallelConfig` | dataclass | 分布式并行策略配置 |
| `TransformerConfig` | dataclass | Transformer 模型配置容器，支持字典式访问 |
| `DiffusionCacheConfig` | dataclass | 缓存加速（TeaCache/cache-dit）配置 |
| `OmniDiffusionConfig` | dataclass | 全局扩散配置，是所有组件的核心配置来源 |
| `DiffusionOutput` | dataclass | 扩散推理输出数据结构 |
| `AttentionBackendEnum` | Enum | 注意力后端类型枚举 |
| `SHUTDOWN_MESSAGE` | dict | 用于通知 worker 关闭的特殊消息 |

## 与其他模块的关系

- `OmniDiffusionConfig` 被几乎所有模块使用：`diffusion_engine.py`、`registry.py`、`scheduler.py`、`worker/` 等。
- `DiffusionOutput` 是推理结果的标准格式，被 `worker/`、`ipc.py`、`scheduler.py` 等传递。
- `DiffusionParallelConfig` 被 `worker/diffusion_worker.py` 用于初始化分布式并行环境。
- `AttentionBackendEnum` 被注意力层选择器使用。

## 总结

`data.py` 是整个 diffusion 模块的数据基础，定义了配置体系和核心数据结构。`OmniDiffusionConfig` 作为中心配置对象，将模型、并行、缓存、量化、LoRA 等所有维度的配置统一管理。理解该文件是理解整个 diffusion 模块的关键前提。
