# distributed/ -- 扩散模型分布式推理模块索引

## 模块概述

`distributed/` 模块为 vllm-omni 的扩散模型提供完整的分布式推理支持，涵盖六种并行维度：

1. **数据并行 (DP)**: 多个 GPU 处理不同的输入样本
2. **CFG 并行**: 正向/负向预测分配到不同 GPU 并行计算
3. **序列并行 (SP)**: 将长序列分片到多个 GPU，支持 Ulysses 和 Ring 两种注意力策略
4. **流水线并行 (PP)**: 将模型层分配到不同 GPU，支持异步 P2P 通信
5. **张量并行 (TP)**: 将单层的参数分片到多个 GPU（复用 vLLM 实现）
6. **Fully Shard (FS/HSDP)**: 混合分片数据并行，将参数分片和副本结合

## 模块结构

```
distributed/
├── __init__.py                # 模块入口
├── cfg_parallel.py            # CFG 并行混入类
├── comm.py                    # 底层通信原语 (All-to-All, Ring)
├── group_coordinator.py       # 进程组协调器
├── hsdp.py                    # HSDP 混合分片
├── parallel_state.py          # 全局并行状态管理 (核心)
├── sp_plan.py                 # 序列并行计划类型定义
├── sp_sharding.py             # 序列并行分片工具
├── utils.py                   # 工具函数
├── vae_patch_parallel.py      # VAE 补丁/瓦片并行 (猴子补丁方式)
└── autoencoders/              # 分布式 VAE 自编码器 (继承方式)
    ├── __init__.py
    ├── autoencoder_kl.py           # 标准 AutoencoderKL 分布式版本
    ├── autoencoder_kl_qwenimage.py # Qwen 图像 VAE 分布式版本
    ├── autoencoder_kl_wan.py       # Wan 视频 VAE 分布式版本
    └── distributed_vae_executor.py # 分布式 VAE 执行框架
```

## 文件教程索引

### 核心文件

| 文件 | 说明 | 教程链接 |
|------|------|---------|
| `__init__.py` | 模块入口 | [__init__.md](__init__.md) |
| `parallel_state.py` | 全局并行状态管理 | [parallel_state.md](parallel_state.md) |
| `group_coordinator.py` | 进程组协调器 | [group_coordinator.md](group_coordinator.md) |
| `cfg_parallel.py` | CFG 并行支持 | [cfg_parallel.md](cfg_parallel.md) |

### 序列并行

| 文件 | 说明 | 教程链接 |
|------|------|---------|
| `sp_plan.py` | SP 配置与计划类型 | [sp_plan.md](sp_plan.md) |
| `sp_sharding.py` | SP 分片工具 | [sp_sharding.md](sp_sharding.md) |
| `comm.py` | 底层通信原语 | [comm.md](comm.md) |

### 模型分片

| 文件 | 说明 | 教程链接 |
|------|------|---------|
| `hsdp.py` | HSDP 混合分片 | [hsdp.md](hsdp.md) |

### VAE 分布式解码

| 文件 | 说明 | 教程链接 |
|------|------|---------|
| `vae_patch_parallel.py` | VAE 补丁/瓦片并行 | [vae_patch_parallel.md](vae_patch_parallel.md) |

### 分布式自编码器

| 文件 | 说明 | 教程链接 |
|------|------|---------|
| `autoencoders/__init__.py` | 子模块入口 | [autoencoders/__init__.md](autoencoders/__init__.md) |
| `autoencoders/distributed_vae_executor.py` | 执行框架 | [autoencoders/distributed_vae_executor.md](autoencoders/distributed_vae_executor.md) |
| `autoencoders/autoencoder_kl.py` | 标准 VAE | [autoencoders/autoencoder_kl.md](autoencoders/autoencoder_kl.md) |
| `autoencoders/autoencoder_kl_qwenimage.py` | Qwen VAE | [autoencoders/autoencoder_kl_qwenimage.md](autoencoders/autoencoder_kl_qwenimage.md) |
| `autoencoders/autoencoder_kl_wan.py` | Wan VAE | [autoencoders/autoencoder_kl_wan.md](autoencoders/autoencoder_kl_wan.md) |

### 工具

| 文件 | 说明 | 教程链接 |
|------|------|---------|
| `utils.py` | 设备获取工具 | [utils.md](utils.md) |

## 架构概览

```
initialize_model_parallel()
        │
        ├── _WORLD (GroupCoordinator)
        ├── _DP   (GroupCoordinator)         数据并行
        ├── _CFG  (GroupCoordinator)         CFG 并行
        │         └── CFGParallelMixin
        ├── _SP   (SequenceParallelGroupCoordinator)  序列并行
        │         ├── ulysses_group          All-to-All
        │         └── ring_group             Ring P2P
        ├── _PP   (PipelineGroupCoordinator) 流水线并行
        ├── _TP   (vLLM GroupCoordinator)    张量并行
        ├── _FS   (GroupCoordinator)         Fully Shard
        │         └── HSDP
        └── _DIT  (ProcessGroup)             DiT 全局组
                  └── DistributedVaeExecutor
```

## VAE 分布式解码的两种方式

| 方式 | 文件 | 特点 |
|------|------|------|
| 猴子补丁 | `vae_patch_parallel.py` | 不修改 VAE 类，运行时替换 decode 方法 |
| 继承 | `autoencoders/*.py` | 创建 VAE 子类，更灵活的 split/exec/merge 控制 |

## 序列并行的两种策略

| 策略 | 通信方式 | 适用场景 |
|------|---------|---------|
| Ulysses | All-to-All | 中等序列长度，带宽充足 |
| Ring | P2P 环形传递 | 超长序列，内存/带宽受限 |

vllm-omni 支持混合 Ulysses-Ring 策略：`sp_size = ulysses_degree * ring_degree`
