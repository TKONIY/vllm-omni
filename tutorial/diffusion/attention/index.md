# attention/ — 扩散模型注意力模块

## 模块概述

`attention/` 模块是 vllm-omni 扩散模型推理引擎的核心注意力子系统。它采用分层架构，将**注意力计算内核**（后端选择）与**并行通信策略**（序列并行）解耦，支持多种硬件平台（CUDA / ROCm / NPU / XPU）和多种并行方式（Ulysses / Ring / 混合模式）。

## 架构设计

```
attention/
├── layer.py          # 核心 Attention 层（三阶段流水线入口）
├── selector.py       # 后端选择器（平台委托 + 环境变量覆盖）
├── backends/         # 注意力计算后端（Flash Attn / SDPA / Sage 等）
│   ├── abstract.py   #   抽象基类与元数据定义
│   ├── flash_attn.py #   Flash Attention 后端
│   ├── sdpa.py       #   PyTorch SDPA 后端
│   ├── sage_attn.py  #   Sage Attention 后端
│   ├── registry.py   #   后端注册表（枚举 + 覆盖）
│   ├── ring_flash_attn.py    # Ring Flash Attention 实现
│   ├── ring_pytorch_attn.py  # Ring PyTorch Attention 实现
│   ├── ring/         #   Ring Attention 底层组件
│   │   ├── ring_globals.py   #     全局依赖检测
│   │   ├── ring_kernels.py   #     计算内核集合
│   │   ├── ring_selector.py  #     内核选择器
│   │   └── ring_utils.py     #     分块结果合并工具
│   └── utils/        #   工具函数
│       └── fa.py     #     Flash Attention 导入管理 & unpad/pad 工具
└── parallel/         # 并行注意力策略
    ├── base.py       #   策略接口 & 默认无并行策略
    ├── factory.py    #   策略工厂
    ├── ring.py       #   Ring 并行策略
    └── ulysses.py    #   Ulysses 并行策略
```

## 核心设计原则

1. **内核与并行解耦**：`backends/` 负责"如何计算注意力"，`parallel/` 负责"如何在设备间通信"
2. **策略模式**：`Attention.forward()` 通过 `ParallelAttentionStrategy` 接口调用并行策略，新增并行方式无需修改核心类
3. **三阶段流水线**：`pre_attention`（通信）→ 内核执行（计算）→ `post_attention`（逆向通信）
4. **平台委托**：后端选择委托给平台层，支持多硬件透明切换
5. **优雅降级**：float32 自动回退 SDPA、SP 非活跃时自动禁用并行、FA 不可用时回退 SDPA

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](./__init__.md) | 包初始化（仅许可证声明） |
| [`layer.py`](./layer.md) | 核心 `Attention` 层，三阶段流水线入口 |
| [`selector.py`](./selector.md) | 后端选择器，支持环境变量覆盖 |

### 子目录

| 目录 | 说明 |
|------|------|
| [`backends/`](./backends/index.md) | 注意力计算后端集合 |
| [`parallel/`](./parallel/index.md) | 并行注意力策略集合 |

## 数据流

```
用户调用 Attention.forward(Q, K, V, metadata)
  ↓
parallel.pre_attention: AllToAll / 联合拼接 / 直通
  ↓
内核执行: Flash Attn / SDPA / Sage / Ring Attention
  ↓
parallel.post_attention: 逆 AllToAll / AllGather / 直通
  ↓
返回注意力输出
```

## 关键依赖

- `vllm_omni.platforms`：平台检测与后端选择
- `vllm_omni.diffusion.distributed`：分布式通信原语（RingComm、SeqAllToAll4D 等）
- `vllm_omni.diffusion.forward_context`：前向上下文管理
