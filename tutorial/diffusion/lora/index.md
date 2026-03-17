# LoRA 子模块索引

## 概述

`lora/` 子模块为扩散模型提供 LoRA（Low-Rank Adaptation）适配器支持。它复用 vLLM 的 LoRA 基础设施，并针对扩散模型推理场景进行了适配：使用 torch matmul 替代 punica_wrapper、支持 packed 投影层映射、提供 LRU 缓存管理。

## 架构设计

```
lora/
├── __init__.py          # 模块入口，导出 DiffusionLoRAManager
├── manager.py           # 核心管理器，负责适配器的加载、缓存和切换
├── utils.py             # 工具函数：模块匹配、名称扩展、层替换工厂
└── layers/              # LoRA 层类型实现
    ├── __init__.py
    ├── base_linear.py
    ├── column_parallel_linear.py
    ├── replicated_linear.py
    └── row_parallel_linear.py
```

## 文件列表

| 文件 | 说明 | 文档链接 |
|------|------|----------|
| `__init__.py` | 模块入口，导出 `DiffusionLoRAManager` | [__init__.md](./__init__.md) |
| `manager.py` | LoRA 适配器管理器：LRU 缓存、动态层替换、适配器激活 | [manager.md](./manager.md) |
| `utils.py` | 工具函数：目标模块匹配、packed 层扩展、层替换工厂 | [utils.md](./utils.md) |

## 核心流程

1. **初始化**：`DiffusionLoRAManager` 扫描 pipeline 中的线性层，计算支持的模块后缀和 packed 映射。
2. **加载适配器**：从 PEFT 检查点加载 LoRA 权重，创建 `LoRAModel`。
3. **层替换**：使用 `from_layer_diffusion` 将匹配的线性层替换为 LoRA 封装层。
4. **激活适配器**：将 LoRA 权重（含 scale）设置到各封装层中。
5. **缓存管理**：LRU 策略自动淘汰不常用的适配器。

## 子目录

- [layers/](./layers/index.md) — LoRA 层类型实现
