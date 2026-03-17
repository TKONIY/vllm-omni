# 离线加载（Offloader）子模块索引

## 概述

`offloader/` 子模块实现了 CPU 卸载（offloading）功能，使大型扩散模型能在有限 GPU 显存下运行。提供两种策略：模型级顺序卸载（DiT 与编码器互斥）和层级卸载（transformer block 滑动窗口）。

## 架构设计

```
offloader/
├── __init__.py              # 入口：工厂函数 get_offload_backend
├── base.py                  # 基础抽象：策略枚举、配置、后端基类
├── module_collector.py      # Pipeline 模块发现
├── layerwise_backend.py     # 层级卸载：异步预取、滑动窗口
└── sequential_backend.py    # 模型级卸载：DiT/编码器互斥
```

## 文件列表

| 文件 | 说明 | 文档链接 |
|------|------|----------|
| `__init__.py` | 工厂函数，根据配置创建卸载后端 | [__init__.md](./__init__.md) |
| `base.py` | 策略枚举、配置数据类、后端抽象基类 | [base.md](./base.md) |
| `module_collector.py` | Pipeline 模块发现工具 | [module_collector.md](./module_collector.md) |
| `layerwise_backend.py` | 层级卸载后端：异步 stream 预取、展平 CPU 存储 | [layerwise_backend.md](./layerwise_backend.md) |
| `sequential_backend.py` | 模型级卸载后端：DiT 与编码器互斥 GPU 占用 | [sequential_backend.md](./sequential_backend.md) |

## 卸载策略对比

| 策略 | 显存需求 | 切换频率 | 传输开销 | 适用场景 |
|------|----------|----------|----------|----------|
| MODEL_LEVEL | 能容纳 max(DiT, 编码器) | 每扩散步 1-2 次 | 中等 | 显存略不足 |
| LAYER_WISE | 能容纳 2-3 个 block + 编码器 | 每 block 1 次 | 较高（但异步重叠） | 显存严重不足 |

## 核心流程

1. **配置提取**：`OffloadConfig.from_od_config()` 从用户配置中确定策略。
2. **模块发现**：`ModuleDiscovery.discover()` 自动识别 DiT、编码器、VAE。
3. **后端创建**：`get_offload_backend()` 根据策略创建对应后端。
4. **启用卸载**：`backend.enable(pipeline)` 注册 hook、移动模块。
5. **自动运行**：推理过程中 hook 自动管理模块的 CPU/GPU 切换。
6. **清理**：`backend.disable()` 移除所有 hook。
