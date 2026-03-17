# GGUF 适配器子模块索引

## 概述

`gguf_adapters/` 子目录实现了 GGUF 格式权重的名称映射适配器。不同模型的 GGUF 检查点使用不同的键名约定，适配器负责将这些键名映射到 Diffusers/HuggingFace 格式的参数名。

## 架构设计

```
gguf_adapters/
├── __init__.py        # 工厂函数 get_gguf_adapter
├── base.py            # 适配器基类 + 通用 GGUF 权重迭代器
├── flux2_klein.py     # Flux2-Klein 模型适配器
└── z_image.py         # Z-Image 模型适配器
```

## 文件列表

| 文件 | 说明 | 文档链接 |
|------|------|----------|
| `__init__.py` | 适配器工厂函数，根据模型类型自动匹配 | [__init__.md](./__init__.md) |
| `base.py` | 适配器基类、GGUF 量化权重迭代器 | [base.md](./base.md) |
| `flux2_klein.py` | Flux2-Klein 适配器：双流/单流 block 映射、adaLN 交换 | [flux2_klein.md](./flux2_klein.md) |
| `z_image.py` | Z-Image 适配器：注意力层映射、前缀移除 | [z_image.md](./z_image.md) |

## 适配器对比

| 适配器 | 目标模型 | 映射复杂度 | 特殊处理 |
|--------|----------|------------|----------|
| `Flux2KleinGGUFAdapter` | Flux2 系列 | 高（4 组映射字典） | adaLN shift/scale 交换 |
| `ZImageGGUFAdapter` | Z-Image 系列 | 低（1 组映射字典） | 无 |

## 新增适配器指南

1. 在本目录创建新文件（如 `my_model.py`）。
2. 定义适配器类继承 `GGUFAdapter`：
   - 实现 `is_compatible()` 静态方法进行模型匹配。
   - 实现 `weights_iterator()` 方法进行键名映射。
3. 在 `__init__.py` 的 `adapter_classes` 元组中注册新类。
