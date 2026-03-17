# `gguf_adapters/__init__.py` — GGUF 适配器入口与工厂

## 文件概述

`gguf_adapters/__init__.py` 提供了 GGUF 适配器的工厂函数 `get_gguf_adapter`，根据模型类型自动选择合适的 GGUF 权重名称映射适配器。

## 关键代码解析

```python
def get_gguf_adapter(gguf_file, model, source, od_config) -> GGUFAdapter:
    adapter_classes = (ZImageGGUFAdapter, Flux2KleinGGUFAdapter)
    for adapter_cls in adapter_classes:
        if adapter_cls.is_compatible(od_config, model, source):
            return adapter_cls(gguf_file, model, source, od_config)

    # 无匹配时抛出错误
    raise ValueError(
        f"No GGUF adapter matched diffusion model "
        f"(model_class_name={od_config.model_class_name!r}, ...)"
    )
```

工厂函数遍历已注册的适配器类，调用每个类的 `is_compatible()` 静态方法检查是否匹配当前模型。匹配成功则实例化并返回；所有适配器都不匹配时抛出错误。

新增模型的 GGUF 支持只需：
1. 创建适配器类继承 `GGUFAdapter`。
2. 实现 `is_compatible()` 和 `weights_iterator()`。
3. 在 `adapter_classes` 元组中注册。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `get_gguf_adapter` | 函数 | GGUF 适配器工厂，根据模型类型选择适配器 |
| `GGUFAdapter` | 类（重导出） | 适配器基类 |
| `Flux2KleinGGUFAdapter` | 类（重导出） | Flux2-Klein 模型适配器 |
| `ZImageGGUFAdapter` | 类（重导出） | Z-Image 模型适配器 |

## 与其他模块的关系

- **`base.py`**：导入 `GGUFAdapter` 基类。
- **`flux2_klein.py`**：导入 `Flux2KleinGGUFAdapter`。
- **`z_image.py`**：导入 `ZImageGGUFAdapter`。
- **`../diffusers_loader.py`**：`_get_gguf_weights_iterator` 调用 `get_gguf_adapter` 获取适配器。

## 总结

此入口文件通过工厂模式管理 GGUF 适配器的选择。模型兼容性检查由各适配器自行实现，工厂函数只负责匹配和实例化。
