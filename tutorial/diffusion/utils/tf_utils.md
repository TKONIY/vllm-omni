# `tf_utils.py` — Transformer 配置工具

## 文件概述

`tf_utils.py` 提供了从 `TransformerConfig` 中提取模型初始化参数的工具函数。它通过 `inspect.signature` 动态检测模型 `__init__` 方法接受的参数，自动过滤不相关的配置项。

## 关键代码解析

### get_transformer_config_kwargs

```python
def get_transformer_config_kwargs(
    tf_model_config: TransformerConfig, model_class: type[Any] | None = None
) -> dict[str, Any]:
    # 1. 从 TransformerConfig 提取所有参数
    tf_config_params = tf_model_config.to_dict()

    # 2. 过滤 diffusers 内部元数据（以 '_' 开头的键）
    filtered_params = {k: v for k, v in tf_config_params.items() if not k.startswith("_")}

    # 3. 如果提供了 model_class，通过签名检查过滤不接受的参数
    if model_class is not None:
        sig = inspect.signature(model_class.__init__)
        accepted_params = {
            name for name, param in sig.parameters.items()
            if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
        }
        filtered_params = {k: v for k, v in filtered_params.items() if k in accepted_params}

    return filtered_params
```

过滤步骤：
1. 移除 diffusers 内部元数据键（如 `_class_name`、`_diffusers_version`）
2. 可选地通过模型类的 `__init__` 签名过滤，仅保留模型实际接受的参数
3. 排除 `**kwargs` 类型的参数（`VAR_KEYWORD`），只匹配显式定义的参数

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `get_transformer_config_kwargs` | 函数 | 从 TransformerConfig 提取模型接受的初始化参数 |

## 与其他模块的关系

- 使用 `data.py` 中的 `TransformerConfig` 数据结构
- 被扩散模型加载器（`model_loader/`）调用，将 HuggingFace 配置转换为模型构造参数

## 总结

`tf_utils.py` 解决了 HuggingFace 配置与模型构造函数之间的参数匹配问题。通过运行时签名检查自动过滤不兼容的参数，避免了传递未知参数导致的 `TypeError`，同时保持了向后兼容性。
