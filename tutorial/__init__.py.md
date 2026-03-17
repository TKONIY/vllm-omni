# `__init__.py` — 包入口与核心组件注册

## 文件概述

`__init__.py` 是 vllm-omni 包的入口文件，负责在导入时完成以下关键初始化工作：

1. 导入 `patch` 模块，触发对 vLLM 的猴子补丁
2. 注册自定义的 transformers 配置（AutoConfig、AutoTokenizer）
3. 导出核心公共 API：`Omni`、`AsyncOmni`、`OmniModelConfig`

## 关键代码解析

### 补丁导入（容错设计）

```python
try:
    from . import patch  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "vllm":
        raise
    patch = None
```

这段代码在包导入时自动执行 `patch.py`，完成对 vLLM 核心类的替换。如果 vLLM 未安装（如文档构建环境），则优雅跳过。

### 自定义配置注册

```python
from vllm_omni.transformers_utils import configs as _configs
```

尽早注册自定义的 HuggingFace 配置类，确保 `AutoConfig.from_pretrained()` 能识别 omni 模型。

### 公共 API 导出

```python
__all__ = [
    "__version__",
    "__version_tuple__",
    "Omni",
    "AsyncOmni",
    "OmniModelConfig",
]
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `Omni` | 类 | 同步推理引擎入口 |
| `AsyncOmni` | 类 | 异步推理引擎入口 |
| `OmniModelConfig` | 类 | 多阶段模型配置 |

## 与其他模块的关系

- 触发 `patch.py` 执行，完成 vLLM 类替换
- 导入 `config.OmniModelConfig` 作为配置入口
- 导入 `entrypoints.Omni` / `entrypoints.AsyncOmni` 作为用户 API
- 注册 `transformers_utils.configs` 自定义配置

## 总结

该文件是 vllm-omni 的"启动引导"，通过导入副作用完成补丁注入和配置注册，同时提供简洁的公共 API 供用户使用。
