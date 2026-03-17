# `configs/__init__.py` — 延迟加载与自动注册

## 文件概述

该文件是 configs 子包的入口，实现了两个关键机制：
1. 通过 `__getattr__` 实现配置类的延迟加载
2. 通过即时导入子模块触发 `AutoConfig.register()` 副作用

## 关键代码解析

### 延迟加载映射

```python
_CLASS_TO_MODULE: dict[str, str] = {
    "Mammothmoda2Config": "vllm_omni.transformers_utils.configs.mammoth_moda2",
    "Mammothmoda2Qwen2_5_VLConfig": "vllm_omni.transformers_utils.configs.mammoth_moda2",
    "Mammothmoda2Qwen2_5_VLTextConfig": "vllm_omni.transformers_utils.configs.mammoth_moda2",
    "Mammothmoda2Qwen2_5_VLVisionConfig": "vllm_omni.transformers_utils.configs.mammoth_moda2",
    "FishSpeechConfig": "vllm_omni.transformers_utils.configs.fish_speech",
    "FishSpeechSlowARConfig": "vllm_omni.transformers_utils.configs.fish_speech",
    "FishSpeechFastARConfig": "vllm_omni.transformers_utils.configs.fish_speech",
}
```

将配置类名映射到其所在的模块路径。

### 延迟导入

```python
def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)
    raise AttributeError(...)
```

当访问 `configs.Mammothmoda2Config` 时，才真正导入对应模块。

### 即时注册

```python
from vllm_omni.transformers_utils.configs import fish_speech as _fish_speech
from vllm_omni.transformers_utils.configs import mammoth_moda2 as _mammoth_moda2
```

同时在模块底部即时导入所有子模块，确保 `AutoConfig.register()` 的副作用立即生效。这看似与延迟加载矛盾，实际上是为了确保 `AutoConfig.from_pretrained()` 能正确识别自定义模型类型。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `_CLASS_TO_MODULE` | 字典 | 类名到模块路径的映射 |
| `__getattr__(name)` | 函数 | 模块级 `__getattr__`，实现延迟加载 |
| `__dir__()` | 函数 | 返回可用的配置类名列表 |

## 与其他模块的关系

- **组织 fish_speech 和 mammoth_moda2 的配置**: 统一入口。
- **触发 AutoConfig 注册**: 导入本包即可让 `AutoConfig.from_pretrained()` 识别自定义模型。

## 总结

该文件巧妙地结合了延迟加载和即时注册两种策略：延迟加载减少不必要的导入开销，即时注册确保 AutoConfig 的全局注册表包含所有自定义配置类。
