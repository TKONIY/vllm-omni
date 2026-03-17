# `__init__.py` — 模块初始化文件

## 文件概述

该文件为空文件（仅包含一个空行），其作用是将 `qwen2_5_omni` 目录标记为 Python 包。

## 关键代码解析

文件内容为空，不包含任何导入或导出声明。各子模块通过完整路径直接导入使用。

## 核心类/函数

无。

## 与其他模块的关系

作为包标识文件，使得以下导入方式成为可能：
```python
from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker import ...
```

## 总结

标准 Python 包初始化文件，无实质内容。实际的模型注册和使用通过 vLLM 的模型注册表机制完成。
