# `__init__.py` — GLM-Image 模块入口

## 文件概述

导出 `GlmImageForConditionalGeneration` 类，使其可被 vLLM-omni 的模型注册系统加载。

## 关键代码解析

```python
from .glm_image_ar import GlmImageForConditionalGeneration
__all__ = ["GlmImageForConditionalGeneration"]
```

## 总结

简单的导出入口，实际实现在 `glm_image_ar.py` 中。
