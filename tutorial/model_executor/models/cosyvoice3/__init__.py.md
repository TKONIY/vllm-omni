# `__init__.py` — CosyVoice3 模块入口

## 文件概述

CosyVoice3 模块的空初始化文件，仅包含 Apache-2.0 许可证声明。模型的注册和导出在其他文件中通过 `@MULTIMODAL_REGISTRY.register_processor` 装饰器完成。

## 总结

空入口文件，CosyVoice3 模型通过 vLLM-omni 的 `OmniModelRegistry` 直接引用 `cosyvoice3.CosyVoice3Model` 进行注册。
