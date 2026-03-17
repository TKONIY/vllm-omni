# `__init__.py` — MammothModa2 模块入口

## 文件概述

模块初始化文件，负责导入配置类、注册 tokenizer、并导出核心模型类。

## 关键代码解析

```python
from transformers import AutoTokenizer
from vllm_omni.tokenizers.mammoth_moda2_tokenizer import MammothUTokenizer
from vllm_omni.transformers_utils.configs.mammoth_moda2 import (
    Mammothmoda2Config, Mammothmoda2Qwen2_5_VLConfig,
)

# 注册 tokenizer：使 AutoTokenizer.from_pretrained 能正确加载 MammothUTokenizer
AutoTokenizer.register(config_class=Mammothmoda2Config, slow_tokenizer_class=MammothUTokenizer)
AutoTokenizer.register(config_class=Mammothmoda2Qwen2_5_VLConfig, slow_tokenizer_class=MammothUTokenizer)
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `MammothModa2ARForConditionalGeneration` | 类 | 导出的核心模型类 |

## 与其他模块的关系

- 导入 `vllm_omni.tokenizers.mammoth_moda2_tokenizer` 的自定义 tokenizer
- 导入 `vllm_omni.transformers_utils.configs.mammoth_moda2` 的配置类（触发 AutoConfig 注册）

## 总结

该文件的核心作用是通过 `AutoTokenizer.register` 将 MammothUTokenizer 与配置类关联，确保模型加载时能正确初始化 tokenizer。
