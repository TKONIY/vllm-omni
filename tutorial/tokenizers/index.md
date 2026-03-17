# tokenizers 模块索引

本模块提供 vllm-omni 的自定义分词器实现，当前主要包含 MammothModa2 模型的专用分词器。

## 模块结构

```
tokenizers/
├── __init__.py                      # 空初始化文件
└── mammoth_moda2_tokenizer.py       # MammothU 分词器实现
```

## 文档列表

| 文件 | 说明 |
|------|------|
| [__init__.md](__init__.md) | 包初始化 |
| [mammoth_moda2_tokenizer.md](mammoth_moda2_tokenizer.md) | MammothU 分词器 |

## 模块间关系

- `MammothUTokenizer` 是 MammothModa2 系列模型的专用分词器。
- 与 `transformers_utils/configs/mammoth_moda2.py` 中的配置类配合使用（配置中指定 `tokenizer_class = "MammothUTokenizer"`）。
- 基于 tiktoken 库实现 BPE 分词，兼容 HuggingFace Transformers 的 `PreTrainedTokenizer` 接口。
