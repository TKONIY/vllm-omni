# inputs 模块教程 — 输入数据与预处理

## 模块概述

`inputs/` 模块定义了 vllm-omni 的输入数据类型和预处理逻辑。它扩展了 vLLM 的输入系统，添加了对 prompt 嵌入、附加信息、扩散模型采样参数等的支持。

## 架构图

```
inputs/
├── __init__.py        # 空文件
├── data.py            # 输入数据类型定义
└── preprocess.py      # 输入预处理器

用户输入 (str / list[int] / dict)
    │
    ▼
┌──────────────────────┐
│  类型识别              │
│  ├─ "prompt" → Text   │
│  ├─ "prompt_token_ids"│
│  │   → Tokens         │
│  └─ "prompt_embeds"   │
│      → Embeds         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────┐
│  OmniInputPreprocessor        │
│  ├─ _process_text()           │
│  │   → OmniTokenInputs       │
│  │   → MultiModalInputs      │
│  ├─ _process_tokens()         │
│  │   → OmniTokenInputs       │
│  │   → MultiModalInputs      │
│  └─ _process_embeds()         │
│      → EmbedsInputs           │
└──────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  OmniTokenInputs              │
│  ├─ prompt_token_ids          │
│  ├─ prompt_embeds (可选)      │
│  ├─ additional_information    │
│  │   (可选)                   │
│  └─ multi_modal_data (可选)   │
└──────────────────────────────┘
```

## 模块文档索引

| 文件 | 说明 |
|------|------|
| [data.py.md](./data.py.md) | 输入数据类型定义 |
| [preprocess.py.md](./preprocess.py.md) | 输入预处理器 |
