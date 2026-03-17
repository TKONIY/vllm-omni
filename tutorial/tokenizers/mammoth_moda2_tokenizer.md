# `mammoth_moda2_tokenizer.py` — MammothU 分词器

## 文件概述

该文件实现了 `MammothUTokenizer` 类，是 MammothModa2 系列多模态模型的专用分词器。基于 `tiktoken` 库的 BPE（Byte Pair Encoding）分词算法，继承 HuggingFace `PreTrainedTokenizer` 接口，支持丰富的特殊 token 体系，包括视觉 token、生成 token 等多模态相关标记。

## 关键代码解析

### 词汇表结构

```python
VOCAB_FILES_NAMES = {
    "vocab_file": "mammothu.tiktoken",          # BPE 合并规则文件
    "special_tokens_file": "mammothu_vision_tokens.txt",  # 视觉特殊 token
}

SPECIAL_START_ID = 151643  # 特殊 token 起始 ID
```

词汇表由三部分组成：
1. **基础 BPE 词汇**: 从 `.tiktoken` 文件加载的合并规则
2. **Qwen 特殊 token**: 18 个预定义 token（vision、tool_call、fim 等）
3. **扩展 token**: EXTRAS 列表（对齐 Qwen2.5 词汇表大小 + 生成 token）

### 特殊 token 体系

```python
QWEN_SPECIAL_TOKENS = (
    "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>",
    "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
    "<|image_pad|>", "<|video_pad|>",
    "<tool_call>", "</tool_call>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
    # ...
)

# 对齐 Qwen2.5 词汇表
EXTRAS = [f"<|extra_{i}|>" for i in range(181)]
EXTRAS += [f"<|extra_margin_{i}|>" for i in range(152064 - 151846)]
EXTRAS += ["<|endofline|>", "<|endoffile|>", "<|gen_placeholder|>", ...]
```

### 初始化流程

```python
class MammothUTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, special_tokens_file, ...):
        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)

        # 加载视觉特殊 token
        with open(special_tokens_file) as f:
            vision_tokens = [t.strip() for t in f.readlines()]

        # 构建完整特殊 token 映射
        SPECIAL_TOKENS = tuple(enumerate(
            (ENDOFTEXT, IMSTART, IMEND) + QWEN_SPECIAL_TOKENS + EXTRAS + tuple(vision_tokens),
            start=SPECIAL_START_ID,
        ))

        # 创建 tiktoken 编码器
        enc = tiktoken.Encoding("mammothu", pat_str=PAT_STR,
                                mergeable_ranks=self.mergeable_ranks,
                                special_tokens=self.special_tokens)

        # 视觉 token 范围（用于多模态处理）
        self.vision_range = (self.get_vocab()[self.boi_token], self.tokenizer.n_vocab - 1)
```

### 序列化支持

```python
def __getstate__(self):
    state = self.__dict__.copy()
    del state["tokenizer"]  # tiktoken 对象不可 pickle
    return state

def __setstate__(self, state):
    self.__dict__.update(state)
    # 反序列化时重建 tiktoken 编码器
    enc = tiktoken.Encoding("mammothu", pat_str=PAT_STR, ...)
    self.tokenizer = enc
```

tiktoken 的 `Encoding` 对象不支持 Python pickle，因此序列化时移除并在反序列化时重建。

### 分词与反分词

```python
def tokenize(self, text, allowed_special="all", disallowed_special=()):
    tokens = []
    text = unicodedata.normalize("NFC", text)
    for t in self.tokenizer.encode(text, allowed_special=allowed_special, ...):
        tokens.append(self.decoder[t])
    return tokens

def _decode(self, token_ids, skip_special_tokens=False, errors=None):
    if skip_special_tokens:
        token_ids = [i for i in token_ids if i < self.eod_id]
    return self.tokenizer.decode(token_ids, errors=errors or self.errors)
```

分词路径：`text -> NFC normalize -> tiktoken encode -> token IDs -> surface forms`

### Token 添加限制

```python
def _add_tokens(self, new_tokens, special_tokens=False):
    if not special_tokens and new_tokens:
        raise ValueError("Adding regular tokens is not supported")
    # 只接受已在预定义集合中的特殊 token
```

该分词器不允许动态添加常规 token，只允许添加已在 `special_tokens_set` 中预定义的特殊 token。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `MammothUTokenizer` | 类 | MammothModa2 专用分词器，继承 `PreTrainedTokenizer` |
| `_load_tiktoken_bpe(tiktoken_bpe_file)` | 函数 | 加载 tiktoken BPE 合并规则文件 |
| `PAT_STR` | 常量 | 正则分词模式（与 Qwen 系列一致） |
| `SPECIAL_START_ID` | 常量 | 特殊 token 起始 ID (151643) |
| `QWEN_SPECIAL_TOKENS` | 常量 | Qwen 风格的特殊 token 列表 |
| `EXTRAS` | 常量 | 扩展 token 列表（对齐词汇表大小） |

## 与其他模块的关系

- **与配置类配合**: `transformers_utils/configs/mammoth_moda2.py` 中的配置指定 `tokenizer_class = "MammothUTokenizer"`。
- **基于 tiktoken**: 使用 tiktoken 库进行实际的 BPE 编码/解码。
- **兼容 HuggingFace**: 继承 `PreTrainedTokenizer`，支持 `from_pretrained`、`save_pretrained` 等标准接口。
- **多模态支持**: 通过 `vision_range`、`visual_tokens`、`gen_placeholder_id` 等属性支持多模态推理。

## 总结

`MammothUTokenizer` 是一个为 MammothModa2 多模态模型定制的分词器，融合了 tiktoken 的高性能 BPE 分词和 HuggingFace 的标准接口。它管理一个复杂的词汇表体系，包含基础文本 token、Qwen 兼容的特殊 token、视觉相关 token 和生成占位 token，为多模态模型的文本-图像-视频联合处理提供了分词基础。
