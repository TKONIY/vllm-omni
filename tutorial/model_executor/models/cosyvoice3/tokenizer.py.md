# `tokenizer.py` — CosyVoice 文本分词器

## 文件概述

定义 CosyVoice2 和 CosyVoice3 的文本分词器，基于 Qwen tokenizer 扩展了大量特殊 token（语气词、音素等）。

## 关键代码解析

```python
class CosyVoice3Tokenizer(CosyVoice2Tokenizer):
    """扩展了大量中英文音素 token"""
    def __init__(self, token_path, skip_special_tokens=True):
        special_tokens = {
            "additional_special_tokens": [
                "[breath]", "[noise]", "[laughter]",  # 非语言声音
                "[AA]", "[AE]", ...,                   # 英文 CMU 音素
                "[a]", "[ai]", "[an]", ...,            # 中文拼音音素
                "[à]", "[á]", "[ā]", ...,              # 带声调拼音
            ],
        }
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(special_tokens)

@cache
def get_qwen_tokenizer(token_path, skip_special_tokens, version="cosyvoice3"):
    """缓存的 tokenizer 工厂函数"""
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `CosyVoice2Tokenizer` | 类 | V2 版分词器，19 个特殊 token |
| `CosyVoice3Tokenizer` | 类 | V3 版分词器，300+ 个音素 token |
| `get_qwen_tokenizer()` | 函数 | 缓存的工厂函数，按版本创建 tokenizer |

## 总结

CosyVoice3 大幅扩展了音素词表，支持更精细的中英文发音控制。使用 `@cache` 装饰器确保同一路径的 tokenizer 只加载一次。
