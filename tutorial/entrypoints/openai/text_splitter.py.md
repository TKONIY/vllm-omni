# `text_splitter.py` — 句子分割器

## 文件概述

实现了多语言句子边界检测器 `SentenceSplitter`，用于流式 TTS 的文本预处理。该分割器缓冲增量输入文本，在检测到句子边界时产出完整句子，特别针对中英文混合场景做了优化。

## 关键代码解析

### 边界正则表达式

```python
# 句子级别：英文句末标点 + 中日韩句号
SPLIT_SENTENCE = re.compile(
    r"(?<=[.!?])\s+"       # 英文句末标点后必须有空格确认
    r"|(?<=[。！？])"       # CJK 句末标点直接分割
)

# 子句级别：额外包含中文逗号和分号
SPLIT_CLAUSE = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?<=[。！？，；])"
)
```

英文标点后要求空格确认（避免 "Dr." "U.S." 等误分割），CJK 标点直接分割。

### 增量分割逻辑

```python
class SentenceSplitter:
    def __init__(self, min_sentence_length=2, boundary_re=None):
        self._buffer = ""
        self._min_sentence_length = min_sentence_length
        self._boundary_re = boundary_re or SPLIT_SENTENCE

    def add_text(self, text: str) -> list[str]:
        """添加文本并返回检测到的完整句子"""
        self._buffer += text
        if len(self._buffer) > _MAX_BUFFER_SIZE:
            raise ValueError("缓冲区溢出")
        return self._extract_sentences()

    def _extract_sentences(self):
        parts = self._boundary_re.split(self._buffer)
        if len(parts) <= 1:
            return []  # 没有边界，继续缓冲
        sentences = []
        carry = ""
        for i in range(len(parts) - 1):
            text = carry + parts[i]
            stripped = text.strip()
            if len(stripped) >= self._min_sentence_length:
                sentences.append(stripped)
                carry = ""
            elif stripped:
                carry = text  # 太短（如 "Dr."），继续拼接
        self._buffer = carry + parts[-1]  # 最后一部分保留在缓冲区
        return sentences

    def flush(self) -> str | None:
        """输入结束时，刷新缓冲区中的剩余文本"""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None
```

关键设计：
1. 最短句子长度保护（默认 2 字符），避免缩写触发误分割
2. 过短的片段向后携带（carry forward），与下一个片段合并
3. 缓冲区大小上限（100K 字符）防止内存泄漏
4. `flush()` 处理最后一个不完整的句子

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `SentenceSplitter` | 类 | 增量句子分割器 |
| `add_text()` | 方法 | 输入文本并获取完整句子 |
| `flush()` | 方法 | 刷新缓冲区 |
| `SPLIT_SENTENCE` | 正则常量 | 句子级分割模式 |
| `SPLIT_CLAUSE` | 正则常量 | 子句级分割模式 |

## 与其他模块的关系

- 被 `serving_speech_stream.py` 的 `OmniStreamingSpeechHandler` 使用
- 分割粒度由 `StreamingSpeechSessionConfig.split_granularity` 配置

## 总结

一个专为流式 TTS 设计的多语言句子分割器，通过增量缓冲和智能边界检测，在文本逐步到达时准确地提取完整句子。其中英文混合处理和最短长度保护机制确保了在各种文本风格下的鲁棒分割。
