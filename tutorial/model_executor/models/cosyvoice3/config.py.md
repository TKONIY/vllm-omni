# `config.py` — CosyVoice3 配置类

## 文件概述

定义 `CosyVoice3Config` 配置类，继承自 HuggingFace 的 `PretrainedConfig`。该配置类集中管理 CosyVoice3 模型所有组件的超参数，包括 LLM、流匹配解码器、HiFT 声码器等。

## 关键代码解析

```python
class CosyVoice3Config(PretrainedConfig):
    model_type = "cosyvoice3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = 24000        # 采样率
        self.llm_input_size = 896       # LLM 输入维度
        self.token_frame_rate = 25      # 每秒 token 帧数
        self.token_mel_ratio = 2        # token 到 mel 帧的比例
        self.vocab_size = 151923        # 词汇表大小
```

### 配置分区

配置被组织为几个嵌套字典：

- **`self.llm`**：LLM 相关参数（语音 token 大小、EOS、采样策略等）
- **`self.flow`**：流匹配解码器参数（CFM 配置、DiT estimator 参数）
- **`self.hift`**：HiFiGAN 声码器参数（上采样率、残差块、F0 预测器）
- **`self.feat_extractor`**：mel 频谱提取参数

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `CosyVoice3Config` | 类 | CosyVoice3 全量配置，model_type="cosyvoice3" |

## 与其他模块的关系

- 被 `cosyvoice3.py`、`cosyvoice3_code2wav.py` 等模块导入使用
- 参数值直接传递给各组件的构造函数

## 总结

集中式配置类，将 LLM、流匹配、声码器三大组件的超参数统一管理，便于模型的序列化和复现。
