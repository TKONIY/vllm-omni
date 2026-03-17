# `qwen3_tts_talker.py` — Talker 模型（AR 文本→codec）

## 文件概述

本文件实现了 Qwen3-TTS 的 Talker 模型，使用 Qwen3 Transformer 解码器自回归生成 codec tokens。包含完整的说话人编码器（ECAPA-TDNN）、Resize MLP 投影层、以及 Code Predictor 集成。

## 关键代码解析

### 1. 说话人编码器（ECAPA-TDNN）

```python
class Qwen3TTSSpeakerEncoder(nn.Module):
    # 组件：TimeDelayNetBlock → Res2NetBlock → SqueezeExcitationRes2NetBlock
    #      → AttentiveStatisticsPooling → Linear
    # 输入：梅尔频谱 → 输出：说话人嵌入向量
```

从参考音频的梅尔频谱提取固定维度的说话人嵌入（x-vector）。

### 2. Resize MLP

```python
class Qwen3TTSTalkerResizeMLP(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size, act, bias=False):
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = ACT2FN[act]
```

用于 text_hidden_size → talker hidden_size 的维度映射。

### 3. Talker 主模型

```python
class Qwen3TTSTalkerForConditionalGeneration(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        self.model = Qwen3Model(vllm_config=vllm_config, ...)   # Qwen3 解码器
        self.lm_head = ParallelLMHead(...)                       # Codec 头
        self.text_projection = Qwen3TTSTalkerResizeMLP(...)      # 文本投影
        self.code_predictor = Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(...)
        self.speaker_encoder = Qwen3TTSSpeakerEncoder(...)
```

### 4. 权重映射

```python
hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
    "talker.model.": "model.",
    "talker.text_projection.": "text_projection.",
    "talker.code_predictor.": "code_predictor.",
    "talker.codec_head.": "lm_head.",
    "speaker_encoder.": "speaker_encoder.",
})
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen3TTSTalkerForConditionalGeneration` | 类 | Talker 主模型 |
| `Qwen3TTSSpeakerEncoder` | 类 | ECAPA-TDNN 说话人编码器 |
| `Qwen3TTSTalkerResizeMLP` | 类 | 维度映射 MLP |
| `TimeDelayNetBlock` | 类 | TDNN 块 |
| `Res2NetBlock` | 类 | Res2Net 块 |
| `SqueezeExcitationRes2NetBlock` | 类 | SE-Res2Net 块 |
| `AttentiveStatisticsPooling` | 类 | 注意力统计池化 |

## 与其他模块的关系

- **被引用**: `pipeline.yaml` 中作为 Stage 0 的模型架构
- **依赖**: `qwen3_tts_code_predictor_vllm.py` 中的 Code Predictor
- **依赖**: `configuration_qwen3_tts.py` 中的配置类
- **依赖**: vLLM 上游 `Qwen3Model` 作为解码器骨干

## 总结

Qwen3-TTS Talker 集成了说话人编码（ECAPA-TDNN）、文本到 codec 的 AR 生成（Qwen3 Transformer）和 Code Predictor（MTP），形成完整的文本到语音编码流水线。与 Qwen3-Omni Talker 的主要区别在于：使用 Dense Transformer（非 MoE）、集成说话人编码器、以及支持声音克隆模式。
