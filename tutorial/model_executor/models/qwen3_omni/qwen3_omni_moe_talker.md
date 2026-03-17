# `qwen3_omni_moe_talker.py` — Talker 模型

## 文件概述

本文件实现了 Qwen3-Omni MoE 的 Talker 模型，将 Thinker 的文本嵌入转换为 RVQ 第 0 层 codec codes。Talker 使用 MoE Transformer 架构，并集成了 Code Predictor 用于生成剩余 RVQ 层。

## 关键代码解析

### 1. 双投影层

```python
class Qwen3OmniMoeTalkerResizeMLP(nn.Module):
    def __init__(self, config: Qwen3OmniMoeTalkerConfig):
        self.linear_fc1 = nn.Linear(config.thinker_hidden_size, config.text_config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.text_config.intermediate_size, config.text_config.hidden_size, bias=True)
        self.act_fn = _ACTIVATION_REGISTRY[config.text_config.hidden_act]
```

两个投影实例：
- `text_projection`: 文本嵌入（thinker embedding → talker dimension）
- `hidden_projection`: 多模态隐藏状态（thinker last layer → talker dimension）

### 2. Codec 嵌入替换

```python
class Qwen3OmniMoeModel(Qwen3MoeLLMForCausalLM):
    def __init__(self, vllm_config, talker_config, prefix):
        super().__init__(...)
        if hasattr(self, "lm_head"): del self.lm_head
        if hasattr(self.model, "embed_tokens"): del self.model.embed_tokens
        self.model.codec_embedding = nn.Embedding(
            talker_config.text_config.vocab_size,
            talker_config.text_config.hidden_size)
```

删除继承的 `lm_head` 和 `embed_tokens`，替换为 `codec_embedding`。

### 3. Code Predictor 集成

```python
def code_predictor_forward(self, input_ids, inputs_embeds, *, last_talker_hidden, ...):
    for pos in range(seq_len):
        layer0_code = input_ids[:, pos:pos+1]
        layer0_embed = embed_fn(layer0_code)
        pos_all_layers, proj_buf = self.code_predictor(layer0_code, layer0_embed, last_talker_hidden)
        result_codes[:, :, pos:pos+1] = pos_all_layers
        summed_embeddings[:, pos, :] = proj_buf[:, 1:, :].sum(dim=1)
```

对每个位置调用 Code Predictor 生成 16 层 codes，并将 codec 嵌入求和作为下一步输入。

### 4. 权重映射

```python
hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
    "talker.model.": "language_model.model.",
    "talker.codec_head.": "codec_head.",
    "talker.code_predictor.": "code_predictor.",
    "talker.text_projection.": "text_projection.",
    "talker.hidden_projection.": "hidden_projection.",
    "talker.": "",
})
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen3OmniMoeTalkerForConditionalGeneration` | 类 | Talker 主模型 |
| `Qwen3OmniMoeTalkerResizeMLP` | 类 | Thinker→Talker 投影 MLP |
| `Qwen3OmniMoeModel` | 类 | Talker 内部 MoE 语言模型 |
| `project_thinker_outputs()` | 方法 | 投影 thinker 输出 |
| `code_predictor_forward()` | 方法 | Code Predictor 前向传播 |
| `compute_logits()` | 方法 | Codec 头计算 logits |
| `embed_multimodal()` | 方法 | 多模态嵌入（profile 用） |

## 与其他模块的关系

- **被引用**: `qwen3_omni.py` 通过架构名实例化
- **依赖**: `qwen3_omni_moe_code_predictor_mtp.py` 中的 Code Predictor
- **依赖**: `qwen3_omni_moe_thinker.py` 中的 Mixin 和视觉/音频编码器
- **依赖**: `qwen3_moe.py` 中的 MoE 因果语言模型

## 总结

Qwen3 Talker 相比 Qwen2.5 Talker 有几个关键改进：(1) 使用 MoE 架构替代 Dense Transformer；(2) 双投影层区分文本和多模态区域；(3) 集成 Code Predictor 生成多层 RVQ codes；(4) 使用 codec_embedding 替代传统的 text embedding。
