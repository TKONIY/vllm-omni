# `cosyvoice3_talker.py` — Talker 阶段 LM 模型

## 文件概述

定义 CosyVoice3 的语言模型组件，包括基于 Qwen2 的编码器封装和多层次语音语言模型类。该文件实现了从文本到语音 token 的自回归生成。

## 关键代码解析

### VLLMQwen2Encoder

```python
class VLLMQwen2Encoder(torch.nn.Module):
    """使用 vLLM Qwen2Model 的编码器，支持 PagedAttention"""
    def __init__(self, vllm_config, prefix=""):
        self.model = Qwen2Model(vllm_config=vllm_config, prefix=prefix)

    def forward(self, inputs_embeds, positions):
        # KV cache 由 GPUARModelRunner 通过 ForwardContext 外部管理
        hidden_states = self.model(input_ids=..., positions=positions_flat,
                                    inputs_embeds=inputs_flat)
```

### CosyVoice3LM（最终 LM 类）

```python
class CosyVoice3LM(Qwen2LM):
    def __init__(self, ...):
        self.sos = speech_token_size + 0         # SOS token ID
        self.eos_token = speech_token_size + 1   # EOS token ID
        self.task_id = speech_token_size + 2     # 任务 ID
        self.fill_token = speech_token_size + 3  # 填充 token
        # 语音嵌入：speech_token_size + 200 个词条
        self.speech_embedding = nn.Embedding(speech_token_size + 200, llm_input_size)
        # 解码头
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 200, bias=False)
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `VLLMQwen2Encoder` | 类 | vLLM 优化的 Qwen2 编码器 |
| `TransformerLM` | 类 | 基础 Transformer LM（文本编码器 + LLM） |
| `Qwen2LM` | 类 | Qwen2 变体 LM（继承自 TransformerLM） |
| `CosyVoice3LM` | 类 | CosyVoice3 专用 LM（扩展词汇表） |

## 与其他模块的关系

- `VLLMQwen2Encoder` 使用 vLLM 的 `Qwen2Model` 实现高效推理
- 被 `cosyvoice3.py` 中的 `CosyVoice3Model` 实例化（talker 阶段）
- 权重从 `llm.pt` 加载

## 总结

该文件定义了三层继承的语音语言模型类（TransformerLM → Qwen2LM → CosyVoice3LM），通过 vLLM 的 PagedAttention 实现高效自回归生成。CosyVoice3LM 特有的设计是将 SOS/EOS/TaskID 等特殊 token 编码到扩展的 speech_embedding 空间中。
