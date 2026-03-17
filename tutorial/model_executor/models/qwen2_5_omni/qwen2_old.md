# `qwen2_old.py` — 旧版 Qwen2 因果语言模型

## 文件概述

本文件实现了 Qwen2 因果语言模型的旧版本，专门用于 Qwen2.5-Omni Talker 内部。与 vLLM 上游的 Qwen2 实现相比，此版本保持了与 Talker 权重检查点的兼容性（使用 `embedding_size` 配置、特定的 RoPE 参数等）。

## 关键代码解析

### 1. Qwen2MLP

```python
class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, ...):
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False, ...)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, ...)
        self.act_fn = SiluAndMul()
```

标准 SwiGLU MLP：gate 和 up 投影融合，使用 SiLU 激活。

### 2. Qwen2Attention

```python
class Qwen2Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, ...):
        self.qkv_proj = QKVParallelLinear(...)  # 融合的 QKV 投影
        self.o_proj = RowParallelLinear(...)
        self.rotary_pos_emb = get_rope(...)  # RoPE 旋转位置编码
        self.attn = Attention(...)           # vLLM 注意力后端
```

注意 `bias=True` 用于 QKV 投影（与 Qwen2 原始设计一致），支持 GQA。

### 3. Qwen2Model

```python
@support_torch_compile(dynamic_arg_dims={...})
class Qwen2Model(nn.Module):
    def __init__(self, *, vllm_config, prefix, decoder_layer_type=Qwen2DecoderLayer):
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            getattr(config, "embedding_size", config.hidden_size), ...)
```

使用 `embedding_size` 而非 `hidden_size` 初始化嵌入层——这是 Talker 版本的关键区别。

### 4. Qwen2ForCausalLM

```python
class Qwen2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    def __init__(self, *, vllm_config, prefix=""):
        self.model = Qwen2Model(vllm_config=vllm_config, ...)
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, ...)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen2MLP` | 类 | SwiGLU MLP |
| `Qwen2Attention` | 类 | 多头注意力（支持 GQA） |
| `Qwen2DecoderLayer` | 类 | Transformer 解码器层 |
| `Qwen2Model` | 类 | Transformer 主体模型 |
| `Qwen2ForCausalLM` | 类 | 完整的因果语言模型 |

## 与其他模块的关系

- **被引用**: `qwen2_5_omni_talker.py` 中通过 `architectures=["Qwen2ForCausalLM_old"]` 注册并实例化
- **区别**: 与 vLLM 上游 `Qwen2ForCausalLM` 相比，使用 `embedding_size` 配置参数
- **用途**: 仅作为 Talker 的内部语言模型骨干

## 总结

`qwen2_old.py` 是 Qwen2 因果语言模型的一个定制版本，专为 Talker 的 codec token 生成优化。它保留了标准的 Transformer 解码器架构（RMSNorm、RoPE、GQA、SwiGLU），但在嵌入层初始化时使用 `embedding_size` 以匹配 Talker 检查点的权重形状。该文件是 vLLM-Omni 对 Qwen2.5-Omni Talker 的忠实还原。
