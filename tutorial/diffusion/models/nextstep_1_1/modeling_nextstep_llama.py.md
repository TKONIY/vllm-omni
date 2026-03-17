# `modeling_nextstep_llama.py` — NextStep-1.1 TP 感知的 LLaMA 层

## 文件概述

该文件实现了 NextStep-1.1 使用的 TP（张量并行）感知 LLaMA 模型组件，包括注意力层、MLP、RMSNorm 和旋转位置编码。这些组件使用 vLLM 的并行线性层替代标准 PyTorch 线性层。

## 关键代码解析

### TP 感知的注意力层

```python
class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # 融合的 QKV 投影（张量并行按 head 分片）
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=config.attention_bias,
        )
        # 行并行输出投影（自动 all-reduce）
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, self.hidden_size,
            bias=getattr(config, "o_attention_bias", config.attention_bias),
        )
```

### TP 感知的 MLP

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        # 融合的 gate + up 投影
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size, [self.intermediate_size] * 2, bias=config.mlp_bias,
        )
        # 行并行 down 投影
        self.down_proj = RowParallelLinear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias,
        )
    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        down, _ = self.down_proj(self.act_fn(gate) * up)
```

### RoPE 和 KV 缓存

```python
class LlamaAttention:
    def forward(self, hidden_states, position_embeddings, past_key_value, ...):
        qkv, _ = self.qkv_proj(hidden_states)
        # 按 TP 分片后的 head 数分割
        query_states, key_states, value_states = qkv.split(split_sizes, dim=2)
        # 应用 RoPE
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # 更新 KV 缓存
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, ...)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `LlamaAttention` | 类 | TP 感知的注意力层 |
| `LlamaMLP` | 类 | TP 感知的 MLP |
| `LlamaDecoderLayer` | 类 | 完整的解码器层 |
| `LlamaRMSNorm` | 类 | RMSNorm 归一化 |
| `LlamaRotaryEmbedding` | 类 | 旋转位置编码 |

## 与其他模块的关系

- 被 `modeling_nextstep.py` 中的 `NextStepModel` 使用
- 使用 vLLM 的 `QKVParallelLinear`、`RowParallelLinear`、`MergedColumnParallelLinear`

## 总结

该文件将标准 LLaMA 的线性层替换为 vLLM 的张量并行版本，实现了 NextStep-1.1 LLM 骨干的高效多 GPU 推理。核心修改点包括：融合 QKV 投影、融合 gate+up 投影，以及使用 `RowParallelLinear` 自动执行 all-reduce。
