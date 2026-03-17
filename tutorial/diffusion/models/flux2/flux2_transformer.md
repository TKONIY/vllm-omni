# `flux2_transformer.py` -- Flux 2 Transformer 模型

## 文件概述

实现 Flux 2 的 Transformer 架构，相较 FLUX.1 的主要改进包括：SwiGLU 替代 GELU 激活、全局 Modulation 参数共享（而非逐块独立）、4 轴 RoPE、无 bias 设计。

**文件路径**: `vllm_omni/diffusion/models/flux2/flux2_transformer.py`

## 关键代码解析

### Flux2SwiGLU 激活函数

```python
class Flux2SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2  # SiLU(x1) * x2
```

### Flux2Modulation 全局调制参数

```python
class Flux2Modulation(nn.Module):
    def __init__(self, dim, mod_param_sets=2):
        self.linear = nn.Linear(dim, dim * 3 * mod_param_sets, bias=False)
    def forward(self, temb):
        mod = self.act_fn(temb)
        mod = self.linear(mod)
        mod_params = torch.chunk(mod, 3 * self.mod_param_sets, dim=-1)
        return tuple(mod_params[3*i:3*(i+1)] for i in range(self.mod_param_sets))
```

与 FLUX.1 的 AdaLayerNormZero（每个块独立计算 shift/scale/gate）不同，Flux 2 在模型顶层计算全局调制参数并传递给所有块。

### Flux2Transformer2DModel 主模型

```python
class Flux2Transformer2DModel(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, guidance):
        temb = self.time_guidance_embed(timestep * 1000, guidance * 1000)
        # 全局调制参数（所有块共享）
        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)[0]
        # 双流块
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt, ...)
        # 单流块
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=None,
                                  temb_mod_params=single_stream_mod, ...)
```

### Flux2ParallelSelfAttention 并行注意力+MLP

```python
class Flux2ParallelSelfAttention(nn.Module):
    def __init__(self, ...):
        # 将 QKV 和 MLP 输入投影融合为一个大矩阵
        self.to_qkv_mlp_proj = ColumnParallelLinear(
            query_dim, inner_dim*3 + mlp_hidden_dim*mlp_mult_factor, ...)
    def forward(self, hidden_states, ...):
        hidden_states, _ = self.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(hidden_states, [...], dim=-1)
        # 并行计算 attention 和 MLP
        attn_output = self.attn(query, key, value, ...)
        mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
```

将 QKV 投影和 MLP 输入投影融合为单次矩阵乘法，减少访存开销。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Flux2Transformer2DModel` | nn.Module | Flux 2 完整 Transformer |
| `Flux2TransformerBlock` | nn.Module | 双流块 |
| `Flux2SingleTransformerBlock` | nn.Module | 单流块 |
| `Flux2Attention` | nn.Module | 联合注意力 |
| `Flux2ParallelSelfAttention` | nn.Module | 融合 QKV+MLP 的并行注意力 |
| `Flux2Modulation` | nn.Module | 全局调制参数 |
| `Flux2PosEmbed` | nn.Module | 4 轴 RoPE (32x4=128) |
| `Flux2TimestepGuidanceEmbeddings` | nn.Module | 时间步+引导嵌入 |
| `Flux2SwiGLU` | nn.Module | SwiGLU 激活 |
| `Flux2FeedForward` | nn.Module | MergedColumnParallel SwiGLU FFN |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 被使用 | `pipeline_flux2.py` | Pipeline 创建并调用 Transformer |
| 依赖 | vLLM 并行层 | QKVParallelLinear、MergedColumnParallelLinear 等 |
| 依赖 | `vllm_omni.diffusion.attention` | 优化注意力后端 |

## 总结

Flux 2 Transformer 相较 FLUX.1 的关键改进：(1) SwiGLU 激活替代 GELU，(2) 全局 Modulation 参数共享减少参数量，(3) 4 轴 RoPE (每轴 32 维)，(4) 无 bias 设计减少参数，(5) 单流块中 QKV 与 MLP 投影融合为单次矩阵运算，提升计算效率。
