# `glm_image_transformer.py` -- GLM-Image Transformer 模型

## 文件概述

实现 GLM-Image 的扩散 Transformer，包含完整的 KV 缓存系统用于图像编辑工作流。该模型使用 prior token（由 AR 阶段生成）作为图像条件，通过 ByT5 字形嵌入支持文本渲染，并支持 KV 缓存的写入/读取模式实现图像编辑。

**文件路径**: `vllm_omni/diffusion/models/glm_image/glm_image_transformer.py`

## 关键代码解析

### KVCacheMode 和 KV 缓存系统

```python
class KVCacheMode(Enum):
    WRITE = "write"   # 存储条件图像的 K/V
    READ = "read"     # 读取缓存并拼接
    SKIP = "skip"     # 不使用缓存

class GlmImageLayerKVCache:
    def store(self, key, value):
        if self.k_cache is None:
            self.k_cache = key
        else:
            self.k_cache = torch.cat([self.k_cache, key], dim=1)  # 沿序列维累积

class GlmImageKVCache:
    def __init__(self, num_layers):
        self.caches = [GlmImageLayerKVCache() for _ in range(num_layers)]
```

三层 KV 缓存结构：
- `GlmImageKVCache`: 管理所有层的缓存，统一设置模式
- `GlmImageLayerKVCache`: 单层的 KV 存储
- `KVCacheMode`: WRITE（处理条件图像）-> READ（去噪目标图像）

### GlmImageRotaryPosEmbed 2D 旋转位置编码

```python
class GlmImageRotaryPosEmbed(nn.Module):
    def forward(self, hidden_states):
        # 分别计算高度和宽度方向的频率
        freqs_h = torch.outer(h_seq, h_inv_freq)  # [H, dim//4]
        freqs_w = torch.outer(w_seq, w_inv_freq)  # [W, dim//4]
        # 扩展并拼接
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)  # [H, W, dim//2]
        freqs = torch.cat([freqs, freqs], dim=-1)       # [H, W, dim]
        return (freqs.cos(), freqs.sin())
```

为 2D 图像 patch 生成位置编码，高度和宽度各占一半维度。

### GlmImageAdaLayerNormZero 自适应层归一化

```python
class GlmImageAdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim, dim):
        self.linear = nn.Linear(embedding_dim, 12 * dim)  # 12 个调制参数

    def forward(self, hidden_states, encoder_hidden_states, temb):
        emb = self.linear(temb)
        (shift_msa, c_shift_msa, scale_msa, c_scale_msa,
         gate_msa, c_gate_msa, shift_mlp, c_shift_mlp,
         scale_mlp, c_scale_mlp, gate_mlp, c_gate_mlp) = emb.chunk(12, dim=1)
```

一次性生成 12 个调制参数（图像流和文本流各 6 个：shift/scale/gate x attention/FFN）。

### GlmImageAttention 联合注意力 + KV 缓存

```python
class GlmImageAttention(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, image_rotary_emb, kv_cache, kv_cache_mode):
        # 拼接文本和图像
        hidden_states_combined = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        qkv, _ = self.to_qkv(hidden_states_combined)
        # 仅对图像 token 应用 RoPE
        query_img = apply_rotary_emb(query_img, image_rotary_emb, ...)
        key_img = apply_rotary_emb(key_img, image_rotary_emb, ...)
        # KV 缓存处理
        if kv_cache_mode == KVCacheMode.WRITE:
            kv_cache.store(key, value)
        elif kv_cache_mode == KVCacheMode.READ:
            k_cached, v_cached = kv_cache.get()
            key = torch.cat([k_cached, key], dim=1)
```

使用 LayerNorm（而非 RMSNorm）进行 QK 归一化，RoPE 仅应用于图像 token。

### GlmImageTransformer2DModel 主模型

```python
class GlmImageTransformer2DModel(CachedTransformer):
    def forward(self, hidden_states, encoder_hidden_states, prior_token_id, ...):
        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)
        # 2. Patch + Prior embedding
        hidden_states = self.image_projector(hidden_states)
        encoder_hidden_states = self.glyph_projector(encoder_hidden_states)
        prior_embedding = self.prior_token_embedding(prior_token_id)
        hidden_states = hidden_states + prior_hidden_states
        # 3. Transformer blocks (with per-layer KV cache)
        for layer_idx, block in enumerate(self.transformer_blocks):
            layer_kv_cache = kv_cache[layer_idx] if kv_cache is not None else None
            hidden_states, encoder_hidden_states = block(..., kv_cache=layer_kv_cache, ...)
        # 4. Unpatchify
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `GlmImageTransformer2DModel` | nn.Module | 完整 Transformer（继承 CachedTransformer） |
| `GlmImageTransformerBlock` | nn.Module | Transformer 块 |
| `GlmImageAttention` | nn.Module | 联合注意力 + KV 缓存 |
| `GlmImageKVCache` | 数据类 | 全层 KV 缓存管理器 |
| `GlmImageLayerKVCache` | 数据类 | 单层 KV 缓存 |
| `KVCacheMode` | 枚举 | 缓存操作模式 |
| `GlmImageImageProjector` | nn.Module | 图像 patch 投影 |
| `GlmImageRotaryPosEmbed` | nn.Module | 2D RoPE |
| `GlmImageAdaLayerNormZero` | nn.Module | 自适应层归一化 (12 参数) |
| `GlmImageAdaLayerNormContinuous` | nn.Module | 输出层 AdaLN（无激活） |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 被使用 | `pipeline_glm_image.py` | Pipeline 调用 |
| 继承 | `CachedTransformer` | 缓存 Transformer 基类 |
| 依赖 | diffusers | `GlmImageCombinedTimestepSizeEmbeddings` |

## 总结

GLM-Image Transformer 的核心特色：(1) prior token 条件——AR 模型生成的离散 token 通过嵌入层转为连续条件，(2) KV 缓存系统支持图像编辑（先 WRITE 缓存条件图像特征，再 READ 用于去噪），(3) 2D RoPE 仅应用于图像 token，(4) ByT5 字形嵌入支持文本渲染，(5) 12 参数 AdaLN 同时调制图像流和文本流。
