# `qwen3_omni_moe_code_predictor_mtp.py` — Code Predictor (MTP)

## 文件概述

本文件实现了 Qwen3-Omni 的 Code Predictor，用于从 Talker 生成的第 0 层 codec code 自回归预测剩余 1~15 层的 RVQ 残差编码。采用 **re-prefill（重新预填充）** 策略而非 KV 缓存，每一步都完整前向传播整个（短）序列。

## 关键代码解析

### 1. 注意力模块（SDPA，无 KV 缓存）

```python
class Qwen3OmniCodePredictorAttention(nn.Module):
    def forward(self, hidden_states, position_ids):
        qkv, _ = self.qkv_proj(hidden_states.reshape(bsz * seq_len, -1))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self._use_gqa)
```

直接使用 PyTorch 的 SDPA，不使用 vLLM 的 paged attention。

### 2. 持久化缓冲区

```python
class Qwen3OmniMoeTalkerCodePredictor(nn.Module):
    def _ensure_buffers(self, bsz, device, dtype):
        max_seq = self.num_code_groups + 1
        self._proj_buf = torch.zeros(bsz, max_seq, self._hidden_size, dtype=dtype, device=device)
        self._pos_ids = torch.arange(max_seq, dtype=torch.long, device=device)
```

预分配嵌入缓冲区和位置 ID，避免每步分配内存。

### 3. Re-Prefill 自回归循环

```python
def forward(self, layer0_code, layer0_embed, last_talker_hidden):
    all_codes = torch.empty(bsz, num_groups, 1, dtype=torch.int64, device=device)
    all_codes[:, 0] = layer0_code
    proj_buf[:bsz, 0:1, :] = last_talker_hidden  # 位置 0: talker 隐藏状态
    proj_buf[:bsz, 1:2, :] = layer0_embed         # 位置 1: layer-0 嵌入

    for step in range(1, num_groups):
        seq_len = step + 1
        projected = proj_buf[:bsz, :seq_len, :]
        hidden_out = model_fwd(projected, step_pos_ids)  # 完整序列前向传播

        # 内联 top-k 采样
        logits = lm_heads[step - 1](hidden_out[:, -1, :])
        topk_vals, _ = logits.topk(top_k, dim=-1)
        logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
        code = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        all_codes[:, step] = code
        proj_buf[:bsz, step+1:step+2, :] = codec_embeds[step-1](code)
```

每一步都重新前向传播整个序列（长度 2→num_code_groups+1），利用序列很短（最多 17 步）的特点。

### 4. torch.compile 加速

```python
def _ensure_model_fwd(self):
    if not current_omni_platform.supports_torch_inductor():
        self._model_fwd = self.model.forward
        return
    self._model_fwd = torch.compile(self.model.forward, mode="default", dynamic=True)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen3OmniCodePredictorAttention` | 类 | SDPA 注意力（无 KV 缓存） |
| `Qwen3OmniCodePredictorMLP` | 类 | SiLU-gated MLP |
| `Qwen3OmniCodePredictorDecoderLayer` | 类 | Transformer 解码层 |
| `Qwen3OmniCodePredictorBaseModel` | 类 | 内部 Transformer 模型 |
| `Qwen3OmniMoeTalkerCodePredictor` | 类 | Code Predictor 包装器 |
| `_ensure_buffers()` | 方法 | 预分配持久化缓冲区 |
| `forward()` | 方法 | Re-prefill 自回归生成 |

## 与其他模块的关系

- **被引用**: `qwen3_omni_moe_talker.py` 中实例化为 `self.code_predictor`
- **被调用**: `qwen3_omni.py` 中的 `talker_mtp()` 调用 `code_predictor_forward()`
- **不使用** vLLM 的 paged attention，是完全独立的 Transformer

## 总结

Code Predictor 是 Qwen3-Omni 区别于 Qwen2.5-Omni 的核心创新之一。它采用 re-prefill 策略（每步重新前向传播完整序列），利用 RVQ 残差序列很短（16 步）的特点，避免了 KV 缓存管理的复杂性。关键优化包括：持久化缓冲区（零分配）、内联 top-k 采样（无 LogitsProcessor 开销）、以及 torch.compile 加速。
