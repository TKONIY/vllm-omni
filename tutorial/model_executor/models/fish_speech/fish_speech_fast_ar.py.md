# `fish_speech_fast_ar.py` — Fast AR 残差 codebook 预测器

## 文件概述

实现 Fish Speech 的 Fast AR 模块：4 层 Transformer，自回归预测残差 codebook 编码（1..num_codebooks-1）。采用 re-prefill 策略（每步完整前向，无 KV cache），优化手段包括 torch.compile 和预分配缓冲区。

## 关键代码解析

### 注意力层（SDPA，无 KV cache）

```python
class _FastARAttention(nn.Module):
    """使用 F.scaled_dot_product_attention 的多头注意力"""
    def forward(self, hidden_states, position_ids):
        q, k = self.rotary_emb(position_ids, q, k)  # 交错 RoPE
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### 自回归预测循环

```python
class FishSpeechFastAR(nn.Module):
    def forward(self, slow_ar_hidden, semantic_token_id):
        # 位置 0: Slow AR 隐状态投影
        embed_buf[:bsz, 0, :] = self.fast_project_in(slow_ar_hidden)
        # 位置 1: 语义 code 嵌入
        embed_buf[:bsz, 1, :] = self.fast_embeddings(semantic_code)
        # 步骤 1~9: 逐步预测残差 code
        for step in range(1, num_cb):
            hidden_out = model_fwd(embed_buf[:bsz, :step+1, :], pos_ids)
            logits = self.fast_output(self.fast_norm(hidden_out[:, -1, :]))
            logits = logits[:, :1024]  # 残差 codebook 只有 1024 词条
            next_ids = torch.multinomial(probs, num_samples=1)  # 采样
            embed_buf[:bsz, step+1, :] = self.fast_embeddings(next_ids)
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `_FastARAttention` | 类 | SDPA 多头注意力（带 RoPE） |
| `_FastARMLP` | 类 | SiLU 门控 MLP |
| `_FastARDecoderLayer` | 类 | 解码层（注意力 + MLP） |
| `FishSpeechFastARModel` | 类 | 4 层 Transformer |
| `FishSpeechFastAR` | 类 | 完整 Fast AR 封装（含采样） |

## 总结

Fast AR 以极低的计算代价（4 层 * 10 步）预测残差编码，是 Fish Speech 实现高质量多 codebook 音频生成的关键。re-prefill 策略对于序列长度不超过 11 的场景几乎没有性能损失。
