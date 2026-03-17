# `fish_speech_slow_ar.py` — Slow AR 语义 token 生成器

## 文件概述

Fish Speech 的核心模型（Stage 0），基于 Qwen3 Transformer 实现文本到语义 token 的自回归生成。集成了多 codebook 嵌入、Fast AR 调用、preprocess/postprocess 钩子，以及 Fish Speech 特有的权重格式转换。

## 关键代码解析

### 权重映射函数

```python
def _remap_fish_speech_weights(weights, ...):
    """Fish Speech → Qwen3 权重名转换"""
    # text_model.model.layers.N.attention.wqkv → 拆分为 q/k/v
    q = tensor[:q_size, :]
    k = tensor[q_size:q_size + kv_size, :]
    v = tensor[q_size + kv_size:, :]
    # w1 → gate_proj, w3 → up_proj, w2 → down_proj
    # attention_norm → input_layernorm, ffn_norm → post_attention_layernorm
    # audio_decoder.* → fast_ar.* (类似转换)
```

### Preprocess/Postprocess 钩子

```python
class FishSpeechSlowARForConditionalGeneration(nn.Module):
    def preprocess(self, input_ids, input_embeds, **info_dict):
        if span_len > 1:  # Prefill
            prompt_embeds = self._build_prefill_embeds(input_ids, info_dict)
            # 语音克隆: 在语义 token 位置添加 codebook 嵌入
            for cb_idx in range(num_codebooks):
                code_with_offset = code + cb_idx * codebook_size
                emb = self.codebook_embeddings(code_with_offset)
                codebook_sum[0, pos, :] += emb
        else:  # Decode
            # 保存 last_slow_ar_hidden 用于 Fast AR
            return input_ids, inputs_embeds_out, {"mtp_inputs": (...)}

    def postprocess(self, hidden_states):
        last = hidden_states[-1, :].detach().to("cpu")
        return {"last_slow_ar_hidden": last}
```

### GPU-side Fast AR 快速路径 (talker_mtp)

```python
def talker_mtp(self, input_ids, input_embeds, last_talker_hidden, text_step):
    """在 GPU 上运行 Fast AR，预测残差 codebook 编码"""
    audio_codes = self.fast_ar(slow_ar_hidden=past_hidden, semantic_token_id=...)
    # 将 codebook 嵌入加到输入嵌入上
    inputs_embeds_out[b] = (embed + codebook_sum) / sqrt(num_codebooks + 1)
    return inputs_embeds_out, audio_codes
```

### RoPE 风格修正

```python
def _fix_rope_style(self):
    """将 NeoX 风格 RoPE 替换为交错（GPT-J）风格"""
    for layer in self.model.layers:
        attn.rotary_emb = get_rope(head_dim, is_neox_style=False, ...)
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `_remap_fish_speech_weights()` | 函数 | 权重名/形状转换 |
| `FishSpeechSlowARForConditionalGeneration` | 类 | 完整 Slow AR 模型 |
| `preprocess()` | 方法 | prefill/decode 预处理钩子 |
| `postprocess()` | 方法 | 保存隐状态给 Fast AR |
| `talker_mtp()` | 方法 | GPU 上的 Fast AR 快速路径 |
| `compute_logits()` | 方法 | logits 计算 + 语义掩码 |

## 总结

Slow AR 是 Fish Speech 最复杂的组件，集成了 Qwen3 推理、多 codebook 嵌入、Fast AR 协调、权重转换、RoPE 修正等多项功能。`talker_mtp` 方法实现了 Slow AR 和 Fast AR 的 GPU 侧联合推理，是性能优化的核心。
