# `qwen3_tts_code_predictor_vllm.py` — Code Predictor (vLLM 版本)

## 文件概述

本文件实现了 Qwen3-TTS 的 Code Predictor，功能与 Qwen3-Omni 的版本类似，但针对 TTS 场景进行了适配。采用独立的 Transformer 层（使用 SDPA 注意力，无 KV 缓存），通过 re-prefill 策略自回归预测 RVQ 残差层。

## 关键代码解析

### 1. 独立注意力层

```python
class _CodePredictorAttention(nn.Module):
    """使用 F.scaled_dot_product_attention 而非 vLLM 的 paged Attention"""
    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, ...):
        self.qkv_proj = QKVParallelLinear(..., disable_tp=True)  # 禁用张量并行
        self.rotary_emb = get_rope(...)
        self.q_norm = RMSNorm(self.head_dim, ...)
        self.k_norm = RMSNorm(self.head_dim, ...)
```

注意 `disable_tp=True`：Code Predictor 不参与张量并行，因为它是一个轻量级辅助模型。

### 2. 与 Qwen3-Omni 版本的区别

- 使用 `Qwen3TTSTalkerCodePredictorConfig` 而非 `code_predictor_config` 子属性
- 权重名直接匹配检查点结构（无额外映射）
- 支持 `torch.compile` 加速

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `_CodePredictorAttention` | 类 | SDPA 注意力 |
| `_CodePredictorMLP` | 类 | SiLU-gated MLP |
| `_CodePredictorDecoderLayer` | 类 | Transformer 解码层 |
| `Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM` | 类 | Code Predictor 包装器 |

## 与其他模块的关系

- **被引用**: `qwen3_tts_talker.py` 中实例化
- **依赖**: `configuration_qwen3_tts.py` 中的 `Qwen3TTSTalkerCodePredictorConfig`

## 总结

TTS 版本的 Code Predictor 与 Omni 版本共享相同的 re-prefill 策略和优化技术（持久化缓冲区、内联采样、torch.compile），但使用不同的配置类和权重映射。
