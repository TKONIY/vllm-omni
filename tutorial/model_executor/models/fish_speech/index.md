# Fish Speech 模型模块架构概览

## 模块简介

Fish Speech S2 Pro 是一个基于 Qwen3 的端到端语音合成模型，采用 Slow AR + Fast AR + DAC 解码的三阶段架构。Slow AR 生成语义 token，Fast AR 预测残差 codebook 编码，DAC 解码器将编码转换为 44.1kHz 波形。

## 架构图

```
文本输入 (+ 可选参考音频)
       │
       ▼
┌──────────────────────────────────┐
│  FishSpeechSlowARForConditional  │  ← Stage 0: Slow AR
│  Generation                      │
│  ├── Qwen3Model (model)         │  ← 36 层 Transformer
│  ├── codebook_embeddings         │  ← 多 codebook 嵌入表
│  ├── lm_head                     │  ← 语义 token 预测头
│  ├── _semantic_allowed_mask      │  ← 语义 token 掩码
│  └── FishSpeechFastAR (fast_ar)  │  ← 内嵌 Fast AR
│      ├── FishSpeechFastARModel   │  ← 4 层 Transformer
│      ├── fast_embeddings          │  ← Fast AR codebook 嵌入
│      ├── fast_output              │  ← 残差 code 预测头
│      └── fast_norm                │
└──────────┬───────────────────────┘
           │ [semantic_tokens + codebook_codes]
           ▼
┌──────────────────────────────────┐
│  FishSpeechDACDecoder            │  ← Stage 1: DAC 解码
│  ├── DAC codec                   │  ← 从 codec.pth 加载
│  │   ├── Encoder                 │
│  │   ├── DownsampleResidualVQ    │
│  │   └── Decoder                 │
│  └── decode(codes) → waveform    │
└──────────────────────────────────┘
           │
           ▼
      44.1kHz 音频波形
```

## 文件列表

| 文件 | 描述 |
|------|------|
| `__init__.py` | 空模块入口 |
| `configuration_fish_speech.py` | 三层配置类（Slow AR / Fast AR / 顶层） |
| `dac_encoder.py` | DAC 编码器（参考音频语音克隆） |
| `dac_utils.py` | DAC codec 模型构建工具 |
| `fish_speech_dac_decoder.py` | DAC 解码器（Stage 1）|
| `fish_speech_fast_ar.py` | Fast AR 残差 codebook 预测器 |
| `fish_speech_slow_ar.py` | Slow AR 语义 token 生成器（Stage 0）|

## 核心设计思想

1. **Slow-Fast AR 分离**：Slow AR（36 层 Qwen3）生成语义 token，每步触发 Fast AR（4 层）预测 10 个 codebook 的残差编码。Fast AR 使用 re-prefill（无 KV cache）策略，以 O(T^2) 注意力换取零缓存管理开销。

2. **权重映射**：Fish Speech 原始权重使用自定义命名（wqkv、wo、w1/w2/w3），通过 `_remap_fish_speech_weights` 函数转换为 Qwen3 格式。

3. **交错 RoPE**：Fish Speech 使用 GPT-J 风格（交错）RoPE 而非 NeoX 风格，在加载后通过 `_fix_rope_style` 修正。

4. **语义掩码**：Slow AR 在 compute_logits 中应用语义 token 掩码，确保只采样合法的语义 token 范围。
