# `cosyvoice3_code2wav.py` — Code2Wav 阶段实现

## 文件概述

实现 CosyVoice3 的 token 到波形转换阶段，包含流匹配解码器（CFM + DiT）和 HiFiGAN 声码器。将离散语音 token 转换为高质量音频波形。

## 关键代码解析

### 组件初始化

```python
class CosyVoice3Code2Wav(nn.Module):
    def __init__(self, config):
        # 1. PreLookahead 卷积层
        pre_lookahead_layer = PreLookaheadLayer(**config.flow["pre_lookahead_layer"])
        # 2. DiT estimator（使用扩散注意力后端）
        estimator = DiT(**decoder_cfg["estimator"])
        # 3. 因果条件流匹配解码器
        self.flow_model = CausalMaskedDiffWithDiT(...)
        # 4. 因果 HiFiGAN 声码器
        self.hift = CausalHiFTGenerator(...)
```

### 前向推理流程

```python
def forward(self, token, prompt_token, prompt_feat, embedding, n_timesteps=10):
    # 1. 说话人嵌入标准化
    embedding = F.normalize(embedding, dim=1)
    embedding = self.spk_embed_affine_layer(embedding)
    # 2. Token 嵌入 + PreLookahead
    token_emb = self.input_embedding(full_token) * mask
    h = self.pre_lookahead_layer(token_emb)
    h = h.repeat_interleave(self.token_mel_ratio, dim=1)
    # 3. 条件构建（参考 mel 特征）
    conds[:, :mel_len1] = prompt_feat
    # 4. 流匹配解码
    feat, _ = self.decoder(mu=h, mask=mel_mask, spks=embedding, cond=conds, n_timesteps=10)
    # 5. 声码器合成
    tts_speech, _ = self.hift.inference(speech_feat=feat[:, :, mel_len1:])
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `CosyVoice3Code2Wav` | 类 | 完整的 token→波形管线 |
| `forward()` | 方法 | 端到端推理：token→mel→wav |
| `load_weights()` | 方法 | 从 flow.pt 和 hift.pt 加载权重 |

## 与其他模块的关系

- 使用 `code2wav_core/cfm.py` 中的 `CausalConditionalCFM` 和 `CausalMaskedDiffWithDiT`
- 使用 `code2wav_core/hifigan.py` 中的 `CausalHiFTGenerator`
- 使用 `code2wav_core/layers.py` 中的 `PreLookaheadLayer`
- DiT estimator 来自 `vllm_omni.diffusion.models.cosyvoice3_audio`

## 总结

Code2Wav 阶段实现了完整的语音合成后端：通过流匹配生成 mel 频谱，再用 HiFiGAN 声码器转换为时域波形。支持流式参数配置（token_overlap_len、mel_cache_len 等）。
