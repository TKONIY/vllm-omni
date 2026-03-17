# `fish_speech_dac_decoder.py` — DAC 解码器（Stage 1）

## 文件概述

Fish Speech 的 Stage 1 解码器，从 codebook 编码解码为 44.1kHz 音频波形。作为 GenerationModelRunner 的组件运行在 GPU 上。

## 关键代码解析

### 输入格式

```
input_ids: [cb0_f0, cb0_f1, ..., cb0_fN, cb1_f0, ..., cb9_fN]
         → reshape 为 [num_codebooks, num_frames]
         → DAC decode → 波形
```

### 前向推理

```python
class FishSpeechDACDecoder(nn.Module):
    def forward(self, input_ids, ...):
        # 1. 将 input_ids 按请求拆分
        request_ids_list = self._split_request_ids(ids)
        # 2. 逐请求重塑为 [q, frames] 并解码
        for codes_qf in valid_codes_qf:
            codes_bqf = codes_qf.unsqueeze(0)  # [1, num_codebooks, frames]
            wav, _ = self._codec.decode(codes_bqf, feature_lengths)
        # 3. 裁剪上下文帧（流式场景）
        if ctx_frames > 0:
            wav = wav[ctx_frames * self._hop_length:]
        return OmniOutput(multimodal_outputs={"model_outputs": audios, "sr": srs})
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `FishSpeechDACDecoder` | 类 | DAC 解码器，codebook codes → 波形 |
| `_ensure_codec_loaded()` | 方法 | 懒加载 codec.pth 到 GPU |
| `_split_request_ids()` | 方法 | 按请求拆分 input_ids |
| `load_weights()` | 方法 | 空实现（codec 从 codec.pth 懒加载） |

## 总结

DAC 解码器是一个无状态的波形生成模块，每次前向调用独立解码。支持流式推理中的左上下文裁剪。
