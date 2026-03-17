# `dac_utils.py` — DAC codec 构建工具

## 文件概述

共享的 DAC codec 模型构建函数，被编码器（语音克隆）和解码器（Stage 1）共同使用。定义了 DAC 的关键常量和完整的模型架构。

## 关键代码解析

```python
DAC_SAMPLE_RATE = 44100
DAC_HOP_LENGTH = 2048   # 512 * 4
DAC_NUM_CODEBOOKS = 10  # 1 semantic + 9 residual

def build_dac_codec() -> nn.Module:
    """构建 DAC codec（未初始化权重）"""
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024, n_codebooks=9, codebook_size=1024,
        semantic_codebook_size=4096,  # 语义 codebook 更大
    )
    codec = DAC(
        encoder_rates=[2, 4, 8, 8],      # 编码器下采样率
        decoder_rates=[8, 8, 4, 2],      # 解码器上采样率
        quantizer=quantizer,
        sample_rate=44100, causal=True,
    )
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `build_dac_codec()` | 函数 | 构建完整的 DAC 模型实例 |

## 总结

DAC codec 使用分层下采样 + 残差向量量化 + 分层上采样的对称架构，语义 codebook (4096 词条) 比残差 codebook (1024 词条) 更大。
