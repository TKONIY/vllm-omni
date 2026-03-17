# `hifigan.py` — HiFiGAN 声码器实现

## 文件概述

实现 HiFTNet（HiFi-GAN + Neural Source Filter + ISTFT）声码器，将 mel 频谱转换为时域音频波形。包含标准版和因果（流式）版两种实现。

## 关键代码解析

### Snake 激活函数

```python
class Snake(nn.Module):
    """基于正弦的周期性激活函数"""
    def forward(self, x):
        # Snake(x) = x + (1/a) * sin^2(a*x)
        x = x + (1.0 / alpha) * pow(sin(x * alpha), 2)
```

### HiFTGenerator 架构

```
mel 特征 → conv_pre → [上采样 + 残差块 + 源信号融合] × N → conv_post → ISTFT → 波形
                                    ↑
F0 预测 → 正弦源生成 → 下采样 ────────┘
```

### 因果版本 CausalHiFTGenerator

```python
class CausalHiFTGenerator(HiFTGenerator):
    """使用因果卷积替换所有标准卷积，支持流式推理"""
    def __init__(self, ...):
        self.conv_pre = CausalConv1d(..., causal_type="right")  # 右因果
        self.ups = [CausalConv1dUpsample(...)]  # 因果上采样
        self.conv_post = CausalConv1d(..., causal_type="left")  # 左因果
```

### 因果卷积模块

```python
class CausalConv1d(torch.nn.Conv1d):
    """因果一维卷积，支持左因果和右因果"""
    def forward(self, x, cache=...):
        if self.causal_type == "left":
            x = torch.concat([cache, x], dim=2)   # 左填充
        else:
            x = torch.concat([x, cache], dim=2)   # 右填充
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `Snake` | 类 | 周期性激活函数 |
| `SineGen` / `SineGen2` | 类 | 正弦波源生成器 |
| `SourceModuleHnNSF` | 类 | 谐波 + 噪声源模块 |
| `ResBlock` | 类 | HiFiGAN 残差块 |
| `HiFTGenerator` | 类 | 标准 HiFT 声码器 |
| `CausalHiFTGenerator` | 类 | 因果（流式）HiFT 声码器 |
| `CausalConv1d` | 类 | 因果一维卷积 |
| `CausalConvRNNF0Predictor` | 类 | 因果 F0 预测器 |

## 总结

HiFiGAN 声码器是 CosyVoice3 音频质量的关键组件。因果变体通过将所有卷积替换为因果版本，实现了流式推理能力，代价是额外的填充计算。
