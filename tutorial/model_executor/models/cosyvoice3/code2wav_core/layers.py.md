# `layers.py` — PreLookahead 卷积层

## 文件概述

实现 PreLookaheadLayer，一个用于因果处理的前瞻卷积层。在流式推理中，该层允许模型在不违反因果性的前提下"看"未来几帧的上下文。

## 关键代码解析

```python
class PreLookaheadLayer(nn.Module):
    def __init__(self, in_channels, channels, pre_lookahead_len=1):
        self.conv1 = nn.Conv1d(in_channels, channels,
                               kernel_size=pre_lookahead_len + 1)
        self.conv2 = nn.Conv1d(channels, in_channels, kernel_size=3)

    def forward(self, inputs, context=torch.zeros(0, 0, 0)):
        outputs = inputs.transpose(1, 2)
        if context.size(2) == 0:
            # 非流式：用零填充替代未来上下文
            outputs = F.pad(outputs, (0, self.pre_lookahead_len))
        else:
            # 流式：使用提供的上下文（未来帧）
            outputs = torch.concat([outputs, context], dim=2)
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs = F.pad(outputs, (self.conv2.kernel_size[0] - 1, 0))
        outputs = self.conv2(outputs)
        return outputs + inputs  # 残差连接
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `PreLookaheadLayer` | 类 | 带残差连接的前瞻卷积层 |

## 总结

PreLookaheadLayer 通过两层卷积 + 残差连接，在因果约束下引入有限的未来上下文信息，提升流式推理的音频质量。
