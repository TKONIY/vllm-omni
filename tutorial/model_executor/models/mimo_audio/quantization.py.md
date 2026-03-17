# `quantization.py` — 残差向量量化（RVQ）

## 文件概述

实现残差向量量化（Residual Vector Quantization）算法，是 MiMo-Audio Tokenizer 将连续音频特征转换为离散编码的核心组件。

## 关键代码解析

### EuclideanCodebook

```python
class EuclideanCodebook(nn.Module):
    """基于欧氏距离的码本"""
    def quantize(self, x):
        """最近邻搜索"""
        dist = -(x.pow(2).sum(1,keepdim=True) - 2*x@embed + embed.pow(2).sum(0))
        return dist.max(dim=-1).indices

    def forward(self, x):
        # K-means 初始化（首次训练 batch）
        self.init_embed_(x)
        embed_ind = self.quantize(x)
        quantize = self.dequantize(embed_ind)
        # EMA 更新码本
        if self.training:
            self.expire_codes_(x)  # 过期死码
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
```

### ResidualVectorQuantization

```python
class ResidualVectorQuantization(nn.Module):
    """多层残差 VQ：逐层量化残差"""
    def forward(self, x, n_q=None):
        residual = x
        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized      # 更新残差
            quantized_out += quantized           # 累加量化结果

    def encode(self, x, n_q=None, st=None):
        """编码：返回所有层的索引"""
        for layer in self.layers[st:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
        return torch.stack(all_indices)

    def decode(self, q_indices, st=0):
        """解码：将索引转回连续表示"""
        for i, indices in enumerate(q_indices):
            quantized_out += self.layers[st + i].decode(indices)
```

### ResidualVectorQuantizer（高层封装）

```python
class ResidualVectorQuantizer(nn.Module):
    """RVQ 的高层封装，提供 encode/decode 接口"""
    def __init__(self, dimension=256, n_q=8, bins=1024):
        self.vq = ResidualVectorQuantization(
            dim=dimension, codebook_size=bins, num_quantizers=n_q)
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `EuclideanCodebook` | 类 | 欧氏距离码本（EMA 更新） |
| `VectorQuantization` | 类 | 单层 VQ（含投影） |
| `ResidualVectorQuantization` | 类 | 多层残差 VQ |
| `ResidualVectorQuantizer` | 类 | RVQ 高层封装 |
| `kmeans()` | 函数 | K-means 初始化 |
| `ema_inplace()` | 函数 | EMA 原地更新 |

## 与其他模块的关系

- 被 `modeling_audio_tokenizer.py` 的 `AudioEncoder` 使用
- 支持分布式训练（`dist.all_reduce`）

## 总结

RVQ 是神经音频编码器的核心量化模块，通过逐层量化残差的方式实现高压缩比编码。MiMo-Audio 默认使用 12 层量化器，码本大小可按通道配置（如 1024）。
