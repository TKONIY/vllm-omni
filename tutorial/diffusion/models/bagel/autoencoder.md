# `autoencoder.py` -- Bagel 自编码器 (VAE)

## 文件概述

本文件实现了 Bagel 模型使用的变分自编码器（VAE），包括编码器、解码器、对角高斯采样等核心组件。VAE 负责将图像压缩到潜空间（编码），以及从潜空间重建图像（解码）。代码改编自 Black Forest Labs 的 FLUX 项目。

**文件路径**: `vllm_omni/diffusion/models/bagel/autoencoder.py`

## 关键代码解析

### AutoEncoderParams 配置

```python
@dataclass
class AutoEncoderParams:
    resolution: int       # 输入分辨率（如 256）
    in_channels: int      # 输入通道数（3 = RGB）
    downsample: int       # 下采样倍率（如 8）
    ch: int               # 基础通道数（如 128）
    out_ch: int           # 输出通道数
    ch_mult: list[int]    # 各级通道倍数（如 [1, 2, 4, 4]）
    num_res_blocks: int   # 每级残差块数
    z_channels: int       # 潜空间通道数（如 16）
    scale_factor: float   # 潜空间缩放因子
    shift_factor: float   # 潜空间偏移因子
```

### AttnBlock 自注意力块

```python
class AttnBlock(nn.Module):
    def attention(self, h_: Tensor) -> Tensor:
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        # 使用 PyTorch 内置的 scaled_dot_product_attention
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
```

使用 `einops.rearrange` 进行张量维度变换，利用 PyTorch 原生的高效注意力实现。

### Encoder 编码器

编码器采用经典的 U-Net 风格下采样结构：
1. 卷积输入层
2. 多级下采样块（ResnetBlock + 可选 AttnBlock + Downsample）
3. 中间块（ResnetBlock + AttnBlock + ResnetBlock）
4. 输出层 (GroupNorm + Swish + Conv2d)

### Decoder 解码器

解码器与编码器对称，使用上采样结构：
1. 卷积输入层
2. 中间块
3. 多级上采样块（ResnetBlock + 可选 AttnBlock + Upsample）
4. 输出层

### DiagonalGaussian 采样

```python
class DiagonalGaussian(nn.Module):
    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean
```

从编码器输出的均值和对数方差中进行重参数化采样。

### AutoEncoder 主类

```python
class AutoEncoder(nn.Module):
    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)
```

编码时先进行采样再缩放偏移；解码时进行逆变换后送入解码器。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `AutoEncoder` | nn.Module | VAE 主类，提供 encode/decode 接口 |
| `Encoder` | nn.Module | 图像编码器，多级下采样 |
| `Decoder` | nn.Module | 图像解码器，多级上采样 |
| `DiagonalGaussian` | nn.Module | 对角高斯采样/取均值 |
| `AutoEncoderParams` | dataclass | VAE 配置参数 |
| `AttnBlock` | nn.Module | 自注意力块 |
| `ResnetBlock` | nn.Module | 残差卷积块 |
| `swish` | 函数 | Swish 激活函数 (`x * sigmoid(x)`) |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 被使用 | `pipeline_bagel.py` | Pipeline 中用于图像编解码 |
| 被使用 | `bagel_transformer.py` | Transformer 中用于 VAE 潜空间操作 |
| 来源 | FLUX (Black Forest Labs) | 改编自 FLUX 项目 |

## 总结

Bagel 的 AutoEncoder 是一个标准的 KL-VAE，采用 ResNet + 注意力的多级编解码器架构。`scale_factor` 和 `shift_factor` 对潜空间进行归一化，确保潜变量分布适合后续的扩散过程。默认配置中 `z_channels=16`、`downsample=8`，即将图像压缩为 16 通道、空间分辨率缩小 8 倍的潜空间表示。
