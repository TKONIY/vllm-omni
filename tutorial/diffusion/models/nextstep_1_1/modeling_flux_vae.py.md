# `modeling_flux_vae.py` — NextStep-1.1 自定义 VAE

## 文件概述

该文件实现了 NextStep-1.1 使用的自定义 VAE (Variational Autoencoder)，基于 Flux 风格的 VAE 架构。与 diffusers 标准 VAE 不同，该实现支持 patchify/unpatchify 操作和可选的 encoder norm，并提供 tiling 分块解码支持。

## 关键代码解析

### VAE 参数配置

```python
@dataclass
class AutoEncoderParams:
    resolution: int = 256
    z_channels: int = 16        # latent 通道数
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159
    deterministic: bool = False
    encoder_norm: bool = False   # 可选的 encoder 归一化
    psz: int | None = None       # 可选的 patch size
```

### 编码器和解码器

编码器使用标准的多尺度下采样架构：

```python
class Encoder(nn.Module):
    # conv_in -> 多级 ResBlock + Downsample -> mid (ResBlock + Attn + ResBlock) -> norm + conv_out
```

解码器使用对称的上采样架构：

```python
class Decoder(nn.Module):
    # conv_in -> mid (ResBlock + Attn + ResBlock) -> 多级 ResBlock + Upsample -> norm + conv_out
```

### Patchify/Unpatchify

```python
def patchify(self, img):
    # img: (bsz, C, H, W) -> x: (bsz, C*p^2, H/p, W/p)
    img = torch.einsum("nchpwq->ncpqhw", img.reshape(bsz, c, h_, p, w_, p))

def unpatchify(self, x):
    # x: (bsz, C*p^2, H/p, W/p) -> img: (bsz, C, H, W)
    x = torch.einsum("ncpqhw->nchpwq", x.reshape(bsz, c, p, p, h_, w_))
```

可选的 patch 操作，在 encoder norm 前后对 latent 进行空间重排。

### Tiling 解码

```python
def blend_v(self, a, b, blend_extent):
    # 垂直方向混合两个 tile 的重叠区域
def blend_h(self, a, b, blend_extent):
    # 水平方向混合两个 tile 的重叠区域
```

支持大分辨率图像的分块解码。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `AutoencoderKL` | 类 | 主 VAE 模型 |
| `Encoder` | 类 | VAE 编码器 |
| `Decoder` | 类 | VAE 解码器 |
| `AutoEncoderParams` | dataclass | VAE 参数配置 |
| `AttnBlock` | 类 | 自注意力块 |
| `ResnetBlock` | 类 | 残差块 |

## 与其他模块的关系

- 被 `pipeline_nextstep_1_1.py` 使用进行 latent 编码/解码
- 独立于 diffusers 的 VAE 实现，使用自定义的 `from_pretrained` 方法

## 总结

该文件实现了 NextStep-1.1 专用的 VAE，支持 patchify 操作、encoder norm 和 tiling 解码。相比 diffusers 标准 VAE，该实现更紧凑且针对 NextStep 的 latent 空间进行了定制化。
