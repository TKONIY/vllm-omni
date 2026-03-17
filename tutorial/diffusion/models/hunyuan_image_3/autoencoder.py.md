# `autoencoder.py` — HunyuanImage3 3D 卷积自编码器

## 文件概述

本文件实现了 HunyuanImage3 使用的基于 3D 卷积的 KL 正则化自编码器 `AutoencoderKLConv3D`。该 VAE 使用 DCAE（Deep Compression AutoEncoder）架构，通过像素重排（pixel shuffle/unshuffle）方式实现空间和时间维度的下采样/上采样，而非传统的步进卷积。支持空间和时间的分块编解码（tiling），以降低显存占用。

## 关键代码解析

### 1. 高效 3D 卷积

```python
class Conv3d(nn.Conv3d):
    def forward(self, input):
        B, C, T, H, W = input.shape
        memory_count = (C * T * H * W) * 2 / 1024**3
        if memory_count > 2:
            n_split = math.ceil(memory_count / 2)
            chunks = torch.chunk(input, chunks=n_split, dim=-3)
            # 对每个 chunk 手动添加时间维度 padding 并逐个卷积
            ...
```

当输入的显存占用超过 2GB 时，自动沿时间维度切分并逐块卷积，避免显存溢出。

### 2. DCAE 下采样

```python
class DownsampleDCAE(nn.Module):
    def forward(self, x):
        h = self.conv(x)
        h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
        return h + shortcut
```

使用像素反洗牌（pixel unshuffle）将空间/时间维度压缩到通道维度，再加上分组平均的残差连接。

### 3. DCAE 上采样

```python
class UpsampleDCAE(nn.Module):
    def forward(self, x):
        h = self.conv(x)
        h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
        shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        return h + shortcut
```

上采样过程与下采样相反，使用像素洗牌（pixel shuffle）恢复空间/时间分辨率。

### 4. 分块编解码（Tiling）

```python
class AutoencoderKLConv3D(ModelMixin, ConfigMixin):
    def spatial_tiled_encode(self, x):
        # 在空间维度进行重叠切分，逐块编码后混合拼接
        for i in range(0, H, overlap_size):
            for j in range(0, W, overlap_size):
                tile = x[:, :, :, i:i+self.tile_sample_min_size, j:j+self.tile_sample_min_size]
                tile = self.encoder(tile)

    def temporal_tiled_encode(self, x):
        # 在时间维度进行重叠切分
        for i in range(0, T, overlap_size):
            tile = x[:, :, i:i+self.tile_sample_min_tsize, :, :]
```

支持空间和时间两个维度的独立或嵌套分块，通过 `blend_h`/`blend_v`/`blend_t` 方法在重叠区域进行线性混合消除拼接伪影。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiagonalGaussianDistribution` | 类 | 对角高斯分布（VAE 潜在分布） |
| `Conv3d` | 类 | 内存优化的 3D 卷积 |
| `AttnBlock` | 类 | 自注意力块（使用 SDPA） |
| `ResnetBlock` | 类 | 3D ResNet 残差块 |
| `DownsampleDCAE` | 类 | DCAE 下采样（像素反洗牌） |
| `UpsampleDCAE` | 类 | DCAE 上采样（像素洗牌） |
| `Encoder` | 类 | 编码器网络 |
| `Decoder` | 类 | 解码器网络 |
| `AutoencoderKLConv3D` | 类 | 完整 KL 自编码器 |

## 与其他模块的关系

- **`pipeline_hunyuan_image_3.py`**：管线中用于图像编码和潜在空间解码
- **diffusers**：继承 `ModelMixin` 和 `ConfigMixin`，兼容 diffusers 权重加载

## 总结

`autoencoder.py` 实现了基于 DCAE 架构的 3D KL 自编码器，通过像素重排实现高效的空间/时间下采样，并支持分块编解码以处理高分辨率或长时间序列的输入。内存优化的 3D 卷积和梯度检查点进一步降低了显存需求。
