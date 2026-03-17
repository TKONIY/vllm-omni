# `autoencoder_kl_3d.py` — 3D KL-VAE 自编码器

## 文件概述

实现基于 3D 卷积的 KL-VAE（变分自编码器），用于图像/视频的潜空间编解码。核心特性包括：3D 卷积处理时空维度、DCAE 风格的上下采样、空间和时间 tiling 策略、以及多 GPU 分布式解码。

## 关键代码解析

### 3D 卷积（分块处理大输入）

```python
class Conv3d(nn.Conv3d):
    """对大张量自动分块计算，避免显存溢出"""
    def forward(self, input):
        memory_count = (C * T * H * W) * 2 / 1024**3
        if memory_count > 2:  # 超过 2GB 则分块
            chunks = torch.chunk(input, n_split, dim=-3)
            # 手动填充相邻块的边界
```

### DCAE 风格下采样

```python
class DownsampleDCAE(nn.Module):
    """像素重排 + 卷积下采样"""
    def forward(self, x):
        h = self.conv(x)
        h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w")
        shortcut = rearrange(x, ...)  # 对应的像素重排快捷连接
        shortcut = shortcut.view(B, h.shape[1], group_size, ...).mean(dim=2)
        return h + shortcut
```

### 分布式空间 tiling 解码

```python
class AutoencoderKLConv3D(ModelMixin, ConfigMixin):
    def spatial_tiled_decode(self, z):
        if dist.is_initialized() and world_size > 1:
            # 1. 每个 rank 分配不同的 tile（round-robin）
            my_linear_indices = list(range(rank, total_tiles, world_size))
            # 2. 各 rank 独立解码
            for lin_idx in my_linear_indices:
                dec = self.decoder(tile)
            # 3. all_gather 汇集所有 tile
            dist.all_gather(tiles_gather_list, decoded_tiles)
            # 4. rank 0 重建完整图像（blend + crop）
```

### 编码/解码接口

```python
def encode(self, x):
    # 支持 tiling（temporal → spatial）
    if self.use_temporal_tiling:
        return self.temporal_tiled_encode(x)
    posterior = DiagonalGaussianDistribution(self.encoder(x))
    return AutoencoderKLOutput(latent_dist=posterior)

def decode(self, z):
    decoded = self.decoder(z)  # 或 tiled decode
    return DecoderOutput(sample=decoded)
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `DiagonalGaussianDistribution` | 类 | 对角高斯分布（采样/KL/NLL） |
| `Conv3d` | 类 | 分块 3D 卷积 |
| `ResnetBlock` | 类 | 3D ResNet 残差块 |
| `AttnBlock` | 类 | 3D 自注意力块 |
| `Downsample` / `DownsampleDCAE` | 类 | 下采样模块 |
| `Upsample` / `UpsampleDCAE` | 类 | 上采样模块 |
| `Encoder` | 类 | VAE 编码器 |
| `Decoder` | 类 | VAE 解码器 |
| `AutoencoderKLConv3D` | 类 | 完整 VAE（含 tiling） |
| `load_weights()` | 函数 | 权重加载工具 |

## 总结

3D KL-VAE 是 HunyuanImage3 图像质量的基础。通过 DCAE 风格的像素重排和分布式 tiling 策略，实现了高分辨率图像的高效编解码。该文件约 935 行，是一个完整的工业级 VAE 实现。
