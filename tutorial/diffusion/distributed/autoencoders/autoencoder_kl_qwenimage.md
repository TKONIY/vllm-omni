# `autoencoder_kl_qwenimage.py` -- Qwen 图像 VAE 分布式解码

## 文件概述

`autoencoder_kl_qwenimage.py` 实现了 Qwen 图像模型 VAE（`AutoencoderKLQwenImage`）的分布式版本。Qwen 的 VAE 具有 5D 张量（B, C, T, H, W）和逐帧解码的特点，需要特殊的瓦片分割和合并逻辑。

## 关键代码解析

### tile_split -- Qwen 特定的瓦片分割

```python
def tile_split(self, z):
    _, _, num_frames, height, width = z.shape
    sample_height = height * self.spatial_compression_ratio
    sample_width = width * self.spatial_compression_ratio

    tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
    tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

    tiletask_list = []
    for i in range(0, height, tile_latent_stride_height):
        for j in range(0, width, tile_latent_stride_width):
            time_list = []
            for k in range(num_frames):
                self._conv_idx = [0]
                tile = z[:, :, k:k+1, i:i+tile_latent_min_height, j:j+tile_latent_min_width]
                time_list.append(tile)
            tiletask_list.append(TileTask(..., time_list, ...))
```

关键区别：
- 输入是 5D 张量 `(B, C, T, H, W)`
- 每个瓦片包含所有时间帧的列表（`time_list`）
- 分割维度为 `(3, 4)` 即 H 和 W

### tile_exec -- 逐帧解码

```python
def tile_exec(self, task):
    self.clear_cache()
    time = []
    for k in range(len(task.tensor)):
        self._conv_idx = [0]
        tile = self.post_quant_conv(task.tensor[k])
        decoded = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx)
        time.append(decoded)
    result = torch.cat(time, dim=2)
    return result
```

Qwen VAE 使用特征缓存（`feat_cache`）逐帧解码，帧间共享卷积特征。

### tile_merge -- 带混合的瓦片合并

```python
def tile_merge(self, coord_tensor_map, grid_spec):
    for i in range(grid_h):
        for j in range(grid_w):
            tile = coord_tensor_map[(i, j)]
            if i > 0:
                tile = self.blend_v(coord_tensor_map[(i-1, j)], tile, blend_height)
            if j > 0:
                tile = self.blend_h(coord_tensor_map[(i, j-1)], tile, blend_width)
            result_row.append(tile[:, :, :, :tile_sample_stride_height, :tile_sample_stride_width])
    # 最终裁剪到目标尺寸
    dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
```

### tiled_decode -- 入口方法

```python
def tiled_decode(self, z, return_dict=True):
    if not self.is_distributed_enabled():
        return super().tiled_decode(z, return_dict=return_dict)

    result = self.distributed_decoder.execute(
        z, DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
        broadcast_result=True  # 广播结果给所有 rank
    )
```

注意 `broadcast_result=True`，Qwen 需要所有 rank 都有完整解码结果。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DistributedAutoencoderKLQwenImage` | 类 | Qwen 图像 VAE 的分布式版本 |
| `tile_split()` | 方法 | 5D 张量的逐帧瓦片分割 |
| `tile_exec()` | 方法 | 逐帧解码（带特征缓存） |
| `tile_merge()` | 方法 | 瓦片混合合并 |
| `tiled_decode()` | 方法 | 分布式解码入口 |

## 与其他模块的关系

- **distributed_vae_executor.py**: 继承 `DistributedVaeMixin`
- **diffusers AutoencoderKLQwenImage**: 继承原始 Qwen VAE

## 总结

`DistributedAutoencoderKLQwenImage` 针对 Qwen 图像 VAE 的特点（5D 张量、逐帧解码、特征缓存）定制了分布式解码逻辑。每个瓦片包含所有时间帧的数据，解码时逐帧处理并利用帧间特征缓存。`broadcast_result=True` 确保所有 rank 获得完整结果。
