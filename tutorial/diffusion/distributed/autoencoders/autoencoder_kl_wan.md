# `autoencoder_kl_wan.py` -- Wan 视频 VAE 分布式解码

## 文件概述

`autoencoder_kl_wan.py` 实现了 Wan 视频模型 VAE（`AutoencoderKLWan`）的分布式版本。Wan VAE 与 Qwen VAE 类似处理 5D 张量，但增加了对 `patch_size` 的支持（用于视频的时空分块）和 `first_chunk` 标志。

## 关键代码解析

### tile_split -- Wan 特定的瓦片分割

```python
def tile_split(self, z):
    _, _, num_frames, height, width = z.shape
    # 考虑 patch_size 对 sample 尺寸的影响
    if self.config.patch_size is not None:
        sample_height = sample_height // self.config.patch_size
        sample_width = sample_width // self.config.patch_size
        tile_sample_stride_height = tile_sample_stride_height // self.config.patch_size
        blend_height = self.tile_sample_min_height // self.config.patch_size - tile_sample_stride_height
```

关键区别：当模型配置了 `patch_size` 时，输出空间的尺寸计算和混合参数需要相应调整。

### tile_exec -- 逐帧解码（带 first_chunk 标志）

```python
def tile_exec(self, task):
    self.clear_cache()
    time = []
    for k in range(len(task.tensor)):
        self._conv_idx = [0]
        tile = self.post_quant_conv(task.tensor[k])
        decoded = self.decoder(tile, feat_cache=self._feat_map,
                              feat_idx=self._conv_idx, first_chunk=(k == 0))
        time.append(decoded)
    result = torch.cat(time, dim=2)
    return result
```

`first_chunk=(k==0)` 告诉解码器第一帧需要特殊处理（如初始化因果卷积的缓存）。

### tile_merge -- 带 unpatchify 的合并

```python
def tile_merge(self, coord_tensor_map, grid_spec):
    # 标准混合合并
    for i in range(grid_h):
        for j in range(grid_w):
            tile = coord_tensor_map[(i, j)]
            if i > 0:
                tile = self.blend_v(...)
            if j > 0:
                tile = self.blend_h(...)
            result_row.append(tile[:, :, :, :stride_h, :stride_w])

    dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

    # 特有: unpatchify 还原
    if self.config.patch_size is not None:
        dec = unpatchify(dec, patch_size=self.config.patch_size)

    dec = torch.clamp(dec, min=-1.0, max=1.0)  # 值域裁剪
    return dec
```

Wan VAE 在合并后可能需要 `unpatchify` 将分块后的输出还原为完整帧。最终值域裁剪到 `[-1, 1]`。

### tiled_decode -- 入口方法

```python
def tiled_decode(self, z, return_dict=True):
    if not self.is_distributed_enabled():
        return super().tiled_decode(z, return_dict=return_dict)

    result = self.distributed_decoder.execute(
        z, DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
        broadcast_result=False  # 只需 rank0 有结果
    )
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DistributedAutoencoderKLWan` | 类 | Wan 视频 VAE 的分布式版本 |
| `tile_split()` | 方法 | 5D 张量分割（考虑 patch_size） |
| `tile_exec()` | 方法 | 逐帧解码（带 first_chunk 标志） |
| `tile_merge()` | 方法 | 混合合并 + unpatchify + 值域裁剪 |

## 与其他模块的关系

- **distributed_vae_executor.py**: 继承 `DistributedVaeMixin`
- **diffusers AutoencoderKLWan**: 继承原始 Wan VAE
- **diffusers unpatchify**: 使用 `unpatchify` 还原分块输出

## 总结

`DistributedAutoencoderKLWan` 针对 Wan 视频模型 VAE 的两个特点进行了定制：
1. **patch_size 支持**: 在分割参数计算和合并后的 unpatchify 中处理时空分块
2. **first_chunk 标志**: 解码第一帧时初始化因果卷积缓存

`broadcast_result=False` 表示只有 rank0 需要最终结果，这与 Qwen VAE 的 `broadcast_result=True` 形成对比，反映了不同管道对分布式解码结果的不同需求。
