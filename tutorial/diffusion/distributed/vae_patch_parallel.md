# `vae_patch_parallel.py` -- VAE 补丁/瓦片并行解码

## 文件概述

`vae_patch_parallel.py` 实现了 VAE 解码器的分布式并行化，通过将潜空间张量分割为多个瓦片（tile）或补丁（patch），分配给不同 GPU 解码后再拼接结果。这种方式可以显著降低单 GPU 的 VAE 解码内存峰值。

## 关键代码解析

### _distributed_tiled_decode -- 分布式瓦片解码

```python
def _distributed_tiled_decode(*, vae, orig_decode, z, group, vae_patch_parallel_size):
    """当潜空间尺寸大于 tile_latent_min_size 时使用。"""

    # 1. 计算瓦片网格
    overlap_size = int(tile_latent_min_size * (1 - tile_overlap_factor))
    h_starts = list(range(0, z.shape[2], overlap_size))
    w_starts = list(range(0, z.shape[3], overlap_size))

    # 2. 按 rank 分配瓦片（轮询分配，偏移 1 以负载均衡）
    tile_id = 0
    for i in h_starts:
        for j in w_starts:
            tile_rank = (tile_id + 1) % pp_size
            if active and (tile_rank == rank):
                tile = z[:, :, i:i+tile_latent_min_size, j:j+tile_latent_min_size]
                decoded = vae.decoder(tile)
                local_tiles.append(decoded)
            tile_id += 1

    # 3. Gather 所有瓦片到 rank0
    dist.gather(meta_tensor, gather_list=meta_gather, dst=0, group=group)
    dist.gather(tile_tensor, gather_list=tile_gather, dst=0, group=group)

    # 4. Rank0 执行混合拼接
    for i, row in enumerate(rows):
        for j, tile in enumerate(row):
            if i > 0:
                tile = vae.blend_v(rows[i-1][j], tile, blend_extent)
            if j > 0:
                tile = vae.blend_h(row[j-1], tile, blend_extent)
```

### _distributed_patch_decode -- 分布式补丁解码

```python
def _distributed_patch_decode(*, vae, orig_decode, z, group, vae_patch_parallel_size, vae_scale_factor):
    """当潜空间尺寸不触发瓦片解码时使用。"""

    # 1. 将潜空间分割为网格
    grid_rows, grid_cols = _factor_pp_grid(pp_size)

    # 2. 每个 rank 解码一个带 halo 的补丁
    halo = max(halo_base, min(core_h, core_w) // 2)
    tile = z[:, :, ph0:ph1, pw0:pw1]  # 带 halo 的补丁
    decoded = vae.decoder(tile)

    # 3. 裁剪到核心区域（移除 halo）
    local_core = decoded[:, :, ch0:ch1, cw0:cw1]

    # 4. Gather 到 rank0 并拼接
    dist.gather(padded, gather_list=block_gather, dst=0, group=group)
    # rank0 拼接所有核心块
    for patch_idx in range(pp_size):
        out[:, :, h0*scale:h1*scale, w0*scale:w1*scale] = tile[:, :, :ph, :pw]
```

补丁解码使用 halo（光环区域）确保边界处解码质量，裁剪时只保留核心区域。

### VaePatchParallelism -- 包装类

```python
class VaePatchParallelism:
    """将 vae.decode 替换为分布式版本的包装器。"""

    def __init__(self, vae, *, vae_patch_parallel_size, group_getter):
        self._orig_decode = vae.decode
        self._vae_scale_factor = _get_vae_spatial_scale_factor(vae)

    def decode(self, z, return_dict=True, *args, **kwargs):
        # 根据条件选择策略
        if should_tile:  # 尺寸大于 tile_latent_min_size
            decoded = _distributed_tiled_decode(...)
        else:            # 尺寸较小
            decoded = _distributed_patch_decode(...)

        # rank0 的结果广播给所有 rank
        dist.broadcast(decoded, src=0, group=group)
        return (decoded,) if not return_dict else DecoderOutput(sample=decoded)
```

### maybe_wrap_vae_decode_with_patch_parallelism -- 便捷包装函数

```python
def maybe_wrap_vae_decode_with_patch_parallelism(pipeline, *, vae_patch_parallel_size, group_getter):
    """将管道的 vae.decode 替换为分布式版本。"""
    vae = getattr(pipeline, "vae", None)
    if vae is None or not hasattr(vae, "decode"):
        return
    if getattr(vae, "_vllm_vae_patch_parallel_installed", False):
        return  # 防止重复安装

    wrapper = VaePatchParallelism(vae, vae_patch_parallel_size=vae_patch_parallel_size, ...)
    vae.decode = wrapper.decode  # 猴子补丁替换
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `_distributed_tiled_decode()` | 函数 | 大尺寸潜空间的分布式瓦片解码 |
| `_distributed_patch_decode()` | 函数 | 小尺寸潜空间的分布式补丁解码 |
| `VaePatchParallelism` | 类 | VAE decode 的分布式包装器 |
| `maybe_wrap_vae_decode_with_patch_parallelism()` | 函数 | 便捷安装函数 |
| `_factor_pp_grid()` | 函数 | 将并行度分解为行列网格 |
| `_get_vae_spatial_scale_factor()` | 函数 | 获取 VAE 的空间缩放因子 |

## 与其他模块的关系

- **parallel_state.py**: 通过 `group_getter` 回调获取通信组
- **管道类**: 通过 `maybe_wrap_vae_decode_with_patch_parallelism` 自动安装
- **autoencoders/**: 提供了更高级的 VAE 分布式支持（继承式而非猴子补丁式）

## 总结

`vae_patch_parallel.py` 提供了两种 VAE 分布式解码策略：
1. **瓦片解码**: 适用于大尺寸输入，使用重叠瓦片 + 混合拼接
2. **补丁解码**: 适用于小尺寸输入，使用 halo 扩展 + 裁剪拼接

两种策略都将工作分散到多个 GPU 上，在 rank0 上完成最终拼接。通过猴子补丁方式替换 `vae.decode`，管道代码无需任何修改即可获得分布式 VAE 解码能力。
