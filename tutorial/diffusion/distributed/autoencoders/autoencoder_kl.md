# `autoencoder_kl.py` -- 标准 AutoencoderKL 分布式解码

## 文件概述

`autoencoder_kl.py` 实现了标准 diffusers `AutoencoderKL` 的分布式版本 `DistributedAutoencoderKL`。它通过继承 `DistributedVaeMixin` 获得分布式执行能力，并实现了瓦片（tile）和补丁（patch）两种分割策略。

## 关键代码解析

### DistributedAutoencoderKL_base -- 基类

```python
class DistributedAutoencoderKL_base(DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()  # 初始化分布式执行器
        return model
```

### tile_split -- 瓦片分割

```python
def tile_split(self, z):
    overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
    blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
    row_limit = self.tile_sample_min_size - blend_extent

    tiletask_list = []
    for i in range(0, z.shape[2], overlap_size):
        for j in range(0, z.shape[3], overlap_size):
            tile = z[:, :, i:i+self.tile_latent_min_size, j:j+self.tile_latent_min_size]
            tiletask_list.append(TileTask(
                len(tiletask_list), (i // overlap_size, j // overlap_size),
                tile, workload=tile.shape[2] * tile.shape[3],
            ))

    grid_spec = GridSpec(split_dims=(2, 3), grid_shape=(...), tile_spec={...})
    return tiletask_list, grid_spec
```

将潜空间按重叠的网格分割为瓦片，每个瓦片的 `workload` 用于负载均衡。

### patch_split -- 补丁分割

```python
def patch_split(self, z):
    """将潜空间按网格分割为带 halo 的补丁。"""
    grid_rows, grid_cols = ...  # 根据并行度计算网格
    for i in range(grid_rows):
        for j in range(grid_cols):
            # 计算核心区域和 halo 区域
            halo = max(halo_base, min(core_h, core_w) // 2)
            tile = z[:, :, ph0:ph1, pw0:pw1]  # 含 halo 的补丁
            tiletask_list.append(TileTask(..., tile, ...))
            halo_size[(i,j)] = {"up": ..., "down": ..., "left": ..., "right": ...}
```

补丁分割使用 halo 扩展确保边界解码质量。

### tile_exec / patch_exec -- 瓦片执行

```python
def tile_exec(self, task):
    tile = task.tensor
    if self.config.use_post_quant_conv:
        tile = self.post_quant_conv(tile)
    return self.decoder(tile)

def patch_exec(self, task):
    return self.tile_exec(task)  # 解码逻辑相同
```

### tile_merge -- 瓦片合并（带混合）

```python
def tile_merge(self, coord_tensor_map, grid_spec):
    for i in range(grid_h):
        for j in range(grid_w):
            tile = coord_tensor_map[(i, j)]
            if i > 0:
                tile = self.blend_v(coord_tensor_map[(i-1, j)], tile, blend_extent)
            if j > 0:
                tile = self.blend_h(coord_tensor_map[(i, j-1)], tile, blend_extent)
            result_row.append(tile[:, :, :row_limit, :row_limit])
```

### patch_merge -- 补丁合并（裁剪 halo）

```python
def patch_merge(self, coord_tensor_map, grid_spec):
    for i in range(grid_h):
        for j in range(grid_w):
            halo = grid_spec.tile_spec["halo_size"][(i, j)]
            scale = grid_spec.tile_spec["scale"]
            core_tile = tile[:, :, halo_up:-halo_down, halo_left:-halo_right]
```

### decode -- 统一入口

```python
def decode(self, z, return_dict=True, *args, **kwargs):
    if not self.is_distributed_enabled():
        return super().decode(z, return_dict=return_dict, *args, **kwargs)

    split, exec, merge = self._strategy_select(z)
    if split is not None:
        result = self.distributed_decoder.execute(
            z, DistributedOperator(split=split, exec=exec, merge=merge),
            broadcast_result=False
        )
```

`_strategy_select` 根据潜空间尺寸自动选择瓦片或补丁策略。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DistributedAutoencoderKL_base` | 类 | 分布式 VAE 基类，实现两种分割策略 |
| `DistributedAutoencoderKL` | 类 | 继承 Diffusers AutoencoderKL 的具体类 |
| `tile_split/exec/merge` | 方法 | 瓦片策略的三步操作 |
| `patch_split/exec/merge` | 方法 | 补丁策略的三步操作 |
| `_strategy_select()` | 方法 | 根据尺寸自动选择策略 |

## 与其他模块的关系

- **distributed_vae_executor.py**: 继承 `DistributedVaeMixin`，使用 `DistributedVaeExecutor` 执行分布式任务
- **diffusers AutoencoderKL**: 继承原始 VAE 功能

## 总结

`DistributedAutoencoderKL` 通过 split-exec-merge 三步操作模式实现了 VAE 的分布式解码。瓦片策略适用于大尺寸输入（使用重叠+混合），补丁策略适用于小尺寸输入（使用 halo 扩展+裁剪）。自动策略选择使得上层管道无需关心具体的分割方式。
