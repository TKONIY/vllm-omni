# autoencoders/ -- 分布式 VAE 自编码器索引

## 模块概述

`autoencoders/` 子模块通过继承方式为 diffusers 的各种 VAE 自编码器提供分布式解码支持。与 `vae_patch_parallel.py` 的猴子补丁方式不同，这里采用面向对象的继承设计，提供更灵活的 split-exec-merge 控制。

## 核心设计: split-exec-merge 模式

```
         split(z)                exec(task)              merge(results)
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │ 将潜空间 z 分割为 │   │ 每个 rank 解码   │   │ rank0 将解码结果 │
    │ TileTask 列表     │──>│ 分配到的瓦片     │──>│ 混合拼接为完整图 │
    │ + GridSpec 元信息 │   │                  │   │                  │
    └──────────────────┘   └──────────────────┘   └──────────────────┘
```

## 框架层级

```
DistributedVaeMixin (混入)
    └── init_distributed() -> DistributedVaeExecutor
                                  │
                                  ├── _balance_tasks()   负载均衡
                                  ├── execute()          分布式执行
                                  │     ├── split -> 分割
                                  │     ├── exec  -> 本地解码
                                  │     ├── gather -> 聚合到 rank0
                                  │     └── merge  -> 拼接
                                  └── _sync_final_result() 可选广播

DistributedAutoencoderKL_base (基类)
    ├── tile_split/exec/merge    瓦片策略（重叠+混合）
    └── patch_split/exec/merge   补丁策略（halo+裁剪）

DistributedAutoencoderKL          <- Diffusers AutoencoderKL
DistributedAutoencoderKLQwenImage <- Diffusers AutoencoderKLQwenImage
DistributedAutoencoderKLWan       <- Diffusers AutoencoderKLWan
```

## 文件索引

| 文件 | 说明 | 核心组件 |
|------|------|---------|
| [`distributed_vae_executor.py`](distributed_vae_executor.md) | 执行框架 | `DistributedVaeExecutor`, `DistributedVaeMixin`, `GridSpec`, `TileTask` |
| [`autoencoder_kl.py`](autoencoder_kl.md) | 标准 VAE | `DistributedAutoencoderKL`（瓦片+补丁策略） |
| [`autoencoder_kl_qwenimage.py`](autoencoder_kl_qwenimage.md) | Qwen 图像 VAE | `DistributedAutoencoderKLQwenImage`（5D 逐帧解码） |
| [`autoencoder_kl_wan.py`](autoencoder_kl_wan.md) | Wan 视频 VAE | `DistributedAutoencoderKLWan`（patch_size + first_chunk） |

## 支持的 VAE 模型

| VAE 类型 | 分布式类 | 张量维度 | 策略 | 特殊处理 |
|---------|---------|---------|------|---------|
| AutoencoderKL | DistributedAutoencoderKL | 4D (B,C,H,W) | 瓦片+补丁 | 自动策略选择 |
| AutoencoderKLQwenImage | DistributedAutoencoderKLQwenImage | 5D (B,C,T,H,W) | 瓦片 | 逐帧解码+特征缓存 |
| AutoencoderKLWan | DistributedAutoencoderKLWan | 5D (B,C,T,H,W) | 瓦片 | patch_size+unpatchify |

## 添加新 VAE 的分布式支持

```python
class DistributedMyVAE(MyVAE, DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def tile_split(self, z):
        # 分割逻辑
        return tiletask_list, grid_spec

    def tile_exec(self, task):
        # 解码单个瓦片
        return decoded_tile

    def tile_merge(self, coord_tensor_map, grid_spec):
        # 合并所有解码后的瓦片
        return full_image

    def tiled_decode(self, z, return_dict=True):
        if not self.is_distributed_enabled():
            return super().tiled_decode(z, return_dict=return_dict)
        result = self.distributed_decoder.execute(
            z, DistributedOperator(self.tile_split, self.tile_exec, self.tile_merge))
        return DecoderOutput(sample=result)
```

## 与 vae_patch_parallel.py 的对比

| 方面 | vae_patch_parallel.py | autoencoders/*.py |
|------|----------------------|-------------------|
| 方式 | 猴子补丁 | 继承 |
| 灵活性 | 通用但有限 | 每个 VAE 完全自定义 |
| 负载均衡 | 轮询分配 | 贪心负载均衡 |
| 适用场景 | 快速适配 | 精细控制 |
| 维护性 | 无需修改 VAE 代码 | 需要创建子类 |
