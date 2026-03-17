# `cfg_parallel.py` -- Classifier-Free Guidance 并行支持

## 文件概述

`cfg_parallel.py` 实现了 `CFGParallelMixin` 混入类，为扩散管道提供 Classifier-Free Guidance（CFG）的并行计算支持。CFG 是扩散模型中常用的技术，需要对同一输入分别进行有条件和无条件推理，再将结果组合。该模块支持两种模式：顺序 CFG 和并行 CFG。

## 关键代码解析

### predict_noise_maybe_with_cfg -- 核心 CFG 调度

```python
def predict_noise_maybe_with_cfg(
    self, do_true_cfg, true_cfg_scale,
    positive_kwargs, negative_kwargs,
    cfg_normalize=True, output_slice=None,
) -> torch.Tensor | None:
    if do_true_cfg:
        cfg_parallel_ready = get_classifier_free_guidance_world_size() > 1
        if cfg_parallel_ready:
            # 并行模式: rank0 计算正向, rank1 计算负向
            cfg_group = get_cfg_group()
            cfg_rank = get_classifier_free_guidance_rank()
            if cfg_rank == 0:
                local_pred = self.predict_noise(**positive_kwargs)
            else:
                local_pred = self.predict_noise(**negative_kwargs)

            # 聚合结果
            gathered = cfg_group.all_gather(local_pred, separate_tensors=True)
            if cfg_rank == 0:
                noise_pred = gathered[0]
                neg_noise_pred = gathered[1]
                noise_pred = self.combine_cfg_noise(noise_pred, neg_noise_pred, true_cfg_scale, cfg_normalize)
                return noise_pred
            else:
                return None  # 非 rank0 返回 None
        else:
            # 顺序模式: 依次计算正向和负向
            positive_noise_pred = self.predict_noise(**positive_kwargs)
            negative_noise_pred = self.predict_noise(**negative_kwargs)
            noise_pred = self.combine_cfg_noise(positive_noise_pred, negative_noise_pred, ...)
            return noise_pred
    else:
        return self.predict_noise(**positive_kwargs)
```

两种 CFG 模式：
- **顺序模式** (`cfg_world_size == 1`): 单 GPU 上依次计算正向和负向预测
- **并行模式** (`cfg_world_size > 1`): rank0 计算正向，rank1 计算负向，通过 all_gather 聚合

### scheduler_step_maybe_with_cfg -- 调度器同步

```python
def scheduler_step_maybe_with_cfg(self, noise_pred, t, latents, do_true_cfg):
    cfg_parallel_ready = do_true_cfg and get_classifier_free_guidance_world_size() > 1
    if cfg_parallel_ready:
        cfg_group = get_cfg_group()
        cfg_rank = get_classifier_free_guidance_rank()
        if cfg_rank == 0:
            latents = self.scheduler_step(noise_pred, t, latents)
        latents = latents.contiguous()
        cfg_group.broadcast(latents, src=0)  # rank0 广播结果
    else:
        latents = self.scheduler_step(noise_pred, t, latents)
    return latents
```

在 CFG 并行模式下，只有 rank0 执行调度器步骤，然后将更新后的 latents 广播给所有 rank。

### combine_cfg_noise -- CFG 组合函数

```python
def combine_cfg_noise(self, noise_pred, neg_noise_pred, true_cfg_scale, cfg_normalize=False):
    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
    if cfg_normalize:
        noise_pred = self.cfg_normalize_function(noise_pred, comb_pred)
    else:
        noise_pred = comb_pred
    return noise_pred
```

标准 CFG 公式：`result = uncond + scale * (cond - uncond)`，可选归一化处理。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `CFGParallelMixin` | Mixin 类 | 为管道提供 CFG 并行能力 |
| `predict_noise_maybe_with_cfg()` | 方法 | CFG 调度核心：自动选择顺序/并行模式 |
| `scheduler_step_maybe_with_cfg()` | 方法 | 带 CFG 同步的调度器步进 |
| `combine_cfg_noise()` | 方法 | CFG 公式组合 |
| `cfg_normalize_function()` | 方法 | CFG 结果归一化 |
| `predict_noise()` | 方法 | transformer 前向传播（子类可覆写） |
| `diffuse()` | 抽象方法 | 子类必须实现的扩散循环 |

## 与其他模块的关系

- **parallel_state.py**: 使用 `get_cfg_group()`, `get_classifier_free_guidance_rank()`, `get_classifier_free_guidance_world_size()`
- **group_coordinator.py**: 通过 `GroupCoordinator.all_gather()` 和 `broadcast()` 进行通信
- **管道类**: 各模型管道继承此 Mixin 获得 CFG 并行能力

## 总结

`CFGParallelMixin` 是一个精心设计的混入类，将 CFG 的并行/顺序模式自动化处理。管道子类只需调用 `predict_noise_maybe_with_cfg()` 和 `scheduler_step_maybe_with_cfg()`，即可透明地获得 CFG 并行加速。该设计将 CFG 计算量减半（每个 GPU 只计算一个分支），并通过广播机制保持所有 rank 的 latents 同步。
