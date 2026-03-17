# `schedulers.py` — MammothModa2 自定义调度器

## 文件概述

该文件实现了 MammothModa2 专用的 Flow Matching Euler 离散调度器。与 diffusers 标准调度器不同，该调度器支持动态时间偏移（dynamic time shift），根据 latent token 数量自适应调整时间步分布。

## 关键代码解析

### 动态时间偏移

```python
def set_timesteps(self, num_inference_steps=None, device=None, timesteps=None, num_tokens=None):
    timesteps_np = np.linspace(0, 1, num_inference_steps + 1, dtype=np.float32)[:-1]
    if self.dynamic_time_shift and num_tokens is not None:
        m = np.sqrt(float(num_tokens)) / 40.0
        timesteps_np = timesteps_np / (m - m * timesteps_np + timesteps_np)
```

时间步根据 `num_tokens`（latent 空间的 token 总数）进行非线性变换。token 数越多，时间步偏移越大。

### Euler 步进

```python
def step(self, model_output, timestep, sample, ...):
    t = self._timesteps[self._step_index]
    t_next = self._timesteps[self._step_index + 1]
    prev_sample = sample_fp32 + (t_next - t) * model_output
```

标准的 Euler 方法：`x_{t-1} = x_t + (t_{next} - t) * v_t`。

### 输出格式

```python
@dataclass
class FlowMatchEulerDiscreteSchedulerOutput:
    prev_sample: torch.FloatTensor
```

简化的输出格式，只包含更新后的样本。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `FlowMatchEulerDiscreteScheduler` | 类 | 支持动态时间偏移的 Flow Matching 调度器 |
| `FlowMatchEulerDiscreteSchedulerOutput` | dataclass | 调度器步进输出 |

## 与其他模块的关系

- 被 `pipeline_mammothmoda2_dit.py` 使用作为扩散调度器
- 独立于 diffusers 的调度器系统，因为需要自定义的动态时间偏移逻辑

## 总结

该调度器的核心特色是动态时间偏移机制：根据图像分辨率（latent token 数量）自适应调整时间步分布，高分辨率图像使用更大的偏移。这使得生成不同分辨率图像时都能保持合理的去噪进度。
