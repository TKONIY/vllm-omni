# `scheduling_helios.py` — Helios 统一调度器

## 文件概述

本文件实现了 Helios 的统一调度器 `HeliosScheduler`，集成了三种去噪算法：**Euler**（一阶离散）、**UniPC**（多阶校正预测）和 **DMD**（分布匹配蒸馏）。调度器还包含金字塔多阶段噪声调度的初始化逻辑，支持动态时间偏移和各阶段独立的 sigma 调度。

## 关键代码解析

### 1. 金字塔多阶段初始化

```python
def init_sigmas_for_each_stage(self):
    self.init_sigmas()
    for i_s in range(stages):
        start_indice = int(stage_range[i_s] * training_steps)
        end_indice = int(stage_range[i_s + 1] * training_steps)
        start_sigma = self.sigmas[start_indice].item()
        # 非首阶段需要校正 sigma（考虑块噪声影响）
        if i_s != 0:
            corrected_sigma = (1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)) * ori_sigma
            start_sigma = 1 - corrected_sigma
```

将完整噪声调度拆分为多个阶段，每个阶段有独立的 `start_sigma` 和 `end_sigma`。非首阶段的 sigma 需要根据 `gamma`（块噪声系数）进行校正。

### 2. Euler 步进

```python
def step_euler(self, model_output, timestep, sample, ...):
    sigma = self.sigmas[self.step_index]
    sigma_next = self.sigmas[self.step_index + 1]
    prev_sample = sample + (sigma_next - sigma) * model_output
    return HeliosSchedulerOutput(prev_sample=prev_sample)
```

最简单的一阶 Euler 离散步进方法。

### 3. UniPC 步进（多阶预测-校正）

```python
def step_unipc(self, model_output, timestep, sample, ...):
    # 校正步：使用先前预测更新当前样本
    if use_corrector:
        sample = self.multistep_uni_c_bh_update(...)

    # 预测步：使用更新后的样本计算下一步
    prev_sample = self.multistep_uni_p_bh_update(...)
    return HeliosSchedulerOutput(prev_sample=prev_sample, ...)
```

UniPC 使用预测-校正（Predictor-Corrector）框架，在每步先用校正器修正当前样本，再用预测器计算下一步样本，能以较少步数获得更高质量结果。

### 4. DMD 步进（蒸馏加速）

```python
def step_dmd(self, model_output, timestep, sample, cur_sampling_step, dmd_noisy_tensor, ...):
    pred_image_or_video = self.convert_flow_pred_to_x0(
        flow_pred=model_output, xt=sample, timestep=..., sigmas=..., timesteps=...
    )
    if cur_sampling_step < len(all_timesteps) - 1:
        prev_sample = self.add_noise(pred_image_or_video, dmd_noisy_tensor, ...)
    else:
        prev_sample = pred_image_or_video
    return HeliosSchedulerOutput(prev_sample=prev_sample)
```

DMD 通过蒸馏训练实现极少步数（如 4-8 步）的快速生成，每步先预测干净样本 x0 再重新加噪。

### 5. 统一 step 入口

```python
def step(self, model_output, timestep, sample, ...):
    if self.config.scheduler_type == "euler":
        return self.step_euler(...)
    elif self.config.scheduler_type == "unipc":
        return self.step_unipc(...)
    elif self.config.scheduler_type == "dmd":
        return self.step_dmd(...)
```

### 6. 动态时间偏移

```python
def time_shift(self, mu, sigma, t):
    if self.config.time_shift_type == "exponential":
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    elif self.config.time_shift_type == "linear":
        return mu / (mu + (1 / t - 1) ** sigma)
```

支持线性和指数两种时间偏移函数，根据图像序列长度自适应调整噪声调度。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `HeliosSchedulerOutput` | 数据类 | 调度器输出：prev_sample、model_outputs 等 |
| `HeliosScheduler` | 类 | 统一调度器（Euler/UniPC/DMD） |
| `init_sigmas_for_each_stage` | 方法 | 初始化金字塔各阶段 sigma |
| `set_timesteps` | 方法 | 设置推理时间步 |
| `step_euler` | 方法 | Euler 一阶步进 |
| `step_unipc` | 方法 | UniPC 多阶预测-校正步进 |
| `step_dmd` | 方法 | DMD 蒸馏步进 |
| `step` | 方法 | 统一步进入口 |
| `convert_model_output` | 方法 | 将模型输出转换为 x0 预测 |
| `multistep_uni_p_bh_update` | 方法 | UniPC 预测器更新 |
| `multistep_uni_c_bh_update` | 方法 | UniPC 校正器更新 |
| `time_shift` | 方法 | 动态时间偏移 |

## 与其他模块的关系

- **`pipeline_helios.py`**：管线在每个去噪步调用 `scheduler.step()`
- **`diffusers.configuration_utils`**：继承 `ConfigMixin` 和 `SchedulerMixin`，兼容 diffusers 生态
- 配置从模型目录的 `scheduler/scheduler_config.json` 加载

## 总结

`scheduling_helios.py` 实现了一个灵活的统一调度器，通过 `scheduler_type` 配置项在 Euler、UniPC 和 DMD 三种算法之间切换。金字塔多阶段初始化为分辨率渐进去噪提供了基础，动态时间偏移则使噪声调度能自适应不同的视频分辨率和序列长度。
