# `scheduling_flow_unipc_multistep.py` -- Flow Matching UniPC 多步调度器

## 文件概述

本文件实现了 `FlowUniPCMultistepScheduler`，一个面向 Flow Matching 扩散模型的无训练快速采样框架。它将 UniPC（Unified Predictor-Corrector）算法适配到 Flow Matching 范式中，在保持生成质量的同时可用更少的步数完成采样（通常 20-30 步即可替代 Euler 方法的 40-50 步）。

**文件路径**: `vllm_omni/diffusion/models/schedulers/scheduling_flow_unipc_multistep.py`

## 关键代码解析

### 初始化与 Sigma 调度

```python
# 线性初始化 alpha/sigma 调度
alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
sigmas = 1.0 - alphas
sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

# 应用 timestep shifting
if not use_dynamic_shifting:
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
```

Shift 参数控制噪声调度的偏移量。例如，Wan2.2 模型在 720p 时使用 `shift=5.0`，480p 时使用 `shift=12.0`。

### UniPC 预测器（Predictor）

```python
def multistep_uni_p_bh_update(self, model_output, *, sample, order):
```

UniP 预测器使用 B(h) 版本的多步更新算法。核心步骤：
1. 计算 lambda 空间中的步长 h
2. 利用历史模型输出构建多项式近似
3. 使用 `torch.linalg.solve` 求解系数
4. 执行更新 `x_t = sigma_t/sigma_s0 * x - alpha_t * h_phi_1 * m0`

### UniPC 校正器（Corrector）

```python
def multistep_uni_c_bh_update(self, this_model_output, *, last_sample, this_sample, order):
```

UniC 校正器在预测步骤后进行修正，有效提升一阶精度。使用与预测器相似的系数计算方式，但额外利用当前时间步的模型输出进行修正。

### Step 方法（核心采样步骤）

```python
def step(self, model_output, timestep, sample, return_dict=True, generator=None):
    # 1. 转换模型输出
    model_output_convert = self.convert_model_output(model_output, sample=sample)
    # 2. 可选的校正步骤
    if use_corrector:
        sample = self.multistep_uni_c_bh_update(...)
    # 3. 更新历史记录
    self.model_outputs[-1] = model_output_convert
    # 4. 预测步骤
    prev_sample = self.multistep_uni_p_bh_update(...)
    return SchedulerOutput(prev_sample=prev_sample)
```

### 动态阈值

```python
def _threshold_sample(self, sample):
    s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
    s = torch.clamp(s, min=1, max=self.config.sample_max_value)
    sample = torch.clamp(sample, -s, s) / s
```

来自 Imagen 论文的动态阈值方法，用于防止像素饱和。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `FlowUniPCMultistepScheduler` | 调度器类 | 继承 `SchedulerMixin`、`ConfigMixin`、`BaseScheduler` |
| `step()` | 方法 | 核心采样步骤：校正 + 预测 |
| `set_timesteps()` | 方法 | 设置推理时间步 |
| `convert_model_output()` | 方法 | 转换模型输出格式 |
| `multistep_uni_p_bh_update()` | 方法 | UniP 预测器更新 |
| `multistep_uni_c_bh_update()` | 方法 | UniC 校正器更新 |
| `add_noise()` | 方法 | 添加噪声到原始样本 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 继承 | `schedulers/base.py` | 继承 `BaseScheduler` 基类 |
| 继承 | `diffusers` | 继承 `SchedulerMixin` 和 `ConfigMixin` |
| 使用方 | Wan2.2 等 Pipeline | 用作去噪循环中的采样调度器 |

## 总结

`FlowUniPCMultistepScheduler` 是专为 Flow Matching 模型设计的高阶求解器。通过 UniPC 的预测-校正框架，它能在较少的采样步数内达到与低阶方法（如 Euler）相同的生成质量。关键配置包括 `solver_order`（建议有条件指导时用 2，无条件时用 3）、`shift`（根据分辨率调整）和 `solver_type`（`bh1` 或 `bh2`）。
