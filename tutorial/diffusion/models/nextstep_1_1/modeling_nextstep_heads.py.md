# `modeling_nextstep_heads.py` — NextStep-1.1 Flow Matching Head

## 文件概述

该文件实现了 NextStep-1.1 的 Flow Matching 采样头，包括 ResBlock、FinalLayer、TimestepEmbedder 和核心的 SimpleMLPAdaLN 网络。该头网络体积较小（dim=1536, 12层），不需要张量并行。

## 关键代码解析

### ResBlock（AdaLN 调制残差块）

```python
class ResBlock(nn.Module):
    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h
```

### SimpleMLPAdaLN（核心网络）

```python
class SimpleMLPAdaLN(nn.Module):
    def forward(self, x, t, c):
        x = self.input_proj(x)       # 投影输入
        t = self.time_embed(t)        # 时间步嵌入
        c = self.cond_embed(c)        # 条件嵌入
        y = t + c                     # 合并条件
        for block in self.res_blocks:
            x = block(x, y)           # AdaLN 调制
        return self.final_layer(x, y)
```

### FlowMatchingHead（采样器）

```python
class FlowMatchingHead(nn.Module):
    @torch.no_grad()
    def sample(self, c, cfg=1.0, cfg_img=1.0, cfg_mult=None, ...):
        noise = randn_tensor((c.shape[0] // cfg_mult, self.input_dim), noise_repeat, self.device)
        x = noise
        for ti, tj in zip(timesteps[:-1], timesteps[1:]):
            velocity = self.net(combined, ti.expand(...), c)
            velocity = self.get_velocity_from_cfg(velocity, cfg, cfg_img, cfg_mult)
            score = self.get_score_from_velocity(velocity, x, ti)
            drift = velocity + (1 - t) * score
            x = x + drift * dt + sqrt(2*(1-t)) * dw  # SDE 采样
```

使用随机微分方程（SDE）采样器，支持 2-branch 和 3-branch CFG。

### CFG 支持

```python
def get_velocity_from_cfg(self, velocity, cfg, cfg_img, cfg_mult):
    if cfg_mult == 2:  # 文本 CFG
        cond_v, uncond_v = velocity.chunk(2)
        velocity = uncond_v + cfg * (cond_v - uncond_v)
    elif cfg_mult == 3:  # 文本 + 图像 CFG
        cond_v, uncond_v1, uncond_v2 = velocity.chunk(3)
        velocity = uncond_v2 + cfg_img*(uncond_v1-uncond_v2) + cfg*(cond_v-uncond_v1)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `FlowMatchingHead` | 类 | Flow Matching 采样头 |
| `SimpleMLPAdaLN` | 类 | AdaLN 调制的 MLP 网络 |
| `ResBlock` | 类 | 自适应调制残差块 |
| `FinalLayer` | 类 | 最终投影层 |
| `TimestepEmbedder` | 类 | 时间步嵌入器 |

## 与其他模块的关系

- 被 `modeling_nextstep.py` 中的 `NextStepModel` 使用
- 在自回归生成的每步中被调用，生成单个图像 token

## 总结

FlowMatchingHead 是一个轻量级的 SDE 采样网络，在每个自回归步中被调用。它接收 LLM 的条件输出，通过多步 SDE 积分生成一个图像 token。支持 2-branch（文本 CFG）和 3-branch（文本+图像 CFG）的引导方式。
