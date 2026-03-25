# 4. CFG Parallel 适配：双输出模型的 all_gather + 本地 combine

## 背景

PR #2063 重构了 `CFGParallelMixin`：
- 删除冗余 broadcast：all_gather 后所有 rank 本地 combine + scheduler step
- 新增 tuple 输出支持：`predict_noise()` 可返回 `(video, action)` 元组
- PR #2160 (LTX2) 提供了双输出适配的参考模式

DreamZero 直接复用这套框架。

## DreamZero 的 CFG 特殊性

| 输出 | CFG 策略 | 原因 |
|------|---------|------|
| 视频 | 标准 CFG: `neg + 5.0 * (pos - neg)` | 视频生成需要条件引导 |
| 动作 | 仅 positive 分支 | 动作应直接跟随指令，无需"无条件"混合 |

## 实现：三个覆写点

### 1. `predict_noise()` → 返回 tuple

```python
def predict_noise(self, **kwargs):
    video_pred, action_pred, _kv = self.transformer(...)
    return (video_pred, action_pred)
```

`CFGParallelMixin.predict_noise_maybe_with_cfg()` 自动处理：
- 用 `_wrap()` 把 tuple 的每个元素分别 all_gather
- 重建 positive/negative tuples → 传给 `combine_cfg_noise()`

### 2. `combine_cfg_noise()` → 视频 CFG + 动作 positive

```python
def combine_cfg_noise(self, pos, neg, scale, normalize):
    (video_pos, action_pos) = pos
    (video_neg, _) = neg
    video_combined = super().combine_cfg_noise(video_pos, video_neg, scale, normalize)
    return (video_combined, action_pos)
```

### 3. `VideoActionScheduler` → 组合调度器

```python
class VideoActionScheduler:
    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        video_out = self.video_scheduler.step(noise_pred[0], t[0], latents[0], ...)[0]
        action_out = self.action_scheduler.step(noise_pred[1], t[1], latents[1], ...)[0]
        return ((video_out, action_out),)
```

通过 `per_request_scheduler=video_action_scheduler` 传给 `scheduler_step_maybe_with_cfg()`。

## CFG Parallel 数据流

```
rank 0 (positive):                    rank 1 (negative):
CausalWanModel(cond_prompt)           CausalWanModel(empty_prompt)
→ (video_pos, action_pos)             → (video_neg, action_neg)
         │                                       │
         └───── all_gather (video) ──────────────┘
         └───── all_gather (action) ─────────────┘
                         │
              ┌──────────┴──────────┐
              │   两个 rank 本地 combine:    │
              │   video = neg + 5*(pos-neg) │
              │   action = pos              │
              └──────────┬──────────┘
                         │
              VideoActionScheduler.step()
              .contiguous() + cuda.synchronize()
                         │
              两个 rank 结果 bit-identical
```

## 同步保障

跟 PR #2160 (LTX2) 相同的模式：

```python
def _synchronize_cfg_parallel_step_output(self, latents, do_true_cfg):
    latents = tuple(tensor.contiguous() for tensor in latents)
    if not self._is_cfg_parallel_enabled(do_true_cfg):
        return latents
    device = next((t.device for t in latents if t.is_cuda), None)
    if device is not None:
        torch.cuda.current_stream(device).synchronize()
    return latents
```

- `.contiguous()` 确保内存布局一致
- `cuda.synchronize()` 确保 all_gather 完成后再 scheduler step
- 仅在 CFG parallel 模式下执行（单 GPU 无需同步）

## generator 透传

PR #2063 的 wtomin review fix：`scheduler_step()` 支持 `generator` 参数透传，为未来非确定性 scheduler（如 DDPM）做准备。当前 FlowUniPC 是确定性的，传 `None` 即可。
