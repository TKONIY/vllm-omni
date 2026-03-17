# `cfm.py` — 条件流匹配（CFM）实现

## 文件概述

实现条件流匹配（Conditional Flow Matching）算法，用于从编码器输出 mu 和噪声 z 之间进行确定性 ODE 求解，生成 mel 频谱。支持 Classifier-Free Guidance（CFG）推理。

## 关键代码解析

### CFG 推理

```python
class ConditionalCFM(BASECFM):
    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """固定步长 Euler ODE 求解器"""
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance: 同时推理有条件/无条件路径
            x_in[:] = x          # 重复两份
            mu_in[0] = mu         # 有条件
            spks_in[0] = spks     # 有说话人
            cond_in[0] = cond     # 有参考
            dphi_dt = self.forward_estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            # CFG 混合
            dphi_dt = (1 + cfg_rate) * dphi_dt - cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
```

### 因果变体

```python
class CausalConditionalCFM(ConditionalCFM):
    """因果版本：不使用缓存和重叠机制，适用于非流式推理"""
    def forward(self, mu, mask, n_timesteps, ...):
        z = torch.randn_like(mu) * temperature
        return self.solve_euler(z, t_span, mu, mask, spks, cond), None
```

### CausalMaskedDiffWithDiT

```python
class CausalMaskedDiffWithDiT(nn.Module):
    """完整的流匹配管线：token嵌入 → PreLookahead → token上采样 → CFM解码"""
    def inference(self, token, token_len, prompt_token, prompt_feat, embedding, finalize):
        token_emb = self.input_embedding(token) * mask
        h = self.pre_lookahead_layer(token_emb)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)  # 2x 上采样
        feat, _ = self.decoder(mu=h, mask=mask, spks=embedding, cond=conds, n_timesteps=10)
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `BASECFM` | 类 | CFM 抽象基类 |
| `ConditionalCFM` | 类 | 带 CFG 的条件流匹配 |
| `CausalConditionalCFM` | 类 | 因果条件流匹配 |
| `CausalMaskedDiffWithDiT` | 类 | 完整的流匹配推理管线 |

## 总结

CFM 通过 Euler ODE 求解器在噪声和目标之间进行插值，结合 CFG 提升生成质量。支持 PyTorch Module 和 TensorRT 两种 estimator 后端。
