# MammothModa2 模型教程索引

## 模块概述

`mammoth_moda2` 模块实现了 MammothModa2 的 DiT 生成阶段，采用 Lumina2 架构风格的扩散 Transformer。该模块作为两阶段架构中的下游非自回归阶段，接收上游 AR 模型的条件 token 进行图像生成。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`mammothmoda2_dit_model.py`](mammothmoda2_dit_model.py.md) | DiT Transformer 模型，含 Refiner 和 Q-Former |
| [`pipeline_mammothmoda2_dit.py`](pipeline_mammothmoda2_dit.py.md) | 生成阶段管线，与 vLLM 原生接口集成 |
| [`rope_real.py`](rope_real.py.md) | 实值旋转位置编码实现 |
| [`schedulers.py`](schedulers.py.md) | 自定义 Flow Matching 调度器 |

## 架构特点

- **两阶段架构**：AR 阶段生成条件 token，DiT 阶段生成图像
- **分阶段精化**：文本、噪声、参考图像各自有独立的 Refiner
- **动态时间偏移**：调度器根据分辨率自适应调整时间步
- **Q-Former 精化**：可选的图像条件精化器
- **vLLM 原生集成**：使用 VllmConfig 接口，返回 OmniOutput
- **CFG 范围控制**：只在指定步数范围内应用 CFG
