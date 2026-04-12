# DreamZero `image_encoder` vs HF `CLIPVisionModel`

本文记录本地 `transformers` 源码与当前 `vllm_omni/diffusion/models/dreamzero/modeling/image_encoder.py`
的逐层对比结论，目标是回答：

1. 我们现在的 `DreamZeroImageEncoder` 和 HF `CLIPVisionModel` 到底哪里实现不一样？
2. 哪个差异会导致真实 DreamZero 推理路径上的数值不对齐？

## 对比对象

- 当前实现：`vllm_omni/diffusion/models/dreamzero/modeling/image_encoder.py`
- HF 源码：`.venv/lib/python3.13/site-packages/transformers/models/clip/modeling_clip.py`
- HF 预处理：`.venv/lib/python3.13/site-packages/transformers/models/clip/image_processing_clip.py`

## 最终结论

- `DreamZeroImageEncoder` 的主体结构和 HF `CLIPVisionModel` 在视觉 tower 上**基本同构**。
- 但 stock HF 版本在 DreamZero 的真实 bf16 服务路径上**不能严格对齐**。
- 首个真实数值分歧点不是 QKV remap、不是 hidden state 索引、也不是 attention backend，
  而是 **LayerNorm 的输出 dtype 语义不同**。
- 另一个独立问题是 `CLIPImageProcessor` 的输入契约和默认预处理流程也不等价，因此也不能直接拿来替代源码路径。

## 1. 结构上哪些地方是等价的

### 1.1 Embedding / patchify / class token / pos embedding

当前实现：

- `DreamZeroVisionTransformer.forward()`：
  - patchify: `image_encoder.py:166`
  - class token: `image_encoder.py:167`
  - pos embedding: `image_encoder.py:171`
  - pre-norm: `image_encoder.py:172`

HF：

- `CLIPVisionEmbeddings.forward()`：
  - patchify: `modeling_clip.py:201`
  - class token: `modeling_clip.py:205`
  - pos embedding: `modeling_clip.py:206`
- `CLIPVisionTransformer.forward()`：
  - pre-norm: `modeling_clip.py:742`

实测结论：

- 在真实 DreamZero 权重 remap 完成后，
  `embeddings` 输出是 **exact match**：
  - `max diff = 0.0`
  - `mean diff = 0.0`

这说明：

- `cls_embedding` 参数 vs HF `nn.Embedding` 参数形式的区别只是**存储形式不同**
- `to_qkv` 合并权重拆成 `q_proj/k_proj/v_proj` 也不是这里的误差来源

### 1.2 Transformer block 本体

当前实现：

- `DreamZeroVisionAttentionBlock`: `image_encoder.py:63`
- `DreamZeroVisionSelfAttention`: `image_encoder.py:32`

HF：

- `CLIPEncoderLayer`: `modeling_clip.py:368`
- `CLIPAttention`: `modeling_clip.py:278`

实测结论：

- 给第 0 个 block 喂**完全相同的输入张量**时：
  - local block 输出与 HF block 输出 **exact match**
  - `max diff = 0.0`
  - `mean diff = 0.0`

这说明：

- QKV 合并实现 vs HF split Q/K/V 不是误差根因
- attention / MLP / residual 拓扑在 remap 后是等价的

## 2. 真正的首个分歧点：LayerNorm 输出 dtype

### 2.1 当前实现

`DreamZeroLayerNorm.forward()`：

- `image_encoder.py:28`
- `image_encoder.py:29`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return super().forward(x).type_as(x)
```

关键点：

- 当前实现会把 `LayerNorm` 输出**强制 cast 回输入 dtype**
- 对 DreamZero 的 bf16 服务路径，这意味着输出保持 `bf16`

### 2.2 HF 实现

HF 在 vision tower 里直接用的是原生 `nn.LayerNorm`：

- top pre-norm：`modeling_clip.py:722`
- block 内 LN1/LN2：`modeling_clip.py:373`、`modeling_clip.py:375`

HF 源码里**没有**对应的 `.type_as(x)` 回写。

### 2.3 实测现象

在真实 DreamZero 服务输入 + 真实权重 remap 下：

- `embeddings` 输出：
  - local dtype = `bfloat16`
  - HF dtype = `bfloat16`
  - diff = `0`
- `pre_norm` 输出：
  - local dtype = `bfloat16`
  - HF dtype = `float32`
  - `max diff = 2.4669647e-02`
  - `mean diff = 1.7555278e-04`
- 把 HF `pre_norm` 输出手动 `to(torch.bfloat16)` 后，再比：
  - `max diff = 0.0`
  - `mean diff = 0.0`

因此可以确认：

- **首个数值分歧点就是 top pre-norm 的输出 dtype 语义**
- 不是权重 remap 错了
- 不是 hidden state 取错了
- 也不是 attention backend 先出错

## 3. 为什么不是 attention backend 的问题

HF `CLIPAttention` 支持不同 attention backend：

- dispatch 入口：`modeling_clip.py:329`
- eager 路径：`modeling_clip.py:250`

我们额外做了对比：

- HF eager
- HF sdpa

结果：

- 两条路径和当前 local 实现的逐层 diff **完全一样**

所以：

- `eager` vs `sdpa` 不是这次 drift 的根因

## 4. `hidden_states[-2]` 是否和 `use_31_block=True` 对齐

当前实现：

- `DreamZeroVisionTransformer.forward()` 在 `use_31_block=True` 时返回
  `self.transformer[:-1](x)`，见 `image_encoder.py:175`-`image_encoder.py:176`
- 这表示只跑前 31 个 blocks（索引 `0..30`）

HF：

- `CLIPEncoder.forward()` 会在每层**进入前**先把 `hidden_states` 记入 `encoder_states`，
  见 `modeling_clip.py:545`-`modeling_clip.py:562`

因此在 32 层模型里：

- `hidden_states[-2]` 对应的是 **layer 30 之后、layer 31 之前** 的状态
- 也就是前 31 个 blocks 的输出

这一点和 DreamZero `use_31_block=True` 是对齐的。

结论：

- `hidden_states[-2]` 这个索引本身**没有问题**

## 5. 为什么 `CLIPImageProcessor` 也不能直接用

DreamZero 源码路径：

- 先对输入 tensor 做固定 `224x224` bicubic resize
- 再做 `(x * 0.5 + 0.5)`
- 再做 CLIP normalize

当前实现对应：

- resize: `image_encoder.py:221`-`image_encoder.py:230`
- 反归一化 + normalize: `image_encoder.py:231`

HF `CLIPImageProcessor` 默认行为：

- `do_resize=True`: `image_processing_clip.py:97`
- `size={"shortest_edge": 224}`: `image_processing_clip.py:111`
- `do_center_crop=True`: `image_processing_clip.py:100`
- `do_rescale=True`: `image_processing_clip.py:102`
- 实际 `preprocess()` 主流程：`image_processing_clip.py:202`

这和 DreamZero 的服务路径至少有两个不一致：

1. 默认是 `shortest_edge + center_crop`，不是固定 `(224, 224)` resize
2. 输入契约偏向 PIL / numpy / 0~255 图像，而 DreamZero 当前路径输入已经是 GPU 上的 `[-1, 1]` tensor

即便把 processor 参数手动改到尽量接近，真实输入上也仍然不能和源码路径完全一致。

## 6. 本次结论对实现决策的影响

因此当前 DreamZero 侧继续保留本地 `DreamZeroImageEncoder` 是合理的：

- 它保留了 DreamZero 源码的参数布局
- `action_head.image_encoder.model.* -> image_encoder.model.*` 可以直接 prefix strip 加载
- 它复现了 DreamZero 自定义 `LayerNorm(...).type_as(x)` 语义
- 它复现了源码 `encode_image()` 的 tensor 预处理路径

相反，直接复用 stock HF：

- `CLIPVisionModel`
- `CLIPImageProcessor`

都不能满足“和 DreamZero 源码严格数值对齐”的要求。

## 7. 一句话结论

> HF vision tower 和我们现在的 DreamZero image encoder，主体结构是同构的；真正导致真实服务路径不对齐的首个实现差异，是 HF `LayerNorm` 没有像 DreamZero 一样把输出 cast 回输入 dtype，而 `CLIPImageProcessor` 的预处理契约也不等价。
