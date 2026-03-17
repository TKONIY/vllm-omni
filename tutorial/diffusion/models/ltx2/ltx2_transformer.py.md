# `ltx2_transformer.py` — LTX2 视频 3D Transformer 模型

## 文件概述

该文件实现了 LTX2 的 3D 视频 Transformer 模型 (`LTX2VideoTransformer3DModel`)，这是一个大型的视频生成扩散 Transformer，支持视频和音频的联合生成。模型使用 vLLM 的张量并行层实现高效推理，支持序列并行。

## 关键代码解析

### 注意力机制 — LTX2Attention

模型使用 `QKVParallelLinear` 实现融合的 QKV 投影，并使用 `RMSNorm` 对 Q/K 进行归一化：

```python
self.to_qkv = QKVParallelLinear(
    hidden_size=query_dim,
    head_size=self.head_dim,
    total_num_heads=self.heads,
    total_num_kv_heads=self.kv_heads,
    bias=qk_norm == "rms_norm_across_heads",
)
```

### 嵌入系统

模型使用 `PixArtAlphaCombinedTimestepSizeEmbeddings` 和 `PixArtAlphaTextProjection` 处理时间步和文本条件：

```python
self.adaln_single = PixArtAlphaCombinedTimestepSizeEmbeddings(...)
self.caption_projection = PixArtAlphaTextProjection(...)
```

### 3D RoPE 位置编码

模型实现了专门的 3D 旋转位置编码，分别处理帧、高度和宽度三个维度：

```python
class LTX2VideoRoPE3D(nn.Module):
    # 为视频的时间和空间维度计算旋转位置编码
    def prepare_video_coords(self, batch_size, num_frames, height, width, device, fps):
        # 生成 (frame, height, width) 坐标
```

### Transformer 缓存支持

模型继承了 `CachedTransformer`，支持推理时的注意力缓存加速。

### 序列并行 (_sp_plan)

```python
_sp_plan = {
    "": {"hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3)},
    "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
}
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `LTX2VideoTransformer3DModel` | 类 | 主 3D Transformer 模型 |
| `LTX2Attention` | 类 | 张量并行的注意力层 |
| `LTX2TransformerBlock` | 类 | Transformer 块，含自注意力和交叉注意力 |
| `LTX2VideoRoPE3D` | 类 | 3D 旋转位置编码 |
| `Transformer2DModelOutput` | dataclass | 模型输出格式 |

## 与其他模块的关系

- 被 `pipeline_ltx2.py` 和 `pipeline_ltx2_image2video.py` 使用
- 使用 `vllm_omni.diffusion.attention.layer.Attention` 进行高效注意力计算
- 支持 `CachedTransformer` 的推理缓存机制
- 使用 `_sp_plan` 实现序列并行

## 总结

LTX2VideoTransformer3DModel 是一个大规模的 3D 视频生成 Transformer，支持视频+音频联合生成。通过 vLLM 的张量并行和序列并行优化，可以高效地处理高分辨率、长时间的视频生成任务。
