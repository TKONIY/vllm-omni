# `fusion.py` — DreamID-Omni 视频-音频融合模型

## 文件概述

本文件实现了 DreamID-Omni 的核心融合模型 `FusionModel`，将视频和音频两个模态的 Transformer 模型耦合在一起进行联合去噪。通过在每个 Transformer 块中注入双向交叉注意力（视频关注音频、音频关注视频），实现视频和音频内容的时空同步。

## 关键代码解析

### 1. 模型初始化与交叉注意力注入

```python
class FusionModel(nn.Module):
    def __init__(self, video_config=None, audio_config=None):
        self.video_model = WanModel(**video_config)
        self.audio_model = WanModel(**audio_config)
        self.inject_cross_attention_kv_projections()

    def inject_cross_attention_kv_projections(self):
        for vid_block in self.video_model.blocks:
            vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim)
            vid_block.cross_attn.norm_k_fusion = WanRMSNorm(vid_block.dim)
```

为每个 Transformer 块的交叉注意力模块动态注入额外的 KV 投影层，用于跨模态融合。

### 2. 融合交叉注意力

```python
def single_fusion_cross_attention_forward(self, cross_attn_block, src_seq, ..., target_seq, ...):
    # 1. 计算原始交叉注意力（文本条件）
    q, k, v = cross_attn_block.qkv_fn(src_seq, context)
    x = self.attn(q, k, v)
    if k_img is not None:
        img_x = self.attn(q, k_img, v_img)
        x = x + img_x

    # 2. 计算跨模态注意力（另一模态的序列）
    target_seq = cross_attn_block.pre_attn_norm_fusion(target_seq)
    k_target = cross_attn_block.norm_k_fusion(cross_attn_block.k_fusion(target_seq))
    v_target = cross_attn_block.v_fusion(target_seq)

    # 3. 对 Q 和 K 应用各自的 RoPE
    q = rope_apply(q, src_grid_sizes, src_freqs, ...)
    k_target = rope_apply(k_target, target_grid_sizes, target_freqs, ...)

    target_x = self.attn(q, k_target, v_target)
    x = x + target_x  # 原始 + 跨模态
```

融合注意力的核心：源模态的 Q 对目标模态的 K/V 进行注意力计算，并加上原始的文本交叉注意力结果。

### 3. 融合块前向

```python
def single_fusion_block_forward(self, vid_block, audio_block, vid, audio, ...):
    # 1. 音频自注意力
    audio_y = audio_block.self_attn(audio, ...)
    audio = audio + audio_y * audio_e[2]

    # 2. 视频自注意力
    vid_y = vid_block.self_attn(vid, ...)
    vid = vid + vid_y * vid_e[2]

    og_audio = audio  # 保存融合前的音频

    # 3. 音频交叉注意力 + FFN（关注视频）
    audio = self.single_fusion_cross_attention_ffn_forward(
        audio_block, audio, ..., target_seq=vid, ...)

    # 4. 视频交叉注意力 + FFN（关注融合前的音频）
    vid = self.single_fusion_cross_attention_ffn_forward(
        vid_block, vid, ..., target_seq=og_audio, ...)

    return vid, audio
```

关键设计：视频关注的是融合前的音频（`og_audio`），避免循环依赖。

### 4. 完整前向

```python
def forward(self, vid, audio, t, vid_context, audio_context, ...):
    # 1. 准备各自的 Transformer 输入
    vid, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(...)
    audio, audio_e, audio_kwargs = self.audio_model.prepare_transformer_block_kwargs(...)

    # 2. 逐块融合
    for i in range(self.num_blocks):
        vid, audio = self.single_fusion_block_forward(
            self.video_model.blocks[i], self.audio_model.blocks[i], vid, audio, ...)

    # 3. 各自的后处理
    vid = self.video_model.post_transformer_block_out(vid, ...)
    audio = self.audio_model.post_transformer_block_out(audio, ...)
    return vid, audio
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `FusionModel` | 类 | 视频-音频融合模型 |
| `inject_cross_attention_kv_projections` | 方法 | 注入跨模态 KV 投影 |
| `single_fusion_cross_attention_forward` | 方法 | 单块融合交叉注意力 |
| `single_fusion_cross_attention_ffn_forward` | 方法 | 融合交叉注意力 + FFN |
| `single_fusion_block_forward` | 方法 | 单融合块前向（自注意力+双向交叉注意力+FFN） |
| `merge_kwargs` | 方法 | 合并视频/音频参数 |
| `set_rope_params` | 方法 | 设置 RoPE 参数 |

## 与其他模块的关系

- **`wan2_2.py`**：`WanModel` 提供视频和音频的基础 Transformer
- **`pipeline_dreamid_omni.py`**：管线调用 `FusionModel.forward` 进行联合去噪
- **`vllm_omni.diffusion.attention.layer.Attention`**：统一注意力层
- **`dreamid_omni` 外部包**：提供 `WanLayerNorm`、`WanRMSNorm`、`rope_apply` 等基础组件

## 总结

`fusion.py` 实现了 DreamID-Omni 的核心创新——视频和音频的双向融合去噪。通过在每个 Transformer 块中注入跨模态交叉注意力，使两个模态在去噪过程中相互感知并保持时空同步，同时使用原始音频快照避免循环依赖。
