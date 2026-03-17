# `cosyvoice3.py` — CosyVoice3 顶层模型类

## 文件概述

CosyVoice3 的顶层模型入口，实现多模态处理器链和两阶段模型路由。根据 `model_stage` 参数初始化 talker（文本到语音 token）或 code2wav（token 到波形）阶段。

## 关键代码解析

### 1. 多模态处理器

```python
class CosyVoice3MultiModalProcessor(BaseMultiModalProcessor):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        # 1. 提取文本 token
        text_token = extract_text_token(prompt, self.tokenizer, config.allowed_special)
        # 2. 提取参考音频的语音 token、mel 特征、说话人嵌入
        speech_token = extract_speech_token(audio, self.speech_tokenizer, device)
        speech_feat = extract_speech_feat(audio, self.feat_extractor, device)
        embedding = extract_spk_embedding(audio, self.campplus_session, device)
```

处理器懒加载 ONNX 推理会话（speech_tokenizer、campplus），并缓存以避免重复加载。

### 2. 模型阶段路由

```python
class CosyVoice3Model(nn.Module, SupportsMultiModal):
    def __init__(self, *, vllm_config, prefix=""):
        if self.model_stage == "talker":
            self.talker = CosyVoice3LM(...)       # 自回归语音 token 生成
        elif self.model_stage == "code2wav":
            self.code2wav = CosyVoice3Code2Wav(...)  # 流匹配 + 声码器
```

### 3. 前向传播

```python
def forward(self, input_ids, positions, ...):
    if self.model_stage == "talker":
        hidden_states = self.model.llm(inputs_embeds, positions)
        return OmniOutput(text_hidden_states=hidden_states, multimodal_outputs=...)
    elif self.model_stage == "code2wav":
        tts_speech = self.code2wav(token, prompt_token, prompt_feat, embedding)
        return OmniOutput(text_hidden_states=None, multimodal_outputs={"audio": tts_speech})
```

### 4. Logits 计算与掩码

```python
def compute_logits(self, hidden_states):
    logits = self.model.llm_decoder(hidden_states)
    logits[..., -200:] = float("-inf")  # 屏蔽尾部 token
    logits[..., self.config.llm["eos_token_id"]] = eos_token_val  # 保留 EOS
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `CosyVoice3MultiModalProcessingInfo` | 类 | 处理信息提供者 |
| `CosyVoice3MultiModalProcessor` | 类 | 多模态处理器，管理 ONNX 运行时组件 |
| `CosyVoice3DummyInputsBuilder` | 类 | 虚拟输入构建器（30秒参考音频） |
| `CosyVoice3Model` | 类 | 顶层模型，路由 talker/code2wav |

## 与其他模块的关系

- **talker 阶段** 依赖 `cosyvoice3_talker.py` 中的 `CosyVoice3LM` 和 `VLLMQwen2Encoder`
- **code2wav 阶段** 依赖 `cosyvoice3_code2wav.py` 中的 `CosyVoice3Code2Wav`
- **工具函数** 来自 `utils.py`（音频处理）和 `tokenizer.py`（文本 tokenizer）
- **权重加载** 从 `llm.pt`、`flow.pt`、`hift.pt` 分别加载各阶段权重

## 总结

作为 CosyVoice3 的入口，该文件协调了多模态预处理、两阶段模型初始化、权重加载和推理流程。通过 `model_stage` 参数实现单一代码路径支持多阶段部署。
