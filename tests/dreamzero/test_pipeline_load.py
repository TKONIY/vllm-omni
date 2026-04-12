"""Test DreamZero pipeline instantiation and weight loading from root checkpoint."""
from dataclasses import dataclass, field
import json
import os
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
import vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero as pipeline_mod
from vllm_omni.diffusion.distributed.parallel_state import init_dit_group
from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

CHECKPOINT_DIR = "/home/yangshen/code/dreamzero/checkpoints/dreamzero"

@pytest.mark.skipif(
    not os.path.exists(CHECKPOINT_DIR),
    reason="DreamZero checkpoint not available",
)
def test_pipeline_init_and_load_weights(default_vllm_config):
    # Init DIT group (needed by DistributedAutoencoderKLWan)
    try:
        init_dit_group(dit_parallel_size=1, backend="nccl")
    except (AssertionError, RuntimeError):
        pass

    @dataclass
    class MockODConfig:
        model: str = CHECKPOINT_DIR
        model_config: dict = field(default_factory=dict)
        model_class_name: str = "DreamZeroPipeline"
        model_paths: dict = field(default_factory=lambda: {"tokenizer": "google/umt5-xxl"})
        dtype: object = torch.bfloat16

    # Step 1: Instantiate pipeline
    t0 = time.time()
    pipeline = DreamZeroPipeline(od_config=MockODConfig())
    print(f"\nInit: {time.time()-t0:.1f}s")
    print(f"transformer: {sum(p.numel() for p in pipeline.transformer.parameters()):,}")
    print(f"text_encoder: {sum(p.numel() for p in pipeline.text_encoder.parameters()):,}")
    print(f"image_encoder: {sum(p.numel() for p in pipeline.image_encoder.parameters()):,}")

    # Step 2: Load weights
    t0 = time.time()
    with open(os.path.join(CHECKPOINT_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)

    shard_keys = defaultdict(list)
    for key, shard_file in index["weight_map"].items():
        shard_keys[shard_file].append(key)

    def weight_iter():
        for shard_file, keys in sorted(shard_keys.items()):
            shard_path = os.path.join(CHECKPOINT_DIR, shard_file)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in keys:
                    yield key, f.get_tensor(key)

    loaded = pipeline.load_weights(weight_iter())
    print(f"Loaded {len(loaded)} params in {time.time()-t0:.1f}s")

    # Step 3: Verify coverage
    all_params = set(dict(pipeline.named_parameters()).keys())
    missing = all_params - loaded

    print(f"All params: {len(all_params)}, loaded: {len(loaded & all_params)}, missing: {len(missing)}")
    if missing:
        for m in sorted(missing)[:15]:
            print(f"  MISSING: {m}")

    assert len(missing) == 0, f"{len(missing)} params not loaded: {sorted(missing)[:5]}"
    print("ALL PARAMETERS LOADED SUCCESSFULLY")


def test_pipeline_hf_root_init_does_not_require_extra_vae_source(monkeypatch):
    root_cfg = {
        "action_head_cfg": {
            "config": {
                "action_dim": 32,
                "num_frames": 33,
                "num_frame_per_block": 2,
                "action_horizon": 24,
                "decouple_inference_noise": False,
                "video_inference_final_noise": 0.8,
                "hidden_size": 64,
                "max_state_dim": 64,
                "max_action_dim": 32,
                "diffusion_model_cfg": {
                    "model_type": "i2v",
                    "dim": 64,
                    "ffn_dim": 128,
                    "num_heads": 4,
                    "num_layers": 2,
                    "in_dim": 36,
                    "out_dim": 16,
                    "freq_dim": 256,
                    "eps": 1e-6,
                    "frame_seqlen": 880,
                    "num_action_per_block": 24,
                    "num_frame_per_block": 2,
                    "num_state_per_block": 1,
                },
            },
        },
    }

    def fake_load_repo_json(model_path: str, relative_path: str, local_files_only: bool):
        assert model_path == "GEAR-Dreams/DreamZero-DROID"
        if relative_path == "config.json":
            return root_cfg
        if relative_path == "experiment_cfg/metadata.json":
            return {}
        return None

    monkeypatch.setattr(DreamZeroPipeline, "_load_repo_json", staticmethod(fake_load_repo_json))
    monkeypatch.setattr(pipeline_mod.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())

    class FakeUMT5EncoderModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.weight = nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(pipeline_mod, "UMT5EncoderModel", FakeUMT5EncoderModel)

    class FakeTransformer(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.weight = nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(pipeline_mod, "CausalWanModel", FakeTransformer)

    class FakeImageEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(pipeline_mod, "DreamZeroImageEncoder", FakeImageEncoder)

    class FakeScheduler:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(pipeline_mod, "FlowUniPCMultistepScheduler", FakeScheduler)

    class FakeVAE(nn.Module):
        from_pretrained_calls = []

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16)
            self.init_distributed_called = False

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.from_pretrained_calls.append((args, kwargs))
            raise AssertionError("HF-root bootstrap should not require `from_pretrained(..., subfolder=\"vae\")`.")

        def init_distributed(self):
            self.init_distributed_called = True

        def to(self, *args, **kwargs):
            return self

    monkeypatch.setattr(pipeline_mod, "DistributedAutoencoderKLWan", FakeVAE)

    @dataclass
    class MockODConfig:
        model: str = "GEAR-Dreams/DreamZero-DROID"
        model_config: dict = field(default_factory=dict)
        model_class_name: str = "DreamZeroPipeline"
        model_paths: dict = field(default_factory=dict)
        dtype: object = torch.bfloat16
        enable_cpu_offload: bool = False
        enable_layerwise_offload: bool = False

    pipeline = DreamZeroPipeline(od_config=MockODConfig())

    assert isinstance(pipeline.vae, FakeVAE)
    assert pipeline.vae.init_distributed_called is True
    assert FakeVAE.from_pretrained_calls == []
    assert pipeline.weights_sources[0].model_or_path == "GEAR-Dreams/DreamZero-DROID"


def test_pipeline_uses_hf_root_config_instead_of_runtime_model_config(monkeypatch):
    root_cfg = {
        "action_head_cfg": {
            "config": {
                "action_dim": 32,
                "num_frames": 33,
                "num_frame_per_block": 2,
                "action_horizon": 24,
                "decouple_inference_noise": False,
                "video_inference_final_noise": 0.8,
                "hidden_size": 64,
                "max_state_dim": 64,
                "max_action_dim": 32,
                "diffusion_model_cfg": {
                    "model_type": "i2v",
                    "dim": 64,
                    "ffn_dim": 128,
                    "num_heads": 4,
                    "num_layers": 2,
                    "in_dim": 36,
                    "out_dim": 16,
                    "freq_dim": 256,
                    "eps": 1e-6,
                    "frame_seqlen": 880,
                    "num_action_per_block": 24,
                    "num_frame_per_block": 2,
                    "num_state_per_block": 1,
                },
            },
        },
    }

    def fake_load_repo_json(model_path: str, relative_path: str, local_files_only: bool):
        if relative_path == "config.json":
            return root_cfg
        if relative_path == "experiment_cfg/metadata.json":
            return {}
        return None

    monkeypatch.setattr(DreamZeroPipeline, "_load_repo_json", staticmethod(fake_load_repo_json))
    monkeypatch.setattr(pipeline_mod.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())

    class FakeUMT5EncoderModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.weight = nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(pipeline_mod, "UMT5EncoderModel", FakeUMT5EncoderModel)

    class FakeTransformer(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.weight = nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(pipeline_mod, "CausalWanModel", FakeTransformer)

    class FakeImageEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(pipeline_mod, "DreamZeroImageEncoder", FakeImageEncoder)

    class FakeScheduler:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(pipeline_mod, "FlowUniPCMultistepScheduler", FakeScheduler)

    class FakeVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16)

        def init_distributed(self):
            pass

        def to(self, *args, **kwargs):
            return self

    monkeypatch.setattr(pipeline_mod, "DistributedAutoencoderKLWan", FakeVAE)

    @dataclass
    class MockODConfig:
        model: str = "GEAR-Dreams/DreamZero-DROID"
        model_config: dict = field(default_factory=lambda: {
            "num_frames": 81,
            "num_frame_per_block": 99,
            "action_horizon": 999,
            "decouple_inference_noise": True,
            "video_inference_final_noise": 0.1,
            "max_state_dim": 999,
            "max_action_dim": 999,
        })
        model_class_name: str = "DreamZeroPipeline"
        model_paths: dict = field(default_factory=dict)
        dtype: object = torch.bfloat16
        enable_cpu_offload: bool = False
        enable_layerwise_offload: bool = False

    pipeline = DreamZeroPipeline(od_config=MockODConfig())

    assert pipeline.num_frames == 33
    assert pipeline.num_frame_per_block == 2
    assert pipeline.action_horizon == 24
    assert pipeline.decouple_inference_noise is False
    assert pipeline.video_inference_final_noise == 0.8
    assert pipeline.max_state_dim == 64
    assert pipeline.max_action_dim == 32
    assert pipeline.transformer.kwargs["action_dim"] == 32
    assert pipeline.transformer.kwargs["max_state_dim"] == 64
    assert pipeline.transformer.kwargs["num_frame_per_block"] == 2
    assert "hidden_size" not in pipeline.transformer.kwargs


def test_pipeline_requires_root_config_fields_instead_of_falling_back(monkeypatch):
    root_cfg = {
        "action_head_cfg": {
            "config": {
                "action_dim": 32,
                "num_frame_per_block": 2,
                "action_horizon": 24,
                "decouple_inference_noise": False,
                "video_inference_final_noise": 0.8,
                "hidden_size": 64,
                "max_state_dim": 64,
                "max_action_dim": 32,
                "diffusion_model_cfg": {
                    "model_type": "i2v",
                    "dim": 64,
                    "ffn_dim": 128,
                    "num_heads": 4,
                    "num_layers": 2,
                    "in_dim": 36,
                    "out_dim": 16,
                    "freq_dim": 256,
                    "eps": 1e-6,
                    "frame_seqlen": 880,
                    "num_action_per_block": 24,
                    "num_frame_per_block": 2,
                    "num_state_per_block": 1,
                },
            },
        },
    }

    def fake_load_repo_json(model_path: str, relative_path: str, local_files_only: bool):
        if relative_path == "config.json":
            return root_cfg
        if relative_path == "experiment_cfg/metadata.json":
            return {}
        return None

    monkeypatch.setattr(DreamZeroPipeline, "_load_repo_json", staticmethod(fake_load_repo_json))
    monkeypatch.setattr(pipeline_mod.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())

    class FakeUMT5EncoderModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.weight = nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(pipeline_mod, "UMT5EncoderModel", FakeUMT5EncoderModel)

    class FakeTransformer(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.weight = nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(pipeline_mod, "CausalWanModel", FakeTransformer)

    class FakeImageEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(pipeline_mod, "DreamZeroImageEncoder", FakeImageEncoder)

    class FakeScheduler:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(pipeline_mod, "FlowUniPCMultistepScheduler", FakeScheduler)

    class FakeVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16)

        def init_distributed(self):
            pass

        def to(self, *args, **kwargs):
            return self

    monkeypatch.setattr(pipeline_mod, "DistributedAutoencoderKLWan", FakeVAE)

    @dataclass
    class MockODConfig:
        model: str = "GEAR-Dreams/DreamZero-DROID"
        model_config: dict = field(default_factory=lambda: {"num_frames": 81})
        model_class_name: str = "DreamZeroPipeline"
        model_paths: dict = field(default_factory=dict)
        dtype: object = torch.bfloat16
        enable_cpu_offload: bool = False
        enable_layerwise_offload: bool = False

    with pytest.raises(KeyError, match="num_frames"):
        DreamZeroPipeline(od_config=MockODConfig())
